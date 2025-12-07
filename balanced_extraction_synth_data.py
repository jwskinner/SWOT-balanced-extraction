# Balanced extraction on synthetic SWOT KaRIn (with optional Nadir)
# Units:: work in [cm] for covariances/obs, spectra in [cpkm] return [meters] for outputs.
import os
import pickle
import numpy as np
import xarray as xr
import scipy.linalg as la
import jws_swot_tools as swot
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import solve_triangular
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from jws_swot_tools.julia_bridge import julia_functions as jl

# =========================
# CONFIG
# =========================
pass_num = 9                                                                   # SWOT pass number
lat_min = 28 
lat_max = 35 

SYN_DIR = f"./synthetic_swot_data/Pass_{pass_num:03d}_Lat{lat_min}_{lat_max}/" 
KARIN_NA_PATH    = f"{SYN_DIR}/karin_synth.pkl"
NADIR_NA_PATH    = f"{SYN_DIR}/nadir_synth.pkl"
KARIN_PATH       = f"{SYN_DIR}/karin_swot.pkl"
NADIR_PATH       = f"{SYN_DIR}/nadir_swot.pkl"
RHO_L_KM         = 4.0                                                         # Gaussian spectral taper scale
COMPUTE_POSTERIOR= False                                                       # toggle posterior on target grid
TAPER_CUTOFF     = 2.0                                                         # "T(k)" cutoff
OUTNAME          = f"Pass_{pass_num:03d}_Lat{lat_min}_{lat_max}_{int(RHO_L_KM)}km"
OUTDIR           = f"./balanced_extraction/SYNTH_data/{OUTNAME}/"
os.makedirs(os.path.dirname(OUTDIR), exist_ok=True)
t = swot.Timer()

# -------------------------
# Load data
# -------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

karin_NA = load_pkl(KARIN_NA_PATH)
nadir_NA = load_pkl(NADIR_NA_PATH)
karin    = load_pkl(KARIN_PATH)
nadir    = load_pkl(NADIR_PATH)
t.lap("Data loaded")
ntime = karin_NA.ssha.shape[0]

# -------------------------
# Spectral fit (cm)
# -------------------------

noisy_karin  = karin_NA.ssh_noisy * 100.0                                      # synthetic SWOT data in m to cm
noisy_nadir = nadir_NA.ssh_noisy * 100.0                                       # synthetic nadir data in m to cm

ssh_noisy_xr = xr.DataArray(
    noisy_karin,
    coords=[np.arange(ntime), karin_NA.y_coord_km, karin_NA.x_coord_km],
    dims=["sample", "line", "pixel"],
)
spec_ssh_noisy = swot.mean_power_spectrum(ssh_noisy_xr, karin_NA.window, "line", ["sample", "pixel"])

nad_noisy_xr = xr.DataArray(
    noisy_nadir,
    coords=[np.arange(ntime), nadir_NA.y_coord_km],
    dims=["sample", "nadir_line"],
)

spec_nad_noisy = swot.mean_power_spectrum(nad_noisy_xr, nadir_NA.window, "nadir_line", ["sample"])

# Fit spectra to get parameters
p_karin, _   = swot.fit_spectrum(karin_NA, spec_ssh_noisy, swot.karin_model)
p_nadir, _   = swot.fit_nadir_spectrum(nadir_NA, spec_nad_noisy, p_karin)
swot.plot_spectral_fits(karin, nadir, p_karin, p_nadir, f'{OUTDIR}fits_synth.pdf')
t.lap("Spectral fits done")

# -------------------------
# Geometry & masks
# -------------------------
index = 2 # can be anything, just for masks
mask_k = np.isfinite(karin.ssha[index])
mask_n = np.isfinite(nadir.ssh[index]).ravel()
    
xkk = (karin.x_grid[mask_k].ravel(order="C")) * 1e-3  # km
ykk = (karin.y_grid[mask_k].ravel(order="C")) * 1e-3
xnn = (nadir.x_grid.ravel()[mask_n]) * 1e-3
ynn = (nadir.y_grid.ravel()[mask_n]) * 1e-3

# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin, unit="km", extend=True)
n_t = xt.size
t.lap("Grids and masks done")

# -------------------------
# Distance matrices in km
# -------------------------
r_kk = swot.pairwise_r(xkk, ykk)
r_nn = swot.pairwise_r(xnn, ynn) 
r_kn = swot.pairwise_r(xkk, ykk, xnn, ynn) 
r_tk = swot.pairwise_r(xt, yt, xkk, ykk)
r_tn = swot.pairwise_r(xt, yt, xnn, ynn)
r_tt = swot.pairwise_r(xt, yt)
t.lap("Distance matrices built")

# -------------------------
# Nadir white-noise std [cm]
# -------------------------
sigma_n = np.sqrt(p_nadir[0] / (2.0 * nadir.dy_km))  # cm

# -------------------------
# Covariance functions with spectral taper G and T
# -------------------------
B_psd = swot.balanced_psd_from_params(p_karin)                                 # B(k) balanced power spectrum model
Nk_psd = swot.karin_noise_psd_from_params(p_karin)                             # N_K(k) noise power spectrum model

# Base kernels in [cm^2]
n_samples = 100000
l_sample = 5000
kk = np.arange(n_samples // 2 + 1) / l_sample                                  # wavenumber grid for transforms

Tfun  = lambda k: swot.taper(k, cutoff=TAPER_CUTOFF)                           # T(k) is taper function  
C_B      = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk), kk), kk)                    # C[B]
C_BT     = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk)*Tfun(kk), kk), kk)           # C[B T]
C_NT2    = jl.cov(jl.abel(jl.iabel(Nk_psd(kk), kk)*Tfun(kk)**2, kk), kk)       # C[N T^2]

# Tapered Kernels (requires Abel transform) in [cm]
SIGMA         = 2 * np.pi * RHO_L_KM                                         # σ convert to angular wavenumber
DELTA         = (np.pi * karin.dx_km) / 2 * np.log(2)                          # δ

# Gaussian Smoothings and tapers combined
G  = lambda k: np.exp(-((SIGMA**2) * (k**2)) / 2.0)                            # Gassian smooth C[BG]
G2 = lambda k: np.exp(-(SIGMA**2) * (k**2))                                    # Target-Target smoothing C[BG^2]
GT = lambda k: np.exp(-(((SIGMA**2 + DELTA**2)* k**2) / 2.0) )                 # Taper + Gaussian Smooth C[BGT]
T2 = lambda k: np.exp(-(DELTA**2) * (k**2))                                    # Taper^2 C[BT^2]

C_B_G   = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G(kk), kk), kk)
C_B_G2  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G2(kk), kk), kk)
C_B_TG  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * GT(kk), kk), kk)
C_B_T2  = jl.cov(jl.abel(jl.iabel(B_psd(kk) + Nk_psd(kk),  kk) * T2(kk), kk), kk) # this is for my k_tt

# -------------------------
# Blocks
# -------------------------
# KaRIn–KaRIn: C[(B+N_K) T^2]
R_KK = np.asarray(C_B_T2(r_kk))

# Target-Target covariance for posterior C[BG^2] 
R_tt = np.asarray(C_B_G2(r_tt))

# Nadir–Nadir: C[B] + σ_N^2 I
R_NN = np.asarray(C_B(r_nn)) + (sigma_n**2) * np.eye(r_nn.shape[0])
     
# Cross KaRIn–Nadir: C[B T]
# This was the section that failed:
R_KN = np.asarray(C_BT(r_kn))  
R_NK = R_KN.T              
     
# Target cross terms C[BTG]
R_tK = np.asarray(C_B_TG(r_tk))     # C[BTG]
R_tN = np.asarray(C_B_TG(r_tn))     # C[BG]
     
# Assemble observation covariance
# np.block and np.concatenate will now work as expected
C_obs = np.block([[R_KK, R_KN],
                  [R_NK, R_NN]])
R = np.concatenate([R_tK, R_tN], axis=1)
t.lap("Covariance blocks built")

# -------------------------
# Cho factor
# -------------------------
eps = 1e-8 * float(np.mean(np.diag(C_obs)))
cho = la.cho_factor(C_obs + 0.0 * eps*np.eye(C_obs.shape[0]), lower=True)      # Turned the jitter off
t.lap("Cholesky done")

# -------------------------
# Extraction 
# -------------------------
print("Running balanced extraction...")
n_times = karin_NA.ssh_noisy.shape[0]

ht_all    = np.full((n_times, nxt, nyt), np.nan, dtype=float)
ug_all    = np.full((n_times, nxt, nyt), np.nan, dtype=float)
vg_all    = np.full((n_times, nxt, nyt), np.nan, dtype=float)
vel_all   = np.full((n_times, nxt, nyt), np.nan, dtype=float)
zetag_all = np.full((n_times, nxt, nyt), np.nan, dtype=float)

for i in range(n_times):
    if (i + 1) % 10 == 0 or i == n_times - 1:
        print(f"  processed {i + 1}/{n_times}")
    h_k = noisy_karin[i][mask_k].ravel()     # in [cm]
    h_n = noisy_nadir[i][mask_n]             # in [cm]
    h_obs = np.concatenate([h_k, h_n])

    z  = la.cho_solve(cho, h_obs)                                              # (C_obs)^{-1} h
    ht = R @ z                                                                 # cm
    ht_all[i] = ((ht / 100.0).reshape(nyt, nxt).T)                             # back to [m] 
    ug, vg, geo_vel, geo_vort = swot.plot_balanced_extraction(karin, nadir, ht_all[i], i, outdir=f"{OUTDIR}/plots/")
    ug_all[i] = ug
    vg_all[i] = vg
    vel_all[i] = geo_vel
    zetag_all[i] = geo_vort
t.lap("Extraction finished")

# -------------------------
# Save outputs
# -------------------------
# save a backup pkl
karin_NA.ssh_balanced = ht_all
karin_NA.ug           = ug_all
karin_NA.vg           = vg_all
karin_NA.velocity     = vel_all
karin_NA.vorticity    = zetag_all
with open(f"{OUTDIR}balanced_extraction_pass{pass_num}.pkl", "wb") as f:
    pickle.dump(karin_NA, f)
t.lap("Balanced field saved")

# -------------------------
# posterior on target (Eq. 10 in paper)
# -------------------------
if COMPUTE_POSTERIOR:
    L, lower = la.cho_factor(C_obs, lower=True, check_finite=False, overwrite_a=False)
    W = solve_triangular(L, R.T, lower=True, check_finite=False, overwrite_b=False)
    C_mean = W.T @ W                                                           # covariance of posterior mean R @ C_obs^{-1} @ R.T
    P = R_tt - C_mean                                                          # posterior 
    
    posterior_variance = np.diag(P)
    posterior_variance_field = posterior_variance.reshape(nyt, nxt)

    # ----- Save outputs -----
    km = int(RHO_L_KM)

    # Posterior covariance (full)
    with open(f"{OUTDIR}posterior.pkl", "wb") as f:
        pickle.dump(P, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior saved: {OUTDIR}posterior.pkl", "wb")

    # Covariance of posterior mean
    with open(f"{OUTDIR}Cmean.pkl", "wb") as f:
        pickle.dump(C_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"C_mean saved: {OUTDIR}Cmean.pkl")

    # Posterior variance (vector)
    with open(f"{OUTDIR}posterior_varvec.pkl", "wb") as f:
        pickle.dump(posterior_variance, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior variance (vector) saved: {OUTDIR}posterior_varvec.pkl")

    # Posterior variance (field)
    with open(f"{OUTDIR}posterior_varfield.pkl", "wb") as f:
        pickle.dump(posterior_variance_field, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior variance (field) saved: {OUTDIR}posterior_varfield.pkl")

t.total()
