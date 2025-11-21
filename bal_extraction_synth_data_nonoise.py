# Balanced extraction on synthetic SWOT KaRIn (with optional Nadir)
# Units in [cm] for covariances/obs, spectra in [cpkm] return [meters] for outputs.
# Put the simulation data through with no SWOT noise and include no noise in the covariances 
# so that the extraction is simply giving us the smoothed field by G(k)

import pickle
import numpy as np
import xarray as xr
import scipy.linalg as la
import JWS_SWOT_toolbox as swot
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import solve_triangular
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from JWS_SWOT_toolbox.julia_bridge import julia_functions as jl

# =========================
# CONFIG
# =========================
PICKLES_DIR       = "./pickles"
KARIN_NA_PATH     = f"{PICKLES_DIR}/karin_NA_tmean.pkl"
NADIR_NA_PATH     = f"{PICKLES_DIR}/nadir_NA_tmean.pkl"
KARIN_PATH        = f"{PICKLES_DIR}/karin.pkl"
NADIR_PATH        = f"{PICKLES_DIR}/nadir.pkl"
RHO_L_KM          = 1.0              # Gaussian spectral taper scale
COMPUTE_POSTERIOR = True             # toggle posterior on target grid
TAPER_CUTOFF      = 2.0              # "T(k)" cutoff
OUTNAME           = f"balanced_extraction_synth_NA_tmean_sm_{int(RHO_L_KM)}km_nonoise"
OUT_PREFIX        = f"{PICKLES_DIR}/{OUTNAME}"

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
noisy_karin  = karin_NA.ssh_noisy * 100.0    # our synthetic SWOT data in m to cm
noisy_nadir = nadir_NA.ssh_noisy * 100.0     # our synthetic nadir data in m to cm

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
swot.plot_spectral_fits(karin_NA, nadir_NA, p_karin, p_nadir, 'fits_synth.pdf')
t.lap("Spectral fits done")

# -------------------------
# Geometry & masks
# -------------------------
#index = 2 # can be anything, just for masks
# mask_k = np.isfinite(karin.ssha[index])
# mask_n = np.isfinite(nadir.ssh[index]).ravel()

# Build a 'KaRIn grid that is just the full simulation grid
# xkk = (karin_NA.x_grid.ravel(order="C")) * 1e-3  # km
# ykk = (karin_NA.y_grid.ravel(order="C")) * 1e-3
# xnn = (nadir_NA.x_grid.ravel()[mask_n]) * 1e-3
# ynn = (nadir_NA.y_grid.ravel()[mask_n]) * 1e-3

# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin_NA, unit="km", extend=False)
n_t = xt.size
t.lap("Grids and masks done")

# -------------------------
# Distance matrices in km
# -------------------------
# r_kk = swot.pairwise_r(xkk, ykk)
# r_nn = swot.pairwise_r(xnn, ynn) 
# r_kn = swot.pairwise_r(xkk, ykk, xnn, ynn) 
# r_tk = swot.pairwise_r(xt, yt, xkk, ykk)
# r_tn = swot.pairwise_r(xt, yt, xnn, ynn)
r_tt = swot.pairwise_r(xt, yt) # we only need target grid because its same as the simulation grid
t.lap("Distance matrices built")

# -------------------------
# Nadir white-noise std [cm]
# -------------------------
sigma_n = np.sqrt(p_nadir[0] / (2.0 * nadir.dy_km))  # cm

# -------------------------
# Covariance functions with spectral taper G and T
# -------------------------
B_psd = swot.balanced_psd_from_params(p_karin)                 # B(k) balanced power spectrum model
Nk_psd = swot.karin_noise_psd_from_params(p_karin)             # N_K(k) noise power spectrum model

n_samples = 100000
l_sample = 5000
kk = np.arange(n_samples // 2 + 1) / l_sample                  # wavenumber grid for transforms

rho           = 2 * np.pi * RHO_L_KM                           # ρ for Gaussian smooth
delta         = (np.pi * karin.dx_km) / (2 * np.log(2))        # δ for taper

# Gaussian and taper
G  = lambda k: np.exp(-((rho**2) * (k**2)) / 2.0)              # Gassian smooth C[BG]
T  = lambda k: np.exp(-((delta**2) * (k**2)) / 2.0)            # Taper C[BT]
G2 = lambda k: np.exp(-(rho**2) * (k**2))                    
GT = lambda k: np.exp(-(((rho**2 + delta**2)* k**2) / 2.0) ) 
T2 = lambda k: np.exp(-(delta**2) * (k**2))                    

#C_B      = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk), kk), kk)             # C[B]
C_B      = jl.cov(B_psd(kk),  kk)                                        # C[B]
# C_BT     = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * T(kk),  kk), kk)  # C[BT]
C_BG     = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G(kk),  kk), kk)    # C[BT]
C_BG2    = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G2(kk), kk), kk)    # C[BG^2]
#C_BTG    = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G(kk), kk), kk)    # C[BGT]
# -------------------------
# Blocks
# -------------------------
# Observation terms
# R_KK = np.asarray(C_BT2(r_kk))
# R_NN = np.asarray(C_B(r_nn)) # Nadir–Nadir: C[B] + σ_N^2 
# R_KN = np.asarray(C_BT(r_kn))  
# R_NK = R_KN.T              

# Target terms 
# R_tt = np.asarray(C_BG2(r_tt))  
# R_tK = np.asarray(C_BTG(r_tk))     
# R_tN = np.asarray(C_BG(r_tn))      
     
# Assemble observation covariance
# C_obs = np.block([[R_KK, R_KN],
#                   [R_NK, R_NN]])
# R = np.concatenate([R_tK, R_tN], axis=1)

# New calculation on full sim data + smooth 
R_KK = np.asarray(C_B(r_tt))                # 'karin-karin' is now just our full fields
R_tK = np.asarray(C_BG(r_tt))               # target-karin grid is target-sim
R_tt = np.asarray(C_BG2(r_tt))              # target-target same as before 
C_obs = R_KK                                # only use the karin part
R = R_tK

# swot.diagnose_positive_definite(C_obs)

t.lap("Covariance blocks built")

# -------------------------
# Cholesky factor
# -------------------------
eps = 0.0 # 1e-8 * float(np.mean(np.diag(C_obs)))
(L, lower) = la.cho_factor(C_obs + eps*np.eye(C_obs.shape[0]), lower=True)
t.lap("Cholesky done")

# -------------------------
# Extraction 
# -------------------------
print("Running balanced extraction...")
n_times = karin_NA.ssh_noisy.shape[0]
ht_all = np.empty((n_times, nyt, nxt), dtype=float)

for i in range(n_times):
    h_k = karin_NA.ssha_full[i, :, 5:65].ravel()*100  # full simulation fields in [cm]
    h_obs = h_k

    z  = la.cho_solve((L, lower), h_obs)   # (C_obs)^{-1} h
    ht = R @ z                      # cm
    ht_all[i] = (ht / 100.0).reshape(nyt, nxt)  # back to [m] 
    if (i + 1) % 10 == 0 or i == n_times - 1:
        print(f"  processed {i + 1}/{n_times}")

t.lap("Extraction finished")

# ~~~~~~~~~~~~~Simple plot 
import matplotlib.pyplot as plt
index_plot = 2  
hk_field = swot.compute_laplacian_4th_order(karin_NA.ssha_full[index_plot, :, 5:65])  
ht_field = swot.compute_laplacian_4th_order(ht_all[index_plot])
vmax = 1e-9
vmin = -vmax

fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150, constrained_layout=True)
im0 = axes[0].imshow(hk_field, origin="lower", cmap='RdBu', vmin=vmin, vmax=vmax)
axes[0].set_title(fr"$\nabla^2 h$ simulation")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

im1 = axes[1].imshow(ht_field, origin="lower",cmap='RdBu',vmin=vmin, vmax=vmax)
axes[1].set_title(fr"$\nabla^2 h$ noiseless extraction $\rho = ${RHO_L_KM} km")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label("Lap SSH [m]")
plt.savefig('test.pdf')
print("PLOTTED")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -------------------------
# Save outputs
# -------------------------
with open(f"{OUT_PREFIX}.pkl", "wb") as f:
    pickle.dump(ht_all, f)
t.lap("Balanced field saved")

# -------------------------
# posterior on target (Eq. 10 in paper)
# -------------------------
if COMPUTE_POSTERIOR:
    W = solve_triangular(L, R.T, lower=True, check_finite=False, overwrite_b=False)
    C_mean = W.T @ W                       # covariance of posterior mean R @ C_obs^{-1} @ R.T
    P = R_tt - C_mean                      # posterior covariance
    
    posterior_variance = np.diag(P)
    posterior_variance_field = posterior_variance.reshape(nyt, nxt)

    # ----- Save outputs -----
    km = int(RHO_L_KM)

    # Posterior covariance (full)
    with open(f"{PICKLES_DIR}/posterior_{OUTNAME}.pkl", "wb") as f:
        pickle.dump(P, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior saved: {PICKLES_DIR}/posterior_{OUTNAME}.pkl")

    # Covariance of posterior mean
    with open(f"{PICKLES_DIR}/Cmean_{OUTNAME}.pkl", "wb") as f:
        pickle.dump(C_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"C_mean saved: {PICKLES_DIR}/Cmean_{OUTNAME}.pkl")

    # Posterior variance (vector)
    with open(f"{PICKLES_DIR}/posterior_varvec_{OUTNAME}.pkl", "wb") as f:
        pickle.dump(posterior_variance, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior variance (vector) saved: {PICKLES_DIR}/posterior_varvec_{OUTNAME}.pkl")

    # Posterior variance (field)
    with open(f"{PICKLES_DIR}/posterior_varfield_{OUTNAME}.pkl", "wb") as f:
        pickle.dump(posterior_variance_field, f, protocol=pickle.HIGHEST_PROTOCOL)
    t.lap(f"Posterior variance (field) saved: {PICKLES_DIR}/posterior_varfield_{OUTNAME}.pkl")

    # print("\n ~~~~~C_obs \n")
    # swot.diagnose_not_positive_definite(C_obs)
    # print("~~~~~R_tt")
    # swot.diagnose_not_positive_definite(R_tt)
    # print("~~~~~WT")
    # swot.diagnose_not_positive_definite(W.T @ W)
    # print("~~~~~P")
    #swot.diagnose_not_positive_definite(P)

t.total()
