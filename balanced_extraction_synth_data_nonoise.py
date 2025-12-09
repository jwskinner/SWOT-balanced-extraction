# Balanced extraction on synthetic SWOT KaRIn (with optional Nadir)
# Units in [cm] for covariances/obs, spectra in [cpkm] return [meters] for outputs.
# Put the simulation data through with no SWOT noise and include no noise in the covariances 
# so that the extraction is simply giving us the smoothed field by G(k)
import os, sys
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
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
pass_num = 7                                                                   # SWOT pass number
lat_min  = 28 
lat_max  = 35 

if len(sys.argv) > 1: 
    pass_num = int(sys.argv[1])

SYN_DIR = f"./synthetic_swot_data/Pass_{pass_num:03d}_Lat{lat_min}_{lat_max}/" 
KARIN_NA_PATH    = f"{SYN_DIR}/karin_synth.pkl"
NADIR_NA_PATH    = f"{SYN_DIR}/nadir_synth.pkl"
KARIN_PATH       = f"{SYN_DIR}/karin_swot.pkl"
NADIR_PATH       = f"{SYN_DIR}/nadir_swot.pkl"
RHO_L_KM         = 4.0                                                         # Gaussian spectral taper scale
COMPUTE_POSTERIOR= False                                                       # toggle posterior on target grid
TAPER_CUTOFF     = 2.0                                                         # "T(k)" cutoff
OUTNAME          = f"Pass_{pass_num:03d}_Lat{lat_min}_{lat_max}_rho{int(RHO_L_KM)}km"
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
swot.plot_spectral_fits(karin_NA, nadir_NA, p_karin, p_nadir, f'{OUTDIR}fits_synth.pdf')
t.lap("Spectral fits done")

# -------------------------
# Geometry & masks
# -------------------------
# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin_NA, unit="km", extend=True)
n_t = xt.size
t.lap("Grids and masks done")
print(nxt, nyt)
print(karin_NA.ssha_full[0, :, 3:67].shape)

# -------------------------
# Distance matrices in km
# -------------------------
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
ht_all    = np.full((ntime, nxt, nyt), np.nan, dtype=float)
ug_all    = np.full((ntime, nxt, nyt), np.nan, dtype=float)
vg_all    = np.full((ntime, nxt, nyt), np.nan, dtype=float)
vel_all   = np.full((ntime, nxt, nyt), np.nan, dtype=float)
zetag_all = np.full((ntime, nxt, nyt), np.nan, dtype=float)

for i in range(ntime):
    if (i + 1) % 10 == 0 or i == ntime - 1:
        print(f"  processed {i + 1}/{ntime}")
    h_k = karin_NA.ssha_full[i, :, 3:67].ravel()*100  # full simulation fields in [cm] (subset fits the extended grid)
    h_obs = h_k
    z  = la.cho_solve((L, lower), h_obs)   # (C_obs)^{-1} h
    ht = R @ z                      # cm
    ht_all[i] = ((ht / 100.0).reshape(nyt, nxt).T)   # back to [m] 
    ug, vg, geo_vel, geo_vort = swot.plot_balanced_extraction(karin, nadir, ht_all[i], i, outdir=f"{OUTDIR}/plots_noiseless/")
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
with open(f"{OUTDIR}balanced_extraction_noiseless_pass{pass_num:03d}.pkl", "wb") as f:
    pickle.dump(karin_NA, f)
t.lap("Balanced field saved")

# save the output to a netcdf file 
x_axis = np.arange(nxt)*karin_NA.dx_km
y_axis = np.arange(nyt)*karin_NA.dy_km
time_axis = pd.to_datetime(karin_NA.date_list[:ntime])

ds = xr.Dataset(
    data_vars={
        "ssh_balanced": (("time", "x", "y"), ht_all,    {"units": "m", "description": "Balanced SSH anomaly"}),
        "ug":           (("time", "x", "y"), ug_all,    {"units": "m/s", "description": "Geostrophic Velocity U"}),
        "vg":           (("time", "x", "y"), vg_all,    {"units": "m/s", "description": "Geostrophic Velocity V"}),
        "velocity":     (("time", "x", "y"), vel_all,   {"units": "m/s", "description": "Geostrophic Velocity Magnitude"}),
        "vorticity":    (("time", "x", "y"), zetag_all, {"units": "1/f", "description": "Geostrophic Relative Vorticity"}),
    },
    coords={
        "time": (("time",), time_axis),
        "x":    (("x",), x_axis, {"units": "km", "description": "X distance"}),
        "y":    (("y",), y_axis, {"units": "km", "description": "Y distance"}),
    },
    attrs={
        "title": f"SWOT Balanced Extraction on Noiseless Simulation Data Pass {pass_num}",
        "smoothing_scale_rho": f"{RHO_L_KM} km",
        "lat_range": f"{lat_min} to {lat_max}",
        "creation_date": datetime.now().isoformat(),
    }
)

nc_path = os.path.join(OUTDIR, f"balanced_extraction_noiseless_pass{pass_num:03d}.nc")

ds.to_netcdf(nc_path)
print(f"Data saved to: {nc_path}")

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

t.total()
