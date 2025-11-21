# Balanced extraction on SWOT KaRIn + Nadir with time loop
# Reuses geometry and covariance matrices, subsetting per time
# Units: covariances/obs in [cm], spectra in [cpkm], outputs in [m]

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import JWS_SWOT_toolbox as swot
from JWS_SWOT_toolbox.julia_bridge import julia_functions as jl

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
t = swot.Timer()

data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'
pass_number = 9
lat_min = 28
lat_max = 35
RHO_L_KM = 0.0  # Gaussian smoothing scale; 0 = no smoothing

outdir = f"./SWOT_Pass{pass_number:03d}_Lat{lat_min}_{lat_max}_rho{int(RHO_L_KM)}km"
os.makedirs(outdir, exist_ok=True)
os.makedirs(f"{outdir}/plots", exist_ok=True)

# --------------------------------------------------
# LOAD SWOT FILES
# --------------------------------------------------
# karin_files, nadir_files are sorted by cycle and contain the same cycles
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(
    data_folder, pass_number
)

# Use some file index to get track length
sample_index = 2  # in case the first one has NaNs
indx, track_length = swot.get_karin_track_indices(
    karin_files[sample_index][0], lat_min, lat_max
)
indxs, track_length_nadir = swot.get_nadir_track_indices(
    nadir_files[sample_index][0], lat_min, lat_max
)

dims = [len(shared_cycles), track_length, track_length_nadir]

# --------------------------------------------------
# INIT DATA CLASSES & LOAD DATA
# --------------------------------------------------
karin, nadir = swot.init_swot_arrays(dims, lat_min, lat_max, pass_number)

# Load and process KaRIn
swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)
karin.shared_cycles = shared_cycles  # store

# Load and process Nadir
swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Build coordinate grids in [m]
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()
t.lap("Data loaded, processed, and spectra computed")

# --------------------------------------------------
# SPECTRAL MODEL FITS
# --------------------------------------------------
p_karin, cov_karin = swot.fit_spectrum(
    karin, karin.spec_alongtrack_av, swot.karin_model
)
p_nadir, cov_nadir = swot.fit_nadir_spectrum(
    nadir, nadir.spec_alongtrack_av, p_karin
)

# Plot and save fits
swot.plot_spectral_fits(karin, nadir, p_karin, p_nadir, f'{outdir}/spectral_fits.png')
swot.save_spectral_fit_results(f'{outdir}/spectral_fits.out', p_karin, cov_karin, p_nadir, cov_nadir)
t.lap("Spectral fits complete")

# --------------------------------------------------
# SAVE RAW SWOT DATA (OPTIONAL)
# --------------------------------------------------
swot.save_swot_to_netcdf(karin, nadir, outdir)
t.lap("SWOT data saved to NetCDF")

# --------------------------------------------------
# GEOMETRY & MASTER MASKS
# --------------------------------------------------
mask_k_master_2d = np.isfinite(karin.ssha).any(axis=0)          # (ny, nx)
mask_n_master_1d = np.isfinite(nadir.ssha).any(axis=0).ravel()  # (npoints,)
mask_k_master_flat = mask_k_master_2d.ravel(order="C")
mask_n_master_flat = mask_n_master_1d  # already 1D

xkk_full = (karin.x_grid.ravel(order="C")[mask_k_master_flat]) * 1e-3
ykk_full = (karin.y_grid.ravel(order="C")[mask_k_master_flat]) * 1e-3

xnn_full = (nadir.x_grid.ravel()[mask_n_master_flat]) * 1e-3
ynn_full = (nadir.y_grid.ravel()[mask_n_master_flat]) * 1e-3

n_k_full = xkk_full.size
n_n_full = xnn_full.size

# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin, unit="km", extend=False)
n_t = xt.size

# --------------------------------------------------
# DISTANCE MATRICES
# --------------------------------------------------
r_kk_full = swot.pairwise_r(xkk_full, ykk_full)
r_nn_full = swot.pairwise_r(xnn_full, ynn_full)
r_kn_full = swot.pairwise_r(xkk_full, ykk_full, xnn_full, ynn_full)
r_tk_full = swot.pairwise_r(xt, yt, xkk_full, ykk_full)
r_tn_full = swot.pairwise_r(xt, yt, xnn_full, ynn_full)
r_tt      = swot.pairwise_r(xt, yt)

t.lap("Distance matrices built")

# --------------------------------------------------
# COVARIANCE FUNCTIONS WITH TAPER / SMOOTHING
# --------------------------------------------------
B_psd = swot.balanced_psd_from_params(p_karin)      # B(k) balanced spectrum model
Nk_psd = swot.karin_noise_psd_from_params(p_karin)  # N_K(k) noise spectrum model

sigma_n = np.sqrt(p_nadir[0] / (2.0 * nadir.dy_km))  # [cm]

# Wavenumber grid for transforms
n_samples = 10000
l_sample = 5000
kk = np.arange(n_samples // 2 + 1) / l_sample  # [cpkm]

rho   = 2 * np.pi * RHO_L_KM
delta = (np.pi * karin.dx_km) / (2 * np.log(2))

# Gaussian and taper
G  = lambda k: np.exp(-((rho**2) * (k**2)) / 2.0)                               # G
T  = lambda k: np.exp(-((delta**2) * (k**2)) / 2.0)                             # T
G2 = lambda k: np.exp(-(rho**2) * (k**2))                                       # G^2
GT = lambda k: np.exp(-(((rho**2 + delta**2) * k**2) / 2.0))                    # G*T
T2 = lambda k: np.exp(-(delta**2) * (k**2))                                     # T^2

# Covariance functions C[r] returned as callables
C_B   = jl.cov(B_psd(kk), kk)                                                   # C[B]
C_BT  = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk) * T(kk), kk),kk)                 # C[BT]
C_BG  = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk) * G(kk), kk),kk)                 # C[BG]
C_BG2 = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk) * G2(kk), kk), kk)               # C[BG^2]
C_BTG = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk) * GT(kk), kk), kk)               # C[BGT]
C_BT2 = jl.cov(jl.abel(jl.iabel(B_psd(kk) + Nk_psd(kk), kk) * T2(kk), kk), kk)  # C[BT^2] for KaRIn obs (signal + noise)
t.lap("Covariance functions built")

# --------------------------------------------------
# COVARIANCE BLOCKS
# --------------------------------------------------
# Observation terms
R_KK_full = np.asarray(C_BT2(r_kk_full), dtype=np.float64)                      # KaRIn-KaRIn
R_NN_full = np.asarray(C_B(r_nn_full), dtype=np.float64)                        # Nadir-Nadir signal
R_NN_full += (sigma_n**2) * np.eye(r_nn_full.shape[0], dtype=np.float64)        # + noise

R_KN_full = np.asarray(C_BT(r_kn_full), dtype=np.float64)                       # KaRIn-Nadir
R_NK_full = R_KN_full.T

# Target terms
R_tt  = np.asarray(C_BG2(r_tt), dtype=np.float64)                               # Target-Target
R_tK_full = np.asarray(C_BTG(r_tk_full), dtype=np.float64)                      # Target-KaRIn
R_tN_full = np.asarray(C_BG(r_tn_full), dtype=np.float64)                       # Target-Nadir

# Full observation covariance for all potential points
C_obs_full = np.block([
    [R_KK_full, R_KN_full],
    [R_NK_full, R_NN_full]
])
R_full = np.concatenate([R_tK_full, R_tN_full], axis=1)
t.lap("Covariance matrices built")

# --------------------------------------------------
# LOOP OVER TIME
# --------------------------------------------------
ntimes = karin.ssha.shape[0]
ht_all = np.full((ntimes, nxt, nyt), np.nan, dtype=float)
ug_all = np.full((ntimes, nxt, nyt), np.nan, dtype=float)
vg_all = np.full((ntimes, nxt, nyt), np.nan, dtype=float)
vel_all = np.full((ntimes, nxt, nyt), np.nan, dtype=float)
zetag_all = np.full((ntimes, nxt, nyt), np.nan, dtype=float)

for t_idx in range(ntimes):
    print(f"--- Time index {t_idx+1}/{ntimes} ---")

    mk_t_full_flat = np.isfinite(karin.ssha[t_idx]).ravel(order="C")
    mn_t_full_flat = np.isfinite(nadir.ssha[t_idx]).ravel()

    mk_t = mk_t_full_flat[mask_k_master_flat]
    mn_t = mn_t_full_flat[mask_n_master_flat]

    # Combined observation mask
    obs_mask = np.concatenate([mk_t, mn_t])
    n_obs_t = obs_mask.sum()
    if n_obs_t == 0:
        print(f"Time {t_idx}: no finite KaRIn or Nadir data -> skipping.") # skip if this is a NaN time
        continue

    n_obs_t = obs_mask.sum()
    if n_obs_t == 0:
        print(f"Time {t_idx}: no valid observations, skipping.")
        continue

    # Slice covariance for this time
    C_obs_t = C_obs_full[np.ix_(obs_mask, obs_mask)]
    R_t     = R_full[:, obs_mask]

    # Build observation vector h_obs_t in same master ordering
    # KaRIn
    h_k_full_flat = karin.ssha[t_idx].ravel(order="C")
    h_k_master    = h_k_full_flat[mask_k_master_flat]   # KaRIn master points
    h_k_t         = h_k_master[mk_t] * 100.0            # [cm]

    # Nadir
    h_n_full_flat = nadir.ssha[t_idx].ravel()
    h_n_master    = h_n_full_flat[mask_n_master_flat]   # Nadir master points
    h_n_t         = h_n_master[mn_t] * 100.0            # [cm]

    h_obs_t = np.concatenate([h_k_t, h_n_t])

    # Solve (C_obs_t) z_t = h_obs_t
    try:
        cho_t = la.cho_factor(C_obs_t, lower=True, check_finite=False)
        z_t   = la.cho_solve(cho_t, h_obs_t, check_finite=False)
    except la.LinAlgError as e:
        print(f"Cholesky failed at time {t_idx}: {e}")
        continue

    # Posterior mean on target grid
    ht_t = R_t @ z_t                      
    ht_map_t = (ht_t / 100.0).reshape(nyt, nxt).T   # back to [m], shape (nx, ny)
    ht_all[t_idx] = ht_map_t

    # Plot for this time and store geostrophic velocities and vorticity
    ug, vg, geo_vel, geo_vort = swot.plot_balanced_extraction(karin, nadir, ht_map_t, t_idx, outdir=f"{outdir}/plots/")
    ug_all[t_idx] = ug
    vg_all[t_idx] = vg
    vel_all[t_idx] = geo_vel
    zetag_all[t_idx] = geo_vort

    # store the data
    swot.save_balanced_step_to_netcdf(outdir,karin, t_idx, ht_map_t, ug, vg, geo_vort, xt, yt)
    t.lap("Step Complete")

t.lap("Balanced extraction completed for all times")
