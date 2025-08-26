#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced reconstruction + per-frame plots for SWOT KaRIn/Nadir.

Outputs:
  - balanced_outputs/P{pass}/fields/P{pass}_C{cycle}_{idx}.npy
  - balanced_outputs/P{pass}/plots/fields/P{pass}_C{cycle}_{idx}.png
  - (optional) ./data_outputs/swot_pass{pass}.nc  (single-file snapshot)

Notes:
  - Requires JWS_SWOT_toolbox.
  - Uses joblib for parallel frames; uses Agg backend (headless-safe).
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "4"  
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import psutil, time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from math import sqrt
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")  # headless-safe for HPC
import matplotlib.pyplot as plt
import cmocean
from joblib import Parallel, delayed
from scipy.linalg import block_diag
import JWS_SWOT_toolbox as swot
import xarray as xr
from scipy.ndimage import gaussian_filter

def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

def log_mem(tag):
    print(f"[{time.strftime('%H:%M:%S')}] [pid {os.getpid()}] {tag} | RSS={rss_gb():.2f} GB", flush=True)


# ------------------------- CONFIG -------------------------
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'  # CAL/VAL data root
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE/'  # CAL/VAL data root
pass_number = 507
lat_min = 28
lat_max = 33

# plotting
SSH_VMIN, SSH_VMAX = -0.20, 0.20     # meters
VORT_VMIN, VORT_VMAX = -1.0, 1.0     # zeta/f color limits
PNG_DPI = 200

# parallel
N_JOBS = 4
BACKEND = "loky"

# ---------------------- PASS COMMAND LINE ARGS ----------------------
parser = argparse.ArgumentParser(description="Balanced SWOT reconstruction")
parser.add_argument(
    "--pass", "-p", type=int, dest="pass_num",
    help=f"SWOT pass number (default: {pass_number})"
)
args = parser.parse_args()
if args.pass_num is not None:
    pass_number = args.pass_num

# ---------------------- OUTPUT PATHS ----------------------
def make_output_dirs(pass_num: int):
    root = f"./balanced_outputs/P{pass_num:03d}"
    fields_dir = os.path.join(root, "fields")
    plots_dir = os.path.join(root, "plots")
    os.makedirs(fields_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return root, fields_dir, plots_dir

ROOT_DIR, FIELDS_DIR, PLOTS_DIR = make_output_dirs(pass_number)

# ------------------- DATA IMPORT & PREP -------------------
# Return files for the pass
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(
    data_folder, pass_number
)

# Choose a sample file to determine array extents (skip potential NaN-leading cycles)
sample_index = 2
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims = [len(shared_cycles), track_length, track_length_nadir]

# Init classes
karin, nadir = swot.init_swot_arrays(dims, lat_min, lat_max, pass_number)

# Load/process
swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)

swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Coordinates in meters, spectra, etc.
karin.coordinates()
nadir.coordinates()

karin.compute_spectra()
nadir.compute_spectra()

# --------------------- SPECTRAL FITS ----------------------
# KaRIn (balanced+unbalanced fit)
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)

# Nadir (white noise, using KaRIn fit for guidance)
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)

# Save the spectral fits to the fields directory 
swot.plot_spectral_fits(karin, nadir, poptcwg_karin, poptcwg_nadir, output_filename=os.path.join(ROOT_DIR, 'swot_karin_nadir_fit.pdf'))

# Save the fit parameters 
fit_params = {"karin": poptcwg_karin, "nadir": poptcwg_nadir}
np.save(os.path.join(ROOT_DIR, "fit_params.npy"), fit_params)

A_b, lam_b, s_param = poptcwg_karin[0], poptcwg_karin[1], poptcwg_karin[2]   # balanced params
A_n, s_n, lam_n = poptcwg_karin[3], poptcwg_karin[5], 1e5                    # unbalanced params (lam_n fixed 100km)
N_n = poptcwg_nadir[0]                                                       # nadir white noise level

# ------------------ COVARIANCE DEFINITIONS ----------------
# Balanced S(k)
S_bal = lambda k: A_b / (1.0 + (lam_b * k)**s_param)
c = swot.cov(S_bal, 5_000_000, 10_000e3)  

# Unbalanced S(k) with small-scale taper
cutoff = 1e3
sigma_taper = 2 * np.pi * cutoff / np.sqrt(2 * np.log(2))
S_unb = lambda k: A_n / (1.0 + (lam_n * k)**2)**(s_n / 2) * np.exp(-0.5 * ((sigma_taper**2) * (k**2)))
nk = swot.cov(S_unb, 5_000, 10_000e3)

# Nadir white noise σ^2
delta_n = nadir.dy
sigma_white = np.sqrt(N_n / (2.0 * delta_n))

# --------------------- TARGET GRID ------------------------
xt, yt, nxt, nyt = swot.make_target_grid(karin, extend=False)  # meters, we extend it slightly for the ST
xt_km = xt * 1e-3
yt_km = yt * 1e-3
XX, YY = np.meshgrid(xt_km, yt_km)  # shapes (nyt, nxt)

# ---------------- PLOTTING (per frame) --------------
def plot_frame(ht, index, karin, nadir, shared_cycles, pass_number, nyt, nxt, out_path=None):

    # ── reshape the map and pick colour limits 
    ht_map = ht.reshape(nyt, nxt).T  # shape: (nxt, nyt) -> [y,x]
    vmin, vmax = -0.20, 0.20

    # Axes coordinates in km (x = along-track, y = across-track)
    x_km = np.linspace(0, nyt * karin.dx * 1e-3, ht_map.shape[1])  # columns
    y_km = np.linspace(0, nxt * karin.dy * 1e-3, ht_map.shape[0])  # rows
    XX, YY = np.meshgrid(x_km, y_km)

    # ── figure with adjusted size for equal aspect
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"hspace": 0.4})

    # Get the original data for scatter plot - don't use the processed versions
    karin_data = karin.ssha[index]
    nadir_data = nadir.ssh[index]

    # Create coordinate meshgrids if needed
    if len(karin.x_grid.shape) == 1:
        x_grid_k, y_grid_k = np.meshgrid(karin.x_grid, karin.y_grid)
    else:
        x_grid_k = karin.x_grid
        y_grid_k = karin.y_grid

    # Apply masks and get valid data
    mask_k_2d = np.isfinite(karin_data)
    mask_n_1d = np.isfinite(nadir_data)

    # KaRIn scatter
    x_k_valid = x_grid_k[mask_k_2d].flatten()
    y_k_valid = y_grid_k[mask_k_2d].flatten()
    ssh_k_valid = karin_data[mask_k_2d].flatten()

    # Nadir scatter
    if len(nadir.x_grid.shape) == 1:
        x_n_valid = nadir.x_grid[mask_n_1d]
        y_n_valid = nadir.y_grid[mask_n_1d]
    else:
        x_n_valid = nadir.x_grid.flatten()[mask_n_1d.flatten()]
        y_n_valid = nadir.y_grid.flatten()[mask_n_1d.flatten()]
    ssh_n_valid = nadir_data[mask_n_1d]

    # ── Panel 0: Observed SSH (scatter)
    sc1 = axes[0].scatter(
        y_k_valid * 1e-3, x_k_valid * 1e-3,  # across (y) vs along (x)
        c=ssh_k_valid, s=5, cmap='Spectral',
        vmin=vmin, vmax=vmax, edgecolor="none"
    )
    # overlay nadir with same scale
    axes[0].scatter(
        y_n_valid * 1e-3, x_n_valid * 1e-3,
        c=ssh_n_valid, s=5, cmap='Spectral',
        vmin=vmin, vmax=vmax, edgecolor="none"
    )
    axes[0].set_title("Observed SSH")
    axes[0].set_title(f"Cycle: {shared_cycles[index]}", fontsize=11, loc='right')
    axes[0].set_title(f"Pass: {pass_number:03d}", fontsize=11, loc='left')
    axes[0].set_ylabel("across-track (km)")
    axes[0].margins(x=0, y=0) 
    axes[0].set_aspect("equal")  # Equal aspect ratio
    cbar0 = fig.colorbar(sc1, ax=axes[0], orientation='vertical', shrink=0.7, pad=0.02)
    cbar0.set_ticks([vmin, 0, vmax])
    cbar0.set_ticklabels([f'{vmin:.2f}', '0.00', f'{vmax:.2f}'])

    # ── Diagnostics
    lats = np.linspace(np.nanmin(karin.lat[index, :, :]), np.nanmax(karin.lat[index, :, :]), ht_map.shape[0])
    # sigma_pixels = 1.0 
    # ssh_smoothed = gaussian_filter(np.ma.masked_invalid(ht_map), sigma=sigma_pixels)
    geo_vort = swot.compute_geostrophic_vorticity_5pt(np.ma.masked_invalid(ht_map), karin.dx, karin.dy, lats)
    grad_mag = swot.compute_gradient_magnitude_5point(ht_map, karin.dx, karin.dy)
    ssh_levels  = np.linspace(vmin, vmax, 400)

    # Gradient: force vmin to include 0 for the 0-tick
    grad_data_min = float(np.nanmin(grad_mag))
    grad_data_max = 0.8*float(np.nanmax(grad_mag))
    grad_vmin = min(0.0, grad_data_min)
    grad_vmax = 0.9* grad_data_max
    grad_levels = np.linspace(grad_vmin, grad_vmax, 100)

    # Vorticity: choose symmetric limits around 0 (you had these already)
    vort_vmin, vort_vmax = -1.0, 1.0
    vort_levels = np.linspace(vort_vmin, vort_vmax, 100)

    # ── Panel 1: Extracted Balanced SSH (contourf)
    cf0 = axes[1].contourf(XX, YY, ht_map, levels=ssh_levels, cmap='Spectral', extend='both')
    axes[1].set_title("Extracted Balanced SSH")
    axes[1].set_ylabel("across-track (km)")
    axes[1].set_aspect("equal")
    cbar1 = fig.colorbar(cf0, ax=axes[1], orientation='vertical', shrink=0.7, pad=0.02)
    cbar1.set_ticks([vmin, 0, vmax])
    cbar1.set_ticklabels([f'{vmin:.2f}', '0.00', f'{vmax:.2f}'])

    # ── Panel 2: Gradient magnitude (contourf) — three ticks: min, 0, max
    cf1 = axes[2].contourf(XX, YY, grad_mag, levels=grad_levels, cmap='cmo.deep_r', extend='both')
    axes[2].set_title("Gradient Magnitude")
    axes[2].set_ylabel("across-track (km)")
    axes[2].set_aspect("equal")
    cbar2 = fig.colorbar(cf1, ax=axes[2], orientation='vertical', shrink=0.7, pad=0.02)
    cbar2.set_ticks([grad_vmin, 0, grad_vmax])
    cbar2.set_ticklabels([f'{grad_vmin:.2e}', '0', f'{grad_vmax:.2e}'])
    cbar2.set_label(r'$|\nabla \mathrm{SSH}|$')

    # ── Panel 3: Geostrophic Vorticity (contourf) — three ticks: min, 0, max
    cf2 = axes[3].contourf(XX, YY, geo_vort, levels=vort_levels, cmap=cmocean.cm.balance, extend='both')
    axes[3].set_title("Geostrophic Vorticity")
    axes[3].set_xlabel("along-track distance (km)")
    axes[3].set_ylabel("across-track (km)")
    axes[3].set_aspect("equal")
    cbar3 = fig.colorbar(cf2, ax=axes[3], orientation='vertical', shrink=0.7, pad=0.02)
    cbar3.set_ticks([vort_vmin, 0, vort_vmax])
    cbar3.set_ticklabels([f'{vort_vmin:.2f}', '0.00', f'{vort_vmax:.2f}'])
    cbar3.set_label(r'$\zeta / f$')

    # save/show
    if out_path is None:
        out_path = "balanced_extraction.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_spectrum_comparison(karin_obj, swot_obj, poptcwg_karin_params, ntx, nyt, ht_map, out_path=None):

    ht_map_2d = ht_map.reshape(nyt, nxt).T
    
    # Extract KaRIn wavenumbers and sample mean spectrum
    k_karin = karin_obj.wavenumbers[int(karin_obj.track_length/2):]
    karin_spec_sample_mean = karin_obj.spec_alongtrack_av[int(karin_obj.track_length/2):]

    # Ensure consistent slicing for model inputs (skipping the first wavenumber)
    k_karin_sliced = k_karin[1:]

    # Put the wavenumbers through the models to get the functional form
    spbalanced = swot_obj.balanced_model(k_karin_sliced, *poptcwg_karin_params[0:3])
    spunbalanced = swot_obj.unbalanced_model(k_karin_sliced, *poptcwg_karin_params[3:7])

    nx_dim = ht_map_2d.shape[1] # Along-track dimension
    ny_dim = ht_map_2d.shape[0] # Across-track dimension 
    ht_map_coords = {
        'pixel': np.arange(0, ny_dim) * karin_obj.dy, # Across-track coordinate
        'line': np.arange(0, nx_dim) * karin_obj.dx   # Along-track coordinate
    }
    ht_map_xr = xr.DataArray(ht_map_2d, coords=ht_map_coords, dims=['pixel', 'line'])
    spec_ht_map_2s = swot_obj.mean_power_spectrum(ht_map_xr, karin_obj.window, 'line', ['pixel'])
    spec_ht_map = spec_ht_map_2s[int(karin_obj.track_length/2):][1:]

    # --- Plotting ---
    fig, axs = plt.subplots(1, 1, figsize=(6, 5), dpi=150, constrained_layout=True)
    k_km = k_karin_sliced * 1e3
    axs.loglog(k_km, karin_spec_sample_mean[1:], 'o', label='KaRIn SSHA')
    axs.loglog(k_km, spunbalanced,
                  label=r'$A_n$=%5.1f, $\lambda_n$=%5.1f, $S_n$=%5.1f' %
                  (poptcwg_karin_params[3], 100, poptcwg_karin_params[5]))
    axs.loglog(k_km, spbalanced,
                  label=r'$A_b$=%5.1f, $\lambda_b$=%5.1f, $S_b$=%5.1f' %
                  (poptcwg_karin_params[0], poptcwg_karin_params[1]*1e-3, poptcwg_karin_params[2]))
    axs.loglog(k_km, (spunbalanced + spbalanced), '--', label='Model (sum)')
    axs.loglog(k_km, spec_ht_map, '-', lw=2, label='Extracted Balanced Flow')
    axs.set_xlabel('wavenumber (cpkm)')
    axs.set_ylabel('PSD (m$^2$ cpm$^{-1}$)')
    axs.set_xlim(1e-3, 3e-1)
    axs.set_ylim(1e-3, 1e4)
    axs.legend(loc='lower left', frameon=False, fontsize=9)

    # save
    if out_path is None:
        out_path = "spec_comp.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig) 

# --------------- PER-FRAME ESTIMATION (worker) -------------
def process_frame(idx: int):
    """
    Builds per-frame covariance blocks, estimates ht, saves npy + png.
    """
    # --- Masks
    mask_k = np.isfinite(karin.ssha[idx])            # 2D
    mask_n = np.isfinite(nadir.ssh[idx]).ravel()     # 1D

    log_mem("start")

    # --- KaRIn (2D) valid
    hkk = karin.ssha[idx][mask_k].ravel()
    xkk = karin.x_grid[mask_k].ravel()
    ykk = karin.y_grid[mask_k].ravel()

    # --- Nadir (1D) valid
    hn = np.ravel(nadir.ssh[idx])
    xn = np.ravel(nadir.x_grid)
    yn = np.ravel(nadir.y_grid)

    hnn = hn[mask_n]
    xnn = xn[mask_n]
    ynn = yn[mask_n]

    # --- Concatenate obs
    h_obs = np.concatenate([hkk, hnn])
    xobs  = np.concatenate([xkk, xnn])
    yobs  = np.concatenate([ykk, ynn])

    # --- Signal covariance (all obs)
    C_obs = swot.build_covariance_matrix(c, xobs, yobs)

    log_mem("cov")

    # --- Noise: KaRIn correlated + Nadir white
    dxk = xkk[:, None] - xkk[None, :]
    dyk = ykk[:, None] - ykk[None, :]
    Nk_obs = nk(np.hypot(dxk, dyk))  # correlated KaRIn
    Nn_obs = (sigma_white**2) * np.eye(len(xnn))  # nadir white
    N_obs = block_diag(Nk_obs, Nn_obs)

    # --- Target grid + estimate
    ht_vec = swot.estimate_signal_on_target(c, xt, yt, xobs, yobs, C_obs, N_obs, h_obs)

    log_mem("estimate")

    # --- SAVE FIELDS AND PLOTS 
    out_npy = os.path.join(FIELDS_DIR, f"P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
    np.save(out_npy, ht_vec.reshape(nyt, nxt).astype("f4"))

    frame_path = os.path.join(
        PLOTS_DIR, f"frame_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
        )
    spec_path = os.path.join(
        PLOTS_DIR, f"spec_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
        )
    
    plot_frame(
        ht=ht_vec,                
        index=idx,
        karin=karin,
        nadir=nadir,
        shared_cycles=shared_cycles,
        pass_number=pass_number,
        nyt=nyt, nxt=nxt,
        out_path=frame_path
    )

    plot_spectrum_comparison(
        karin_obj=karin,
        swot_obj=swot,
        poptcwg_karin_params=poptcwg_karin,
        ntx=nxt,
        nyt=nyt,
        ht_map=ht_vec, 
        out_path=spec_path

    )

    print(f"Frame {idx:03d} → npy: {os.path.basename(out_npy)} | png: {os.path.basename(frame_path)} | png: {os.path.basename(spec_path)}")
    return out_npy, frame_path

# -------------------------- RUN ---------------------------
if __name__ == "__main__":
    n_frames = karin.ssha.shape[0]

    results = Parallel(n_jobs=N_JOBS, backend=BACKEND)(
        delayed(process_frame)(idx) for idx in range(n_frames)  # test run
    )

# ----------------- write a NetCDF output -----------------
try:
    T = len(results)
    print(f"Processing {T} frames for NetCDF output")
    processed_indices = list(range(T)) 
    
    ht_stack = np.empty((T, nyt, nxt), dtype="f4")
    for i, idx in enumerate(processed_indices):
        path = os.path.join(FIELDS_DIR, f"P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
        ht_stack[i] = np.load(path)

    # Times
    time_coords = np.arange(T, dtype=np.float64)
    cycle_numbers = [shared_cycles[idx] for idx in processed_indices]
    datetime_data = [karin.time_dt[idx] for idx in processed_indices]  # Actual datetime info
    
    x_coords = np.linspace(xt_km.min(), xt_km.max(), nxt)  # along-track, length nxt
    y_coords = np.linspace(yt_km.min(), yt_km.max(), nyt)  # across-track, length nyt
    karin_x_coords = np.linspace(xt_km.min(), xt_km.max(), karin.total_width)  # along-track, length nxt
    karin_y_coords = np.linspace(yt_km.min(), yt_km.max(), karin.track_length)  # across-track, length nyt

    if "CALVAL" in data_folder.upper():
        mission_phase = "CalVal"
    else:
        mission_phase = "Science"
    
    # Create the xarray Dataset
    ds = xr.Dataset(
        {
            'balanced_ssh': (['time', 'y', 'x'], ht_stack, {
                'long_name': 'Balanced Sea Surface Height',
                'units': 'm',
                'description': 'Extracted balanced component of SSH using optimal interpolation'
            }),
            
            'karin_ssha': (['time', 'karin_y', 'karin_x'], karin.ssha[processed_indices], {
                'long_name': 'KaRIn Sea Surface Height Anomaly',
                'units': 'm',
                'description': 'Original KaRIn SSHA observations'
            }),
            
            'nadir_ssh': (['time', 'nadir_along'], nadir.ssh[processed_indices], {
                'long_name': 'Nadir Altimeter Sea Surface Height',
                'units': 'm',
                'description': 'Original Nadir SSH observations'
            }),
            
            'karin_lon': (['time', 'karin_y', 'karin_x'], karin.lon[processed_indices], {
                'long_name': 'KaRIn Longitude',
                'units': 'degrees_east'
            }),
            'karin_lat': (['time', 'karin_y', 'karin_x'], karin.lat[processed_indices], {
                'long_name': 'KaRIn Latitude',
                'units': 'degrees_north'
            }),
            'nadir_lon': (['time', 'nadir_along'], nadir.lon[processed_indices], {
                'long_name': 'Nadir Longitude',
                'units': 'degrees_east'
            }),
            'nadir_lat': (['time', 'nadir_along'], nadir.lat[processed_indices], {
                'long_name': 'Nadir Latitude',
                'units': 'degrees_north'
            }),
            
            'cycle_number': (['time'], cycle_numbers, {
                'long_name': 'SWOT Cycle Number',
                'description': 'Original SWOT repeat cycle numbers',
                'units': 'cycle'
            }),
            
            'time_datetime': (['time'], datetime_data, {
                'long_name': 'Acquisition DateTime',
                'description': 'Actual datetime of data acquisition'
            }),
        },
        coords={
            'time': ('time', time_coords, {
                'long_name': 'Time Index',
                'units': 'index',
                'description': 'Sequential integer time index for ParaView animation',
                'standard_name': 'time',
                'axis': 'T',
                '_CoordinateAxisType': 'Time'
            }),
            'x': ('x', x_coords, {
                'long_name': 'Along-track Distance',
                'units': 'km',
                'description': 'Distance along satellite track'
            }),
            'y': ('y', y_coords, {
                'long_name': 'Across-track Distance', 
                'units': 'km',
                'description': 'Distance across satellite track'
            }),
            'karin_x': ('karin_x', karin_x_coords, {
                'long_name': 'KaRIn across-track pixel index'
            }),
            'karin_y': ('karin_y', karin_y_coords, {
                'long_name': 'KaRIn along-track pixel index'
            }),
            'nadir_along': ('nadir_along', np.arange(nadir.ssh.shape[1])*nadir.dy_km, {
                'long_name': 'Nadir along-track index'
            })
        },
        attrs={
            'title': f'SWOT Balanced SSH Extraction - Pass {pass_number:03d}',
            'description': 'Balanced component of sea surface height extracted from SWOT KaRIn and Nadir data using optimal interpolation',
            'pass_number': pass_number,
            'latitude_range': f'{lat_min}°N to {lat_max}°N',
            'processing_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'spatial_resolution_x': f'{karin.dx_km:.3f} km',
            'spatial_resolution_y': f'{karin.dy_km:.3f} km',
            'n_cycles_processed': T,
            'mission_phase': mission_phase,
            'balanced_correlation_length': f'{lam_b/1000:.1f} km',
            'unbalanced_correlation_length': f'{lam_n/1000:.1f} km',
            'signal_variance_balanced': f'{A_b:.4f} m²',
            'signal_variance_unbalanced': f'{A_n:.4f} m²',
            'nadir_noise_std': f'{sigma_white:.4f} m',
            'balanced_params_A_b': float(poptcwg_karin[0]),
            'balanced_params_lambda_b': float(poptcwg_karin[1]),
            'balanced_params_s_param': float(poptcwg_karin[2]),
            'unbalanced_params_A_n': float(poptcwg_karin[3]),
            'unbalanced_params_lambda_n': float(poptcwg_karin[4]),
            'unbalanced_params_s_n': float(poptcwg_karin[5]),
            'source_code': 'balanced_swot_reconstruction.py',
            'source': 'JWS, Caltech 2025',
        }
    )
    
    # Set up compression for all data variables
    comp = dict(zlib=True, complevel=4, shuffle=True)
    encoding = {v: comp for v in ds.data_vars}
    encoding['time'] = {'dtype': 'float64', 'zlib': True, 'complevel': 4}
    
    # Write the NetCDF file
    out_nc = os.path.join(ROOT_DIR, f"swot_pass{pass_number:03d}.nc")
    ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding, unlimited_dims=['time'])
    print(f"[NetCDF] Wrote {out_nc}")

except Exception as e:
    print(f"[NetCDF] Error writing compact snapshot: {e}")
    import traceback
    traceback.print_exc()