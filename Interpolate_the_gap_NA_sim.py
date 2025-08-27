#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced reconstruction tests on the NA simulation data. The functions in this script are tested in "generate_NA_SWOT_data_simple.ipynb" 

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
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs


# ------------------------- CONFIG -------------------------
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'  # CAL/VAL data root
#data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE/'  # CAL/VAL data root
NA_folder = "/expanse/lustre/projects/cit197/jskinner1/NA_daily_snapshots"

pass_number = 9
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
    root = f"./NA-Sim_balanced_outputs/P{pass_num:03d}"
    fields_dir = os.path.join(root, "fields")
    plots_dir = os.path.join(root, "plots")
    os.makedirs(fields_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return root, fields_dir, plots_dir

ROOT_DIR, FIELDS_DIR, PLOTS_DIR = make_output_dirs(pass_number)

# ------------------- SWOT DATA IMPORT & PREP -------------------
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_number)

for sample_index in range(len(karin_files)): # find a valid file and use its structure to setup the arrays
    try:
        indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
        indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
        break  # success, we will use sample_index throughout now for plots etc.
    except IndexError:
        continue
else:
    raise RuntimeError("No valid index found in karin/nadir files")

dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]
karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_number)

swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)

swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Generate coordinates
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()

print(f"Loaded SWOT pass {pass_number:03d} | cycles {shared_cycles[0]:03d}-{shared_cycles[-1]:03d} | lat {lat_min}°N to {lat_max}°N")

# ------------------- NA SIMULATION DATA IMPORT & PREP -------------------
# 1) choose sim dates for KaRIn times
_, _, matched_dates = swot.pick_range_from_karin_times(
    karin_time_dt=karin.time_dt,
    data_folder=NA_folder,
    mode="cyclic"    # or 'absolute' if sim year == SWOT year
)

# 2) interpolate each sim day onto the KaRIn and Nadir grids
#  Match the simulation dates with the SWOT dates
NA_folder = "/expanse/lustre/projects/cit197/jskinner1/NA_daily_snapshots"

# -- choose sim dates for KaRIn times
_, _, matched_dates = swot.pick_range_from_karin_times(
    karin_time_dt=karin.time_dt,
    data_folder=NA_folder,
    mode="cyclic"    # or 'absolute' if sim year == SWOT year
)

# -- interpolate each sim day onto the KaRIn grid - we get a full grid (the pass), one with a gap (KaRIn), and the nadir points
NA_karin_full_ssh, NA_karin_ssh, NA_nadir_ssh, used_dates = swot.load_sim_on_karin_nadir_grids(
    karin, 
    nadir, 
    data_folder=NA_folder, 
    matched_dates=matched_dates 
)

# Now the data is processed we can init all out data classes with the NA simulation data (NA_Karin/Nadir)
ncycles = NA_karin_ssh.shape[0]
track_length_karin =  NA_karin_ssh.shape[1]
track_length_nadir = NA_nadir_ssh.shape[1]
dims_NA = [ncycles, track_length_karin, track_length_nadir]

karin_NA, nadir_NA = swot.init_swot_arrays(dims_NA, lat_min, lat_max, pass_number) # init a class for the karin/nadir parts of the data
karin_NA.ssh_orig = NA_karin_full_ssh # save the original ssh
karin_NA.ssh = NA_karin_ssh
karin_NA.ssha = NA_karin_ssh - np.nanmean(NA_karin_ssh, axis=(1, 2), keepdims=True)
karin_NA.lat = karin.lat  # init the NA pass with the KaRIn lat/lon grids
karin_NA.lon = karin.lon
karin_NA.date_list=matched_dates  

nadir_NA.ssh = NA_nadir_ssh
nadir_NA.ssha = NA_nadir_ssh - np.nanmean(NA_nadir_ssh, axis=(1), keepdims=True)
nadir_NA.lat = nadir.lat
nadir_NA.lon = nadir.lon

karin_NA.coordinates()
nadir_NA.coordinates()

karin_NA.compute_spectra()
nadir_NA.compute_spectra()

print(f"Loaded NA sim data for {len(used_dates)} dates")

# ------ FIGURE: SWOT vs Sim data + Spectra ------
swot.set_plot_style()

vmin, vmax = -0.5, 0.5
ylims = (1e-5, 1e5)
cmap = cmocean.cm.balance

fig = plt.figure(figsize=(18, 6), dpi=150)
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.6], figure=fig)

# ───── Simulation Map ─────
ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
sc0 = ax0.scatter(
    karin.lon[sample_index].flatten(), karin.lat[sample_index].flatten(),
    c=karin_NA.ssha[sample_index].flatten(), s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o'
)
ax0.scatter(
    nadir.lon[sample_index], nadir.lat[sample_index], c=nadir_NA.ssha[sample_index], vmin=vmin, vmax=vmax,
    cmap=cmap, s=1, marker='o', transform=ccrs.PlateCarree()
)
ax0.coastlines()
ax0.set_title("NA Simulation KaRIn + Nadir")
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
gl0.top_labels = gl0.right_labels = False
cbar0 = fig.colorbar(sc0, ax=ax0, orientation='vertical', shrink=0.7, pad=0.03)
cbar0.set_label("SSHA (m)")

# ───── SWOT Map ─────
ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
sc1 = ax1.scatter(
    karin.lon[sample_index], karin.lat[sample_index],
    c=karin.ssha[sample_index], s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o'
)
ax1.scatter(
    nadir.lon[sample_index], nadir.lat[sample_index],
    c=nadir.ssh[sample_index], s=1, vmin=vmin, vmax=vmax,
    cmap=cmap, transform=ccrs.PlateCarree(), marker='o'
)
ax1.coastlines()
ax1.set_title(f'SWOT KaRIn + Nadir')
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
gl1.top_labels = gl1.right_labels = False
cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', shrink=0.7, pad=0.03)
cbar1.set_label("SSHA (m)")

# ───── Power Spectrum ─────
ax2 = fig.add_subplot(gs[0, 2])
ax2.loglog(karin_NA.wavenumbers_cpkm, karin_NA.spec_alongtrack_av, label='Sim KaRIn SSH', linewidth=2)
ax2.loglog(nadir_NA.wavenumbers_cpkm, nadir_NA.spec_alongtrack_av, label='Sim Nadir SSH', linewidth=2)
#ax2.loglog(karin_NA.wavenumbers * 1e3, karin_NA.spec_ssh, label='Sim KaRIn SSH', linewidth=1.5)

ax2.loglog(karin.wavenumbers_cpkm, karin.spec_alongtrack_av, label='SWOT KaRIn SSHA', linewidth=2)
ax2.loglog(nadir.wavenumbers_cpkm, nadir.spec_alongtrack_av, label='SWOT Nadir SSHA', linewidth=2)
ax2.loglog(karin.wavenumbers_cpkm, karin.spec_ssh, label='SWOT KaRIn SSH', linewidth=2.0)
ax2.loglog(karin.wavenumbers_cpkm, karin.spec_tmean, label='SWOT Time-mean', linewidth=2.0)
#ax2.loglog(karin.wavenumbers * 1e3, karin.spec_filt_tmean, label='SWOT Filtered', linewidth=1.2)
ax2.loglog(karin.wavenumbers_cpkm, karin.spec_tide, label='SWOT HRET', linewidth=2.0)

ax2.set_xlabel("Wavenumber (cpkm)")
ax2.set_ylabel("PSD (m$^2$/cpkm)")
ax2.set_ylim(ylims)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
ax2.set_title("Power Spectra")
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "swot_vs_sim_comparison.pdf"), dpi=200, bbox_inches="tight")

# --------------------- SPECTRAL FITS SWOT DATA ----------------------
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model) # KaRIn model fit
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin) # Nadir model fit
swot.plot_spectral_fits(karin, nadir, poptcwg_karin, poptcwg_nadir, output_filename=os.path.join(ROOT_DIR, 'swot_karin_nadir_fit.pdf'))

# Save the fit parameters 
fit_params = {"karin": poptcwg_karin, "nadir": poptcwg_nadir}
np.save(os.path.join(ROOT_DIR, "SWOT_fit_params.npy"), fit_params)

# ───── Generate Synthetic Fields from the SWOT Models ─────
A_b, lam_b, s_b = poptcwg_karin[0], poptcwg_karin[1], poptcwg_karin[2]
A_n, s_n, lam_n = poptcwg_karin[3], poptcwg_karin[5], 1e5  # lam_n fixed at 100 km
N_n = poptcwg_nadir[0]

# --- Grid and Spacing ---
nx, ny = 2 * karin.swath_width, karin.track_length
nn = nadir.track_length
dx, dy, dn = karin.dx, karin.dy, nadir.dy

# --- Covariance Functions ---
S_bal = lambda k: A_b / (1 + (lam_b * k)**s_b)
sigma_taper = 2 * np.pi * 1e3 / np.sqrt(2 * np.log(2))
S_unb = lambda k: A_n / (1 + (lam_n * k)**2)**(s_n/2) * np.exp(-0.5 * (sigma_taper**2) * k**2)
sigma_noise = np.sqrt(N_n / (2 * dn))

c_bal = swot.cov(S_bal, 5000000, 10000e3)
c_unb = swot.cov(S_unb, 5000, 10000e3)

# --- Observation Points ---
xk, yk = karin.x_obs_grid.flatten(), karin.y_obs_grid.flatten()
xn, yn = nadir.x_grid.flatten(), nadir.y_grid.flatten()
xobs = np.concatenate((xk, xn))
yobs = np.concatenate((yk, yn))

# --- Covariance and Noise Matrices ---
C = swot.build_covariance_matrix(c_bal, xobs, yobs)
N, Nk = swot.build_noise_matrix(c_unb, xk, yk, sigma_noise, nn, nx*ny)

# --- Cholesky Decomposition ---
F = swot.cholesky_decomp(C, "C")
Fk = swot.cholesky_decomp(Nk, "Nk")

# --- Use SWOT Covariance to generate n realizations of the random signal and noise ---
n_realizations = karin_NA.ssha.shape[0]
hs, etas, etas_k, etas_n = swot.generate_synthetic_realizations(swot, F, Fk, sigma_noise, nx, ny, nn, n_realizations)

# --------------------- GENERATE SYNTHETIC SWOT DATA ----------------------
ssh_noisy = np.empty_like(karin_NA.ssha) # new arrays for holding the synthetic SWOT NA data
ssh_nadir_noisy = np.empty_like(nadir_NA.ssha)

for t in range(0, n_realizations):

    ssh = karin_NA.ssha[t, :, :] 
    mask = np.isfinite(ssh)      # mask the gap out

    eta_k_reshaped = np.full_like(ssh, np.nan)  
    eta_k_current = etas_k[t, :, :] 

    # Direct assignment of flattened valid data:
    eta_k_reshaped[mask] = eta_k_current.flatten()

    # Add noise to SSH, preserving gaps:
    ssh_noisy[t, :, :] = ssh + eta_k_reshaped

    # --- Nadir ---
    ssh_nadir = nadir_NA.ssha[t, :]
    ssh_nadir_noisy[t] = ssh_nadir + etas_n[t]

karin_NA.ssh_noisy = ssh_noisy # save the generated noisy fields to our NA simulation classes 
nadir_NA.ssh_noisy = ssh_nadir_noisy

# --------------------- FIT SPECTRAL MODEL TO SYNTHETIC SWOT DATA ----------------------
kt_NA_coords    = [np.arange(n_realizations), karin.y_coord, karin.x_coord]
ssh_noisy_xr = xr.DataArray(ssh_noisy, coords = kt_NA_coords, dims = ['sample', 'line', 'pixel'])
spec_ssh_noisy = swot.mean_power_spectrum(ssh_noisy_xr, karin.window, 'line', ['sample', 'pixel'])

poptcwg_karin_NA, pcovcwg_karin_NA = swot.fit_spectrum(karin_NA, spec_ssh_noisy, swot.karin_model) 
swot.plot_spectral_fits(karin_NA, nadir_NA, poptcwg_karin_NA, poptcwg_nadir, output_filename=os.path.join(ROOT_DIR, 'synth_karin_nadir_fit.pdf'))

# --------------------- TARGET GRID ------------------------
xt, yt, nxt, nyt, x_target, y_target = swot.make_target_grid(karin, extend=True)  # meters, we extend it slightly for the ST
xt_km = xt * 1e-3
yt_km = yt * 1e-3
XX, YY = np.meshgrid(xt_km, yt_km)  # shapes (nyt, nxt)

# --------------- PER-FRAME ESTIMATION (on SWOT + Synth) -------------
def process_frame(idx: int):
    # SWOT Data 
    # --- Masks
    mask_k = np.isfinite(karin.ssha[idx])            # 2D
    mask_n = np.isfinite(nadir.ssh[idx]).ravel()     # 1D

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

    # --- skip frame if no finite values
    if hkk.size == 0 or hnn.size == 0:
        print(f"Frame {idx:03d} skipped (no finite SWOT data)")
        return None

    # --- Concatenate obs
    h_obs = np.concatenate([hkk, hnn])
    xobs  = np.concatenate([xkk, xnn])
    yobs  = np.concatenate([ykk, ynn])

    # --- Signal covariance (all obs)
    C_obs = swot.build_covariance_matrix(c_bal, xobs, yobs)

    # --- Noise: KaRIn correlated + Nadir white
    dxk = xkk[:, None] - xkk[None, :]
    dyk = ykk[:, None] - ykk[None, :]
    Nk_obs =c_unb(np.hypot(dxk, dyk))  # correlated KaRIn
    Nn_obs = (sigma_noise**2) * np.eye(len(xnn))  # nadir white
    N_obs = block_diag(Nk_obs, Nn_obs)

    # --- Target grid + estimate
    ht_vec = swot.estimate_signal_on_target(c_bal, xt, yt, xobs, yobs, C_obs, N_obs, h_obs)

    # --- SAVE FIELDS AND PLOTS 
    out_npy = os.path.join(FIELDS_DIR, f"SWOT_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
    np.save(out_npy, ht_vec.reshape(nyt, nxt).astype("f4"))

    frame_path = os.path.join(
        PLOTS_DIR, f"SWOT_frame_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
        )
    spec_path = os.path.join(
        PLOTS_DIR, f"SWOT_spec_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
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

    print(f"Frame SWOT  {idx:03d} → npy: {os.path.basename(out_npy)} | png: {os.path.basename(frame_path)} | png: {os.path.basename(spec_path)}")
    
    # SYNTHETIC Data 
    # --- Masks
    mask_k = np.isfinite(karin_NA.ssh_noisy[idx])        # we put our noisy NA Sim data in here to extract on it
    mask_n = np.isfinite(nadir_NA.ssh_noisy[idx]).ravel()

    # --- KaRIn (2D) valid
    hkk = karin_NA.ssh_noisy[idx][mask_k].ravel()
    xkk = karin_NA.x_grid[mask_k].ravel()
    ykk = karin_NA.y_grid[mask_k].ravel()

    # --- Nadir (1D) valid
    hn = np.ravel(nadir_NA.ssh_noisy[idx])
    xn = np.ravel(nadir_NA.x_grid)
    yn = np.ravel(nadir_NA.y_grid)

    hnn = hn[mask_n]
    xnn = xn[mask_n]
    ynn = yn[mask_n]

    # --- Concatenate obs
    h_obs = np.concatenate([hkk, hnn])
    xobs  = np.concatenate([xkk, xnn])
    yobs  = np.concatenate([ykk, ynn])

    # --- Signal covariance (all obs)
    C_obs = swot.build_covariance_matrix(c_bal, xobs, yobs)

    # --- Noise: KaRIn correlated + Nadir white
    dxk = xkk[:, None] - xkk[None, :]
    dyk = ykk[:, None] - ykk[None, :]
    Nk_obs = c_unb(np.hypot(dxk, dyk))  # correlated KaRIn
    Nn_obs = (sigma_noise**2) * np.eye(len(xnn))  # nadir white
    N_obs = block_diag(Nk_obs, Nn_obs)

    # --- Target grid + estimate
    ht_syn = swot.estimate_signal_on_target(c_bal, xt, yt, xobs, yobs, C_obs, N_obs, h_obs)

    # --- SAVE FIELDS AND PLOTS 
    out_npy = os.path.join(FIELDS_DIR, f"SYN_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
    np.save(out_npy, ht_syn.reshape(nyt, nxt).astype("f4"))

    frame_path = os.path.join(
        PLOTS_DIR, f"SYN_frame_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
        )
    spec_path = os.path.join(
        PLOTS_DIR, f"SYN_spec_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.png"
        )
    
    plot_frame(
        ht=ht_syn,                
        index=idx,
        karin=karin_NA,
        nadir=nadir_NA,
        shared_cycles=shared_cycles,
        pass_number=pass_number,
        nyt=nyt, nxt=nxt,
        out_path=frame_path
    )

    plot_spectrum_comparison(
        karin_obj=karin_NA,
        swot_obj=swot,
        poptcwg_karin_params=poptcwg_karin_NA,
        ntx=nxt,
        nyt=nyt,
        ht_map=ht_syn, 
        out_path=spec_path

    )

    print(f"Frame SYN  {idx:03d} → npy: {os.path.basename(out_npy)} | png: {os.path.basename(frame_path)} | png: {os.path.basename(spec_path)}")
    
    return out_npy, frame_path

# ---------------- PLOTTING (per frame) --------------
def plot_frame(ht, index, karin, nadir, shared_cycles, pass_number, nyt, nxt, out_path=None):

    def _time_or_static(arr, index):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[index]
        elif arr.ndim == 2:
            return arr
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

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
    if np.ndim(karin.lat) == 3:
        lats = np.linspace(np.nanmin(karin.lat[index]), np.nanmax(karin.lat[index]), ht_map.shape[0])
    else:  # 2-D (L, W)
        lats = np.linspace(np.nanmin(karin.lat), np.nanmax(karin.lat), ht_map.shape[0])
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
    axs.set_ylim(1e-3, 1e5)
    axs.legend(loc='lower left', frameon=False, fontsize=9)

    # save
    if out_path is None:
        out_path = "spec_comp.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig) 

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

    # stack list of 2D arrays (pad if needed) 
    def _stack_2d_list(arr_list, dtype="f4"):
        arrs = [np.asarray(a) for a in arr_list]
        ny = max(a.shape[0] for a in arrs)
        nx = max(a.shape[1] for a in arrs)
        out = np.full((len(arrs), ny, nx), np.nan, dtype=dtype)
        for i, a in enumerate(arrs):
            out[i, :a.shape[0], :a.shape[1]] = a
        return out

    # returns array floats for data output
    def _f(x): return float(np.asarray(x))

    # ---- gather balanced fields (SWOT and SYN) saved per-frame ----
    ht_swot_stack = np.empty((T, nyt, nxt), dtype="f4")
    ht_syn_stack  = np.empty((T, nyt, nxt), dtype="f4")
    for i, idx in enumerate(processed_indices):
        p_swot = os.path.join(FIELDS_DIR, f"SWOT_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
        p_syn  = os.path.join(FIELDS_DIR, f"SYN_P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy")
        ht_swot_stack[i] = np.load(p_swot)
        ht_syn_stack[i]  = np.load(p_syn)

    # ---- NA Sim SSH along the pass ----
    orig_ssh_stack = _stack_2d_list([karin_NA.ssh_orig[i]  for i in processed_indices], dtype="f4")
    ny_box, nx_box = orig_ssh_stack.shape[1:] # The shape of the cropped sim pass

    # ---- coords (km) ----
    x_coords = np.linspace(xt_km.min(), xt_km.max(), nxt)      # along-track, balanced target grid
    y_coords = np.linspace(yt_km.min(), yt_km.max(), nyt)      # across-track, balanced target grid

    sim_x_km = np.asarray(karin_NA.x_coord) * 1e-3            # along-track on sim-on-KaRIn grid
    sim_y_km = np.asarray(karin_NA.y_coord) * 1e-3            # across-track on sim-on-KaRIn grid

    # ---- time metadata ----
    time_coords = np.arange(T, dtype=np.float64)
    cycle_numbers = [shared_cycles[idx] for idx in processed_indices]
    swot_times = np.array([karin.time_dt[idx] for idx in processed_indices], dtype="datetime64[ns]")
    sim_times  = np.array([used_dates[idx] for idx in processed_indices], dtype="datetime64[ns]")

    # # ---- parameter names/arrays for fits ----
    param6_names = np.array(["A_b", "lambda_b_m", "s_b", "A_n", "lambda_n_m", "s_n"], dtype="U")
    nadir_param_names = np.array(["N_n"], dtype="U")

    # ---- dataset ----
    ds = xr.Dataset(
        {
            # SWOT Observations
            "karin_ssha": (["time", "karin_y", "karin_x"], karin.ssha[processed_indices], {
                "long_name": "KaRIn SSHA (SWOT)",
                "units": "m"
            }),
            "nadir_ssh": (["time", "nadir_along"], nadir.ssh[processed_indices], {
                "long_name": "Nadir SSH (SWOT)",
                "units": "m"
            }),
            "karin_lon": (["time", "karin_y", "karin_x"], karin.lon[processed_indices], {
                "long_name": "KaRIn Longitude (SWOT)",
                "units": "degrees_east"
            }),
            "karin_lat": (["time", "karin_y", "karin_x"], karin.lat[processed_indices], {
                "long_name": "KaRIn Latitude (SWOT)",
                "units": "degrees_north"
            }),
            "nadir_lon": (["time", "nadir_along"], nadir.lon[processed_indices], {
                "long_name": "Nadir Longitude (SWOT)",
                "units": "degrees_east"
            }),
            "nadir_lat": (["time", "nadir_along"], nadir.lat[processed_indices], {
                "long_name": "Nadir Latitude (SWOT)",
                "units": "degrees_north"
            }),
            # Balanced results
            "balanced_ssh_swot": (["time", "y", "x"], ht_swot_stack, {
                "long_name": "Balanced SSH (SWOT obs)",
                "units": "m"
            }),
            # NA Sim results
            "sim_pass_ssh": (["time", "box_y", "box_x"], orig_ssh_stack, {
                "long_name": "Original simulation SSH along the SWOT pass",
                "units": "m"
            }),
            "sim_pass_lon": (["box_y", "box_x"], karin.lon_full, {
                "long_name": "Simulation longitude along SWOT pass",
                "units": "degrees_east"
            }),
            "sim_pass_lat": (["box_y", "box_x"], karin.lon_full, {
                "long_name": "Simulation latitudes along SWOT Pass",
                "units": "degrees_north"
            }),
            # Simulation interpolated to KaRIn/Nadir geometry
            "sim_karin_ssha": (["time", "sim_y", "sim_x"], karin_NA.ssha[processed_indices], {
                "long_name": "Simulation SSHA on KaRIn grid",
                "units": "m"
            }),
            "sim_nadir_ssha": (["time", "sim_nadir_along"], nadir_NA.ssha[processed_indices], {
                "long_name": "Simulation SSHA on Nadir track",
                "units": "m"
            }),
            "sim_karin_lon": (["sim_y", "sim_x"], karin_NA.lon[sample_index], {
                "long_name": "Simulation longitude on KaRIn grid",
                "units": "degrees_east"
            }),
            "sim_karin_lat": (["sim_y", "sim_x"], karin_NA.lon[sample_index], {
                "long_name": "Simulation latitude on KaRIn grid",
                "units": "degrees_north"
            }),
            # Synthetic SWOT observations
            "synth_karin_ssha": (["time", "sim_y", "sim_x"], ssh_noisy[processed_indices], {
                "long_name": "Synthetic KaRIn SSHA from simulation",
                "units": "m"
            }),
            "synth_nadir_ssha": (["time", "sim_nadir_along"], ssh_nadir_noisy[processed_indices], {
                "long_name": "Synthetic Nadir SSHA from simulation",
                "units": "m"
            }),
            "synth_balanced_ssh": (["time", "y", "x"], ht_syn_stack, {
                "long_name": "Balanced SSH extracted from synthetic SWOT data",
                "units": "m"
            }),
            # Fit parameters (as variables with named coords)
            "fit_params_swot_karin": (["param6"], np.asarray(poptcwg_karin, dtype="f8"), {
                "long_name": "SWOT KaRIn spectral fit parameters",
                "description": "[A_b, lambda_b (m), s_b, A_n, lambda_n (m), s_n]"
            }),
            "fit_params_sim_karin": (["param6"], np.asarray(poptcwg_karin_NA, dtype="f8"), {
                "long_name": "Synthetic KaRIn spectral fit parameters",
                "description": "[A_b, lambda_b (m), s_b, A_n, lambda_n (m), s_n]"
            }),
            "fit_params_swot_nadir": (["param_nadir"], np.atleast_1d(poptcwg_nadir).astype("f8"), {
                "long_name": "SWOT Nadir spectral fit parameters",
                "description": "[N_n]"
            }),

            # Time & meta
            "swot_cycle_number": (["time"], cycle_numbers, {"long_name": "SWOT Cycle Number", "units": "cycle"}), 
            "swot_pass_time": (["time"], swot_times, {"long_name": "SWOT pass datetime"}),
            "sim_pass_time": (["time"], sim_times, {"long_name": "Simulation datetime matched to SWOT"}),
        },
        coords={
            "time": ("time", time_coords, {
                "long_name": "Time Index", "units": "index",
                "standard_name": "time", "axis": "T",
                "_CoordinateAxisType": "Time"
            }),
            # Balanced target coordinates (km)
            "xt": ("x", x_target, {"long_name": "Balanced Target Coords", "units": "km"}),
            "yt": ("y", y_target, {"long_name": "Balanced Target Coords", "units": "km"}),

            # KaRIn/Nadir native axes
            "karin_x": ("karin_x", karin.x_coord,
                        {"long_name": "KaRIn across-track index (km)", "units": "km"}),
            "karin_y": ("karin_y", karin.y_coord,
                        {"long_name": "KaRIn along-track index (km)", "units": "km"}),
            "nadir_x":  ("nadir_x", nadir.x_coord,
                            {"long_name": "Nadir along-track index", "units": "km"}),
            "nadir_y":  ("nadir_y", nadir.y_coord,
                            {"long_name": "Nadir across-track index", "units": "km"}),

            # Sim-on-KaRIn axes (km)
            "sim_x": ("sim_x", sim_x_km, {"long_name": "Along-track (sim on KaRIn)", "units": "km"}),
            "sim_y": ("sim_y", sim_y_km, {"long_name": "Across-track (sim on KaRIn)", "units": "km"}),

            # Original box axes (index only; lat/lon provided as variables)
            "box_x": ("box_x", np.arange(nx_box), {"long_name": "Original sim box x-index"}),
            "box_y": ("box_y", np.arange(ny_box), {"long_name": "Original sim box y-index"}),

            # Parameter name coords
            "param6": ("param6", param6_names),
            "param_nadir": ("param_nadir", nadir_param_names),
        },
        attrs={
            "title": f"SWOT Balanced SSH Extraction - Pass {pass_number:03d}",
            "description": "Balanced SSH (SWOT & synthetic), original and simulated fields, plus spectral fit parameters.",
            "pass_number": pass_number,
            "latitude_range": f"{lat_min}°N to {lat_max}°N",
            "processing_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "KaRIn spatial_resolution_x [km]": karin.dx_km,
            "KaRInspatial_resolution_y [km]": karin.dy_km,
            "nadir_resolution": nadir.dy_km,
            "n_cycles_processed": T,
            "mission_phase": "CalVal" if "CALVAL" in data_folder.upper() else "Science",
            "swot_fit_karin_A_b":          _f(poptcwg_karin[0]),
            "swot_fit_karin_lambda_b_m":   _f(poptcwg_karin[1]),
            "swot_fit_karin_s_b":          _f(poptcwg_karin[2]),
            "swot_fit_karin_A_n":          _f(poptcwg_karin[3]),
            "swot_fit_karin_lambda_n_m":   _f(poptcwg_karin[4]),
            "swot_fit_karin_s_n":          _f(poptcwg_karin[5]),
            "swot_fit_nadir_N_n":          _f(np.atleast_1d(poptcwg_nadir)[0]),
            "nasim_fit_karin_A_b":         _f(poptcwg_karin_NA[0]),
            "nasim_fit_karin_lambda_b_m":  _f(poptcwg_karin_NA[1]),
            "nasim_fit_karin_s_b":         _f(poptcwg_karin_NA[2]),
            "nasim_fit_karin_A_n":         _f(poptcwg_karin_NA[3]),
            "nasim_fit_karin_lambda_n_m":  _f(poptcwg_karin_NA[4]),
            "nasim_fit_karin_s_n":         _f(poptcwg_karin_NA[5]),
    
            "fit_param_names": "A_b, lambda_b_m, s_b, A_n, lambda_n_m, s_n",
            "source_code": "Interpolate_the_gap_NA_sim.py",
        }
    )

    # compression
    comp = dict(zlib=True, complevel=4, shuffle=True)
    encoding = {v: comp for v in ds.data_vars}
    encoding["time"] = {"dtype": "float64", "zlib": True, "complevel": 4}

    out_nc = os.path.join(ROOT_DIR, f"swot_pass{pass_number:03d}.nc")
    ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding, unlimited_dims=["time"])
    print(f"[NetCDF] Wrote {out_nc}")

except Exception as e:
    print(f"[NetCDF] Error writing compact snapshot: {e}")
    import traceback; traceback.print_exc()
