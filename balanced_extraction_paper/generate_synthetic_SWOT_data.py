# Estimate the balanced signal for the SWOT data and use to
# generate synthetic SWOT data based on the signal and noise which we can used for the NA simulation. 

import JWS_SWOT_toolbox as swot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy
import cartopy.crs as ccrs
import numpy as np
import cmocean
import xarray as xr 
import scipy.linalg as la
import pickle
from JWS_SWOT_toolbox.julia_bridge import julia_functions as jl
import cartopy.feature as cfeature
 
# ───── Read and Process Data ─────
# Read in the SWOT data for this pass
pass_num = 9
lat_max = 35
lat_min = 28

data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'
#data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE/'

_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num)

sample_index = 2 
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]
karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_num)

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

karin.shared_cycles = shared_cycles # save

#  Match the simulation dates with the SWOT dates
NA_folder = "/expanse/lustre/projects/cit197/jskinner1/NA_daily_snapshots"

# 1) choose sim dates for KaRIn times
_, _, matched_dates = swot.pick_range_from_karin_times(
    karin_time_dt=karin.time_dt,
    data_folder=NA_folder,
    mode="cyclic"    # or 'absolute' if sim year == SWOT year
)
print("Dates matched")

#2) interpolate each sim day onto the ONE constant KaRIn grid
NA_karin_full_ssh, NA_karin_ssh, NA_nadir_ssh, used_dates = swot.load_sim_on_karin_nadir_grids(
    karin, 
    nadir, 
    data_folder=NA_folder, 
    matched_dates=matched_dates 
)

# Now the data is processed we can init new data classes with NA_Karin/Nadir
ncycles = NA_karin_ssh.shape[0]
track_length_karin =  NA_karin_ssh.shape[1]
track_length_nadir = NA_nadir_ssh.shape[1]
dims_NA = [ncycles, track_length_karin, track_length_nadir]

karin_NA, nadir_NA = swot.init_swot_arrays(dims_NA, lat_min, lat_max, pass_num) # init a class for the karin/nadir parts of the data

# SWOT geometry fields
karin_NA.ssh = NA_karin_ssh 
karin_NA.lat = karin.lat
karin_NA.lon = karin.lon
karin_NA.date_list=matched_dates  

nadir_NA.ssh = NA_nadir_ssh 
nadir_NA.lat = nadir.lat
nadir_NA.lon = nadir.lon

# compute the SSHA's 
NA_karin_smean = np.nanmean(NA_karin_ssh, axis=(1, 2)) 
karin_NA.ssha  = NA_karin_ssh - NA_karin_smean[:, None, None] 
nadir_NA.ssha  = NA_nadir_ssh - NA_karin_smean[:, None]  # subtract the karin mean from nadir (better constrained)
karin_NA.ssha_full = NA_karin_full_ssh - NA_karin_smean[:, None, None] 

# Builds the coordinate grids -- in [m]
karin_NA.coordinates()
nadir_NA.coordinates()

# Compute spectra
karin_NA.compute_spectra()
nadir_NA.compute_spectra()
print("NA sim done")

#───── Fit SWOT Data Spectra ─────
# KaRIn model fit
p_karin, _ = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)

# Nadir model fit
p_nadir, _ = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, p_karin)

swot.plot_spectral_fits(karin, nadir, p_karin, p_nadir, 'fits.pdf')

# --- Grid and Spacing ---
nx, ny = 2 * karin.swath_width, karin.track_length
nn = nadir.track_length
dx, dy, dn = karin.dx_km, karin.dy_km, nadir.dy_km

# --- Covariance Functions ---
c_bal = swot.balanced_covariance_func(p_karin)
c_unb = swot.noise_covariance_func(p_karin)

N_n = p_nadir[0]
sigma_noise = np.sqrt(N_n / (2 * dn))

# --- Observation Points ---
xk, yk = karin.x_obs_grid.flatten()*1e-3, karin.y_obs_grid.flatten()*1e-3
xn, yn = nadir.x_grid.flatten()*1e-3, nadir.y_grid.flatten()*1e-3
xobs = np.concatenate((xk, xn))
yobs = np.concatenate((yk, yn))

# --- Covariance and Noise Matrices ---
C = swot.build_covariance_matrix(c_bal, xobs, yobs)
N, Nk = swot.build_noise_matrix(c_unb, xk, yk, sigma_noise, nn, nx*ny)

# --- Cholesky Decomposition ---
F = swot.cholesky_decomp(C, "C")
Fk = swot.cholesky_decomp(Nk, "Nk")
cho_tuple = la.cho_factor(C + N, lower=True)

# --- Generate Synthetic SWOT Signal+ Noise ---
h, eta, eta_k, eta_n = swot.generate_signal_and_noise(F, Fk, sigma_noise, nx*ny, nn)

# --- Target (Reconstruction) Grid ---
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin, unit='km', extend=False)

# --- Cross covariance between target and observation points
R = c_bal(np.hypot(xt[:, None] - xobs, yt[:, None] - yobs))

# --- Estimate Signal on Target Grid ---
ht = swot.estimate_signal_on_target_cho_solve(R, cho_tuple, h + eta) # faster function

# generates n synthetic realisations of the signal and noise
n_realizations = karin_NA.ssha.shape[0]
hs, etas, etas_k, etas_n = swot.generate_synthetic_realizations(swot, F, Fk, sigma_noise, nx, ny, nn, n_realizations)

ntime = karin_NA.ssha.shape[0] # simulation times
vmin = -0.3
vmax = 0.3 

ssh_noisy = np.empty_like(karin_NA.ssha) # new arrays for synthetic SWOT NA data
ssh_nadir_noisy = np.empty_like(nadir_NA.ssha)

for t in range(0, ntime):

    ssh = karin_NA.ssha[t, :, :] 
    mask = np.isfinite(ssh)      # mask the gap out

    eta_k_reshaped = np.full_like(ssh, np.nan)  
    eta_k_current = etas_k[t, :, :] - np.nanmean(etas_k[t, :, :])

    # Direct assignment of flattened valid data:
    eta_k_reshaped[mask] = eta_k_current.flatten()

    # Add noise to SSH, preserving gaps:
    ssh_noisy[t, :, :] = ssh + 1e-2 * eta_k_reshaped

    # --- Nadir ---
    ssh_nadir = nadir_NA.ssha[t, :]
    ssh_nadir_noisy[t] = ssh_nadir + 1e-2 * etas_n[t]

karin_NA.ssh_noisy = ssh_noisy
nadir_NA.ssh_noisy = ssh_nadir_noisy

karin_NA.poptcwg_karin = p_karin
karin.poptcwg_karin = p_karin

# save the synthetic SWOT data to pickle file
with open("./pickles/karin_NA_tmean.pkl", "wb") as f:
   pickle.dump(karin_NA, f)

with open("./pickles/nadir_NA_tmean.pkl", "wb") as f:
   pickle.dump(nadir_NA, f)

with open("./pickles/nadir.pkl", "wb") as f:
   pickle.dump(nadir, f)

with open("./pickles/karin.pkl", "wb") as f:
   pickle.dump(karin, f)

print("Saved")

# --------------------
# Paper Fig. 6 a): Three field maps
# --------------------

index = 40

cmap = cmocean.cm.balance

vmin = -0.5 
vmax = 0.5
phi = np.deg2rad(31.5)

fig1 = plt.figure(figsize=(8, 5), dpi=200)

ax0 = fig1.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
sc0 = ax0.scatter(
    karin.lon_full[:, 5:65], karin.lat_full[:, 5:65],
    c=NA_karin_full_ssh[index][:, 5:65] - np.nanmean(NA_karin_full_ssh[index][:, 5:65]),
    s=1, marker='o', vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), rasterized=True
)
ax0.set_title("Simulation")
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl0.top_labels = gl0.right_labels = False
gl0.xlabel_style = {'size': 9}
gl0.ylabel_style = {'size': 9}
ax0.set_aspect(1 / np.cos(phi))
ax0.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
ax0.set_rasterization_zorder(1) 

ax1 = fig1.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
sc1 = ax1.scatter(
    karin.lon[index].flatten(), karin.lat[index].flatten(),
    c=ssh_noisy[index],
    s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax1.scatter(
    nadir.lon[index], nadir.lat[index],
    c=ssh_nadir_noisy[index],
    s=0.5, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)

ax1.set_title("Synthetic Data")
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl1.top_labels = gl1.right_labels = False
gl1.xlabel_style = {'size': 9}
gl1.ylabel_style = {'size': 9}
ax1.set_aspect(1 / np.cos(phi))
ax1.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
ax1.set_rasterization_zorder(1) 

ax2 = fig1.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
sc2 = ax2.scatter(
    karin.lon[index].flatten(), karin.lat[index].flatten(),
    c=karin.ssha[index].flatten(),
    s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax2.scatter(
    nadir.lon[index], nadir.lat[index],
    c=nadir.ssha[index] if hasattr(nadir, "ssha") else (nadir.ssh[index] - np.nanmean(nadir.ssh[index])),
    s=0.5, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax2.set_title("SWOT Data")
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl2.top_labels = gl2.right_labels = False
gl2.xlabel_style = {'size': 9}
gl2.ylabel_style = {'size': 9}
ax2.set_aspect(1 / np.cos(phi))
ax2.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
ax2.set_rasterization_zorder(1) 

cbar_ax = fig1.add_axes([0.92, 0.22, 0.015, 0.35])
cbar = fig1.colorbar(sc2, cax=cbar_ax, orientation='vertical', shrink = 0.5)
cbar.set_label("SSHA [m]")

norm = plt.Normalize(vmin=vmin, vmax=vmax)
sc0.set_norm(norm)
sc1.set_norm(norm)
sc2.set_norm(norm)

plt.suptitle(f"Pass {pass_num:03d}  Cycle {shared_cycles[index]:03d}", fontsize=11)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
fig1.savefig('synthetic_swot_fields.pdf', bbox_inches='tight')

print("plotted")

# --------------------
# Paper Fig. 6 b) Power spectrum
# --------------------

# Build xarray wrappers and compute spectra (convert to cm for PSD units)
kt_NA_coords = [np.arange(ntime), karin.y_coord_km, karin.x_coord_km]
ssh_noisy_xr = xr.DataArray(ssh_noisy * 100.0, coords=kt_NA_coords, dims=['sample', 'line', 'pixel'])
spec_ssh_noisy = swot.mean_power_spectrum(ssh_noisy_xr, karin.window, 'line', ['sample', 'pixel'])
spec_ssh_noisy = spec_ssh_noisy[int(karin.track_length/2):]  # one-sided (positive k)

if 'eta_nt_coords' not in locals():
    eta_nt_coords = {'sample': np.arange(ssh_nadir_noisy.shape[0]),
                     'nadir_line': np.arange(0.5, ssh_nadir_noisy.shape[1], 1.0) * nadir.dy_km}

nad_noisy_xr   = xr.DataArray(ssh_nadir_noisy * 100.0, coords=eta_nt_coords, dims=['sample', 'nadir_line'])
spec_nad_noisy = swot.mean_power_spectrum(nad_noisy_xr, nadir.window, 'nadir_line', ['sample'])

print(karin.time_dt[index])
print(shared_cycles[index])

fig2 = plt.figure(figsize=(5, 5), dpi=150)
ax2 = fig2.add_subplot(1, 1, 1)

k_karin = karin.wavenumbers_cpkm
k_nadir = nadir.wavenumbers_cpkm

ax2.loglog(karin_NA.wavenumbers_cpkm, karin_NA.spec_alongtrack_av, color='tab:purple', label='Simulation', linewidth=2.0)
ax2.loglog(spec_ssh_noisy.freq_line,    spec_ssh_noisy,           '-',  color='tab:orange', label='Synthetic KaRIn', linewidth=2.0)
ax2.loglog(k_karin,                     karin.spec_alongtrack_av, '-',  color='tab:blue',   label='SWOT KaRIn', linewidth=2.0)
ax2.loglog(k_nadir,                     spec_nad_noisy,           '-',  color='tab:green',  label='Synthetic Nadir', linewidth=2.0)
ax2.loglog(k_nadir,                     nadir.spec_alongtrack_av, '-',  color='tab:red',    label='SWOT Nadir', linewidth=2.0)

ax2.set_xlabel("Wavenumber [cpkm]", fontsize=13)
ax2.set_ylabel("Power spectral density [cm$^2$cpkm$^{-1}$]", fontsize=13)
ax2.legend(loc='lower left', fontsize=12, frameon=False)

# yticks = ax2.get_yticks()
# if len(yticks) > 0:
#     ax2.set_yticks(yticks[::2])

ax2.set_ylim(1e-5, 1e7)
ax2.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('synthetic_swot_spectra.pdf')