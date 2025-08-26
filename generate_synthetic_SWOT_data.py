# In this script we estimate the balanced signal for the SWOT data 
# we also generate synthetic SWOT data based on the signal and noise which we can used for the NA simulation. 
# First, we fit spectra to the KaRIn and Nadir data for a specified region and use this to generate synthetic noise 
# which we then save for later use in the NA simulation.

import JWS_SWOT_toolbox as swot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy
import cartopy.crs as ccrs
import numpy as np
import cmocean
import xarray as xr 

pass_num = 9
lat_max = 38
lat_min = 28
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/' # where our data is stored

# ~~~~ SWOT Data Part ~~~~ 
# ───── Read and Process Data ─────
# finds overlapping cycles between the karin and nadir datasets
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num) 
sample_index = 2  # some index for setting up the grids 
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]

karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_num)

# Read and process the karin data
swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)

# Read and process the nadir data
swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Generate coordinates
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()

# ───── Fit Models to the Spectra ─────
# KARIN
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model) # Fit the Model to the spectrum, returns fit vectors

# NADIR
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)

# ───── Generate Synthetic Fields from the SWOT Models ─────
# --- Extract Parameters from SWOT data fit ---
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

# --- Generate Synthetic SWOT Signal+ Noise ---
h, eta, eta_k, eta_n = swot.generate_signal_and_noise(F, Fk, sigma_noise, nx*ny, nn)

# --- Target (Reconstruction) Grid ---
xt, yt, nxt, nyt = swot.make_target_grid(karin, extend=False)

# --- Estimate Signal on Target Grid ---
ht = swot.estimate_signal_on_target(c_bal, xt, yt, xobs, yobs, C, N, h + eta)

# Now we loop over the time indecies and create a new synthetic realization for each time step. 
hs_list   = []
etas_list = []
etas_k = [] # karin noise model
etas_n = [] # nadir noise model 

for i in range(0, karin.ssha.shape[0]): # make one for every simulation time
    h, eta, eta_k, eta_n = swot.generate_signal_and_noise(F, Fk, sigma_noise, nx * ny, nn)
    etas_k.append(eta_k.reshape(ny, nx))
    etas_n.append(eta_n)
    hs_list.append(h)
    etas_list.append(eta)

hs     = np.array(hs_list, dtype=object)
etas   = np.array(etas_list, dtype=object)
etas_k = np.array(etas_k, dtype=object)
etas_n = np.array(etas_n, dtype=object)

# karin component
h_k = hs[:, :nx*ny]  # karin signal  
h_combined = h_k  + etas_k.reshape(etas_k.shape[0], -1) # karin signal + noise

# h_syn is the combined generated signal and noise reshaped into (time, nx, ny)
h_syn = np.zeros((hs.shape[0], ny, nx))
for i in range(hs.shape[0]):
    h_syn[i, :, :] = h_combined[i, :].reshape(ny, nx)

# ---- Make Plot of the Generated Data ----
# Now we generate our Karin and Nadir noise based on synthetic realizations of the SWOT models 
index = 10

ntime = karin_NA.ssha.shape[0] # simulation times
vmin = -0.25
vmax = 0.25 

ssh_noisy = np.empty_like(karin_NA.ssha) # new arrays for synthetic SWOT NA data
ssh_nadir_noisy = np.empty_like(nadir_NA.ssha)

for t in range(0, ntime):

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


kt_NA_coords    = [np.arange(ntime), karin.y_coord, karin.x_coord]
ssh_noisy_xr = xr.DataArray(ssh_noisy, coords = kt_NA_coords, dims = ['sample', 'line', 'pixel'])
spec_ssh_noisy = swot.mean_power_spectrum(ssh_noisy_xr, karin.window, 'line', ['sample', 'pixel'])
spec_ssh_noisy = spec_ssh_noisy[int(karin.track_length/2):] # take the half spectrum

nad_noisy_xr   = xr.DataArray(ssh_nadir_noisy, coords = eta_nt_coords, dims = ['sample', 'nadir_line'])
spec_nad_noisy = swot.mean_power_spectrum(nad_noisy_xr, nadir.window, 'nadir_line', ['sample'])
spec_nad_noisy = spec_nad_noisy[int(nadir.track_length/2):]

karin_NA.ssh_noisy = ssh_noisy # save the generated noisy fields to our NA simulation classes 
nadir_NA.ssh_noisy = ssh_nadir_noisy

fig = plt.figure(figsize=(18, 6), dpi=150)
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.6], figure=fig)

# ───── Simulation Map ─────
ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
sc0 = ax0.scatter(
    NA_karin_lon.flatten(), NA_karin_lat.flatten(),
    c=ssh_noisy[index], s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o'
)
ax0.scatter(
    NA_nadir_lon, NA_nadir_lat, c=ssh_nadir_noisy[index], vmin=vmin, vmax=vmax,
    cmap=cmap, s=1, marker='o', transform=ccrs.PlateCarree()
)
ax0.coastlines()
ax0.set_title("NA Sim. + SWOT Noise Model")
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
gl0.top_labels = gl0.right_labels = False
cbar0 = fig.colorbar(sc0, ax=ax0, orientation='vertical', shrink=0.5, pad=0.05)
cbar0.set_label("SSHA (m)")

# ───── SWOT Map ─────
ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
sc1 = ax1.scatter(
    karin.lon[index], karin.lat[index],
    c=karin.ssha[index], s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o'
)
ax1.scatter(
    nadir.lon[index], nadir.lat[index],
    c=nadir.ssh[index], s=1, vmin=vmin, vmax=vmax,
    cmap=cmap, transform=ccrs.PlateCarree(), marker='o'
)
ax1.coastlines()
ax1.set_title(f'SWOT KaRIn + Nadir')
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
gl1.top_labels = gl1.right_labels = False
cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', shrink=0.5, pad=0.05)
cbar1.set_label("SSHA (m)")

# ───── Power Spectrum ─────
ax2 = fig.add_subplot(gs[0, 2])
ax2.loglog(karin_NA.wavenumbers_cpkm, karin_NA.spec_alongtrack_av, label='NA Sim Spectrum', linewidth=2.5)
# ax2.loglog(nadir_NA.wavenumbers * 1e3, nadir_NA.spec_alongtrack_av, label='Sim Nadir SSH', linewidth=2)
ax2.loglog(spec_ssh_noisy.freq_line * 1e3, spec_ssh_noisy, label='NA Sim + KaRIn Noise', linewidth=2.5)
ax2.loglog(k_nadir*1e3, spec_nad_noisy, label='NA Sim + Nadir Noise', linewidth=2.5)

ax2.loglog(karin.wavenumbers_cpkm, karin.spec_alongtrack_av, '-', label='SWOT KaRIn SSHA', linewidth=2.5)
ax2.loglog(nadir.wavenumbers_cpkm, nadir.spec_alongtrack_av, '-', label='SWOT Nadir SSHA', linewidth=2.5)
ax2.loglog(k_karin * 1e3, spec_eta_k,'--', lw=2.5, label='Synthetic KaRIn Noise')
ax2.loglog(k_nadir * 1e3, spec_eta_n, '--', lw=2.5, label='Synthetic Nadir Noise')

ax2.set_xlabel("Wavenumber (cpkm)")
ax2.set_ylabel("PSD (m$^2$/cpkm)")
ax2.set_ylim(ylims)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
#ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.set_title("SSHA Spectra")

plt.suptitle(f"Pass {pass_num:03d}", x=0.0, ha='center', fontsize=13)

plt.tight_layout()
plt.show()

