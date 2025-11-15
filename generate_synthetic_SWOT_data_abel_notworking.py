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
 
# ───── Read and Process Data ─────
# Read in the SWOT data for this pass
pass_num = 9
lat_max = 32
lat_min = 30

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

#  Match the simulation dates with the SWOT dates, we do both NA sim and butterworth filter
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

# Now the data is processed we can init all out data classes with NA_Karin/Nadir
ncycles = NA_karin_ssh.shape[0]
track_length_karin =  NA_karin_ssh.shape[1]
track_length_nadir = NA_nadir_ssh.shape[1]
dims_NA = [ncycles, track_length_karin, track_length_nadir]

karin_NA, nadir_NA = swot.init_swot_arrays(dims_NA, lat_min, lat_max, pass_num) # init a class for the karin/nadir parts of the data
karin_NA.ssh_full = NA_karin_full_ssh # save the original ssh

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

#───── Covariances ─────
# -------------------------
# Geometry & masks for NA sim data
# -------------------------
mask_k = np.isfinite(karin_NA.ssha[sample_index])
mask_n = np.isfinite(nadir_NA.ssh[sample_index]).ravel()
    
xkk = (karin_NA.x_grid[mask_k].ravel(order="C")) * 1e-3  # km
ykk = (karin_NA.y_grid[mask_k].ravel(order="C")) * 1e-3
xnn = (nadir_NA.x_grid.ravel()[mask_n]) * 1e-3
ynn = (nadir_NA.y_grid.ravel()[mask_n]) * 1e-3

xobs = np.concatenate((xkk, xnn))
yobs = np.concatenate((ykk, ynn))

# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin_NA, unit="km", extend=False)
n_t = xt.size
nx, ny = 2 * karin_NA.swath_width, karin_NA.track_length
nn = nadir_NA.track_length

# -------------------------
# Distance matrices in km
# -------------------------
def pairwise_r(x0, y0, x1=None, y1=None):
    # Euclidean distances between (x0,y0) and (x1,y1)
    if x1 is None:  # square
        dx = x0[:, None] - x0[None, :]
        dy = y0[:, None] - y0[None, :]
    else:
        dx = x0[:, None] - x1[None, :]
        dy = y0[:, None] - y1[None, :]
    return np.hypot(dx, dy)

r_kk = pairwise_r(xkk, ykk)
r_nn = pairwise_r(xnn, ynn) 
r_kn = pairwise_r(xkk, ykk, xnn, ynn)
r_tk = pairwise_r(xt, yt, xkk, ykk)
r_tn = pairwise_r(xt, yt, xnn, ynn) 
r_tt = pairwise_r(xt, yt)
print("Distance matrices built")

# 4) ───── Generate Covariances ─────
# Build Covariances 
# -------------------------

# Specral Fit Parameters
B_psd = swot.balanced_psd_from_params(p_karin)                 # B(k) balanced power spectrum model
Nk_psd = swot.karin_noise_psd_from_params(p_karin)             # N_K(k) noise power spectrum model
sigma_n = np.sqrt(p_nadir[0] / (2.0 * nadir.dy_km))  # cm

# Base kernels in [cm^2]
n_samples = 10000
l_sample = 5000
kk = np.arange(n_samples // 2 + 1) / l_sample                             # wavenumber grid for transforms (200000 samples over 10000km as in swot.cov())

Tfun  = lambda k: swot.taper(k, cutoff=2.0)                               # T(k) is taper function cutoff at 2km  
C_B      = jl.cov(B_psd(kk), kk)                                          # C[B]
C_BT     = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk)*Tfun(kk), kk), kk)      # C[B T]

# Tapered Kernels (requires Abel transform) in [cm]
SIGMA_L_KM = 0.0
SIGMA    = 2 * np.pi * SIGMA_L_KM                               # σ convert to angular wavenumber
DELTA    = (np.pi * karin_NA.dx_km) / (2 * np.log(2))           # δ

# Gaussian Smoothings and tapers combined
G  = lambda k: np.exp(-((SIGMA**2) * (k**2)) / 2.0)            # Gassian smooth C[G]
G2 = lambda k: np.exp(-(SIGMA**2) * (k**2))                    # Target-Target smoothing C[G^2]
GT = lambda k: np.exp(-(((SIGMA**2 + DELTA**2)* k**2) / 2.0) ) # Taper + Gaussian Smooth C[GT]
T2 = lambda k: np.exp(-(DELTA**2) * (k**2))                    # Taper^2 C[T^2]

C_B_G   = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G(kk), kk), kk)   # C[B G]
C_B_G2  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G2(kk), kk), kk)  # C[B G^2]
C_B_TG  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * GT(kk), kk), kk)  # C[B TG]
C_B_T2  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * T2(kk), kk), kk)  # C[B T^2]
C_NT2   = jl.cov(jl.abel(jl.iabel(Nk_psd(kk), kk) * T2(kk), kk), kk)  # C[N T^2]


# -------------------------
# Blocks
# -------------------------
# KaRIn–KaRIn: C[(B+N_K) T^2]
R_KK = np.asarray(C_B_T2(r_kk))
N_KK = np.asarray(C_NT2(r_kk)) # Noise Karin-Karin term

# Target-Target covariance for posterior C[BG^2] 
R_tt = np.asarray(C_B_G2(r_tt))

# Nadir–Nadir: C[B] + σ_N^2 I
R_NN = np.asarray(C_B(r_nn))
N_NN = (sigma_n**2) * np.eye(r_nn.shape[0])
     
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

swot.diagnose_not_positive_definite(C_obs)

N_obs = la.block_diag(N_KK, N_NN)

swot.diagnose_not_positive_definite(N_obs)

R = np.concatenate([R_tK, R_tN], axis=1) # target covariance
print("Covariance Blocks Built")

# --- Cholesky Decomposition ---
F = swot.cholesky_decomp(C_obs, "C", jitter=True)
Fk = swot.cholesky_decomp(N_KK, "Nk") # Choleky of KaRIn noise 
cho = la.cho_factor(C_obs + N_obs, lower=True)

# --- Create realizations of the noise 
n_realizations = karin_NA.ssha.shape[0]
hs, etas, etas_k, etas_n = swot.generate_synthetic_realizations(swot, F, Fk, sigma_n, nx, ny, nn, n_realizations)

# --- Add noise to the NA simulation data ---
ssh_noisy = np.empty_like(karin_NA.ssha) # new arrays for synthetic SWOT NA data
ssh_nadir_noisy = np.empty_like(nadir_NA.ssha)

ntime = karin_NA.ssha.shape[0] # simulation times
for t in range(0, ntime):

    ssh = karin_NA.ssha[t, :, :] 
    mask = np.isfinite(ssh)      # mask the gap out

    eta_k_reshaped = np.full_like(ssh, np.nan)  
    eta_k_current = etas_k[t, :, :] 

    # Direct assignment of flattened valid data:
    eta_k_reshaped[mask] = eta_k_current.flatten()

    # Add noise to SSH, preserving gaps:
    ssh_noisy[t, :, :] = ssh + 1e-2 * eta_k_reshaped

    # --- Nadir ---
    ssh_nadir = nadir_NA.ssha[t, :]
    ssh_nadir_noisy[t] = ssh_nadir + 1e-2 * etas_n[t]


kt_NA_coords    = [np.arange(ntime), karin.y_coord_km, karin.x_coord_km]
ssh_noisy_xr = xr.DataArray(ssh_noisy * 100, coords = kt_NA_coords, dims = ['sample', 'line', 'pixel'])
spec_ssh_noisy = swot.mean_power_spectrum(ssh_noisy_xr, karin.window, 'line', ['sample', 'pixel'])
spec_ssh_noisy = spec_ssh_noisy[int(karin.track_length/2):] # take the half spectrum

eta_nt_coords = {'sample': np.arange(etas_n.shape[0]),'nadir_line': np.arange(0.5, etas_n.shape[1], 1.0) * nadir.dy_km}
nad_noisy_xr   = xr.DataArray(ssh_nadir_noisy * 100, coords = eta_nt_coords, dims = ['sample', 'nadir_line'])
spec_nad_noisy = swot.mean_power_spectrum(nad_noisy_xr, nadir.window, 'nadir_line', ['sample'])
spec_nad_noisy = spec_nad_noisy[int(nadir.track_length/2):]

karin_NA.ssh_noisy = ssh_noisy # save the generated noisy fields to our NA simulation classes 
nadir_NA.ssh_noisy = ssh_nadir_noisy

# Save noisy fields back to simulation classes 
karin_NA.ssh_noisy = ssh_noisy
nadir_NA.ssh_noisy = ssh_nadir_noisy

# # ### Save the pickle files so we can read in later
# karin_NA.poptcwg_karin = poptcwg_karin_NA
# karin.poptcwg_karin = poptcwg_karin

with open("./pickles/karin_NA_tmean_small.pkl", "wb") as f:
    pickle.dump(karin_NA, f)

with open("./pickles/nadir_NA_tmean_small.pkl", "wb") as f:
    pickle.dump(nadir_NA, f)

with open("./pickles/nadir_small.pkl", "wb") as f:
    pickle.dump(nadir, f)

with open("./pickles/karin_small.pkl", "wb") as f:
    pickle.dump(karin, f)

print("Saved")

# --------------------
# Figure 1: Fields (three maps)
# --------------------

index = 10

cmap = cmocean.cm.balance
vals_for_range = np.hstack([
    (NA_karin_full_ssh[index][:, 4:64] - np.nanmean(NA_karin_full_ssh[index][:, 4:64])).ravel(),
    (ssh_noisy[index] - np.nanmean(ssh_noisy[index])).ravel(),
    karin.ssha[index].ravel(),
    nadir_NA.ssh_noisy[index].ravel() if 'nadir_NA' in globals() and hasattr(nadir_NA, 'ssh_noisy')
        else (nadir.ssh[index] - np.nanmean(nadir.ssh[index])).ravel()
])
v1, v2 = np.nanpercentile(vals_for_range, [1, 99])
v = np.nanmax(np.abs([v1, v2]))
vmin, vmax = -v, v

fig1 = plt.figure(figsize=(8.5, 5), dpi=200)

ax0 = fig1.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
sc0 = ax0.scatter(
    karin.lon_full[:, 5:65], karin.lat_full[:, 5:65],
    c=NA_karin_full_ssh[index][:, 4:64] - np.nanmean(NA_karin_full_ssh[index][:, 4:64]),
    s=1, marker='o', vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), rasterized=True
)
ax0.coastlines()
ax0.set_title("NA Simulation")
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl0.top_labels = gl0.right_labels = False
gl0.xlabel_style = {'size': 9}
gl0.ylabel_style = {'size': 9}

ax1 = fig1.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
sc1 = ax1.scatter(
    karin.lon[index].flatten(), karin.lat[index].flatten(),
    c=(ssh_noisy[index] - np.nanmean(ssh_noisy[index])).flatten(),
    s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax1.scatter(
    nadir.lon[index], nadir.lat[index],
    c=(ssh_nadir_noisy[index] - np.nanmean(ssh_nadir_noisy[index])),
    s=0.5, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax1.coastlines()
ax1.set_title("Synthetic Data")
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl1.top_labels = gl1.right_labels = False
gl1.xlabel_style = {'size': 9}
gl1.ylabel_style = {'size': 9}

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
ax2.coastlines()
ax2.set_title("SWOT Data")
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl2.top_labels = gl2.right_labels = False
gl2.xlabel_style = {'size': 9}
gl2.ylabel_style = {'size': 9}

cbar_ax = fig1.add_axes([0.92, 0.22, 0.015, 0.56])
cbar = fig1.colorbar(sc2, cax=cbar_ax, orientation='vertical')
cbar.set_label("SSHA (m)")

plt.suptitle(f"Pass {pass_num:03d}  Cycle {shared_cycles[index]:03d}", fontsize=11)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
fig1.savefig('synthetic_swot_fields.pdf', bbox_inches='tight')
print("plotted")

# --------------------
# Spectrum Figure
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


fig2 = plt.figure(figsize=(5.2, 5), dpi=150)
ax2 = fig2.add_subplot(1, 1, 1)

k_karin = karin.wavenumbers_cpkm
k_nadir = nadir.wavenumbers_cpkm

ax2.loglog(karin_NA.wavenumbers_cpkm, karin_NA.spec_alongtrack_av, 'k', label='NA Sim.', linewidth=2.0)
ax2.loglog(spec_ssh_noisy.freq_line,    spec_ssh_noisy,           '-',  color='tab:orange', label='Synthetic KaRIn', linewidth=2.0)
ax2.loglog(k_karin,                     karin.spec_alongtrack_av, '-',  color='tab:blue',   label='SWOT KaRIn', linewidth=2.0)
ax2.loglog(k_nadir,                     spec_nad_noisy,           '-',  color='tab:green',  label='Synthetic Nadir', linewidth=2.0)
ax2.loglog(k_nadir,                     nadir.spec_alongtrack_av, '-',  color='tab:red',    label='SWOT Nadir', linewidth=2.0)

ax2.set_xlabel("Wavenumber (cpkm)", fontsize=11)
ax2.set_ylabel("PSD (cm$^2$/cpkm)", fontsize=11)
#ax2.set_ylim(1e-4, 1e6)
ax2.legend(loc='lower left', fontsize=10)

yticks = ax2.get_yticks()
if len(yticks) > 0:
    ax2.set_yticks(yticks[::2])

ax2.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('synthetic_swot_spectra.pdf', bbox_inches='tight')