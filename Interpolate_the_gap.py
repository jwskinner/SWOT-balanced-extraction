# Python script for generating balanced and interpolated data can by run as a job 
# (I'd prefer using the jupyter notebook though for diagnostics)
import netCDF4 as nc
import xarray as xr
import xrft
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist 
import mpmath as mp
import cartopy.crs as ccrs
from math import sin, cos, sqrt, atan2, radians
import os
import cmocean 
from glob import glob
import numpy as np
import time
import mpmath as mp
from scipy.special import gamma
import matplotlib.pyplot as plt
import JWS_SWOT_toolbox as swot
import pickle
import numpy as np
from joblib import Parallel, delayed
os.environ["NUMEXPR_MAX_THREADS"] = "4" # With 128GB of RAM 4 works

# Config. parameters
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'

min_cycle = 476
max_cycle = 580

pass_number = 4
lat_min = 26 #29
lat_max = 32 #35

# return the files in the data directory for the specified pass number
# karin_files, nadir_files are sorted by cycle and contain the same cycles
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_number)

# Returns indexes for the Karin and Nadir data between lat_min and lat_max 
indx, track_length = swot.get_karin_track_indices(karin_files[10][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[10][0], lat_min, lat_max)
dims = [len(shared_cycles), track_length, track_length_nadir]

# Init the data classes -- (time, along_track, across_track)
karin, nadir = swot.init_swot_arrays(dims, lat_min, lat_max)

# Load and process Karin data
swot.load_karin_data(karin_files, indx, karin)
swot.process_karin_data(karin)

# Load and process Nadir data
swot.load_nadir_data(nadir_files, indxs, nadir)
swot.process_nadir_data(nadir)

# compute the dx, dy, dy_nadir of the grids -- in [m]
karin.distances()
nadir.distances()

# Builds the coordinate grids -- in [m]
karin.coordinates()
nadir.coordinates()

# --- windows and grids ---
karin.window = xr.DataArray(swot.sin2_window_func(karin.track_length), dims=['line'])
nadir.window = xr.DataArray(swot.sin2_window_func(nadir.track_length), dims=['nadir_line'])
k_coords     = [karin.y_coord, karin.x_coord]
kt_coords    = [karin.t_coord, karin.y_coord, karin.x_coord]
n_coords     = [nadir.y_coord]
nt_coords    = [nadir.t_coord, nadir.y_coord]

# --- xarrays ---
# KaRIn
karin_mean           = xr.DataArray(karin.ssh_mean, coords = k_coords, dims = ['line', 'pixel'])
karin_mean_filtered  = xr.DataArray(karin.ssha_mean_highpass, coords = k_coords, dims = ['line', 'pixel'])
karin_ssh            = xr.DataArray(karin.ssh, coords = kt_coords, dims = ['sample', 'line', 'pixel']) # full ssh
karin_ssha           = xr.DataArray(karin.ssha, coords = kt_coords, dims = ['sample', 'line', 'pixel']) # ssh with time mean removed 
# Nadir
nadir_ssh            = xr.DataArray(nadir.ssh, coords = nt_coords, dims=['sample', 'nadir_line'])

# --- remove spatial mean ---
karin_spatial_mean = swot.spatial_mean(karin_ssha, ['line', 'pixel'])
karin_anomsp       = karin_ssha - karin_spatial_mean
nadir_spatial_mean = swot.spatial_mean(nadir_ssh, ['nadir_line'])
nadir_anomsp       = nadir_ssh - nadir_spatial_mean

# --- spectral analysis ---
# Karin spectra 
karin.spec_ssh          = swot.mean_power_spectrum(karin_ssh, karin.window, 'line', ['sample', 'pixel'])
karin.spec_tmean        = swot.mean_power_spectrum(karin_mean, karin.window, 'line', ['pixel'])
karin.spec_filt_tmean   = swot.mean_power_spectrum(karin_mean_filtered, karin.window, 'line', ['pixel'])
karin.spec_ssha         = swot.mean_power_spectrum(karin_ssha, karin.window, 'line', ['sample', 'pixel'])
karin.spec_alongtrack_av  = swot.mean_power_spectrum(karin_anomsp, karin.window, 'line', ['sample', 'pixel']) # Full mean spectrum
karin.spec_alongtrack_ins = swot.mean_power_spectrum(karin_anomsp, karin.window, 'line', ['pixel']) # spec at each sample
karin.wavenumbers         = karin.spec_alongtrack_ins.freq_line 

# Nadir spectra
nadir.spec_ssh            = swot.mean_power_spectrum(nadir_ssh, nadir.window, 'nadir_line', ['sample'])
nadir.spec_alongtrack_av  = swot.mean_power_spectrum(nadir_anomsp, nadir.window, 'nadir_line', ['sample'])
nadir.spec_alongtrack_ins = swot.mean_power_spectrum(nadir_anomsp, nadir.window, 'nadir_line', []) # spec at each sample
nadir.wavenumbers         = nadir.spec_alongtrack_ins.freq_nadir_line

# fit the model to the data
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)

# Assume k_nadir, spec_nadir_sample_mean, track_length, poptcwg_karin already obtained from the KaRIn fit
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)
print("Fitted noise floor N =", poptcwg_nadir[0])

# Parameters from spectral estimation above
A_b, lam_b, s_param = poptcwg_karin[0], poptcwg_karin[1], poptcwg_karin[2] # balanced params from fit
A_n, s_n, lam_n = poptcwg_karin[3], poptcwg_karin[5], 1e5 # unbalanced params from fit, we fixed lam_k to 100km in unbalanced model above
N_n = poptcwg_nadir[0] # Nadir noise

ny = 2*karin.swath_width
gap = karin.middle_width
delta_kx = karin.dx
delta_ky = karin.dy
delta_n = nadir.dy
nx = karin.track_length
nn = nadir.track_length 

# balanced model covariance
S = lambda k: A_b / (1 + (lam_b * k)**s_param)
c = swot.cov(S, 5000000, 10000e3)

# unbalanced model covariance
cutoff = 1e3
sigma = 2 * np.pi * cutoff/np.sqrt(2*np.log(2)) 
Sk = lambda k: A_n / (1 + (lam_n * k)**2)**(s_n / 2) * np.exp(-0.5 * ((sigma**2)*(k**2))) # add guassian taper to smallest scales
nk = swot.cov(Sk, 5000, 10000e3)
sigma = np.sqrt(N_n / (2 * delta_n))

# Build the KaRIn and Nadir grids
xk, yk = swot.make_karin_points(karin)
xn, yn = swot.make_nadir_points(karin, nadir)
xobs = np.concatenate((xk, xn))
yobs = np.concatenate((yk, yn))

# Build the observation masks
index = 1 # time index 
mask_k = ~np.isnan(karin.ssha[index])
mask_n = ~np.isnan(nadir.ssh[index])
mask_k_flat = mask_k.T.flatten(order="C")             # transpose to match meshgrid order
mask_full   = np.concatenate((mask_k_flat, mask_n))   # len = nx*ny + nn

# Build Covariance Matrices 
C_obs = swot.build_covariance_matrix(c,  xobs, yobs) # covariance for observed points

Nk_obs  = nk(np.hypot(
                    xobs[:mask_k_flat.sum(), None] - xobs[:mask_k_flat.sum()],
                    yobs[:mask_k_flat.sum(), None] - yobs[:mask_k_flat.sum()])) # Karin noise block on observed pixels

Nn_obs  = sigma**2 * np.eye(mask_n.sum()) # Nadir noise block (white noise on diagonal)

N_obs = block_diag(Nk_obs, Nn_obs) # assemble block-diagonal noise matrix

# Covariance matrices
C = swot.build_covariance_matrix(c, xobs, yobs)
N, Nk = swot.build_noise_matrix(nk, xk, yk, sigma, nn, nx * ny)

# Cholesky decompositions
F = swot.cholesky_decomp(C, "C")
Fk = swot.cholesky_decomp(Nk, "Nk")

# Target grid
xt, yt, nxt, nyt = swot.make_target_grid(karin)

# ---- Save the karin and nadir classes ----
with open(f'./data_outputs/karin_class.pkl', 'wb') as f:
    pickle.dump(karin, f)
with open(f'./data_outputs/nadir_class.pkl', 'wb') as f:
    pickle.dump(nadir, f)
print("Saved karin and nadir class objects to './data_outputs/'")

# ---- Function for processing a single frame that we paralelize ----
def process_frame(idx):
    h_obs = np.concatenate((
        karin.ssha[idx].T.flatten(order="C")[mask_k_flat],  # Karin
        nadir.ssh[idx][mask_n]                              # Nadir
    ))
    ht = swot.estimate_signal_on_target(
        c, xt, yt,
        xobs, yobs,
        C_obs, N_obs,
        h_obs
    )
    filename = f"./data_outputs/P{pass_number:03d}_C{shared_cycles[idx]:03d}_{idx:03d}.npy"
    np.save(filename, ht.reshape(nyt, nxt))
    print(f"Frame {idx:03d} → saved `{filename}`")
    return filename

# ---- Run in parallel ----
n_frames = karin.ssha.shape[0]
results = Parallel(n_jobs=4, backend='loky')(
    delayed(process_frame)(idx) for idx in range(n_frames)
)
