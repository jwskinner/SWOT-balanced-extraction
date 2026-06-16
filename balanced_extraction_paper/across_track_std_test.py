"""
In this script we will test the calc of the across track std from the posterior on
synthetic random data with same spectral characteristics as the SWOT data 
to evaluate the effect of the nadir noise on the across-track profiles.
"""

import jws_swot_tools as swot
import pickle 
from datetime import datetime, timedelta
import numpy as np
import scipy.linalg as la
from jws_swot_tools import list_available_sim_dates, DATE_FMT
from jws_swot_tools.julia_bridge import julia_functions as jl
from scipy.linalg import solve_triangular, cholesky
import matplotlib.pyplot as plt

#==================================================
# Read in the SWOT data for this pass
pass_num = 9
lat_max = 35
lat_min = 28

data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num)

sample_index = 6 
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]
karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_num)

swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)
karin.sample_index = sample_index

swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Clear Nadir Outliers 
bad_track_index = 63
nadir.ssh[bad_track_index, :] = np.nan
nadir.ssha[bad_track_index, :] = np.nan

# Generate coordinates
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()

#==================================================
# Fit spectral models

# KaRIn model fit
p_karin, _ = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)

# Nadir model fit
p_nadir, _= swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, p_karin)

#==================================================
# Grids and Covariances

# --- Grid and Spacing ---
nx, ny = 2 * karin.swath_width, karin.track_length
nn = nadir.track_length
dx, dy, dn = karin.dx_km, karin.dy_km, nadir.dy_km

# --- Covariance Functions ---
# KaRIn
c_bal = swot.balanced_covariance_func(p_karin, taper = False)
c_unb = swot.noise_covariance_func(p_karin, taper = False)

# Nadir 
N_n = p_nadir[0]
sigma_noise = 5.0 #np.sqrt(N_n / (2 * dn))
print(f"Sigma Nadir = {sigma_noise}")

# --- Observation Points ---
xk, yk = karin.x_obs_grid.flatten()*1e-3, karin.y_obs_grid.flatten()*1e-3
xn, yn = nadir.x_grid.flatten()*1e-3, nadir.y_grid.flatten()*1e-3
xobs = np.concatenate((xk, xn))
yobs = np.concatenate((yk, yn))

# --- Covariance and Noise Matrices ---
C = swot.build_covariance_matrix(c_bal, xobs, yobs)
N, Nk = swot.build_noise_matrix(c_unb, xk, yk, sigma_noise, nn, nx*ny)

# As a test, we build uncorrelated white noise on the KaRIn
def build_noise_matrix(nk_func, xk, yk, sigma, nn, n_obs):
    print("Calculating noise matrices...")

    karin_variance = nk_func(np.zeros(1))[0] # Get variance at distance 0
    Nk = karin_variance * np.eye(n_obs)      # Diagonal matrix
    
    Nn = sigma**2 * np.eye(nn)
    N = np.block([[Nk, np.zeros((n_obs, nn))], [np.zeros((nn, n_obs)), Nn]])
    return N, Nk

#N, Nk = build_noise_matrix(c_unb, xk, yk, sigma_noise, nn, nx*ny)

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

# --- Plot the generated fields ---
fig, axes = plt.subplots(5, 1, figsize=(8, 13), sharex=True,gridspec_kw={'hspace': 0.35})
vmin = -0.2
vmax = 0.2
axes[0].scatter(yk, xk, s=1,  c='blue', label='Karin', alpha=0.5)
axes[0].scatter(yn, xn, s=1, c='red',  label='Nadir', alpha=0.8)
axes[0].set_title('Karin and Nadir Grid')
axes[0].set_ylabel('cross-track (km)')
axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
axes[1].scatter(yobs, xobs, c = h/100,  s=1, vmin=vmin, vmax=vmax)
axes[1].set_title('signal')
axes[1].set_ylabel('cross-track (km)')
axes[2].scatter(yobs, xobs, c = eta/100, s=1, vmin=vmin, vmax=vmax)
axes[2].set_title('noise')
axes[2].set_ylabel('cross-track (km)')
axes[3].scatter(yobs, xobs, c = (h + eta)/100, s=1, vmin=vmin, vmax=vmax)
axes[3].set_title('signal + noise')
axes[3].set_ylabel('cross-track (km)')
im = axes[4].imshow(
    ht.reshape((nyt, nxt)).T/100, origin='lower',
    extent=np.array([0, nyt * karin.dy, 0, nxt * karin.dx]) * 1e-3,
    vmin=vmin, vmax=vmax, aspect='auto'
)
axes[4].set_title('mapped signal')
axes[4].set_xlabel('along-track distance (km)')
axes[4].set_ylabel('cross-track (km)')
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.3, pad=0.05)
cbar.set_label('SSH anomaly (m)')
plt.tight_layout()
plt.savefig("./test_field_gen.png")

#==================================================
# Posterior estimates
print("Calculating posterior covariance")

# --- Posterior Covariance Diagonal ---
L_cho, lower = cho_tuple  # already computed above
W = solve_triangular(L_cho, R.T, lower=True, check_finite=False)

# Target-target covariance (signal only, no noise)
c_bal_tt = c_bal(np.hypot(xt[:, None] - xt, yt[:, None] - yt))
C_tt = c_bal_tt  # shape (n_t, n_t)

C_mean = W.T @ W                                # R @ (C+N)^{-1} @ R^T
P_diag = np.diag(C_tt) - np.sum(W**2, axis=0)   # diag(P) = diag(C_tt - C_mean)

post_var = P_diag.reshape(nyt, nxt)
ssh_posterior_std = np.mean(np.sqrt(np.maximum(post_var, 0)), axis=0)

# --- Across-track coordinate ---
X = xt.reshape(nyt, nxt)
x_km = X[0, :]

# --- Plot across-track std profile ---
fig2, ax = plt.subplots(figsize=(7, 3))
ax.plot(x_km[1:-1], ssh_posterior_std[1:-1], lw=1.8)
ax.set_xlabel('Across-track distance (km)')
ax.set_ylabel('Posterior std (cm)')
ax.set_title('SSHA posterior std — across-track profile')
plt.tight_layout()
plt.savefig("./test_across_track_std.png")
