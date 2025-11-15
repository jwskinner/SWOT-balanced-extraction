# -*- coding: utf-8 -*-
# Cleaned SWOT KaRIn + Nadir extraction, posterior variance, and 1D x-profiles

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.linalg import block_diag
import scipy.linalg as la
import scipy.sparse as sp
import JWS_SWOT_toolbox as swot
import pickle
import os, sys

CACHE = "nadir_vs_nonadir.pkl"

def plot_from_cached(out):
    print("Plotting from cached file")
    x_km = out["x_km"]
    ssh_posterior_std      = out["ssh_posterior_std"]
    grad_u_posterior_std   = out["grad_u_posterior_std"]
    grad_v_posterior_std   = out["grad_v_posterior_std"]
    lap_posterior_std      = out["lap_posterior_std"]
    ssh_posterior_std_k    = out["ssh_posterior_std_k"]
    grad_u_posterior_std_k = out["grad_u_posterior_std_k"]
    grad_v_posterior_std_k = out["grad_v_posterior_std_k"]
    lap_posterior_std_k    = out["lap_posterior_std_k"]
    ssh_posterior_std_n    = out["ssh_posterior_std_n"]
    grad_u_posterior_std_n = out["grad_u_posterior_std_n"]
    grad_v_posterior_std_n = out["grad_v_posterior_std_n"]
    lap_posterior_std_n    = out["lap_posterior_std_n"]

    print(">>> Plotting (from cache)")
    import matplotlib.pyplot as plt

    # ---- Combined plot: KaRIn+Nadir vs KaRIn-only (NO nadir curves here) ----
    fig, axs = plt.subplots(4, 1, figsize=(4.8, 11), dpi=150, sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    # 1) SSH (cm)
    axs[0].plot(x_km[1:-1], ssh_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[0].plot(x_km[1:-1], ssh_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[0].set_title('SSHA', fontsize=11)
    axs[0].set_ylabel('Std. [cm]')
    axs[0].set_ylim(0.65, 0.85)
    axs[0].legend(fontsize=9)

    # 2) Geostrophic u (cm s^-1)
    axs[1].plot(x_km[1:-1], grad_u_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[1].plot(x_km[1:-1], grad_u_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[1].set_title(r'along-track velocity $u_g$', fontsize =11)
    axs[1].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs[1].set_ylim(7, 9)

    # 3) Geostrophic v (cm s^-1)
    axs[2].plot(x_km[1:-1], grad_v_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[2].plot(x_km[1:-1], grad_v_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[2].set_title(r'across-track velocity $v_g$ ', fontsize=11)
    axs[2].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs[2].set_ylim(7, 9)

    # 4) Geostrophic vorticity ζ/f (—)
    axs[3].plot(x_km[1:-1], lap_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[3].plot(x_km[1:-1], lap_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[3].set_title(r'Geostrophic vorticity $\zeta_g / f$', fontsize=11)
    axs[3].set_ylabel(r'Std.')
    axs[3].set_xlabel('Across Track [km]')
    axs[3].set_ylim(0.45, 0.53)

    plt.savefig("karin_vs_nadir_std.pdf", bbox_inches="tight")
    plt.close(fig)
    print(">>> Saved: karin_vs_nadir_std.pdf")

    # ---- separate nadir-only plot ----
    fig2, axs2 = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
    fig2.subplots_adjust(hspace=0.4, wspace=0.25)

    axs2[0].plot(x_km[1:-1], ssh_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[0].set_title('SSHA [cm]')
    axs2[0].set_ylabel('Std. [cm]')
    axs2[0].legend(fontsize=9, loc='lower right')
    axs2[0].set_ylim(0, 10)

    axs2[1].plot(x_km[1:-1], grad_u_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[1].set_title(r'$u_g$ [cm s$^{-1}$]')
    axs2[1].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs2[1].set_ylim(25, 35)

    axs2[2].plot(x_km[1:-1], grad_v_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[2].set_title(r'$v_g$ [cm s$^{-1}$]')
    axs2[2].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs2[2].set_ylim(10, 35)

    axs2[3].plot(x_km[1:-1], lap_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[3].set_title(r'$\zeta / f$')
    axs2[3].set_ylabel(r'Std. [—]')
    axs2[3].set_xlabel('Across Track [km]')
    axs2[3].set_ylim(0.50, 0.65)

    plt.savefig("nadir_only_std.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(">>> Saved: nadir_only_std.pdf")

# ---- Early exit path if cache exists ----
if os.path.exists(CACHE):
    print(f">>> Found {CACHE} — loading and plotting only.")
    with open(CACHE, "rb") as f:
        _out = pickle.load(f)
    plot_from_cached(_out)
    sys.exit(0)

# -----------------------------
# Config
# -----------------------------
DATA_FOLDER = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/'
PASS_NUMBER = 9
LAT_MIN, LAT_MAX = 28, 35
INDEX = 40  # time index to estimate on

print(">>> Gathering file lists")
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(DATA_FOLDER, PASS_NUMBER)

# pick a sample index to build array sizes
sample_index = 2
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], LAT_MIN, LAT_MAX)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], LAT_MIN, LAT_MAX)
dims = [len(shared_cycles), track_length, track_length_nadir]

# -----------------------------
# Init data objects & load
# -----------------------------
print(">>> Initializing data classes")
karin, nadir = swot.init_swot_arrays(dims, LAT_MIN, LAT_MAX, PASS_NUMBER)

print(">>> Loading KaRIn")
swot.load_karin_data(karin_files, LAT_MIN, LAT_MAX, karin, verbose=False)
swot.process_karin_data(karin)

print(">>> Loading Nadir")
swot.load_nadir_data(nadir_files, LAT_MIN, LAT_MAX, nadir)
swot.process_nadir_data(nadir)

# Coordinates (meters) + spectra
print(">>> Building coordinates and spectra")
karin.coordinates()
nadir.coordinates()
karin.compute_spectra()
nadir.compute_spectra()

# -----------------------------
# Spectral fits -> covariance fns
# -----------------------------
print(">>> Fitting spectra")
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)
poptcwg_nadir, _ = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)

# signal and noise covariance functions (expect km)
c  = swot.balanced_covariance_func(poptcwg_karin)
nk = swot.noise_covariance_func(poptcwg_karin)

N_n = poptcwg_nadir[0]         # Nadir white-noise variance per sample (data cm^2)
dn  = nadir.dy_km              # km spacing along the nadir track
sigma = np.sqrt(N_n / (2 * dn)) # Nadir white-noise std dev (cm/km)
print("sigma (nadir noise std dev) = %.3f cm" % sigma)

print(f">>> Using cycle: {shared_cycles[INDEX]}")

# -----------------------------
# Observation points (km)
# -----------------------------
xk_km, yk_km = swot.make_karin_points(karin, unit='km')
xn_km, yn_km = swot.make_nadir_points(karin, nadir, unit='km')

# masks
mask_k = np.isfinite(karin.ssha[INDEX])
mask_n = np.isfinite(nadir.ssha[INDEX]).ravel()

# KaRIn observed values/coords
hkk     = karin.ssha[INDEX][mask_k].ravel()*100 # convert to [cm]
xkk_km  = (karin.x_grid[mask_k].ravel()) * 1e-3 # KaRIn grid [in km]
ykk_km  = (karin.y_grid[mask_k].ravel()) * 1e-3

print(karin.ssha.shape)

# Nadir observed values/coords
hn      = np.ravel(nadir.ssha[INDEX]) # [in meters]
xn_m    = np.ravel(nadir.x_grid)
yn_m    = np.ravel(nadir.y_grid)
hnn     = hn[mask_n] * 100 # convert to [cm]
xnn_km  = (xn_m[mask_n]) * 1e-3 # nadir grid[in km]
ynn_km  = (yn_m[mask_n]) * 1e-3

# Concatenate obs
h_obs   = np.concatenate([hkk, hnn])
xobs_km = np.concatenate([xkk_km, xnn_km]) # across-track, goes from 0-120
yobs_km = np.concatenate([ykk_km, ynn_km]) # along-track

# -----------------------------
# Covariances (obs/target)
# -----------------------------
print(">>> Building (C_obs, N_obs)")
C_obs = swot.build_covariance_matrix(c, xobs_km, yobs_km)

# KaRIn correlated noise among KaRIn points
dxk = xkk_km[:, None] - xkk_km[None, :]
dyk = ykk_km[:, None] - ykk_km[None, :] # grid spacing in [km]
Nk_obs = nk(np.hypot(dxk, dyk))  # nk expects [km]

# Nadir white noise block
Nn_obs = (sigma**2) * np.eye(len(xnn_km))

# block-diagonal noise
N_obs = block_diag(Nk_obs, Nn_obs)

# target grid (km)
print(">>> Making target grid")
xt_km, yt_km, nxt, nyt, _, _ = swot.make_target_grid(karin, unit='km', extend=False) # xt is 0 120km accross track, yt is along track

# cross cov R(target, obs)
R = c(np.hypot(xt_km[:, None] - xobs_km[None, :],
               yt_km[:, None] - yobs_km[None, :]))

# -----------------------------
# Estimate signal on target
# -----------------------------
print(">>> Solving for posterior mean (KaRIn + Nadir)")
CF = cho_factor(C_obs + N_obs, lower=False, check_finite=False)
ht_vec = R @ cho_solve(CF, h_obs, check_finite=False)              # posterior mean
ht_map = ht_vec.reshape(nyt, nxt)                                  # now put out in [cm]

# -----------------------------
# Posterior covariance on target
# P = C_tt - R (C_obs + N_obs)^{-1} R^T
# -----------------------------
print(">>> Computing posterior covariance diagonal (KaRIn + Nadir)")
C_target = swot.build_covariance_matrix(c, xt_km, yt_km)

term = cho_solve(CF, R.T, check_finite=False)
P = C_target - R @ term
P = 0.5 * (P + P.T)  # enforce symmetry

# diagonal -> posterior variance (SSH)
posterior_variance = np.diag(P)
posterior_variance_field = posterior_variance.reshape(nyt, nxt)

# -----------------------------
# Differential operators & posterior std of u,v,zeta
# -----------------------------
print(">>> Building finite-difference operators and mapping to u,v,zeta/f")
g = 9.81 #[in m/s]
omega = 7.2921e-5 # Earth's rotation rate [rad/s]
eps_f = 1e-12 # min Coriolis param

mean_lat_deg = float(np.nanmean(karin.lat[INDEX, :, :]))
f = 2.0 * omega * np.sin(np.deg2rad(mean_lat_deg))
if abs(f) < eps_f:
    f = np.sign(f) * eps_f if f != 0 else eps_f

def gradient_operator(nx, ny, dx_km, dy_km):
    N = ny * nx
    Gx = sp.lil_matrix((N, N))
    Gy = sp.lil_matrix((N, N))
    def idx(i, j): return j * nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            # x-derivative
            if i == 0:
                Gx[k, idx(0, j)] = -1.0 / dx_km
                Gx[k, idx(1, j)] =  1.0 / dx_km
            elif i == nx - 1:
                Gx[k, idx(nx-1, j)] =  1.0 / dx_km
                Gx[k, idx(nx-2, j)] = -1.0 / dx_km
            else:
                Gx[k, idx(i-1, j)] = -0.5 / dx_km
                Gx[k, idx(i+1, j)] =  0.5 / dx_km
            # y-derivative
            if j == 0:
                Gy[k, idx(i, 0)]  = -1.0 / dy_km
                Gy[k, idx(i, 1)]  =  1.0 / dy_km
            elif j == ny - 1:
                Gy[k, idx(i, ny-1)] =  1.0 / dy_km
                Gy[k, idx(i, ny-2)] = -1.0 / dy_km
            else:
                Gy[k, idx(i, j-1)] = -0.5 / dy_km
                Gy[k, idx(i, j+1)] =  0.5 / dy_km
    return Gx.tocsr(), Gy.tocsr()
def laplacian_operator(nx, ny, dx_km, dy_km):
    N = ny * nx
    L = sp.lil_matrix((N, N))
    dx2, dy2 = dx_km**2, dy_km**2
    def idx(i, j): return j * nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            # interior (5-point)
            if 1 <= i < nx - 1 and 1 <= j < ny - 1:
                L[k, idx(i-1, j)] += 1.0 / dx2
                L[k, idx(i+1, j)] += 1.0 / dx2
                L[k, idx(i, j-1)] += 1.0 / dy2
                L[k, idx(i, j+1)] += 1.0 / dy2
                L[k, k]           += -2.0 / dx2 - 2.0 / dy2
            else:
                # x-boundary
                if i == 0:
                    L[k, k]           += -2.0 / dx2
                    L[k, idx(i+1, j)] +=  2.0 / dx2
                elif i == nx - 1:
                    L[k, k]           += -2.0 / dx2
                    L[k, idx(i-1, j)] +=  2.0 / dx2
                else:
                    L[k, idx(i-1, j)] += 1.0 / dx2
                    L[k, idx(i+1, j)] += 1.0 / dx2
                    L[k, k]           += -2.0 / dx2
                # y-boundary
                if j == 0:
                    L[k, k]           += -2.0 / dy2
                    L[k, idx(i, j+1)] +=  2.0 / dy2
                elif j == ny - 1:
                    L[k, k]           += -2.0 / dy2
                    L[k, idx(i, j-1)] +=  2.0 / dy2
                else:
                    L[k, idx(i, j-1)] += 1.0 / dy2
                    L[k, idx(i, j+1)] += 1.0 / dy2
                    L[k, k]           += -2.0 / dy2
    return L.tocsr()

# grid spacings in km (uniform)
dx_km = float(karin.dx_km)
dy_km = float(karin.dy_km)
Gx, Gy = gradient_operator(nxt, nyt, dx_km, dy_km)
Lap    = laplacian_operator(nxt, nyt, dx_km, dy_km)

# Scale to meters for physical units
scale_grad = 1e-3   # d/dx_km -> d/dx_m
scale_lap  = 1e-6   # d2/dx2_km -> d2/dx2_m

# Operators mapping SSH 
g = 9.81
Vop = (- (g / f) * scale_grad) * Gy # y is the across track here
Uop = (  (g / f) * scale_grad) * Gx # x is the along track here
Zop = (  (g / (f**2)) * scale_lap * 0.01) * Lap # The 0.01 convert SSH [cm] to [m] because f is in [m]

# Cholesky of posterior covariance
print(">>> Cholesky of P (KaRIn + Nadir)")
try:
    Lp = cholesky(P, lower=True)
except np.linalg.LinAlgError:
    print("    P not PD, adding tiny jitter")
    Lp = cholesky(P + 1e-12 * np.eye(P.shape[0]), lower=True)

Lp_arr = np.asarray(Lp)
ULp = Uop.dot(Lp_arr)
VLp = Vop.dot(Lp_arr)
ZLp = Zop.dot(Lp_arr)

# diag(A P A^T) = sum_k (A Lp)_{:,k}^2
var_u_vec    = np.sum(np.asarray(ULp)**2, axis=1)
var_v_vec    = np.sum(np.asarray(VLp)**2, axis=1)
var_zeta_vec = np.sum(np.asarray(ZLp)**2, axis=1)
var_speed_vec = var_u_vec + var_v_vec

std_u    = np.sqrt(var_u_vec).reshape(nyt, nxt)
std_v    = np.sqrt(var_v_vec).reshape(nyt, nxt)
std_speed= np.sqrt(var_speed_vec).reshape(nyt, nxt)
std_zeta = np.sqrt(var_zeta_vec).reshape(nyt, nxt)

# -----------------------------
# KaRIn-only posterior
# -----------------------------
print(">>> KaRIn-only posterior (for comparison)")
xk_obs = xkk_km
yk_obs = ykk_km
h_obs_k = hkk # apply to just karin

C_obs_k = swot.build_covariance_matrix(c, xk_obs, yk_obs)
dxk = xk_obs[:, None] - xk_obs[None, :]
dyk = yk_obs[:, None] - yk_obs[None, :]
N_obs_k = nk(np.hypot(dxk, dyk)) # karin noise covariance

CF_k = cho_factor(C_obs_k + N_obs_k, lower=False, check_finite=False)
Rk   = c(np.hypot(xt_km[:, None] - xk_obs[None, :],
                  yt_km[:, None] - yk_obs[None, :]))

ht_vec_k = Rk @ cho_solve(CF_k, h_obs_k, check_finite=False)
ht_map_k = ht_vec_k.reshape(nyt, nxt)

# Posterior covariance (KaRIn-only)
term_k = cho_solve(CF_k, Rk.T, check_finite=False)
Pk = C_target - Rk @ term_k
Pk = 0.5 * (Pk + Pk.T)
post_var_k = np.clip(np.diag(Pk), a_min=0.0, a_max=None).reshape(nyt, nxt)

print(">>> Cholesky of P (KaRIn-only)")
try:
    Lp_k = cholesky(Pk, lower=True)
except np.linalg.LinAlgError:
    print("    Pk not PD, adding tiny jitter")
    Lp_k = cholesky(Pk + 1e-12 * np.eye(Pk.shape[0]), lower=True)

Lp_k_arr = np.asarray(Lp_k)
ULpk = Uop.dot(Lp_k_arr)
VLpk = Vop.dot(Lp_k_arr)
ZLpk = Zop.dot(Lp_k_arr)

var_u_vec_k    = np.sum(np.asarray(ULpk)**2, axis=1)
var_v_vec_k    = np.sum(np.asarray(VLpk)**2, axis=1)
var_zeta_vec_k = np.sum(np.asarray(ZLpk)**2, axis=1)
var_speed_vec_k = var_u_vec_k + var_v_vec_k

std_u_k     = np.sqrt(var_u_vec_k).reshape(nyt, nxt)
std_v_k     = np.sqrt(var_v_vec_k).reshape(nyt, nxt)
std_speed_k = np.sqrt(var_speed_vec_k).reshape(nyt, nxt)
std_zeta_k  = np.sqrt(var_zeta_vec_k).reshape(nyt, nxt)

# -----------------------------
# Nadir-only posterior 
# -----------------------------
print(">>> Nadir-only posterior (for comparison)")
xn_obs = xnn_km
yn_obs = ynn_km
h_obs_n = hnn  # nadir only

C_obs_n = swot.build_covariance_matrix(c, xn_obs, yn_obs)
N_obs_n = (sigma**2) * np.eye(len(xn_obs)) # nadir noise covariance

CF_n = cho_factor(C_obs_n + N_obs_n, lower=False, check_finite=False)
Rn   = c(np.hypot(xt_km[:, None] - xn_obs[None, :],
                  yt_km[:, None] - yn_obs[None, :]))

ht_vec_n = Rn @ cho_solve(CF_n, h_obs_n, check_finite=False)
ht_map_n = ht_vec_n.reshape(nyt, nxt)

# Posterior covariance (Nadir-only)
term_n = cho_solve(CF_n, Rn.T, check_finite=False)
Pn = C_target - Rn @ term_n
Pn = 0.5 * (Pn + Pn.T)
post_var_n = np.clip(np.diag(Pn), a_min=0.0, a_max=None).reshape(nyt, nxt)

print(">>> Cholesky of P (Nadir-only)")
try:
    Lp_n = cholesky(Pn, lower=True)
except np.linalg.LinAlgError:
    print("    Pn not PD, adding tiny jitter")
    Lp_n = cholesky(Pn + 1e-12 * np.eye(Pn.shape[0]), lower=True)

Lp_n_arr = np.asarray(Lp_n)
ULpn = Uop.dot(Lp_n_arr)
VLpn = Vop.dot(Lp_n_arr)
ZLpn = Zop.dot(Lp_n_arr)

var_u_vec_n    = np.sum(np.asarray(ULpn)**2, axis=1)
var_v_vec_n    = np.sum(np.asarray(VLpn)**2, axis=1)
var_zeta_vec_n = np.sum(np.asarray(ZLpn)**2, axis=1)
var_speed_vec_n = var_u_vec_n + var_v_vec_n

std_u_n     = np.sqrt(var_u_vec_n).reshape(nyt, nxt)
std_v_n     = np.sqrt(var_v_vec_n).reshape(nyt, nxt)
std_speed_n = np.sqrt(var_speed_vec_n).reshape(nyt, nxt)
std_zeta_n  = np.sqrt(var_zeta_vec_n).reshape(nyt, nxt)


# -----------------------------
# Build 1D across-track profiles for plotting
# Use x-coordinate from the target grid (reshape then take a row)
# -----------------------------
X = xt_km.reshape(nyt, nxt)
x_km = X[0, :]  # across-track coordinate

ssh_posterior_std     = np.mean(np.sqrt(posterior_variance_field), axis=0)
grad_u_posterior_std   = np.mean(std_u, axis=0)
grad_v_posterior_std   = np.mean(std_v, axis=0)
lap_posterior_std     = np.mean(std_zeta, axis=0)

ssh_posterior_std_k   = np.mean(np.sqrt(post_var_k), axis=0)
grad_u_posterior_std_k = np.mean(std_u_k, axis=0)
grad_v_posterior_std_k = np.mean(std_v_k, axis=0)
lap_posterior_std_k   = np.mean(std_zeta_k, axis=0)

ssh_posterior_std_n     = np.mean(np.sqrt(post_var_n), axis=0)
grad_u_posterior_std_n  = np.mean(std_u_n, axis=0)
grad_v_posterior_std_n  = np.mean(std_v_n, axis=0)
lap_posterior_std_n     = np.mean(std_zeta_n, axis=0)

out = {
    "x_km": np.asarray(x_km),
    "ssh_posterior_std":       np.asarray(ssh_posterior_std),
    "grad_u_posterior_std":    np.asarray(grad_u_posterior_std),
    "grad_v_posterior_std":    np.asarray(grad_v_posterior_std),
    "lap_posterior_std":       np.asarray(lap_posterior_std),
    "ssh_posterior_std_k":     np.asarray(ssh_posterior_std_k),
    "grad_u_posterior_std_k":  np.asarray(grad_u_posterior_std_k),
    "grad_v_posterior_std_k":  np.asarray(grad_v_posterior_std_k),
    "lap_posterior_std_k":     np.asarray(lap_posterior_std_k),
    "ssh_posterior_std_n":     np.asarray(ssh_posterior_std_n),
    "grad_u_posterior_std_n":  np.asarray(grad_u_posterior_std_n),
    "grad_v_posterior_std_n":  np.asarray(grad_v_posterior_std_n),
    "lap_posterior_std_n":     np.asarray(lap_posterior_std_n),
}

with open("nadir_vs_nonadir.pkl", "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved nadir_vs_nonadir.pkl")

# -----------------------------
# Plot
# -----------------------------
print(">>> Plotting")

# ---- Combined plot: KaRIn+Nadir vs KaRIn-only (NO nadir curves here) ----
fig, axs = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
fig.subplots_adjust(hspace=0.4, wspace=0.25)

# 1) SSH (cm)
axs[0].plot(x_km[1:-1], ssh_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[0].plot(x_km[1:-1], ssh_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[0].set_title('SSHA')
axs[0].set_ylabel('Std. [cm]')
axs[0].legend(fontsize=9)

# 2) Geostrophic u (cm s^-1)
axs[1].plot(x_km[1:-1], grad_u_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[1].plot(x_km[1:-1], grad_u_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[1].set_title(r'along-track velocity $u_g$')
axs[1].set_ylabel(r'Std. [cm s$^{-1}$]')

# 3) Geostrophic v (cm s^-1)
axs[2].plot(x_km[1:-1], grad_v_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[2].plot(x_km[1:-1], grad_v_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[2].set_title(r'across-track velocity $v_g$')
axs[2].set_ylabel(r'Std. [cm s$^{-1}$]')

# 4) Geostrophic vorticity ζ/f (dimensionless)
axs[3].plot(x_km[1:-1], lap_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[3].plot(x_km[1:-1], lap_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[3].set_title(r'Geostrophic Vorticity $\zeta / f$')
axs[3].set_ylabel(r'Std.')
axs[3].set_xlabel('Across Track [km]')

plt.savefig("karin_vs_nadir_std.pdf", bbox_inches="tight")
plt.close(fig)
print(">>> Saved: karin_vs_nadir_std.pdf")

# ---- separate nadir-only plot ----
fig2, axs2 = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
fig2.subplots_adjust(hspace=0.4, wspace=0.25)

axs2[0].plot(x_km[1:-1], ssh_posterior_std_n[1:-1], ':',  lw=1.8, label='Nadir only')
axs2[0].set_title('SSHA [cm]')
axs2[0].set_ylabel('Std. [cm]')
axs2[0].legend(fontsize=9)

axs2[1].plot(x_km[1:-1], grad_u_posterior_std_n[1:-1], ':',  lw=1.8, label='Nadir only')
axs2[1].set_title(r'$u_g$ [cm s$^{-1}$]')
axs2[1].set_ylabel(r'Std. [cm s$^{-1}$]')

axs2[2].plot(x_km[1:-1], grad_v_posterior_std_n[1:-1], ':',  lw=1.8, label='Nadir only')
axs2[2].set_title(r'$v_g$ [cm s$^{-1}$]')
axs2[2].set_ylabel(r'Std. [cm s$^{-1}$]')

axs2[3].plot(x_km[1:-1], lap_posterior_std_n[1:-1], ':',  lw=1.8, label='Nadir only')
axs2[3].set_title(r'Geostrophic Vorticity $\zeta_g / f$')
axs2[3].set_ylabel(r'Std. [—]')
axs2[3].set_xlabel('Across Track [km]')

plt.savefig("nadir_only_std.pdf", bbox_inches="tight")
plt.close(fig2)
print(">>> Saved: nadir_only_std.pdf")

print("\n=== BASIC CONTEXT ===")
print(f"Cycle index: {INDEX}  (cycle ID: {shared_cycles[INDEX]})")
print(f"Target grid: nyt={nyt}, nxt={nxt}, dx_km={karin.dx_km:.3f}, dy_km={karin.dy_km:.3f}")
print(f"Obs counts -> KaRIn: {hkk.size},  Nadir: {hnn.size},  Total: {h_obs.size}")

print("\n=== NOISE / COVARIANCE BLOCKS ===")
print(f"C_obs shape: {C_obs.shape},  N_obs shape: {N_obs.shape}")
print(f"KaRIn noise block Nk_obs shape: {Nk_obs.shape}")
print(f"Nadir noise block Nn_obs shape: {Nn_obs.shape}")
print(f"Nadir white-noise sigma (data units): {sigma:.6g}")

print("\n=== POSITIVE-DEFINITE CHECKS ===")
print(f"P (with nadir) min diag: {np.min(np.diag(P)):.6g}, max diag: {np.max(np.diag(P)):.6g}")
print(f"Pk (KaRIn-only) min diag: {np.min(np.diag(Pk)):.6g}, max diag: {np.max(np.diag(Pk)):.6g}")

print("\n=== CENTERLINE / KEY INDICES ===")
ix0 = int(np.argmin(np.abs(x_km)))
print(f"Across-track center index ix0={ix0}, x_km[ix0]={x_km[ix0]:.3f}")

print("\n=== SSH POSTERIOR STD (cm) ===")
ssh_std_cm      = 100.0*np.mean(np.sqrt(posterior_variance_field), axis=0)
ssh_std_k_cm    = 100.0*np.mean(np.sqrt(post_var_k), axis=0)
print(f"SSH std at center: K+N={ssh_std_cm[ix0]:.3f} cm,  K-only={ssh_std_k_cm[ix0]:.3f} cm,  Δ={ssh_std_k_cm[ix0]-ssh_std_cm[ix0]:.3f} cm")
print(f"SSH std (min/mean/max): K+N=({ssh_std_cm.min():.3f}/{ssh_std_cm.mean():.3f}/{ssh_std_cm.max():.3f}) cm, "
      f"K-only=({ssh_std_k_cm.min():.3f}/{ssh_std_k_cm.mean():.3f}/{ssh_std_k_cm.max():.3f}) cm")

print("\n=== VELOCITY STD (cm s^-1) ===")
ug_std_cm       = 100.0*np.mean(std_u, axis=0)
vg_std_cm       = 100.0*np.mean(std_v, axis=0)
ug_std_k_cm     = 100.0*np.mean(std_u_k, axis=0)
vg_std_k_cm     = 100.0*np.mean(std_v_k, axis=0)
print(f"u_g std at center: K+N={ug_std_cm[ix0]:.3f},  K-only={ug_std_k_cm[ix0]:.3f},  Δ={ug_std_k_cm[ix0]-ug_std_cm[ix0]:.3f}")
print(f"v_g std at center: K+N={vg_std_cm[ix0]:.3f},  K-only={vg_std_k_cm[ix0]:.3f},  Δ={vg_std_k_cm[ix0]-vg_std_cm[ix0]:.3f}")
print(f"u_g std (min/mean/max): K+N=({ug_std_cm.min():.3f}/{ug_std_cm.mean():.3f}/{ug_std_cm.max():.3f}), "
      f"K-only=({ug_std_k_cm.min():.3f}/{ug_std_k_cm.mean():.3f}/{ug_std_k_cm.max():.3f})")
print(f"v_g std (min/mean/max): K+N=({vg_std_cm.min():.3f}/{vg_std_cm.mean():.3f}/{vg_std_cm.max():.3f}), "
      f"K-only=({vg_std_k_cm.min():.3f}/{vg_std_k_cm.mean():.3f}/{vg_std_k_cm.max():.3f})")

print("\n=== VORTICITY STD (ζ/f, dimensionless) ===")
zeta_std        = np.mean(std_zeta, axis=0)
zeta_std_k      = np.mean(std_zeta_k, axis=0)
print(f"ζ/f std at center: K+N={zeta_std[ix0]:.5f},  K-only={zeta_std_k[ix0]:.5f},  Δ={zeta_std_k[ix0]-zeta_std[ix0]:.5f}")
print(f"ζ/f std (min/mean/max): K+N=({zeta_std.min():.5f}/{zeta_std.mean():.5f}/{zeta_std.max():.5f}), "
      f"K-only=({zeta_std_k.min():.5f}/{zeta_std_k.mean():.5f}/{zeta_std_k.max():.5f})")

print("\n=== PERCENT REDUCTIONS AT CENTERLINE (K+N vs K-only) ===")
def pct(old, new): 
    return 100.0 * (old - new) / max(old, 1e-12)
print(f"SSH:   {pct(ssh_std_k_cm[ix0],  ssh_std_cm[ix0]):6.2f}%")
print(f"u_g:   {pct(ug_std_k_cm[ix0],   ug_std_cm[ix0]):6.2f}%")
print(f"v_g:   {pct(vg_std_k_cm[ix0],   vg_std_cm[ix0]):6.2f}%")
print(f"ζ/f:   {pct(zeta_std_k[ix0],    zeta_std[ix0]):6.2f}%")

print("\n=== GLOBAL MEAN PERCENT REDUCTIONS (y-mean profiles) ===")
def pct_arr(old, new):
    return 100.0*np.nanmean((old - new)/np.maximum(old, 1e-12))
print(f"SSH mean Δ%: {pct_arr(ssh_std_k_cm, ssh_std_cm):6.2f}%")
print(f"u_g mean Δ%: {pct_arr(ug_std_k_cm,  ug_std_cm ):6.2f}%")
print(f"v_g mean Δ%: {pct_arr(vg_std_k_cm,  vg_std_cm ):6.2f}%")
print(f"ζ/f mean Δ%: {pct_arr(zeta_std_k,   zeta_std   ):6.2f}%")

print("\n=== SHAPES FOR SANITY ===")
print(f"x_km: {x_km.shape},  profiles -> SSH: {ssh_std_cm.shape}, u: {ug_std_cm.shape}, v: {vg_std_cm.shape}, ζ/f: {zeta_std.shape}")

x_nadir = float(np.nanmedian(xnn_km))            # nadir x [km]
ix_nadir = int(np.argmin(np.abs(x_km - x_nadir)))
print(f"\n[NADIR CENTERLINE] x_nadir≈{x_nadir:.3f} km  -> ix_nadir={ix_nadir}, x_km[ix_nadir]={x_km[ix_nadir]:.3f} km")

plt.figure(figsize=(6,1.5))
plt.scatter(xkk_km, np.zeros_like(xkk_km), s=5, label='KaRIn')
plt.scatter(xnn_km, np.zeros_like(xnn_km)+0.1, s=10, label='Nadir')
plt.xlabel("Across-track x [km]")
plt.yticks([])
plt.legend()
plt.title("Observation geometry (KaRIn vs Nadir)")
plt.savefig("test.pdf", bbox_inches="tight")

# ---------- higher-precision centerline numbers ----------
def pct(old, new): 
    return 100.0 * (old - new) / max(old, 1e-12)

np.set_printoptions(precision=6, suppress=True)

print("\n=== HIGH-PRECISION AT NADIR COLUMN ===")
print(f"ix_nadir={ix_nadir}, x_km[ix_nadir]={x_km[ix_nadir]:.6f} km, x_nadir≈{x_nadir:.6f} km")

# SSH (cm), u/v (cm/s), zeta/f (—)
ssh_std_cm   = 100.0*np.mean(np.sqrt(posterior_variance_field), axis=0)
ssh_std_k_cm = 100.0*np.mean(np.sqrt(post_var_k),               axis=0)

ug_std_cm    = 100.0*np.mean(std_u,   axis=0)
vg_std_cm    = 100.0*np.mean(std_v,   axis=0)
ug_std_k_cm  = 100.0*np.mean(std_u_k, axis=0)
vg_std_k_cm  = 100.0*np.mean(std_v_k, axis=0)

zeta_std     = np.mean(std_zeta,   axis=0)
zeta_std_k   = np.mean(std_zeta_k, axis=0)

def show(label, k_only, k_n, units):
    d_abs = k_only[ix_nadir] - k_n[ix_nadir]
    d_pct = pct(k_only[ix_nadir], k_n[ix_nadir])
    print(f"{label:7s}: K+N={k_n[ix_nadir]:.6f}  K-only={k_only[ix_nadir]:.6f}  "
          f"Δabs={d_abs:.6f} {units}  Δ%={d_pct:.6f}%")

show("SSH",   ssh_std_k_cm, ssh_std_cm,   "cm")
show("u_g",   ug_std_k_cm,  ug_std_cm,    "cm/s")
show("v_g",   vg_std_k_cm,  vg_std_cm,    "cm/s")
show("zeta/f",zeta_std_k,   zeta_std,     "—")

# ---------- delta profiles (so small reductions are visible) ----------
d_ssh_cm  = ssh_std_k_cm - ssh_std_cm
d_ug_cm   = ug_std_k_cm  - ug_std_cm
d_vg_cm   = vg_std_k_cm  - vg_std_cm
d_zeta    = zeta_std_k   - zeta_std

print("\n=== DELTA PROFILE SUMMARIES (K-only minus K+N) ===")
def summarize(name, arr, units):
    print(f"{name:6s}: min={arr.min():.6f}  mean={arr.mean():.6f}  max={arr.max():.6f} ({units})")
summarize("SSH",  d_ssh_cm, "cm")
summarize("u_g",  d_ug_cm,  "cm/s")
summarize("v_g",  d_vg_cm,  "cm/s")
summarize("zeta", d_zeta,   "—")
