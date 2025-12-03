# Computes the extraction for each of the KaRIn and nadir instruments independently and combined and plots Fig. 4. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, cholesky, block_diag, solve_triangular
import scipy.linalg as la
import scipy.sparse as sp
import jws_swot_tools as swot
import pickle
import os, sys
from jws_swot_tools.julia_bridge import julia_functions as jl

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
    fig, axs = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    # 1) SSH (cm)
    axs[0].plot(x_km[1:-1], ssh_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[0].plot(x_km[1:-1], ssh_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[0].set_title('SSHA', fontsize = 11)
    axs[0].set_ylabel('Std. [cm]')
    axs[0].set_ylim(0.65, 0.85)
    axs[0].legend(fontsize=9)

    # 2) Geostrophic u (cm s^-1)
    axs[1].plot(x_km[1:-1], grad_u_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[1].plot(x_km[1:-1], grad_u_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[1].set_title(r'Along-track geostrophic velocity $u_g$ ', fontsize = 11)
    axs[1].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs[1].set_ylim(7, 9)

    # 3) Geostrophic v (cm s^-1)
    axs[2].plot(x_km[1:-1], grad_v_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[2].plot(x_km[1:-1], grad_v_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[2].set_title(r'Across-track geostrophic velocity $v_g$', fontsize = 11)
    axs[2].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs[2].set_ylim(7, 9)

    # 4) Geostrophic vorticity ζ/f (—)
    axs[3].plot(x_km[1:-1], lap_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
    axs[3].plot(x_km[1:-1], lap_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
    axs[3].set_title(r'Geostrophic vorticity $\zeta_g / f$', fontsize = 11)
    axs[3].set_ylabel(r'Std.')
    axs[3].set_xlabel('Across track [km]')
    axs[3].set_ylim(0.45, 0.525)

    plt.savefig("karin_vs_nadir_std.pdf", bbox_inches="tight")
    plt.close(fig)
    print(">>> Saved: karin_vs_nadir_std.pdf")

    # ---- separate nadir-only plot ----
    fig2, axs2 = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
    fig2.subplots_adjust(hspace=0.4, wspace=0.25)

    axs2[0].plot(x_km[1:-1], ssh_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[0].set_title('SSHA')
    axs2[0].set_ylabel('Std. [cm]')
    axs2[0].legend(fontsize=9, loc='lower right')
    axs2[0].set_ylim(0, 10)

    axs2[1].plot(x_km[1:-1], grad_u_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[1].set_title(r'Along-track velocity $u_g$ ')
    axs2[1].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs2[1].set_ylim(25, 35)

    axs2[2].plot(x_km[1:-1], grad_v_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[2].set_title(r'Across-track velocity$v_g$')
    axs2[2].set_ylabel(r'Std. [cm s$^{-1}$]')
    axs2[2].set_ylim(10, 35)

    axs2[3].plot(x_km[1:-1], lap_posterior_std_n[1:-1], '-', color='tab:green', lw=1.8, label='Nadir only')
    axs2[3].set_title(r'Geostrophic vorticity $\zeta_g / f$')
    axs2[3].set_ylabel(r'Std.')
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
p_karin, _ = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)
p_nadir, _ = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, p_karin)

N_n = p_nadir[0]                # Nadir white-noise variance per sample (data cm^2)
dn  = nadir.dy_km               # km spacing along the nadir track
sigma = np.sqrt(N_n / (2 * dn)) # Nadir white-noise std dev (cm/km)
print("sigma (nadir noise std dev) = %.3f cm" % sigma)
print(f">>> Using cycle: {shared_cycles[INDEX]}")

# -----------------------------
# Observation points (km)
# -----------------------------
index = 2
mask_k = np.isfinite(karin.ssha[index])
mask_n = np.isfinite(nadir.ssh[index]).ravel()
    
xkk = (karin.x_grid[mask_k].ravel(order="C")) * 1e-3  # km
ykk = (karin.y_grid[mask_k].ravel(order="C")) * 1e-3
xnn = (nadir.x_grid.ravel()[mask_n]) * 1e-3
ynn = (nadir.y_grid.ravel()[mask_n]) * 1e-3

# Target grid (km)
xt, yt, nxt, nyt, _, _ = swot.make_target_grid(karin, unit="km", extend=False)
n_t = xt.size

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

# -----------------------------
# Covariances (obs/target)
# -----------------------------
print("  Building (C_obs, N_obs)")
SIGMA_L_KM = 0.0

# Specral Fit Parameters
B_psd = swot.balanced_psd_from_params(p_karin)                 # B(k) balanced power spectrum model
Nk_psd = swot.karin_noise_psd_from_params(p_karin)             # N_K(k) noise power spectrum model
sigma_n = np.sqrt(p_nadir[0] / (2.0 * nadir.dy_km))  # cm

# Base kernels in [cm^2]
n_samples = 100000
l_sample = 5000
kk = np.arange(n_samples // 2 + 1) / l_sample                  # wavenumber grid for transforms (200000 samples over 10000km as in swot.cov())

Tfun  = lambda k: swot.taper(k, cutoff=2.0)                    # T(k) is taper function cutoff at 2km  
C_B      = jl.cov(B_psd(kk), kk)                                          # C[B]
C_BT     = jl.cov(jl.abel(jl.iabel(B_psd(kk), kk)*Tfun(kk), kk), kk)      # C[B T]
C_NT2    = jl.cov(jl.abel(jl.iabel(Nk_psd(kk), kk)*Tfun(kk)**2, kk), kk)  # C[N T^2]

# Tapered Kernels (requires Abel transform) in [cm]
SIGMA    = 2 * np.pi * SIGMA_L_KM                              # σ convert to angular wavenumber
DELTA    = (np.pi * karin.dx_km) / (2 * np.log(2))              # δ

# Gaussian Smoothings and tapers combined
G  = lambda k: np.exp(-((SIGMA**2) * (k**2)) / 2.0)            # Gassian smooth C[BG]
G2 = lambda k: np.exp(-(SIGMA**2) * (k**2))                    # Target-Target smoothing C[BG^2]
GT = lambda k: np.exp(-(((SIGMA**2 + DELTA**2)* k**2) / 2.0) ) # Taper + Gaussian Smooth C[BGT]
T2 = lambda k: np.exp(-(DELTA**2) * (k**2))                    # Taper^2 C[BT^2]

C_B_G   = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G(kk), kk), kk)
C_B_G2  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * G2(kk), kk), kk)
C_B_TG  = jl.cov(jl.abel(jl.iabel(B_psd(kk),  kk) * GT(kk), kk), kk)
C_B_T2  = jl.cov(jl.abel(jl.iabel(B_psd(kk) + Nk_psd(kk),  kk) * T2(kk), kk), kk) # this is for k_tt

# -------------------------
# Blocks
# -------------------------
# KaRIn–KaRIn: C[(B+N_K) T^2]
# Convert the Julia Array result to a NumPy array
R_KK = np.asarray(C_B_T2(r_kk))

# Target-Target covariance for posterior C[BG^2] 
R_tt = np.asarray(C_B_G2(r_tt))

# Nadir–Nadir: C[B] + σ_N^2 I
R_NN = np.asarray(C_B(r_nn)) + (sigma_n**2) * np.eye(r_nn.shape[0])
     
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
R = np.concatenate([R_tK, R_tN], axis=1)

# -----------------------------
# Estimate signal on target
# -----------------------------
print("  Solving for posterior mean (KaRIn + Nadir)")
h_k = karin.ssha[index][mask_k].ravel()*100     # in [cm]
h_n = nadir.ssha[index][mask_n]*100             # in [cm]
h_obs = np.concatenate([h_k, h_n])

CF = la.cho_factor(C_obs, lower=True)
z  = la.cho_solve(CF, h_obs)               # (C_obs)^{-1} h
ht = R @ z                                 
ht_map = (ht / 100.0).reshape(nyt, nxt).T       # back to [m] 

# -----------------------------
# Posterior covariance on target
# P = C_tt - R (C_obs + N_obs)^{-1} R^T
# -----------------------------
print("  Computing posterior covariance diagonal (KaRIn + Nadir)")
L, lower = la.cho_factor(C_obs, lower=True, check_finite=False, overwrite_a=False)
W = solve_triangular(L, R.T, lower=True, check_finite=False, overwrite_b=False)

C_mean = W.T @ W                                        # covariance of posterior mean R @ C_obs^{-1} @ R.T
P = R_tt - C_mean                                       # Posterior covariance
posterior_variance = np.diag(P)
post_var = posterior_variance.reshape(nyt, nxt)

# -----------------------------
# Differential operators & posterior std of u,v,zeta
# -----------------------------
print("  Building finite-difference operators and mapping to u,v,zeta/f")
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
print("  Cholesky of P (KaRIn + Nadir)")
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
print("KaRIn-only posterior")
xk_obs = xkk
yk_obs = ykk

h_obs_k = h_k                        # apply to just karin
C_obs_k = R_KK                      # only karin-karin covarariance
R_k = R_tK                          # No nadir cross terms

CF_k = la.cho_factor(C_obs_k, lower=True)
z_k  = la.cho_solve(CF_k, h_obs_k)               # (C_obs)^{-1} h
ht_k = R_k @ z_k                                 
ht_map_k = (ht_k / 100.0).reshape(nyt, nxt).T       # back to [m] 

# -----------------------------
# Posterior covariance on target
# P = C_tt - R (C_obs + N_obs)^{-1} R^T
# -----------------------------
print("  Computing posterior covariance (KaRIn Only)")
L_k, lower = la.cho_factor(C_obs_k, lower=True, check_finite=False, overwrite_a=False)
W_k = solve_triangular(L_k, R_k.T, lower=True, check_finite=False, overwrite_b=False)

C_mean_k = W_k.T @ W_k                                      # covariance of posterior mean R @ C_obs^{-1} @ R.T
P_k = R_tt - C_mean_k                                       # Posterior covariance
posterior_variance_k = np.diag(P_k)
post_var_k = posterior_variance_k.reshape(nyt, nxt)

Lp_k = cholesky(P_k + 1e-12 * np.eye(P_k.shape[0]), lower=True)

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
print("  Nadir-only posterior (for comparison)")
xn_obs = xnn
yn_obs = ynn
h_obs_n = h_n  # nadir only

C_obs_n = R_NN
R_n = R_tN

CF_n = la.cho_factor(C_obs_n, lower=True)
z_n  = la.cho_solve(CF_n, h_obs_n)                  # (C_obs)^{-1} h
ht_n = R_n @ z_n                                 
ht_map_n = (ht_n / 100.0).reshape(nyt, nxt).T       # back to [m] 

print("  Computing posterior covariance (Nadir Only)")
L_n, lower = la.cho_factor(C_obs_n, lower=True, check_finite=False, overwrite_a=False)
W_n = solve_triangular(L_n, R_n.T, lower=True, check_finite=False, overwrite_b=False)

C_mean_n = W_n.T @ W_n                                        # covariance of posterior mean R @ C_obs^{-1} @ R.T
P_n = R_tt - C_mean_n                                       # Posterior covariance
posterior_variance_n = np.diag(P_n)
post_var_n = posterior_variance_n.reshape(nyt, nxt)
Lp_n = cholesky(P_n + 1e-12 * np.eye(P_n.shape[0]), lower=True)

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
X = xt.reshape(nyt, nxt)
x_km = X[0, :]  # across-track coordinate

ssh_posterior_std     = np.mean(np.sqrt(post_var), axis=0)
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
axs[0].set_title('SSHA [cm]')
axs[0].set_ylabel('Std. [cm]')
axs[0].legend(fontsize=9)

# 2) Geostrophic u (cm s^-1)
axs[1].plot(x_km[1:-1], grad_u_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[1].plot(x_km[1:-1], grad_u_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[1].set_title(r'$u_g$ [cm s$^{-1}$]')
axs[1].set_ylabel(r'Std. [cm s$^{-1}$]')

# 3) Geostrophic v (cm s^-1)
axs[2].plot(x_km[1:-1], grad_v_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[2].plot(x_km[1:-1], grad_v_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[2].set_title(r'$v_g$ [cm s$^{-1}$]')
axs[2].set_ylabel(r'Std. [cm s$^{-1}$]')

# 4) Geostrophic vorticity ζ/f (dimensionless)
axs[3].plot(x_km[1:-1], lap_posterior_std[1:-1], '-',  lw=1.8, label='KaRIn + Nadir')
axs[3].plot(x_km[1:-1], lap_posterior_std_k[1:-1], '--', lw=1.8, label='KaRIn only')
axs[3].set_title(r'Geostrophic Vorticity $\zeta / f$')
axs[3].set_ylabel(r'Std. [—]')
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
axs2[3].set_title(r'Geostrophic Vorticity $\zeta / f$')
axs2[3].set_ylabel(r'Std. [—]')
axs2[3].set_xlabel('Across Track [km]')

plt.savefig("nadir_only_std.pdf", bbox_inches="tight")
plt.close(fig2)
print(">>> Saved: nadir_only_std.pdf")


