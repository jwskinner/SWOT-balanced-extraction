import numpy as np 
import scipy 
from scipy.special import gamma
from scipy.spatial.distance import cdist
import numpy.linalg as la
import time

def cov(s, n, L):
    k = np.arange(n // 2 + 1) / L
    r = np.arange(n // 2 + 1) * L / n
    S_k = s(k)
    C_r = scipy.fft.dct(S_k, type=1) / (2 * L)
    variance_spectrum = (S_k[0]/2 + np.sum(S_k[1:n//2]) + S_k[n//2]/2) / L
    variance_covariance = C_r[0]
    scaling_factor = variance_spectrum / variance_covariance if variance_covariance != 0 else 1.0
    C_r *= scaling_factor
    print(f"Variance from spectrum:   {variance_spectrum:8.6f}")
    print(f"Variance from covariance: {C_r[0]:8.6f}")
    return scipy.interpolate.interp1d(r, C_r, kind='linear', bounds_error=False, fill_value="extrapolate")

def make_karin_points(nx, ny, gap, delta_k):
    xk_1d = np.arange(0.5, nx, 1) * delta_k
    yk_1d_upper = np.arange(0.5, ny // 2, 1) * delta_k
    yk_1d_lower = np.arange(ny // 2 + gap + 0.5, ny + gap, 1) * delta_k
    yk_1d = np.concatenate((yk_1d_upper, yk_1d_lower))
    Xk, Yk = np.meshgrid(xk_1d, yk_1d)
    return Xk.flatten(), Yk.flatten()

def make_nadir_points(nn, ny, gap, delta_k, delta_n):
    xn = (np.arange(0.5, nn, 1)) * delta_n
    yn = (ny // 2 + gap / 2) * delta_k * np.ones(nn)
    return xn, yn

def build_covariance_matrix(cov_func, x, y):
    return cov_func(np.hypot(x[:, None] - x, y[:, None] - y))

def build_noise_matrix(nk_func, xk, yk, sigma, nn, n_obs):
    Nk = nk_func(np.hypot(xk[:, None] - xk, yk[:, None] - yk))
    Nn = sigma**2 * np.eye(nn)
    N = np.block([[Nk, np.zeros((n_obs, nn))], [np.zeros((nn, n_obs)), Nn]])
    return N, Nk

def cholesky_decomp(M, name="Matrix", jitter=False):
    print(f"Performing Cholesky decomposition for {name}...")
    start_time = time.time()
    if jitter: 
        eps = 1e-2 * np.trace(M) / M.shape[0]
        M_jittered = M + eps * np.eye(M.shape[0]) # jitter the diagonal if we need it but turned off for now
        F = la.cholesky(M_jittered)
    else: 
        F = la.cholesky(M)
    print(f"Cholesky({name}) time: {time.time() - start_time:.4f} seconds")
    return F

def generate_signal_and_noise(F, Fk, sigma, nxny, nn):
    h = F @ np.random.randn(nxny + nn)
    eta_k = Fk @ np.random.randn(nxny)
    eta_n = sigma * np.random.randn(nn)
    eta = np.concatenate((eta_k, eta_n))
    return h, eta, eta_k, eta_n

def make_target_grid(nx, ny, gap, delta_k):
    nxt = nx
    nyt = ny + gap
    xt_1d = np.arange(0.5, nxt, 1) * delta_k
    yt_1d = np.arange(0.5, nyt, 1) * delta_k
    Xt, Yt = np.meshgrid(xt_1d, yt_1d)
    return Xt.flatten(), Yt.flatten(), nxt, nyt

def estimate_signal_on_target(c, xt, yt, x, y, C, N, h):
    print("Estimating signal on target points...")
    start_time = time.time()
    R = c(np.hypot(xt[:, None] - x, yt[:, None] - y))
    print(f"shape h: {h.shape}")
    print(f"shape R: {R.shape}") 
    ht = R @ la.solve(C + N, h)
    print(f"Signal estimation time: {time.time() - start_time:.4f} seconds")
    return ht