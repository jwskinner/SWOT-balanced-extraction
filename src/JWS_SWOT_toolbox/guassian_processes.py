import numpy as np 
import scipy 
from scipy.special import gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import numpy.linalg as la
import time
import numpy as np
import JWS_SWOT_toolbox as swot

# --- Pairwise distance matrix ---
def pairwise_r(x0, y0, x1=None, y1=None):
    # Euclidean distances between (x0,y0) and (x1,y1)
    if x1 is None:  # square
        dx = x0[:, None] - x0[None, :]
        dy = y0[:, None] - y0[None, :]
    else:
        dx = x0[:, None] - x1[None, :]
        dy = y0[:, None] - y1[None, :]
    return np.hypot(dx, dy)

# --- Grid Functions ---
def make_karin_points(karin, unit):
    nx = karin.track_length
    ny = 2 * karin.swath_width
    gap = karin.middle_width
    delta_k = karin.dx

    xk_1d = np.arange(0.5, nx, 1) * delta_k
    yk_1d_upper = np.arange(0.5, ny // 2, 1) * delta_k
    yk_1d_lower = np.arange(ny // 2 + gap + 0.5, ny + gap, 1) * delta_k

    yk_1d = np.concatenate((yk_1d_upper, yk_1d_lower))
    Xk, Yk = np.meshgrid(xk_1d, yk_1d)
    if unit =='km':
        return Xk.flatten()*1e-3, Yk.flatten()*1e-3
    else: 
        return Xk.flatten(), Yk.flatten()

def make_nadir_points(karin, nadir, unit, offset=0):
    nn = nadir.track_length
    ny = 2*karin.swath_width
    delta_k = karin.dx
    delta_n = nadir.dy
    gap = karin.middle_width

    xn = (np.arange(0.5, nn, 1)) * delta_n + offset
    yn = (ny // 2 + gap / 2) * delta_k * np.ones(nn)
    if unit=='km':
        return xn*1e-3, yn*1e-3
    else:
        return xn, yn

def make_karin_points_from_data(karin, index): # Converts karin lats, lons into m grid
    lons = karin.lon[index, :, :]
    lats = karin.lat[index, :, :]

    nx, ny = np.shape(lons)

    valid_mask = (~np.isnan(lons)) & (~np.isnan(lats))

    lat0 = lats[0, 0]
    lon0 = lons[0, 0]

    # x: distance along latitude circle (east-west) - take dx component
    x_m, _, _ = swot.projected_distance(lon0, lat0, lons, np.full_like(lats, lat0))
    
    # y: distance along longitude circle (north-south) - take dy component  
    _, y_m, _ = swot.projected_distance(lon0, lat0, np.full_like(lons, lon0), lats)

    # Shift to make all distances positive
    x_shifted = x_m - np.nanmin(x_m)
    y_shifted = y_m - np.nanmin(y_m)

    x_valid = x_shifted[valid_mask]
    y_valid = y_shifted[valid_mask]

    # Using the karin grid structure we will return the target grid also 
    x_target, y_target = make_target_grid_from_data(x_shifted, y_shifted, valid_mask)

    return x_valid, y_valid, x_target, y_target

def make_target_grid(karin, unit, extend=False, dx=None, dy=None):

    # Use observed x/y extent from the data class 
    x_min = np.nanmin(karin.x_grid)
    x_max = np.nanmax(karin.x_grid)
    y_min = np.nanmin(karin.y_grid)
    y_max = np.nanmax(karin.y_grid)

    # Default to KaRIn spacing if not provided
    if dx is None:
        dx = karin.dx
    if dy is None:
        dy = karin.dy

    # Extension for ST analysis (pads with ~2 grid points on each side)
    if extend:
        x_min -= 2 * dx
        x_max += 2 * dx

    # 1D grid arrays
    x_target = np.arange(x_min, x_max + dx, dx) # 0.5dx because we are on a half grid
    y_target = np.arange(y_min, y_max + dy, dy)

    if unit == 'km':
        x_target *= 1e-3
        y_target *= 1e-3
        
    Xt, Yt = np.meshgrid(x_target, y_target)

    return Xt.flatten(), Yt.flatten(), len(x_target), len(y_target), x_target, y_target

def make_target_grid_from_data(x_shifted, y_shifted, valid_mask, extra_width=4):
    """Create target grid that fills the nadir gap between swaths"""
    
    n_rows, n_cols = x_shifted.shape
    
    # Create the target grid by filling the nadir gap
    x_target = x_shifted.copy()
    y_target = y_shifted.copy()
    
    # For each row, interpolate across the nadir gap
    for i in range(n_rows):
        row_x = x_shifted[i, :]
        row_y = y_shifted[i, :]
        
        # Find valid (non-NaN) indices
        valid_indices = np.where(~np.isnan(row_x))[0]
        
        if len(valid_indices) > 0:
            # Find gaps in the valid indices (nadir region)
            if len(valid_indices) > 1:
                # Check for gaps larger than 1 (indicating nadir)
                gaps = np.diff(valid_indices)
                large_gaps = np.where(gaps > 1)[0]
                
                for gap_idx in large_gaps:
                    # Indices of the gap boundaries
                    left_valid_idx = valid_indices[gap_idx]
                    right_valid_idx = valid_indices[gap_idx + 1]
                    
                    # Fill the gap between left_valid_idx and right_valid_idx
                    gap_start = left_valid_idx + 1
                    gap_end = right_valid_idx
                    gap_length = gap_end - gap_start
                    
                    if gap_length > 0:
                        # Linear interpolation
                        x_left = row_x[left_valid_idx]
                        x_right = row_x[right_valid_idx]
                        y_left = row_y[left_valid_idx]
                        y_right = row_y[right_valid_idx]
                        
                        # Interpolate x and y values
                        for j in range(gap_length):
                            alpha = (j + 1) / (gap_length + 1)
                            x_target[i, gap_start + j] = x_left + alpha * (x_right - x_left)
                            y_target[i, gap_start + j] = y_left + alpha * (y_right - y_left)
    
    # Add extra width if requested
    if extra_width > 0:
        n_cols_new = n_cols + extra_width
        x_target_extended = np.full((n_rows, n_cols_new), np.nan)
        y_target_extended = np.full((n_rows, n_cols_new), np.nan)
        
        # Copy filled data
        x_target_extended[:, :n_cols] = x_target
        y_target_extended[:, :n_cols] = y_target
        
        # Extrapolate the extra columns
        for i in range(n_rows):
            row_x = x_target[i, :]
            row_y = y_target[i, :]
            valid_indices = np.where(~np.isnan(row_x))[0]
            
            if len(valid_indices) >= 2:
                last_idx = valid_indices[-1]
                second_last_idx = valid_indices[-2]
                
                dx = row_x[last_idx] - row_x[second_last_idx]
                dy = row_y[last_idx] - row_y[second_last_idx]
                
                for j in range(extra_width):
                    x_target_extended[i, n_cols + j] = row_x[last_idx] + (j + 1) * dx
                    y_target_extended[i, n_cols + j] = row_y[last_idx] + (j + 1) * dy
        
        return x_target_extended, y_target_extended
    
    return x_target, y_target

def make_nadir_points_from_data(karin, nadir, index):
    lons = nadir.lon[index, :]  
    lats = nadir.lat[index, :]

    # we use our KaRIn point as our reference here
    k_lons = karin.lon[index, :, :]
    k_lats = karin.lat[index, :, :]
    lat0 = k_lats[0, 0]
    lon0 = k_lons[0, 0]
    
    if lons.ndim > 1:
        lons = lons.squeeze()
        lats = lats.squeeze()
    
    valid_mask = (~np.isnan(lons)) & (~np.isnan(lats))
    
    # x: distance along latitude circle (east-west) - take dx component
    x_m, _, _ = swot.projected_distance(lon0, lat0, lons, np.full_like(lats, lat0))
    
    # y: distance along longitude circle (north-south) - take dy component  
    _, y_m, _ = swot.projected_distance(lon0, lat0, np.full_like(lons, lon0), lats)
    
    # Apply the shifts from the karin data
    karin_x_m, _, _ = swot.projected_distance(lon0, lat0, k_lons, np.full_like(k_lats, lat0))
    _, karin_y_m, _ = swot.projected_distance(lon0, lat0, np.full_like(k_lons, lon0), k_lats)
    
    # Use KaRIn min values to ensure consistent coordinate system
    x_min_karin = np.nanmin(karin_x_m)
    y_min_karin = np.nanmin(karin_y_m)
    
    # Apply same shifts as KaRIn
    x_shifted = x_m - x_min_karin
    y_shifted = y_m - y_min_karin
    
    x_valid = x_shifted[valid_mask]
    y_valid = y_shifted[valid_mask]
    
    return x_valid, y_valid

# --- Covariance Functions --- 
def cov(s, L = 5000, n=100000): # default 10,000km with 200,000 samples so 50m sampling resolution
    k = np.arange(n // 2 + 1) / L
    r = np.arange(n // 2 + 1) * L / n
    if callable(s): # we can pass either a function or an array in here
        S_k = s(k)
    else: 
        S_k = s
    C_r = scipy.fft.dct(S_k, type=1) / (2 * L)
    variance_spectrum = (S_k[0]/2 + np.sum(S_k[1:n//2]) + S_k[n//2]/2) / L
    variance_covariance = C_r[0]
    scaling_factor = variance_spectrum / variance_covariance if variance_covariance != 0 else 1.0
    C_r *= scaling_factor
    
    # Compute spacing and max wavenumber
    delta_k = 1 / L
    k_max = n / (2 * L)
    print("")
    print("---- Hankel Transform ----")
    print(f"Δk (spacing): {delta_k:.6f} cpkm")
    print(f"k_max (maximum wavenumber): {k_max:.6f} cpkm")
    print(f"Variance from spectrum:   {variance_spectrum:8.6f}")
    print(f"Variance from covariance: {C_r[0]:8.6f}")
    print("")
    return scipy.interpolate.interp1d(r, C_r, kind='cubic', bounds_error=False, fill_value="extrapolate")

def cov_2d_isotropic(s, L, n):
    """
    Isotropic 2D covariance C(r) from a radial PSD S(k) using the Hankel transform:
      C(r) = ∫_0^∞ S(k) J0(2π k r) 2π k dk
    """
    from scipy.special import j0

    # k-grid (uniform like yours)
    k = np.arange(n // 2 + 1) / L                         # [cpkm]
    dk = 1.0 / L                                          # [cpkm]
    S_k = s(k)                                            # [variance per cpkm^2]

    # r-grid to match your output domain [0, L/2]
    r = np.arange(n // 2 + 1) * L / n                     # [km]

    # 2D Hankel (order 0): C(r) = Σ S(k) J0(2π k r) (2π k dk)
    kr = (2.0 * np.pi) * np.outer(r, k)                   # dimensionless
    J = j0(kr)
    weights = 2.0 * np.pi * k * dk                        # (2π k dk)

    C_r = (J * (S_k * weights)[None, :]).sum(axis=1)
    integrand = S_k * k
    variance_spectrum = 2.0 * np.pi * (
        0.5 * integrand[0] + integrand[1:-1].sum() + 0.5 * integrand[-1]
    ) * dk
    variance_covariance = C_r[0]
    if variance_covariance != 0:
        C_r *= (variance_spectrum / variance_covariance)

    # Report the same diagnostics you print today
    delta_k = dk
    k_max = n / (2.0 * L)
    print("")
    print("---- Hankel Transform ----")
    print(f"Δk (spacing): {delta_k:.6f} cpkm")
    print(f"k_max (maximum wavenumber): {k_max:.6f} cpkm")
    print(f"Variance from spectrum:   {variance_spectrum:8.6f}")
    print(f"Variance from covariance: {C_r[0]:8.6f}")
    print("")

    return scipy.interpolate.interp1d(r, C_r, kind="cubic", bounds_error=False, fill_value="extrapolate")

def make_cov_from_psd(S, L=5_000, n=5_000_000, isotropic_2d=False):
    """
    Wrapper to compute covariance from a PSD S(k). isotropic_2d=True uses 2D isotropic Hankel transform.
    """
    if isotropic_2d:
        return cov_2d_isotropic(S, L, n)
    else:
        return swot.cov(S, L, n)

def build_covariance_matrix(cov_func, x, y):
    print("Calculating covariance matrices...")
    return cov_func(np.hypot(x[:, None] - x, y[:, None] - y))

def build_noise_matrix(nk_func, xk, yk, sigma, nn, n_obs):
    print("Calculating noise matrices...")
    Nk = nk_func(np.hypot(xk[:, None] - xk, yk[:, None] - yk))
    Nn = sigma**2 * np.eye(nn)
    N = np.block([[Nk, np.zeros((n_obs, nn))], [np.zeros((nn, n_obs)), Nn]])
    return N, Nk

def balanced_covariance_func(
    params,
    cutoff=2.0,
    max_radius=5_000,
    num_points=100_000, # gives 5m resolution
    smooth=False,
    scale_km=None,
):

    # Unpack parameters
    A_b, lam_b, s_param = params[0], params[1], params[2]

    # Base spectral shape S0(k)
    S0 = lambda k: (A_b / (1.0 + (lam_b * k) ** s_param)) * (swot.taper(k, cutoff) ** 2)

    # Apply optional smoothing in spectral space
    if smooth and (scale_km is not None) and (scale_km > 0):
        sigma_L = float(scale_km)
        G = lambda k: np.exp(-2.0 * (np.pi * k * sigma_L)**2)
        S = lambda k: S0(k) * G(k)
        print(f"Gaussian smoothing: σ_L={sigma_L:.3f} km")
    else:
        S = S0

    # Build and return the spatial covariance function
    cov_func = swot.cov(S, max_radius, num_points)
    return cov_func

def noise_covariance_func(params, lam_n=100, cutoff=2.0, max_radius=5_000, num_points=100_000):
    """
    Creates a spatial covariance function for the unbalanced (KaRIN noise) component
    from a parameter vector.
    """
    # Unpack parameters from the input vector
    A_n, s_n = params[3], params[5]
    
    # Define the spectral shape Sk(k) with a Gaussian taper
    Sk = lambda k: A_n / (1 + (lam_n * k)**2)**(s_n / 2) * swot.taper(k, cutoff)
    
    # Use swot.cov to get the spatial covariance function
    cov_func = swot.cov(Sk, max_radius, num_points)
    
    return cov_func

def cholesky_decomp(M, name="Matrix", jitter=False):
    print(f"Performing Cholesky decomposition for {name}...")
    start_time = time.time()
    if jitter: 
        eps = 1e-8 * np.trace(M) / M.shape[0]
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

def generate_synthetic_realizations(swot, F, Fk, sigma_noise, nx, ny, nn, n_realizations):
    """Generate n number of synthetic signal and noise realizations."""
    hs_list = []
    etas_list = []
    etas_k = []
    etas_n = []
    
    for i in range(n_realizations):
        h, eta, eta_k, eta_n = swot.generate_signal_and_noise(F, Fk, sigma_noise, nx * ny, nn)
        etas_k.append(eta_k.reshape(ny, nx))
        etas_n.append(eta_n)
        hs_list.append(h)
        etas_list.append(eta)
    
    return (np.array(hs_list, dtype=object), 
            np.array(etas_list, dtype=object),
            np.array(etas_k, dtype=object), 
            np.array(etas_n, dtype=object))

def estimate_signal_on_target_cho_solve(R, cho_factor_tuple, h, output=True):
    """
    Estimates the signal using the factor from cho_factor and cho_solve.
    """
    start_time = time.time()
    
    # Solve (C+N)z = h efficiently using the pre-computed factor
    z = cho_solve(cho_factor_tuple, h)
    
    # Calculate the final estimate
    ht = R @ z

    if output:
        print(f"Signal estimation time: {time.time() - start_time:.4f} seconds")
    return ht

def estimate_signal_on_target(c, xt, yt, x, y, C, N, h):
    print("Estimating signal on target points...")
    start_time = time.time()
    R = c(np.hypot(xt[:, None] - x, yt[:, None] - y))
    ht = R @ la.solve(C + N, h)
    print(f"Signal estimation time: {time.time() - start_time:.4f} seconds")
    return ht

def estimate_signal_on_target_blocked(c, xt, yt, x, y, C, N, h, block_size=2000):
    """
    Compute ht = R @ (C+N)^{-1} h in blocks to reduce memory.
    Arguments are the same as your original function.
    """
    print("Estimating signal on target points...")
    start_time = time.time()

    # precompute weights once
    w = la.solve(C + N, h)

    M = len(xt)
    ht = np.empty(M, dtype=w.dtype)

    # loop over blocks of target points
    for i0 in range(0, M, block_size):
        i1 = min(M, i0 + block_size)
        dx = xt[i0:i1, None] - x[None, :]
        dy = yt[i0:i1, None] - y[None, :]
        r = np.hypot(dx, dy)       # (block, K)
        Rb = c(r)                  # (block, K)
        ht[i0:i1] = Rb @ w         # multiply block

    print(f"Signal estimation time: {time.time() - start_time:.4f} seconds")
    return ht

def estimate_signal_on_target_fast(R, C, N, h):
    # same as above but we already have the R matrix, good for loops
    print("Estimating signal on target points...")
    start_time = time.time()
    ht = R @ la.solve(C + N, h)
    print(f"Signal estimation time: {time.time() - start_time:.4f} seconds")
    return ht

# ---- Updated covariances with different tapers for each component (eqs 6,7 in paper) ----
def balanced_psd_from_params(params):
    A_b, lam_b, s_b = params[0], params[1], params[2]
    return lambda k: A_b / (1.0 + (lam_b * k)**s_b)

def karin_noise_psd_from_params(params, lam_n=100):
    # Using (A_n, s_n) at indices [3], [5] from the KaRIn fit
    A_n, s_n = params[3], params[5]
    return lambda k: A_n / (1.0 + (lam_n * k)**2)**(0.5 * s_n)

def nadir_noise_std(N_n, delta_n):
    return np.sqrt(N_n / (2 * delta_n))