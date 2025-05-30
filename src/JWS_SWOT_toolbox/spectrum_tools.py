def sin2_window_func(n):
    '''sine-squared window function for spectral analysis normalized for unit variance'''
    return np.sqrt(8/3) * np.sin(np.pi * np.arange(n) / n) ** 2

def mean_power_spectrum(data, window, dim, average_dims, pers_check=True):
    '''Computes the power spectrum using xarray'''
    #pspec = xrft.power_spectrum(data, dim=dim, window='tukey', window_correction=True, scaling='density') # we can test other windows
    pspec = xrft.power_spectrum(data * window, dim=dim)
    return 2 * pspec.mean(dim=average_dims) #(factor of two is because we use one sided spectrum)

def onesided_spectrum(data,  window, dx = 2e3):
    '''Computes the one-sided power spectrum using RFFT'''
    num_useful_strips, track_length, track_width = np.shape(data)
    window_3d = np.broadcast_to(window.reshape(1, track_length, 1), (num_useful_strips, track_length, track_width))
    windowed = data * window_3d
    amp = np.fft.rfft(windowed, axis = 1) # rfft applied to windowed ssh [sample,y/2,x]
    ampsq = dx * (np.abs(amp)) ** 2 /track_length
    ampsq_1side = np.empty(ampsq.shape)
    ampsq_1side[:] = np.nan
    ampsq_1side[:, 1: ,:] = 2 * ampsq[:,1:,:]
    ampsq_1side[:, 0, :] = ampsq[:,0,:]
    spec_1side_ts = np.nanmean(ampsq_1side, axis = 2)
    spec_1side = np.nanmean(ampsq_1side, axis = tuple([0,2]))
    wavenumbers = np.fft.rfftfreq(track_length, d=dx) # Compute wavenumbers (in 1/m)
    return spec_1side, spec_1side_ts, wavenumbers

# ------- Models for fitting to power spectrum for balanced and unbalanced flows 
def nadir_noise_model(k, N):
    return np.full_like(k, N) 

def nadir_model(k, sigma_b, rho_b, sigma_u, gm_u, rho_u, N):
    nadir_bal = balanced_model(k, sigma_b, rho_b)
    nadir_noise = nadir_noise_model(k, N)
    return np.log(nadir_bal + nadir_noise)

def balanced_model(k,sigma,rho):# for balanced part
    sp = rho/(0.34026*(1+(2*np.pi*rho*k)**5))
    return sigma**2*sp

def matern_spec(k, sigma, gm, rho):# for unbalanced part
    sp = 2*np.sqrt(np.pi)*gamma(gm+1/2)*(2*gm)**gm / (gamma(gm)*rho**(2*gm)) * (2*gm/rho**2 + 4*np.pi**2*k**2)**-(gm+1/2)
    return sigma**2*sp

def karin_model(k, sigma_b, rho_b, sigma_u, gm_u, rho_u):
    return np.log(balanced_model(k, sigma_b, rho_b) + matern_spec(k, sigma_u, gm_u, rho_u))  

def parseval_check(data, pspec, dx, freq_dim='freq_line'):
    '''Use perseval's theorem to check the spectra'''
    data_flat = np.ravel(data).astype(float)
    data_flat = data_flat[np.isfinite(data_flat)]
    var_space = np.mean((data_flat - np.mean(data_flat))**2)

    if hasattr(pspec, 'dims') and hasattr(pspec, 'coords') and freq_dim in pspec.coords:
        freq_coords = pspec[freq_dim].values
        dk = np.abs(freq_coords[1] - freq_coords[0]) if len(freq_coords) > 1 else 1.0
        var_spec = np.nansum(pspec.values) * dk # we need to x 2 because its dk dl and we are doing one side
    else:
        df = 1.0 / (len(data_flat) * dx)
        var_spec = np.nansum(pspec) * df
    
    rel_error = np.abs(var_space - var_spec) / np.abs(var_space) if var_space != 0 else np.nan
    print(f"Parseval check: var_space = {var_space:.5g}, var_spec = {var_spec:.5g}, rel_error = {rel_error:.3g}")
    return var_space, var_spec, rel_error