import numpy as np
from scipy.optimize import curve_fit

def nadir_noise_model(k, N):
    return np.full_like(k, N) 

def nadir_model(k, A_b, lam_b, s_param, N):
    nadir_bal = balanced_model(k, A_b, lam_b, s_param)
    nadir_noise = nadir_noise_model(k, N)
    return np.log(nadir_bal + nadir_noise)

def balanced_model(k, A_b, lam, s_param):# for balanced part
    #sp = (lam / A_b) * 1/(1+(lam*k)**6)
    sp = A_b / (1 + (lam * k)**s_param)
    return sp

def matern_spec(k, gm, lam_u):# for unbalanced part
    sp = 2*np.pi*gamma(gm+1/2)*(2*gm)**gm / (gamma(gm)*lam_u**(2*gm)) * (2*gm/lam_u**2 + 4*np.pi**2*k**2)**-(gm+1/2)
    return sp

def unbalanced_model_notaper(k, A_n, lam_n, s_n):
    '''Model of unbalanced component without a Guassian taper at high wavenumbers'''
    lam_n = 1e5 # set to 100km because it is not well constrained
    sp = A_n / (1 + (lam_n * k)**2)**(s_n/2)
    return sp

def unbalanced_model(k, A_n, lam_n, s_n, cutoff=1e3):
    '''Model of unbalanced component with a taper at high wavenumbers'''
    sigma = 2 * np.pi * cutoff/np.sqrt(2*np.log(2))
    lam_n = 1e5  # set to 100km because it is not well constrained 
    taper = np.exp(-sigma**2*k**2/2)
    sp = A_n / (1 + (lam_n * k)**2)**(s_n/2)
    return sp * taper

def karin_model(k, A_b, lam_b, s_param, A_n, lam_n, s_n):
    return np.log(balanced_model(k, A_b, lam_b, s_param) + unbalanced_model(k, A_n, lam_n, s_n))

def fit_spectrum(k, spectrum, track_length, model, initial_guess=None, bounds=None):
    '''Fits the balanced/unbalanced models to the power spectrum'''
    spectrum_onesided = spectrum[int(track_length // 2):]
    weights = np.sqrt(k[1:])

    # Defaults for initial guess and bounds
    if initial_guess is None:
        initial_guess = [2.5e3, 200e3, 4.6, 10.0, 100e3, 1.3]
    if bounds is None:
        lower_bounds = [0, 0, 3, 0, 0, 0]
        upper_bounds = [1e9, 1e9, 10, 1e9, 1e9, 10]
        bounds = (lower_bounds, upper_bounds)

    # Fit the model (excluding zero)
    popt, pcov = curve_fit(
        model,
        k[1:],
        np.log(spectrum_onesided[1:]),
        p0=initial_guess,
        sigma=weights,
        bounds=bounds,
    )
    return popt, pcov

def fit_nadir_spectrum(k, spectrum, poptcwg_karin, initial_guess=None, bounds=None):
    ''' Fit only the noise parameter N in the nadir model, with balanced parameters fixed from KaRIn fit. '''

    weights = np.sqrt(k[1:])

    # Fixed balanced parameters from KaRIn fit
    A_b_fixed     = poptcwg_karin[0]
    lam_b_fixed   = poptcwg_karin[1]
    s_param_fixed = poptcwg_karin[2]

    # Wrapper model: only N is a free parameter!
    def model_fixed(k, N):
        return nadir_model(k, A_b_fixed, lam_b_fixed, s_param_fixed, N)

    # Set defaults for noise floor
    if initial_guess is None:
        initial_guess = [1.0]
    if bounds is None:
        bounds = ([0], [1e9])

    # Fit!
    popt, pcov = curve_fit(
        model_fixed,
        k[1:],
        np.log(spectrum[1:]),
        p0=initial_guess,
        sigma=weights,
        bounds=bounds,
    )
    return popt, pcov
