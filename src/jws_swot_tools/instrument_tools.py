import numpy as np
from scipy.ndimage import gaussian_filter

def onboard_smoothing(field, dx, dy, drop_to=0.5, frac=0.5):
    """
    Apply a Gaussian filter such that the field's autocorrelation
    drops to a given value (default 0.5) at a specified fraction of the pixel size.
    """

    # --- convert grid spacing to km
    dx_km, dy_km = dx * 1e-3, dy * 1e-3
    d_km = min(dx_km, dy_km)
    r_km = frac * d_km  # distance where autocorr = drop_to

    # Solve for sigma_km from e^{-(r^2)/(4σ^2)} = drop_to
    sigma_km = r_km / (2.0 * np.sqrt(np.log(1.0 / drop_to)))

    # Convert to pixel units
    sigma_pix_x = sigma_km / dx_km
    sigma_pix_y = sigma_km / dy_km

    # --- NaN-aware filtering ---
    mask = np.isfinite(field).astype(float)
    field_filled = np.where(np.isfinite(field), field, 0.0)

    num = gaussian_filter(field_filled, sigma=[sigma_pix_y, sigma_pix_x], mode="nearest")
    den = gaussian_filter(mask,         sigma=[sigma_pix_y, sigma_pix_x], mode="nearest")

    field_filt = num / np.maximum(den, 1e-12)
    field_filt[den < 1e-6] = np.nan

    return field_filt
