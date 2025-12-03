import numpy as np
import JWS_SWOT_toolbox as swot


def compute_geostrophic_velocity(ssh, dx, dy, lat, order = 4, g=9.81, omega=7.2921e-5, eps_f=1e-12):
    """
    Computes geostrophic velocity using nth-order finite differences for the SSH gradient.
    
    """
    ssh = np.asarray(ssh, dtype=float)
    M, N = ssh.shape

    # Latitude handling
    lat = np.asarray(lat)
    if lat.ndim == 0:  # Scalar latitude
        lat_row = np.full(M, float(lat))
    elif lat.ndim == 1 and lat.shape[0] == M:  # Latitude array per row
        lat_row = lat
    else:
        raise ValueError(f"lat must be scalar or 1D of length {M}")

    # Coriolis parameter f, broadcast to the shape of the ssh field
    lat_rad = np.deg2rad(lat_row)
    f = 2 * omega * np.sin(lat_rad)[:, None]
    
    # Avoid division by zero at the equator
    f = np.where(np.abs(f) < eps_f, np.sign(f) * eps_f + eps_f, f)

    # Compute SSH gradients using 4th-order accurate stencils everywhere
    if order == 4: 
        dssh_dx, dssh_dy = swot.compute_gradient_4th_order(ssh, dx, dy)
    else: 
        dssh_dx, dssh_dy = swot.compute_gradient_2nd_order(ssh, dx, dy)

    # Geostrophic velocities
    u_geo = -(g / f) * dssh_dy  # Zonal / across-track velocity
    v_geo = (g / f) * dssh_dx   # Meridional / along-track velocity
    speed = np.sqrt(u_geo**2 + v_geo**2)

    return u_geo, v_geo, speed

def compute_geostrophic_vorticity(ssh, dx, dy, lat, order=4, g=9.81, omega=7.2921e-5, R_e=6371e3):
    """
    Computes geostrophic vorticity using nth-order finite differences.
    """
    ssh = np.asarray(ssh)
    M, N = ssh.shape
    
    # Ensure latitude array matches number of rows for df/dy calculation
    lat = np.asarray(lat)
    if lat.ndim != 1 or lat.shape[0] != M:
        raise ValueError(f"lat must be a 1D array of length {M}")
    
    # Use 4th-order accurate functions for all derivatives
    if order==4:
        laplacian = swot.compute_laplacian_4th_order(ssh, dx, dy)
        dssh_dx, _ = swot.compute_gradient_4th_order(ssh, dx, dy) # Only need the x-derivative
    else: 
        laplacian = swot.compute_laplacian_2nd_order(ssh, dx, dy)
        dssh_dx, _ = swot.compute_gradient_2nd_order(ssh, dx, dy) 

    # Coriolis parameter and its meridional gradient
    lat_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(lat_rad)
    df_dy = 2 * omega * np.cos(lat_rad) / R_e  # β = ∂f/∂y (s⁻¹ m⁻¹)
    
    f = f[:, None]
    df_dy = df_dy[:, None]

    # To prevent division by zero, ensure f is not too small
    f[np.abs(f) < 1e-12] = 1e-12 
    
    zeta = (g / f) * laplacian
    zeta_over_f = (g / f**2) * laplacian

    return zeta_over_f

def spatial_mean(anom, dims):
    '''returns spatial mean over specified dimensions'''
    return anom.mean(dim=dims, skipna=True)

# def compute_geostrophic_velocity(ssh, dx, dy, lat, g=9.81, omega=7.2921e-5, eps_f=1e-12):
 
#     ssh = np.asarray(ssh, dtype=float)
#     M, N = ssh.shape

#     # latitude handling
#     lat = np.asarray(lat)
#     if lat.ndim == 0:               # scalar
#         lat_row = np.full(M, float(lat))
#     elif lat.ndim == 1 and lat.shape[0] == M:   # per-row
#         lat_row = lat
#     else:
#         raise ValueError(f"lat must be scalar or 1D of length {M}")

#     lat_rad = np.deg2rad(lat_row)
#     f = 2 * omega * np.sin(lat_rad)[:, None]  # shape (M,1)
#     f = np.where(np.abs(f) < eps_f, np.sign(f) * eps_f + eps_f, f)

#     # derivatives: centered differences
#     dssh_dx = np.zeros_like(ssh)
#     dssh_dx[:, 1:-1] = (ssh[:, 2:] - ssh[:, :-2]) / (2 * dx)
#     dssh_dx[:, 0] = (ssh[:, 1] - ssh[:, 0]) / dx
#     dssh_dx[:, -1] = (ssh[:, -1] - ssh[:, -2]) / dx

#     dssh_dy = np.zeros_like(ssh)
#     dssh_dy[1:-1, :] = (ssh[2:, :] - ssh[:-2, :]) / (2 * dy)
#     dssh_dy[0, :] = (ssh[1, :] - ssh[0, :]) / dy
#     dssh_dy[-1, :] = (ssh[-1, :] - ssh[-2, :]) / dy

#     # geostrophic velocities
#     u_geo = - (g / f) * dssh_dy   # zonal / across-track
#     v_geo =   (g / f) * dssh_dx   # meridional / along-track
#     speed = np.sqrt(u_geo**2 + v_geo**2)

#     return u_geo, v_geo, speed

