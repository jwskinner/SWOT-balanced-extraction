import numpy as np

def spatial_mean(anom, dims):
    '''returns spatial mean over specified dimensions'''
    return anom.mean(dim=dims, skipna=True)

def compute_gradient(field, dx=2000, dy=2000):
    field = np.array(field)
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)

    grad_x[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * dx)
    grad_y[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * dy)
    
    grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
    grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx
    
    grad_y[0, :] = (field[1, :] - field[0, :]) / dy
    grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dy
    return np.sqrt(grad_x**2 + grad_y**2)

def compute_laplacian(field, dx=2000, dy=2000):
    """
    Compute ∇² field with central differences in the interior and
    one-sided 2-point stencils at the boundaries.
    """
    f = np.array(field, dtype=float)
    lap = np.zeros_like(f)
    ny, nx = f.shape

    # Interior (central 3-point in both x and y)
    lap[1:-1,1:-1] = (
        (f[1:-1,2:]   - 2*f[1:-1,1:-1]   + f[1:-1,:-2]) / dx**2 +
        (f[2:,1:-1]   - 2*f[1:-1,1:-1]   + f[:-2,1:-1]) / dy**2
    )

    # Top edge (i=0): forward difference in y, central in x
    lap[0,1:-1] = (
        (f[0,2:]    - 2*f[0,1:-1]    + f[0,:-2])   / dx**2 +
        (f[2,1:-1]  - 2*f[1,1:-1]    + f[0,1:-1])   / dy**2
    )

    # Bottom edge (i=ny-1): backward difference in y, central in x
    lap[-1,1:-1] = (
        (f[-1,2:]   - 2*f[-1,1:-1]   + f[-1,:-2])  / dx**2 +
        (f[-1,1:-1] - 2*f[-2,1:-1]   + f[-3,1:-1]) / dy**2
    )

    # Left edge (j=0): forward difference in x, central in y
    lap[1:-1,0] = (
        (f[1:-1,1]  - 2*f[1:-1,0]    + f[1:-1,0])   / dx**2 +
        (f[2:,0]    - 2*f[1:-1,0]    + f[:-2,0])    / dy**2
    )

    # Right edge (j=nx-1): backward difference in x, central in y
    lap[1:-1,-1] = (
        (f[1:-1,-1]  - 2*f[1:-1,-2]  + f[1:-1,-3])  / dx**2 +
        (f[2:,-1]    - 2*f[1:-1,-1]  + f[:-2,-1])   / dy**2
    )

    # Corners (use forward/backward in both directions)
    # Top‐left (0,0):
    lap[0,0] = (
        (f[0,1]   - 2*f[0,0]  + f[0,0])  / dx**2 +
        (f[1,0]   - 2*f[0,0]  + f[0,0])  / dy**2
    )
    # Top‐right (0,nx-1):
    lap[0,-1] = (
        (f[0,-1]  - 2*f[0,-2] + f[0,-3]) / dx**2 +
        (f[1,-1]  - 2*f[0,-1] + f[0,-1]) / dy**2
    )
    # Bottom‐left (ny-1,0):
    lap[-1,0] = (
        (f[-1,1]  - 2*f[-1,0] + f[-1,0]) / dx**2 +
        (f[-1,0]  - 2*f[-2,0] + f[-3,0]) / dy**2
    )
    # Bottom‐right (ny-1,nx-1):
    lap[-1,-1] = (
        (f[-1,-1] - 2*f[-1,-2] + f[-1,-3]) / dx**2 +
        (f[-1,-1] - 2*f[-2,-1] + f[-3,-1]) / dy**2
    )

    return lap

def compute_geostrophic_vorticity_fd(ssh, dx, dy, lat, g=9.81, omega=7.2921e-5, R_e=6371e3):

    ssh = np.asarray(ssh)
    M, N = ssh.shape
    
    # Ensure latitude array matches number of rows
    lat = np.asarray(lat)
    if lat.ndim != 1 or lat.shape[0] != M:
        raise ValueError(f"lat must be 1D array of length {M}")
    
    laplacian = compute_laplacian(ssh, dx, dy)  # Laplacian of SSH

    # Coriolis parameter and its meridional gradient
    lat_rad = np.deg2rad(lat)
    f      = 2 * omega * np.sin(lat_rad)          # shape (M,)
    df_dy  = 2 * omega * np.cos(lat_rad) / R_e     # ∂f/∂y (s⁻¹ m⁻¹)
    
    f      = f[:, None]     # broadcast to (M x N)
    df_dy  = df_dy[:, None]

    # First derivative in x for beta‐term
    dssh_dx = np.zeros_like(ssh)
    dssh_dx[:, 1:-1] = (ssh[:, 2:] - ssh[:, :-2]) / (2 * dx)
    dssh_dx[:, 0]    = (ssh[:, 1] - ssh[:, 0])   / dx
    dssh_dx[:, -1]   = (ssh[:, -1] - ssh[:, -2]) / dx

    # Geostrophic vorticity
    vorticity = (g / f) * laplacian #+ (df_dy / f) * dssh_dx
    return vorticity / f