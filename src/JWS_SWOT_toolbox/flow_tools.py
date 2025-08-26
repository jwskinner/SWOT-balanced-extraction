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

def compute_gradient_5point(field, dx=2000, dy=2000):
    """
    Compute gradient using 5-point finite difference stencils for higher accuracy.
    
    5-point stencil: [1, -8, 0, 8, -1] / (12*h)
    Provides O(h⁴) accuracy vs O(h²) for 3-point
    """
    field = np.array(field, dtype=float)
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    ny, nx = field.shape

    # Interior: 5-point stencil (O(h⁴) accuracy)
    # ∂f/∂x ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)
    grad_x[2:-2, 2:-2] = (
        field[2:-2, :-4] - 8*field[2:-2, 1:-3] + 8*field[2:-2, 3:-1] - field[2:-2, 4:]
    ) / (12 * dx)
    
    grad_y[2:-2, 2:-2] = (
        field[:-4, 2:-2] - 8*field[1:-3, 2:-2] + 8*field[3:-1, 2:-2] - field[4:, 2:-2]
    ) / (12 * dy)

    # Near boundaries: 3-point central difference  
    # Second row/column from edges
    grad_x[2:-2, 1] = (field[2:-2, 2] - field[2:-2, 0]) / (2 * dx)
    grad_x[2:-2, -2] = (field[2:-2, -1] - field[2:-2, -3]) / (2 * dx)
    grad_x[1, 2:-2] = (field[1, 3:-1] - field[1, 1:-3]) / (2 * dx)  
    grad_x[-2, 2:-2] = (field[-2, 3:-1] - field[-2, 1:-3]) / (2 * dx)
    
    grad_y[1, 2:-2] = (field[2, 2:-2] - field[0, 2:-2]) / (2 * dy)
    grad_y[-2, 2:-2] = (field[-1, 2:-2] - field[-3, 2:-2]) / (2 * dy)
    grad_y[2:-2, 1] = (field[3:-1, 1] - field[1:-3, 1]) / (2 * dy)
    grad_y[2:-2, -2] = (field[3:-1, -2] - field[1:-3, -2]) / (2 * dy)

    # Edges: one-sided differences
    # Left edge (j=0)
    grad_x[:, 0] = (-3*field[:, 0] + 4*field[:, 1] - field[:, 2]) / (2 * dx)
    # Right edge (j=-1)  
    grad_x[:, -1] = (3*field[:, -1] - 4*field[:, -2] + field[:, -3]) / (2 * dx)
    # Top edge (i=0)
    grad_y[0, :] = (-3*field[0, :] + 4*field[1, :] - field[2, :]) / (2 * dy)
    # Bottom edge (i=-1)
    grad_y[-1, :] = (3*field[-1, :] - 4*field[-2, :] + field[-3, :]) / (2 * dy)

    # Near-edge regions that couldn't use 5-point
    # Fill in the gaps with 3-point central where possible
    for i in [1, -2]:
        for j in [1, -2]:
            if 1 <= abs(i) <= ny-2 and 1 <= abs(j) <= nx-2:
                grad_x[i, j] = (field[i, j+1] - field[i, j-1]) / (2 * dx)
                grad_y[i, j] = (field[i+1, j] - field[i-1, j]) / (2 * dy)

    # Corners and remaining edge points: use nearby values or one-sided
    # These are handled by the edge calculations above
    
    return grad_x, grad_y

def compute_gradient_magnitude_5point(field, dx=2000, dy=2000):
    """
    Compute gradient magnitude using 5-point stencils.
    Returns |∇field| = √((∂f/∂x)² + (∂f/∂y)²)
    """
    grad_x, grad_y = compute_gradient_5point(field, dx, dy)
    return np.sqrt(grad_x**2 + grad_y**2)

def compute_gradient_components_5point(field, dx=2000, dy=2000):
    """
    Compute individual gradient components using 5-point stencils.
    Returns grad_x, grad_y separately for when you need both components.
    """
    return compute_gradient_5point(field, dx, dy)

def compute_laplacian(field, dx=2000, dy=2000):
    """
    Compute ∇² field using finite differences.
    Interior: central differences
    Edges: one-sided differences 
    Corners: simple extrapolation
    """
    f = np.array(field, dtype=float)
    lap = np.zeros_like(f)
    ny, nx = f.shape

    # Interior points: central differences
    lap[1:-1, 1:-1] = (
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
    )

    # Edges using one-sided differences
    
    # Top edge (i=0): forward difference in y
    lap[0, 1:-1] = (
        (f[0, 2:] - 2*f[0, 1:-1] + f[0, :-2]) / dx**2 +
        (f[2, 1:-1] - 2*f[1, 1:-1] + f[0, 1:-1]) / dy**2  # forward in y
    )

    # Bottom edge (i=ny-1): backward difference in y
    lap[-1, 1:-1] = (
        (f[-1, 2:] - 2*f[-1, 1:-1] + f[-1, :-2]) / dx**2 +
        (f[-1, 1:-1] - 2*f[-2, 1:-1] + f[-3, 1:-1]) / dy**2  # backward in y
    )

    # Left edge (j=0): forward difference in x
    lap[1:-1, 0] = (
        (f[1:-1, 2] - 2*f[1:-1, 1] + f[1:-1, 0]) / dx**2 +  # forward in x
        (f[2:, 0] - 2*f[1:-1, 0] + f[:-2, 0]) / dy**2
    )

    # Right edge (j=nx-1): backward difference in x
    lap[1:-1, -1] = (
        (f[1:-1, -1] - 2*f[1:-1, -2] + f[1:-1, -3]) / dx**2 +  # backward in x
        (f[2:, -1] - 2*f[1:-1, -1] + f[:-2, -1]) / dy**2
    )

    # Corners: use neighboring edge values (simple approach)
    lap[0, 0] = (lap[0, 1] + lap[1, 0]) / 2
    lap[0, -1] = (lap[0, -2] + lap[1, -1]) / 2  
    lap[-1, 0] = (lap[-2, 0] + lap[-1, 1]) / 2
    lap[-1, -1] = (lap[-2, -1] + lap[-1, -2]) / 2

    return lap

def compute_laplacian_5point(field, dx=2000, dy=2000):
    """
    Compute ∇² field using 5-point finite difference stencils.
    More accurate and stable than 3-point for smooth data.
    
    5-point stencil: [-1, 16, -30, 16, -1] / (12 * h²)
    """
    f = np.array(field, dtype=float)
    lap = np.zeros_like(f)
    ny, nx = f.shape

    # Interior points: 5-point stencil (O(h⁴) accuracy)
    lap[2:-2, 2:-2] = (
        # Second derivative in x direction
        (-f[2:-2, 4:] + 16*f[2:-2, 3:-1] - 30*f[2:-2, 2:-2] + 
         16*f[2:-2, 1:-3] - f[2:-2, :-4]) / (12 * dx**2) +
        # Second derivative in y direction  
        (-f[4:, 2:-2] + 16*f[3:-1, 2:-2] - 30*f[2:-2, 2:-2] + 
         16*f[1:-3, 2:-2] - f[:-4, 2:-2]) / (12 * dy**2)
    )

    # Edge regions: fall back to 3-point stencil
    
    # Top/bottom edges (but not corners)
    lap[0, 2:-2] = (
        (-f[0, 4:] + 16*f[0, 3:-1] - 30*f[0, 2:-2] + 16*f[0, 1:-3] - f[0, :-4]) / (12 * dx**2) +
        (f[2, 2:-2] - 2*f[1, 2:-2] + f[0, 2:-2]) / dy**2
    )
    lap[1, 2:-2] = (
        (-f[1, 4:] + 16*f[1, 3:-1] - 30*f[1, 2:-2] + 16*f[1, 1:-3] - f[1, :-4]) / (12 * dx**2) +
        (f[3, 2:-2] - 2*f[2, 2:-2] + f[1, 2:-2]) / dy**2
    )
    lap[-2, 2:-2] = (
        (-f[-2, 4:] + 16*f[-2, 3:-1] - 30*f[-2, 2:-2] + 16*f[-2, 1:-3] - f[-2, :-4]) / (12 * dx**2) +
        (f[-2, 2:-2] - 2*f[-3, 2:-2] + f[-4, 2:-2]) / dy**2
    )
    lap[-1, 2:-2] = (
        (-f[-1, 4:] + 16*f[-1, 3:-1] - 30*f[-1, 2:-2] + 16*f[-1, 1:-3] - f[-1, :-4]) / (12 * dx**2) +
        (f[-1, 2:-2] - 2*f[-2, 2:-2] + f[-3, 2:-2]) / dy**2
    )

    # Left/right edges (but not corners)  
    lap[2:-2, 0] = (
        (f[2:-2, 2] - 2*f[2:-2, 1] + f[2:-2, 0]) / dx**2 +
        (-f[4:, 0] + 16*f[3:-1, 0] - 30*f[2:-2, 0] + 16*f[1:-3, 0] - f[:-4, 0]) / (12 * dy**2)
    )
    lap[2:-2, 1] = (
        (f[2:-2, 3] - 2*f[2:-2, 2] + f[2:-2, 1]) / dx**2 +
        (-f[4:, 1] + 16*f[3:-1, 1] - 30*f[2:-2, 1] + 16*f[1:-3, 1] - f[:-4, 1]) / (12 * dy**2)
    )
    lap[2:-2, -2] = (
        (f[2:-2, -2] - 2*f[2:-2, -3] + f[2:-2, -4]) / dx**2 +
        (-f[4:, -2] + 16*f[3:-1, -2] - 30*f[2:-2, -2] + 16*f[1:-3, -2] - f[:-4, -2]) / (12 * dy**2)
    )
    lap[2:-2, -1] = (
        (f[2:-2, -1] - 2*f[2:-2, -2] + f[2:-2, -3]) / dx**2 +
        (-f[4:, -1] + 16*f[3:-1, -1] - 30*f[2:-2, -1] + 16*f[1:-3, -1] - f[:-4, -1]) / (12 * dy**2)
    )

    # Corner and near-corner regions: 3-point stencil
    # Top-left quadrant
    for i in range(2):
        for j in range(2):
            lap[i, j] = (
                (f[i, j+2] - 2*f[i, j+1] + f[i, j]) / dx**2 +
                (f[i+2, j] - 2*f[i+1, j] + f[i, j]) / dy**2
            )
    
    # Top-right quadrant  
    for i in range(2):
        for j in range(nx-2, nx):
            lap[i, j] = (
                (f[i, j] - 2*f[i, j-1] + f[i, j-2]) / dx**2 +
                (f[i+2, j] - 2*f[i+1, j] + f[i, j]) / dy**2
            )
    
    # Bottom-left quadrant
    for i in range(ny-2, ny):
        for j in range(2):
            lap[i, j] = (
                (f[i, j+2] - 2*f[i, j+1] + f[i, j]) / dx**2 +
                (f[i, j] - 2*f[i-1, j] + f[i-2, j]) / dy**2
            )
    
    # Bottom-right quadrant
    for i in range(ny-2, ny):
        for j in range(nx-2, nx):
            lap[i, j] = (
                (f[i, j] - 2*f[i, j-1] + f[i, j-2]) / dx**2 +
                (f[i, j] - 2*f[i-1, j] + f[i-2, j]) / dy**2
            )

    return lap

def compute_geostrophic_vorticity_5pt(ssh, dx, dy, lat, g=9.81, omega=7.2921e-5, R_e=6371e3):
    ssh = np.asarray(ssh)
    M, N = ssh.shape
    
    # Ensure latitude array matches number of rows
    lat = np.asarray(lat)
    if lat.ndim != 1 or lat.shape[0] != M:
        raise ValueError(f"lat must be 1D array of length {M}")
    
    laplacian = compute_laplacian_5point(ssh, dx, dy)  # Laplacian of SSH

    # Coriolis parameter and its meridional gradient
    lat_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(lat_rad)          # shape (M,)
    df_dy = 2 * omega * np.cos(lat_rad) / R_e     # ∂f/∂y (s⁻¹ m⁻¹)
    
    f = f[:, None]     # broadcast to (M x N)
    df_dy = df_dy[:, None]

    # First derivative in x for beta-term
    dssh_dx = np.zeros_like(ssh)
    dssh_dx[:, 1:-1] = (ssh[:, 2:] - ssh[:, :-2]) / (2 * dx)
    dssh_dx[:, 0] = (ssh[:, 1] - ssh[:, 0]) / dx
    dssh_dx[:, -1] = (ssh[:, -1] - ssh[:, -2]) / dx

    # Relative vorticity ζ/f (dimensionless)
    zeta_over_f = (g / f**2) * laplacian + (df_dy / f**2) * dssh_dx
    return zeta_over_f
