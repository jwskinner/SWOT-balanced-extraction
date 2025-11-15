import numpy as np

def deriv1d_2nd_order(f, h):
    """Computes the 1D first derivative using 2nd-order stencils everywhere."""
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    
    if f.size < 3:
        raise ValueError("Input array must have at least 3 points for 2nd-order derivatives.")

    # --- Interior: 2nd-order central difference ---
    # f'(x) ≈ [f(x+h) - f(x-h)] / 2h
    out[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    
    # --- Boundaries: 2nd-order forward/backward differences ---
    # Boundary i=0 (forward)
    out[0] = (-3*f[0] + 4*f[1] - f[2]) / (2 * h)
    
    # Boundary i=-1 (backward)
    out[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2 * h)
    
    return out

def laplacian1d_2nd_order(f, h):
    """Computes the 1D second derivative using 2nd-order stencils everywhere."""
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    h2 = h**2
    
    if f.size < 4:
        raise ValueError("Input array must have at least 4 points for 2nd-order boundary derivatives.")

    # --- Interior: 2nd-order central difference ---
    # f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    out[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h2

    # --- Boundaries: 2nd-order forward/backward differences ---
    # Boundary i=0 (forward)
    out[0] = (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / h2
    
    # Boundary i=-1 (backward)
    out[-1] = (2*f[-1] - 5*f[-2] + 4*f[-3] - f[-4]) / h2

    return out

def deriv1d_4th_order(f, h):
    """
    Computes the 1D first derivative using 4th-order, 5-point stencils everywhere.

    """
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)

    # Check if the array is large enough for 5-point stencils
    if f.size < 5:
        raise ValueError("Input array must have at least 5 points for 4th-order derivatives.")

    # Interior points: 5-point, 4th-order centered difference
    # Uses coeficients [1/12, -8/12, 0, 8/12, -1/12]
    # Formula: (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2]) / (12*h) 
    out[2:-2] = (f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:]) / (12 * h)

    # --- Boundary Points (5-point stencils, 4th-order accuracy) ---
    # These use coeficients [-25/12, 48/12, -36/12, 16/12, -3/12]
    # Boundary i=0: 5-point, 4th-order forward difference
    out[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12 * h)

    # Boundary i=1: 5-point, 4th-order forward difference
    out[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + 1*f[4]) / (12 * h)

    # Boundary i=-1 (last point): 5-point, 4th-order backward difference
    out[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12 * h)

    # Boundary i=-2 (second to last point): 5-point, 4th-order backward difference
    out[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - 1*f[-5]) / (12 * h)

    return out

def laplacian1d_4th_order(f, h):
    """
    Computes the 1D second derivative (Laplacian) using 4th-order stencils everywhere.

    This function maintains a consistent order of accuracy across the entire
    domain. It uses a 5-point centered difference for the interior and the
    appropriate 6-point one-sided differences at the boundaries to preserve
    4th-order accuracy.

    """
    f = np.asarray(f, dtype=float)
    out = np.zeros_like(f)
    h2 = h**2

    # Array must be large enough for 6-point boundary stencils
    if f.size < 6:
        raise ValueError("Input array must have at least 6 points for this method.")

    # Interior points: 5-point, 4th-order centered difference
    # coeficients [-1/12 4/3 -5/2 4/3 -1/12]
    out[2:-2] = (-f[:-4] + 16*f[1:-3] - 30*f[2:-2] + 16*f[3:-1] - f[4:]) / (12 * h2)

    # --- Boundary Points (6-point stencils, 4th-order accuracy) ---
    # Boundary i=0: 6-point, 4th-order forward difference
    # coeficients [45/12 -77/6 -5/2 107/6 -13 61/12 -5/6]
    out[0] = (45*f[0] - 154*f[1] + 214*f[2] - 156*f[3] + 61*f[4] - 10*f[5]) / (12 * h2)

    # Boundary i=1: 6-point, 4th-order forward difference
    out[1] = (10*f[0] - 15*f[1] - 4*f[2] + 14*f[3] - 6*f[4] + f[5]) / (12 * h2)

    # Boundary i=-1 (last point): 6-point, 4th-order backward difference
    out[-1] = (45*f[-1] - 154*f[-2] + 214*f[-3] - 156*f[-4] + 61*f[-5] - 10*f[-6]) / (12 * h2)

    # Boundary i=-2 (second to last point): 6-point, 4th-order backward difference
    out[-2] = (10*f[-1] - 15*f[-2] - 4*f[-3] + 14*f[-4] - 6*f[-5] + f[-6]) / (12 * h2)

    return out

def compute_gradient_2nd_order(field, dx=2000, dy=2000):
    """
    Computes the 2D gradient and its magnitude using 2nd-order differences.

    Returns:
        tuple: A tuple containing (grad_x, grad_y, magnitude)
    """
    field = np.array(field, dtype=float)
    
    grad_x = np.apply_along_axis(deriv1d_2nd_order, 1, field, dx)
    grad_y = np.apply_along_axis(deriv1d_2nd_order, 0, field, dy)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_x, grad_y

def compute_laplacian_2nd_order(field, dx=2000, dy=2000):
    """Computes the 2D Laplacian using 2nd-order finite differences everywhere."""
    field = np.array(field, dtype=float)
    
    d2x = np.apply_along_axis(laplacian1d_2nd_order, 1, field, dx)
    d2y = np.apply_along_axis(laplacian1d_2nd_order, 0, field, dy)
    
    return d2x + d2y


def compute_gradient_4th_order(field, dx=2000, dy=2000):
    """Computes the 2D gradient using 4th-order finite differences everywhere."""
    field = np.array(field, dtype=float)
    
    grad_x = np.apply_along_axis(deriv1d_4th_order, 1, field, dx)
    grad_y = np.apply_along_axis(deriv1d_4th_order, 0, field, dy)
    
    return grad_x, grad_y

def compute_laplacian_4th_order(field, dx=2000, dy=2000):
    """Computes the 2D Laplacian using 4th-order finite differences everywhere."""
    field = np.array(field, dtype=float)
    
    d2x = np.apply_along_axis(laplacian1d_4th_order, 1, field, dx)
    d2y = np.apply_along_axis(laplacian1d_4th_order, 0, field, dy)
    
    return d2x + d2y

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
