#!/usr/bin/env python3
import numpy as np
from scipy.fft import dct
from scipy.interpolate import interp1d
import scipy.linalg as la
import matplotlib.pyplot as plt

# -------------------------
# 1. Covariance from spectrum (Abel-related)
# -------------------------

def cov(S, k):
    """
    Given spectrum S(k) on wavenumber grid k,
    compute the radial covariance via a DCT-I (FFTW REDFT00 equivalent).

    Returns a callable C(r_values).
    """
    S = np.asarray(S, dtype=float)
    k = np.asarray(k, dtype=float)

    n = 2 * (len(k) - 1)
    L = 1.0 / k[1]        # k[2] in Julia (1-based)
    r = np.arange(0, n // 2 + 1) * L / n

    # FFTW.REDFT00 ~ DCT-I, unnormalized
    C = dct(S, type=1) / (2.0 * L)

    # Diagnostics: variance
    var_spec = (S[0] / 2.0 + S[1:n // 2].sum() + S[n // 2] / 2.0) / L
    print(f"Variance from spectrum:   {var_spec:8.6f}")
    print(f"Variance from covariance: {C[0]:8.6f}")

    # Interpolant C(r)
    itp = interp1d(
        r, C,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True
    )

    def C_interp(r_values):
        return itp(r_values)

    return C_interp


# -------------------------
# 2. sqrtint: \int_k^∞ S(κ)/sqrt(κ² - k²) dκ
# -------------------------

def sqrtint(S, k):
    """
    Calculate ∫_{k0}^∞ S(κ)/sqrt(κ² - k0²) dκ
    where k0 = k[0] in the local arrays.
    Direct translation of the Julia routine.
    """
    S = np.asarray(S, dtype=float)
    k = np.asarray(k, dtype=float)

    I = 0.0
    n = len(S)

    if np.isclose(k[0], 0.0):
        # k0 = 0 branch
        I += S[1]
        for i in range(1, n - 1):   # Julia: i = 2:length(S)-1
            ki = k[i]
            kip1 = k[i + 1]
            Si = S[i]
            Sip1 = S[i + 1]
            I += (
                (Sip1 - Si)
                + (Sip1 * ki - Si * kip1)
                  * np.log(kip1 / ki) / (ki - kip1)
            )
    else:
        # k0 != 0 branch
        k0 = k[0]
        for i in range(0, n - 1):   # Julia: i = 1:length(S)-1
            ki = k[i]
            kip1 = k[i + 1]
            Si = S[i]
            Sip1 = S[i + 1]

            sq1 = np.sqrt(ki**2 - k0**2)
            sq2 = np.sqrt(kip1**2 - k0**2)

            num = ((Sip1 - Si) * (sq1 - sq2)
                   + (Sip1 * ki - Si * kip1)
                     * np.log((ki - sq1) / (kip1 - sq2)))
            I += num / (ki - kip1)

    return I


# -------------------------
# 3. Forward Abel transform
#    S1(k) = 2/π ∫_k^∞ Sr(κ)/sqrt(κ² - k²) dκ
# -------------------------

def abel(Sr, k):
    Sr = np.asarray(Sr, dtype=float)
    k = np.asarray(k, dtype=float)
    n = len(k)
    S1 = np.zeros_like(Sr)

    for i in range(n):
        S1[i] = 2.0 / np.pi * sqrtint(Sr[i:], k[i:])

    return S1


# -------------------------
# 4. Inverse Abel transform
#    Sr(κ) = -κ ∫_κ^∞ S1'(k)/sqrt(k² - κ²) dk
# -------------------------

def iabel(S1, kappa):
    S1 = np.asarray(S1, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)

    dkap = kappa[1] - kappa[0]

    # Derivative S1'(k) with 2nd-order one-sided at ends, central inside
    S1p = np.zeros_like(S1)
    S1p[0] = (-3.0 * S1[0] + 4.0 * S1[1] - S1[2]) / (2.0 * dkap)
    S1p[1:n-1] = (S1[2:] - S1[0:n-2]) / (2.0 * dkap)
    S1p[n-1] = (S1[n-3] - 4.0 * S1[n-2] + 3.0 * S1[n-1]) / (2.0 * dkap)

    Sr = np.zeros_like(S1)
    for i in range(n):
        Sr[i] = -kappa[i] * sqrtint(S1p[i:], kappa[i:])

    return Sr


# -------------------------
# 5. Main script
# -------------------------

# Parameters
nx = 200
ny = 60
ng = 10
dk = 2e3            # Δk

nn = 59
dn = 6.8e3          # Δn

Ab = 2.7e3
lb = 223.4e3
sb = 4.7

An = 0.0            # 4.36
ln = 100e3
sn = 1.7

sigma = 0.0         # 5.2e-2

g = 9.81
f = 1e-4

# Grids
xk = (np.arange(0.5, nx, 1.0)) * dk
yk = np.concatenate([
    np.arange(-ny//2 + 0.5, -ng//2 - 0.5 + 1, 1.0),
    np.arange( ng//2 + 0.5,  ny//2 - 0.5 + 1, 1.0)
]) * dk   # length ny - ng

xn = (np.arange(0.5, nn, 1.0)) * dn
yn = np.zeros(nn)

xs = (np.arange(0.5, nx, 1.0)) * dk
ys = (np.arange(-ny//2 + 0.5, ny//2 - 0.5 + 1, 1.0)) * dk

rho = 2.0 * np.pi * 4e3
delta = np.pi * dk / (2.0 * np.log(2.0))

def B(k):
    return Ab / (1.0 + (lb * k)**sb)

def N(k):
    return An / (1.0 + (ln * k)**2)**(sn / 2.0)

# Wavenumber grid
k = np.arange(0, 200_000 + 1) / 1e7

# Spectra → covariance kernels
Bk = B(k)
Nk = N(k)

print("done")

# ckk
tmp = iabel(Bk + Nk, k)
tmp = tmp * np.exp(-delta**2 * k**2)
ckk = cov(abel(tmp, k), k)

# cnn
cnn = cov(Bk, k)
print("done")

# ckn
tmp2 = iabel(Bk, k)
tmp2 = tmp2 * np.exp(-delta**2 * k**2 / 2.0)
ckn = cov(abel(tmp2, k), k)

# css
tmp3 = iabel(Bk, k)
tmp3 = tmp3 * np.exp(-rho**2 * k**2)
css = cov(abel(tmp3, k), k)
print("done")

# csk
tmp4 = iabel(Bk, k)
tmp4 = tmp4 * np.exp(-(rho**2 + delta**2) * k**2 / 2.0)
csk = cov(abel(tmp4, k), k)
print("done")

# csn (same as cnn here)
csn = cov(Bk, k)
print("done")

# -------------------------
# 6. Coordinate vectors for pairwise distances
# -------------------------

# KaRIn (no-gap Ny)
Xk, Yk = np.meshgrid(xk, yk, indexing='ij')
Xk = Xk.ravel()
Yk = Yk.ravel()

# Full swath grid
Xs, Ys = np.meshgrid(xs, ys, indexing='ij')
Xs = Xs.ravel()
Ys = Ys.ravel()
print("done")

# Pairwise distance matrices
def pairwise_r(x1, y1, x2=None, y2=None):
    if x2 is None:
        x2, y2 = x1, y1
    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]
    return np.hypot(dx, dy)

rkk = pairwise_r(Xk, Yk)
rnn = pairwise_r(xn, yn)
rkn = pairwise_r(Xk, Yk, xn, yn)
rss = pairwise_r(Xs, Ys)
rsk = pairwise_r(Xs, Ys, Xk, Yk)
rsn = pairwise_r(Xs, Ys, xn, yn)

# Covariance matrices
Rkk = ckk(rkk)
Rnn = ckk(rnn) + (sigma**2) * np.eye(nn)
Rkn = ckk(rkn)
Rss = css(rss)
Rsk = csk(rsk)
Rsn = csk(rsn)

# -------------------------
# 7. Draw from prior
# -------------------------

jitter = 1e-15

L_prior = la.cholesky(Rss + jitter * np.eye(Rss.shape[0]), lower=True)
h = L_prior @ np.random.randn(Rss.shape[0])

# -------------------------
# 8. Posterior covariance
# -------------------------

# Block obs covariance
R_obs = np.block([
    [Rkk, Rkn],
    [Rkn.T, Rnn]
])

L_obs = la.cholesky(R_obs + jitter * np.eye(R_obs.shape[0]), lower=True)

# Stack Rsk and Rsn (note the transpose to match Julia)
RS = np.vstack([Rsk.T, Rsn.T])      # shape (Nobs, Ngrid)

# Solve L_obs v = RS
v = la.solve_triangular(L_obs, RS, lower=True)

# Posterior covariance on swath
C = Rss - v.T @ v

# -------------------------
# 9. Draw from posterior
# -------------------------

L_post = la.cholesky(C + jitter * np.eye(C.shape[0]), lower=True)
eta = L_post @ np.random.randn(C.shape[0])

# Reshape fields
h = h.reshape((nx, ny))
eta = eta.reshape((nx, ny))

# -------------------------
# 10. Vorticity (discrete Laplacian, interior)
# -------------------------

zeta = g / f * (
    h[0:-2, 1:-1] + h[2:, 1:-1] +
    h[1:-1, 0:-2] + h[1:-1, 2:] -
    4.0 * h[1:-1, 1:-1]
) / dk**2

eps = g / f * (
    eta[0:-2, 1:-1] + eta[2:, 1:-1] +
    eta[1:-1, 0:-2] + eta[1:-1, 2:] -
    4.0 * eta[1:-1, 1:-1]
) / dk**2

# -------------------------
# 11. Plot
# -------------------------

fig, axes = plt.subplots(
    4, 1,
    sharex=True,
    sharey=True,
    figsize=(6.4, 9.6),
    constrained_layout=True
)

# Panel 1: prior height
vmax = np.max(np.abs(h))
img1 = axes[0].imshow(
    h.T,
    extent=1e-3 * dk * np.array([0, nx, -ny/2, ny/2]),
    origin='lower',
    cmap='RdBu_r',
    vmin=-vmax,
    vmax=vmax,
    aspect='auto'
)

# Panel 2: prior vorticity / f
vmax = np.max(np.abs(zeta)) / f
img2 = axes[1].imshow(
    (zeta / f).T,
    extent=1e-3 * dk * np.array([1, nx-1, -ny/2+1, ny/2-1]),
    origin='lower',
    cmap='RdBu_r',
    vmin=-vmax,
    vmax=vmax,
    aspect='auto'
)

# Panel 3: posterior height
vmax = np.max(np.abs(eta))
img3 = axes[2].imshow(
    eta.T,
    extent=1e-3 * dk * np.array([0, nx, -ny/2, ny/2]),
    origin='lower',
    cmap='RdBu_r',
    vmin=-vmax,
    vmax=vmax,
    aspect='auto'
)

# Panel 4: posterior vorticity / f
vmax = np.max(np.abs(eps)) / f
img4 = axes[3].imshow(
    (eps / f).T,
    extent=1e-3 * dk * np.array([1, nx-1, -ny/2+1, ny/2-1]),
    origin='lower',
    cmap='RdBu_r',
    vmin=-vmax,
    vmax=vmax,
    aspect='auto'
)

axes[0].set_title("prior draw height (m)")
axes[1].set_title(r"prior draw vorticity ($f$)")
axes[2].set_title("posterior draw height (m)")
axes[3].set_title(r"posterior draw vorticity ($f$)")

axes[0].set_xlim(0, 1e-3 * dk * nx)
axes[0].set_ylim(-1e-3 * dk * ny / 2.0, 1e-3 * dk * ny / 2.0)
axes[-1].set_xlabel("x (km)")

# Colorbars
fig.colorbar(img1, ax=axes[0], fraction=0.06, pad=0.025, shrink=0.75)
fig.colorbar(img2, ax=axes[1], fraction=0.06, pad=0.025, shrink=0.75)
fig.colorbar(img3, ax=axes[2], fraction=0.06, pad=0.025, shrink=0.75)
fig.colorbar(img4, ax=axes[3], fraction=0.06, pad=0.025, shrink=0.75)

plt.savefig('test.png')
