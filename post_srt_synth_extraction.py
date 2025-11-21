#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read saved posterior P and reconstructed fields (ht_all), then compare:
- Posterior std (theory) vs Empirical std over time (truth - recon)
for SSH [cm], u_g, v_g [cm/s], and zeta/f [—], as 1D across-track (y-mean) profiles.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import cholesky

# -------------------
# Config / paths
# -------------------
PICKLES = "./pickles"
KARIN_NA_PATH   = f"{PICKLES}/karin_NA_tmean.pkl"                                         # synthetic truth & grids
HT_ALL_PATH     = f"{PICKLES}/balanced_extraction_synth_NA_tmean_sm_0km.pkl"              # saved recon (m) 
POST_PATH       = f"{PICKLES}/posterior_balanced_extraction_synth_NA_tmean_sm_0km.pkl"    # saved P (SSH, cm^2)
FIG_OUT         = "cross_track_std.pdf"

# -------------------
# Load saved products
# -------------------
print(">>> Loading saved files...")
with open(KARIN_NA_PATH, "rb") as f:
    karin_NA = pickle.load(f)

with open(HT_ALL_PATH, "rb") as f:
    ht_all_m = pickle.load(f)  # (ntime, ny, nx), meters   

with open(POST_PATH, "rb") as f:
    P = pickle.load(f)         # (ny*nx, ny*nx), cm^2      

# -------------------
# Pull truth and grid
# -------------------
h_truth_all_m = getattr(karin_NA, "ssha_full")   # (ntime, ny, nx), meters  
h_truth_all_m = h_truth_all_m[:, :, 5:64] # crop our truth field to match the SWOT area we start from index 5 and go + 50 + 9 in the gap

xg_m   = getattr(karin_NA, "x_grid")             # (ny, nx), meters
yg_m   = getattr(karin_NA, "y_grid")             # (ny, nx), meters
dx_km  = float(getattr(karin_NA, "dx_km"))
dy_km  = float(getattr(karin_NA, "dy_km"))
lat_2d = getattr(karin_NA, "lat")                # (ny, nx)

# -------------------
# convert SSH fields to cm
# -------------------
ht_all_cm      = ht_all_m * 100.0
h_truth_all_cm = h_truth_all_m * 100.0

ntime, ny, nx = h_truth_all_cm.shape
xt_km = (xg_m * 1e-3).reshape(ny, nx)[:,:nx]   
yt_km = (yg_m * 1e-3).reshape(ny, nx)[:,:nx]
xt_vec = xt_km.ravel()
yt_vec = yt_km.ravel()

print(f">>> Shapes: time={ntime}, ny={ny}, nx={nx}, P={P.shape} (cm^2)")

# -------------------
# Posterior SSH std from P
# -------------------
post_std_h_cm = np.sqrt(np.maximum(np.diag(P), 0.0)).reshape(ny, nx)  # [cm]
post_std_h_cm_1d = np.mean(post_std_h_cm, axis=0)                     # [cm], y-mean

# -------------------
# Finite-difference operators (2nd-order one-sided at edges)
# -------------------
def laplacian_operator(ny, nx, dx, dy):
    """∇^2 with 2nd-order one-sided boundaries, 2nd-order central interior; returns csr in 1/km^2."""
    L = sp.lil_matrix((ny*nx, ny*nx))
    cx = 1.0 / (dx**2); cy = 1.0 / (dy**2)
    def idx(i, j): return j*nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            # x
            if i == 0:
                L[k, idx(i, j)]     +=  2*cx
                L[k, idx(i+1, j)]   += -5*cx
                L[k, idx(i+2, j)]   +=  4*cx
                L[k, idx(i+3, j)]   += -1*cx
            elif i == nx-1:
                L[k, idx(i, j)]     +=  2*cx
                L[k, idx(i-1, j)]   += -5*cx
                L[k, idx(i-2, j)]   +=  4*cx
                L[k, idx(i-3, j)]   += -1*cx
            else:
                L[k, idx(i-1, j)]   +=  cx
                L[k, idx(i, j)]     += -2*cx
                L[k, idx(i+1, j)]   +=  cx
            # y
            if j == 0:
                L[k, idx(i, j)]     +=  2*cy
                L[k, idx(i, j+1)]   += -5*cy
                L[k, idx(i, j+2)]   +=  4*cy
                L[k, idx(i, j+3)]   += -1*cy
            elif j == ny-1:
                L[k, idx(i, j)]     +=  2*cy
                L[k, idx(i, j-1)]   += -5*cy
                L[k, idx(i, j-2)]   +=  4*cy
                L[k, idx(i, j-3)]   += -1*cy
            else:
                L[k, idx(i, j-1)]   +=  cy
                L[k, idx(i, j)]     += -2*cy
                L[k, idx(i, j+1)]   +=  cy
    return L.tocsr()

def gradient_operator(ny, nx, dx, dy):
    """(Gx,Gy) with 2nd-order one-sided boundaries; returns csr in 1/km."""
    Gx = sp.lil_matrix((ny*nx, ny*nx))
    Gy = sp.lil_matrix((ny*nx, ny*nx))
    def idx(i, j): return j*nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            # d/dx
            if i == 0:
                Gx[k, idx(i, j)]     = -1.5/dx
                Gx[k, idx(i+1, j)]   =  2.0/dx
                Gx[k, idx(i+2, j)]   = -0.5/dx
            elif i == nx-1:
                Gx[k, idx(i, j)]     =  1.5/dx
                Gx[k, idx(i-1, j)]   = -2.0/dx
                Gx[k, idx(i-2, j)]   =  0.5/dx
            else:
                Gx[k, idx(i-1, j)]   = -0.5/dx
                Gx[k, idx(i+1, j)]   =  0.5/dx
            # d/dy
            if j == 0:
                Gy[k, idx(i, j)]     = -1.5/dy
                Gy[k, idx(i, j+1)]   =  2.0/dy
                Gy[k, idx(i, j+2)]   = -0.5/dy
            elif j == ny-1:
                Gy[k, idx(i, j)]     =  1.5/dy
                Gy[k, idx(i, j-1)]   = -2.0/dy
                Gy[k, idx(i, j-2)]   =  0.5/dy
            else:
                Gy[k, idx(i, j-1)]   = -0.5/dy
                Gy[k, idx(i, j+1)]   =  0.5/dy
    return Gx.tocsr(), Gy.tocsr()

Gx, Gy = gradient_operator(ny, nx, dx_km, dy_km)
Lap    = laplacian_operator(ny, nx, dx_km, dy_km)

# -------------------
# Physical scaling & Coriolis
# -------------------
g      = 9.81
omega  = 7.2921e-5
latdeg = float(np.nanmean(lat_2d))
f      = 2.0 * omega * np.sin(np.deg2rad(latdeg))
if abs(f) < 1e-12:
    f = 1e-12 * (1.0 if f >= 0 else -1.0)

# Geometry scalings (km → m)
scale_grad = 1e-3   # 1/km -> 1/m
scale_lap  = 1e-6   # 1/km^2 -> 1/m^2

# -------------------
# Operators # i've  flipped U and V so U is along track in x-coordinate 
# -------------------
cm_to_m = 1e-2 
Vop_cm = (-(g/f) * scale_grad) * Gy * cm_to_m    # [m/s] per [cm]
Uop_cm = ( (g/f) * scale_grad) * Gx * cm_to_m    # [m/s] per [cm]
Zop_cm = ( (g/(f**2)) * scale_lap) * Lap * cm_to_m  # [—] per [cm]

# -------------------
# Posterior std for u,v,zeta/f from P (P is cm^2)
# -------------------
Ntot = P.shape[0]
try:
    Lp_cm = cholesky(P, lower=True, check_finite=False)        # [cm]
except np.linalg.LinAlgError:
    Lp_cm = cholesky(P + 1e-12*np.eye(Ntot), lower=True, check_finite=False)
Lp_cm = np.asarray(Lp_cm)

ULp = Uop_cm.dot(Lp_cm)    # [m/s]
VLp = Vop_cm.dot(Lp_cm)    # [m/s]
ZLp = Zop_cm.dot(Lp_cm)    # [—]

post_var_u = np.sum(ULp**2, axis=1).reshape(ny, nx)  # (m/s)^2
post_var_v = np.sum(VLp**2, axis=1).reshape(ny, nx)
post_var_z = np.sum(ZLp**2, axis=1).reshape(ny, nx)  # (—)^2

post_std_u_cs = np.mean(np.sqrt(post_var_u), axis=0) * 100.0  # [cm/s], y-mean
post_std_v_cs = np.mean(np.sqrt(post_var_v), axis=0) * 100.0  # [cm/s]
post_std_zeta = np.mean(np.sqrt(post_var_z), axis=0)          # [—]

# -------------------
# Empirical std over time
# -------------------
def apply_ops_cm(h2d_cm):
    """Apply operators that expect SSH in cm → (u,v,zeta/f)."""
    h_vec = np.asarray(h2d_cm).ravel()
    u = Uop_cm.dot(h_vec).reshape(ny, nx)  # [m/s]
    v = Vop_cm.dot(h_vec).reshape(ny, nx)  # [m/s]
    z = Zop_cm.dot(h_vec).reshape(ny, nx)  # [—]
    return u, v, z

print(">>> Computing empirical std over time for SSH [cm], u,v [cm/s], zeta/f [—]")
diff_h_stack_cm = h_truth_all_cm - ht_all_cm             # [cm]
diff_u_stack    = np.empty_like(ht_all_cm, dtype=float)  # [m/s]
diff_v_stack    = np.empty_like(ht_all_cm, dtype=float)  # [m/s]
diff_zeta_stack = np.empty_like(ht_all_cm, dtype=float)  # [—]

for t in range(ntime):
    uT, vT, zT = apply_ops_cm(h_truth_all_cm[t])
    uR, vR, zR = apply_ops_cm(ht_all_cm[t])
    diff_u_stack[t]    = uT - uR
    diff_v_stack[t]    = vT - vR
    diff_zeta_stack[t] = zT - zR

# y-mean of time-stds in desired units
emp_std_h_cm = np.sqrt(np.nanmean(diff_h_stack_cm**2, axis=(0, 1)))
emp_std_u_cs = np.sqrt(np.nanmean(diff_u_stack**2, axis=(0,1))) * 100
emp_std_v_cs = np.sqrt(np.nanmean(diff_v_stack**2, axis=(0,1))) * 100
emp_std_zeta = np.sqrt(np.nanmean(diff_zeta_stack**2, axis=(0,1)))

print(emp_std_h_cm.shape, emp_std_u_cs.shape, emp_std_v_cs.shape, emp_std_zeta.shape)

# old way
#emp_std_u_cs = np.nanmean(np.nanstd(diff_u_stack,     axis=0), axis=0) * 100  # [cm/s]
#emp_std_zeta = np.nanmean(np.nanstd(diff_zeta_stack,  axis=0), axis=0)        # [—]
#emp_std_v_cs = np.nanmean(np.nanstd(diff_v_stack,     axis=0), axis=0) * 100  # [cm/s]
#emp_std_h_cm = np.nanmean(np.nanstd(diff_h_stack_cm, axis=0), axis=0)         # [cm]

x_km = xt_km[0, :]

# -------------------
# Plot comparisons
# -------------------
print(f">>> Plotting: {FIG_OUT}")
fig, axs = plt.subplots(4, 1, figsize=(4.8, 10), dpi=150, sharex=True)
fig.subplots_adjust(hspace=0.4, wspace=0.25)

# 1) SSH [cm]
axs[0].plot(x_km, post_std_h_cm_1d,lw=1.8, ls='-', color='tab:blue', label='Posterior std.')
axs[0].plot(x_km, emp_std_h_cm, 'k', lw=1.8, label=r'$\Delta h$ std.')
axs[0].set_title('SSHA', fontsize=10)
axs[0].set_ylabel('Std. [cm]')
axs[0].set_ylim(0.65, 1.0)
axs[0].legend(fontsize=9, loc='upper left')

# 2) u_g [cm/s]
axs[1].plot(x_km, post_std_u_cs, lw=1.8, ls='-', color='tab:blue', label=r'Posterior std.')
axs[1].plot(x_km, emp_std_u_cs, 'k', lw=1.8, label=r'$\Delta u_g$ std.')
axs[1].set_title(r'Along-track geostrophic velocity $u_g$', fontsize=10)
axs[1].set_ylabel(r'Std. [cm s$^{-1}$]')
axs[1].set_ylim(7, 11.0)
axs[1].legend(fontsize=9, loc='upper left')

# 3) v_g [cm/s]
axs[2].plot(x_km, post_std_v_cs, lw=1.8, ls='-', color='tab:blue', label=r'Posterior std.')
axs[2].plot(x_km, emp_std_v_cs,'k', lw=1.8, label=r'$\Delta v_g$ std.')
axs[2].set_title(r'Across-track geostrophic velocity $v_g$', fontsize=10)
axs[2].set_ylabel(r'Std. [cm s$^{-1}$]')
axs[2].set_ylim(6, 12.0)
axs[2].legend(fontsize=9, loc='upper left')

# 4) zeta/f [—]
axs[3].plot(x_km, post_std_zeta, lw=1.8, ls='-', color='tab:blue', label=r'Posterior std.')
axs[3].plot(x_km, emp_std_zeta, 'k', lw=1.8, label=r'$\Delta \zeta_g/f$ std.')
axs[3].set_title(r'Geostrophic vorticity $\zeta_g/f$', fontsize=10)
axs[3].set_xlabel('Across track [km]')
axs[3].set_ylabel('Std.')
axs[3].set_ylim(0.3, 0.8)
axs[3].legend(fontsize=9, loc='upper left')

for lab, ax in zip(["(a)", "(b)", "(c)", "(d)"], axs):
    fsize = 9
    ax.text(0.001, 1.07, lab, transform=ax.transAxes, fontsize=fsize + 2,
            va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

#fig.tight_layout()
fig.savefig(FIG_OUT, bbox_inches='tight')
print(f">>> Saved: {FIG_OUT}")

# -------------------
# Print summary stats
# -------------------
def summarize(name, arr, units=""):
    a = np.asarray(arr)
    mid_val = a[len(a)//2] if len(a) > 0 else np.nan
    print(f"{name:>14}: "
          f"min={np.nanmin(a):.3f}{(' ' + units) if units else ''}, "
          f"mean={np.nanmean(a):.3f}{(' ' + units) if units else ''}, "
          f"max={np.nanmax(a):.3f}{(' ' + units) if units else ''}, "
          f"mid={mid_val:.3f}{(' ' + units) if units else ''}")

print("\n>>> Across-track summary (y-mean profiles):")
summarize("SSHA post σ", post_std_h_cm_1d, "cm")
summarize("SSHA emp σ",  emp_std_h_cm,    "cm")
summarize("u_g post σ",  post_std_u_cs,   "cm/s")
summarize("u_g emp σ",   emp_std_u_cs,    "cm/s")
summarize("v_g post σ",  post_std_v_cs,   "cm/s")
summarize("v_g emp σ",   emp_std_v_cs,    "cm/s")
summarize("ζ/f post σ",  post_std_zeta,   "")
summarize("ζ/f emp σ",   emp_std_zeta,    "")
