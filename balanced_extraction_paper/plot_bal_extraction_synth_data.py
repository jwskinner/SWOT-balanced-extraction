import pickle
import sys 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
import cmocean
CMAP_BAL = cmocean.cm.balance
CMAP_CURL = cmocean.cm.curl
import importlib
import jws_swot_tools as swot
importlib.reload(swot)
from scipy.ndimage import gaussian_filter
sys.modules["JWS_SWOT_toolbox"] = swot

KARIN_NA_PATH = f"./synthetic_swot_data/Pass_009_Lat28_35/karin_synth.pkl" # synthetic swot data paths
NADIR_NA_PATH = f"./synthetic_swot_data/Pass_009_Lat28_35/nadir_synth.pkl"
BALANCED_PATH = f"./balanced_extraction/SYNTH_data/Pass_009_Lat28_35_rho0km/balanced_extraction_pass009.pkl"   # balanced extraction result path

INDEX = 40  # time index to plot

# -------------------
# Load data
# -------------------
with open(KARIN_NA_PATH, "rb") as f:
    karin_NA = pickle.load(f)

with open(NADIR_NA_PATH, "rb") as f:
    nadir_NA = pickle.load(f)

with open(BALANCED_PATH, "rb") as f:
    ht_all = pickle.load(f)  

# Arrays
ssh_noisy = np.asarray(karin_NA.ssh_noisy, dtype=float)  
ssha_full = np.asarray(karin_NA.ssha_full, dtype=float)  

# Grid spacings [meters]
dx_m = float(karin_NA.dx_km) * 1e3 
dy_m = float(karin_NA.dy_km) * 1e3  

# Array shapes (should all be same)
_, ny, nx = ssh_noisy.shape
nyt, nxt = ny, nx 

# function for filling the gap with NaNs
def fill_nans_rowwise(field2d):
    """Fill NaNs rowwise (across-track direction) with linear interp / nearest."""
    f = np.array(field2d, dtype=float)
    nrows, ncols = f.shape
    x = np.arange(ncols)
    for i in range(nrows):
        row = f[i]
        good = ~np.isnan(row)
        if good.sum() == 0:
            continue
        elif good.sum() == 1:
            f[i, :] = row[good][0]
        else:
            f[i, ~good] = np.interp(x[~good], x[good], row[good])
    return f

# build the lat 1D and 2D arrays
lat_2d = np.asarray(karin_NA.lat[INDEX], dtype=float)  
lats_1d = np.nanmean(lat_2d, axis=1)


# Fields 
obs_map = ssh_noisy[INDEX]                 
bal_map = np.asarray(ht_all.ssh_balanced[INDEX]).T  
truth_full = ssha_full[INDEX]          
truth_map = truth_full #truth_full[:, 5:65] 

# find the subset that minimizes the difference between
# truth map and balanced map because truth map is the full 
# observation slice and balanced is a slight crop in from the
# edges of the swath
w = bal_map.shape[1]
best_i = min(range(truth_map.shape[1] - w + 1), key=lambda i: np.nanmean((truth_map[:, i:i+w] - bal_map)**2))
print(f"Optimal truth_map slice: [{best_i}:{best_i+w}]")
truth_map = truth_map[:, best_i:best_i+w]

ssha_diff = truth_map - bal_map

# Extents (for plotting)
dy_km = karin_NA.dy_km  
dx_km = karin_NA.dx_km
shift_x = 1.0 * dx_km 

# Apply shift to the Across-track dimension (indices 0 and 1)
extent = [0, (ny * dy_km), 0, nx * dx_km + shift_x]
extent = [0, (ny * dy_km), 0, 119.5]

# -------------------
# Paper Fig. 7: SSHA
# -------------------
cmap = CMAP_BAL
vmin, vmax = -0.5, 0.5  # meters
fsize = 7

cb_kwargs = dict(fraction=0.2, pad=0.01, shrink=0.8)

fig, axs = plt.subplots(4, 1, figsize=(7, 5), sharex=True, dpi=200)

# 1) Synthetic Observation
im0 = axs[0].imshow(obs_map.T, origin="upper", cmap=cmap, aspect="equal",
                    extent=extent, vmin=vmin, vmax=vmax)

ynn = nadir_NA.y_grid.ravel()*1e-3
axs[0].scatter(
    ynn,                       
    np.full_like(ynn, 59.5),   
    c=nadir_NA.ssh_noisy[INDEX],
    s=2, cmap=cmap,
    edgecolor="none",
    vmin=vmin, vmax=vmax,
    zorder=3
)
axs[0].set_title("Synthetic Observation", fontsize=fsize)
cbar = fig.colorbar(im0, ax=axs[0], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$h$ [m]", size=fsize)

# 2) Balanced Extraction
im1 = axs[1].imshow(bal_map.T, origin="upper", cmap=cmap, aspect="equal",
                    extent=extent, vmin=vmin, vmax=vmax)
axs[1].set_title("Balanced Extraction", fontsize=fsize)
cbar = fig.colorbar(im1, ax=axs[1], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$h$ [m]", size=fsize)

# 3) NA Simulation 
im2 = axs[2].imshow(truth_map.T, origin="upper", cmap=cmap, aspect="equal",
                    extent=extent, vmin=vmin, vmax=vmax)
axs[2].set_title("Simulation", fontsize=fsize)
cbar = fig.colorbar(im2, ax=axs[2], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$h$ [m]", size=fsize)

# 4) Difference (truth - balanced) 
im3 = axs[3].imshow(ssha_diff.T*100, origin="upper", cmap="RdGy", aspect="equal",
                    extent=extent, vmin=10 * vmin, vmax=10 * vmax)
axs[3].set_title(r"Simulation $-$ Balanced Extraction", fontsize=fsize)
cbar = fig.colorbar(im3, ax=axs[3], **cb_kwargs)
cbar.set_label(r"$\Delta h$ [cm]", size=fsize)
cbar.ax.tick_params(labelsize=fsize)
cbar.update_ticks()

# Labels 
for lab, ax in zip(["(a)", "(b)", "(c)", "(d)"], axs):
    ax.text(0.001, 1.07, lab, transform=ax.transAxes, fontsize=fsize,
            va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

# Axis labels and ticks
axs[3].set_xlabel("Along track [km]", fontsize=fsize)

yticks = np.arange(0, 120 + 1, 40)
for ax in axs:
    ax.set_ylabel("Across track [km]", fontsize=fsize)
    ax.tick_params(axis="both", labelsize=fsize)
    ax.set_yticks(yticks)

plt.tight_layout()
plt.savefig("ssh_extraction.pdf", bbox_inches="tight")
print("Saved ssh_extraction.pdf")
plt.close(fig)

# -------------------
# Paper Fig 9. Geostrophic speed |u_g|
# -------------------
nan_mask = np.isnan(obs_map)
obs_filled = fill_nans_rowwise(obs_map)

# Balanced & truth maps 
h_truth_map = gaussian_filter(truth_map, sigma=0.0)  # no smoothing by default

# Geostrophic velocities (dx,dy in meters; lats length ny)
ug_obs, vg_obs, g_obs = swot.compute_geostrophic_velocity(obs_filled, dx_m, dy_m, lats_1d, order = 2)
g_obs[nan_mask] = np.nan

ug_recon, vg_recon, g_recon = swot.compute_geostrophic_velocity(bal_map, dx_m, dy_m, lats_1d, order = 2)
ug_truth, vg_truth, g_truth = swot.compute_geostrophic_velocity(h_truth_map, dx_m, dy_m, lats_1d, order = 2)

# Difference magnitude
g_diff_mag = np.sqrt((ug_truth - ug_recon) ** 2 + (vg_truth - vg_recon) ** 2)

# Plot
cmap_speed = "YlGnBu_r"

fig, axs = plt.subplots(4, 1, figsize=(7, 5), sharex=True, dpi=300)

# Panel 1: Observation speed
im0 = axs[0].imshow(g_obs.T, origin="upper", cmap=cmap_speed, aspect="equal",
                    extent=extent, vmin=0, vmax=2.0)
axs[0].set_title("Synthetic Observation", fontsize=fsize)
cbar = fig.colorbar(im0, ax=axs[0], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$|{\bf u}_g|$ [m s$^{-1}$]", size=fsize)

# Panel 2: Recovered speed
im1 = axs[1].imshow(g_recon.T, origin="upper", cmap=cmap_speed, aspect="equal",
                    extent=extent,
                    vmin=0, vmax=2.0)
axs[1].set_title("Balanced Extraction", fontsize=fsize)
cbar = fig.colorbar(im1, ax=axs[1], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$|{\bf u}_g|$ [m s$^{-1}$]", size=fsize)

# Panel 3: Truth speed
im2 = axs[2].imshow(g_truth.T, origin="upper", cmap=cmap_speed, aspect="equal",
                    extent=extent,
                    vmin=0, vmax=2.0)
axs[2].set_title("Simulation", fontsize=fsize)
cbar = fig.colorbar(im2, ax=axs[2], **cb_kwargs)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_label(r"$|{\bf u}_g|$ [m s$^{-1}$]", size=fsize)

# Panel 4: Difference magnitude
im3 = axs[3].imshow(np.abs(g_diff_mag).T, origin="upper", cmap=cmap_speed, aspect="equal",
                    extent=extent,
                    vmin=0, vmax=0.5)
axs[3].set_title(r"Difference Magnitude $|\Delta {\bf u}_g|$", fontsize=fsize)
cbar = fig.colorbar(im3, ax=axs[3], **cb_kwargs)
cbar.set_label(r"$|\Delta {\bf u}_g|$ [m s$^{-1}$]", size=fsize)
cbar.ax.tick_params(labelsize=fsize)
cbar.set_ticks([0, 0.25, 0.5])
cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x:.1f}")
cbar.update_ticks()

for lab, ax in zip(["(a)", "(b)", "(c)", "(d)"], axs):
    ax.text(0.001, 1.07, lab, transform=ax.transAxes, fontsize=fsize,
            va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

axs[3].set_xlabel("Along track [km]", fontsize=fsize)

for ax in axs:
    ax.set_ylabel("Across track [km]", fontsize=fsize)
    ax.tick_params(axis="both", labelsize=fsize)
    ax.set_yticks(yticks)

plt.tight_layout()
plt.savefig("grad_extraction.pdf", bbox_inches="tight")
print("Saved grad_extraction.pdf")
plt.close(fig)

# -------------------
# Paper Fig. 10 Geostrophic vorticity ζ/f
# -------------------
# Compute vorticities (use meters for dx/dy; expects lats length ny)
fsize = 8 # a bit larger because more panels
vort_obs = swot.compute_geostrophic_vorticity(obs_filled, dx_m, dy_m, lats_1d, order = 2)
vort_obs[nan_mask] = np.nan

vort_recon = swot.compute_geostrophic_vorticity(bal_map, dx_m, dy_m, lats_1d, order = 2)
vort_truth = swot.compute_geostrophic_vorticity(truth_map, dx_m, dy_m, lats_1d, order = 2)

vort_diff = vort_truth - vort_recon

# Gaussian filtered truth for comparison
sigma_km = 4.0
sigma_y = (sigma_km * 1e3) / dx_m   # along-track pixels
sigma_x = (sigma_km * 1e3) / dy_m   # across-track pixels

vort_truth = swot.onboard_smoothing(vort_truth, dx_m, dy_m, drop_to=0.5, frac=0.5) # apply the onboard smoothing to the NA sim truth data
vort_recon = swot.onboard_smoothing(vort_recon, dx_m, dy_m, drop_to=0.5, frac=0.5) # apply the onboard smoothing to the balanced extraction data

vort_truth_gf = gaussian_filter(vort_truth, sigma=(sigma_y, sigma_x), mode="nearest")
vort_recon_gf = gaussian_filter(vort_recon, sigma=(sigma_y, sigma_x), mode="nearest")

# Plot
cmap_vort = CMAP_CURL
vmin_v, vmax_v = -1.0, 1.0  # ζ/f range

fig, axs = plt.subplots(7, 1, figsize=(8, 9), sharex=True, dpi=300)

# 1) Noisy 
im0 = axs[0].imshow(vort_obs.T, origin="upper", cmap=cmap_vort, aspect="equal",
                    extent=extent, vmin=10*vmin_v, vmax=10*vmax_v)
axs[0].set_title("Synthetic Observation", fontsize=fsize)
cbar = fig.colorbar(im0, ax=axs[0], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# 2) Balanced (full)
im1 = axs[1].imshow(vort_recon.T, origin="upper", cmap=cmap_vort, aspect="equal",
                    extent=extent, vmin=vmin_v, vmax=vmax_v)
axs[1].set_title("Balanced Extraction", fontsize=fsize)
cbar = fig.colorbar(im1, ax=axs[1], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# 3) Truth
im3 = axs[2].imshow(vort_truth.T, origin="upper", cmap=cmap_vort, aspect="equal",
                    extent=extent, vmin=vmin_v, vmax=vmax_v)
axs[2].set_title("Simulation", fontsize=fsize)
cbar = fig.colorbar(im3, ax=axs[2], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# 4) Difference
im4 = axs[3].imshow(vort_diff.T, origin="upper", cmap="RdGy", aspect="equal",
                    extent=extent, vmin=vmin_v, vmax=vmax_v)
axs[3].set_title(r"Simulation $−$ Balanced Extraction", fontsize=fsize)
cbar = fig.colorbar(im4, ax=axs[3], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# 5) Balanced filtered
im2 = axs[4].imshow(vort_recon_gf.T, origin="upper", cmap=cmap_vort, aspect="equal",
                    extent=extent, vmin=vmin_v, vmax=vmax_v)
axs[4].set_title("Balanced Extraction Filtered", fontsize=fsize)
cbar = fig.colorbar(im2, ax=axs[4], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# 6) Truth filtered
im2 = axs[5].imshow(vort_truth_gf.T, origin="upper", cmap=cmap_vort, aspect="equal",
                    extent=extent, vmin=vmin_v, vmax=vmax_v)
axs[5].set_title("Simulation Filtered", fontsize=fsize)
cbar = fig.colorbar(im2, ax=axs[5], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)


# 7) Difference filtered
im4 = axs[6].imshow(vort_truth_gf.T - vort_recon_gf.T, origin="upper", cmap="RdGy",
                    aspect="equal", extent=extent,
                    vmin=vmin_v, vmax=vmax_v)
axs[6].set_title(r"Simulation Filtered $−$ Balanced Extraction Filtered", fontsize=fsize)
cbar = fig.colorbar(im4, ax=axs[6], **cb_kwargs)
cbar.set_label(r"$\zeta_g/f$", size=fsize)
cbar.ax.tick_params(labelsize=fsize)

# Labels (a–e)
for lab, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"], axs):
    ax.text(0.005, 1.07, lab, transform=ax.transAxes, fontsize=fsize,
            va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

for ax in axs:
    ax.set_ylabel("Across track [km]", fontsize=fsize)
    ax.tick_params(axis="both", labelsize=fsize)
    ax.set_yticks(yticks)

axs[6].set_xlabel("Along track [km]", fontsize=fsize)
plt.tight_layout()
plt.savefig("vort_extraction.pdf", bbox_inches="tight")
print("Saved vort_extraction.pdf")
plt.close(fig)

print("All figures saved: ssh_extraction.pdf, grad_extraction.pdf, vort_extraction.pdf")

print()
print("RMS(ΔSSHA) =", np.sqrt(np.nanmean(ssha_diff**2)))
print("RMS(|Δu_g|) =", np.sqrt(np.nanmean(g_diff_mag**2)))
print("RMS(Δζ/f) =", np.sqrt(np.nanmean(vort_diff**2)))
print("RMS Filt (Δζ/f) =", np.sqrt(np.nanmean(vort_truth_gf.T**2 - vort_recon_gf.T**2)))

# Coordinates (km)
x_km = np.linspace(0, ny * karin_NA.dy_km, ny)
y_km = np.linspace(0, nx * karin_NA.dx_km, nx)
extent_arr = np.array(extent)

np.savez(
    "Fig10.npz",

    # Coordinates / grid
    x_km=x_km,
    y_km=y_km,
    extent=extent_arr,
    dx_m=dx_m,
    dy_m=dy_m,
    lats_1d=lats_1d,

    # Raw vorticities
    vort_obs=vort_obs,
    vort_recon=vort_recon,
    vort_sim=vort_truth,
    vort_diff=vort_diff,

    # Filtered fields
    vort_recon_filt=vort_recon_gf,
    vort_sim_filt=vort_truth_gf,
    vort_diff_filt=(vort_truth_gf - vort_recon_gf),

    # Plot limits
    vort_limits=np.array([vmin_v, vmax_v]),
)

print("Saved: Fig10.npz")
