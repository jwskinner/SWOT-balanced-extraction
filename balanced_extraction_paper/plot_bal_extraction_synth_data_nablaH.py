import importlib
import pickle
import sys

import cmocean
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import gaussian_filter

import jws_swot_tools as swot

importlib.reload(swot)
sys.modules["JWS_SWOT_toolbox"] = swot

KARIN_NA_PATH = f"./synthetic_swot_data/Pass_009_Lat28_35/karin_synth.pkl"
NADIR_NA_PATH = f"./synthetic_swot_data/Pass_009_Lat28_35/nadir_synth.pkl"
BALANCED_PATH = f"./balanced_extraction/SYNTH_data/Pass_009_Lat28_35_rho0km/balanced_extraction_pass009.pkl"

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

# Array shapes
_, ny, nx = ssh_noisy.shape


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


# Fields
obs_map = ssh_noisy[INDEX]
bal_map = np.asarray(ht_all.ssh_balanced[INDEX]).T
truth_full = ssha_full[INDEX]
truth_map = truth_full

w = bal_map.shape[1]
best_i = min(
    range(truth_map.shape[1] - w + 1),
    key=lambda i: np.nanmean((truth_map[:, i : i + w] - bal_map) ** 2),
)
print(f"Optimal truth_map slice: [{best_i}:{best_i+w}]")
truth_map = truth_map[:, best_i : best_i + w]

# Extents & layout parameters
dy_km = karin_NA.dy_km
dx_km = karin_NA.dx_km
extent = [0, (ny * dy_km), 0, 119.5]
yticks = np.arange(0, 120 + 1, 40)
cb_kwargs = dict(fraction=0.18, pad=0.01, shrink=0.85)
fsize = 7

# -------------------
# Compute Individual Fields
# -------------------
nan_mask = np.isnan(obs_map)
obs_filled = fill_nans_rowwise(obs_map)

# 1. Noiseless simulation: truth_map
# 2. Synthetic data (Simulation + Noise): obs_filled
# 3. Balanced extraction: bal_map
# 4. Noise field: n = obs_filled - truth_map
synth_noise = obs_filled - truth_map
# 5. Extraction + Noise: bal_map + synth_noise
bal_plus_noise = bal_map + synth_noise


# -------------------
# Compute Gradient Magnitude |\nabla h| [m / km]
# -------------------
def compute_grad_mag(h_2d, dy_m, dx_m):
    dh_dy, dh_dx = np.gradient(h_2d, dy_m, dx_m)
    return np.sqrt(dh_dx**2 + dh_dy**2) * 1e3  # Convert m/m to m/km


grad_sim = compute_grad_mag(truth_map, dy_m, dx_m)
grad_obs = compute_grad_mag(obs_filled, dy_m, dx_m)
grad_bal = compute_grad_mag(bal_map, dy_m, dx_m)
grad_noise = compute_grad_mag(synth_noise, dy_m, dx_m)
grad_bal_noise = compute_grad_mag(bal_plus_noise, dy_m, dx_m)

# Apply missing data masks (nadir gap)
grad_sim[nan_mask] = np.nan
grad_obs[nan_mask] = np.nan
grad_noise[nan_mask] = np.nan
grad_bal_noise[nan_mask] = np.nan

# -------------------
# Plot 5-Panel Figure
# -------------------
cmap_speed = "YlGnBu_r"
vmin, vmax = 0.0, 0.025  # m/km

fig, axs = plt.subplots(5, 1, figsize=(7, 7.5), sharex=True, dpi=300)

panels = [
    (
        grad_sim,
        r"Noiseless Simulation $|\nabla h_{\mathrm{sim}}|$",
    ),
    (
        grad_obs,
        r"Synthetic Data (Simulation + Noise) $|\nabla h_{\mathrm{obs}}|$",
    ),
    (
        grad_bal,
        r"Balanced Extraction $|\nabla h_{\mathrm{bal}}|$",
    ),
    (
        grad_noise,
        r"Synthetic Noise $|\nabla n|$",
    ),
    (
        grad_bal_noise,
        r"Extracted + Synthetic Noise $|\nabla (h_{\mathrm{bal}} + n)|$",
    ),
]

labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

for idx, (grad_field, title) in enumerate(panels):
    ax = axs[idx]
    im = ax.imshow(
        grad_field.T,
        origin="upper",
        cmap=cmap_speed,
        aspect="equal",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=fsize)

    cbar = fig.colorbar(im, ax=ax, **cb_kwargs)
    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label(r"$|\nabla h|$ [m km$^{-1}$]", size=fsize)

    ax.text(
        0.001,
        1.07,
        labels[idx],
        transform=ax.transAxes,
        fontsize=fsize,
        va="bottom",
        ha="left",
        bbox=dict(
            facecolor="white", alpha=0.6, edgecolor="none", pad=1.5
        ),
    )
    ax.set_ylabel("Across track [km]", fontsize=fsize)
    ax.tick_params(axis="both", labelsize=fsize)
    ax.set_yticks(yticks)

axs[-1].set_xlabel("Along track [km]", fontsize=fsize)

plt.tight_layout()
plt.savefig("grad_extraction_five_panel.pdf", bbox_inches="tight")
print("Saved grad_extraction_five_panel.pdf")
plt.show()
