#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

# -------------------
# Config
# -------------------
PICKLES = "./pickles"
KARIN_NA_PATH  = f"{PICKLES}/karin_NA.pkl"   # synthetic NA (model truth + synthetic noisy KaRIn)
NADIR_NA_PATH  = f"{PICKLES}/nadir_NA.pkl"   # synthetic NA nadir
KARIN_PATH     = f"{PICKLES}/karin.pkl"      # original SWOT KaRIn
NADIR_PATH     = f"{PICKLES}/nadir.pkl"      # original SWOT nadir
INDEX          = 40                           # time index (0..len-1)
C0, C1         = 5, 65                        # crop for truth columns (60 wide)
cmap           = cmocean.cm.balance         # diverging for SSH anomalies

# -------------------
# Load
# -------------------
with open(KARIN_NA_PATH, "rb") as f:  karin_NA = pickle.load(f)
with open(NADIR_NA_PATH, "rb") as f:  nadir_NA = pickle.load(f)
with open(KARIN_PATH, "rb") as f:     karin    = pickle.load(f)
with open(NADIR_PATH, "rb") as f:     nadir    = pickle.load(f)

# -------------------
# make arrays
# -------------------
# Truth (model simulation) – use synthetic NA full-SSHA at the same time

truth_full = np.asarray(karin_NA.ssha_full[INDEX], dtype=float)  # meters
truth_crop = truth_full[:, C0:C1]
lon_truth = np.asarray(karin.lon_full, dtype=float)
lat_truth = np.asarray(karin.lat_full, dtype=float)
lon_truth = lon_truth[:, C0:C1]
lat_truth = lat_truth[:, C0:C1]

ssh_noisy_karin = np.asarray(karin_NA.ssh_noisy[INDEX], dtype=float)  # meters
ssh_noisy_nadir = np.asarray(getattr(nadir_NA, "ssh_noisy")[INDEX], dtype=float)

# Original SWOT fields (KaRIn + nadir)
ssha_karin = np.asarray(karin.ssha[INDEX], dtype=float)  # meters
ssh_nadir  = None
for key in ("ssh", "ssha"):
    if hasattr(nadir, key):
        ssh_nadir = np.asarray(getattr(nadir, key)[INDEX], dtype=float)
        break
if ssh_nadir is None:
    raise AttributeError("Could not find a nadir SSH/SSHA field on nadir (tried ssh, ssha).")

# Coordinates for KaRIn/Nadir scatter
lon_karin = np.asarray(karin.lon[INDEX], dtype=float).flatten()
lat_karin = np.asarray(karin.lat[INDEX], dtype=float).flatten()
lon_nadir = np.asarray(nadir.lon[INDEX], dtype=float)
lat_nadir = np.asarray(nadir.lat[INDEX], dtype=float)

# Synthetic NA coords (for the middle panel’s KaRIn + nadir)
lon_karin_na = np.asarray(karin_NA.lon[INDEX], dtype=float).flatten()
lat_karin_na = np.asarray(karin_NA.lat[INDEX], dtype=float).flatten()
lon_nadir_na = np.asarray(nadir_NA.lon[INDEX], dtype=float)
lat_nadir_na = np.asarray(nadir_NA.lat[INDEX], dtype=float)

# -------------------
# Anomalies & color limits
# -------------------

truth_anom = truth_crop - np.nanmean(truth_crop)
karin_na_anom = (ssh_noisy_karin - np.nanmean(ssh_noisy_karin)).flatten()
nadir_na_anom = (ssh_noisy_nadir - np.nanmean(ssh_noisy_nadir))

# Decide global color range from all three panels robustly (2–98th percentile)
vals_for_scale = np.concatenate([
    truth_anom.flatten(),
    karin_na_anom,
    nadir_na_anom.flatten(),
    (ssha_karin - np.nanmean(ssha_karin)).flatten(),
    (ssh_nadir  - np.nanmean(ssh_nadir)).flatten()
])
vmin, vmax = -0.6, 0.6 #np.nanpercentile(vals_for_scale, [2, 98])

# -------------------
# Plot
# -------------------
fig1 = plt.figure(figsize=(8, 5), dpi=200)

# 1) left: model simulation (truth)
ax0 = fig1.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
sc0 = ax0.scatter(
    lon_truth, lat_truth, c=truth_anom,
    s=1, marker='o', vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), rasterized=True
)
ax0.coastlines()
ax0.add_feature(cfeature.LAND, facecolor='#e6e8ea')
ax0.set_title("NA Simulation")
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl0.top_labels = gl0.right_labels = False
gl0.xlabel_style = {'size': 9}
gl0.ylabel_style = {'size': 9}

# 2) middle: synthetic noisy fields (KaRIn + nadir)
ax1 = fig1.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
sc1 = ax1.scatter(
    lon_karin_na, lat_karin_na,
    c=karin_na_anom, s=3,
    vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax1.scatter(
    lon_nadir_na, lat_nadir_na,
    c=nadir_na_anom, s=0.5,
    vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax1.coastlines()
ax1.add_feature(cfeature.LAND, facecolor='#e6e8ea')
ax1.set_title("Synthetic Data")
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl1.top_labels = gl1.right_labels = False
gl1.xlabel_style = {'size': 9}
gl1.ylabel_style = {'size': 9}

# 3) right: original SWOT fields (KaRIn + nadir)
ax2 = fig1.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
sc2 = ax2.scatter(
    lon_karin, lat_karin,
    c=(ssha_karin - np.nanmean(ssha_karin)).flatten(),
    s=3, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax2.scatter(
    lon_nadir, lat_nadir,
    c=(ssh_nadir - np.nanmean(ssh_nadir)),
    s=0.5, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o', rasterized=True
)
ax2.coastlines()
ax2.add_feature(cfeature.LAND, facecolor='#e6e8ea')
ax2.set_title("SWOT Data")
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl2.top_labels = gl2.right_labels = False
gl2.xlabel_style = {'size': 9}
gl2.ylabel_style = {'size': 9}

# Shared colorbar
cbar_ax = fig1.add_axes([0.92, 0.25, 0.015, 0.3])
cbar = fig1.colorbar(sc2, cax=cbar_ax, orientation='vertical')
cbar.set_label("SSHA (m)")

# Title: pass/cycle/date if available
pass_str = f"Pass {int(getattr(karin, 'pass_number', -1)):03d}" if hasattr(karin, 'pass_number') else "Pass N/A"
cycle_str = f"Cycle {int(karin_NA.t_coord[INDEX]):03d}" if hasattr(karin_NA, 't_coord') else "Cycle N/A"
date_str  = ""
if hasattr(karin_NA, "date_list"):
    try:
        date_str = f" — {karin_NA.date_list[INDEX].isoformat()}"
    except Exception:
        pass

ax0.set_rasterized(True)
ax0.set_rasterization_zorder(0)

# after ax1 = fig1.add_subplot(...)
ax1.set_rasterized(True)
ax1.set_rasterization_zorder(0)

# after ax2 = fig1.add_subplot(...)
ax2.set_rasterized(True)
ax2.set_rasterization_zorder(0)

# after creating the colorbar (cbar = fig1.colorbar(...))
cbar.ax.set_rasterized(True)

plt.suptitle(f"{pass_str}  {cycle_str}{date_str}", fontsize=11)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
fig1.savefig('synthetic_swot_fields.pdf', bbox_inches='tight', dpi=300)
print("Saved synthetic_swot_fields.pdf")
