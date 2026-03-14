# Simple script that returns the Pass numbers in a lat-lon box for SWOT L2 data and downloads them.
import earthaccess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from collections import defaultdict
import jws_swot_tools as swot

# --------------------------------------------------
# CONFIG
lat_min, lat_max, lon_min, lon_max = [30.0, 38.0, 154.0, 162.0]
# SO Region: [-59.0, -51.0, 144.0, 152.0]
# KE region 1: [30.0, 38.0, 154.0, 162.0]
# KE region 2: [22.0, 30.0, 154.0, 162.0] 

tmin = "2023-07-10 00:00:00"
tmax = "2026-03-20 23:59:59"
short_karin_name = "SWOT_L2_LR_SSH_D"
short_nadir_name = "SWOT_L2_NALT_GDR_D"

download_dir = "/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE_VD/"
download     = True       # download the passes or just plot them
min_points   = 25000      # minimum data points inside box needed to count as a valid pass
min_granules = 100        # minimum number of granules a pass must have to be included

# --------------------------------------------------
# FIND PASSES IN BOUNDING BOX
earthaccess.login()

print("Searching granules in bounding box...")
karin_results = earthaccess.search_data(
    short_name="SWOT_L2_LR_SSH_D",
    temporal=(tmin, tmax),
    bounding_box=(lon_min, lat_min, lon_max, lat_max),
)

def extract_pass_number(granule):
    name = granule["umm"]["RelatedUrls"][0]["URL"].split("/")[-1]
    return int(name.split("_")[6])

def extract_date(granule):
    tr = granule["umm"]["TemporalExtent"]["RangeDateTime"]
    return tr["BeginningDateTime"][:10]

def pass_covers_box(filepath, lat_min, lat_max, lon_min, lon_max, min_points=10000):
    """
    Return (True, n_pts) if file has enough valid points inside the box.
    This is a quick check that reads lat/lon variables from the first valid file 
    to avoid errors in the NASA metadata that causes some passes to appear though 
    a box when their lat-lon data is either missing or does not intersect the box at all. 
    """
    with nc.Dataset(filepath) as ds:
        lat = ds.variables["latitude"][:].ravel()
        lon = ds.variables["longitude"][:].ravel()
        lon[lon < 0] += 360
        inside = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        n_inside = int(np.sum(inside))
    return n_inside >= min_points, n_inside

# Group granules by pass
pass_granules = defaultdict(list)
pass_dates    = defaultdict(list)
for g in karin_results:
    p = extract_pass_number(g)
    pass_granules[p].append(g)
    pass_dates[p].append(extract_date(g))

pass_list = sorted(pass_granules.keys())
print(f"\nFound {len(pass_list)} unique passes (before granule filter):")
for p in pass_list:
    dates = sorted(pass_dates[p])
    n = len(dates)
    flag = "" if n >= min_granules else "  ← below min_granules threshold"
    print(f"  Pass {p:03d}: {dates[0]} → {dates[-1]}  ({n} granules){flag}")

# Filter passes that don't meet the minimum granule count
pass_list = [p for p in pass_list if len(pass_granules[p]) >= min_granules]
print(f"\n{len(pass_list)} passes remaining after requiring >= {min_granules} granules:")
print(f"  {pass_list}")

# --------------------------------------------------
# PLOT FIRST FRAME OF EACH PASS
tmp_dir = "./tmp"
os.makedirs(tmp_dir, exist_ok=True)
pad_deg = 20  # pad the map on each side to give context around the box

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([
    max(lon_min - pad_deg, -180), min(lon_max + pad_deg, 180),
    max(lat_min - pad_deg,  -90), min(lat_max + pad_deg,  90),
], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
colors = plt.cm.tab20(np.linspace(0, 1, len(pass_list)))

valid_passes = []

for color, pass_num in zip(colors, pass_list):
    sorted_granules = sorted(pass_granules[pass_num], key=extract_date)

    # Try granules in order until one downloads successfully
    files = None
    for attempt, granule in enumerate(sorted_granules):
        try:
            files = earthaccess.download([granule], local_path=tmp_dir)
            if files:
                break
        except Exception as e:
            print(f"  Pass {pass_num:03d}: granule {attempt} failed ({e}), trying next...")

    if not files:
        print(f"  Pass {pass_num:03d}: all granules failed, skipping")
        continue

    try:
        covers, n_pts = pass_covers_box(files[0], lat_min, lat_max, lon_min, lon_max, min_points)
        if not covers:
            print(f"  Pass {pass_num:03d}: only {n_pts} pts in box — skipping")
            continue

        with nc.Dataset(files[0]) as ds:
            lat = ds.variables["latitude"][:].ravel()
            lon = ds.variables["longitude"][:].ravel()
            lon[lon < 0] += 360

        ax.scatter(lon, lat, s=0.3, color=color, label=f"Pass {pass_num:03d} ({n_pts} pts)", transform=ccrs.PlateCarree())
        valid_passes.append(pass_num)
        print(f"  Pass {pass_num:03d}: plotted ({n_pts} pts in box)")

    except Exception as e:
        print(f"  Pass {pass_num:03d}: error reading file — {e}")

# Overlay bounding box
box = mpatches.Rectangle(
    (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
    linewidth=2, edgecolor="black", facecolor="none",
    linestyle="-", label="Search box", zorder=5,
    transform=ccrs.PlateCarree(),
)
ax.add_patch(box)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Returned Passes")
ax.legend(loc="upper right", markerscale=8, fontsize=7)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("pass_preview.png", dpi=150)
plt.close()

print(f"\nValid passes after screening: {valid_passes}")
print("Saved pass_preview.png")

# --------------------------------------------------
# DOWNLOAD (uses valid_passes from screening above)
def download_batch(granules, download_dir, verbose=True):
    os.makedirs(download_dir, exist_ok=True)
    downloaded = earthaccess.download(granules, local_path=download_dir, threads=4)
    if verbose:
        print(f"Downloaded {len(downloaded)} files to {download_dir}")
    return downloaded

if download:
    for pass_num in valid_passes:
        print(f"\nDownloading pass {pass_num}...")

        # Search KaRIn and nadir independently so nadir availability
        # does not truncate the KaRIn download via shared_cycles filtering as in the return_cloud_files function. 
        karin_pass = earthaccess.search_data(
            short_name=short_karin_name,
            temporal=(tmin, tmax),
            granule_name=f"*_LR_SSH_Basic_*_{pass_num:03d}_*",
        )
        nadir_pass = earthaccess.search_data(
            short_name=short_nadir_name,
            temporal=(tmin, tmax),
            granule_name=f"*_{pass_num:03d}_*",
        )

        print(f"  KaRIn: {len(karin_pass)} granules | Nadir: {len(nadir_pass)} granules")
        download_batch(karin_pass, download_dir)
        download_batch(nadir_pass, download_dir)
        print(f"Completed: {pass_num}")

print("Download Complete")
print("\nValid Passes:")
print(f"PASSES=({' '.join(str(p) for p in valid_passes)})")
