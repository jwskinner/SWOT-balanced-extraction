# Simple script to read in, process, and plot the SWOT KaRIn and Nadir data spectra and spatial map

import JWS_SWOT_toolbox as swot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy
import cartopy.crs as ccrs
import numpy as np
import cmocean

pass_num = 9
lat_max = 38
lat_min = 28
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/' # where our data is stored

# ───── Read and Process Data ─────
# finds overlapping cycles between the karin and nadir datasets
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num) 

sample_index = 2 # this is just some index we use to pull a preliminary file and get our track indices 
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]

karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_num)

# Read and process the karin data
swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)

# Read and process the nadir data
swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# Generate coordinates
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()

# ───── SWOT Map And Spectrum Plot ─────
index = 2
vmin, vmax = -0.25, 0.25
ylims = (1e-3, 1e5)
cmap = 'YlGnBu' #cmocean.cm.balance

swot.set_plot_style()

fig = plt.figure(figsize=(10, 5), dpi=150)
gs = GridSpec(1, 2, width_ratios=[1, 1.1], figure=fig)

pad = 0.5 # padding around the map

lon_min = np.nanmin(karin.lon[index]) - pad
lon_max = np.nanmax(karin.lon[index]) + pad
lat_min = karin.lat_min - pad
lat_max = karin.lat_max + pad

ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

sc0 = ax0.scatter(
    karin.lon[index], karin.lat[index],
    c=karin.ssha[index], s=1, vmin=vmin, vmax=vmax, cmap=cmap,
    transform=ccrs.PlateCarree(), marker='o'
)
ax0.scatter(
    nadir.lon[index], nadir.lat[index],
    c=nadir.ssh[index], s=0.1, vmin=vmin, vmax=vmax,
    cmap=cmap, transform=ccrs.PlateCarree(), marker='o'
)
#ax0.coastlines()
ax0.add_feature(cartopy.feature.LAND, facecolor='lightgrey', edgecolor='none', zorder=0)
ax0.set_title(f'Pass {pass_num:03d}')
gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.0)
gl0.top_labels = gl0.right_labels = False
cbar0 = fig.colorbar(sc0, ax=ax0, orientation='vertical', shrink=0.5, pad=0.03)
cbar0.set_label("SSHA (m)")

# ───── Power Spectrum ─────
ax1 = fig.add_subplot(gs[0, 1])

ax1.loglog(karin.wavenumbers_cpkm, karin.spec_alongtrack_av, label='KaRIn', linewidth=2.5)
ax1.loglog(nadir.wavenumbers_cpkm, nadir.spec_alongtrack_av, label='Nadir', linewidth=2.5)
#ax1.loglog(karin.wavenumbers_cpkm, karin.spec_ssh, label='SWOT KaRIn SSH', linewidth=2.0)
#ax1.loglog(karin.wavenumbers_cpkm, karin.spec_tmean, label='SWOT Time-mean', linewidth=2.0)
#ax1.loglog(karin.wavenumbers_cpkm, karin.spec_tide, label='SWOT HRET', linewidth=2.0)

# Reference slope lines (k^-5 and k^-2)
k_ref = karin.wavenumbers_cpkm  # range of wavenumbers (cpkm)
C1 = 9e-8  # vertical placement for k^-5
C2 = 1e-3  # vertical placement for k^-2
ax1.loglog(k_ref, C1 * k_ref**-4.2, 'k--', linewidth=1, label=r'$k^{-4.2}$')
ax1.loglog(k_ref, C2 * k_ref**-1.7, 'gray', linestyle='--', linewidth=1, label=r'$k^{-1.7}$')

ax1.set_xlabel("Wavenumber (cpkm)")
ax1.set_ylabel("PSD (m$^2$/cpkm)")
ax1.set_ylim(ylims)
ax1.legend()

plt.tight_layout()
fig.savefig("swot_map_spectrum.pdf", bbox_inches='tight')
plt.show()