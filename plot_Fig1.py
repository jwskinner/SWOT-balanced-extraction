import JWS_SWOT_toolbox as swot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read in the SWOT data for this pass
pass_num = 9
lat_max = 35 #38 #38
lat_min = 28 #28 #28

data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/' # where our data is stored

# finds overlapping cycles between the karin and nadir datasets
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num) 

sample_index = 2 # this is just some index we use to pull a preliminary file and get our track indices 
indx, track_length = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
indxs, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)
dims_SWOT = [len(shared_cycles), track_length, track_length_nadir]

karin, nadir = swot.init_swot_arrays(dims_SWOT, lat_min, lat_max, pass_num)

# read and process the karin data
swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
swot.process_karin_data(karin)

# read and process the nadir data
swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
swot.process_nadir_data(nadir)

# generate the coordinates
karin.coordinates()
nadir.coordinates()

# Compute spectra
karin.compute_spectra()
nadir.compute_spectra()

# KaRIn model fit
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model)

# Nadir model fit
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)

# Make the figure 1 plot 
# optional style (if available)
try:
    swot.set_plot_style()
except Exception:
    pass

# index for the pass to plot
index = 10
print(karin.time_dt[index])

print(shared_cycles[index])

# one-sided wavenumbers & spectra (take positive half)
tlen_k = int(karin.track_length // 2)
tlen_n = int(nadir.track_length // 2)

k_karin = np.asarray(karin.wavenumbers_cpkm[tlen_k:])
karin_spec = np.asarray(karin.spec_alongtrack_av[tlen_k:])

k_nadir = np.asarray(nadir.wavenumbers_cpkm[tlen_n:])
nadir_spec = np.asarray(nadir.spec_alongtrack_av[tlen_n:])

# model components
sp_balanced_karin = swot.balanced_model_tapered(k_karin, *poptcwg_karin[0:3])
sp_unbalanced_karin = swot.unbalanced_model_tapered(k_karin, *poptcwg_karin[3:6])
model_sum_karin = sp_balanced_karin + sp_unbalanced_karin

sp_noise_nadir = swot.nadir_noise_model(k_nadir, poptcwg_nadir[0])
sp_balanced_nadir = swot.balanced_model(k_nadir, *poptcwg_karin[0:3])
model_sum_nadir = sp_balanced_nadir + sp_noise_nadir

# --- shared x-axis limits ---
kmin = 1e-3
kmax = 3e-1

# figure layout: left large map, right column with 3 stacked spectra (sharing x)
fig = plt.figure(figsize=(12, 12), dpi=150)
gs = GridSpec(nrows=3, ncols=2, figure=fig,
              width_ratios=[2.0, 1.2], height_ratios=[1, 1, 1],
              hspace=0.3, wspace=0.25)

fig.subplots_adjust(left=0.01)  # push leftmost column all the way left

# ----- Large map (left column spanning all rows) -----
ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
pad = 0.5
lon_min = np.nanmin(karin.lon[index]) - pad
lon_max = np.nanmax(karin.lon[index]) + pad
lat_min = np.nanmin(karin.lat[index]) - pad
lat_max = np.nanmax(karin.lat[index]) + pad
ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

sc = ax_map.scatter(karin.lon[index], karin.lat[index],
                    c=karin.ssha[index], s=1.0, cmap='cmo.balance',
                    vmin=-0.25, vmax=0.25, transform=ccrs.PlateCarree(),
                    marker='o', rasterized=True, zorder=2)
ax_map.scatter(nadir.lon[index], nadir.lat[index],
               c=nadir.ssh[index], s=1.5, cmap='cmo.balance',
               vmin=-0.25, vmax=0.25, transform=ccrs.PlateCarree(),
               marker='o', rasterized=True, zorder=1)
ax_map.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)
ax_map.coastlines(resolution='50m', linewidth=0.5, zorder=3)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.8)
gl.top_labels = False
gl.right_labels = False
ax_map.set_title(f'Pass {pass_num:03d}', fontsize=15)
cbar = fig.colorbar(sc, ax=ax_map, orientation='vertical', shrink=0.3, pad=0.05)
cbar.set_label('SSHA (m)')

# increase font size for latitude/longitude labels
gl.xlabel_style = {'size': 13}  # longitude labels
gl.ylabel_style = {'size': 13}  # latitude labels

# --- Y-limits for PSD ---
ylims = (1e-3, 1e5)

# create three stacked spectra sharing x-axis
ax_top  = fig.add_subplot(gs[0, 1])
ax_mid  = fig.add_subplot(gs[1, 1], sharex=ax_top)
ax_bot  = fig.add_subplot(gs[2, 1], sharex=ax_top)

# ----- Top-right: observed spectra -----
k_ref = karin.wavenumbers_cpkm
C1 = 2.5e-7  # vertical placement for balanced
C2 = 1.5e-2  # vertical placement for noise
ax_top.loglog(k_karin, karin_spec, label='KaRIn', linewidth=3.0)
ax_top.loglog(k_nadir, nadir_spec, label='Nadir', linewidth=3.0)
ax_top.loglog(k_ref, C1 * k_ref**-4.7, 'k--', linewidth=1, label=r'$k^{-4.7}$')
ax_top.loglog(k_ref, C2 * k_ref**-1.7, 'k', linestyle='-.', linewidth=1, label=r'$k^{-1.7}$')
ax_top.set_ylabel('PSD (cm$^2$/cpkm)')
ax_top.set_ylim(ylims)
ax_top.set_xlim(kmin, kmax)
ax_top.set_title('Observed spectra')
ax_top.legend(fontsize=9, loc='lower left')
plt.setp(ax_top.get_xticklabels(), visible=False)

# ----- Middle-right: KaRIn data + fits -----
ax_mid.loglog(k_karin, karin_spec, 'o',markersize = 6, label='KaRIn')
ax_mid.loglog(k_karin, sp_balanced_karin, '-', color='tab:green', label='Balanced model')
ax_mid.loglog(k_karin, sp_unbalanced_karin, '-', color='tab:orange', label='Noise model')
ax_mid.loglog(k_karin, model_sum_karin, '--k', label='Model sum')
ax_mid.set_ylabel('PSD (cm$^2$/cpkm)')
ax_mid.set_ylim(ylims)
ax_mid.legend(fontsize=9, loc='lower left')
ax_mid.set_title('KaRIn model')
plt.setp(ax_mid.get_xticklabels(), visible=False)

# ----- Bottom-right: Nadir data + fits -----
ax_bot.loglog(k_nadir, nadir_spec, 'o',markersize = 6, color='tab:red', label='Nadir')
ax_bot.loglog(k_nadir, sp_balanced_nadir, '-',  color='tab:green', label='Balanced (KaRIn)')
ax_bot.loglog(k_nadir, sp_noise_nadir, '-', color='tab:orange', label='Noise floor (N)')
ax_bot.loglog(k_nadir, model_sum_nadir, '--k', label='Model sum')
ax_bot.set_xlabel('Wavenumber (cpkm)')
ax_bot.set_ylabel('PSD (cm$^2$/cpkm)')
ax_bot.set_ylim(ylims)
ax_bot.legend(fontsize=9, loc='lower left')
ax_bot.set_title('Nadir model')

# ----- Add labels ---- 
fig.text(0.045, 0.905, "(a)", fontsize=15, va='top', ha='left')
fig.text(0.6, 0.905, "(b)", fontsize=15,  va='top', ha='left')
fig.text(0.6, 0.627, "(c)", fontsize=15,  va='top', ha='left')
fig.text(0.6, 0.3475, "(d)", fontsize=15,  va='top', ha='left')


# add a grey line for the transition scale 

# ---- Find where balanced and noise models are equal ----
# KaRIn crossover (balanced vs unbalanced/aliased)
ratio_karin = sp_balanced_karin / sp_unbalanced_karin
cross_idx_karin = np.argmin(np.abs(np.log10(ratio_karin)))
k_cross_karin = k_karin[cross_idx_karin]
lambda_cross_karin = 1 / k_cross_karin
print(f"KaRIn crossover: k = {k_cross_karin:.4f} cpkm  (λ ≈ {lambda_cross_karin:.1f} km)")

# Nadir crossover (balanced vs noise floor)
ratio_nadir = sp_balanced_nadir / sp_noise_nadir
cross_idx_nadir = np.argmin(np.abs(np.log10(ratio_nadir)))
k_cross_nadir = k_nadir[cross_idx_nadir]
lambda_cross_nadir = 1 / k_cross_nadir
print(f"Nadir crossover: k = {k_cross_nadir:.4f} cpkm  (λ ≈ {lambda_cross_nadir:.1f} km)")

# ---- Add vertical grey lines ----

for ax in [ax_top, ax_mid]:
    # KaRIn crossover
    ax.axvline(k_cross_karin, color='grey', linestyle='-', linewidth=1.0, alpha=1.0)
    ax.text(k_cross_karin * 1.1, 3e-3,
            f'{lambda_cross_karin:.1f} km',
            rotation=90, color='grey', fontsize=9, va='bottom', ha='left')
for ax in [ax_bot]:
    # Nadir crossover
    ax.axvline(k_cross_nadir, color='grey', linestyle='-', linewidth=1.0, alpha=1.0)
    ax.text(k_cross_nadir * 1.1, 3e-3,
            f'{lambda_cross_nadir:.1f} km',
            rotation=90, color='grey', fontsize=9, va='bottom', ha='left')

plt.tight_layout()
fig.savefig("fig1.pdf", bbox_inches='tight')
plt.show()
