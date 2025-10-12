import matplotlib.pyplot as plt
import os
import numpy as np
import cmocean
import JWS_SWOT_toolbox as swot

def set_plot_style(font='Fira Sans'):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial', 'Liberation Sans', 'Verdana']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['text.usetex'] = False

    # Figure size and DPI
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams['figure.dpi'] = 150

    # Font sizes
    plt.rcParams['axes.titlesize'] =  12
    plt.rcParams['axes.labelsize'] =  12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # Axes, lines, and markers
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 7

    # Grid style
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5

    # Color cycle (optional)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    )

    # Tick parameters
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 3

    # Legend
    plt.rcParams['legend.frameon'] = False

    # Savefig
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 200

def plot_ssh_summary(ht, nyt, nxt,
                     ykk, xkk, mask_k,
                     ynn, xnn, mask_n,
                     karin, nadir,
                     index,
                     shared_cycles,
                     pass_number,
                     swot,
                     plots_dir,
                     basename):

    # ── reshape the map and pick colour limits 
    ht_map = ht.reshape(nyt, nxt)
    #vmin, vmax = np.nanpercentile(ht_map, [2, 98])
    
    vmin = -0.25 
    vmax = 0.25 

    # ── figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True,
                            gridspec_kw={"hspace": 0.4})

    sc = axes[0].scatter(
        ykk * 1e-3, xkk * 1e-3, c = karin.ssha[index, :, :][mask_k].flatten(), s=5, cmap=cmocean.cm.balance,
        vmin=vmin, vmax=vmax, edgecolor="none"
    )
    sc = axes[0].scatter(
        ynn * 1e-3, xnn * 1e-3, c = nadir.ssh[index, :][mask_n], s=5, cmap=cmocean.cm.balance,
        vmin=vmin, vmax=vmax, edgecolor="none"
    )
    axes[0].set_title("Observed SSH")
    axes[0].set_title("Cycle: {}".format(shared_cycles[index]), fontsize=11, loc='right')
    axes[0].set_title("Pass: {:03d}".format(pass_number), fontsize=11, loc='left')
    #axes[0].set_xlabel("along-track (km)")
    axes[0].set_ylabel("across-track (km)")
    axes[0].set_aspect("auto")
    fig.colorbar(sc, ax=axes[0], orientation='vertical', shrink=0.7, pad=0.02)

    lats = np.linspace(np.nanmin(karin.lat[index, :, :]), np.nanmax(karin.lat[index, :, :]), nyt)
    geo_vort = swot.compute_geostrophic_vorticity_fd(np.ma.masked_invalid(ht_map), 2000, 2000, lats)

    # 1. gridded estimate (imshow)
    im0 = axes[1].imshow(
        ht_map,
        origin="lower",
        extent=(0, nxt * karin.dx * 1e-3,        # km
                0, nyt * karin.dy * 1e-3),       # km
        aspect="auto",
        cmap=cmocean.cm.balance,
        vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Balanced SSH")
    axes[1].set_ylabel("across-track (km)")
    fig.colorbar(im0, ax=axes[1], orientation='vertical', shrink=0.7, pad=0.02)

    im1 = axes[2].imshow(
        swot.compute_gradient(ht_map),
        origin="lower",
        extent=(0, nxt * karin.dx * 1e-3,        # km
                0, nyt * karin.dy * 1e-3),       # km
        aspect="auto",
        cmap=cmocean.cm.deep_r
    )

    axes[2].set_title("Gradient")
    axes[2].set_ylabel("across-track (km)")
    fig.colorbar(im1, ax=axes[2], orientation='vertical', shrink=0.7, pad=0.02, label=r'$\nabla(SSH)$')

    im1 = axes[3].imshow(
        geo_vort,
        origin="lower",
        extent=(0, nxt * karin.dx * 1e-3,        # km
                0, nyt * karin.dy * 1e-3),       # km
        aspect="auto",
        vmin = -0.5, 
        vmax = 0.5,
        cmap=cmocean.cm.balance
    )
    
    axes[3].set_title("Geostrophic Vorticity")
    axes[3].set_xlabel("along-track distance (km)")
    axes[3].set_ylabel("across-track (km)")
    fig.colorbar(im1, ax=axes[3], orientation='vertical', shrink=0.7, pad=0.02, label=r'$\zeta / f$')

    # shared colour-bar
    plt.tight_layout()
    plt.show()


    # Layout adjustment
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"{basename}.png")

    # Save and close
    fig.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {plot_filename}")

def plot_spectral_fits(karin, nadir, poptcwg_karin, poptcwg_nadir, 
                               output_filename='swot_karin_nadir_fit.pdf', 
                               figsize=(8, 4), dpi=120):
    
    # Get the one-sided spectra
    k_karin = karin.wavenumbers_cpkm[int(karin.track_length/2):]  # units [1/m]
    karin_spec_sample_mean = karin.spec_alongtrack_av[int(karin.track_length/2):]
    k_nadir = nadir.wavenumbers_cpkm[int(nadir.track_length/2):]  # units [1/m]
    nadir_spec_sample_mean = nadir.spec_alongtrack_av[int(nadir.track_length/2):]
    
    # Put the wavenumbers through the models to get the functional form
    spbalanced = swot.balanced_model(k_karin[1:], *poptcwg_karin[0:3])
    spunbalanced = swot.unbalanced_model_aliased(k_karin, *poptcwg_karin[3:7])
    spnoise_nadir = swot.nadir_noise_model(k_nadir, poptcwg_nadir[0])
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    
    # ----- Panel 1: KaRIn -----
    k_km = k_karin # wavenumbers in cycles/km
    axs[0].loglog(k_km[1:], karin_spec_sample_mean[1:], 'o', label='KaRIn SSHA')
    axs[0].loglog(k_km[1:], spunbalanced[1:], 
                  label=r'$A_n$=%5.1f, $\lambda_n$=%5.1f, $S_n$=%5.1f' % 
                  (poptcwg_karin[3], 100, poptcwg_karin[5]))
    axs[0].loglog(k_km[1:], spbalanced, 
                  label=r'$A_b$=%5.1f, $\lambda_b$=%5.1f, $S_b$=%5.1f' % 
                  (poptcwg_karin[0], poptcwg_karin[1], poptcwg_karin[2]))
    axs[0].loglog(k_km[1:], (spunbalanced[1:] + spbalanced), '--', label='Model (sum)')
    axs[0].set_xlabel('wavenumber (cpkm)')
    axs[0].set_ylabel('PSD (cm$^2$ / cpkm)')
    # axs[0].set_xlim(1e-3, 3e-1)
    axs[0].set_ylim(1e-3, 1e6)
    axs[0].set_title('KaRIn')
    axs[0].legend(loc='lower left', frameon=False, fontsize=9)
    
    # ----- Panel 2: Nadir -----
    axs[1].loglog(k_nadir[1:], nadir_spec_sample_mean[1:], 'o', label='Nadir SSHA')
    axs[1].loglog(k_nadir, spnoise_nadir, label=r'$N$=%5.1f' % (poptcwg_nadir[0]))
    axs[1].loglog(k_karin[1:], spbalanced, '-', label=r'KaRIn balanced model')
    axs[1].loglog(k_karin[1:], (spbalanced + spnoise_nadir[:1]), '--', label='Model (sum)')
    axs[1].set_xlabel('wavenumber (cpkm)')
    axs[1].set_ylabel('PSD (cm$^2$ / cpkm)')
    # axs[1].set_xlim(1e-3, 3e-1)
    axs[1].set_ylim(1e-3, 1e6)
    axs[1].set_title('Nadir')
    axs[1].legend(loc='lower left', frameon=False, fontsize=9)
    
    # Save figure
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    
    return fig, axs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

# Plot the NA simulation vs SWOT maps and spectra
def plot_swot_sim_maps_and_spectra(
    *,
    karin_NA, nadir_NA,
    NA_karin_lon, NA_karin_lat,
    NA_nadir_lon, NA_nadir_lat,
    karin, nadir,
    index: int = 2,
    vmin: float = -0.5, vmax: float = 0.5,
    ylims=(1e-5, 1e5),
    cmap=None,
    plot_style_fn=None,           # e.g., swot.set_plot_style
    figsize=(18, 6), dpi=150,
    show: bool = True,
    savepath: str | None = None
):
    """
    Plot (1) simulation KaRIn+Nadir map, (2) SWOT KaRIn+Nadir map, and (3) power spectra.

    Parameters
    ----------
    karin_NA, nadir_NA : objects with fields
        Simulation objects providing:
          - ssha[time, ...]
          - wavenumbers_cpkm (1D), spec_alongtrack_av (1D)
    NA_karin_lon/lat : 2D arrays
        Simulation KaRIn lon/lat grids.
    NA_nadir_lon/lat : 1D arrays
        Simulation Nadir lon/lat tracks.
    karin, nadir : objects with fields
        SWOT objects providing:
          - lat[time,...], lon[time,...], ssha[time,...] (KaRIn)
          - lat[time,...], lon[time,...], ssh[time,...]  (Nadir)
          - wavenumbers_cpkm (1D), spec_alongtrack_av (1D)
          - karin also: spec_ssh, spec_tmean, spec_tide
    index : int
        Time index to plot.
    vmin, vmax : float
        Color scale for maps.
    ylims : tuple
        Y-limits for log–log PSD plot.
    cmap : matplotlib colormap or None
        Defaults to cmocean.balance or 'RdBu_r' if not available.
    plot_style_fn : callable or None
        If provided, called at start (e.g., swot.set_plot_style).
    figsize, dpi : figure sizing
    show : bool
        If True, plt.show() at end.
    savepath : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    fig, axes : (matplotlib.figure.Figure, dict)
        axes = {"sim": ax0, "swot": ax1, "spec": ax2}
    """
    # Optional style hook
    if plot_style_fn is not None:
        try:
            plot_style_fn()
        except Exception:
            pass

    # Colormap default
    if cmap is None:
        try:
            import cmocean
            cmap = cmocean.cm.balance
        except Exception:
            cmap = "RdBu_r"

    # Figure & layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.6], figure=fig)

    # ───── (0) Helpers ─────
    def _finite_mask(*arrs):
        m = np.ones_like(np.asarray(arrs[0]).ravel(), dtype=bool)
        for a in arrs:
            a_flat = np.asarray(a).ravel()
            m &= np.isfinite(a_flat)
        return m

    # ───── (1) Simulation Map ─────
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    sim_k_lon = np.asarray(NA_karin_lon).ravel()
    sim_k_lat = np.asarray(NA_karin_lat).ravel()
    sim_k_val = np.asarray(karin_NA.ssha[index]).ravel()
    m0 = _finite_mask(sim_k_lon, sim_k_lat, sim_k_val)
    sc0 = ax0.scatter(
        sim_k_lon[m0], sim_k_lat[m0], c=sim_k_val[m0],
        s=3, vmin=vmin, vmax=vmax, cmap=cmap,
        transform=ccrs.PlateCarree(), marker='o', rasterized=True
    )

    sim_n_lon = np.asarray(NA_nadir_lon).ravel()
    sim_n_lat = np.asarray(NA_nadir_lat).ravel()
    sim_n_val = np.asarray(nadir_NA.ssha[index]).ravel()
    m1 = _finite_mask(sim_n_lon, sim_n_lat, sim_n_val)
    ax0.scatter(
        sim_n_lon[m1], sim_n_lat[m1], c=sim_n_val[m1],
        vmin=vmin, vmax=vmax, cmap=cmap, s=1, marker='o',
        transform=ccrs.PlateCarree(), rasterized=True
    )
    ax0.coastlines()
    ax0.set_title("NA Simulation KaRIn + Nadir")
    gl0 = ax0.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
    try:
        gl0.top_labels = False
        gl0.right_labels = False
    except Exception:
        pass
    cbar0 = fig.colorbar(sc0, ax=ax0, orientation='vertical', shrink=0.7, pad=0.03)
    cbar0.set_label("SSHA (m)")

    # ───── (2) SWOT Map ─────
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    k_lon = np.asarray(karin.lon[index]).ravel()
    k_lat = np.asarray(karin.lat[index]).ravel()
    k_val = np.asarray(karin.ssha[index]).ravel()
    mk = _finite_mask(k_lon, k_lat, k_val)
    sc1 = ax1.scatter(
        k_lon[mk], k_lat[mk], c=k_val[mk], s=3, vmin=vmin, vmax=vmax,
        cmap=cmap, transform=ccrs.PlateCarree(), marker='o', rasterized=True
    )

    n_lon = np.asarray(nadir.lon[index]).ravel()
    n_lat = np.asarray(nadir.lat[index]).ravel()
    n_val = np.asarray(nadir.ssh[index]).ravel()
    mn = _finite_mask(n_lon, n_lat, n_val)
    ax1.scatter(
        n_lon[mn], n_lat[mn], c=n_val[mn], s=1, vmin=vmin, vmax=vmax,
        cmap=cmap, transform=ccrs.PlateCarree(), marker='o', rasterized=True
    )
    ax1.coastlines()
    ax1.set_title("SWOT KaRIn + Nadir")
    gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.25)
    try:
        gl1.top_labels = False
        gl1.right_labels = False
    except Exception:
        pass
    cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', shrink=0.7, pad=0.03)
    cbar1.set_label("SSHA (m)")

    # ───── (3) Power Spectrum ─────
    ax2 = fig.add_subplot(gs[0, 2])

    def _loglog_safe(x, y, label, lw=2):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.any():
            ax2.loglog(x[m], y[m], label=label, linewidth=lw)

    # Simulation spectra
    _loglog_safe(karin_NA.wavenumbers_cpkm, karin_NA.spec_alongtrack_av, "Sim KaRIn SSH", 2)
    _loglog_safe(nadir_NA.wavenumbers_cpkm, nadir_NA.spec_alongtrack_av, "Sim Nadir SSH", 2)

    # SWOT spectra
    _loglog_safe(karin.wavenumbers_cpkm, karin.spec_alongtrack_av, "SWOT KaRIn SSHA", 2)
    _loglog_safe(nadir.wavenumbers_cpkm, nadir.spec_alongtrack_av, "SWOT Nadir SSHA", 2)
    _loglog_safe(karin.wavenumbers_cpkm, getattr(karin, "spec_ssh", np.nan), "SWOT KaRIn SSH", 2.0)
    _loglog_safe(karin.wavenumbers_cpkm, getattr(karin, "spec_tmean", np.nan), "SWOT Time-mean", 2.0)
    _loglog_safe(karin.wavenumbers_cpkm, getattr(karin, "spec_tide", np.nan), "SWOT HRET", 2.0)

    ax2.set_xlabel("Wavenumber (cpkm)")
    ax2.set_ylabel(r"PSD (m$^2$/cpkm)")
    ax2.set_ylim(ylims)
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax2.set_title("Power Spectra")

    # Legend outside
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()

    return fig, {"sim": ax0, "swot": ax1, "spec": ax2}

