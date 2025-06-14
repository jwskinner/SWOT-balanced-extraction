import matplotlib.pyplot as plt
import os
import numpy as np
import cmocean

def set_plot_style(font='Fira Sans'):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial', 'Liberation Sans', 'Verdana']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['text.usetex'] = False

    # Figure size and DPI
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams['figure.dpi'] = 150

    # Font sizes
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 12
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
    
    vmin = -0.4 
    vmax = 0.4 

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
        vmin = -1, 
        vmax = 1,
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