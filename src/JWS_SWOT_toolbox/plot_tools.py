import matplotlib.pyplot as plt

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