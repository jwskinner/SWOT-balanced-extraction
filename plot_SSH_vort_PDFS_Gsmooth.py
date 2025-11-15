import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis
import JWS_SWOT_toolbox as swot
from scipy.stats.mstats import winsorize
# Removed: import scipy.linalg as la
# Removed: from scipy.ndimage import gaussian_filter

# =========================
#  --- CONFIGURATION ---
# =========================
PICKLES = "./pickles"
KARIN_NA_PATH = f"{PICKLES}/karin_NA.pkl"
SCALES = [1, 2, 4, 8, 16]  # km
FINAL_DATA_PATH = f"{PICKLES}/final_plot_data.pkl"  # somewhere to save the plot data so we dont need to compute it each time

def path_nonoise(scale_km):
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_nonoise_sm_{scale_km}km.pkl"

def path_withnoise(scale_km):
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_sm_{scale_km}km.pkl"

def path_posterior(scale_km):
    """Path to the pre-smoothed posterior SSH."""
    return f"{PICKLES}/posterior_full_synth_NA_tmean_sm_{scale_km}km.pkl"

# =========================
#  --- LOAD GRID ---
# =========================

os.makedirs(PICKLES, exist_ok=True) 

with open(KARIN_NA_PATH, "rb") as f:
    karin_NA = pickle.load(f)

dx_m = float(karin_NA.dx_km) * 1e3
dy_m = float(karin_NA.dy_km) * 1e3

lats = np.asarray(getattr(karin_NA, "lat"), dtype=float)
if lats.ndim == 3:
    lat_1d = lats[0, :, 0]
elif lats.ndim == 2:
    lat_1d = lats[:, 0]
else:
    raise ValueError("Unexpected latitude array shape for 'lat'")

# =========================
#  --- FUNCTS ---
# =========================
def finite_flat(a):
    """Flatten an array and return only its finite values."""
    if a is None:
        return np.array([]) # Handle None input
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

def stack_vort(ssh_stack):
    """Compute geostrophic vorticity for each time slice in a stack."""
    out = []
    # Get latitude dimension from the 2D slice shape
    lat_dim_size = ssh_stack.shape[1] 
    lats_sliced = lat_1d[0:lat_dim_size]
    
    for i in range(ssh_stack.shape[0]):
        eta = np.asarray(ssh_stack[i], dtype=float)
        zeta = swot.compute_geostrophic_vorticity(eta, dx_m, dy_m, lats_sliced)
        out.append(zeta)
    return np.stack(out, axis=0)

def pooled_bins(*arrays, nbins=120):
    """Create histogram bins from the pooled range of multiple arrays."""
    pool = np.concatenate([finite_flat(a) for a in arrays if a is not None and a.size > 0])
    if pool.size == 0:
        return np.linspace(-1, 1, nbins + 1) # Default fallback
    lo = np.percentile(pool, 0.25)
    hi = np.percentile(pool, 99.75)
    bound = max(abs(lo), abs(hi))
    if bound == 0:
        bound = 1.0
    return np.linspace(-bound, bound, nbins + 1)

def compute_stats_from_stack(z_stack, edges):
    """
    function to compute time-mean PDFs and time-mean moments
    in a single loop over the time dimension (nT).
    """
    nT = z_stack.shape[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdfs = np.zeros((nT, centers.size))
    
    stds, vars_, skews, kurts = [], [], [], []

    for i in range(nT):
        vals_finite = finite_flat(z_stack[i])
        
        if vals_finite.size < 2:
            pdfs[i, :] = np.nan # Mark as nan if no data
            continue

        # --- 1. PDF Calculation ---
        h, _ = np.histogram(vals_finite, bins=edges, density=True)
        pdfs[i, :] = h
        
        # --- 2. Moments Calculation ---
        # remove extreme outliers (only for unprocessed NA sim's kurtosis estimate)
        vals_win = winsorize(vals_finite, limits=[0.001, 0.001]).data 

        stds.append(np.std(vals_win))
        vars_.append(np.var(vals_win))
        skews.append(skew(vals_win))
        kurts.append(kurtosis(vals_win, fisher=True, bias=False))

    # --- Process PDF Stats ---
    pdf_mu = np.nanmean(pdfs, axis=0)
    pdf_sig = np.nanstd(pdfs, axis=0, ddof=1)
    
    # --- Process Moment Stats ---
    stds = np.array(stds); vars_ = np.array(vars_)
    skews = np.array(skews); kurts = np.array(kurts)
    
    moments_out = {
        "nT": len(stds),
        "std_mu": np.nanmean(stds),  "std_sig": np.nanstd(stds, ddof=1),
        "var_mu": np.nanmean(vars_), "var_sig": np.nanstd(vars_, ddof=1),
        "skw_mu": np.nanmean(skews), "skw_sig": np.nanstd(skews, ddof=1),
        "kur_mu": np.nanmean(kurts), "kur_sig": np.nanstd(kurts, ddof=1),
    }
    
    return (centers, pdf_mu, pdf_sig), moments_out


def safe_load(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

# =================================================
#  --- LOAD DATA OR RUN PROCESSING ---
# =================================================

if os.path.exists(FINAL_DATA_PATH):
    # --- FILE EXISTS, LOAD DATA ---
    print(f"Loading pre-computed data from {FINAL_DATA_PATH}...")
    with open(FINAL_DATA_PATH, "rb") as f:
        data_loaded = pickle.load(f)
    stats = data_loaded["stats"]
    all_pdf_data = data_loaded["all_pdf_data"]
    print("...Load complete. Skipping processing.")

else:
    # --- FILE DOESN'T EXIST, RUN PROCESSING ---
    print(f"No pre-computed data found at {FINAL_DATA_PATH}. Running processing...")
    
    # =========================
    #  --- STORAGE FOR STATS ---
    # =========================
    stats = {
        "scale": [],
        # nonoise (sim) means & time-sigmas
        "var_no_mu": [], "var_no_sig": [],
        "std_no_mu": [], "std_no_sig": [],
        "skw_no_mu": [], "skw_no_sig": [],
        "kur_no_mu": [], "kur_no_sig": [],
        # with-noise (extracted) means & time-sigmas
        "var_wi_mu": [], "var_wi_sig": [],
        "std_wi_mu": [], "std_wi_sig": [],
        "skw_wi_mu": [], "skw_wi_sig": [],
        "kur_wi_mu": [], "kur_wi_sig": [],
        # posterior (sampled)
        "var_post": [],
        "std_post": [],
        "skw_post": [],
        "kur_post": [],
        # other stuff
        "nT_no": [], "nT_wi": [], "nT_post": [], # Added nT_post
    }
    
    all_pdf_data = [] # Storage for PDF data in time

    # ======================================
    #  --- MAIN PROCESSING LOOP ---
    # ======================================
    for j, scale in enumerate(SCALES):
        print(f"\n--- Processing Scale: {scale} km ---")
        
        # --- Load nonoise and withnoise ---
        ht_nonoise = np.asarray(safe_load(path_nonoise(scale)), dtype=float)
        ht_with    = np.asarray(safe_load(path_withnoise(scale)), dtype=float)

        z_nonoise = stack_vort(ht_nonoise)
        z_with    = stack_vort(ht_with)

        # --- Load posterior ---
        path_post = path_posterior(scale)
        ht_posterior = None
        z_posterior = None
        if os.path.exists(path_post):
            try:
                ht_posterior = np.asarray(safe_load(path_post), dtype=float)
                z_posterior = stack_vort(ht_posterior)
                print(f"Loaded and processed posterior from {path_post}")
            except Exception as e:
                print(f"Could not load or process posterior {path_post}: {e}")
        else:
            print(f"Posterior file not found, skipping: {path_post}")


        # ======= Compute PDFs and Moments in one go =======
        # Add z_posterior to the pool (it's fine if it's None)
        edges = pooled_bins(z_nonoise, z_with, z_posterior, nbins=120) 
        
        print("Calculating stats for 'nonoise'...")
        (x, m_no, s_no), tm_no = compute_stats_from_stack(z_nonoise, edges)
        
        print("Calculating stats for 'withnoise'...")
        (_, m_wi, s_wi), tm_wi = compute_stats_from_stack(z_with, edges)

        # --- Store nonoise and withnoise stats ---
        stats["scale"].append(scale)
        stats["nT_no"].append(tm_no["nT"])
        stats["nT_wi"].append(tm_wi["nT"])

        stats["var_no_mu"].append(tm_no["var_mu"]); stats["var_no_sig"].append(tm_no["var_sig"])
        stats["std_no_mu"].append(tm_no["std_mu"]); stats["std_no_sig"].append(tm_no["std_sig"])
        stats["skw_no_mu"].append(tm_no["skw_mu"]); stats["skw_no_sig"].append(tm_no["skw_sig"])
        stats["kur_no_mu"].append(tm_no["kur_mu"]); stats["kur_no_sig"].append(tm_no["kur_sig"])

        stats["var_wi_mu"].append(tm_wi["var_mu"]); stats["var_wi_sig"].append(tm_wi["var_sig"])
        stats["std_wi_mu"].append(tm_wi["std_mu"]); stats["std_wi_sig"].append(tm_wi["std_sig"])
        stats["skw_wi_mu"].append(tm_wi["skw_mu"]); stats["skw_wi_sig"].append(tm_wi["skw_sig"])
        stats["kur_wi_mu"].append(tm_wi["kur_mu"]); stats["kur_wi_sig"].append(tm_wi["kur_sig"])

        # --- NEW: Posterior Calculation & Stats (from loaded file) ---
        m_post = np.full_like(x, np.nan) # Initialize posterior PDF as NaN
        post_calculated = False
        
        if z_posterior is not None:
            print("Calculating stats for 'posterior'...")
            (_, m_post, s_post), tm_post = compute_stats_from_stack(z_posterior, edges)
            
            # Store posterior stats
            stats["nT_post"].append(tm_post["nT"])
            stats["var_post"].append(tm_post["var_mu"])
            stats["std_post"].append(tm_post["std_mu"])
            stats["skw_post"].append(tm_post["skw_mu"])
            stats["kur_post"].append(tm_post["kur_mu"])
            post_calculated = True
        
        if not post_calculated:
            # Append NaNs if posterior wasn't loaded/processed
            stats["nT_post"].append(np.nan)
            stats["var_post"].append(np.nan)
            stats["std_post"].append(np.nan)
            stats["skw_post"].append(np.nan)
            stats["kur_post"].append(np.nan)

        # --- (REMOVED) Old posterior sampling loop ---

        # --- Store PDF data for plotting ---
        all_pdf_data.append({
            "scale": scale,
            "x": x,
            "m_no": m_no,
            "m_wi": m_wi,
            "m_post": m_post  # This will be populated or NaN
        })

    # --- End of processing loop ---
    
    # --- Save the computed data ---
    data_to_save = {"stats": stats, "all_pdf_data": all_pdf_data}
    with open(FINAL_DATA_PATH, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"\nProcessing complete. Saved data to {FINAL_DATA_PATH}")


# =================================================
#  --- PLOTTING  ---
# =================================================

# =========================
#  --- FIGURE 1: PDFs by scale ---
# =========================
print("Generating Figure 1: Vorticity PDFs by Scale...")
fig1, pdf_axes = plt.subplots(
    len(SCALES), 1, figsize=(5, 3.6 * len(SCALES)),
    sharey=True, constrained_layout=True, sharex=True
)
if len(SCALES) == 1:
    pdf_axes = [pdf_axes]

# Loop through the loaded or computed PDF data
for j, pdf_data in enumerate(all_pdf_data):
    ax = pdf_axes[j]
    
    # Extract data for this scale
    scale = pdf_data["scale"]
    x = pdf_data["x"]
    m_no = pdf_data["m_no"]
    m_wi = pdf_data["m_wi"]
    m_post = pdf_data["m_post"]

    # --- Plotting ---
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
    
    line_no, = ax.plot(x, m_no, lw=2, label="NA sim. ζ") 
    line_wi, = ax.plot(x, m_wi, lw=2, color="tab:red", label="Extracted ζ")
    
    # Only plot posterior if it's not all NaNs
    if not np.all(np.isnan(m_post)):
        ax.plot(x, m_post, lw=2, color='green', linestyle='-', label='Posterior ζ')

    # --- Finish Plotting PDF ---
    ax.text(
        0.02, 0.98, f"{scale} km",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=20,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
    )

    if j == 4 : # Only label bottom axis
        ax.set_xlabel("ζ / f", fontsize=20)
    if j == 0:
        ax.legend(frameon=True, fontsize=16)
    ax.set_ylabel("PDF", fontsize=20)
    ax.set_xlim(-2, 2)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(False)

fig1.savefig("Vorticity_PDFs_by_Scale.pdf", dpi=300)
print("Saved Vorticity_PDFs_by_Scale.pdf")

# =========================
#  --- FIGURE 2: Moments vs scale --- (function of time so we could have error bars if needed)
# =========================
print("Generating Figure 2: Vorticity Stats vs Scale...")

for k in stats:
    stats[k] = np.array(stats[k], dtype=float)

scales = stats["scale"]
plot_scales = np.array([s if s > 0 else 0.5 for s in scales], dtype=float)

fig2, axs = plt.subplots(3, 1, figsize=(7, 10), constrained_layout=True, sharex=True)
axs = axs.ravel()

# (y_mean_key, y_err_key, y_mean_key_wi, y_err_key_wi, y_key_post, label)
pairs = [
    ("var_no_mu", "var_no_sig", "var_wi_mu", "var_wi_sig", "var_post", "Variance"),
    # ("std_no_mu", "std_no_sig", "std_wi_mu", "std_wi_sig", "std_post", "Std. dev."),
    ("skw_no_mu", "skw_no_sig", "skw_wi_mu", "skw_wi_sig", "skw_post", "Skewness"),
    ("kur_no_mu", "kur_no_sig", "kur_wi_mu", "kur_wi_sig", "kur_post", "Kurtosis"),
]

for i, (ax, (k_no_mu, k_no_er, k_wi_mu, k_wi_er, k_post, ylab)) in enumerate(zip(axs, pairs)):

    ax.plot(plot_scales, stats[k_no_mu], 'o-', label="NA sim. ζ", color='tab:blue', markersize=6)
    ax.plot(plot_scales, stats[k_wi_mu], 'o-', label="Extracted ζ", color='tab:red', markersize=6)
    ax.plot(plot_scales, stats[k_post], 'go-', label="Posterior ζ", markersize=8, markerfacecolor='none')

    ax.set_title(ylab, fontsize=18)
    ax.set_xscale("log", base=2)
    ax.set_xticks(plot_scales)
    ax.set_xticklabels([f"{int(x)}" for x in plot_scales])
    if i == 0:
        ax.legend(frameon=True, fontsize=16)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=16)
    if i == 2: # Only label bottom axis
        ax.set_xlabel("Smoothing scale [km]", fontsize=16)

fig2.savefig("Vorticity_Stats_vs_Scale.pdf", dpi=150)
print("Saved Vorticity_Stats_vs_Scale.pdf")

plt.show()
print("\nDone.")

# =========================
#  --- FIGURE 3: Moments computed directly from time-mean PDFs ---
# =========================
print("Generating Figure 3: Moments from time-mean PDFs vs Scale...")

def moments_from_pdf(x, p):
    """
    Compute mean, variance, skewness, and excess kurtosis from a discrete PDF p(x).
    p is assumed to be values at centers x. We renormalize with Δx.
    Returns: (mean, var, skew, kurt_excess)
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # Guard: need at least 2 finite points
    m = np.isfinite(x) & np.isfinite(p)
    x = x[m]; p = p[m]
    if x.size < 2 or np.all(p == 0) or np.any(np.isnan(p)):
        return (np.nan, np.nan, np.nan, np.nan)

    # Assume uniform binning (your pooled_bins uses linspace)
    dx = np.mean(np.diff(x))
    if not np.isfinite(dx) or dx <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    # Normalize PDF to integrate to 1
    Z = np.sum(p) * dx
    if Z <= 0 or not np.isfinite(Z):
        return (np.nan, np.nan, np.nan, np.nan)
    p = p / Z

    mu = np.sum(x * p) * dx  # mean
    c2 = np.sum(((x - mu) ** 2) * p) * dx
    if c2 <= 0 or not np.isfinite(c2):
        return (mu, np.nan, np.nan, np.nan)
    sig = np.sqrt(c2)

    c3 = np.sum(((x - mu) ** 3) * p) * dx
    c4 = np.sum(((x - mu) ** 4) * p) * dx

    skewness = c3 / (sig ** 3)
    kurt_excess = (c4 / (sig ** 4)) - 3.0  # Fisher/excess kurtosis

    return (mu, c2, skewness, kurt_excess)

# Collect arrays vs scale
scales_pdf = []
var_no_pdf, var_wi_pdf, var_po_pdf = [], [], []
skw_no_pdf, skw_wi_pdf, skw_po_pdf = [], [], []
kur_no_pdf, kur_wi_pdf, kur_po_pdf = [], [], []

for pdf_data in all_pdf_data:
    scale = pdf_data["scale"]
    x     = pdf_data["x"]
    m_no  = pdf_data["m_no"]
    m_wi  = pdf_data["m_wi"]
    m_po  = pdf_data["m_post"]

    # NA sim.
    _, v_no, s_no, k_no = moments_from_pdf(x, m_no)
    # Extracted
    _, v_wi, s_wi, k_wi = moments_from_pdf(x, m_wi)
    # Posterior (may be NaN if not computed)
    if np.all(np.isnan(m_po)):
        v_po = s_po = k_po = np.nan
    else:
        _, v_po, s_po, k_po = moments_from_pdf(x, m_po)

    scales_pdf.append(scale)
    var_no_pdf.append(v_no);  var_wi_pdf.append(v_wi);  var_po_pdf.append(v_po)
    skw_no_pdf.append(s_no);  skw_wi_pdf.append(s_wi);  skw_po_pdf.append(s_po)
    kur_no_pdf.append(k_no);  kur_wi_pdf.append(k_wi);  kur_po_pdf.append(k_po)

scales_pdf = np.asarray(scales_pdf, dtype=float)
plot_scales_pdf = np.array([s if s > 0 else 0.5 for s in scales_pdf], dtype=float)

fig3, axs3 = plt.subplots(3, 1, figsize=(7, 15), sharex=False)
plt.subplots_adjust(hspace=0.5) 
axs3 = axs3.ravel()

def _line(ax, x, y, label, style_kwargs):
    ax.plot(x, y, **style_kwargs, label=label)

# 1) Variance
ax = axs3[0]
_line(ax, plot_scales_pdf, var_no_pdf, "NA sim. ζ", dict(marker='o', linestyle='-', linewidth=1.8   , color='tab:blue'  ))
_line(ax, plot_scales_pdf, var_wi_pdf, "Extracted ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:red'   ))
_line(ax, plot_scales_pdf, var_po_pdf, "Posterior ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:green' ))
ax.set_title("Variance", fontsize=18)
ax.set_xscale("log", base=2)
ax.set_xticks(plot_scales_pdf)
ax.set_xticklabels([f"{int(x)}" for x in plot_scales_pdf])
ax.legend(frameon=True, fontsize=16)
ax.grid(False)
ax.set_ylim(0, 0.6)
ax.tick_params(axis='both', labelsize=16)

# 2) Skewness
ax = axs3[1]
_line(ax, plot_scales_pdf, skw_no_pdf, "NA sim. ζ", dict(marker='o', linestyle='-', linewidth=1.8   , color='tab:blue'))
_line(ax, plot_scales_pdf, skw_wi_pdf, "Extracted ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:red' ))
_line(ax, plot_scales_pdf, skw_po_pdf, "Posterior ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:green' ))
ax.set_title("Skewness", fontsize=18)
ax.set_xscale("log", base=2)
ax.set_xticks(plot_scales_pdf)
ax.set_xticklabels([f"{int(x)}" for x in plot_scales_pdf])
ax.grid(False)
ax.set_ylim(0, 1.5)
ax.tick_params(axis='both', labelsize=16)

# 3) Kurtosis (excess)
ax = axs3[2]
_line(ax, plot_scales_pdf, kur_no_pdf, "NA sim. ζ", dict(marker='o', linestyle='-', linewidth=1.8   , color='tab:blue'  ))
_line(ax, plot_scales_pdf, kur_wi_pdf, "Extracted ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:red'   ))
_line(ax, plot_scales_pdf, kur_po_pdf, "Posterior ζ", dict(marker='o', linestyle='-', linewidth=1.8 , color='tab:green' ))
ax.set_title("Kurtosis", fontsize=18)
ax.set_xscale("log", base=2)
ax.set_xticks(plot_scales_pdf)
ax.set_xticklabels([f"{int(x)}" for x in plot_scales_pdf])
ax.set_xlabel("Smoothing scale [km]", fontsize=16)
ax.grid(False)
ax.set_ylim(0, 4.0)
ax.tick_params(axis='both', labelsize=16)

fig3.savefig("Vorticity_Stats_from_PDFs_vs_Scale.pdf", dpi=150)
print("Saved Vorticity_Stats_from_PDFs_vs_Scale.pdf")

# =========================
#  --- FIGURE 4: Example ζ fields by Scale and Case (adds Posterior sample) ---
# =========================
print("Generating Figure 4: Example ζ fields (NA sim., Extracted, Posterior) by Scale...")

def _first_vort(ssh_stack, t_idx=0):
    """Compute ζ for a single time slice."""
    z = stack_vort(ssh_stack[t_idx:t_idx+1])[0]
    return z

# --- REVISED Figure 4 helper function ---
def _posterior_vort_t0(scale_km, t_idx=0):
    """Load 2D posterior and get ζ for it."""
    try:
        path_post = path_posterior(scale_km) # Get path
        ht_po = np.asarray(safe_load(path_post), dtype=float)
        
        if ht_po.ndim != 2:
             raise ValueError(f"Expected 2D array but got {ht_po.ndim}D")

        # It's a single 2D field (time-mean).
        print(f"Fig 4: Loaded 2D posterior field for scale {scale_km}.")
        lat_dim_size = ht_po.shape[0]
        lats_sliced = lat_1d[0:lat_dim_size]
        z = swot.compute_geostrophic_vorticity(ht_po, dx_m, dy_m, lats_sliced)
        
        return z
    
    except Exception as e:
        # Add scale_km to the print statement
        print(f"Fig 4: Could not load/process posterior for scale {scale_km}: {e}")
        return None

# Build figure: rows = scales, cols = 3 (NA sim., Extracted, Posterior sample)
nrows = len(SCALES); ncols = 3
fig4, axs4 = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows), sharex=False, sharey=False)
plt.subplots_adjust(hspace=0.35, wspace=0.18)

if nrows == 1:
    axs4 = np.array([axs4])  # ensure 2D indexing

dx_km = dx_m / 1e3
dy_km = dy_m / 1e3
t0 = 0  # same time slice for all panels
last_im = None

for r, scale in enumerate(SCALES):
    # Load stacks for this scale
    ht_no = np.asarray(safe_load(path_nonoise(scale)), dtype=float)
    ht_wi = np.asarray(safe_load(path_withnoise(scale)), dtype=float)

    # Compute ζ for NA sim. and Extracted at t0
    z_no = _first_vort(ht_no, t_idx=t0)
    z_wi = _first_vort(ht_wi, t_idx=t0)

    # --- UPDATED: Get posterior ζ at t0 from file ---
    z_po = _posterior_vort_t0(scale_km=scale, t_idx=t0) # t_idx is ignored but kept for consistency

    # Symmetric color scale per row using 99th percentile over available fields
    pools = [finite_flat(z_no), finite_flat(z_wi)]
    if z_po is not None:
        pools.append(finite_flat(z_po))
    both_abs = np.abs(np.concatenate(pools)) if len(pools) > 0 else np.array([])
    vmax = np.nanpercentile(both_abs, 99) if both_abs.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # Extent in km
    ny, nx = z_no.shape
    extent = [0, nx * dx_km, 0, ny * dy_km]

    # Column 0: NA sim.
    ax = axs4[r, 0]
    im0 = ax.imshow(z_no, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    if r == 0:
        ax.set_title("NA sim. ζ", fontsize=14)
    ax.set_ylabel(f"{scale} km\n y [km]", fontsize=12)
    ax.tick_params(labelsize=11)

    # Column 1: Extracted
    ax = axs4[r, 1]
    im1 = ax.imshow(z_wi, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    if r == 0:
        ax.set_title("Extracted ζ", fontsize=14)
    ax.tick_params(labelsize=11)

    # Column 2: Posterior sample (or N/A)
    ax = axs4[r, 2]
    if z_po is not None:
        im2 = ax.imshow(z_po, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        origin="lower", extent=extent, aspect="auto", interpolation="nearest")
        last_im = im2  # for colorbar
    else:
        # Grey panel + text if unavailable
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "Posterior N/A", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        im2 = im1  # keep something valid for colorbar hookup
        if r == nrows - 1: # If last row is also N/A, use last valid im
             last_im = im1


    if r == 0:
        ax.set_title("Posterior sample ζ", fontsize=14)
    ax.tick_params(labelsize=11)


# Label bottom x-axes
for c in range(ncols):
    axs4[-1, c].set_xlabel("x [km]", fontsize=12)

# Single colorbar on the right
if last_im is not None:
    cbar = fig4.colorbar(last_im, ax=axs4, location="right", fraction=0.025, pad=0.04)
    cbar.set_label("ζ / f", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
else:
    print("Warning: Could not create colorbar for Figure 4 (no valid images).")


fig4.savefig("Vorticity_Example_Fields_with_Posterior_by_Scale.pdf", dpi=200, bbox_inches="tight")
print("Saved Vorticity_Example_Fields_with_Posterior_by_Scale.pdf")