#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------ Minimal imports ------
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.ndimage import gaussian_filter
import JWS_SWOT_toolbox as swot

# =========================
# --- CONFIG (edit here) ---
# =========================
PICKLES = "./pickles"
KARIN_NA_PATH = f"{PICKLES}/karin_NA.pkl"
SCALES = [1, 2, 4, 8, 16]  # km

# Posterior covariance (OPTIONAL). If missing or wrong size, code skips posterior gracefully.
POSTERIOR_COV_PATH = f"{PICKLES}/posterior_full_synth_NA_tmean_sm_1km.pkl"
N_POST_SAMPLES = 20   # per time for posterior PDFs
T0 = 0                # same time-index for example fields and posterior

def path_nonoise(scale_km):
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_nonoise_sm_{scale_km}km.pkl"

def path_withnoise(scale_km):
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_sm_{scale_km}km.pkl"

# =========================
# --- LOAD GRID ONCE ---
# =========================
os.makedirs(PICKLES, exist_ok=True)
karin = pickle.load(open(KARIN_NA_PATH, "rb"))
dx_m = float(karin.dx_km) * 1e3
dy_m = float(karin.dy_km) * 1e3
lat_1d = np.asarray(karin.lat, float)[0, :, 0]

# =========================
# --- SMALL HELPERS ---
# =========================
def finite_flat(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

def _sigma_pixels(scale_km):
    # use std = scale / (2π) so "scale_km" ≈ e-folding diameter scale
    sig_km = scale_km / (2.0 * np.pi)
    return (sig_km * 1e3 / dy_m, sig_km * 1e3 / dx_m)  # (sy, sx) in pixels

def stack_vort(ssh_stack):
    """Compute ζ for each time slice in ssh_stack (t,y,x)."""
    out = []
    ny = ssh_stack.shape[1]
    lats = lat_1d[:ny]
    for i in range(ssh_stack.shape[0]):
        eta = np.asarray(ssh_stack[i], float)
        zeta = swot.compute_geostrophic_vorticity(eta, dx_m, dy_m, lats)
        out.append(zeta)
    return np.stack(out, axis=0)

def pooled_bins(*arrays, nbins=120):
    pool = np.concatenate([finite_flat(a) for a in arrays])
    if pool.size == 0:
        return np.linspace(-1, 1, nbins + 1)
    lo, hi = np.percentile(pool, [0.25, 99.75])
    bound = max(abs(lo), abs(hi))
    bound = 1.0 if bound == 0 else bound
    return np.linspace(-bound, bound, nbins + 1)

def pdf_and_moments(z_stack, edges):
    """Time-mean PDF and simple moments computed per-time then averaged."""
    nT = z_stack.shape[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdfs = np.zeros((nT, centers.size))
    var, skew, kurt = np.zeros(nT), np.zeros(nT), np.zeros(nT)

    for i in range(nT):
        v = finite_flat(z_stack[i])
        if v.size < 2:
            pdfs[i, :] = np.nan
            var[i] = skew[i] = kurt[i] = np.nan
            continue
        h, _ = np.histogram(v, bins=edges, density=True)
        pdfs[i, :] = h
        m = np.mean(v)
        c2 = np.mean((v - m)**2)
        c3 = np.mean((v - m)**3)
        c4 = np.mean((v - m)**4)
        var[i] = c2
        skew[i] = 0 if c2 == 0 else c3 / (c2**1.5)
        kurt[i] = 0 if c2 == 0 else c4 / (c2**2) - 3.0

    pdf_mean = np.nanmean(pdfs, axis=0)
    return centers, pdf_mean, np.nanmean(var), np.nanmean(skew), np.nanmean(kurt)

# =========================
# --- OPTIONAL: POSTERIOR ---
# =========================
def load_L(path):
    if not os.path.exists(path):
        return None
    C = pickle.load(open(path, "rb"))
    C = 0.5 * (C + C.T)
    try:
        return la.cholesky(C + np.eye(C.shape[0]) * 1e-8, lower=True)
    except la.LinAlgError:
        return None

L_post = load_L(POSTERIOR_COV_PATH)

def posterior_flat_vort_samples(mean_ssh_t, L, n_samples, scale_km):
    """Draw n_samples SSH (same size as mean_ssh_t), smooth, return all ζ values (flattened)."""
    nY, nX = mean_ssh_t.shape
    n = nY * nX
    if L is None or L.shape[0] != n:
        return None
    mu_cm = mean_ssh_t.ravel() * 100.0
    Z = np.random.randn(n, n_samples)
    ssh_samples_cm = mu_cm[:, None] + L @ Z
    ssh_samples_m = (ssh_samples_cm / 100.0).T.reshape(n_samples, nY, nX)
    sy, sx = _sigma_pixels(scale_km)
    ssh_smooth = gaussian_filter(ssh_samples_m, sigma=(0, sy, sx), mode="nearest")
    z_stack = stack_vort(ssh_smooth)
    return finite_flat(z_stack)

# =========================
# --- MAIN: COMPUTE & PLOT ---
# =========================
# Storage vs scale
scale_list = []
var_no, var_wi, var_po = [], [], []
skw_no, skw_wi, skw_po = [], [], []
kur_no, kur_wi, kur_po = [], [], []
pdf_data = []  # list of dicts with keys: scale, x, m_no, m_wi, m_po

# For Figure 4 example fields
example_rows = []  # each: (scale, z_no_t0, z_wi_t0, z_po_t0_or_None, extent_km, vmax)

for scale in SCALES:
    print(f"=== Scale {scale} km ===")
    ht_no = np.asarray(pickle.load(open(path_nonoise(scale), "rb")), float)  # (t,y,x)
    ht_wi = np.asarray(pickle.load(open(path_withnoise(scale), "rb")), float)

    # ζ stacks
    z_no = stack_vort(ht_no)
    z_wi = stack_vort(ht_wi)

    # Common bins
    edges = pooled_bins(z_no, z_wi, nbins=120)

    # PDFs + moments
    x, m_no, v_no, s_no, k_no = pdf_and_moments(z_no, edges)
    _, m_wi, v_wi, s_wi, k_wi = pdf_and_moments(z_wi, edges)

    # Posterior PDFs/moments (optional)
    m_po = np.full_like(x, np.nan)
    v_po = s_po = k_po = np.nan
    z_po_t0 = None
    if L_post is not None and ht_wi.size > 0:
        v_samples = posterior_flat_vort_samples(ht_wi[T0], L_post, N_POST_SAMPLES, scale_km=scale)
        if v_samples is not None and v_samples.size > 1:
            h_po, _ = np.histogram(v_samples, bins=edges, density=True)
            m_po = h_po
            mm = np.mean(v_samples)
            c2 = np.mean((v_samples - mm)**2)
            c3 = np.mean((v_samples - mm)**3)
            c4 = np.mean((v_samples - mm)**4)
            v_po = c2
            s_po = 0 if c2 == 0 else c3 / (c2**1.5)
            k_po = 0 if c2 == 0 else c4 / (c2**2) - 3.0


            nY, nX = ht_wi[T0].shape
            n = nY * nX
            Z = np.random.randn(n, 1)
            ssh1 = (ht_wi[T0].ravel() * 100.0 + (L_post @ Z).ravel())/100.0
            ssh1 = ssh1.reshape(nY, nX)
            sy, sx = _sigma_pixels(scale)
            ssh1s = gaussian_filter(ssh1, sigma=(sy, sx), mode="nearest")
            z_po_t0 = stack_vort(ssh1s[None, ...])[0]
            plt.figure(figsize=(5,4))
            plt.imshow(z_po_t0, origin="lower")
            plt.colorbar()
            plt.title("Posterior ζ example")
            plt.tight_layout()
            plt.savefig("test.png", dpi=150)
            plt.close()

    # Store stats
    scale_list.append(scale)
    var_no.append(v_no); var_wi.append(v_wi); var_po.append(v_po)
    skw_no.append(s_no); skw_wi.append(s_wi); skw_po.append(s_po)
    kur_no.append(k_no); kur_wi.append(k_wi); kur_po.append(k_po)
    pdf_data.append(dict(scale=scale, x=x, m_no=m_no, m_wi=m_wi, m_po=m_po))

    # Prepare example fields for Figure 4
    z_no_t0 = z_no[T0]
    z_wi_t0 = z_wi[T0]
    pools = np.abs(np.concatenate([finite_flat(z_no_t0), finite_flat(z_wi_t0)]))
    if np.isfinite(v_po) and z_po_t0 is not None:
        pools = np.abs(np.concatenate([pools, finite_flat(z_po_t0)]))
    vmax = np.nanpercentile(pools, 99) if pools.size else 1.0
    ny, nx = z_no_t0.shape
    extent = [0, nx * dx_m/1e3, 0, ny * dy_m/1e3]  # km
    example_rows.append((scale, z_no_t0, z_wi_t0, z_po_t0, extent, vmax))

# -------------------------
# Figure 1: PDFs by scale
# -------------------------
fig1, axes = plt.subplots(len(SCALES), 1, figsize=(5, 3.2 * len(SCALES)), sharex=True)
if len(SCALES) == 1: axes = [axes]
for ax, d in zip(axes, pdf_data):
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax.plot(d["x"], d["m_no"], lw=2, label="NA sim. ζ")
    ax.plot(d["x"], d["m_wi"], lw=2, label="Extracted ζ", color="tab:red")
    if not np.all(np.isnan(d["m_po"])):
        ax.plot(d["x"], d["m_po"], lw=2, label="Posterior ζ", color="tab:green")
    ax.text(0.02, 0.95, f'{d["scale"]} km', transform=ax.transAxes, va="top")
    ax.set_ylabel("PDF")
axes[-1].set_xlabel("ζ / f")
axes[0].legend()
fig1.tight_layout()
fig1.savefig("Vorticity_PDFs_by_Scale.pdf", dpi=300)
print("Saved Vorticity_PDFs_by_Scale.pdf")

# -------------------------
# Figure 2: moments vs scale
# -------------------------
sc = np.array(scale_list, float)
def _plot(ax, y_no, y_wi, y_po, title):
    ax.plot(sc, y_no, "o-", label="NA sim. ζ")
    ax.plot(sc, y_wi, "o-", label="Extracted ζ", color="tab:red")
    ax.plot(sc, y_po, "o-", label="Posterior ζ", color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sc)
    ax.set_xticklabels([str(int(s)) for s in sc])
    ax.set_title(title); ax.grid(False)

fig2, axs = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True)
_plot(axs[0], var_no, skw_wi=[], y_po=[], title="Variance")  # placeholder signature fix below
axs[0].cla()
_plot(axs[0], var_no, var_wi, var_po, "Variance")
_plot(axs[1], skw_no, skw_wi, skw_po, "Skewness")
_plot(axs[2], kur_no, kur_wi, kur_po, "Kurtosis (excess)")
axs[2].set_xlabel("Smoothing scale [km]")
axs[0].legend()
fig2.tight_layout()
fig2.savefig("Vorticity_Stats_vs_Scale.pdf", dpi=200)
print("Saved Vorticity_Stats_vs_Scale.pdf")

# -------------------------
# Figure 3: example ζ fields
# -------------------------
nrows, ncols = len(example_rows), 3
fig3, axs3 = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows))
if nrows == 1: axs3 = np.array([axs3])
for r, (scale, z0, z1, zp, extent, vmax) in enumerate(example_rows):
    # col 0: NA sim
    ax = axs3[r, 0]
    im0 = ax.imshow(z0, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", extent=extent, aspect="auto")
    if r == 0: ax.set_title("NA sim. ζ")
    ax.set_ylabel(f"{scale} km\n y [km]")

    # col 1: Extracted
    ax = axs3[r, 1]
    im1 = ax.imshow(z1, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", extent=extent, aspect="auto")
    if r == 0: ax.set_title("Extracted ζ")

    # col 2: Posterior sample (optional)
    ax = axs3[r, 2]
    if zp is not None:
        im2 = ax.imshow(zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", extent=extent, aspect="auto")
    else:
        ax.set_facecolor("#efefef"); ax.text(0.5, 0.5, "Posterior N/A", ha="center", va="center", transform=ax.transAxes)
        im2 = im1
    if r == 0: ax.set_title("Posterior ζ (example)")
    axs3[r, 0].set_xlabel("x [km]") if r == nrows-1 else None
    axs3[r, 1].set_xlabel("x [km]") if r == nrows-1 else None
    axs3[r, 2].set_xlabel("x [km]") if r == nrows-1 else None

# one colorbar
fig3.subplots_adjust(right=0.88, wspace=0.2, hspace=0.35)
cax = fig3.add_axes([0.9, 0.15, 0.02, 0.7])
cb = fig3.colorbar(im2, cax=cax)
cb.set_label("ζ / f")
fig3.savefig("Vorticity_Example_Fields_with_Posterior_by_Scale.pdf", dpi=200)
print("Saved Vorticity_Example_Fields_with_Posterior_by_Scale.pdf")

print("\nDone.")
