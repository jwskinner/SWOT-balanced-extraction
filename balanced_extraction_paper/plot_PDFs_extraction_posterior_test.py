import numpy as np
import os, pickle
from scipy.sparse.linalg import eigsh
import jws_swot_tools as swot
import matplotlib.pyplot as plt
import scipy.linalg as la
import h5py

timer = swot.Timer()

PICKLES = "./pickles"
KARIN_NA_PATH = f"{PICKLES}/karin_NA_tmean.pkl"  # where SWOT data is held
RESULTS_PATH = f"{PICKLES}/posterior_vorticity_results.pkl"

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def path_nonoise(km):  # "ground truth"
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_sm_{km}km_nonoise.pkl"

def path_withnoise(km):  # extracted balanced field
    return f"{PICKLES}/balanced_extraction_synth_NA_tmean_sm_{km}km.pkl"

def finite_flat(*arrays):
    outs = []
    for a in arrays:
        v = np.asarray(a, dtype=float).ravel()
        outs.append(v[np.isfinite(v)])
    return outs

# Loop over times and samples for the posterior and for each scale 
# and save all the vorticies and ssh from extraction and posterior samples
SCALES = [0, 1, 2, 4, 8, 16]
NSAMPLES = 20 # number of posterior samples per time

karin = load(KARIN_NA_PATH)
dx_m = float(karin.dx_km) * 1e3
dy_m = float(karin.dy_km) * 1e3
lat_1d = np.asarray(karin.lat)[0, :, 0]  # for gvort
timer.lap("Loaded files")

# =========================
# Compute or load results
# =========================
results = {}

for km in SCALES:
    print(f"{km} km")
    ht_sim = np.asarray(load(path_nonoise(km)), dtype=float)
    ht_ext = np.asarray(load(path_withnoise(km)), dtype=float)
    T, ny, nx = ht_ext.shape
    sim_vort_all = np.zeros((T, ny, nx))
    ext_vort_all = np.zeros((T, ny, nx))
    post_vort_all = np.zeros((T, NSAMPLES, ny, nx))
    post_ssh_all  = np.zeros((T, NSAMPLES, ny, nx)) 

    L_PATH = f"{PICKLES}/posterior_balanced_extraction_synth_NA_tmean_sm_{km}km.pkl"
    C = load(L_PATH)
    Lfac, lower = la.cho_factor(C + np.eye(C.shape[0]) * 1.0e-10, lower=True) 
    Ltri = np.tril(Lfac)
    timer.lap("Cholesky factorisation")

    for t in range(0, T):
        print(f"t = {t}", end="\r", flush=True)
        # posterior sample: x ~ N(mu, C) with C ≈ L L^T
        mu = (ht_ext[t] * 100).ravel()  # in cm for extraction
        r = C.shape[1]
        for s in range(0, NSAMPLES):  # sample posterior
            z = np.random.randn(r)  # new seed each s
            # posterior sample: x ~ N(mu, C) with C ≈ L L^T
            ht_post = (mu + Ltri @ z).reshape(ny, nx) / 100.0  # meters
            post_ssh_all[t, s] = ht_post
            post_vort = swot.compute_geostrophic_vorticity(ht_post, dx_m, dy_m, lat_1d)
            post_vort_all[t, s] = post_vort

        sim_vort = swot.compute_geostrophic_vorticity(ht_sim[t], dx_m, dy_m, lat_1d)
        ext_vort = swot.compute_geostrophic_vorticity(ht_ext[t], dx_m, dy_m, lat_1d)
        sim_vort_all[t] = sim_vort
        ext_vort_all[t] = ext_vort

    timer.lap(f"processed: {km} km")

    results[km] = dict(
        sim      = sim_vort_all,
        ext      = ext_vort_all,
        post     = post_vort_all,
        ssh_sim  = ht_sim,         
        ssh_ext  = ht_ext,          
        ssh_post = post_ssh_all,   
    )

timer.lap("Processing Done")

# Save to pickle for future runs
with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved results to {RESULTS_PATH}")