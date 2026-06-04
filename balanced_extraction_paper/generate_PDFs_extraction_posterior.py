import numpy as np
import os
import pickle
import scipy.linalg as la
import jws_swot_tools as swot

KARIN_NA_PATH = "./synthetic_swot_data/Pass_009_Lat28_35/karin_synth.pkl"
RESULTS_PATH  = "./balanced_extraction/posterior_vorticity_samples.pkl"

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def path_nonoise(km):
    return f"./balanced_extraction/SYNTH_data_noiseless/Pass_009_Lat28_35_rho{km}km/balanced_extraction_noiseless_pass009.pkl"

def path_withnoise(km):
    return f"./balanced_extraction/SYNTH_data/Pass_009_Lat28_35_rho{km}km/balanced_extraction_pass009.pkl"

def posterior_path(km):
    return f"./balanced_extraction/SYNTH_data/Pass_009_Lat28_35_rho{km}km/posterior.pkl"

def main():
    karin  = load(KARIN_NA_PATH)
    dx     = float(karin.dx_km) * 1e3
    dy     = float(karin.dy_km) * 1e3
    lat_1d = np.asarray(karin.lat)[0, :, 0]

    scales_to_run = [1, 2, 4, 8, 16]
    n_samples = 20

    all_results = {}

    for km in scales_to_run:
        
        print(f"Processing scale: {km} km")

        ht_sim = np.asarray(load(path_nonoise(km)).ssh_balanced, dtype=float)
        ht_sim = ht_sim[:, 5:64, :] # crop into extraction swath
        ht_ext = np.asarray(load(path_withnoise(km)).ssh_balanced,   dtype=float)
        T, ny, nx = ht_ext.shape

        C = load(posterior_path(km))
        print(f"Loaded Posterior: {posterior_path(km)}")
        L = la.cholesky(C + np.eye(C.shape[0]) * 1e-8, lower=True)
        r = C.shape[1]
        print("Cholesky Done")

        vort_sim_all  = np.zeros((T,           nx, ny))
        vort_ext_all  = np.zeros((T,           nx, ny))
        ssh_post_all  = np.zeros((T, n_samples, nx, ny))
        vort_post_all = np.zeros((T, n_samples, nx, ny))

        for t in range(T):
            print(f"  t = {t}/{T-1}", end="\r", flush=True)

            vort_sim_all[t] = swot.compute_geostrophic_vorticity(ht_sim[t].T, dx, dy, lat_1d)
            vort_ext_all[t] = swot.compute_geostrophic_vorticity(ht_ext[t].T, dx, dy, lat_1d)

            mu = (ht_ext[t].T.ravel() * 100)

            for s in range(n_samples):
                z = np.random.randn(r)
                ht_post_2d = (mu + L @ z).reshape(nx, ny) / 100.0

                ssh_post_all[t, s]  = ht_post_2d
                vort_post_all[t, s] = swot.compute_geostrophic_vorticity(ht_post_2d, dx, dy, lat_1d)

        print(f"  Done")

        all_results[km] = {
            'ssh_sim':  ht_sim,
            'ssh_ext':  ht_ext,
            'ssh_post': ssh_post_all,
            'sim':      vort_sim_all,
            'ext':      vort_ext_all,
            'post':     vort_post_all,
        }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved -> {RESULTS_PATH}")

if __name__ == "__main__":
    main()
