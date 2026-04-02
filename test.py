import pickle
import jws_swot_tools as swot

data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE_VD/'

REGIONS = [
    {"name": "SO1",  "lat_min": -58.0, "lat_max": -50.0, "passes": [60, 88, 116, 144, 183, 211, 239, 267, 366, 394, 450, 489, 517, 573]},
    {"name": "SO2",  "lat_min": -58.0, "lat_max": -50.0, "passes": [11, 39, 67, 95, 166, 194, 222, 289, 317, 345, 373, 472, 500, 528]},
    {"name": "SO3",  "lat_min": -58.0, "lat_max": -50.0, "passes": [80, 108, 136, 175, 203, 231, 259, 330, 358, 386, 414, 453, 509, 537, 565]},
    {"name": "NA2",  "lat_min":  27.0, "lat_max":  35.0, "passes": [7, 20, 48, 257, 298, 313, 326, 563, 576]},
    {"name": "KE1",  "lat_min":  30.0, "lat_max":  38.0, "passes": [2, 17, 30, 45, 58, 295, 308, 323, 336, 351, 364]},
    {"name": "KE2",  "lat_min":  22.0, "lat_max":  30.0, "passes": [17, 30, 45, 58, 86, 267, 295, 323, 336, 364, 573]},
]

for region in REGIONS:
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    print(f"\n=== Region {region['name']} (lat {lat_min} to {lat_max}) ===")

    for pass_number in region["passes"]:
        print(f"  Pass {pass_number}...")
        outdir = f"./balanced_extraction/SWOT_data_VD_NoNad/Pass_{pass_number:03d}_Lat{lat_min}_{lat_max}_rho4km"

        try:
            _, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_number)
            sample_index = swot.get_best_sample_index(karin_files, lat_min, lat_max)

            _, track_length       = swot.get_karin_track_indices(karin_files[sample_index][0], lat_min, lat_max)
            _, track_length_nadir = swot.get_nadir_track_indices(nadir_files[sample_index][0], lat_min, lat_max)

            dims = [len(shared_cycles), track_length, track_length_nadir]
            karin, nadir = swot.init_swot_arrays(dims, lat_min, lat_max, pass_number)
            karin.sample_index = sample_index

            swot.load_karin_data(karin_files, lat_min, lat_max, karin, verbose=False)
            swot.process_karin_data(karin)
            karin.coordinates()

            swot.load_nadir_data(nadir_files, lat_min, lat_max, nadir)
            swot.process_nadir_data(nadir)
            nadir.coordinates()
            nadir.compute_spectra()

            with open(f"{outdir}/nadir_pass{pass_number:03d}.pkl", "wb") as f:
                pickle.dump(nadir, f)
            print(f"    Saved → {outdir}/nadir_pass{pass_number:03d}.pkl")

        except Exception as e:
            print(f"    FAILED: {e}")