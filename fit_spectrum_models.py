# Script for reading in and processing the SWOT KaRIn and Nadir data and fitting models to their spectra 
# There are two options for the model fit: 1) fit the cross-track averaged spectrum and 2) fit the spectrum at each along-track point.

import numpy as np
import JWS_SWOT_toolbox as swot

pass_num = 9
lat_max = 38
lat_min = 28
data_folder = '/expanse/lustre/projects/cit197/jskinner1/SWOT/CALVAL/' # where our data is stored

# ───── Read and Process Data ─────
# finds overlapping cycles between the karin and nadir datasets
_, _, shared_cycles, karin_files, nadir_files = swot.return_swot_files(data_folder, pass_num) 
sample_index = 2  # some index for setting up the grids 
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

# ───── Fit Models to the Spectra -- Across Track Average ─────
# KARIN
poptcwg_karin, pcovcwg_karin = swot.fit_spectrum(karin, karin.spec_alongtrack_av, swot.karin_model) # Fit the Model to the spectrum, returns fit vectors

# NADIR
poptcwg_nadir, covcwg_nadir = swot.fit_nadir_spectrum(nadir, nadir.spec_alongtrack_av, poptcwg_karin)

# plot the fits
swot.plot_spectral_fits(karin, nadir, poptcwg_karin, poptcwg_nadir)

# ───── Fit Models to the Spectra -- Across Track ─────
poptcwg_karins, pcovcwg_karins = swot.fit_spectrum_across_track(karin, karin.spec_alongtrack_time_av, swot.karin_model)
