
import netCDF4 as nc
import numpy as np
from glob import glob
import math
import re
import xarray as xr
import xrft
from math import sin, cos, sqrt, atan2, radians
from scipy.special import gamma, kv
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt

#  ------- Functions for loading and processing SWOT data
# Loads the SWOT files from the specified folder and aligns the KaRIn and Nadir files based on their cycles.
def return_karin_files(filepath):
    all_files = glob(filepath)
    files_with_numbers = [] # now order by pass to get order in time
    for filename in all_files:
        match = re.search(r'Expert_(\d+)_(\d+)', filename)
        if match:
            cycle = int(match.group(1))
            pass_num = int(match.group(2))
            files_with_numbers.append((filename, cycle, pass_num))

    files_with_numbers.sort(key=lambda x: x[1])
    return files_with_numbers

def return_nadir_files(filepath_nadir, valid_cycles):
    all_files = glob(filepath_nadir)
    nadir_files = []
    for filename in all_files:
        match = re.search(r'GPR_2PfP(\d+)_0*(\d+)', filename)
        if match:
            cycle = int(match.group(1))
            pass_num = int(match.group(2))
            if cycle in valid_cycles:
                nadir_files.append((filename, cycle, pass_num))
    nadir_files.sort(key=lambda x: x[1])  # sort by cycle
    return nadir_files

def load_swot_files(folder, folder_nadir):
    filepath = os.path.join(folder, 'SWOT_L2_LR_SSH_Expert_*.nc')
    filepath_nadir = os.path.join(folder_nadir, 'SWOT_GPR_*.nc')
    
    karin_files_with_numbers = return_karin_files(filepath)
    karin_cycles = {cycle for _, cycle, _ in karin_files_with_numbers}
    nadir_files_with_numbers = return_nadir_files(filepath_nadir, karin_cycles)
    
    karin_dict = {cycle: item for item, cycle, _ in karin_files_with_numbers}
    nadir_dict = {cycle: item for item, cycle, _ in nadir_files_with_numbers}
    
    shared_cycles = sorted(set(karin_dict) & set(nadir_dict))
    
    karin_aligned = [(karin_dict[c], c) for c in shared_cycles]
    nadir_aligned = [(nadir_dict[c], c) for c in shared_cycles]
    return karin_files_with_numbers, nadir_files_with_numbers, shared_cycles, karin_aligned, nadir_aligned


# Returns the indices of the track in the KaRIn file that fall within the specified latitude range.
def get_karin_track_indices(karin_file, lat_min, lat_max):
    karin_ref = nc.Dataset(karin_file, 'r')
    fp_latitude = karin_ref['latitude']
    mid_col = fp_latitude.shape[1] // 2
    lat_center = fp_latitude[:, mid_col]
    indx = np.where((lat_center >= lat_min) & (lat_center <= lat_max))[0]
    track_length = len(indx)
    karin_ref.close()
    return indx, track_length

# Returns the indices of the nadir track in the Nadir file that fall within the specified latitude range.
def get_nadir_track_indices(nadir_file, lat_min, lat_max):
    nadir_ref = nc.Dataset(nadir_file, 'r')
    lats_fp = nadir_ref['data_01']['latitude'][:]
    indxs = np.where((lats_fp >= lat_min) & (lats_fp <= lat_max))[0]
    track_length_nadir = len(indxs)
    nadir_ref.close()
    return indxs, track_length_nadir

# Loads the KaRIn data for the specified cycles and processes it to extract SSH anomalies.
def load_karin_data(shared_cycles, folder, pass_number, indx, swath_width, 
                   lat_karin, lon_karin, ssha_karin, swh_karin, tide_karin):
    good_strips_list = []
    bad_strips_list = []
    num_useful_strips = 0
    num_bad_strips = 0

    for n, cycle in enumerate(shared_cycles):
        path = glob(folder + f'*_{int(cycle):03d}_{pass_number:03d}_*.nc')
        if not path:
            continue

        data = nc.Dataset(path[0], 'r')

        for side in [0, 1]:
            i0 = 34 * side + 5
            i1 = i0 + swath_width

            ssha = data['ssha_karin_2'][indx, i0:i1]
            qual = data['ssha_karin_2_qual'][indx, i0:i1]
            xcor = data['height_cor_xover'][indx, i0:i1]
            xqual = data['height_cor_xover_qual'][indx, i0:i1]
            tide = data['internal_tide_hret'][indx, i0:i1]
            lat  = data['latitude'][indx, i0:i1]
            lon  = data['longitude'][indx, i0:i1]
            tvar = data['time'][indx]
            swh = data['swh_karin'][indx, i0:i1]

            # Removes bad sides completely
            # if np.ma.is_masked(ssha) or np.ma.is_masked(xcor) or np.any(qual):
            #     num_bad_strips += 1
            #     bad_strips_list.append((cycle, side))
            #     continue
            
            # Masks bad data points and leaves the rest 
            ssha_masked = np.where(qual, np.nan, ssha + xcor - tide) # mask the poor quality data
            swh_masked = np.where(qual, np.nan, swh) # mask the poor quality data
            tide_masked = np.where(qual, np.nan, tide) # mask the poor quality data
            j = slice(0, swath_width) if side == 0 else slice(-swath_width, None)
            lat_karin[n, :, j] = lat
            lon_karin[n, :, j] = lon
            ssha_karin[n, :, j] = ssha_masked
            swh_karin[n, :, j] = swh_masked
            tide_karin[n, :, j] = tide_masked
            num_useful_strips += 1
            good_strips_list.append((cycle, side))

        data.close()

    print(f"Number of good KaRIn strips: {num_useful_strips}")
    print(f"Number of bad KaRIn strips: {num_bad_strips}")
    print(f"Good strips list: {len(good_strips_list)}")
    print(f"Bad strips list: {bad_strips_list}")
    return good_strips_list, bad_strips_list, num_useful_strips, num_bad_strips

def process_karin_data(ssha_karin, swath_width, middle_width, cutoff_m=100e3, delta=2e3):
    '''Processes the KaRIn data to remove the high-pass filtered time-mean and return SSH anomalies'''
    
    ssha_karin[:, :, swath_width:swath_width + middle_width] = np.nan # Fill middle section with NaN
    ssha_karin_arr = np.asarray(ssha_karin, dtype=float)
    nan_mask = np.isnan(ssha_karin_arr)
    ssha_karin_arr[nan_mask] = 0.0 # replace nans with 0.0 for the filter step
    ssha_mean_karin = np.nanmean(ssha_karin_arr, axis=0)

    # Apply Gaussian filter to time mean
    sigma_m = cutoff_m / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_m / delta
    lowpass = gaussian_filter(ssha_mean_karin, sigma=(sigma_pixels, sigma_pixels), mode='reflect')
    lowpass[:, swath_width:swath_width + middle_width] = np.nan
    
    ssha_highpass_karin = ssha_mean_karin - lowpass
    ssha_anom_karin = ssha_karin - ssha_highpass_karin[None, :, :]
    ssha_anom_karin[nan_mask] = np.nan  # Restore NaN mask in anomalies
    return ssha_mean_karin, ssha_highpass_karin, ssha_anom_karin

def load_nadir_data(nadir_files_with_numbers, lat_min, lat_max, 
                   ssha_nadir, lat_nadir, lon_nadir, track_length_nadir):
    num_good_cycles = 0
    num_bad_cycles = 0
    
    for n, (filename, cycle, pass_number) in enumerate(nadir_files_with_numbers):
        data = nc.Dataset(filename, 'r')
        group = data['data_01']
        
        lats = group['latitude'][:]
        lons = data['data_01']['longitude'][:]
        indxs = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        
        if len(indxs) == 0 or len(group['ku']['ssha']) <= indxs[-1]:
            num_bad_cycles += 1
            data.close()
            continue
        
        tide = group['internal_tide_hret'][indxs]
        ssha = group['ku']['ssha'][indxs] - tide
        
        if np.all(ssha.mask):
            num_bad_cycles += 1
            data.close()
            continue
        
        num_good_cycles += 1
        n_valid = min(track_length_nadir, len(indxs))
        lat_nadir[n, :n_valid] = lats[indxs[:n_valid]]
        lon_nadir[n, :n_valid] = lons[indxs[:n_valid]]
        ssha_nadir[n, :n_valid] = ssha[:n_valid].filled(np.nan)
        data.close()
    
    print(f"Number of good nadir cycles: {num_good_cycles}")
    print(f"Number of bad nadir cycles: {num_bad_cycles}")
    return num_good_cycles, num_bad_cycles

def process_nadir_data(ssha_nadir, cutoff_m=100e3, delta=6.8e3):
    ssha_mean_nadir = np.nanmean(ssha_nadir, axis=0)
    
    # Apply Gaussian filter to the time mean
    sigma_m = cutoff_m / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_m / delta
    lowpass = gaussian_filter(ssha_mean_nadir, sigma=(sigma_pixels), mode='reflect')
    
    # Calculate high-pass residual and anomalies
    ssha_highpass_nadir = ssha_mean_nadir - lowpass
    ssha_anom_nadir = ssha_nadir# - ssha_highpass_nadir[None, :] # We dont remove the time mean from the nadir because its large scale signal
    return ssha_mean_nadir, ssha_highpass_nadir, ssha_anom_nadir


