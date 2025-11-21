
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
import pandas as pd
from IPython.display import display, Markdown
import JWS_SWOT_toolbox as swot
from netCDF4 import num2date

#  ------- Functions for loading and processing SWOT data
# Loads the SWOT files from the specified folder and aligns the KaRIn and Nadir files based on their cycles.
def return_karin_files(filepath, pnum, basic = True):
    all_files = glob(filepath)
    files_with_numbers = [] # now order by pass to get order in time
    for filename in all_files:
        if basic: 
            match = re.search(r'Basic_(\d+)_(\d+)', filename)
        else: 
            match = re.search(r'Expert_(\d+)_(\d+)', filename)
        if match:
            cycle = int(match.group(1))
            pass_num = int(match.group(2))
            if pass_num == pnum:
                files_with_numbers.append((filename, cycle, pass_num))

    files_with_numbers.sort(key=lambda x: x[1])
    return files_with_numbers

def return_nadir_files(filepath_nadir, valid_cycles, pnum):
    all_files = glob(filepath_nadir)
    nadir_files = []
    for filename in all_files:
        match = re.search(r'GPR_2PfP(\d+)_0*(\d+)', filename)
        if match:
            cycle = int(match.group(1))
            pass_num = int(match.group(2))
            if cycle in valid_cycles and pass_num == pnum:
                nadir_files.append((filename, cycle, pass_num))
    nadir_files.sort(key=lambda x: x[1])  # sort by cycle
    return nadir_files

def return_swot_files(folder, pnum, basic = True):
    
    if basic: 
        filepath = os.path.join(folder, 'SWOT_L2_LR_SSH_Basic_*.nc')
    else: 
        filepath = os.path.join(folder, 'SWOT_L2_LR_SSH_Expert_*.nc')
    
    filepath_nadir = os.path.join(folder, 'SWOT_GPR_*.nc')
    
    karin_files_with_numbers = return_karin_files(filepath, pnum, basic)
    karin_cycles = {cycle for _, cycle, _ in karin_files_with_numbers}

    nadir_files_with_numbers = return_nadir_files(filepath_nadir, karin_cycles, pnum)
    
    karin_dict = {cycle: item for item, cycle, _ in karin_files_with_numbers}
    nadir_dict = {cycle: item for item, cycle, _ in nadir_files_with_numbers}
    
    shared_cycles = sorted(set(karin_dict) & set(nadir_dict))
    
    karin_aligned = [(karin_dict[c], c) for c in shared_cycles]
    nadir_aligned = [(nadir_dict[c], c) for c in shared_cycles]

    return karin_files_with_numbers, nadir_files_with_numbers, shared_cycles, karin_aligned, nadir_aligned


def init_swot_arrays(num_shared_cycles, track_length, total_width, track_length_nadir):
    """
    Initializes arrays for SWOT KaRIn and Nadir data.
    """
    lat_karin  = np.full((num_shared_cycles, track_length, total_width), np.nan)
    lon_karin  = np.full_like(lat_karin, np.nan)
    ssha_karin = np.full_like(lat_karin, np.nan)
    time_array = np.full((num_shared_cycles, track_length), np.nan)
    ssha_nadir = np.full((num_shared_cycles, track_length_nadir), np.nan)
    lat_nadir  = np.full_like(ssha_nadir, np.nan)
    lon_nadir  = np.full_like(ssha_nadir, np.nan)
    tide       = np.full((num_shared_cycles, track_length, total_width), np.nan)

    return lat_karin, lon_karin, ssha_karin, time_array, ssha_nadir, lat_nadir, lon_nadir, tide


# Returns the indices of the track in the KaRIn file that fall within the specified latitude range.
def get_karin_track_indices(karin_file, lat_min, lat_max):
    
    if lat_min > lat_max: # I made this mistake too many times...
        print("Error: lat_max must be larger than lat_min.")
        return [], 0
    
    karin_ref = nc.Dataset(karin_file, 'r')
    fp_latitude = karin_ref['latitude']
    mid_col = fp_latitude.shape[1] // 2
    lat_center = fp_latitude[:, mid_col]
    indx = np.where((lat_center >= lat_min) & (lat_center <= lat_max))[0]
    track_length = len(indx)
    if track_length ==0: 
        print("KaRIn track length = 0, choose different sampling index")
    karin_ref.close()
    return indx, track_length

# Returns the indices of the nadir track in the Nadir file that fall within the specified latitude range.
def get_nadir_track_indices(nadir_file, lat_min, lat_max, nadir_dy=None): 
    
    nadir_ref = nc.Dataset(nadir_file, 'r')
    lats_fp = nadir_ref['data_01']['latitude'][:] 
    indxs = np.where((lats_fp >= lat_min) & (lats_fp <= lat_max))[0]
    track_length_nadir = len(indxs) 
    if track_length_nadir ==0: 
        print("Nadir track length = 0, choose different sampling index")
    nadir_ref.close()
    
    return indxs, track_length_nadir

def load_karin_data(karin_files_with_numbers, lat_min, lat_max, karin, verbose=True, dropqual=False):

    swath_width = karin.swath_width
    initial_good_strips = []
    
    # Lists to track specific cycle-level outcomes
    bad_strips_quality = []      # Tracks specific (cycle, side) tuples
    cycles_dropped_quality = []  # Tracks whole cycles dropped due to >20% bad data
    cycles_passed_quality = []   # Tracks cycles that passed the 20% check
    
    # one value per cycle
    time_cycle_num = np.full(karin.num_cycles, np.nan)
    time_cycle_dt  = np.empty(karin.num_cycles, dtype='datetime64[ns]')
    time_cycle_dt[:] = np.datetime64('NaT')

    for n, (filename, cycle) in enumerate(karin_files_with_numbers):
        try:
            with nc.Dataset(filename, 'r') as data:
                # build the along-track index 
                fp_latitude = data['latitude']
                mid_col = fp_latitude.shape[1] // 2
                lat_center = fp_latitude[:, mid_col]
                indx = np.where((lat_center >= lat_min) & (lat_center <= lat_max))[0]
                if indx.size == 0:
                    continue
            
                # --- mean time over those indices (NaNs ignored) ---
                if 'time' in data.variables:
                    tvar_nc = data.variables['time']
                    tvals   = np.asarray(tvar_nc[indx])
                    if tvals.size > 0:
                        tmean = float(np.nanmean(tvals))
                        time_cycle_num[n] = tmean
                        # Assuming cf_to_datetime64 is available in your scope
                        time_cycle_dt[n]  = cf_to_datetime64([tmean], tvar_nc)[0]
            

                # WHOLE CYCLE DROP 
                drop_cycle = False
                
                # Check both sides to see if EITHER fails the 20% threshold
                for side in [0, 1]:
                    i0 = 34 * side + 5
                    i1 = i0 + swath_width
                    qual = data['ssha_karin_2_qual'][indx, i0:i1]
                    total_pts = np.size(qual)
                    
                    # Calculate fraction of bad points
                    bad_frac = np.sum(qual != 0) / total_pts if total_pts > 0 else 1.0
                    
                    if bad_frac > 0.20:
                        drop_cycle = True
                        break # Stop checking sides, the whole cycle is dead
                
                if drop_cycle:
                    # Record the cycle as dropped
                    cycles_dropped_quality.append(cycle)
                    
                    # Mark both strips as bad for consistency in strip-level tracking
                    bad_strips_quality.append((cycle, 0))
                    bad_strips_quality.append((cycle, 1))
                    
                    if verbose:
                        print(f"KaRIn Cycle {cycle} dropped: >20% bad-quality points in at least one strip.")
                    continue  # Skip this cycle entirely, do not load into karin object

                cycles_passed_quality.append(cycle)

                # ---------------------------------------------------------
                # PROCESS DATA (Cycle Passed Pre-check)
                # ---------------------------------------------------------
                for side in [0, 1]:
                    i0 = 34 * side + 5
                    i1 = i0 + swath_width
            
                    ssha = data['ssha_karin_2'][indx, i0:i1]
                    qual = data['ssha_karin_2_qual'][indx, i0:i1]
                    xcor = data['height_cor_xover'][indx, i0:i1]
                    tide = data['internal_tide_hret'][indx, i0:i1]
                    lat  = data['latitude'][indx, i0:i1]
                    lon  = data['longitude'][indx, i0:i1]

                    karin.lat_full = np.array(data.variables['latitude'][indx,:], copy=True)
                    karin.lon_full = np.array(data.variables['longitude'][indx,:], copy=True)

                    if np.ma.is_masked(ssha) or np.ma.is_masked(xcor) or np.any(qual != 0):
                        bad_strips_quality.append((cycle, side))
                        if dropqual:
                            continue

                    ssha_combined = ssha + xcor + tide
                    ssha_masked   = np.where(qual != 0, np.nan, ssha_combined)
                    tide_masked   = np.where(qual != 0, np.nan, tide)

                    j_slice = slice(0, swath_width) if side == 0 else slice(-swath_width, None)

                    if karin.ssh.ndim == 3 and karin.ssh.shape[1] == 1:
                        karin.lat[n, 0, j_slice]  = lat
                        karin.lon[n, 0, j_slice]  = lon
                        karin.ssh[n, 0, j_slice]  = ssha_masked
                        karin.tide[n, 0, j_slice] = tide_masked
                    else:
                        karin.lat[n, :, j_slice]  = lat
                        karin.lon[n, :, j_slice]  = lon
                        karin.ssh[n, :, j_slice]  = ssha_masked
                        karin.tide[n, :, j_slice] = tide_masked

                    initial_good_strips.append((cycle, side))
                    karin.lon_min = np.nanmin(karin.lon[n, :, :])
                    karin.lon_max = np.nanmax(karin.lon[n, :, :])

        except FileNotFoundError:
            print(f"Error: File not found {filename}")
        except Exception as e:
            print(f"Error processing file {filename}, cycle {cycle}: {e}")

    # --- variance filtering & summary ---
    karin.good_strips_list = np.empty((0, 2))
    karin.removed_strips_high_variance = []

    if not initial_good_strips:
        print("No strips passed initial quality checks. Skipping variance-based filtering.")
        outlier_n_indices = np.array([], dtype=int)
        karin.good_strips_list = np.array(initial_good_strips)
    else:
        threshold = 10.0
        ssha_array = np.copy(karin.ssh)
        varts = np.nanvar(ssha_array, axis=(1, 2))
        
        if np.all(np.isnan(varts)) or varts.size == 0:
            print("Warning: All per-file variances are NaN or no data. Skipping high-variance removal.")
            outlier_n_indices = np.array([], dtype=int)
            karin.good_strips_list = np.array(initial_good_strips)
        else:
            overall_var = np.nanvar(ssha_array)
            if np.isnan(overall_var) or overall_var == 0:
                print(f"Warning: Overall SSH variance is {overall_var}.")
                outlier_n_indices = np.array([], dtype=int)
            else:
                outlier_n_indices = np.where(varts > threshold * overall_var)[0]
            
            print(f"Overall SSH variance (overall_var): {overall_var}")
            
            # Map outlier indices back to Cycle numbers
            outlier_cycle_numbers = set()
            if len(outlier_n_indices) > 0:
                print(f"File indices with outlier variance: {outlier_n_indices}")
                for n_idx in outlier_n_indices:
                    if 0 <= n_idx < len(karin_files_with_numbers):
                        outlier_cycle_numbers.add(karin_files_with_numbers[n_idx][1])
                print(f"High variance cycle numbers: {outlier_cycle_numbers}")

            initial_good_strips_np = np.array(initial_good_strips)
            
            if initial_good_strips_np.size > 0 and outlier_cycle_numbers:
                strip_cycles = initial_good_strips_np[:, 0].astype(int)
                is_high_var = np.isin(strip_cycles, list(outlier_cycle_numbers))
                karin.good_strips_list = initial_good_strips_np[~is_high_var]
                
                removed_temp = initial_good_strips_np[is_high_var]
                karin.removed_strips_high_variance = sorted([tuple(map(int, row)) for row in removed_temp])
            else:
                karin.good_strips_list = initial_good_strips_np

    if 'outlier_n_indices' in locals() and len(outlier_n_indices) > 0:
        for n_idx in outlier_n_indices:
            if 0 <= n_idx < karin.ssh.shape[0]:
                karin.ssh[n_idx, :, :] = np.nan

    karin.bad_strips_quality = sorted(list(set(bad_strips_quality)))

    print(f"----------------------------------")
    print(f"Total Number of Good KaRIn strips : {len(karin.good_strips_list)}")
    print(f"Number of Quality Masked KaRIn strips : {len(karin.bad_strips_quality)}")
    print(f"Number of High Variance strips removed : {len(karin.removed_strips_high_variance)}")
    print(f"Number of Good Cycles: {len(cycles_passed_quality)}")
    print(f"Number of Cycles dropped (>20% masked): {len(cycles_dropped_quality)}")
    print(f"----------------------------------\n")

    if verbose:
        print("Good strips (cycle, side):")
    good_cycles = []
    if karin.good_strips_list.size > 0:
        for c, s in karin.good_strips_list:
            if verbose:
                print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
            good_cycles.append(c)

    if verbose:
        print("\nHigh variance strips removed (cycle, side):")
    hvar_cycles = []
    if karin.removed_strips_high_variance:
        for c, s in karin.removed_strips_high_variance:
            if verbose:
                print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
            hvar_cycles.append(c)

    if verbose:
        print("\nStrips failing quality checks (includes dropped cycles):")
    bad_cycles_from_strips = []
    if karin.bad_strips_quality:
        for c, s in karin.bad_strips_quality:
            if verbose:
                print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
            bad_cycles_from_strips.append(c)

    # Final sets
    karin.good_cycles = sorted(set(good_cycles))
    karin.bad_cycles  = sorted(set(bad_cycles_from_strips))
    karin.hvar_cycles = sorted(set(hvar_cycles))
    
    karin.cycles_dropped_quality = sorted(cycles_dropped_quality)
    karin.cycles_passed_quality = sorted(cycles_passed_quality)

    summary = {
        "good_strips": [tuple(map(int, x)) for x in karin.good_strips_list.tolist()] if karin.good_strips_list.size > 0 else [],
        "removed_high_variance": karin.removed_strips_high_variance,
        "other_strips_quality_issues": karin.bad_strips_quality,
        "cycles_dropped_quality": karin.cycles_dropped_quality,
        "cycles_passed_quality": karin.cycles_passed_quality
    }

    # store one-per-cycle times (mean over selected along-track indices)
    karin.time = time_cycle_num
    karin.time_dt  = time_cycle_dt

    return summary

# def load_karin_data(karin_files_with_numbers, lat_min, lat_max, karin, verbose=True, dropqual=False):

#     swath_width = karin.swath_width
#     initial_good_strips = []
#     bad_strips_quality = []

#     # one value per cycle
#     time_cycle_num = np.full(karin.num_cycles, np.nan)
#     time_cycle_dt  = np.empty(karin.num_cycles, dtype='datetime64[ns]')
#     time_cycle_dt[:] = np.datetime64('NaT')

#     for n, (filename, cycle) in enumerate(karin_files_with_numbers):
#         try:
#             with nc.Dataset(filename, 'r') as data:
#                 # build the along-track index 
#                 fp_latitude = data['latitude']
#                 mid_col = fp_latitude.shape[1] // 2
#                 lat_center = fp_latitude[:, mid_col]
#                 indx = np.where((lat_center >= lat_min) & (lat_center <= lat_max))[0]
#                 if indx.size == 0:
#                     continue
            
#                 # --- mean time over those indices (NaNs ignored) ---
#                 if 'time' in data.variables:
#                     tvar_nc = data.variables['time']
#                     tvals   = np.asarray(tvar_nc[indx])
#                     if tvals.size > 0:
#                         tmean = float(np.nanmean(tvals))
#                         time_cycle_num[n] = tmean
#                         time_cycle_dt[n]  = cf_to_datetime64([tmean], tvar_nc)[0]
            
#                 # ---- pre-check quality for both swaths ----
#                 drop_cycle = False
#                 for side in [0, 1]:
#                     i0 = 34 * side + 5
#                     i1 = i0 + swath_width
#                     qual = data['ssha_karin_2_qual'][indx, i0:i1]
#                     total_pts = np.size(qual)
#                     bad_frac = np.sum(qual != 0) / total_pts if total_pts > 0 else 1.0
#                     if bad_frac > 0.20:
#                         drop_cycle = True # drop the entire cycle if theres too many bad qual data
            
#                 if drop_cycle:
#                     # record both strips as bad
#                     bad_strips_quality.append((cycle, 0))
#                     bad_strips_quality.append((cycle, 1))
#                     print(f"KaRIn Cycle {cycle} dropped: >20% bad-quality points")
#                     continue   # skip this cycle entirely

#                 # ---- process normally if not dropped ----
#                 for side in [0, 1]:
#                     i0 = 34 * side + 5
#                     i1 = i0 + swath_width
            
#                     ssha = data['ssha_karin_2'][indx, i0:i1]
#                     qual = data['ssha_karin_2_qual'][indx, i0:i1]
#                     xcor = data['height_cor_xover'][indx, i0:i1]
#                     tide = data['internal_tide_hret'][indx, i0:i1]
#                     lat  = data['latitude'][indx, i0:i1]
#                     lon  = data['longitude'][indx, i0:i1]

#                     karin.lat_full = np.array(data.variables['latitude'][indx,:], copy=True)
#                     karin.lon_full = np.array(data.variables['longitude'][indx,:], copy=True)

#                     # quality
#                     if np.ma.is_masked(ssha) or np.ma.is_masked(xcor) or np.any(qual != 0):
#                         bad_strips_quality.append((cycle, side))
#                         if dropqual:
#                             continue

#                     ssha_combined = ssha + xcor + tide
#                     ssha_masked   = np.where(qual != 0, np.nan, ssha_combined)
#                     tide_masked   = np.where(qual != 0, np.nan, tide)

#                     j_slice = slice(0, swath_width) if side == 0 else slice(-swath_width, None)

#                     if karin.ssh.ndim == 3 and karin.ssh.shape[1] == 1:
#                         karin.lat[n, 0, j_slice]  = lat
#                         karin.lon[n, 0, j_slice]  = lon
#                         karin.ssh[n, 0, j_slice]  = ssha_masked
#                         karin.tide[n, 0, j_slice] = tide_masked
#                     else:
#                         karin.lat[n, :, j_slice]  = lat
#                         karin.lon[n, :, j_slice]  = lon
#                         karin.ssh[n, :, j_slice]  = ssha_masked
#                         karin.tide[n, :, j_slice] = tide_masked

#                     initial_good_strips.append((cycle, side))
#                     karin.lon_min = np.nanmin(karin.lon[n, :, :])
#                     karin.lon_max = np.nanmax(karin.lon[n, :, :])

#         except FileNotFoundError:
#             print(f"Error: File not found {filename}")
#         except Exception as e:
#             print(f"Error processing file {filename}, cycle {cycle}: {e}")

#     # --- variance filtering & summary ---
#     karin.good_strips_list = np.empty((0, 2))
#     karin.removed_strips_high_variance = []

#     if not initial_good_strips:
#         print("No strips passed initial quality checks. Skipping variance-based filtering.")
#         outlier_n_indices = np.array([], dtype=int)
#         karin.good_strips_list = np.array(initial_good_strips)
#     else:
#         threshold = 10.0
#         ssha_array = np.copy(karin.ssh)
#         varts = np.nanvar(ssha_array, axis=(1, 2))
#         if np.all(np.isnan(varts)) or varts.size == 0:
#             print("Warning: All per-file variances are NaN or no data. Skipping high-variance removal.")
#             outlier_n_indices = np.array([], dtype=int)
#             karin.good_strips_list = np.array(initial_good_strips)
#         else:
#             overall_var = np.nanvar(ssha_array)
#             if np.isnan(overall_var) or overall_var == 0:
#                 print(f"Warning: Overall SSH variance is {overall_var}.")
#                 outlier_n_indices = np.array([], dtype=int)
#             else:
#                 outlier_n_indices = np.where(varts > threshold * overall_var)[0]
#             print(f"Overall SSH variance (overall_var): {overall_var}")
#             print(f"File indices with outlier variance: {outlier_n_indices}")

#             initial_good_strips_np = np.array(initial_good_strips)
#             if len(outlier_n_indices) > 0:
#                 outlier_cycle_numbers = set()
#                 for n_idx in outlier_n_indices:
#                     if 0 <= n_idx < len(karin_files_with_numbers):
#                         outlier_cycle_numbers.add(karin_files_with_numbers[n_idx][1])
#                 print(f"High variance cycle numbers: {outlier_cycle_numbers}")

#                 if initial_good_strips_np.size > 0 and outlier_cycle_numbers:
#                     strip_cycles = initial_good_strips_np[:, 0].astype(int)
#                     is_high_var = np.isin(strip_cycles, list(outlier_cycle_numbers))
#                     karin.good_strips_list = initial_good_strips_np[~is_high_var]
#                     removed_temp = initial_good_strips_np[is_high_var]
#                     karin.removed_strips_high_variance = sorted([tuple(map(int, row)) for row in removed_temp])
#                 else:
#                     karin.good_strips_list = initial_good_strips_np
#             else:
#                 karin.good_strips_list = initial_good_strips_np

#     if 'outlier_n_indices' in locals() and len(outlier_n_indices) > 0:
#         for n_idx in outlier_n_indices:
#             if 0 <= n_idx < karin.ssh.shape[0]:
#                 karin.ssh[n_idx, :, :] = np.nan

#     karin.bad_strips_quality = sorted(list(set(bad_strips_quality)))

#     print(f"----------------------------------")
#     print(f"Total Number of Good KaRIn strips : {len(karin.good_strips_list)}")
#     print(f"Number of Quality Masked KaRIn strips : {len(karin.bad_strips_quality)}")
#     print(f"Number of High Variance strips removed : {len(karin.removed_strips_high_variance)}\n")

#     if verbose:
#         print("Good strips (cycle, side):")
#     good_cycles = []
#     if karin.good_strips_list.size > 0:
#         for c, s in karin.good_strips_list:
#             if verbose:
#                 print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
#             good_cycles.append(c)

#     if verbose:
#         print("\nHigh variance strips removed (cycle, side):")
#     hvar_cycles = []
#     if karin.removed_strips_high_variance:
#         for c, s in karin.removed_strips_high_variance:
#             if verbose:
#                 print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
#             hvar_cycles.append(c)

#     if verbose:
#         print("\nStrips failing initial quality checks (cycle, side):")
#     bad_cycles = []
#     if karin.bad_strips_quality:
#         for c, s in karin.bad_strips_quality:
#             if verbose:
#                 print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
#             bad_cycles.append(c)

#     summary = {
#         "good_strips": [tuple(map(int, x)) for x in karin.good_strips_list.tolist()] if karin.good_strips_list.size > 0 else [],
#         "removed_high_variance": karin.removed_strips_high_variance,
#         "other_strips_quality_issues": karin.bad_strips_quality,
#     }

#     karin.good_cycles = sorted(set(good_cycles))
#     karin.bad_cycles  = sorted(set(bad_cycles))
#     karin.hvar_cycles = sorted(set(hvar_cycles))

#     # store one-per-cycle times (mean over selected along-track indices)
#     karin.time = time_cycle_num
#     karin.time_dt  = time_cycle_dt

#     return summary


def process_karin_data(karin, cutoff_m=100e3, delta=2e3):
    '''Processes the KaRIn data to remove the high-pass filtered time-mean and return SSH anomalies'''
    swath_width = karin.swath_width
    middle_width = karin.middle_width 
    ssh_karin = karin.ssh
    
    ssh_karin[:, :, swath_width:swath_width + middle_width] = np.nan # Fill middle section with NaN
    ssha_karin_arr = np.asarray(ssh_karin, dtype=float)
    nan_mask = np.isnan(ssha_karin_arr)
    ssha_karin_arr[nan_mask] = 0.0 # replace nans with 0.0 for the filter step
    ssh_mean = np.nanmean(ssha_karin_arr, axis=0)

    # Apply Gaussian filter to time mean
    sigma_m = cutoff_m / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_m / delta
    lowpass = gaussian_filter(ssh_mean, sigma=(sigma_pixels, sigma_pixels), mode='reflect')
    lowpass[:, swath_width:swath_width + middle_width] = np.nan
    
    ssh_mean_highpass = ssh_mean - lowpass
    ssha = ssh_karin - ssh_mean_highpass[None, :, :]
    ssha[nan_mask] = np.nan  # Restore NaN mask in anomalies
    karin.ssh_mean = ssh_mean
    karin.ssha_mean_highpass = ssh_mean_highpass
    karin.ssha = ssha
    return 

def load_nadir_data(nadir_files_with_numbers, lat_min, lat_max, nadir):

    num_good_cycles = 0
    num_bad_cycles  = 0

    time_cycle_num = np.full(nadir.num_cycles, np.nan)
    time_cycle_dt  = np.empty(nadir.num_cycles, dtype='datetime64[ns]')
    time_cycle_dt[:] = np.datetime64('NaT')

    for n, (filename, cycle) in enumerate(nadir_files_with_numbers):
        data = nc.Dataset(filename, 'r')
        try:
            group = data['data_01']
            lats  = group['latitude'][:]
            lons  = group['longitude'][:]
            indxs = np.where((lats >= lat_min) & (lats <= lat_max))[0]
            if indxs.size == 0:
                num_bad_cycles += 1
                continue

            # --- mean time over exactly those indices (NaNs ignored) ---
            if 'time' in group.variables:
                tvar_nc = group.variables['time']
                tvals   = np.asarray(tvar_nc[indxs])
                if tvals.size > 0:
                    tmean = float(np.nanmean(tvals))
                    time_cycle_num[n] = tmean
                    time_cycle_dt[n]  = cf_to_datetime64([tmean], tvar_nc)[0]

            if len(group['ku']['ssha']) <= indxs[-1]:
                num_bad_cycles += 1
                continue

            tide = group['internal_tide_hret'][indxs]
            ssha = group['ku']['ssha'][indxs] + tide

            # --- Implement the 20% rule for bad quality points ---
            total_pts = ssha.size
            if total_pts > 0:
                num_bad_pts = np.sum(ssha.mask)
                bad_frac = num_bad_pts / total_pts
                if bad_frac > 0.20:
                    # This print statement is optional but helpful for transparency
                    print(f"Nadir Cycle {cycle} dropped: >20% bad-quality nadir points ({bad_frac:.2%}).")
                    num_bad_cycles += 1
                    continue

            num_good_cycles += 1
            n_valid = min(nadir.track_length, len(indxs))
            if n_valid == 0:
                print(f"Warning: No valid indices found for cycle {cycle} in file {filename}.")

            nadir.lat[n, :n_valid] = lats[indxs[:n_valid]]
            nadir.lon[n, :n_valid] = lons[indxs[:n_valid]]
            nadir.ssh[n, :n_valid] = ssha[:n_valid].filled(np.nan)

        finally:
            data.close()

    # store one-per-cycle times (mean over selected along-track indices)
    nadir.time = time_cycle_num
    nadir.time_dt  = time_cycle_dt

    print(f"Number of good nadir cycles: {num_good_cycles}")
    print(f"Number of bad nadir cycles: {num_bad_cycles}")
    return num_good_cycles, num_bad_cycles


def process_nadir_data(nadir, cutoff_m=100e3, delta=6.8e3, ):
    ssha_nadir = nadir.ssh
    ssha_mean_nadir = np.nanmean(ssha_nadir, axis=0)
    
    # Apply Gaussian filter to the time mean as for the karin data
    sigma_m = cutoff_m / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma_m / delta
    lowpass = gaussian_filter(ssha_mean_nadir, sigma=(sigma_pixels), mode='reflect')
    
    # Calculate high-pass residual and anomalies
    ssha_highpass_nadir = ssha_mean_nadir - lowpass
    ssha_anom_nadir = ssha_nadir - ssha_mean_nadir # We dont remove the time mean from the nadir because its large scale signal
    nadir.ssh_mean = ssha_mean_nadir
    nadir.ssha_mean_highpass = ssha_highpass_nadir
    nadir.ssha = ssha_anom_nadir 
    return

def remove_outlier_strips(ssha_array, good_strips_list, threshold=10):
    """
    Remove strips from good_strips_list where the variance of the corresponding ssha_array slice 
    is an outlier (> threshold * overall nanvar). From Xihans code
    """

    varts = np.nanvar(ssha_array, axis=(0, 1,2))
    overall_var = np.nanvar(ssha_array)
    outlierindx = np.where(varts > threshold * overall_var)[0]

    # Mark outliers in good_strips_list
    for outlier in outlierindx:
        indcom = np.where(good_strips_list[:,0]==outlier)[0]
        good_strips_list[indcom,0] = 1e3  # marker

    # Remove marked outliers
    badstripind = np.where(good_strips_list[:,0]==1e3)[0]
    good_strips_list_clean = np.delete(good_strips_list, badstripind, axis=0)
    num_useful_strips = good_strips_list_clean.shape[0]

    return good_strips_list_clean, num_useful_strips

def cf_to_datetime64(vals, tvar):
    """
    Convert CF-compliant numeric time values (1D) to numpy datetime64[ns].
    vals: array-like numeric times
    tvar: netCDF Variable that has 'units' and (optionally) 'calendar'
    """
    units = getattr(tvar, 'units', 'seconds since 2000-01-01 00:00:00')
    cal   = getattr(tvar, 'calendar', 'standard')
    dt = num2date(np.asarray(vals), units=units, calendar=cal)
    return np.array(dt, dtype='datetime64[ns]')


