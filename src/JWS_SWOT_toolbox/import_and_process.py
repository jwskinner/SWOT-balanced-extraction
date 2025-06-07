
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
    karin_ref.close()
    return indx, track_length

# Returns the indices of the nadir track in the Nadir file that fall within the specified latitude range.
def get_nadir_track_indices(nadir_file, lat_min, lat_max, nadir_dy=None): 
    
    nadir_ref = nc.Dataset(nadir_file, 'r')
    lats_fp = nadir_ref['data_01']['latitude'][:] 
    indxs = np.where((lats_fp >= lat_min) & (lats_fp <= lat_max))[0]
    track_length_nadir = len(indxs) 
    nadir_ref.close()
    
    return indxs, track_length_nadir

# Loads the KaRIn data for the specified cycles and processes it to extract SSH anomalies.
def load_karin_data_old(karin_files_with_numbers, indx, karin):
    swath_width = karin.swath_width
    good_strips_list = []
    bad_strips_list = []
    num_useful_strips = 0
    num_bad_strips = 0

    for n, (filename, cycle) in enumerate(karin_files_with_numbers):
  
        data = nc.Dataset(filename, 'r')

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
     
            # Removes bad sides completely
            if np.ma.is_masked(ssha) or np.ma.is_masked(xcor) or np.any(qual):
                num_bad_strips += 1
                bad_strips_list.append((cycle, side))
                #continue
            
            # Masks bad data points and leaves the rest 
            ssha_masked = np.where(qual, np.nan, ssha + xcor - tide) # mask the poor quality data
            tide_masked = np.where(qual, np.nan, tide) # mask the poor quality data

            
            j = slice(0, swath_width) if side == 0 else slice(-swath_width, None)
            karin.lat[n, :, j] = lat
            karin.lon[n, :, j] = lon
            karin.ssh[n, :, j] = ssha_masked
            karin.tide[n, :, j] = tide_masked
            num_useful_strips += 1
            good_strips_list.append((cycle, side))

    # After loop and loading all strips
    threshold = 10
    ssha_array = np.copy(karin.ssh)
    varts = np.nanvar(ssha_array, axis=(1,2)) 
    overall_var = np.nanvar(ssha_array)
    print(varts)
    print(overall_var)
    outlierindx = np.where(varts > threshold * overall_var)[0]
    print(np.where(varts > threshold * overall_var)[0])
    
    good_strips_list_np = np.array(good_strips_list)
    print(good_strips_list_np.shape)
    print(karin.ssh.shape)
    
    if len(outlierindx) > 0:
        mask = ~np.isin(good_strips_list_np[:, 0], outlierindx)
        good_strips_list_clean = good_strips_list_np[mask]
    
        print(f"Removed {len(good_strips_list_np) - len(good_strips_list_clean)} rows")
        print(f"Original length: {len(good_strips_list_np)}, New length: {len(good_strips_list_clean)}")
    
    else:
        good_strips_list_clean = good_strips_list_np
        print("No outliers found")

    print(f"Final array shape: {good_strips_list_clean.shape}")
    
    karin.good_strips_list = good_strips_list_clean
    karin.removed_strips = [tuple(good_strips_list_np[i]) for i in to_remove] if len(outlierindx) > 0 else []
    karin.num_useful_strips = len(karin.good_strips_list)
    data.close()

    print(f"----------------------------------")
    print(f"Total Number of Good KaRIn strips : {len(good_strips_list_clean)}")
    print(f"Number of Quality Masked KaRIn strips        : {len(bad_strips_list)}")
    print(f"Number of High Variance strips    : {len(karin.removed_strips)}\n")

    print("Good strips (cycle, side):")
    for c, s in good_strips_list_clean:
        print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")

    print("\nHigh variance strips removed (cycle, side):")
    for c, s in sorted(karin.removed_strips):
        print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")

    print("\nOther strips (cycle, side):")
    for c, s in bad_strips_list:
        print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")

    # Optionally, return as dict for downstream use
    summary = {
        "good_strips": [tuple(map(int, x)) for x in good_strips_list_clean],
        "removed_high_variance": [tuple(map(int, x)) for x in karin.removed_strips],
        "other_strips": [tuple(map(int, x)) for x in bad_strips_list],
    }

def load_karin_data(karin_files_with_numbers, indx, karin):

    swath_width = karin.swath_width
    initial_good_strips = []         
    bad_strips_quality_issues = []   

    for n, (filename, cycle) in enumerate(karin_files_with_numbers):
        try:
            with nc.Dataset(filename, 'r') as data:
                for side in [0, 1]:  # 0 for left, 1 for right
                    i0 = 34 * side + 5
                    i1 = i0 + swath_width

                    ssha = data['ssha_karin_2'][indx, i0:i1]
                    qual = data['ssha_karin_2_qual'][indx, i0:i1]
                    xcor = data['height_cor_xover'][indx, i0:i1]
                    # xqual = data['height_cor_xover_qual'][indx, i0:i1] # Not directly used in ssha_masked below
                    tide = data['internal_tide_hret'][indx, i0:i1]
                    lat = data['latitude'][indx, i0:i1]
                    lon = data['longitude'][indx, i0:i1]
                    # tvar = data['time'][indx] # Not used in the provided snippet logic

                    # Initial quality check for the entire side.
                    if np.ma.is_masked(ssha) or np.ma.is_masked(xcor) or np.any(qual != 0):
                        bad_strips_quality_issues.append((cycle, side))
                        continue

                    # Mask bad data points (where qual is non-zero)
                    ssha_combined = ssha + xcor - tide
                    ssha_masked = np.where(qual != 0, np.nan, ssha_combined)
                    tide_masked = np.where(qual != 0, np.nan, tide) # Mask tide similarly

                    j_slice = slice(0, swath_width) if side == 0 else slice(-swath_width, None)

                    # Assign data to karin
                    if karin.ssh.ndim == 3 and karin.ssh.shape[1] == 1:
                        karin.lat[n, 0, j_slice] = lat
                        karin.lon[n, 0, j_slice] = lon
                        karin.ssh[n, 0, j_slice] = ssha_masked
                        karin.tide[n, 0, j_slice] = tide_masked
                    else:
                        karin.lat[n, :, j_slice] = lat
                        karin.lon[n, :, j_slice] = lon
                        karin.ssh[n, :, j_slice] = ssha_masked
                        karin.tide[n, :, j_slice] = tide_masked

                    initial_good_strips.append((cycle, side))
        except FileNotFoundError:
            print(f"Error: File not found {filename}")
        except Exception as e:
            print(f"Error processing file {filename}, cycle {cycle}, side {side}: {e}")


    # --- Variance-based outlier removal from Xihan ---
    karin.good_strips_list = np.empty((0, 2)) # Initialize as empty numpy array
    karin.removed_strips_high_variance = []   # Initialize as empty list

    if not initial_good_strips:
        print("No strips passed initial quality checks. Skipping variance-based filtering.")
    else:
        threshold = 10.0
        ssha_array = np.copy(karin.ssh) 
        varts = np.nanvar(ssha_array, axis=(1, 2))

        if np.all(np.isnan(varts)) or varts.size == 0:
            print("Warning: All per-file variances are NaN or no data. Skipping high-variance removal.")
            karin.good_strips_list = np.array(initial_good_strips)
        else:
            overall_var = np.nanvar(ssha_array) 

            outlier_n_indices = np.array([], dtype=int) 
            if np.isnan(overall_var) or overall_var == 0:
                print(f"Warning: Overall SSH variance is {overall_var}.")
            else:
                outlier_n_indices = np.where(varts > threshold * overall_var)[0]
            
            print(f"Overall SSH variance (overall_var): {overall_var}")
            print(f"File indices with outlier variance: {outlier_n_indices}")

            initial_good_strips_np = np.array(initial_good_strips)

            if len(outlier_n_indices) > 0:
                outlier_cycle_numbers = set()
                for n_idx in outlier_n_indices:
                    if 0 <= n_idx < len(karin_files_with_numbers):
                        outlier_cycle_numbers.add(karin_files_with_numbers[n_idx][1])
                    else:
                        print(f"Warning: outlier_n_index {n_idx} is out of bounds for karin_files_with_numbers.")
                
                print(f"High variance cycle numbers: {outlier_cycle_numbers}")

                if initial_good_strips_np.size > 0 and outlier_cycle_numbers:
                    strip_cycles = initial_good_strips_np[:, 0].astype(type(next(iter(outlier_cycle_numbers))))
                    is_high_variance_strip_mask = np.isin(strip_cycles, list(outlier_cycle_numbers))
                    
                    karin.good_strips_list = initial_good_strips_np[~is_high_variance_strip_mask]
                    removed_strips_temp = initial_good_strips_np[is_high_variance_strip_mask]
                    karin.removed_strips_high_variance = sorted([tuple(map(int, row)) for row in removed_strips_temp])
                else: 
                    karin.good_strips_list = initial_good_strips_np
            else: 
                karin.good_strips_list = initial_good_strips_np
    
    if len(outlier_n_indices) > 0: # Set all the high variance outliers to NaN
        for n_idx in outlier_n_indices:
            if 0 <= n_idx < karin.ssh.shape[0]: # Boundary check
                karin.ssh[n_idx, :, :] = np.nan
    
    karin.bad_strips_quality_issues = sorted(list(set(bad_strips_quality_issues)))
    karin.num_useful_strips = len(karin.good_strips_list)

    # --- Summary ---
    print(f"----------------------------------")
    print(f"Total Number of Good KaRIn strips : {karin.num_useful_strips}")
    print(f"Number of Quality Masked KaRIn strips : {len(karin.bad_strips_quality_issues)}")
    print(f"Number of High Variance strips removed : {len(karin.removed_strips_high_variance)}\n")

    print("Good strips (cycle, side):")
    if karin.good_strips_list.size > 0:
        for c, s in karin.good_strips_list: # It's already a numpy array
            print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
    else:
        print("  No good strips.")

    print("\nHigh variance strips removed (cycle, side):")
    if karin.removed_strips_high_variance: # This is a list of tuples
        for c, s in karin.removed_strips_high_variance: # Already sorted
            print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
    else:
        print("  No high variance strips removed.")

    print("\nStrips failing initial quality checks (cycle, side):")
    if karin.bad_strips_quality_issues: # This is a list of tuples
        for c, s in karin.bad_strips_quality_issues: # Already sorted
            print(f"  - Cycle: {int(c):4d}, Side: {int(s)}")
    else:
        print("  No strips failed initial quality checks.")

    summary = {
        "good_strips": [tuple(map(int, x)) for x in karin.good_strips_list.tolist()] if karin.good_strips_list.size > 0 else [],
        "removed_high_variance": karin.removed_strips_high_variance, # Already list of int tuples
        "other_strips_quality_issues": karin.bad_strips_quality_issues, # Already list of int tuples
    }

    return summary

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

def load_nadir_data(nadir_files_with_numbers, indxs, nadir):
    num_good_cycles = 0
    num_bad_cycles = 0
    
    for n, (filename, cycle) in enumerate(nadir_files_with_numbers):
        data = nc.Dataset(filename, 'r')
        group = data['data_01']
        
        lats = group['latitude'][:]
        lons = data['data_01']['longitude'][:]
        
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
        n_valid = min(nadir.track_length, len(indxs))
        nadir.lat[n, :n_valid] = lats[indxs[:n_valid]]
        nadir.lon[n, :n_valid] = lons[indxs[:n_valid]]
        nadir.ssh[n, :n_valid] = ssha[:n_valid].filled(np.nan)
        data.close()
    
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



