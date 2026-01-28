import netCDF4 as nc
import numpy as np
from glob import glob
import re
import os
from scipy.ndimage import gaussian_filter
from netCDF4 import num2date
from scipy.interpolate import interp1d

def cf_to_datetime64(vals, tvar):
    """
    Convert numeric time values (1D) to numpy datetime64[ns] using Level 3 time format 
    """
    units = getattr(tvar, 'units', 'seconds since 2000-01-01 00:00:00')
    cal   = getattr(tvar, 'calendar', 'standard')
    dt = num2date(np.asarray(vals), units=units, calendar=cal)
    return np.array(dt, dtype='datetime64[ns]')

def return_swot_l3_files(folder, pnum, basic=True, phase="science"):
    """
    Finds L3 files and aligns Karin/Nadir (which are in the same file).
    """
    prefix = 'SWOT_L3_LR_SSH_Basic' if basic else 'SWOT_L3_LR_SSH_Expert'
    filepath = os.path.join(folder, f'{prefix}_*.nc')
    all_files = glob(filepath)
    
    files_with_numbers = []
    for filename in all_files:
        # Match Cycle and Pass from L3 naming: ...Basic_007_048...
        match = re.search(r'(?:Basic|Expert)_(\d+)_(\d+)', filename)
        if match:
            cycle = int(match.group(1))
            if phase == "science" and cycle >= 100:
                continue  # skip cycles >= 100
            pass_num = int(match.group(2))
            if pass_num == pnum:
                files_with_numbers.append((filename, cycle, pass_num))
    
    files_with_numbers.sort(key=lambda x: x[1])
    shared_cycles = [f[1] for f in files_with_numbers]
    
    # In L3, Karin and Nadir files are identical
    karin_aligned = [(f[0], f[1]) for f in files_with_numbers]
    nadir_aligned = karin_aligned 
    
    print(f"L3 Shared Cycles: {shared_cycles}")
    return files_with_numbers, shared_cycles, karin_aligned, nadir_aligned

def get_l3_indices(sample_file, lat_min, lat_max):
    """
    Gets the indices for KaRIn and Nadir between two lats using 1D variables.
    """
    with nc.Dataset(sample_file, 'r') as ds:
        # --- 1. KaRIn Track ---
        lats_2d = ds.variables['latitude'][:]
        grid_width = lats_2d.shape[1]
        mid_col = grid_width // 2
        lat_center = lats_2d[:, mid_col]
        indx = np.where((lat_center >= lat_min) & (lat_center <= lat_max))[0]
        track_len = len(indx)
        
        # --- 2. Nadir Track ---
        rows = ds.variables['i_num_line'][:]
        cols = ds.variables['i_num_pixel'][:]
            
        # Mask invalid indices before using them
        if np.ma.is_masked(rows):
            valid_mask = ~rows.mask
            rows = rows[valid_mask]
            cols = cols[valid_mask]
            
        nadir_lats = lats_2d[rows, cols]

        # Filter nadir data within lat bounds
        nad_indx = np.where((nadir_lats >= lat_min) & (nadir_lats <= lat_max))[0]
        nadir_pt_count = len(nad_indx)

        #print(nadir_lats[nad_indx])
        
        print(f"KaRIn Rows: {track_len}")
        print(f"Valid Nadir Points: {nadir_pt_count}")
        
        return indx, track_len, grid_width, nadir_pt_count, nad_indx

def load_l3_data(aligned_files, indx, karin_obj, nadir_obj, lat_min, lat_max, ssh_key='ssha_unfiltered'):
    swath_width = karin_obj.swath_width
    
    if len(indx) == 0:
        return

    # KaRIn slices (Static)
    idx_slice = slice(indx[0], indx[-1] + 1)
    i0s = [5, 39]
    i1s = [5 + swath_width, 39 + swath_width]
    j_slices = [slice(0, swath_width), slice(-swath_width, None)]

    karin_time_list = []

    for n, (filename, cycle) in enumerate(aligned_files):
        try:
            with nc.Dataset(filename, 'r') as ds:
                lat_var = ds.variables['latitude']
                lon_var = ds.variables['longitude']
                time_var = ds.variables['time'][:][indx]
                karin_time_list.append(np.nanmean(time_var))
                
                ssh_var = ds.variables[ssh_key]

                # --- 1. KaRIn (Fast Block Read) ---
                for side in (0, 1):
                    lats = lat_var[idx_slice, i0s[side]:i1s[side]]
                    lons = lon_var[idx_slice, i0s[side]:i1s[side]]
                    ssh  = ssh_var[idx_slice, i0s[side]:i1s[side]]
                    
                    # Handle Descending Passes (KaRIn points in North-South direction)
                    # If the 0th index is higher latitude than the last index, 
                    # satellite moving N->S. We flip to enforce S->N (Low->High).
                    if lats.shape[0] > 1 and np.nanmean(lats[0, :]) > np.nanmean(lats[-1, :]):
                        lats = np.flip(lats, axis=0)
                        lons = np.flip(lons, axis=0)
                        ssh  = np.flip(ssh, axis=0)
                    
                    ssh[np.abs(ssh) > 1e4] = np.nan
                    karin_obj.lat[n, :, j_slices[side]] = lats
                    karin_obj.lon[n, :, j_slices[side]] = lons
                    karin_obj.ssh[n, :, j_slices[side]] = ssh
                    

                # --- 2. Nadir (Optimized Bounding Box) ---
                rows = ds.variables['i_num_line'][:]
                cols = ds.variables['i_num_pixel'][:]
                
                # Filter invalid indices
                if np.ma.is_masked(rows) or np.ma.is_masked(cols):
                    row_mask = rows.mask if np.ma.is_masked(rows) else np.zeros(rows.shape, dtype=bool)
                    col_mask = cols.mask if np.ma.is_masked(cols) else np.zeros(cols.shape, dtype=bool)
                    valid = ~(row_mask | col_mask)
                    rows = rows[valid]
                    cols = cols[valid]

                if len(rows) > 0:
                    # A. Define Bounding Box (Tall, Narrow Rectangle)
                    # This reads the whole track range in 1 fast sequential read
                    r_min, r_max = rows.min(), rows.max()
                    c_min, c_max = cols.min(), cols.max()

                    # B. Read Data Blocks
                    lat_block = lat_var[r_min:r_max+1, c_min:c_max+1]
                    lon_block = lon_var[r_min:r_max+1, c_min:c_max+1]
                    ssh_block = ssh_var[r_min:r_max+1, c_min:c_max+1]

                    # C. Extract Track Points from Block (In Memory)
                    local_rows = rows - r_min
                    local_cols = cols - c_min
                    
                    track_lats = lat_block[local_rows, local_cols]
                    track_lons = lon_block[local_rows, local_cols]
                    track_ssh  = ssh_block[local_rows, local_cols]

                    # D. Filter by Latitude
                    mask = (track_lats >= lat_min) & (track_lats <= lat_max)
                    
                    final_ssh = track_ssh[mask]
                    final_lat = track_lats[mask]
                    final_lon = track_lons[mask]

                    # E. Clean and Assign
                    final_ssh[np.abs(final_ssh) > 1e4] = np.nan

                   # F. Interpolate to evenly spaced along-track points
                    npts = nadir_obj.ssh.shape[1]  
                    master_lat = np.linspace(final_lat.min(), final_lat.max(), npts)

                    f_ssh = interp1d(final_lat, final_ssh, kind='linear', bounds_error=False, fill_value="NaN")
                    nadir_obj.ssh[n, :] = f_ssh(master_lat)

                    f_lon = interp1d(final_lat, final_lon, kind='linear', bounds_error=False, fill_value="NaN")
                    nadir_obj.lon[n, :] = f_lon(master_lat)

                    nadir_obj.lat[n, :] = master_lat

        except KeyError as e:
            print(f"Cycle {cycle}: Variable not found {e}")
        except Exception as e:
            print(f"Error loading Cycle {cycle}: {e}")
    
    # add the time to KaRIn class
    karin_obj.time = np.array(karin_time_list) 
    karin_obj.time_dt = cf_to_datetime64(karin_time_list, ds.variables['time'])

    # Remove the spatial means (should be demeaned anyway)
    spatial_mean = np.nanmean(karin_obj.ssh[n, :, :])
    karin_obj.ssh[n, :, :] -= spatial_mean
    nadir_obj.ssh[n, :] -= spatial_mean


def process_l3_karin(karin, cutoff_m=100e3, delta=2e3):
    """
    Removes high-pass filtered time-mean from KaRIn 2D data.
    """
    ssh_arr = np.asarray(karin.ssh, dtype=float)
    ssh_mean = np.nanmean(ssh_arr, axis=0)
    
    # Fill NaNs for filtering
    nan_mask = np.isnan(ssh_mean)
    ssh_mean_filled = np.where(nan_mask, 0, ssh_mean)

    sigma_pixels = (cutoff_m / (2 * np.sqrt(2 * np.log(2)))) / delta
    
    # 2D Filter
    lowpass = gaussian_filter(ssh_mean_filled, sigma=(sigma_pixels, sigma_pixels), mode='reflect')
    lowpass[nan_mask] = np.nan
    
    ssh_mean_highpass = ssh_mean - lowpass
    karin.ssha = karin.ssh #- ssh_mean_highpass[None, :, :] dont need to extract these use raw product
    karin.ssh_mean_highpass = ssh_mean_highpass
    karin.ssh_mean = ssh_mean

def process_l3_nadir(nadir, cutoff_m=100e3, delta=2e3):
    """
    Removes high-pass filtered time-mean from Nadir 1D data.
    """
    ssha_mean_nadir = np.nanmean(nadir.ssh, axis=0)
    
    sigma_pixels = (cutoff_m / (2 * np.sqrt(2 * np.log(2)))) / delta
    
    # 1D Filter
    lowpass = gaussian_filter(ssha_mean_nadir, sigma=sigma_pixels, mode='reflect')
    ssh_highpass_nadir = ssha_mean_nadir - lowpass
    
    nadir.ssha = nadir.ssh #- ssh_highpass_nadir[None, :]
    nadir.ssh_mean_highpass = ssh_highpass_nadir