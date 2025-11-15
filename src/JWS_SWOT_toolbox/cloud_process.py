# Functions and classes for processing SWOT Level 2 data in the cloud, useful for the science data processing pipeline.
import earthaccess
import numpy as np
import xarray as xr
import JWS_SWOT_toolbox as swot

import earthaccess
import xarray as xr
from typing import List, Optional
import numpy as np
import earthaccess
import xarray as xr
from typing import List, Optional
import numpy as np
import h5py

def load_cloud_data(
    ssh_results: list,
    latmin: Optional[float] = None,
    latmax: Optional[float] = None,
    shared_cycles: Optional[List[int]] = None
    ):
    print("------- Loading Data -------")
    file_objs = earthaccess.open(ssh_results)
    datasets = [xr.open_dataset(f, engine='h5netcdf') for f in file_objs]

    if not datasets:
        return xr.Dataset()

    print(" ")
    print("------- Combining Data -------")
    combined = xr.concat(datasets, dim="Cycle")


    if shared_cycles is not None:
        if 'cycle' in combined:
            combined = combined.where(combined.cycle.isin(shared_cycles), drop=True)
        else:
            print("Warning: 'cycle' variable not found. Skipping cycle filter.")

    if latmin is not None and latmax is not None:
        if 'lat' in combined.coords:

            if combined.lat.values[0] > combined.lat.values[-1]:

                combined = combined.sel(lat=slice(latmax, latmin))
            else:

                combined = combined.sel(lat=slice(latmin, latmax))
        else:
            print("Warning: 'lat' coordinate not found. Skipping latitude filter.")

    return combined


def find_matching_cycles(karin_results, nadir_results):
    try:
        
        karin_cycles = {granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Track']['Cycle'] for granule in karin_results}
        karin_passes = {granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Track']['Passes'][0]['Pass'] for granule in karin_results}
        nadir_cycles = {granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Track']['Cycle'] for granule in nadir_results}
        nadir_passes = {granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Track']['Passes'][0]['Pass'] for granule in nadir_results}
        
    except (KeyError, TypeError) as e:
        print(f"Error: Could not parse cycle information due to unexpected data structure: {e}")
        print("Please ensure the results lists are not empty and contain the expected 'Track' and 'Cycle' keys.")
        return []

    # Find the intersection of the two sets to get the matching cycles
    matching_cycles = karin_cycles.intersection(nadir_cycles)

    print(str(len(matching_cycles))+" Matching Cycles Found")

    # Return the result as a sorted list
    return sorted(list(matching_cycles)), karin_cycles, nadir_cycles, karin_passes, nadir_passes

def return_cloud_files(pass_num, tmin, tmax): 
    
    # Authenticate
    print("------- Authenticating User -------")
    auth = earthaccess.login("netrc")
    print("User authenticated:", auth.authenticated)

    print(" ")
    print("Searching Earthaccess ...")
    # Search for all granules matching this pass
    karin_results = earthaccess.search_data(
        short_name='SWOT_L2_LR_SSH_2.0',
        temporal=(tmin, tmax),
        granule_name=f'*_SSH_Basic_*_{pass_num:03d}_*'
    )

    # Search for all granules matching this pass
    nadir_results = earthaccess.search_data(
        short_name='SWOT_L2_NALT_GDR_2.0',
        temporal=(tmin, tmax),
        granule_name=f'*_{pass_num:03d}_*'
    )

    shared_cycles, karin_cycles, nadir_cycles, karin_passes, nadir_passes = find_matching_cycles(karin_results, nadir_results)

    print("------- Available Data -------")
    print(f'KaRIn Cycles: {karin_cycles}')
    print(f'Nadir Cycles: {nadir_cycles}')

    print(f'KaRIn passes: {karin_passes}')
    print(f'Nadir passes: {nadir_passes}')

    return shared_cycles, karin_results, nadir_results

def open_batch_cloud_data(cloud_files, verbose=True): 
    file_objs = earthaccess.open(cloud_files)

    datasets = []
    for f in file_objs:
        if verbose:
            print(f'Loading: {f}')
        with f:
            ds = xr.open_dataset(f, engine='h5netcdf')  
            datasets.append(ds)
    return datasets
