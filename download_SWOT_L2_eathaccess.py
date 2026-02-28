import netCDF4 as nc
import numpy as np
import earthaccess 
import os 
from glob import glob
import numpy as np
import time
import mpmath as mp
import jws_swot_tools as swot


# Import the data from the cloud
# region NA [48, 35, 7, 354, 326, 313]
# region K.E [267, 573, 280, 002, 545, 550]
# region S. O [461, 185, 155, 088, 386, 394]

pass_list = [2, 17, 30, 295, 308, 323] # KE Region
pass_list = [394, 489, 88, 60, 183, 211, 366] # SO region

tmin = "2023-07-10 00:00:00"
tmax = "2026-03-20 23:59:59"
download_dir = "/expanse/lustre/projects/cit197/jskinner1/SWOT/SCIENCE_VD/"

def download_batch_cloud_data(granules, download_dir, verbose=True):
    os.makedirs(download_dir, exist_ok=True)

    downloaded_files= earthaccess.download(
        granules, 
        local_path=download_dir, 
        threads = 4 
    )
    if verbose:
        print(f"Downloaded {len(downloaded_files)} files to {download_dir}")

    return downloaded_files

for pass_num in pass_list:
    print(pass_num)

    shared_cycles, karin_results, nadir_results = swot.return_cloud_files(
        pass_num, tmin, tmax, 
        karin_short_name = "SWOT_L2_LR_SSH_D", 
        nadir_short_name = "SWOT_L2_NALT_GDR_D"
    )

    # Download KaRIn and nadir to same directory
    download_batch_cloud_data(karin_results, download_dir)
    download_batch_cloud_data(nadir_results, download_dir)
    print(f"Completed: {pass_num}")