#!/bin/bash
# This script downloads all of the Karin CalVal data to a specified location
#
# Requirements 
#  1) Install PODAAC data subscriber 
# 	pip install podaac-data-subscriber

#  2) Create a ~/.netrc file with the following contents

#  	  machine urs.earthdata.nasa.gov
#     login YOUR_EARTHDATA_USERNAME
#     password YOUR_EARTHDATA_PASSWORD
 
#  3) Fix .netrc permissions 
#  	chmod 600 ~/.netrc

#  4) Change download location below and run this script

DOWNLOAD_LOCATION="./SWOT_data/CALVAL/Karin/"

# Loop through the sequence of cycle numbers and execute the podaac-data-downloader command
for i in $(seq 474 577)
do
    echo "Starting download for cycle $i..."
    podaac-data-downloader -c SWOT_L2_LR_SSH_D -d "$DOWNLOAD_LOCATION" --cycle $i
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded data for cycle $i."
    else
        echo "Failed to download data for cycle $i."
    fi
done

echo "All downloads completed."