# --- Downloads AVISO+ SWOT Level 3 data ---

import ftplib
import os
import re

# 1. Load credentials
with open("aviso.txt", "r") as f:
    lines = f.read().splitlines()
    user, passwd = lines[0], lines[1]

# 2. Configuration
parent_dir = '/expanse/lustre/projects/cit197/jskinner1/SWOT/LEVEL_3/'
base_path = '/swot_products/l3_karin_nadir/l3_lr_ssh/v2_0_1/Basic/'
target_pass_num = "48" 

# Regex to match: Basic_[Cycle]_[Pass]_
pass_pattern = re.compile(fr"SWOT_L3_LR_SSH_Basic_\d{{3}}_{target_pass_num.zfill(3)}_")

local_dir = os.path.join(parent_dir, f"pass_{target_pass_num.zfill(3)}")
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

try:
    # 3. Connect
    ftp = ftplib.FTP('ftp-access.aviso.altimetry.fr')
    ftp.login(user=user, passwd=passwd)
    ftp.set_pasv(True) 
    print(f"Connected. Searching for Pass {target_pass_num}...")

    # 4. Get and filter cycle folders
    ftp.cwd(base_path)
    all_dirs = sorted(ftp.nlst())
    
    for d in all_dirs:
        if 'cycle_' in d:
            try:
                # Extract cycle number; skip Cal/Val phase (cycles 400+)
                cycle_num = int(d.split('_')[1])
                if cycle_num >= 100:
                    continue
                
                # Enter cycle folder
                ftp.cwd(os.path.join(base_path, d))
                files = ftp.nlst()
                
                for f in files:
                    if pass_pattern.search(f):
                        local_filepath = os.path.join(local_dir, f)
                        
                        # if os.path.exists(local_filepath):
                        #     print(f"  [Exists] {f}")
                        #     continue
                        
                        print(f"  [Downloading] {f}...")
                        with open(local_filepath, 'wb') as local_file:
                            ftp.retrbinary(f'RETR {f}', local_file.write)
            except Exception as e:
                print(f"  [Skip] Error in {d}: {e}")
                continue

    ftp.quit()
    print(f"\nFinished! Files saved to: {local_dir}")

except Exception as e:
    print(f"Connection error: {e}")