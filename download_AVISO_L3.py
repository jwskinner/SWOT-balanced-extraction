import ftplib
import os
import re

# Load credentials
with open("aviso.txt", "r") as f:
    lines = f.read().splitlines()
    user, passwd = lines[0], lines[1]

parent_dir = '/expanse/lustre/projects/cit197/jskinner1/SWOT/LEVEL_3/'
base_path = '/swot_products/l3_karin_nadir/l3_lr_ssh/v2_0_1/Basic/'

target_pass_list = ["461", "183", "155", "086", "366", "394", "267", "280", "002", "545", "573", "558"]  

# Compile a dictionary of regex patterns and create local directories
pass_patterns = {}
for p in target_pass_list:
    p_padded = p.zfill(3)
    pass_patterns[p] = re.compile(fr"SWOT_L3_LR_SSH_Basic_\d{{3}}_{p_padded}_")
    
    # Create directory for each pass
    local_dir = os.path.join(parent_dir, f"pass_{p_padded}")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

try:
    # Connect to AVISO
    ftp = ftplib.FTP('ftp-access.aviso.altimetry.fr')
    ftp.login(user=user, passwd=passwd)
    ftp.set_pasv(True) 
    print(f"Connected. Searching for passes: {', '.join(target_pass_list)}")

    # 4. Get cycle folders
    ftp.cwd(base_path)
    all_dirs = sorted(ftp.nlst())
    
    for d in all_dirs:
        if 'cycle_' in d:
            try:
                cycle_num = int(d.split('_')[1])
                if cycle_num >= 100:  # Skip Cal/Val
                    continue
                
                print(f"\nChecking {d}...")
                ftp.cwd(os.path.join(base_path, d))
                files = ftp.nlst()
                
                for f in files:
                    # Check this file against every pass in list
                    for p_num, pattern in pass_patterns.items():
                        if pattern.search(f):
                            local_dest = os.path.join(parent_dir, f"pass_{p_num.zfill(3)}", f)
                            
                            # Optional: Skip if exists
                            if os.path.exists(local_dest):
                                print(f"  [Exists] {f}")
                                continue
                                
                            print(f"  [Downloading] {f} to pass_{p_num.zfill(3)}...")
                            with open(local_dest, 'wb') as local_file:
                                ftp.retrbinary(f'RETR {f}', local_file.write)
                                
            except Exception as e:
                print(f"  [Skip] Error in {d}: {e}")
                continue

    ftp.quit()
    print(f"\nFinished processing all passes!")

except Exception as e:
    print(f"Connection error: {e}")