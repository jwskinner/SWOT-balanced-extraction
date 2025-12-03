# script for handling the output of SWOT data and processing
from datetime import datetime
import os
import xarray as xr
import numpy as np

def save_spectral_fit_results(filename, karin_popt, karin_pcov, nadir_popt, nadir_pcov, header_info=None):
    """
    Writes spectral fit parameters to a text file.
    """
    
   # Calculate standard deviation errors
    karin_perr = np.sqrt(np.diag(karin_pcov))
    nadir_perr = np.sqrt(np.diag(nadir_pcov))
    
    karin_names = [
        'Amp. balanced', 'lambda balanced', 'slope balanced', 
        'Amp. noise', 'lambda noise', 'slope noise'
    ]

    # 1. Write the human-readable report
    with open(filename, 'w') as f:
        # --- Header ---
        f.write("===================================================\n")
        f.write(f"SWOT SPECTRAL FIT RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if header_info:
            f.write(f"Config: {header_info}\n")
        f.write("===================================================\n\n")

        # --- KaRIn Section ---
        f.write("---- KaRIn Spectrum Parameters ----\n")
        f.write(f"{'Parameter':<20} | {'Value':<15} | {'Error (+/-)':<15}\n")
        f.write("-" * 56 + "\n")
        
        for name, val, err in zip(karin_names, karin_popt, karin_perr):
            f.write(f"{name:<20} | {val:12.4e}    | {err:12.4e}\n")
        f.write("\n")

        # --- Nadir Section ---
        f.write("---- Nadir Spectrum Parameters ----\n")
        f.write(f"{'Parameter':<20} | {'Value':<15} | {'Error (+/-)':<15}\n")
        f.write("-" * 56 + "\n")
        
        # Nadir only fits Noise (index 0), others are fixed from KaRIn
        f.write(f"{'Noise floor (N)':<20} | {nadir_popt[0]:12.4e}    | {nadir_perr[0]:12.4e}\n")
        
        # Log the fixed parameters for record-keeping
        f.write("-" * 56 + "\n")
        f.write("(Fixed Parameters derived from KaRIn fit):\n")
        f.write(f"Amp. balanced:      {karin_popt[0]:.4e}\n")
        f.write(f"lambda balanced:    {karin_popt[1]:.4e}\n")
        f.write(f"slope balanced:     {karin_popt[2]:.4e}\n")

    # 2. Write the machine-readable CSV
    base_name, _ = os.path.splitext(filename)
    csv_filename = f"{base_name}.csv"
    
    with open(csv_filename, 'w') as f_csv:
        f_csv.write("parameter,value,error\n")
        
        # Write KaRIn parameters
        for name, val, err in zip(karin_names, karin_popt, karin_perr):
            # sanitize name: "Amp. balanced" -> "Amp_balanced"
            sanitized_name = "karin_" + name.replace('.', '').replace(' ', '_')
            f_csv.write(f"{sanitized_name},{val:.8e},{err:.8e}\n")
            
        # Write Nadir parameter
        f_csv.write(f"nadir_noise_floor,{nadir_popt[0]:.8e},{nadir_perr[0]:.8e}\n")

    print(f"Fit parameters saved to: {filename}")
    print(f"CSV saved to: {csv_filename}")


def save_swot_to_netcdf(karin, nadir, outdir):
    """
    Saves SWOT object data to a NetCDF file with 'time' as the primary dimension.
    """
    filename = f"{outdir}/SWOT_data.nc"

    time_coords = karin.time_dt
    cycles_data = getattr(karin, 'shared_cycles', []) # all cycles

    def get_wn(obj):
        raw = getattr(obj, 'wavenumbers_cpkm', None)
        return raw.values if hasattr(raw, 'values') else raw

    karin_wavenumbers = get_wn(karin)
    nadir_wavenumbers = get_wn(nadir)

    def get(obj, attr):
        val = getattr(obj, attr, None)
        return val.values if hasattr(val, 'values') else val

    # 3. Data Variables
    data_vars = {
        "cycle_number":     (("time",), cycles_data),

        # --- KaRIn Data ---
        "karin_ssh":        (("time", "karin_y", "karin_x"), karin.ssh),
        "karin_ssha":       (("time", "karin_y", "karin_x"), karin.ssha),
        "karin_tide":       (("time", "karin_y", "karin_x"), karin.tide),
        
        "karin_time":       (("time",), karin.time),

        "karin_lat":        (("time", "karin_y", "karin_x"), karin.lat), 
        "karin_lon":        (("time", "karin_y", "karin_x"), karin.lon),
        "karin_x_km":       (("time", "karin_y", "karin_x"), karin.x_km),
        "karin_y_km":       (("time", "karin_y", "karin_x"), karin.y_km),
        
        "karin_ssh_time_mean": (("karin_y", "karin_x"), karin.ssh_mean),

        # Spectral Data (Wavenumber dimension remains unchanged)
        "karin_spec_ssh":             (("karin_wavenumber",), get(karin, 'spec_ssh')),
        "karin_spec_ssha":            (("karin_wavenumber",), get(karin, 'spec_ssha')),
        "karin_spec_alongtrack_av":   (("karin_wavenumber",), get(karin, 'spec_alongtrack_av')),
        "karin_spec_tide":            (("karin_wavenumber",), get(karin, 'spec_tide')),
        "karin_spec_time_mean":       (("karin_wavenumber",), get(karin, 'spec_tmean')),
        "karin_spec_filt_time_mean":  (("karin_wavenumber",), get(karin, 'spec_filt_tmean')),

        # Spectral Data with Time Dimension
        "karin_spec_alongtrack_ins":     (("time", "karin_wavenumber"), get(karin, 'spec_alongtrack_ins')),
        "karin_spec_alongtrack_time_av": (("karin_wavenumber", "karin_x"), get(karin, 'spec_alongtrack_time_av')),

        # --- Nadir Data ---
        "nadir_ssh":        (("time", "nadir_points"), nadir.ssh),
        "nadir_ssha":       (("time", "nadir_points"), nadir.ssha),
        "nadir_time":       (("time",), nadir.time),

        "nadir_lat":        (("time", "nadir_points",), nadir.lat if nadir.lat.ndim > 1 else nadir.lat),
        "nadir_lon":        (("time", "nadir_points",), nadir.lon if nadir.lon.ndim > 1 else nadir.lon),
        "nadir_x_km":       (("time", "nadir_points",), nadir.x_km),
        "nadir_y_km":       (("time", "nadir_points",), nadir.y_km),

        "nadir_spec_ssh":             (("nadir_wavenumber",), get(nadir, 'spec_ssh')),
        "nadir_spec_ssha":            (("nadir_wavenumber",), get(nadir, 'spec_ssha')),
        "nadir_spec_alongtrack_av":   (("nadir_wavenumber",), get(nadir, 'spec_alongtrack_av')),
        "nadir_spec_alongtrack_ava":  (("nadir_wavenumber",), get(nadir, 'spec_alongtrack_ava')),

        "nadir_spec_alongtrack_ins":  (("time", "nadir_wavenumber"), get(nadir, 'spec_alongtrack_ins')),
    }

    # Filter out None values
    final_vars = {k: v for k, v in data_vars.items() if v[1] is not None}

    coords = {
        "time": time_coords  
    }
    
    if karin_wavenumbers is not None:
        coords["karin_wavenumber"] = (("karin_wavenumber",), karin_wavenumbers)
        
    if nadir_wavenumbers is not None:
        coords["nadir_wavenumber"] = (("nadir_wavenumber",), nadir_wavenumbers)

    ds = xr.Dataset(
        data_vars=final_vars,
        coords=coords,
        attrs={
            "description": "SWOT SSH/SSHA Data",
            "pass_number": getattr(karin, 'pass_number', 0),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    if os.path.exists(filename):
        os.remove(filename)
        
    ds.to_netcdf(filename)
    print(f"Saved SWOT NetCDF to: {filename}")

    # Save quality report
    report_file = f"{outdir}/cycle_quality.txt"

def save_quality_report(filename, karin):
    """
    Saves a text report listing good and bad cycles found during processing.
    """
    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Retrieve lists from object, default to empty if not present
    good = getattr(karin, 'cycles_passed_quality', [])
    bad_qual = getattr(karin, 'cycles_dropped_quality', [])
    bad_var = getattr(karin, 'hvar_cycles', [])
    
    total_processed = len(good) + len(bad_qual) + len(bad_var)

    with open(filename, 'w') as f:
        f.write("==============================================\n")
        f.write(f"SWOT CYCLE QUALITY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==============================================\n\n")
        
        f.write(f"Total Cycles Processed: {total_processed}\n")
        f.write(f"Good Cycles: {len(good)}\n")
        f.write(f"Rejected (Quality): {len(bad_qual)}\n")
        f.write(f"Rejected (High Variance): {len(bad_var)}\n\n")

        f.write("----------------------------------------------\n")
        f.write("GOOD CYCLES (Kept)\n")
        f.write("----------------------------------------------\n")
        f.write(", ".join(map(str, good)) + "\n\n")

        f.write("----------------------------------------------\n")
        f.write("REJECTED CYCLES (Poor Quality Flags)\n")
        f.write("----------------------------------------------\n")
        if bad_qual:
            f.write(", ".join(map(str, bad_qual)) + "\n\n")
        else:
            f.write("None\n\n")

        f.write("----------------------------------------------\n")
        f.write("REJECTED CYCLES (High Variance / Outliers)\n")
        f.write("----------------------------------------------\n")
        if bad_var:
            f.write(", ".join(map(str, bad_var)) + "\n\n")
        else:
            f.write("None\n\n")

    print(f"Quality saved to: {filename}")

def save_balanced_step_to_netcdf(outfolder, karin, t_idx, ht_map_t, ug, vg, zetag, xt, yt):

    outdir = f"{outfolder}/balanced_outputs"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Shapes
    nxt, nyt = ht_map_t.shape

    if ug.shape != (nxt, nyt) or vg.shape != (nxt, nyt) or zetag.shape != (nxt, nyt):
        raise ValueError("ht_map_t, ug, vg, zetag must all have shape (nxt, nyt)")

    if xt.size != nxt * nyt or yt.size != nxt * nyt:
        raise ValueError("xt/yt size must equal nxt*nyt to reshape into 2D grid")

    # 2D coordinate grids in km (same shape as fields)
    x2d = xt.reshape(nyt, nxt).T   # (nxt, nyt)
    y2d = yt.reshape(nyt, nxt).T   # (nxt, nyt)

    # Speed magnitude
    speed = np.sqrt(ug**2 + vg**2)

    time_val = np.array([karin.time[t_idx]]).astype("datetime64[ns]")
    cycle = karin.shared_cycles[t_idx]

    ds = xr.Dataset(
        coords={
            "time": ("time", time_val),
            "x": ("x", np.arange(nxt)*karin.dx_km),
            "y": ("y", np.arange(nyt)*karin.dy_km),
            "x_km": (("x", "y"), x2d),
            "y_km": (("x", "y"), y2d),
        },
        data_vars={
            "h":     (("time", "x", "y"), ht_map_t[np.newaxis, :, :]),
            "u":     (("time", "x", "y"), ug[np.newaxis, :, :]),
            "v":     (("time", "x", "y"), vg[np.newaxis, :, :]),
            "speed": (("time", "x", "y"), speed[np.newaxis, :, :]),
            "zeta":  (("time", "x", "y"), zetag[np.newaxis, :, :]),
        },
        attrs={
            "description": "Balanced fields from SWOT KaRIn + Nadir extraction",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    ds["h"].attrs["units"] = "m"
    ds["u"].attrs["units"] = "m s-1"
    ds["v"].attrs["units"] = "m s-1"
    ds["speed"].attrs["units"] = "m s-1"
    ds["zeta"].attrs["units"] = "1/f"
    ds["x_km"].attrs["units"] = "km"
    ds["y_km"].attrs["units"] = "km"

    fname = os.path.join(outdir, f"balanced_output_cycle{cycle}_t{t_idx:03d}.nc")
    ds.to_netcdf(fname)
    print(f"Saved {fname}")
    return fname
