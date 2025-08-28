# Functions specific for importing the NA simulation data and interpolating the data to a SWOT track 
import os
import scipy.io as sio
from datetime import datetime
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import JWS_SWOT_toolbox as swot

import os
import re
from datetime import datetime, timedelta
import numpy as np

DATE_FMT = "%Y_%m_%d"
FNAME_RE = re.compile(r"snapshot_(\d{4})_(\d{2})_(\d{2})\.mat$")

# ---------- imports ----------
import os, re
import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
from scipy.interpolate import griddata

# ---------- constants & regex ----------
DATE_FMT = "%Y_%m_%d"
FNAME_RE = re.compile(r"snapshot_(\d{4})_(\d{2})_(\d{2})\.mat$")

# ---------- date utilities ----------
def list_available_sim_dates(data_folder):
    """Return sorted list of datetime.date objects for all snapshot_YYYY_MM_DD.mat files."""
    dates = []
    for fn in os.listdir(data_folder):
        m = FNAME_RE.match(fn)
        if m:
            y, mth, d = map(int, m.groups())
            dates.append(datetime(y, mth, d).date())
    dates = sorted(set(dates))
    if not dates:
        raise FileNotFoundError("No snapshot_YYYY_MM_DD.mat files found.")
    return dates

def _doy(date_obj):
    """Day-of-year with a stable handling for Feb 29 (map to Feb 28)."""
    if date_obj.month == 2 and date_obj.day == 29:
        return 59
    return int(date_obj.strftime("%j"))

def _closest_cyclic_day_month(target_dt, available_dates):
    """Closest by calendar day (month/day) ignoring year, with cyclic wrap."""
    t_doy = _doy(target_dt.date())
    best, best_dist = None, 10**9
    for d in available_dates:
        adoy = _doy(datetime(d.year, d.month, d.day))
        delta = abs(adoy - t_doy)
        delta = min(delta, 365 - delta)
        if delta < best_dist:
            best_dist = delta
            best = d
    return best, best_dist

def _closest_absolute_date(target_dt, available_dates):
    """Closest by absolute date (includes year)."""
    t = target_dt.date()
    best = min(available_dates, key=lambda d: abs(d - t))
    return best, abs(best - t).days

def pick_range_from_karin_times(karin_time_dt, data_folder, mode="cyclic", window_days=0):
    """
    Map each karin time to nearest available simulation date and return:
      date_min_str, date_max_str, matched_dates_list
    Skips None entries in karin_time_dt.
    """
    avail = list_available_sim_dates(data_folder)

    matched = []
    for t in karin_time_dt:
        if t is None:
            continue
        # numpy datetime64 support
        if hasattr(t, 'astype') and 'datetime64' in str(type(t)):
            ts = np.datetime64(t, 's').astype('datetime64[s]').astype(object)
        else:
            ts = t
        if ts is None:
            continue

        d, _ = (_closest_cyclic_day_month(ts, avail) if mode == "cyclic"
                else _closest_absolute_date(ts, avail))
        matched.append(d)

    if not matched:
        raise RuntimeError("No valid karin_time_dt entries after skipping None values.")

    dmin = min(matched) - timedelta(days=window_days)
    dmax = max(matched) + timedelta(days=window_days)
    return dmin.strftime(DATE_FMT), dmax.strftime(DATE_FMT), matched

# Functions specific for importing the NA simulation data and interpolating the data to a SWOT track 
import os
import scipy.io as sio
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import griddata, interp1d
import JWS_SWOT_toolbox as swot
import re

# ---------- constants & regex ----------
DATE_FMT = "%Y_%m_%d"
FNAME_RE = re.compile(r"snapshot_(\d{4})_(\d{2})_(\d{2})\.mat$")

# ---------- date utilities ----------
def list_available_sim_dates(data_folder):
    """Return sorted list of datetime.date objects for all snapshot_YYYY_MM_DD.mat files."""
    dates = []
    for fn in os.listdir(data_folder):
        m = FNAME_RE.match(fn)
        if m:
            y, mth, d = map(int, m.groups())
            dates.append(datetime(y, mth, d).date())
    dates = sorted(set(dates))
    if not dates:
        raise FileNotFoundError("No snapshot_YYYY_MM_DD.mat files found.")
    return dates

def _doy(date_obj):
    """Day-of-year with a stable handling for Feb 29 (map to Feb 28)."""
    if date_obj.month == 2 and date_obj.day == 29:
        return 59
    return int(date_obj.strftime("%j"))

def _closest_cyclic_day_month(target_dt, available_dates):
    """Closest by calendar day (month/day) ignoring year, with cyclic wrap."""
    t_doy = _doy(target_dt.date())
    best, best_dist = None, 10**9
    for d in available_dates:
        adoy = _doy(datetime(d.year, d.month, d.day))
        delta = abs(adoy - t_doy)
        delta = min(delta, 365 - delta)
        if delta < best_dist:
            best_dist = delta
            best = d
    return best, best_dist

def _closest_absolute_date(target_dt, available_dates):
    """Closest by absolute date (includes year)."""
    t = target_dt.date()
    best = min(available_dates, key=lambda d: abs(d - t))
    return best, abs(best - t).days

def pick_range_from_karin_times(karin_time_dt, data_folder, mode="cyclic", window_days=0):
    """
    Map each karin time to nearest available simulation date and return:
      date_min_str, date_max_str, matched_dates_list
    Skips None entries in karin_time_dt.
    """
    avail = list_available_sim_dates(data_folder)

    matched = []
    for t in karin_time_dt:
        if t is None:
            continue
        # numpy datetime64 support
        if hasattr(t, 'astype') and 'datetime64' in str(type(t)):
            ts = np.datetime64(t, 's').astype('datetime64[s]').astype(object)
        else:
            ts = t
        if ts is None:
            continue

        d, _ = (_closest_cyclic_day_month(ts, avail) if mode == "cyclic"
                else _closest_absolute_date(ts, avail))
        matched.append(d)

    if not matched:
        raise RuntimeError("No valid karin_time_dt entries after skipping None values.")

    dmin = min(matched) - timedelta(days=window_days)
    dmax = max(matched) + timedelta(days=window_days)
    return dmin.strftime(DATE_FMT), dmax.strftime(DATE_FMT), matched

# ---------- spatial interpolation functions ----------
def interpolate_onto_karin_grid(XC, YC, ssh_model, karin_lon, karin_lat, buffer=0.5):
    """
    Interpolate model SSH onto the SWOT KaRIn grid, matching KaRIn's lon convention.
    Robust to NaNs in target coords: only interpolates to finite (lat,lon) points.
    """
    # --- coerce shapes ---
    XC = np.asarray(XC).squeeze()
    YC = np.asarray(YC).squeeze()
    Z  = np.asarray(ssh_model).squeeze()
    KL = np.asarray(karin_lat)
    KX = np.asarray(karin_lon)

    if KL.ndim != 2 or KX.ndim != 2:
        raise ValueError(f"karin_lat/lon must be 2-D; got {KL.shape}, {KX.shape}")
    if XC.ndim != 2 or YC.ndim != 2 or Z.ndim != 2:
        raise ValueError(f"XC/YC/ssh must be 2-D; got XC={XC.shape}, YC={YC.shape}, ssh={Z.shape}")

    # --- convert SIM longitudes to match KaRIn convention (leave KaRIn as-is) ---
    kmin = float(np.nanmin(KX)); kmax = float(np.nanmax(KX))
    if (kmax > 180.0) or (kmin >= 0.0):
        XCadj = np.mod(XC, 360.0)                              # KaRIn uses 0..360
    else:
        XCadj = (XC % 360.0 + 180.0) % 360.0 - 180.0           # KaRIn uses -180..180

    # --- bbox around KaRIn target (dateline-aware for either convention) ---
    lat_min, lat_max = float(np.nanmin(KL)), float(np.nanmax(KL))
    lon_min, lon_max = float(np.nanmin(KX)), float(np.nanmax(KX))

    if lon_max - lon_min <= 180.0:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                (XCadj >= lon_min - buffer) & (XCadj <= lon_max + buffer))
    else:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XCadj >= lon_min - buffer) | (XCadj <= lon_max + buffer)))

    # --- choose source set (subset if possible; else full) and drop non-finite ---
    if np.any(mask) and np.isfinite(Z[mask]).any():
        Ys, Xs, Vs = YC[mask], XCadj[mask], Z[mask]
    else:
        Ys, Xs, Vs = YC, XCadj, Z

    sfin = np.isfinite(Ys) & np.isfinite(Xs) & np.isfinite(Vs)
    Ys, Xs, Vs = Ys[sfin], Xs[sfin], Vs[sfin]
    if Ys.size == 0:
        return np.full(KL.shape, np.nan, dtype=float)

    # --- target: only interpolate where target coords are finite ---
    Yt, Xt = KL, KX
    tfin = np.isfinite(Yt) & np.isfinite(Xt)

    out = np.full(KL.shape, np.nan, dtype=float)
    src_pts = np.column_stack((Ys.ravel(), Xs.ravel()))
    tgt_pts = np.column_stack((Yt[tfin].ravel(), Xt[tfin].ravel()))

    # linear first (needs at least 3 non-collinear points)
    if Ys.size >= 3:
        lin = griddata(src_pts, Vs.ravel(), tgt_pts, method="linear", fill_value=np.nan)
    else:
        lin = np.full((tgt_pts.shape[0],), np.nan, dtype=float)

    # nearest fill for remaining NaNs (tgt_pts contains no NaNs now)
    bad = np.isnan(lin)
    if bad.any():
        lin[bad] = griddata(src_pts, Vs.ravel(), tgt_pts[bad], method="nearest")

    out[tfin] = lin
    return out

def interpolate_onto_nadir_grid(XC, YC, nadir_lon, nadir_lat, ssh_model=None, buffer=0.5):

    if ssh_model is None:
        return np.full(nadir_lat.shape, np.nan, dtype=float)
    
    # Similar logic to interpolate_onto_karin_grid but for 1D nadir track
    XC = np.asarray(XC).squeeze()
    YC = np.asarray(YC).squeeze()
    Z = np.asarray(ssh_model).squeeze()
    
    nadir_lat = np.asarray(nadir_lat)
    nadir_lon = np.asarray(nadir_lon)
    
    # Handle longitude conventions
    nmin = float(np.nanmin(nadir_lon)); nmax = float(np.nanmax(nadir_lon))
    if (nmax > 180.0) or (nmin >= 0.0):
        XCadj = np.mod(XC, 360.0)
    else:
        XCadj = (XC % 360.0 + 180.0) % 360.0 - 180.0
    
    # Create bounding box
    lat_min, lat_max = float(np.nanmin(nadir_lat)), float(np.nanmax(nadir_lat))
    lon_min, lon_max = float(np.nanmin(nadir_lon)), float(np.nanmax(nadir_lon))
    
    if lon_max - lon_min <= 180.0:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                (XCadj >= lon_min - buffer) & (XCadj <= lon_max + buffer))
    else:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XCadj >= lon_min - buffer) | (XCadj <= lon_max + buffer)))
    
    # Select source points
    if np.any(mask) and np.isfinite(Z[mask]).any():
        Ys, Xs, Vs = YC[mask], XCadj[mask], Z[mask]
    else:
        Ys, Xs, Vs = YC, XCadj, Z
    
    sfin = np.isfinite(Ys) & np.isfinite(Xs) & np.isfinite(Vs)
    Ys, Xs, Vs = Ys[sfin], Xs[sfin], Vs[sfin]
    
    if Ys.size == 0:
        return np.full(nadir_lat.shape, np.nan, dtype=float)
    
    # Target points
    tfin = np.isfinite(nadir_lat) & np.isfinite(nadir_lon)
    out = np.full(nadir_lat.shape, np.nan, dtype=float)
    
    if not np.any(tfin):
        return out
    
    src_pts = np.column_stack((Ys.ravel(), Xs.ravel()))
    tgt_pts = np.column_stack((nadir_lat[tfin].ravel(), nadir_lon[tfin].ravel()))
    
    # Linear interpolation first
    if Ys.size >= 3:
        lin = griddata(src_pts, Vs.ravel(), tgt_pts, method="linear", fill_value=np.nan)
    else:
        lin = np.full((tgt_pts.shape[0],), np.nan, dtype=float)
    
    # Fill with nearest neighbor
    bad = np.isnan(lin)
    if bad.any():
        lin[bad] = griddata(src_pts, Vs.ravel(), tgt_pts[bad], method="nearest")
    
    out[tfin] = lin
    return out


def load_sim_on_karin_nadir_grids(karin, nadir, data_folder, matched_dates):
 
    import os
    import numpy as np
    import scipy.io as sio

    def _wrap_like_karin(lon, karin_lon2d):
        """Wrap sim longitudes to the same convention as KaRIn lon."""
        kmin = np.nanmin(karin_lon2d); kmax = np.nanmax(karin_lon2d)
        if kmax > 180 and kmin >= 0:
            out = np.mod(lon, 360.0); out[out < 0] += 360.0
        else:
            out = (lon + 180.0) % 360.0 - 180.0
        return out

    # ---- inputs & basic coercions ----
    karin_lat = np.asarray(karin.lat)[0]
    karin_lon = np.asarray(karin.lon)[0]
    karin_lat_full = np.asarray(karin.lat_full)
    karin_lon_full = np.asarray(karin.lon_full)
    nadir_lat = np.asarray(nadir.lat)[0]
    nadir_lon = np.asarray(nadir.lon)[0]

    ssh_karin_list, ssh_karin_full_list, ssh_nadir_list, used = [], [], [], []
    ssh_full_box_list, lat_full_box_list, lon_full_box_list = [], [], []

    # ---- main loop ----
    for d in matched_dates:
        fpath = os.path.join(data_folder, f"snapshot_{d.strftime(DATE_FMT)}.mat")
        if not os.path.exists(fpath):
            continue

        mat = sio.loadmat(fpath)
        XC  = np.asarray(mat["XC"]).squeeze()    # (Ny,Nx)
        YC  = np.asarray(mat["YC"]).squeeze()    # (Ny,Nx)
        ssh = np.asarray(mat["ssh"]).squeeze()   # (Ny,Nx)
        if ssh.ndim != 2 or XC.ndim != 2 or YC.ndim != 2:
            raise ValueError(f"In {fpath}: XC, YC, ssh must be 2-D; got {XC.shape}, {YC.shape}, {ssh.shape}")

        # Match longitude convention to KaRIn
        XC = _wrap_like_karin(XC, karin_lon)

        # Interpolations (your existing helpers)
        ssh_karin = np.asarray(interpolate_onto_karin_grid(XC, YC, ssh, karin_lon, karin_lat))
        ssh_karin_full = np.asarray(interpolate_onto_karin_grid(XC, YC, ssh, karin_lon_full, karin_lat_full))
        ssh_nadir = np.asarray(interpolate_onto_nadir_grid(XC, YC, nadir_lon, nadir_lat, ssh))
        if ssh_karin.ndim != 2:
            raise ValueError(f"Interpolated KaRIn SSH is not 2-D: {ssh_karin.shape}")

        ssh_karin_list.append(ssh_karin)
        ssh_karin_full_list.append(ssh_karin_full)
        ssh_nadir_list.append(ssh_nadir)
        used.append(d)

    if not ssh_karin_list:
        raise RuntimeError("No simulation snapshots could be interpolated.")

    # ---- stack & return ----
    ssh_karin_out = np.stack(ssh_karin_list, axis=0)       # (T,L,W)
    ssh_karin_full_out = np.stack(ssh_karin_full_list, axis=0)
    ssh_nadir_out = np.stack(ssh_nadir_list, axis=0)       # (T, L) or (T,)


    return (ssh_karin_full_out, ssh_karin_out, ssh_nadir_out,
            used)


# ---------- Legacy functions (kept for backward compatibility) ----------

def interpolate_swot_pass_griddata_optimized(XC, YC, ssh_model, lon_swot, lat_swot, buffer=0.5):
    """Legacy function - use interpolate_onto_karin_grid instead."""
    
    # Bounding box
    lat_min, lat_max = lat_swot.min(), lat_swot.max()
    lon_min, lon_max = lon_swot.min(), lon_swot.max()

    # Adjust longitude convention
    XC_adj = (XC + 180) % 360 - 180
    mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
            (XC_adj >= lon_min - buffer) & (XC_adj <= lon_max + buffer))
    
    if lon_max - lon_min > 180:
         mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XC_adj >= lon_min - buffer) | (XC_adj <= lon_max + buffer)))

    if not np.any(mask):
        return np.full(lat_swot.shape, np.nan)

    YC_subset = YC[mask]
    XC_adj_subset = XC_adj[mask]
    ssh_model_subset = ssh_model[mask]

    source_points = np.column_stack((YC_subset, XC_adj_subset))
    source_values = ssh_model_subset
    target_points = np.column_stack((lat_swot.ravel(), lon_swot.ravel()))

    ssh_interpolated = griddata(source_points, source_values, target_points,
                                method='linear', fill_value=np.nan)

    return ssh_interpolated.reshape(lat_swot.shape)

def extract_pass_swath(pass_num, pass_coords, data_folder, date_min, date_max, lat_min=None, lat_max=None):
    """Extract swath data for a specific pass."""
    
    date_fmt = "%Y_%m_%d"
    tmin = datetime.strptime(date_min, date_fmt)
    tmax = datetime.strptime(date_max, date_fmt)

    mat_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".mat")])
    if not mat_files:
        raise FileNotFoundError("No .mat files found in data folder.")

    sample_file = os.path.join(data_folder, mat_files[0])
    mat = sio.loadmat(sample_file)
    XC = mat['XC']
    YC = mat['YC']

    entry = next((e for e in pass_coords if e[0] == pass_num), None)
    if entry is None:
        raise ValueError(f"Pass {pass_num} not found in pass_coords.")

    lat = entry[1]
    lon = (entry[2] + 180) % 360 - 180

    if lat_min is not None and lat_max is not None:
        if lat.ndim != 2:
            raise ValueError("Expected 2D lat/lon arrays for filtering")
        
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        row_mask = np.any(lat_mask, axis=1)

        if not np.any(row_mask):
            print(f"Warning: The latitude filter ({lat_min}, {lat_max}) removed all SWOT data for pass {pass_num}.")
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        lat = lat[row_mask, :]
        lon = lon[row_mask, :]
        
    ssh_list = []

    for fname in mat_files:
        try:
            date_str = fname.replace("snapshot_", "").replace(".mat", "")
            t = datetime.strptime(date_str, date_fmt)
        except ValueError:
            continue

        if not (tmin <= t <= tmax):
            continue

        fpath = os.path.join(data_folder, fname)
        mat = sio.loadmat(fpath)
        ssh = mat['ssh']
        
        ssh_interpolated = interpolate_swot_pass_griddata_optimized(XC, YC, ssh, lon, lat)
        ssh_list.append(ssh_interpolated)

    if not ssh_list:
        raise RuntimeError("No valid snapshot files found in time range.")

    ssh_all = np.stack(ssh_list)
    
    return ssh_all, lat, lon


#### ------ OLD ------

def interpolate_swot_pass_griddata_optimized(XC, YC, ssh_model, lon_swot, lat_swot, buffer=0.5):

    # bouding box
    lat_min, lat_max = lat_swot.min(), lat_swot.max()
    lon_min, lon_max = lon_swot.min(), lon_swot.max()

    # mask it
    XC_adj = (XC + 180) % 360 - 180
    mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
            (XC_adj >= lon_min - buffer) & (XC_adj <= lon_max + buffer))
    
    if lon_max - lon_min > 180:
         mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XC_adj >= lon_min - buffer) | (XC_adj <= lon_max + buffer)))

    if not np.any(mask):
        return np.full(lat_swot.shape, np.nan)

    YC_subset = YC[mask]
    XC_adj_subset = XC_adj[mask]
    ssh_model_subset = ssh_model[mask]

    source_points = np.column_stack((YC_subset, XC_adj_subset))
    source_values = ssh_model_subset
    target_points = np.column_stack((lat_swot.ravel(), lon_swot.ravel()))

    ssh_interpolated = griddata(source_points, source_values, target_points,
                                method='linear', fill_value=np.nan)

    return ssh_interpolated.reshape(lat_swot.shape), lat_swot, lon_swot


def extract_pass_swath(pass_num, pass_coords, data_folder, date_min, date_max, lat_min=None, lat_max=None):

    date_fmt = "%Y_%m_%d"
    tmin = datetime.strptime(date_min, date_fmt)
    tmax = datetime.strptime(date_max, date_fmt)

    mat_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".mat")])
    if not mat_files:
        raise FileNotFoundError("No .mat files found in data folder.")

    sample_file = os.path.join(data_folder, mat_files[0])
    mat = sio.loadmat(sample_file)
    XC = mat['XC']
    YC = mat['YC']

    entry = next((e for e in pass_coords if e[0] == pass_num), None)
    if entry is None:
        raise ValueError(f"Pass {pass_num} not found in pass_coords.")

    lat = entry[1]
    lon = (entry[2] + 180) % 360 - 180

    if lat_min is not None and lat_max is not None:
        if lat.ndim != 2:
            raise ValueError("Expected 2D lat/lon arrays for filtering")
        
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        row_mask = np.any(lat_mask, axis=1)

        # If the filter removes all data, return empty arrays to avoid processing.
        if not np.any(row_mask):
            print(f"Warning: The latitude filter ({lat_min}, {lat_max}) removed all SWOT data for pass {pass_num}.")
            return np.empty((0,)), np.empty((0,)), np.empty((0,))

        # Apply the mask to the coordinates that will be used in the loop
        lat = lat[row_mask, :]
        lon = lon[row_mask, :]
        
    ssh_list = []

    for fname in mat_files:
        try:
            date_str = fname.replace("snapshot_", "").replace(".mat", "")
            t = datetime.strptime(date_str, date_fmt)
        except ValueError:
            continue

        if not (tmin <= t <= tmax):
            continue

        fpath = os.path.join(data_folder, fname)
        mat = sio.loadmat(fpath)
        ssh = mat['ssh']
        
        ssh_interpolated = interpolate_swot_pass_griddata_optimized(XC, YC, ssh, lon, lat)
        ssh_list.append(ssh_interpolated)

    if not ssh_list:
        raise RuntimeError("No valid snapshot files found in time range.")

    ssh_all = np.stack(ssh_list)
    
    return ssh_all, lat, lon

def sample_NA_sim_to_karin_and_nadir(ssh_model, lat_model, lon_model, 
                                     karin_target_shape, nadir_target_shape,
                                     return_full_sim=True):

    ntime, track_len, NA_total_width = ssh_model.shape
    _, track_len_target, total_width = karin_target_shape
    _, track_len_nadir = nadir_target_shape

    swath_width = 25
    track_axis = np.linspace(0, track_len - 1, track_len_target)

    # Interpolate SSH
    ssh_interp = np.empty((ntime, track_len_target, NA_total_width))
    for i in range(ntime):
        f = interp1d(np.arange(track_len), ssh_model[i], axis=0, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        ssh_interp[i] = f(track_axis)

    # Interpolate LAT and LON
    lat_interp = interp1d(np.arange(track_len), lat_model, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')(track_axis)
    lon_interp = interp1d(np.arange(track_len), lon_model, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')(track_axis)

    # Build KaRIn swath: gap in center, swath_width = 25 pixels per side
    NA_sim_karin_ssh = np.full((ntime, track_len_target, total_width), np.nan)
    NA_sim_karin_ssh[:, :, :swath_width] = ssh_interp[:, :, :swath_width]
    NA_sim_karin_ssh[:, :, 35:60] = ssh_interp[:, :, 35:60]

    NA_sim_karin_lat = np.full((track_len_target, total_width), np.nan)
    NA_sim_karin_lon = np.full((track_len_target, total_width), np.nan)
    NA_sim_karin_lat[:, :swath_width] = lat_interp[:, :swath_width]
    NA_sim_karin_lon[:, :swath_width] = lon_interp[:, :swath_width]
    NA_sim_karin_lat[:, 35:60] = lat_interp[:, 35:60]
    NA_sim_karin_lon[:, 35:60] = lon_interp[:, 35:60]

    # Sample Nadir track
    nadir_track_axis = np.linspace(0, track_len_target - 1, track_len_nadir).astype(int)
    NA_sim_nadir_ssh = ssh_interp[:, nadir_track_axis, 30]
    NA_sim_nadir_lat = lat_interp[nadir_track_axis, 30]
    NA_sim_nadir_lon = lon_interp[nadir_track_axis, 30]

    if return_full_sim:
        return (
            NA_sim_karin_ssh, NA_sim_karin_lat, NA_sim_karin_lon,
            NA_sim_nadir_ssh, NA_sim_nadir_lat, NA_sim_nadir_lon,
            ssh_interp, lat_interp, lon_interp
        )
    else:
        return (
            NA_sim_karin_ssh, NA_sim_karin_lat, NA_sim_karin_lon,
            NA_sim_nadir_ssh, NA_sim_nadir_lat, NA_sim_nadir_lon
        )