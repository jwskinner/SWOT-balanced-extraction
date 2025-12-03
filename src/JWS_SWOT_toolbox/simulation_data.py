# Functions specific for importing the NA simulation data and interpolating the data to a SWOT track 
import os, re
import scipy.io as sio
from scipy.interpolate import interp1d
import JWS_SWOT_toolbox as swot
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
def _wrap_longitudes_to_match(lon_to_wrap, ref_lon2d):
    """Return lon_to_wrap wrapped into the same convention as ref_lon2d."""
    lon = np.asarray(lon_to_wrap)
    kmin = float(np.nanmin(ref_lon2d)); kmax = float(np.nanmax(ref_lon2d))
    if (kmax > 180.0) and (kmin >= 0.0):
        # reference uses 0..360
        out = np.mod(lon, 360.0)
        # ensure non-negative
        out = np.where(out < 0, out + 360.0, out)
    else:
        # reference uses -180..180
        out = (lon % 360.0 + 180.0) % 360.0 - 180.0
    return out
    
def interpolate_onto_karin_grid(XC, YC, ssh_model, karin_lon, karin_lat, buffer=0.5):

    XC = np.asarray(XC).squeeze()
    YC = np.asarray(YC).squeeze()
    Z  = np.asarray(ssh_model).squeeze()
    KL = np.asarray(karin_lat)
    KX = np.asarray(karin_lon)

    if KL.ndim != 2 or KX.ndim != 2:
        raise ValueError(f"karin_lat/lon must be 2-D; got {KL.shape}, {KX.shape}")
    if XC.ndim != 2 or YC.ndim != 2 or Z.ndim != 2:
        raise ValueError(f"XC/YC/ssh must be 2-D; got XC={XC.shape}, YC={YC.shape}, ssh={Z.shape}")

    # --- convert sim longitudes to match KaRIn convention  ---
    XCadj = _wrap_longitudes_to_match(XC, KX)

    # --- bbox around KaRIn target ---
    lat_min, lat_max = float(np.nanmin(KL)), float(np.nanmax(KL))
    lon_min, lon_max = float(np.nanmin(KX)), float(np.nanmax(KX))

    if lon_max - lon_min <= 180.0:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                (XCadj >= lon_min - buffer) & (XCadj <= lon_max + buffer))
    else:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XCadj >= lon_min - buffer) | (XCadj <= lon_max + buffer)))

    # --- choose source set and drop non-finite ---
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
    # Use (lon, lat) ordering explicitly (x, y)
    src_pts = np.column_stack((Xs.ravel(), Ys.ravel()))
    tgt_pts = np.column_stack((Xt[tfin].ravel(), Yt[tfin].ravel()))

    # Debugging: number of source/target points
    # print("interpolate_onto_karin_grid: src pts:", src_pts.shape[0], " tgt pts:", tgt_pts.shape[0])

    # linear first (needs at least 3 non-collinear points)
    if src_pts.shape[0] >= 3:
        lin = griddata(src_pts, Vs.ravel(), tgt_pts, method="cubic", fill_value=np.nan)
    else:
        lin = np.full((tgt_pts.shape[0],), np.nan, dtype=float)

    # nearest fill for remaining NaNs
    bad = np.isnan(lin)
    if bad.any():
        lin[bad] = griddata(src_pts, Vs.ravel(), tgt_pts[bad], method="nearest")

    out[tfin] = lin
    return out

def interpolate_onto_nadir_grid(XC, YC, nadir_lon, nadir_lat, ssh_model=None, buffer=0.5):
    if ssh_model is None:
        return np.full(nadir_lat.shape, np.nan, dtype=float)

    XC = np.asarray(XC).squeeze()
    YC = np.asarray(YC).squeeze()
    Z = np.asarray(ssh_model).squeeze()

    nadir_lat = np.asarray(nadir_lat)
    nadir_lon = np.asarray(nadir_lon)

    try:
        # determine convention from XC itself
        XC_u = _wrap_longitudes_to_match(XC, XC)  # returns in consistent convention
    except Exception:
        XC_u = XC

    # rewrap nadir lon to the same convention as XC_u
    nadir_lon_wrapped = _wrap_longitudes_to_match(nadir_lon, XC_u)

    # Create bounding box for nadir track
    lat_min, lat_max = float(np.nanmin(nadir_lat)), float(np.nanmax(nadir_lat))
    lon_min, lon_max = float(np.nanmin(nadir_lon_wrapped)), float(np.nanmax(nadir_lon_wrapped))

    if lon_max - lon_min <= 180.0:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                (XC_u >= lon_min - buffer) & (XC_u <= lon_max + buffer))
    else:
        mask = ((YC >= lat_min - buffer) & (YC <= lat_max + buffer) &
                ((XC_u >= lon_min - buffer) | (XC_u <= lon_max + buffer)))

    if np.any(mask) and np.isfinite(Z[mask]).any():
        Ys, Xs, Vs = YC[mask], XC_u[mask], Z[mask]
    else:
        Ys, Xs, Vs = YC, XC_u, Z

    sfin = np.isfinite(Ys) & np.isfinite(Xs) & np.isfinite(Vs)
    Ys, Xs, Vs = Ys[sfin], Xs[sfin], Vs[sfin]
    if Ys.size == 0:
        return np.full(nadir_lat.shape, np.nan, dtype=float)

    # target points (use wrapped nadir lon)
    tfin = np.isfinite(nadir_lat) & np.isfinite(nadir_lon_wrapped)
    out = np.full(nadir_lat.shape, np.nan, dtype=float)
    if not np.any(tfin):
        return out

    # use (lon, lat) ordering
    src_pts = np.column_stack((Xs.ravel(), Ys.ravel()))
    tgt_pts = np.column_stack((nadir_lon_wrapped[tfin].ravel(), nadir_lat[tfin].ravel()))

    # Debug
    # print("interpolate_onto_nadir_grid: src pts:", src_pts.shape[0], " tgt pts:", tgt_pts.shape[0])

    if src_pts.shape[0] >= 3:
        lin = griddata(src_pts, Vs.ravel(), tgt_pts, method="linear", fill_value=np.nan)
    else:
        lin = np.full((tgt_pts.shape[0],), np.nan, dtype=float)

    bad = np.isnan(lin)
    if bad.any():
        lin[bad] = griddata(src_pts, Vs.ravel(), tgt_pts[bad], method="nearest")

    out[tfin] = lin
    return out


def load_sim_on_karin_nadir_grids(karin, nadir, data_folder, matched_dates):

    def _wrap_like_karin(lon, karin_lon2d):
        """Wrap sim longitudes to the same convention as KaRIn lon."""
        kmin = np.nanmin(karin_lon2d); kmax = np.nanmax(karin_lon2d)
        if kmax > 180 and kmin >= 0:
            out = np.mod(lon, 360.0); out[out < 0] += 360.0
        else:
            out = (lon + 180.0) % 360.0 - 180.0
        return out
    
    print("Computing time mean over all files")
    ssh_all = []
    for fn in os.listdir(data_folder):
        if fn.endswith(".mat") and "snapshot_" in fn:
            fpath = os.path.join(data_folder, fn)
            ssh = np.asarray(sio.loadmat(fpath)["ssh"]).squeeze()
            ssh_all.append(ssh)

    ssh_all = np.stack(ssh_all, axis=0)
    ssh_tmean = np.nanmean(ssh_all, axis=0)
    print("Computed time mean over", ssh_all.shape[0], "files with shape", ssh_tmean.shape)

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
        ssh_in = np.asarray(mat.get("ssh", mat.get("ssh_daily_inst_filtered"))).squeeze()
        if ssh_in.ndim != 2 or XC.ndim != 2 or YC.ndim != 2:
            raise ValueError(f"In {fpath}: XC, YC, ssh must be 2-D; got {XC.shape}, {YC.shape}, {ssh_in.shape}")

        # Match longitude convention to KaRIn
        XC = _wrap_like_karin(XC, karin_lon)

        ssh = ssh_in - ssh_tmean # subtract the global time mean 

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