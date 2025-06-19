import numpy as np
import JWS_SWOT_toolbox as swot

class KarinData:
    def __init__(self, num_cycles, track_length, lat_min, lat_max, pass_number):
        self.swath_width = 25
        self.middle_width = 10 
        self.num_cycles = num_cycles
        self.track_length = track_length
        self.total_width = 2 * self.swath_width + self.middle_width
        self.lat  = np.full((num_cycles, track_length, self.total_width), np.nan)
        self.lon  = np.full_like(self.lat, np.nan)
        self.ssh = np.full_like(self.lat, np.nan)
        self.tide = np.full_like(self.lat, np.nan)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.pass_number = pass_number
    
    def distances(self, samp_indx=1):
        for i in range(self.lon.shape[0]):
            lon_row = self.lon[i, :, 0]
            lat_row = self.lat[i, :, 0]
            lon_col = self.lon[i, 0, :]
            lat_col = self.lat[i, 0, :]
            if np.any(np.isfinite(lon_row)) and np.any(np.isfinite(lat_row)) and \
            np.any(np.isfinite(lon_col)) and np.any(np.isfinite(lat_col)):
                samp_indx = i
                self.dx = 1e3 * swot.haversine_dx(lon_row, lat_row)
                self.dy = 1e3 * swot.haversine_dx(lon_col, lat_col)
                print(f"Using index {samp_indx}. KaRIn spacing: dx = {self.dx:.2f} m, dy = {self.dy:.2f} m")
                return
        raise RuntimeError("No valid sampling index with finite values found.")

    def coordinates(self):
        y_idx       = np.arange(0.5, self.track_length, 1.0) 
        self.y_coord = self.dy * y_idx

        x_idx       = np.arange(0.5, self.total_width, 1.0)
        self.x_coord = self.dx * x_idx

        left  = x_idx[:self.swath_width]
        right = x_idx[self.swath_width + self.middle_width:]
        self.x_obs = self.dx * np.concatenate((left, right))

        self.t_coord = np.arange(self.num_cycles)
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)


class NadirData:
    def __init__(self, num_cycles, track_length_nadir, lat_min, lat_max, pass_number):
        self.num_cycles = num_cycles
        self.track_length = track_length_nadir
        self.ssh = np.full((num_cycles, track_length_nadir), np.nan)
        self.ssha = np.full((num_cycles, track_length_nadir), np.nan)
        self.lat  = np.full_like(self.ssha, np.nan)
        self.lon  = np.full_like(self.ssha, np.nan)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.pass_number = pass_number
        
    def distances(self, samp_indx=1):
        for i in range(self.lon.shape[0]):
            # skip lines with no data
            if not np.any(np.isfinite(self.ssh[i, :])):
                continue

            # compute spacing (could be scalar or array)
            dy_vals = 1e3 * swot.haversine_dx(self.lon[i, :], self.lat[i, :])
            avg_dy = np.nanmean(np.atleast_1d(dy_vals))

            # if mean is valid, set and exit
            if not np.isnan(avg_dy):
                self.dy = avg_dy
                print(f"Using index {i}. Nadir spacing: dy = {self.dy:.2f} m")
                return
        raise RuntimeError("No valid sampling index with any finite dy found.")
    
    def coordinates(self):
        y_idx       = np.arange(0.5, self.track_length, 1.0)     # length = track_length
        self.y_coord = self.dy * y_idx + getattr(self, 'offset', 0.0)

        x_idx       = np.arange(0.5, self.karin.total_width, 1.0)
        centre_idx  = self.karin.swath_width + self.karin.middle_width / 2
        centre_dist = self.karin.dx * x_idx[int(centre_idx)]
        self.x_coord = np.full(self.track_length, centre_dist)

        self.t_coord = np.arange(self.num_cycles)
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)


def init_swot_arrays(dims, lat_min, lat_max, pass_number):
    ncycles, track_length, track_length_nadir = dims
    karin = KarinData(ncycles, track_length, lat_min, lat_max, pass_number)
    nadir = NadirData(ncycles, track_length_nadir, lat_min, lat_max, pass_number)
    nadir.karin = karin
    return karin, nadir
