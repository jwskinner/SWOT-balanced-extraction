import numpy as np
import JWS_SWOT_toolbox as swot

class KarinData:
    def __init__(self, num_cycles, track_length):
        self.swath_width = 25
        self.middle_width = 10 
        self.num_cycles = num_cycles
        self.track_length = track_length
        self.total_width = 2 * self.swath_width + self.middle_width
        self.lat  = np.full((num_cycles, track_length, self.total_width), np.nan)
        self.lon  = np.full_like(self.lat, np.nan)
        self.ssh = np.full_like(self.lat, np.nan)
        self.time = np.full((num_cycles, track_length), np.nan)
        self.tide = np.full_like(self.lat, np.nan)
    
    def distances(self, samp_indx=10):
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

    def coordinates(self): # Returns the full grid coordinates
        self.y_coord    = self.dy * np.arange(self.track_length)
        self.x_coord    = self.dx * np.arange(self.total_width) # Total points in grid 
        self.x_obs      = self.dx * np.arange(self.total_width - self.middle_width) # Observed points in grid (i.e., remove the gap)
        self.t_coord    = np.arange(self.num_cycles)
        self.x_grid, self.y_grid     = np.meshgrid(self.x_coord, self.y_coord)


class NadirData:
    def __init__(self, num_cycles, track_length_nadir):
        self.num_cycles = num_cycles
        self.track_length = track_length_nadir
        self.ssh = np.full((num_cycles, track_length_nadir), np.nan)
        self.ssha = np.full((num_cycles, track_length_nadir), np.nan)
        self.lat  = np.full_like(self.ssha, np.nan)
        self.lon  = np.full_like(self.ssha, np.nan)
        
    def distances(self, samp_indx=10):
        for i in range(self.lon.shape[0]):
            lon_line = self.lon[i, :]
            lat_line = self.lat[i, :]
            if np.any(np.isfinite(lon_line)) and np.any(np.isfinite(lat_line)):
                samp_indx = i
                self.dy = 1e3 * swot.haversine_dx(lon_line, lat_line)
                print(f"Using index {samp_indx}. Nadir spacing: dy = {self.dy:.2f} m")
                return
        raise RuntimeError("No valid sampling index with any finite values found.")
    
    def coordinates(self):
        self.y_coord    = self.dy * np.arange(self.track_length)
        self.t_coord    = np.arange(self.num_cycles)

def init_swot_arrays(dims):
    ncycles, track_length, track_length_nadir = dims
    karin = KarinData(ncycles, track_length)
    nadir = NadirData(ncycles, track_length_nadir)
    return karin, nadir
