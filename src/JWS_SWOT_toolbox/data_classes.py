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
    
    def distances(self, samp_indx = 1):  # some index to sample the distances where data is nonzero
        self.dx = 1e3 * swot.haversine_dx(self.lon[samp_indx, :, 0], self.lat[samp_indx, :, 0])
        self.dy = 1e3 * swot.haversine_dx(self.lon[samp_indx, 0, :], self.lat[samp_indx, 0, :])
        print(f"KaRIn spacing: dx = {self.dx:.2f} m, dy = {self.dy:.2f} m")

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
        
    def distances(self, samp_indx = 1):
        self.dy   = 1e3 * swot.haversine_dx(self.lon[samp_indx, :], self.lat[samp_indx, :])
        print(f"Nadir spacing: dy = {self.dy:.2f} m")
    
    def coordinates(self):
        self.y_coord    = self.dy * np.arange(self.track_length)
        self.t_coord    = np.arange(self.num_cycles)

def init_swot_arrays(dims):
    ncycles, track_length, track_length_nadir = dims
    karin = KarinData(ncycles, track_length)
    nadir = NadirData(ncycles, track_length_nadir)
    return karin, nadir
