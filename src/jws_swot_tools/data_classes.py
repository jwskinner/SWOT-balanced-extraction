import numpy as np
import matplotlib.pyplot as plt
import jws_swot_tools as swot
import xarray as xr
import xrft

class KarinData:
    def __init__(self, num_cycles, track_length, lat_min, lat_max, pass_number):
        self.swath_width = 25
        self.middle_width = 9 
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

        # --- Spectra attributes ---
        self.spec_ssh = None
        self.spec_tmean = None
        self.spec_filt_tmean = None
        self.spec_ssha = None
        self.spec_alongtrack_av = None
        self.spec_alongtrack_ins = None
        self.spec_tide = None
        self.wavenumbers = None
        self.time = None                 # seconds (num_cycles, track_length)
        self.time_dt = None              # np.datetime64[ns] (num_cycles, track_length)
        self.cycle_dates = None          # np.datetime64[ns] (num_cycles,)

    def coordinates(self):
        # Convert the lats and lons to a grid in km
        self.x_km, self.y_km = swot.convert_to_xy_grid(self) # returns the euclidian grid spacing in km from the lat/lon coordinates
        x_min, x_max = np.nanmin(self.x_km), np.nanmax(self.x_km)
        y_min, y_max = np.nanmin(self.y_km), np.nanmax(self.y_km)

        # Print the range and the calculated span
        print(f"X grid range (km): {x_min:.2f} to {x_max:.2f} (span: {x_max - x_min:.2f} km)")
        print(f"Y grid range (km): {y_min:.2f} to {y_max:.2f} (span: {y_max - y_min:.2f} km)")

        track_length = np.nanmax(self.y_km) - np.nanmin(self.y_km)
        x_row = self.x_km[self.x_km.shape[0] // 2, :]
        sorted_x = np.sort(x_row[~np.isnan(x_row)])
        total_swath_width = (sorted_x.max() - sorted_x.min())
        swath_width = total_swath_width / 2
        print(f"Track Length: {track_length:.2f} km")
        print(f"Swath Width: {swath_width:.2f} km")

        # Ensure the grids are at least 2D
        if self.x_km.ndim >= 2:
            # Calculate dy_km (along-track spacing)
            # Find distance between points in adjacent rows (axis=-2)
            delta_x_y = np.diff(self.x_km, axis=-2)
            delta_y_y = np.diff(self.y_km, axis=-2)
            distances_y = np.sqrt(delta_x_y**2 + delta_y_y**2)
            self.dy_km = np.nanmean(distances_y)
            self.dy = self.dy_km * 1.0e3

            # Calculate dx_km (cross-track spacing)
            # Find distance between points in adjacent columns (axis=-1)
            delta_x_x = np.diff(self.x_km, axis=-1)
            delta_y_x = np.diff(self.y_km, axis=-1)
            distances_x = np.sqrt(delta_x_x**2 + delta_y_x**2)
            self.dx_km = np.nanmean(distances_x)
            self.dx = self.dx_km * 1.0e3

            print(f"Karin spacing: dx = {self.dx_km:.2f} km, dy = {self.dy_km:.2f} km")

        y_idx = np.arange(self.track_length) + 0.5 
        self.y_coord = self.dy * y_idx
        self.y_coord_km = self.y_coord * 1e-3

        x_idx = np.arange(self.total_width) + 0.5 
        self.x_coord = self.dx * x_idx
        self.x_coord_km = self.x_coord * 1e-3

        self.t_coord = np.arange(self.num_cycles)
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)

        left = x_idx[:self.swath_width]
        right = x_idx[self.swath_width + self.middle_width:]
        self.x_obs = self.dx * np.concatenate((left, right))
        self.y_obs = self.y_coord
        self.x_obs_grid, self.y_obs_grid = np.meshgrid(self.x_obs, self.y_obs)

    def compute_spectra(self):
        """Computes all power spectra for the Karin data."""
        print("Computing KaRIn spectra...")
        
        # 1. dims, window and coordinates
        dim_name = 'line' # along-track
        avg_dims = ['sample', 'pixel'] # Dimensions to average over: Time, Cross-track
        self.window = xr.DataArray(swot.sin2_window_func(self.track_length), dims=[dim_name]) #spectrum window function
        
        k_coords = [self.y_coord_km, self.x_coord_km] 
        kt_coords = [self.t_coord, self.y_coord_km, self.x_coord_km] 

        # 2. Create xarrays for analysis, we do the spectra in cm so the output is [cm/cpkm]
        karin_ssh = xr.DataArray(self.ssh*100, coords=kt_coords, dims=['sample', 'line', 'pixel'])
        karin_ssha = xr.DataArray(self.ssha*100, coords=kt_coords, dims=['sample', 'line', 'pixel'])
        
        if hasattr(self, 'ssh_mean') and self.ssh_mean is not None:
            karin_mean = xr.DataArray(self.ssh_mean*100, coords=k_coords, dims=['line', 'pixel'])
            self.spec_tmean = swot.mean_power_spectrum(karin_mean, self.window, 'line', ['pixel'])
        
        if hasattr(self, 'ssh_mean_highpass') and self.ssh_mean_highpass is not None:
            karin_mean_filtered = xr.DataArray(self.ssh_mean_highpass*100, coords=k_coords, dims=['line', 'pixel'])
            self.spec_filt_tmean = swot.mean_power_spectrum(karin_mean_filtered, self.window, 'line', ['pixel'])
        
        if hasattr(self, 'tide') and self.tide is not None:
            karin_tide = xr.DataArray(self.tide*100, coords=kt_coords, dims=['sample', 'line', 'pixel'])
            self.spec_tide = swot.mean_power_spectrum(karin_tide, self.window, 'line', ['sample', 'pixel'])
        
        # 3. compute and remove spatial mean for anomaly 
        karin_spatial_mean = swot.spatial_mean(karin_ssha, ['line', 'pixel'])
        karin_anomsp = karin_ssha - karin_spatial_mean

        # 4. Across-Track and Time-averaged spectra 
        self.spec_ssh = swot.mean_power_spectrum(karin_ssh, self.window, dim_name, avg_dims)
        self.spec_ssha = swot.mean_power_spectrum(karin_ssha, self.window, dim_name, avg_dims)
        self.spec_alongtrack_av = swot.mean_power_spectrum(karin_anomsp, self.window, dim_name, avg_dims)

        # 5. Across-Track Averaged Instantaneous Spectra
        self.spec_alongtrack_ins = swot.mean_power_spectrum(karin_anomsp, self.window, dim_name, ['pixel'])

        # 6. Time-Averaged Across-Track Spectra
        self.spec_alongtrack_time_av = swot.mean_power_spectrum(karin_anomsp, self.window, dim_name, ['sample'])

        # 5. Store wavenumbers in various useful forms 
        waves = swot.get_wavenumbers(self.spec_alongtrack_ins, dim_name)
        self.wavenumbers_ord = waves['ord']
        self.wavenumbers_m = waves['m']
        self.wavenumbers_cpkm = waves['cpkm']
        self.wavenumbers_ang = waves['ang']
        self.wavenumbers_length = waves['length']

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
        self.time = None                 # seconds (num_cycles, track_length_nadir)
        self.time_dt = None              # np.datetime64[ns] (num_cycles, track_length_nadir)
        self.cycle_dates = None          # np.datetime64[ns] (num_cycles,)
        self.lat_full = None
        self.lon_full = None
    
    def coordinates(self):
        self.x_km, self.y_km = swot.convert_to_xy_grid(self, self.karin)

        delta_x = np.diff(self.x_km)
        delta_y = np.diff(self.y_km)
        distances = np.sqrt(delta_x**2 + delta_y**2)
        self.dy_km = np.nanmean(distances)
        self.dy = self.dy_km * 1.0e3 # Convert to meters
        print(f"Nadir spacing: dy = {self.dy_km:.2f} km")

        y_idx = np.arange(self.track_length) + 0.5 # include the extra pixel from half-grid shift
        self.y_coord = self.dy * y_idx 
        self.y_coord_km = self.y_coord * 1e-3
        
        center_x_position = (self.karin.total_width * self.karin.dx) / 2.0
        self.x_coord = np.array([center_x_position])
        self.x_coord_km = self.x_coord * 1e-3
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)

        self.t_coord = np.arange(self.num_cycles)

    def compute_spectra(self):
        """Computes all power spectra for the Nadir data."""
        print("Computing Nadir spectra...")
        
        # 1. window and coordinates
        dim_name = 'nadir_line'
        avg_dims = ['sample']
        self.window = xr.DataArray(swot.sin2_window_func(self.track_length), dims=dim_name)
        nt_coords = [self.t_coord, self.y_coord_km]
        
        # 2. Create xarrays for analysis in cm^2/cpkm
        nadir_ssh = xr.DataArray(self.ssh * 100, coords=nt_coords, dims=['sample', 'nadir_line'])
        nadir_ssha = xr.DataArray(self.ssha * 100, coords=nt_coords, dims=['sample', 'nadir_line'])
        
        # 3. Remove spatial mean for anomaly spectra
        nadir_spatial_mean = swot.spatial_mean(nadir_ssh, ['nadir_line'])
        nadir_anomsp = nadir_ssh - nadir_spatial_mean
        nadir_anomspa = nadir_ssha - nadir_spatial_mean

        # 4. Time and Across-Track Av. SSH and SSHA spectra
        self.spec_ssh = swot.mean_power_spectrum(nadir_ssh, self.window, dim_name, avg_dims)
        self.spec_ssha = swot.mean_power_spectrum(nadir_ssha, self.window, dim_name, avg_dims)
        self.spec_alongtrack_av = swot.mean_power_spectrum(nadir_anomsp, self.window, dim_name, avg_dims)
        self.spec_alongtrack_ava = swot.mean_power_spectrum(nadir_anomspa, self.window, dim_name, avg_dims)
        self.spec_alongtrack_ins = swot.mean_power_spectrum(nadir_anomsp, self.window, dim_name, [])
        
        # 5. Store wavenumbers in various useful forms 
        waves = swot.get_wavenumbers(self.spec_alongtrack_ins, dim_name)
        self.wavenumbers_ord = waves['ord']
        self.wavenumbers_m = waves['m']
        self.wavenumbers_cpkm = waves['cpkm']
        self.wavenumbers_ang = waves['ang']
        self.wavenumbers_length = waves['length']
        

def init_swot_arrays(dims, lat_min, lat_max, pass_number):
    ncycles, track_length, track_length_nadir = dims
    karin = KarinData(ncycles, track_length, lat_min, lat_max, pass_number)
    nadir = NadirData(ncycles, track_length_nadir, lat_min, lat_max, pass_number)
    nadir.karin = karin
    return karin, nadir