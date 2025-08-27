import numpy as np
import matplotlib.pyplot as plt
import JWS_SWOT_toolbox as swot
import xarray as xr
import xrft

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

        x_idx = np.arange(self.total_width) + 0.5 
        self.x_coord = self.dx * x_idx

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
        
        # 1. window and coordinates
        self.window = xr.DataArray(swot.sin2_window_func(self.track_length), dims=['line'])
        k_coords = [self.y_coord, self.x_coord] # do spectra in cpkm
        kt_coords = [self.t_coord, self.y_coord, self.x_coord] # do spectra in cpkm

        # 2. Create xarrays for analysis
        karin_ssh = xr.DataArray(self.ssh, coords=kt_coords, dims=['sample', 'line', 'pixel'])
        karin_ssha = xr.DataArray(self.ssha, coords=kt_coords, dims=['sample', 'line', 'pixel'])
        
        if hasattr(self, 'ssh_mean') and self.ssh_mean is not None:
            karin_mean = xr.DataArray(self.ssh_mean, coords=k_coords, dims=['line', 'pixel'])
            self.spec_tmean = swot.mean_power_spectrum(karin_mean, self.window, 'line', ['pixel'])
        
        if hasattr(self, 'ssh_mean_highpass') and self.ssh_mean_highpass is not None:
            karin_mean_filtered = xr.DataArray(self.ssha_mean_highpass, coords=k_coords, dims=['line', 'pixel'])
            self.spec_filt_tmean = swot.mean_power_spectrum(karin_mean_filtered, self.window, 'line', ['pixel'])
        
        if hasattr(self, 'tide') and self.tide is not None:
            karin_tide = xr.DataArray(self.tide, coords=kt_coords, dims=['sample', 'line', 'pixel'])
            self.spec_tide = swot.mean_power_spectrum(karin_tide, self.window, 'line', ['sample', 'pixel'])
        
        # 3. Remove spatial mean for anomaly spectra
        karin_spatial_mean = swot.spatial_mean(karin_ssha, ['line', 'pixel'])
        karin_anomsp = karin_ssha - karin_spatial_mean

        # 4. Perform spectral analysis using the object's own data
        self.spec_ssh = swot.mean_power_spectrum(karin_ssh, self.window, 'line', ['sample', 'pixel'])
        self.spec_ssha = swot.mean_power_spectrum(karin_ssha, self.window, 'line', ['sample', 'pixel'])
        self.spec_alongtrack_av = swot.mean_power_spectrum(karin_anomsp, self.window, 'line', ['sample', 'pixel'])
        self.spec_alongtrack_ins = swot.mean_power_spectrum(karin_anomsp, self.window, 'line', ['pixel'])
        self.spec_alongtrack_time_av = swot.mean_power_spectrum(karin_anomsp, self.window, 'line', ['sample'])

        # 5. Store wavenumbers in various useful forms 
        self.wavenumbers_ord = self.spec_alongtrack_ins.freq_line        # ordinary wavenumbers in cycles/m
        self.wavenumbers = self.wavenumbers_ord                          # we default to using the ordinary wavenumbers
        self.wavenumbers_cpkm = self.wavenumbers_ord * 1e3               # cycles/km
        self.wavenumbers_ang = self.wavenumbers_cpkm * 2 * np.pi         # angular wavenumbers in rads/m
        self.wavenumbers_length = 1 / self.wavenumbers_ord               # lengths in km

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

        y_idx = np.arange(self.track_length) + 0.5
        self.y_coord = self.dy * y_idx
        center_x_position = (self.karin.total_width * self.karin.dx) / 2.0
        self.x_coord = np.array([center_x_position])
        self.x_grid, self.y_grid = np.meshgrid(self.x_coord, self.y_coord)

        self.t_coord = np.arange(self.num_cycles)

    def compute_spectra(self):
        """Computes all power spectra for the Nadir data."""
        print("Computing Nadir spectra...")
        # 1. window and coordinates
        self.window = xr.DataArray(swot.sin2_window_func(self.track_length), dims=['nadir_line'])
        nt_coords = [self.t_coord, self.y_coord]
        
        # 2. Create xarrays for analysis
        nadir_ssh = xr.DataArray(self.ssh, coords=nt_coords, dims=['sample', 'nadir_line'])
        
        # 3. Remove spatial mean for anomaly spectra
        nadir_spatial_mean = swot.spatial_mean(nadir_ssh, ['nadir_line'])
        nadir_anomsp = nadir_ssh - nadir_spatial_mean

        # 4. Perform spectral analysis
        self.spec_ssh = swot.mean_power_spectrum(nadir_ssh, self.window, 'nadir_line', ['sample'])
        self.spec_alongtrack_av = swot.mean_power_spectrum(nadir_anomsp, self.window, 'nadir_line', ['sample'])
        self.spec_alongtrack_ins = swot.mean_power_spectrum(nadir_anomsp, self.window, 'nadir_line', [])
        
        # 5. Store wavenumbers in various useful forms 
        self.wavenumbers_ord = self.spec_alongtrack_ins.freq_nadir_line  # ordinary wavenumbers in cycles/m
        self.wavenumbers = self.wavenumbers_ord                          # we default to using the ordinary wavenumbers
        self.wavenumbers_cpkm = self.wavenumbers_ord * 1e3               # cycles/km
        self.wavenumbers_ang = self.wavenumbers_cpkm * 2 * np.pi         # angular wavenumbers in rads/m
        self.wavenumbers_length = 1 / self.wavenumbers_ord               # lengths in km
        

def init_swot_arrays(dims, lat_min, lat_max, pass_number):
    ncycles, track_length, track_length_nadir = dims
    karin = KarinData(ncycles, track_length, lat_min, lat_max, pass_number)
    nadir = NadirData(ncycles, track_length_nadir, lat_min, lat_max, pass_number)
    nadir.karin = karin
    return karin, nadir

def plot_grids(karin, nadir):
    
    plt.figure()
    plt.scatter(nadir.x_km, nadir.y_km, s=1)
    plt.scatter(karin.x_km, karin.y_km, s=1)

    # Create the scatter plot
    plt.figure(figsize=(8, 10))
    plt.scatter(karin.x_obs_grid*1e-3, karin.y_obs_grid*1e-3, s=1, alpha=0.9)
    plt.scatter(karin.x_grid*1e-3, karin.y_grid*1e-3, s=1, alpha=0.1)
    plt.scatter(nadir.x_grid*1e-3, nadir.y_grid*1e-3, s=1, alpha=1.0)

    # Add labels and a title for clarity
    plt.title('Scatter Plot of Synthetic Grid Coordinates')
    plt.xlabel('Cross-Track Distance (km)')
    plt.ylabel('Along-Track Distance (km)')

    # Add grid lines and ensure correct aspect ratio
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    # Display the plot
    plt.tight_layout()
    plt.show()