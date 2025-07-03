import numpy as np
import matplotlib.pyplot as plt
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
    
    # def distances(self, samp_indx=1):
    #     # 3D case
    #     if self.lon.ndim == 3:
    #         for i in range(self.lon.shape[0]):
    #             lon_row = self.lon[i, :, 0]
    #             lat_row = self.lat[i, :, 0]
    #             lon_col = self.lon[i, 0, :]
    #             lat_col = self.lat[i, 0, :]
    #             if np.any(np.isfinite(lon_row)) and np.any(np.isfinite(lat_row)) and \
    #             np.any(np.isfinite(lon_col)) and np.any(np.isfinite(lat_col)):
    #                 samp_indx = i
    #                 self.dx = 1e3 * swot.haversine_dx(lon_row, lat_row)
    #                 self.dy = 1e3 * swot.haversine_dx(lon_col, lat_col)
    #                 print(f"Using index {samp_indx}. KaRIn spacing: dx = {self.dx:.2f} m, dy = {self.dy:.2f} m")
    #                 return

    #     elif self.lon.ndim == 2:
    #         # Find a finite row
    #         for i in range(self.lon.shape[0]):
    #             lon_row = self.lon[i, :]
    #             lat_row = self.lat[i, :]
    #             mask_row = np.isfinite(lon_row) & np.isfinite(lat_row)
    #             if np.count_nonzero(mask_row) >= 2:
    #                 lon_row = lon_row[mask_row]
    #                 lat_row = lat_row[mask_row]
    #                 break
    #         else:
    #             raise RuntimeError("No valid finite row found in 2D array.")

    #         # Find a finite column
    #         for j in range(self.lon.shape[1]):
    #             lon_col = self.lon[:, j]
    #             lat_col = self.lat[:, j]
    #             mask_col = np.isfinite(lon_col) & np.isfinite(lat_col)
    #             if np.count_nonzero(mask_col) >= 2:
    #                 lon_col = lon_col[mask_col]
    #                 lat_col = lat_col[mask_col]
    #                 break
    #         else:
    #             raise RuntimeError("No valid finite column found in 2D array.")

    #         self.dy = 1e3 * swot.haversine_dx(lon_row, lat_row)
    #         self.dx = 1e3 * swot.haversine_dx(lon_col, lat_col)
    #         print(f"KaRIn spacing: dx = {self.dx:.2f} m, dy = {self.dy:.2f} m")
    #         return

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
        
    # def distances(self, samp_indx=1):
    #     if self.lon.ndim == 2:
    #         # Case: 2D arrays of shape (ntime, track_length)
    #         for i in range(self.lon.shape[0]):
    #             if not np.any(np.isfinite(self.ssh[i, :])):
    #                 continue
    #             if not np.any(np.isfinite(self.lon[i, :])) or not np.any(np.isfinite(self.lat[i, :])):
    #                 continue

    #             dy_vals = 1e3 * swot.haversine_dx(self.lon[i, :], self.lat[i, :])
    #             avg_dy = np.nanmean(dy_vals)
    #             if not np.isnan(avg_dy):
    #                 self.dy = avg_dy
    #                 print(f"Using index {i}. Nadir spacing: dy = {self.dy:.2f} m")
    #                 return

    #     elif self.lon.ndim == 1:
    #         # Case: 1D arrays of shape (track_length,)
    #         if not np.any(np.isfinite(self.ssh)):
    #             raise RuntimeError("No valid SSH data for nadir.")

    #         if not np.any(np.isfinite(self.lon)) or not np.any(np.isfinite(self.lat)):
    #             raise RuntimeError("Invalid lat/lon in nadir.")

    #         dy_vals = 1e3 * swot.haversine_dx(self.lon, self.lat)
    #         avg_dy = np.nanmean(dy_vals)
    #         if not np.isnan(avg_dy):
    #             self.dy = avg_dy
    #             print(f"Nadir spacing: dy = {self.dy:.2f} m")
    #             return

    #     else:
    #         raise ValueError("nadir.lon must be 1D or 2D")

    #     raise RuntimeError("No valid sampling index with finite dy found.")
    
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