from math import asin, sin, cos, sqrt, atan2, radians
import numpy as np
import pyproj

# Computes the distance between consecutive points in a track given their longitude and latitude.
def distance(lon,lat):
    R = 6373.0
    dis = np.empty(lon.shape)
    for n in range(len(lon)-1):
        lat1 = radians(lat[n])
        lon1 = radians(lon[n])
        lat2 = radians(lat[n+1])
        lon2 = radians(lon[n+1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        dis[n] = R * c
    return np.nanmean(dis)

def haversine_dx(lon, lat): 
    ''' returns the grid spacing between points based on the lat lon coords'''
    R = 6371.0  # Earth's radius in km
    lat1, lon1 = radians(lat[0]), radians(lon[0])
    lat2, lon2 = radians(lat[-1]), radians(lon[-1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
    total_distance = R * c
    return total_distance / (len(lon) - 1)

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r

def convert_to_xy_grid(data_object, origin_object=None):
    lat = data_object.lat
    lon = data_object.lon
    
    # Determine the origin for the coordinate system
    if origin_object is not None: # use the karin data object to center the nadir
        # Use the second data object for the origin
        lat_origin = np.nanmin(origin_object.lat) 
        lon_origin = np.nanmin(origin_object.lon) 
    else:
        # Default behavior: calculate origin from the primary data object
        lat_origin = np.nanmin(lat)
        lon_origin = np.nanmin(lon)

    y_km = haversine_distance(lon_origin, lat_origin, lon_origin, lat)
    x_km = haversine_distance(lon_origin, lat, lon, lat)
    return x_km, y_km

def projected_distance(lon1, lat1, lon2, lat2, utm_zone=None):
    """
    Calculate signed distance using UTM projection
    """
    # Define UTM projection (adjust zone as needed)
    if utm_zone is None:
        utm_zone = int((lon1 + 180) / 6) + 1
    
    proj_utm = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    
    # Transform to UTM
    x1, y1 = pyproj.transform(proj_latlon, proj_utm, lon1, lat1)
    x2, y2 = pyproj.transform(proj_latlon, proj_utm, lon2, lat2)
    
    # Calculate signed distances
    dx = x2 - x1
    dy = y2 - y1
    
    return np.array(dx), np.array(dy), np.array(np.sqrt(dx**2 + dy**2))

# Calculating the distance in meters between consecutive points in a track given their longitude and latitude.
def km2m(distance):
    return distance * 1e3