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

def haversine(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance, returns distance in km
    """
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

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