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
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    total_distance = R * c
    return total_distance / (len(lon) - 1)

# Calculating the distance in meters between consecutive points in a track given their longitude and latitude.
def km2m(distance):
    return distance * 1e3