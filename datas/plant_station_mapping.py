# find closest station for each plant
import math

def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate distance between two lat/lon points
    # https://blog.csdn.net/weixin_42146230/article/details/147923997
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_closest_station(plant_lat, plant_lon, stations):
    closest_station = None
    min_distance = float('inf')
    
    for station_name, coords in stations.items():
        station_lat = coords['lat']
        station_lon = coords['lon']
        distance = calculate_distance(plant_lat, plant_lon, station_lat, station_lon)
        
        if distance < min_distance:
            min_distance = distance
            closest_station = station_name
            
    return closest_station, min_distance

if __name__ == "__main__":
    import csv
    
    # read station data
    stations = {}
    with open('./datas/station_positions.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_name = row['站名'].strip()
            lat = float(row['緯度'].strip())
            lon = float(row['經度'].strip())
            stations[station_name] = {'lat': lat, 'lon': lon}
    
    # read plant data and find closest station for each
    results = []
    with open('./datas/powerplant_position.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_name = row['站名'].strip()
            plant_lat = float(row['緯度'].strip())
            plant_lon = float(row['經度'].strip())
            
            closest_station, distance = find_closest_station(plant_lat, plant_lon, stations)
            
            results.append({
                'powerplant': plant_name,
                'closest_station': closest_station,
            })
            
            print(f"{plant_name} -> {closest_station} ({distance:.2f} km)")
    
    # export to csv
    with open('./datas/plant_station_mapping.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['powerplant', 'closest_station']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTotal plants: {len(results)}")
    print("Results exported to ./datas/plant_station_mapping.csv")