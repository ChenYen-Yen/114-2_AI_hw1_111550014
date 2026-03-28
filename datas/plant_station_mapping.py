# find closest stations for each plant, sorted by distance
import math
import csv

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

def find_closest_stations(plant_lat, plant_lon, stations, top_n=5):
    """
    找到距離電站最近的N個測站，按距離排序
    返回: [(station_name, distance), ...] 按距離升序排列
    """
    distances = []
    
    for station_name, coords in stations.items():
        station_lat = coords['lat']
        station_lon = coords['lon']
        distance = calculate_distance(plant_lat, plant_lon, station_lat, station_lon)
        distances.append((station_name, distance))
    
    # 按距離排序
    distances.sort(key=lambda x: x[1])
    
    # 回傳前N個
    return distances[:top_n]

if __name__ == "__main__":
    # read station data
    stations = {}
    with open('./datas/station_positions.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_name = row['站名'].strip()
            lat = float(row['緯度'].strip())
            lon = float(row['經度'].strip())
            stations[station_name] = {'lat': lat, 'lon': lon}
    
    print(f"讀取 {len(stations)} 個測站\n")
    
    # read plant data and find closest stations for each
    results = []
    with open('./datas/powerplant_position.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_name = row['站名'].strip()
            plant_lat = float(row['緯度'].strip())
            plant_lon = float(row['經度'].strip())
            
            closest_stations = find_closest_stations(plant_lat, plant_lon, stations, top_n=5)
            
            # 建立結果行（只包含站點名稱，不包含距離）
            result = {'powerplant': plant_name}
            for rank, (station_name, distance) in enumerate(closest_stations, 1):
                result[f'station_{rank}'] = station_name
            
            results.append(result)
            
            # 打印最近的3個站
            top_3 = closest_stations[:3]
            stations_str = ' -> '.join([f"{name}({dist:.2f}km)" for name, dist in top_3])
            print(f"{plant_name:40s} {stations_str}")
    
    # 設定CSV欄位名（只有站點名稱）
    fieldnames = ['powerplant']
    for i in range(1, 6):
        fieldnames.append(f'station_{i}')
    
    # export to csv
    with open('./datas/plant_station_mapping.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n共處理 {len(results)} 個電廠")
    print("結果已儲存至 ./datas/plant_station_mapping.csv")
    print("\n檔案格式說明:")
    print("  - powerplant: 電廠名稱")
    print("  - station_1~5: 按距離排序的測站名稱")