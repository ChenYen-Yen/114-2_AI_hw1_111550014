# for each row in 001.csv, find the closest station with plant_station_mapping.csv
# and save the monthly weather data in C-B0026-002 and C-B0026-2025 to the row
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import os

def safe_float(value):
    """
    安全地將值轉換為float，如果無法轉換則返回None
    處理特殊值如'T'(Trace/痕量), 'M'(Missing/缺失), '-'等
    """
    if value is None:
        return None
    
    value_str = str(value).strip()
    if not value_str:
        return None
    
    # 嘗試直接轉換
    try:
        return float(value_str)
    except ValueError:
        # 如果失敗，返回None（不會導致錯誤）
        return None

def parse_weather_xml(xml_file):
    """
    解析氣象XML檔案，提取氣象站名稱和該月份的所有氣象數據
    返回:
      [(station_name, year_month, weather_data_dict), ...]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 定義命名空間
    ns = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
    
    results = []
    
    # 找所有location元素
    for location in root.findall('.//cwa:location', ns):
        # 找station元素
        station_elem = location.find('cwa:station', ns)
        if station_elem is None:
            continue
            
        station_name_elem = station_elem.find('cwa:StationName', ns)
        if station_name_elem is None:
            continue
            
        station_name = station_name_elem.text
        
        # 找stationObsStatistics元素
        stat_elem = location.find('cwa:stationObsStatistics', ns)
        if stat_elem is None:
            continue
        
        year_month_elem = stat_elem.find('cwa:YearMonth', ns)
        if year_month_elem is None:
            continue
            
        year_month = year_month_elem.text
        
        # 提取氣象數據
        weather_data = {}
        
        # 氣溫
        air_temp = stat_elem.find('cwa:AirTemperature', ns)
        if air_temp is not None:
            monthly = air_temp.find('cwa:monthly', ns)
            if monthly is not None:
                mean = monthly.find('cwa:Mean', ns)
                if mean is not None:
                    weather_data['溫度_平均'] = safe_float(mean.text)
                max_temp = monthly.find('cwa:Maximum', ns)
                if max_temp is not None:
                    weather_data['溫度_最高'] = safe_float(max_temp.text)
                max_date = monthly.find('cwa:MaximumDate', ns)
                if max_date is not None:
                    weather_data['溫度_最高日期'] = max_date.text if max_date.text else None
                min_temp = monthly.find('cwa:Minimum', ns)
                if min_temp is not None:
                    weather_data['溫度_最低'] = safe_float(min_temp.text)
                min_date = monthly.find('cwa:MinimumDate', ns)
                if min_date is not None:
                    weather_data['溫度_最低日期'] = min_date.text if min_date.text else None

        # 降水
        precip = stat_elem.find('cwa:Precipitation', ns)
        if precip is not None:
            monthly = precip.find('cwa:monthly', ns)
            if monthly is not None:
                accum = monthly.find('cwa:Accumulation', ns)
                if accum is not None:
                    weather_data['累積雨量'] = safe_float(accum.text)
                GE01day = monthly.find('cwa:GE01Days', ns)
                if GE01day is not None:
                    weather_data['降雨日數'] = safe_float(GE01day.text)

        # 風速
        wind_speed = stat_elem.find('cwa:WindSpeed', ns)
        if wind_speed is not None:
            monthly = wind_speed.find('cwa:monthly', ns)
            if monthly is not None:
                max_wind = monthly.find('cwa:Maximum', ns)
                if max_wind is not None:
                    weather_data['最大十分鐘風'] = safe_float(max_wind.text)
                max_date = monthly.find('cwa:MaximumDate', ns)
                if max_date is not None:
                    weather_data['最大十分鐘風速日期'] = max_date.text if max_date.text else None
        wind_dir = stat_elem.find('cwa:WindDirection', ns)
        if wind_dir is not None:
            monthly = wind_dir.find('cwa:monthly', ns)
            if monthly is not None:
                max_dir = monthly.find('cwa:Maximum', ns)
                if max_dir is not None:
                    weather_data['最大十分鐘風向'] = safe_float(max_dir.text)

        peak_gust = stat_elem.find('cwa:PeakGustSpeed', ns)
        if peak_gust is not None:
            monthly = peak_gust.find('cwa:monthly', ns)
            if monthly is not None:
                max_wind = monthly.find('cwa:Maximum', ns)
                if max_wind is not None:
                    weather_data['最大瞬間風'] = safe_float(max_wind.text)
                max_date = monthly.find('cwa:MaximumDate', ns)
                if max_date is not None:
                    weather_data['最大瞬間風日期'] = max_date.text if max_date.text else None
        wind_dir = stat_elem.find('cwa:PeakGustDirection', ns)
        if wind_dir is not None:
            monthly = wind_dir.find('cwa:monthly', ns)
            if monthly is not None:
                max_dir = monthly.find('cwa:Maximum', ns)
                if max_dir is not None:
                    weather_data['最大瞬間風向'] = safe_float(max_dir.text)

        # 相對濕度
        humidity = stat_elem.find('cwa:RelativeHumidity', ns)
        if humidity is not None:
            monthly = humidity.find('cwa:monthly', ns)
            if monthly is not None:
                mean = monthly.find('cwa:Mean', ns)
                if mean is not None:
                    weather_data['相對濕度_平均'] = safe_float(mean.text)
                min_humidity = monthly.find('cwa:Minimum', ns)
                if min_humidity is not None:
                    weather_data['相對濕度_最低'] = safe_float(min_humidity.text)
                min_date = monthly.find('cwa:MinimumDate', ns)
                if min_date is not None:
                    weather_data['相對濕度_最低日期'] = min_date.text if min_date.text else None
        
        # 氣壓
        air_pressure = stat_elem.find('cwa:AirPressure', ns)
        if air_pressure is not None:
            monthly = air_pressure.find('cwa:monthly', ns)
            if monthly is not None:
                mean = monthly.find('cwa:Mean', ns)
                if mean is not None:
                    weather_data['氣壓_平均'] = safe_float(mean.text)
        
        # 日照時數
        sunshine = stat_elem.find('cwa:SunshineDuration', ns)
        if sunshine is not None:
            monthly = sunshine.find('cwa:monthly', ns)
            if monthly is not None:
                total = monthly.find('cwa:Total', ns)
                if total is not None:
                    weather_data['日照時數'] = safe_float(total.text)
        
        results.append((station_name, year_month, weather_data))
    
    return results

def load_all_weather_data(base_path):
    """
    讀取所有XML檔案，建立{station_name: {year_month: weather_data}}的字典
    """
    weather_dict = defaultdict(lambda: defaultdict(dict))
    
    # 讀取C-B0026-002檔案
    data_dir_002 = Path(base_path) / 'C-B0026-002'
    if data_dir_002.exists():
        for xml_file in sorted(data_dir_002.glob('mn_Report_*.xml')):
            print(f"正在處理: {xml_file.name}")
            try:
                results = parse_weather_xml(xml_file)
                for station_name, year_month, weather_data in results:
                    weather_dict[station_name][year_month] = weather_data
            except Exception as e:
                print(f"  錯誤: {e}")
    
    # 讀取C-B0025-2025檔案
    data_dir_2025 = Path(base_path) / 'C-B0025-2025'
    if data_dir_2025.exists():
        for xml_file in sorted(data_dir_2025.glob('mn_Report_*.xml')):
            print(f"正在處理: {xml_file.name}")
            try:
                results = parse_weather_xml(xml_file)
                for station_name, year_month, weather_data in results:
                    weather_dict[station_name][year_month] = weather_data
            except Exception as e:
                print(f"  錯誤: {e}")
    
    return weather_dict

def combine_data(output_path='../test_output/combined_data.csv'):
    """
    主函式：合併所有數據
    使用plant_station_mapping中多個候選測站，優先使用有數據的測站
    """
    # 讀取必要的CSV檔案
    print("讀取001.csv...")
    df_power = pd.read_csv('./datas/001.csv')
    
    print("讀取plant_station_mapping.csv...")
    df_mapping = pd.read_csv('./datas/plant_station_mapping.csv')
    
    # 讀取所有氣象數據
    print("讀取氣象XML檔案...")
    weather_data = load_all_weather_data('./datas')
    
    print("\n開始合併數據...")
    
    # 建立年月字符串
    df_power['YearMonth'] = df_power['年度/Year'].astype(str) + '-' + df_power['月份/Month'].astype(str).str.zfill(2)
    
    # 添加用於記錄實際使用的測站名稱
    df_power['closest_station'] = None
    
    # 初始化氣象欄位
    weather_columns = set()
    
    # 逐行合併數據
    for idx, row in df_power.iterrows():
        plant_name = row['發電站名稱/Name of The Power Station']
        year_month = row['YearMonth']
        
        # 找到該電廠的映射資料
        plant_mapping = df_mapping[df_mapping['powerplant'] == plant_name]
        if plant_mapping.empty:
            continue
        
        # 嘗試每個候選測站（優先級: station_1 > station_2 > ... > station_5）
        station_found = False
        for rank in range(1, 6):
            station_col = f'station_{rank}'
            if station_col not in plant_mapping.columns:
                continue
            
            station_name = plant_mapping.iloc[0][station_col]
            if pd.isna(station_name):
                continue
            
            station_name = str(station_name).strip()
            
            # 檢查該測站是否有該月份的數據
            if station_name in weather_data and year_month in weather_data[station_name]:
                weather_info = weather_data[station_name][year_month]
                
                # 將氣象數據添加到該行
                for key, value in weather_info.items():
                    if key not in df_power.columns:
                        df_power[key] = None
                        weather_columns.add(key)
                    df_power.at[idx, key] = value
                
                # 記錄使用的測站名稱
                df_power.at[idx, 'closest_station'] = station_name
                station_found = True
                break
        
        if not station_found:
            # 如果沒有找到任何有數據的測站，記錄第一個候選測站的名稱
            if 'station_1' in plant_mapping.columns:
                first_station = plant_mapping.iloc[0]['station_1']
                if not pd.isna(first_station):
                    df_power.at[idx, 'closest_station'] = str(first_station).strip()
    
    # 移除中間欄位
    df_power = df_power.drop(['YearMonth'], axis=1)
    
    # 保存結果
    print(f"\n保存結果到 {output_path}...")
    df_power.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 統計信息
    has_data = df_power['closest_station'].notna().sum()
    print(f"完成！共處理 {len(df_power)} 筆記錄")
    print(f"其中 {has_data} 筆有氣象數據")
    print(f"輸出檔案: {output_path}")
    
    return df_power

if __name__ == '__main__':
    df_result = combine_data('datas/combined_data.csv')
    print("\n前5筆記錄:")
    print(df_result.head())