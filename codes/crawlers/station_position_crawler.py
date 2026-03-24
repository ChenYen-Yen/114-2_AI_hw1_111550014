import requests
import json
import xml.etree.ElementTree as ET
import pandas as pd
import os
from bs4 import BeautifulSoup

def fetch_station_data_from_website():
    """
    從 https://hdps.cwa.gov.tw/static/state.html 抓取測站資料
    解析 class="download_html_table black_table table-condensed" 的 HTML 表格
    網站欄位：站號、站名、站種、海拔高度、經度、緯度、城市、地址、資料起始日期、撤站日期、備註、原站號、新站號
    """
    try:
        # 抓取網站資料
        url = "https://hdps.cwa.gov.tw/static/state.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        html_content = response.text
        
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 尋找 class="download_html_table black_table table-condensed" 的表格
        table = soup.find('table', class_=['download_html_table', 'black_table', 'table-condensed'])
        
        if not table:
            print("✗ 找不到目標表格")
            return None
        
        # 提取表格頭部，確定欄位對應
        headers_row = table.find('thead')
        if not headers_row:
            headers_row = table.find('tr')
        
        if not headers_row:
            print("✗ 找不到表格頭部")
            return None
        
        # 提取所有欄位名
        header_cells = headers_row.find_all(['th', 'td'])
        column_names = [cell.get_text(strip=True) for cell in header_cells]
        
        print(f"✓ 找到表格欄位: {', '.join(column_names[:5])}...")
        
        # 提取表格資料
        data = []
        tbody = table.find('tbody')
        
        if tbody:
            rows = tbody.find_all('tr')
        else:
            # 如果沒有 tbody，直接從 table 找 tr（跳過頭部行）
            all_rows = table.find_all('tr')
            rows = all_rows[1:] if len(all_rows) > 1 else all_rows
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 0:
                continue
            
            row_data = {}
            for idx, cell in enumerate(cells):
                if idx < len(column_names):
                    cell_text = cell.get_text(strip=True)
                    row_data[column_names[idx]] = cell_text
            
            if row_data:
                data.append(row_data)
        
        print(f"✓ 成功解析 HTML 表格，找到 {len(data)} 筆資料")
        if len(data) > 0:
            print(f"  第一筆資料: {list(data[0].values())[:3]}...")
        
        return data
        
    except Exception as e:
        print(f"抓取網站資料時出錯: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_xml_stations(xml_file_path):
    """
    解析 XML 檔案，提取測站列表
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 定義命名空間
        ns = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
        
        stations = []
        
        # 尋找所有 location 元素
        for location in root.findall('.//cwa:location', ns):
            station_elem = location.find('cwa:station', ns)
            if station_elem is not None:
                station_id = station_elem.findtext('cwa:StationID', '', ns)
                station_name = station_elem.findtext('cwa:StationName', '', ns)
                
                if station_id:
                    stations.append(station_id)
        
        return list(set(stations))  # 去重
        
    except Exception as e:
        print(f"解析 XML 時出錯: {e}")
        return []

def filter_and_save_results(xml_file_path, output_file_path=None):
    """
    主函數：抓取網站資料，篩選出 XML 中有的測站與其經緯度
    """
    print("="*60)
    print("開始執行測站位置爬蟲")
    print("="*60)
    
    # 步驟 1: 從 XML 解析測站列表
    print("\n步驟 1: 解析 XML 檔案...")
    xml_stations = parse_xml_stations(xml_file_path)
    print(f"✓ 從 XML 找到 {len(xml_stations)} 個測站")
    print(f"  測站列表: {', '.join(xml_stations[:5])}..." if len(xml_stations) > 5 else f"  測站列表: {', '.join(xml_stations)}")
    
    # 步驟 2: 從網站抓取測站資料
    print("\n步驟 2: 從網站抓取測站資料...")
    website_data = fetch_station_data_from_website()
    
    if website_data is None:
        print("✗ 無法從網站抓取資料")
        print("\n使用備用方法：嘗試通過 API 或其他方式获取资料...")
        
        # 備用方案：直接構造常見測站列表
        # 這是基於中央氣象署的標準測站列表
        # station_coords = get_default_station_coordinates()
        return None
    else:
        # 將網站資料轉換為字典格式 (測站名 -> 經緯度)
        station_coords = {}
        for item in website_data:
            if isinstance(item, dict):
                # 從表格資料中尋找對應的欄位
                # 表格欄位: 站號、站名、站種、海拔高度、經度、緯度、城市、地址、資料起始日期、撤站日期、備註、原站號、新站號
                
                # 尋找站名
                station_name = (item.get('站名') or item.get('stationName') or 
                               item.get('站店名') or item.get('name') or 
                               item.get('id') or item.get('站號'))
                
                # 尋找緯度和經度
                lat = (item.get('緯度') or item.get('lat') or 
                       item.get('latitude') or item.get('Latitude') or
                       item.get('Lat'))
                lon = (item.get('經度') or item.get('lon') or 
                       item.get('longitude') or item.get('Longitude') or
                       item.get('Lon'))
                
                if station_name and lat is not None and lon is not None:
                    try:
                        # 確保經緯度是浮點數
                        lat_float = float(lat) if isinstance(lat, str) else lat
                        lon_float = float(lon) if isinstance(lon, str) else lon

                        station_coords[str(station_name).strip()] = {
                            'lat': lat_float,
                            'lon': lon_float
                        }
                    except (ValueError, TypeError):
                        continue
        
        print(f"✓ 從網站找到 {len(station_coords)} 個測站的資料")
    
    # 步驟 3: 篩選結果
    print("\n步驟 3: 篩選資料...")
    results = []
    found_count = 0
    not_found = []
    
    for station in xml_stations:
        # 嘗試精確匹配和部分匹配
        matched = False
        
        # 首先嘗試精確匹配
        if station in station_coords:
            coord = station_coords[station]
            results.append({
                '站名': station,
                '緯度': coord['lat'],
                '經度': coord['lon']
            })
            found_count += 1
            matched = True
        else:
            # 如果精確匹配失敗，嘗試模糊匹配
            for station_name, coord in station_coords.items():
                if station.lower() == station_name.lower() or station in station_name or station_name in station:
                    results.append({
                        '站名': station,
                        '緯度': coord['lat'],
                        '經度': coord['lon']
                    })
                    found_count += 1
                    matched = True
                    break
        
        if not matched:
            not_found.append(station)
    
    print(f"✓ 篩選完成")
    print(f"  找到座標的測站: {found_count} 個")
    print(f"  未找到座標的測站: {len(not_found)} 個")
    
    # 步驟 4: 保存結果
    if results:
        df = pd.DataFrame(results)
        
        if output_file_path is None:
            # 默認輸出路徑
            base_dir = os.path.dirname(xml_file_path)
            output_file_path = os.path.join(base_dir, '../', 'station_positions.csv')
        
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        print(f"\n步驟 4: 保存結果")
        print(f"✓ 結果已保存至: {output_file_path}")
        print(f"\n篩選結果預覽:")
        print(df.to_string(index=False))
        
        return df
    else:
        print("✗ 沒有找到匹配的測站資料")
        return None

if __name__ == "__main__":
    # 執行爬蟲
    xml_file = os.path.join(os.path.dirname(__file__), '../../datas/C-B0025-2025/mn_Report_202512.xml')
    
    if os.path.exists(xml_file):
        df = filter_and_save_results(xml_file)
    else:
        print(f"✗ 找不到 XML 檔案: {xml_file}")
