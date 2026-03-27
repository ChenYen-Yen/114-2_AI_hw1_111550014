"""
中央氣象署月份氣象資料爬蟲
爬取網址: https://www.cwa.gov.tw/V8/C/C/Statistics/monthlydata.html
目標: 下載 2025/01~2025/12 的氣象資料，生成 XML 和 CSV 檔案
"""

import os
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import xml.etree.ElementTree as ET
import time
import uuid
import calendar
class WeatherCrawler:
    """氣象資料爬蟲類"""
    
    def __init__(self, output_dir="datas/C-B0025-2025"):
        """
        初始化爬蟲
        
        Args:
            output_dir: 輸出資料夾路徑
        """
        self.output_dir = output_dir
        self.base_url = "https://www.cwa.gov.tw/V8/C/C/Statistics/monthlydata.html"
        self.dataset_id = "S0001"
        self.dataset_name = "過去氣象觀測"
        self.data_item_id = "S0001-01000"
        self.driver = None
        
        # 建立輸出資料夾
        os.makedirs(output_dir, exist_ok=True)
        print("成功創建")
        
    def setup_driver(self):
        """設置 Selenium 瀏覽器驅動"""
        print("正在初始化driver...")
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--headless')
        
        driver = webdriver.Chrome(options=options)
        return driver
    
    def fetch_monthly_data(self, year, month):
        """
        爬取特定月份的氣象資料頁面，轉換成 DataFrame
        
        Args:
            year: 年份 (如 2025)
            month: 月份 (1-12)
            
        Returns:
            pd.DataFrame: 原始氣象資料表（未拆分）
        """
        # 第一次爬取時初始化 driver
        if self.driver is None:
            print("driver not set yet, initializing...")
            self.driver = self.setup_driver()
            self.driver.get(self.base_url)
            print("driver 初始化完成")
        
        try:
            print(f"正在獲取 {year} 年 {month} 月的資料...")
            
            # 等待頁面載入
            wait = WebDriverWait(self.driver, 10)
            
            # 等待年份 select 元素出現
            year_select = wait.until(
                EC.presence_of_element_located((By.ID, "Year"))
            )
            month_select = wait.until(
                EC.presence_of_element_located((By.ID, "Month"))
            )
            
            # 使用 Select 類來操作下拉菜單
            select_year = Select(year_select)
            select_month = Select(month_select)
            
            # 選擇年份
            # 因option沒給value因此使用visible text而非value
            select_year.select_by_visible_text(str(year))
            time.sleep(0.3)
            # 選擇月份
            select_month.select_by_visible_text(f"{month}")
            
            # 等待數據表格被更新
            wait.until(
                EC.presence_of_element_located((By.ID, "MonthlyData_MOD"))
            )
            time.sleep(0.3)
            
            # 獲取頁面 HTML
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            
            # 找到資料所在table
            data_tbody_element = soup.find('tbody', {'id': 'MonthlyData_MOD'})
            if not data_tbody_element:
                raise ValueError("未找到 id='MonthlyData_MOD' 的 tbody")
            table = data_tbody_element.find_parent('table')
            
            # 移除<tr class="th_row_1">
            thead = table.find('thead')
            if thead:
                first_tr = thead.find('tr', {'class': 'th_row_1'})
                if first_tr:
                    first_tr.decompose()
            
            try:
                # 使用 StringIO 将 HTML 字符串转为文件对象，再用 pandas 解析
                html_string = str(table)
                df = pd.read_html(StringIO(html_string))[0]
                
                print(f"抓取成功: {len(df)} 筆資料")
                return df
            except Exception as e:
                raise Exception(f"表格抓取失敗: {e}")
                
        except Exception as e:
            raise Exception(f"爬蟲錯誤 ({year}-{month}): {e}")
    
    def extract_data(self, df, month):
        """
        解析和拆分氣象資料，將複合欄位拆開
        
        Args:
            df: 原始氣象資料 DataFrame
            
        Returns:
            pd.DataFrame: 拆分後的氣象資料表
        """
        if df is None or df.empty:
            return None
        
        try:
            # 複製 DataFrame 避免修改原始數據
            df_expanded = df.copy()
            
            # 定義需要拆分的欄位及其拆分方式
            split_rules = {
                '最高/日期': ['最高溫', '最高溫日期'],
                '最低/日期': ['最低溫', '最低溫日期'],
                '最大十分鐘風': ['最大十分鐘風速', '最大十分鐘風向', '最大十分鐘風日期'],
                '最大瞬間風': ['最大瞬間風速', '最大瞬間風向', '最大瞬間風日期'],
                '最小/日期': ['最小相對溼度', '最小相對溼度日期'],
            }
            
            # 遍歷每一欄進行拆分
            for original_col, new_cols in split_rules.items():
                if original_col in df_expanded.columns:
                    # 根據 "/" 拆分欄位
                    split_data = df_expanded[original_col].str.split('/', expand=True)
                    
                    # 確保新欄位數量一致
                    if len(new_cols) <= split_data.shape[1]:
                        for i, new_col in enumerate(new_cols):
                            df_expanded[new_col] = split_data[i] if i < split_data.shape[1] else None
                    else:
                        raise ValueError(f"欄位 '{original_col}' 的拆分數量不符")
                    
                    # 移除原始欄位
                    df_expanded = df_expanded.drop(columns=[original_col])
            
            # print(f"解析完成: 從 {len(df.columns)} 欄擴展到 {len(df_expanded.columns)} 欄")
            return df_expanded
            
        except Exception as e:
            raise Exception(f"解析失敗: {e}")
        
    def close_driver(self):
        """關閉瀏覽器驅動"""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            print("✓ 已關閉瀏覽器驅動")
    
    def create_xml(self, year, month, data_df):
        """
        根據 CSV 數據生成 XML，使用 Map 的方式填充統計欄位
        
        Args:
            year: 年份
            month: 月份
            data_df: 氣象資料 DataFrame
        """
        if data_df is None or data_df.empty:
            print(f"警告: {year}-{month} 沒有資料，跳過 XML 生成")
            return
        
        try:
            # 定義 CSV 欄位到 XML 欄位的映射表
            field_mapping = {
                # 溫度相關
                '平均': ('AirTemperature', 'Mean'),
                '最高溫': ('AirTemperature', 'Maximum'),
                '最高溫日期': ('AirTemperature', 'MaximumDate'),
                '最低溫': ('AirTemperature', 'Minimum'),
                '最低溫日期': ('AirTemperature', 'MinimumDate'),
                # 降水相關
                '(毫米)': ('Precipitation', 'Accumulation'),
                '(天)': ('Precipitation', 'GE01Days'),
                # 風速相關 (10分鐘風)
                '最大十分鐘風速': ('WindSpeed', 'Maximum'),
                '最大十分鐘風日期': ('WindSpeed', 'MaximumDate'),
                '最大十分鐘風向': ('WindDirection', 'Maximum'),
                # 瞬間風相關
                '最大瞬間風速': ('PeakGustSpeed', 'Maximum'),
                '最大瞬間風日期': ('PeakGustSpeed', 'MaximumDate'),
                '最大瞬間風向': ('PeakGustDirection', 'Maximum'),
                # 相對濕度
                '平均.1': ('RelativeHumidity', 'Mean'),
                '最小相對溼度': ('RelativeHumidity', 'Minimum'),
                '最小相對溼度日期': ('RelativeHumidity', 'MinimumDate'),
                # 氣壓
                '(百帕)': ('AirPressure', 'Mean'),
                # 日照
                '(小時)': ('SunshineDuration', 'Total'),
            }
            
            # 建立測站數據對象列表
            locations = []
            for idx, row in data_df.iterrows():
                station_name = row.get('測站', f'Station_{idx}')
                station_data = {'station_name': station_name}
                
                # 根據映射表填充數據
                for csv_col, (xml_element, xml_method) in field_mapping.items():
                    if csv_col in row.index:
                        value = row[csv_col]
                        # 格式化日期（如果是日期欄位）
                        if 'Date' in xml_method or 'Date' in str(csv_col):
                            if pd.notna(value) and str(value).isdigit():
                                value = f"{year}-{month:02d}-{int(float(value)):02d}"
                        
                        key = f"{xml_element}_{xml_method}"
                        station_data[key] = value if pd.notna(value) else None
                
                locations.append(station_data)
            
            # 生成 XML
            self._generate_xml_file(year, month, locations)
            print(f"  XML 檔案已生成: {os.path.join(self.output_dir, f'mn_Report_{year}{month:02d}.xml')}")
            
        except Exception as e:
            print(f"生成 XML 檔案失敗 ({year}-{month:02d}): {e}")
    
    def _generate_xml_file(self, year, month, locations):
        """
        內部方法：根據位置數據生成 XML 檔案
        
        Args:
            year: 年份
            month: 月份
            locations: 測站數據列表
        """
        # 建立根元素
        root = ET.Element('cwaopendata')
        root.set('xmlns', 'urn:cwa:gov:tw:cwacommon:0.1')
        
        # 建立 resources 元素
        resources = ET.SubElement(root, 'resources')
        resource = ET.SubElement(resources, 'resource')
        
        # 建立 metadata 元素
        metadata = ET.SubElement(resource, 'metadata')
        
        # 添加基本信息
        resource_id = ET.SubElement(metadata, 'resourceID')
        resource_id.text = 'S0001-01000-003'
        
        resource_name = ET.SubElement(metadata, 'resourceName')
        resource_name.text = '中央氣象署氣候觀測_逐月_多要素氣象資料'
        
        resource_desc = ET.SubElement(metadata, 'resourceDescription')
        resource_desc.text = '氣象署所屬氣象觀測站，逐月的各項氣象因子資料。'
        
        language = ET.SubElement(metadata, 'language')
        language.text = 'zh'
        
        # 添加時間資訊
        temporal = ET.SubElement(metadata, 'temporal')
        start_time = ET.SubElement(temporal, 'startTime')
        start_time.text = f"{year}-{month:02d}-01T00:00:00+08:00"
        end_time = ET.SubElement(temporal, 'endTime')
        # 計算該月的最後一天
        last_day = calendar.monthrange(year, month)[1]
        end_time.text = f"{year}-{month:02d}-{last_day}T00:00:00+08:00"
        
        # 添加統計信息
        statistics = ET.SubElement(metadata, 'statistics')
        self._add_statistics_info(statistics)
        
        # 建立 data 元素
        data = ET.SubElement(resource, 'data')
        surface_obs = ET.SubElement(data, 'surfaceObs')
        
        # 添加每個測站的數據
        for location_data in locations:
            location = ET.SubElement(surface_obs, 'location')
            station = ET.SubElement(location, 'station')
            
            station_id = ET.SubElement(station, 'StationID')
            station_id.text = str(location_data.get('station_name', 'Unknown'))
            
            station_name = ET.SubElement(station, 'StationName')
            station_name.text = location_data.get('station_name', 'Unknown')
            
            station_obs = ET.SubElement(location, 'stationObsStatistics')
            
            year_month = ET.SubElement(station_obs, 'YearMonth')
            year_month.text = f"{year}-{month:02d}"
            
            # 添加各氣象元素
            self._add_weather_elements(station_obs, location_data)
        
        # 保存 XML 檔案
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"mn_Report_{year}{month:02d}.xml")
        
        # 添加縮進到 XML 元素
        self._indent_xml(root)
        
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
    
    def _add_statistics_info(self, statistics_elem):
        """添加統計信息到 XML"""
        # 添加統計周期
        stat_periods = ET.SubElement(statistics_elem, 'statisticalPeriods')
        stat_period = ET.SubElement(stat_periods, 'statisticalPeriod')
        period_tag = ET.SubElement(stat_period, 'periodTagName')
        period_tag.text = 'monthly'
        description = ET.SubElement(stat_period, 'description')
        description.text = '逐月'
        
        # 添加氣象要素定義
        weather_elems = ET.SubElement(statistics_elem, 'weatherElements')
        
        # 溫度
        self._add_weather_element_def(weather_elems, 'AirTemperature', '溫度', '˚C',
                                      [('Mean', '平均溫度'), ('Maximum', '最高溫度'), 
                                       ('MaximumDate', '最高溫度發生日期'),
                                       ('Minimum', '最低溫度'), ('MinimumDate', '最低溫度發生日期')])
        
        # 降水
        self._add_weather_element_def(weather_elems, 'Precipitation', '降水量', 'mm',
                                      [('Accumulation', '累積降水量'), 
                                       ('GE01Days', '降雨日數, 降雨量>=0.1mm日數')])
        
        # 風速 (10分鐘)
        self._add_weather_element_def(weather_elems, 'WindSpeed', '10分鐘風風速', 'm s^-1',
                                      [('Maximum', '最大10分鐘風風速'), 
                                       ('MaximumDate', '最大10分鐘風發生日期')])
        
        # 風向 (10分鐘)
        self._add_weather_element_def(weather_elems, 'WindDirection', '10分鐘風風向', '˚',
                                      [('Maximum', '最大10分鐘風風向')])
        
        # 瞬間風速
        self._add_weather_element_def(weather_elems, 'PeakGustSpeed', '瞬間風風速', 'm s^-1',
                                      [('Maximum', '最大瞬間風風速'), 
                                       ('MaximumDate', '最大瞬間風發生日期')])
        
        # 瞬間風向
        self._add_weather_element_def(weather_elems, 'PeakGustDirection', '瞬間風風向', '˚',
                                      [('Maximum', '最大瞬間風風向')])
        
        # 相對濕度
        self._add_weather_element_def(weather_elems, 'RelativeHumidity', '相對溼度', '%',
                                      [('Mean', '平均相對濕度'), 
                                       ('Minimum', '最小相對濕度'),
                                       ('MinimumDate', '最小相對濕度發生日期')])
        
        # 氣壓
        self._add_weather_element_def(weather_elems, 'AirPressure', '測站氣壓', 'hPa',
                                      [('Mean', '平均測站氣壓')])
        
        # 日照時數
        self._add_weather_element_def(weather_elems, 'SunshineDuration', '日照時數', 'hr',
                                      [('Total', '總日照時數')])
        
        # 特殊值
        special_values = ET.SubElement(statistics_elem, 'specialValues')
        special_value = ET.SubElement(special_values, 'specialValue')
        value_elem = ET.SubElement(special_value, 'value')
        value_elem.text = ''
        value_desc = ET.SubElement(special_value, 'description')
        value_desc.text = '無觀測'
    
    def _add_weather_element_def(self, parent, tag_name, description, units, methods):
        """添加單個氣象要素定義"""
        elem = ET.SubElement(parent, 'weatherElement')
        tag = ET.SubElement(elem, 'tagName')
        tag.text = tag_name
        desc = ET.SubElement(elem, 'description')
        desc.text = description
        unit = ET.SubElement(elem, 'units')
        unit.text = units
        
        stat_methods = ET.SubElement(elem, 'statisticalMethods')
        for method_tag, method_desc in methods:
            method = ET.SubElement(stat_methods, 'statisticalMethod')
            method_tag_elem = ET.SubElement(method, 'methodTagName')
            method_tag_elem.text = method_tag
            method_desc_elem = ET.SubElement(method, 'description')
            method_desc_elem.text = method_desc
    
    def _add_weather_elements(self, parent, location_data):
        """添加實際的氣象數據"""
        # 溫度
        air_temp = ET.SubElement(parent, 'AirTemperature')
        monthly = ET.SubElement(air_temp, 'monthly')
        self._add_value_if_exists(monthly, 'Mean', location_data, 'AirTemperature_Mean')
        self._add_value_if_exists(monthly, 'Maximum', location_data, 'AirTemperature_Maximum')
        self._add_value_if_exists(monthly, 'MaximumDate', location_data, 'AirTemperature_MaximumDate')
        self._add_value_if_exists(monthly, 'Minimum', location_data, 'AirTemperature_Minimum')
        self._add_value_if_exists(monthly, 'MinimumDate', location_data, 'AirTemperature_MinimumDate')
        
        # 降水
        precip = ET.SubElement(parent, 'Precipitation')
        monthly = ET.SubElement(precip, 'monthly')
        self._add_value_if_exists(monthly, 'Accumulation', location_data, 'Precipitation_Accumulation')
        self._add_value_if_exists(monthly, 'GE01Days', location_data, 'Precipitation_GE01Days')
        
        # 風速 (10分鐘)
        wind_speed = ET.SubElement(parent, 'WindSpeed')
        monthly = ET.SubElement(wind_speed, 'monthly')
        self._add_value_if_exists(monthly, 'Maximum', location_data, 'WindSpeed_Maximum')
        self._add_value_if_exists(monthly, 'MaximumDate', location_data, 'WindSpeed_MaximumDate')
        
        # 風向
        wind_dir = ET.SubElement(parent, 'WindDirection')
        monthly = ET.SubElement(wind_dir, 'monthly')
        self._add_value_if_exists(monthly, 'Maximum', location_data, 'WindDirection_Maximum')
        
        # 瞬間風速
        gust_speed = ET.SubElement(parent, 'PeakGustSpeed')
        monthly = ET.SubElement(gust_speed, 'monthly')
        self._add_value_if_exists(monthly, 'Maximum', location_data, 'PeakGustSpeed_Maximum')
        self._add_value_if_exists(monthly, 'MaximumDate', location_data, 'PeakGustSpeed_MaximumDate')
        
        # 瞬間風向
        gust_dir = ET.SubElement(parent, 'PeakGustDirection')
        monthly = ET.SubElement(gust_dir, 'monthly')
        self._add_value_if_exists(monthly, 'Maximum', location_data, 'PeakGustDirection_Maximum')
        
        # 相對濕度
        rel_humid = ET.SubElement(parent, 'RelativeHumidity')
        monthly = ET.SubElement(rel_humid, 'monthly')
        self._add_value_if_exists(monthly, 'Mean', location_data, 'RelativeHumidity_Mean')
        self._add_value_if_exists(monthly, 'Minimum', location_data, 'RelativeHumidity_Minimum')
        self._add_value_if_exists(monthly, 'MinimumDate', location_data, 'RelativeHumidity_MinimumDate')
        
        # 氣壓
        air_pres = ET.SubElement(parent, 'AirPressure')
        monthly = ET.SubElement(air_pres, 'monthly')
        self._add_value_if_exists(monthly, 'Mean', location_data, 'AirPressure_Mean')
        
        # 日照時數
        sunshine = ET.SubElement(parent, 'SunshineDuration')
        monthly = ET.SubElement(sunshine, 'monthly')
        self._add_value_if_exists(monthly, 'Total', location_data, 'SunshineDuration_Total')
    
    def _add_value_if_exists(self, parent, tag_name, location_data, data_key):
        """如果值存在則添加到 XML"""
        if data_key in location_data and location_data[data_key] is not None:
            elem = ET.SubElement(parent, tag_name)
            elem.text = str(location_data[data_key])
    
    def _indent_xml(self, elem, level=0):
        """遞歸為 XML 添加縮進和換行"""
        indent = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

def test_fetch_and_extract():
    """
    測試 fetch_monthly_data、extract_data 和 create_xml 功能
    並將結果輸出至 test_output 目錄
    """
    print("\n" + "="*60)
    print("開始測試 fetch_monthly_data、extract_data 和 XML 生成")
    print("="*60)
    
    # 建立 test_output 資料夾
    test_output_dir = "test_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    crawler = WeatherCrawler(output_dir="datas/C-B0025-2025")
    
    try:
        year = 2025
        month = 1
        
        print(f"\n[步驟 1] 爬取 {year} 年 {month} 月的資料...")
        raw_data_df = crawler.fetch_monthly_data(year, month)
        
        print(f"\n[步驟 2] 顯示原始資料 (前 3 行):")
        print(raw_data_df.head(3))
        
        # 儲存原始資料到 CSV
        raw_data_file = os.path.join(test_output_dir, "raw_data.csv")
        raw_data_df.to_csv(raw_data_file, index=False, encoding='utf-8')
        print(f"\n✓ 原始資料已儲存至: {raw_data_file}")
        
        print(f"\n[步驟 3] 解析資料...")
        extracted_data_df = crawler.extract_data(raw_data_df, month=month)
        
        print(f"\n[步驟 4] 顯示拆分後的資料 (前 3 行):")
        print(extracted_data_df.head(3))
        
        # 儲存拆分後的資料到 CSV
        extracted_data_file = os.path.join(test_output_dir, "extracted_data.csv")
        extracted_data_df.to_csv(extracted_data_file, index=False, encoding='utf-8')
        print(f"\n✓ 拆分後的資料已儲存至: {extracted_data_file}")
        
        print(f"\n[步驟 5] 生成 XML 檔案...")
        crawler.create_xml(year, month, extracted_data_df)
        print(f"✓ XML 檔案已生成至: datas/C-B0025-2025/")
        
        # 儲存資料統計資訊
        stats_file = os.path.join(test_output_dir, "data_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"氣象資料測試統計\n")
            f.write(f"{'='*50}\n")
            f.write(f"年份: {year}\n")
            f.write(f"月份: {month}\n")
            f.write(f"原始資料:\n")
            f.write(f"  - 行數: {len(raw_data_df)}\n")
            f.write(f"  - 欄數: {len(raw_data_df.columns)}\n")
            f.write(f"  - 欄位名稱: {list(raw_data_df.columns)}\n")
            f.write(f"\n拆分後的資料:\n")
            f.write(f"  - 行數: {len(extracted_data_df)}\n")
            f.write(f"  - 欄數: {len(extracted_data_df.columns)}\n")
            f.write(f"  - 欄位名稱: {list(extracted_data_df.columns)}\n")
        print(f"✓ 統計資訊已儲存至: {stats_file}")
        
        print("\n" + "="*60)
        print("✓ 測試完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        crawler.close_driver()

def fetch_year_data(year):
    crawler = WeatherCrawler(output_dir="datas/C-B0025-2025")
    for month in range(1, 13):
        try:
            print(f"抓取 {year} 年 {month} 月的資料 : ")
            df = crawler.fetch_monthly_data(year, month)
            extracted_df = crawler.extract_data(df, month)
            crawler.create_xml(year, month, extracted_df)
            print(f"{year} 年 {month} 月 抓取完成")
        except Exception as e:
            print(f"抓取 {year}-{month} 失敗: {e}")
    crawler.close_driver()

def main():    
    """
    主程式：爬取 2025 年全年氣象資料
    """
    # 執行測試
    # test_fetch_and_extract()
    
    fetch_year_data(2025)

if __name__ == "__main__":
    main()

