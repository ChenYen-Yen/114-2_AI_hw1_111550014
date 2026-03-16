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
        print("正在初始化瀏覽器驅動...")
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--headless')
        
        driver = webdriver.Chrome(options=options)
        print("✓ 瀏覽器驅動初始化完成")
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
            print(f"正在獲取 {year} 年 {month:02d} 月的資料...")
            
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
            print(f"  ✓ 已選擇年份: {year}")
            time.sleep(0.3)
            
            # 選擇月份
            select_month.select_by_visible_text(f"{month}")
            print(f"  ✓ 已選擇月份: {month:02d}")
            
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
            
            # 移除第一行表頭 (<tr class="th_row_1">)
            thead = table.find('thead')
            if thead:
                first_tr = thead.find('tr', {'class': 'th_row_1'})
                if first_tr:
                    first_tr.decompose()
                    print("已移除第一行表頭 (th_row_1)")
            
            try:
                # 使用 StringIO 将 HTML 字符串转为文件对象，再用 pandas 解析
                html_string = str(table)
                df = pd.read_html(StringIO(html_string))[0]
                
                print(f"抓取成功: {len(df)} 筆資料")
                return df
            except Exception as e:
                raise Exception(f"表格抓取失敗: {e}")
                
        except Exception as e:
            raise Exception(f"爬蟲錯誤 ({year}-{month:02d}): {e}")
    
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
            
            print(f"解析完成: 從 {len(df.columns)} 欄擴展到 {len(df_expanded.columns)} 欄")
            return df_expanded
            
        except Exception as e:
            raise Exception(f"解析失敗: {e}")
        
    def close_driver(self):
        """關閉瀏覽器驅動"""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            print("✓ 已關閉瀏覽器驅動")

def test_fetch_and_extract():
    """
    測試 fetch_monthly_data 和 extract_data 功能
    並將結果輸出至 test_output 目錄
    """
    print("\n" + "="*60)
    print("開始測試 fetch_monthly_data 和 extract_data")
    print("="*60)
    
    # 建立 test_output 資料夾
    test_output_dir = "test_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    crawler = WeatherCrawler(output_dir="datas/C-B0025-2025")
    
    try:
        # 測試單一月份的資料爬取和解析
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
        
        # 儲存資料統計資訊
        stats_file = os.path.join(test_output_dir, "data_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"氣象資料測試統計\n")
            f.write(f"{'='*50}\n")
            f.write(f"年份: {year}\n")
            f.write(f"月份: {month:02d}\n")
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
    finally:
        crawler.close_driver()

def main():    
    """
    主程式：爬取 2025 年全年氣象資料
    """
    # 執行測試
    test_fetch_and_extract()

if __name__ == "__main__":
    main()

