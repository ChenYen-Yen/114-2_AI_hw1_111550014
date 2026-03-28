import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SolarDataPreprocessor:
    """太陽能發電數據預處理模塊"""
    
    def __init__(self, data_path=None):
        """
        初始化預處理器
        
        Args:
            data_path: combined_data.csv 的路徑，如果為 None 則使用預設路徑
        """
        if data_path is None:
            # 預設路徑：假設從 codes/models 目錄執行
            data_path = Path(__file__).parents[3] / 'datas' / 'combined_data.csv'
        
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        
    def load_data(self):
        """讀取數據"""
        print(f"讀取數據: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ 數據形狀: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """數據探索和統計"""
        print("\n【數據探索】")
        print("\n--- 基本信息 ---")
        print(f"時間範圍: {self.df['年度/Year'].min()}-{self.df['月份/Month'].astype(str).str.zfill(2).min()} "
              f"至 {self.df['年度/Year'].max()}-{self.df['月份/Month'].astype(str).str.zfill(2).max()}")
        
        unique_stations = self.df['發電站名稱/Name of The Power Station'].nunique()
        print(f"發電站數量: {unique_stations}")
        print(f"唯一發電站列表:")
        for station in self.df['發電站名稱/Name of The Power Station'].unique():
            count = len(self.df[self.df['發電站名稱/Name of The Power Station'] == station])
            print(f"  - {station} ({count} 筆)")
        
        print("\n--- 缺失值統計 ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            '欄位': missing.index,
            '缺失數': missing.values,
            '缺失率%': missing_pct.values
        })
        missing_df = missing_df[missing_df['缺失數'] > 0].sort_values('缺失率%', ascending=False)
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("✓ 沒有缺失值")
        
        print("\n--- 數值欄位統計 ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe().round(2))
        
        print("\n--- 發電量統計 ---")
        print(f"發電量(度)範圍: {self.df['發電量(度)/Power Generation(kWh)'].min():.0f} ~ "
              f"{self.df['發電量(度)/Power Generation(kWh)'].max():.0f}")
        print(f"裝置容量範圍: {self.df['裝置容量(瓩)/Installed Capacity(kW)'].min():.2f} ~ "
              f"{self.df['裝置容量(瓩)/Installed Capacity(kW)'].max():.2f}")
        
        return missing_df
    
    def clean_data(self, 
                   drop_columns=None,
                   handle_missing='mean',
                   remove_outliers=False):
        """
        數據清理
        
        Args:
            drop_columns: 要刪除的欄位列表
            handle_missing: 缺失值處理方式 ('mean', 'drop', 'forward_fill')
            remove_outliers: 是否移除異常值
        """
        print("\n【數據清理】")
        self.df_processed = self.df.copy()
        
        # 1. 刪除不需要的欄位
        if drop_columns:
            print(f"✓ 刪除欄位: {drop_columns}")
            self.df_processed = self.df_processed.drop(columns=drop_columns, errors='ignore')
        
        # 2. 處理缺失值
        print(f"\n✓ 處理缺失值 (方法: {handle_missing})")
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        
        if handle_missing == 'mean':
            for col in numeric_cols:
                if self.df_processed[col].isnull().sum() > 0:
                    self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
        elif handle_missing == 'forward_fill':
            self.df_processed[numeric_cols] = self.df_processed[numeric_cols].fillna(method='ffill')
            self.df_processed[numeric_cols] = self.df_processed[numeric_cols].fillna(method='bfill')
        elif handle_missing == 'drop':
            self.df_processed = self.df_processed.dropna()
        
        print(f"  缺失值處理後: {self.df_processed.isnull().sum().sum()} 個缺失值")
        
        # 3. 移除異常值（可選）
        if remove_outliers:
            print("\n✓ 移除異常值 (使用 IQR 方法)")
            for col in numeric_cols:
                Q1 = self.df_processed[col].quantile(0.25)
                Q3 = self.df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before = len(self.df_processed)
                self.df_processed = self.df_processed[
                    (self.df_processed[col] >= lower_bound) & 
                    (self.df_processed[col] <= upper_bound)
                ]
                after = len(self.df_processed)
                if before != after:
                    print(f"  {col}: 移除 {before - after} 筆")
        
        print(f"\n✓ 清理後數據形狀: {self.df_processed.shape}")
        return self.df_processed
    
    def add_features(self, include_efficiency=True, 
                     include_time_features=True,
                     include_station_encoding=True):
        """
        特徵工程
        
        Args:
            include_efficiency: 是否添加發電效率特徵
            include_time_features: 是否添加時間特徵
            include_station_encoding: 是否對測站進行編碼
        """
        print("\n【特徵工程】")
        
        # 1. 發電效率特徵
        if include_efficiency:
            print("✓ 添加發電效率特徵")
            # 重新計算發電效率（如果需要驗證原始計算）
            self.df_processed['calculated_efficiency'] = (
                self.df_processed['發電量(度)/Power Generation(kWh)'] / 
                self.df_processed['裝置容量(瓩)/Installed Capacity(kW)'] / 30  # 平均日效率
            )
            # 與原始的對比
            self.df_processed['efficiency_original'] = self.df_processed['平均單位裝置容量每日發電量/Average of Each Unit Power Generatioon Per Day']
        
        # 2. 時間特徵
        if include_time_features:
            print("✓ 添加時間特徵")
            self.df_processed['year_month'] = (
                self.df_processed['年度/Year'].astype(str) + '-' + 
                self.df_processed['月份/Month'].astype(str).str.zfill(2)
            )
            # 季節特徵
            self.df_processed['season'] = self.df_processed['月份/Month'].apply(self._get_season)
            # 季節編碼
            season_map = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
            self.df_processed['season_encoded'] = self.df_processed['season'].map(season_map)
        
        # 3. 測站編碼
        if include_station_encoding:
            print("✓ 添加測站編碼")
            # 保留測站名稱
            self.df_processed['station_chi_name'] = self.df_processed['closest_station']
            # 將測站名稱轉換為數值編碼
            station_map = {station: idx for idx, station in enumerate(
                self.df_processed['closest_station'].unique()
            )}
            self.df_processed['station_encoded'] = self.df_processed['closest_station'].map(station_map)
            
            print(f"  測站映射表: {station_map}")
        
        return self.df_processed
    
    def prepare_train_test(self, 
                          target_col='發電量(度)/Power Generation(kWh)',
                          test_ratio=0.2,
                          by_time=True):
        """
        準備訓練和測試集
        
        Args:
            target_col: 目標欄位
            test_ratio: 測試集比例
            by_time: 是否按時間分割（建議 True，因為數據有時間序列特性）
        """
        print("\n【準備訓練集和測試集】")
        
        # 檢查特徵和標籤
        print(f"目標變量: {target_col}")
        print(f"✓ 使用按時間分割的方式: {by_time}")
        
        if by_time:
            # 按時間順序分割（避免數據洩漏）
            split_idx = int(len(self.df_processed) * (1 - test_ratio))
            train_data = self.df_processed.iloc[:split_idx]
            test_data = self.df_processed.iloc[split_idx:]
            print(f"✓ 訓練集: {train_data.shape[0]} 筆 ({train_data['年度/Year'].min()}-{train_data['月份/Month'].min():.0f} 至 {train_data['年度/Year'].max()}-{train_data['月份/Month'].max():.0f})")
            print(f"✓ 測試集: {test_data.shape[0]} 筆 ({test_data['年度/Year'].min()}-{test_data['月份/Month'].min():.0f} 至 {test_data['年度/Year'].max()}-{test_data['月份/Month'].max():.0f})")
        else:
            # 隨機分割
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                self.df_processed, test_size=test_ratio, random_state=42
            )
            print(f"✓ 訓練集: {train_data.shape[0]} 筆")
            print(f"✓ 測試集: {test_data.shape[0]} 筆")
        
        return train_data, test_data
    
    def get_feature_columns(self):
        """
        獲取模型的特徵欄位
        
        Returns:
            包含不同類別特徵的字典
        """
        weather_cols = [
            '溫度_平均', '溫度_最高', '溫度_最低',
            '累積雨量', '降雨日數',
            '最大十分鐘風', '最大瞬間風',
            '相對濕度_平均', '相對濕度_最低',
            '氣壓_平均', '日照時數'
        ]
        
        power_cols = [
            '裝置容量(瓩)/Installed Capacity(kW)',
            '平均單位裝置容量每日發電量/Average of Each Unit Power Generatioon Per Day'
        ]
        
        if 'calculated_efficiency' in self.df_processed.columns:
            power_cols.append('calculated_efficiency')
            power_cols.append('efficiency_original')
        
        time_cols = []
        if 'season_encoded' in self.df_processed.columns:
            time_cols.append('season_encoded')
        
        location_cols = []
        if 'station_encoded' in self.df_processed.columns:
            location_cols.append('station_encoded')
        
        return {
            'weather': weather_cols,
            'power': power_cols,
            'time': time_cols,
            'location': location_cols,
            'all': weather_cols + power_cols + time_cols + location_cols
        }
    
    @staticmethod
    def _get_season(month):
        """根據月份返回季節"""
        if month in [12, 1, 2]:
            return '冬'
        elif month in [3, 4, 5]:
            return '春'
        elif month in [6, 7, 8]:
            return '夏'
        else:
            return '秋'
    
    def save_processed_data(self, output_path=None):
        """保存已處理的數據"""
        if output_path is None:
            output_path = Path(self.data_path).parent / 'processed_data.csv'
        
        self.df_processed.to_csv(output_path, index=False)
        print(f"\n✓ 已保存處理後的數據: {output_path}")
        return output_path


def main():
    """示例使用"""
    # 初始化預處理器
    preprocessor = SolarDataPreprocessor()
    
    # 1. 讀取數據
    df = preprocessor.load_data()
    
    # 2. 數據探索
    preprocessor.explore_data()
    
    # 3. 數據清理
    # 建議刪除的欄位：
    # - 溫度_最高日期, 溫度_最低日期, 最大十分鐘風速日期, 最大瞬間風日期, 最大瞬間風向, 最大十分鐘風向
    #   這些日期欄位對模型預測幫助有限
    drop_cols = [
        '溫度_最高日期', '溫度_最低日期', 
        '最大十分鐘風速日期', '最大瞬間風日期',
        '最大十分鐘風向', '最大瞬間風向'
    ]
    
    preprocessor.clean_data(
        drop_columns=drop_cols,
        handle_missing='mean',
        remove_outliers=False
    )
    
    # 4. 特徵工程
    preprocessor.add_features(
        include_efficiency=True,
        include_time_features=True,
        include_station_encoding=True
    )
    
    # 5. 獲取特徵列表
    feature_cols = preprocessor.get_feature_columns()
    print("\n【特徵列表】")
    for category, cols in feature_cols.items():
        print(f"{category}: {cols}")
    
    # 6. 準備訓練集和測試集
    train_data, test_data = preprocessor.prepare_train_test(by_time=True)
    
    # 7. 保存處理後的數據
    preprocessor.save_processed_data()
    
    print("\n✓ 預處理完成！")
    return preprocessor


if __name__ == '__main__':
    preprocessor = main()
