import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SolarDataPreprocessor:
    """太陽能發電數據預處理模塊
    
    特別注意事項：
    - 降雨量為 NaN 且降雨日數為 0 的情況代表 T 值（微量降水），在填充時應使用 0.0
    - 這類情況通常表示該月份沒有測定的可測量降水量
    """
    
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
        self.missing_rainfall_t_value_count = 0  # 追蹤 T 值（NaN + 降雨日數=0）的數量
        
    def load_data(self):
        """讀取數據"""
        print(f"讀取數據: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ 數據形狀: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """數據探索和統計"""
        print("\n【數據探索】")
        
        # 檢查 T 值情況（降雨量 NaN + 降雨日數 0）
        rainfall_nan = self.df['累積雨量'].isna()
        rainy_days_zero = self.df['降雨日數'] == 0.0
        t_value_records = rainfall_nan & rainy_days_zero
        self.missing_rainfall_t_value_count = t_value_records.sum()
        
        print(f"⚠️  T 值記錄 (降雨量 NaN + 降雨日數 0): {self.missing_rainfall_t_value_count} 筆")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            '欄位': missing.index,
            '缺失數': missing.values,
            '缺失率%': missing_pct.values
        })
        missing_df = missing_df[missing_df['缺失數'] > 0].sort_values('缺失率%', ascending=False)
        
        if len(missing_df) > 0:
            print("缺失值統計：")
            print(missing_df.to_string(index=False))
        
        return missing_df
    
    def clean_data(self, 
                   drop_columns=None,
                   handle_missing='smart',
                   remove_outliers=False,
                   drop_date_columns=True):
        """
        數據清理
        
        Args:
            drop_columns: 要刪除的欄位列表
            handle_missing: 缺失值處理方式 ('smart', 'mean', 'zero', 'drop', 'forward_fill')
            remove_outliers: 是否移除異常值
            drop_date_columns: 是否刪除日期欄位
        """
        print("\n【數據清理】")
        self.df_processed = self.df.copy()
        
        # 先刪除所有日期欄位
        if drop_date_columns:
            date_cols = [col for col in self.df_processed.columns if '日期' in col]
            if date_cols:
                self.df_processed = self.df_processed.drop(columns=date_cols, errors='ignore')
        
        # 1. 刪除不需要的欄位
        if drop_columns:
            self.df_processed = self.df_processed.drop(columns=drop_columns, errors='ignore')
        
        # 2. 處理缺失值
        print(f"✓ 缺失值處理 (方法: {handle_missing})")
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        
        if handle_missing == 'smart':
            # 特別處理 T 值：降雨量 NaN + 降雨日數 0 → 填為 0.0
            t_value_mask = (self.df_processed['累積雨量'].isna() & 
                           (self.df_processed['降雨日數'] == 0.0))
            
            if t_value_mask.sum() > 0:
                print(f"  → T 值填充（降雨量=0.0）: {t_value_mask.sum()} 筆")
                self.df_processed.loc[t_value_mask, '累積雨量'] = 0.0
            
            # 其他缺失值用平均值填充
            for col in numeric_cols:
                remaining_na = self.df_processed[col].isna().sum()
                if remaining_na > 0:
                    self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
                    print(f"  → {col}: 平均值填充 {remaining_na} 筆")
        
        elif handle_missing == 'mean':
            for col in numeric_cols:
                if self.df_processed[col].isnull().sum() > 0:
                    self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
        
        elif handle_missing == 'zero':
            # 用 0 填充所有缺失值
            for col in numeric_cols:
                na_count = self.df_processed[col].isna().sum()
                if na_count > 0:
                    self.df_processed[col].fillna(0, inplace=True)
                    print(f"  → {col}: 零值填充 {na_count} 筆")
        
        elif handle_missing == 'forward_fill':
            self.df_processed[numeric_cols] = self.df_processed[numeric_cols].fillna(method='ffill')
            self.df_processed[numeric_cols] = self.df_processed[numeric_cols].fillna(method='bfill')
        
        elif handle_missing == 'drop':
            self.df_processed = self.df_processed.dropna()
        
        print(f"✓ 處理後缺失值: {self.df_processed.isnull().sum().sum()} 個")
        print(f"✓ 數據形狀: {self.df_processed.shape}")
        
        return self.df_processed
    
    def add_features(self, include_efficiency=False, 
                     include_time_features=False,
                     include_station_encoding=False):
        """特徵工程"""
        if not include_efficiency and not include_time_features and not include_station_encoding:
            return self.df_processed
        
        if include_efficiency:
            self.df_processed['calculated_efficiency'] = (
                self.df_processed['發電量(度)/Power Generation(kWh)'] / 
                self.df_processed['裝置容量(瓩)/Installed Capacity(kW)'] / 30
            )
        
        if include_time_features:
            self.df_processed['season'] = self.df_processed['月份/Month'].apply(self._get_season)
            season_map = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
            self.df_processed['season_encoded'] = self.df_processed['season'].map(season_map)
        
        if include_station_encoding:
            station_map = {station: idx for idx, station in enumerate(
                self.df_processed['closest_station'].unique()
            )}
            self.df_processed['station_encoded'] = self.df_processed['closest_station'].map(station_map)
        
        return self.df_processed
    
    def create_simplified_mode(self):
        """
        創建簡化模式：
        - 移除：年度、月份、日期、平均單位裝置容量每日發電量
        - 合併：發電比 = 發電量 / 裝置容量
        - 需在 clean_data 之後調用
        
        Returns:
            處理後的 DataFrame
        """
        print("\n【簡化模式轉換】")
        
        if self.df_processed is None:
            raise ValueError("請先調用 clean_data() 方法")
        
        # 移除指定欄位
        cols_to_remove = [
            '年度/Year',
            '月份/Month',
            '平均單位裝置容量每日發電量/Average of Each Unit Power Generatioon Per Day'
        ]
        
        existing_cols_to_remove = [col for col in cols_to_remove if col in self.df_processed.columns]
        
        if existing_cols_to_remove:
            self.df_processed = self.df_processed.drop(columns=existing_cols_to_remove)
            print(f"✓ 已移除欄位: {', '.join(existing_cols_to_remove)}")
        
        # 創建發電比特徵
        capacity_col = '裝置容量(瓩)/Installed Capacity(kW)'
        generation_col = '發電量(度)/Power Generation(kWh)'
        
        if capacity_col in self.df_processed.columns and generation_col in self.df_processed.columns:
            # 計算發電比 (避免除以零)
            self.df_processed['發電比_Power_Ratio'] = (
                self.df_processed[generation_col] / 
                self.df_processed[capacity_col].replace(0, np.nan)
            )
            print(f"✓ 已創建發電比特徵: {self.df_processed['發電比_Power_Ratio'].describe()}")
            
            # 移除原始的裝置容量和發電量欄位
            self.df_processed = self.df_processed.drop(
                columns=[capacity_col, generation_col],
                errors='ignore'
            )
            print(f"✓ 已移除原始特徵: {capacity_col}, {generation_col}")
        
        print(f"✓ 簡化模式後數據形狀: {self.df_processed.shape}")
        
        return self.df_processed
    
    def prepare_train_test(self, 
                          target_col='發電量(度)/Power Generation(kWh)',
                          test_ratio=0.2,
                          by_time=True):
        """準備訓練和測試集"""
        if by_time:
            split_idx = int(len(self.df_processed) * (1 - test_ratio))
            train_data = self.df_processed.iloc[:split_idx]
            test_data = self.df_processed.iloc[split_idx:]
        else:
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                self.df_processed, test_size=test_ratio, random_state=42
            )
        
        print(f"✓ 時間分割: 訓練集 {len(train_data)} | 測試集 {len(test_data)}")
        
        return train_data, test_data
    
    def get_feature_columns(self):
        """
        獲取模型的特徵欄位（當前版本：只返回原始欄位）
        
        Returns:
            包含不同類別特徵的字典
        """
        # 原始欄位（不添加新的計算欄位）
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
        
        location_cols = [
            'closest_station'
        ]
        
        time_cols = [
            '年度/Year',
            '月份/Month'
        ]
        
        return {
            'weather': weather_cols,
            'power': power_cols,
            'location': location_cols,
            'time': time_cols,
            'all': weather_cols + power_cols + location_cols + time_cols
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
    """示例使用 - 輕量級路線：先用原始欄位測試"""
    preprocessor = SolarDataPreprocessor()
    
    preprocessor.load_data()
    preprocessor.explore_data()
    
    # 模式 1: 預設模式（使用平均值填充）
    print("\n" + "="*80)
    print("【模式 1：預設模式（平均值填充）】")
    print("="*80)
    preprocessor.clean_data(
        drop_columns=None,
        handle_missing='smart',
        drop_date_columns=True
    )
    
    feature_cols = preprocessor.get_feature_columns()
    train_data_1, test_data_1 = preprocessor.prepare_train_test(by_time=True)
    preprocessor.save_processed_data()
    
    print(f"✓ 預設模式完成: {preprocessor.df_processed.shape} | 欄位: {preprocessor.df_processed.shape[1]}")
    
    # 模式 2: 簡化模式（零值填充 + 發電比特徵）
    print("\n" + "="*80)
    print("【模式 2：簡化模式（零值填充 + 發電比特徵）】")
    print("="*80)
    preprocessor2 = SolarDataPreprocessor()
    preprocessor2.load_data()
    preprocessor2.explore_data()
    preprocessor2.clean_data(
        drop_columns=None,
        handle_missing='zero',  # 用零值填充缺失值
        drop_date_columns=True
    )
    preprocessor2.create_simplified_mode()  # 創建簡化模式
    
    train_data_2, test_data_2 = preprocessor2.prepare_train_test(by_time=True)
    
    print(f"✓ 簡化模式完成: {preprocessor2.df_processed.shape} | 欄位: {preprocessor2.df_processed.shape[1]}")
    print(f"\n【簡化模式新欄位】")
    print(f"欄位列表: {list(preprocessor2.df_processed.columns)}")
    
    return preprocessor, preprocessor2


if __name__ == '__main__':
    preprocessor1, preprocessor2 = main()
