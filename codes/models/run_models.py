import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# 設定繪圖風格和中文字體
sns.set_style("whitegrid")
# 修復中文字顯示問題
try:
    # macOS 上的中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
except:
    pass
plt.rcParams['axes.unicode_minus'] = False  # 修復負號顯示
plt.rcParams['figure.figsize'] = (12, 6)

# 相對匯入
from modules import (
    SolarDataPreprocessor as preprocessor,
    train_XGBRegressor,
    predict_with_XGB,
    save_XGB,
    load_XGB,
    train_TabNetRegressor,
    predict_with_TabNet,
    save_TabNet,
    load_TabNet
)


def prepare_data(mode='remove_date'):
    """
    準備訓練數據
    
    Args:
        mode: 預處理模式
            - 'remove_date': 移除日期欄位，使用平均值填充缺失值（默認）
            - 'simplified_data': 簡化模式，移除年度/月份/平均單位發電量，合併發電比，用零值填充
    
    Returns:
        prep: 預處理器實例
        train_data: 訓練數據
        test_data: 測試數據
    """
    print("【載入資料】")
    
    # 初始化預處理器
    prep = preprocessor()
    
    # 讀取原始數據
    df = prep.load_data()
    
    # 探索數據
    prep.explore_data()
    
    # 根據模式選擇不同的清理策略
    if mode == 'remove_date':
        print(f"\n【預處理模式：{mode}】")
        print("✓ 移除日期欄位 + 平均值填充")
        
        # 清理數據（刪除日期欄位，處理缺失值）
        prep.clean_data(
            drop_columns=None,
            handle_missing='smart',  # 特別處理 T 值 + 平均值填充
            drop_date_columns=True   # 刪除日期欄位
        )
        
        print("\n【特徵工程】")
        print("✓ 保持原始欄位結構")
        
    elif mode == 'simplified_data':
        print(f"\n【預處理模式：{mode}】")
        print("✓ 簡化模式：移除年度/月份，合併發電比，零值填充")
        
        # 清理數據（使用零值填充）
        prep.clean_data(
            drop_columns=None,
            handle_missing='zero',   # 零值填充
            drop_date_columns=True
        )
        
        # 創建簡化模式
        prep.create_simplified_mode()
        
    else:
        raise ValueError(f"未知的預處理模式: {mode}。請使用 'remove_date' 或 'simplified_data'")
    
    # 準備訓練集和測試集
    train_data, test_data = prep.prepare_train_test(by_time=True)
    
    return prep, train_data, test_data


def extract_features_and_target(data, prep, 
                                 target_col=None):
    """
    從數據框中提取特徵和目標變量
    
    Args:
        data: 輸入 DataFrame
        prep: SolarDataPreprocessor 實例
        target_col: 目標欄位名稱（如果為 None 則自動檢測）
    
    Returns:
        features: 特徵 DataFrame（僅數值列）
        targets: 目標 Series
    """
    # 自動檢測目標列（優先檢測簡化模式的目標列）
    if target_col is None:
        if '發電比_Power_Ratio' in data.columns:
            target_col = '發電比_Power_Ratio'
        else:
            target_col = '發電量(度)/Power Generation(kWh)'
    
    # 取得預處理器中定義的特徵欄位
    feature_cols = prep.get_feature_columns()
    
    # 使用 'all' 類別中的特徵
    available_features = feature_cols['all']
    
    # 過濾出實際存在於數據中的特徵
    existing_features = [col for col in available_features if col in data.columns]
    
    # 刪除非數值欄位和特殊欄位
    drop_cols = [
        '發電站名稱/Name of The Power Station',
        'closest_station',  # 分類欄位，不適合直接用於數值模型
        'year_month',
        'season',
        'season_encoded',
        'calculated_efficiency',
        'efficiency_original',
        'station_chi_name',
        'station_encoded',
    ]
    
    # 獲得最終的特徵欄位
    features = data[existing_features].drop(columns=drop_cols, errors='ignore')
    
    # 確保所有特徵都是數值型態
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    features = features[numeric_features]
    
    targets = data[target_col]
    
    return features, targets


def cross_validate_xgb(X_train, y_train, n_splits=5,
                       n_estimators=200,
                       learning_rate=0.1,
                       max_depth=6):
    """
    對 XGBoost 模型執行 K-Fold 交叉驗證
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練目標
        n_splits: 交叉驗證的折數
        n_estimators: XGBoost 樹的數量
        learning_rate: 學習率
        max_depth: 樹的最大深度
    
    Returns:
        cv_results: 包含各指標的均值和標準差的字典
    """
    print("\n" + "="*80)
    print(f"【XGBoost K-Fold 交叉驗證 (k={n_splits})】")
    print("="*80)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_metrics = {
        'r2': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    fold_num = 1
    for train_idx, val_idx in kfold.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        print(f"\nFold {fold_num}/{n_splits}...")
        
        # 訓練模型
        model = train_XGBRegressor(
            X_train=X_fold_train,
            y_train=y_fold_train,
            X_valid=X_fold_val,
            y_valid=y_fold_val,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            random_state=42,
            verbose=0,
            use_grid_search=False,
            use_random_search=False,
        )
        
        # 預測
        y_pred = predict_with_XGB(model, X_fold_val)
        
        # 計算指標
        mse = mean_squared_error(y_fold_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_fold_val, y_pred)
        r2 = r2_score(y_fold_val, y_pred)
        mape = np.mean(np.abs((y_fold_val - y_pred) / y_fold_val)) * 100
        
        cv_metrics['mse'].append(mse)
        cv_metrics['rmse'].append(rmse)
        cv_metrics['mae'].append(mae)
        cv_metrics['r2'].append(r2)
        cv_metrics['mape'].append(mape)
        
        print(f"  R²: {r2:.4f}, RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, MAPE: {mape:.2f}%")
        fold_num += 1
    
    # 計算統計信息
    print("\n" + "-"*80)
    print("【交叉驗證結果統計】")
    print("-"*80)
    
    cv_results = {}
    for metric_name, metric_values in cv_metrics.items():
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        cv_results[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'values': metric_values
        }
        
        print(f"{metric_name.upper():<10} Mean: {mean_val:>12.4f}  Std: {std_val:>10.4f}")
    
    return cv_results


def cross_validate_tabnet(X_full, y_full, n_splits=5, verbose=0):
    """
    對 TabNet 模型執行 K-Fold 交叉驗證
    
    Args:
        X_full: 完整的訓練特徵
        y_full: 完整的訓練目標
        n_splits: 交叉驗證的折數
        verbose: 詳細程度
    
    Returns:
        cv_results: 包含各指標的均值和標準差的字典
    """
    print("\n" + "="*80)
    print(f"【TabNet K-Fold 交叉驗證 (k={n_splits})】")
    print("="*80)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 轉換為 numpy array
    X_np = X_full.to_numpy()
    y_np = y_full.to_numpy().reshape(-1, 1)
    
    cv_metrics = {
        'r2': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    fold_num = 1
    for train_idx, val_idx in kfold.split(X_np):
        X_fold_train = X_np[train_idx]
        y_fold_train = y_np[train_idx]
        X_fold_val = X_np[val_idx]
        y_fold_val = y_np[val_idx]
        
        print(f"\nFold {fold_num}/{n_splits}...")
        
        # 標準化特徵和目標
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_fold_train_scaled = X_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = X_scaler.transform(X_fold_val)
        y_fold_train_scaled = y_scaler.fit_transform(y_fold_train)
        y_fold_val_scaled = y_scaler.transform(y_fold_val)
        
        # 訓練模型
        model = train_TabNetRegressor(
            X_train=X_fold_train_scaled,
            y_train=y_fold_train_scaled,
            X_valid=X_fold_val_scaled,
            y_valid=y_fold_val_scaled,
            cat_idxs=[],
            cat_dims=[],
            feature_names=X_full.columns.tolist(),
            max_epochs=200,
            patience=20,
            batch_size=64,
            virtual_batch_size=32,
            n_steps=5,
            gamma=2.0,
            lambda_sparse=1e-4,
            verbose=verbose
        )
        
        # 預測
        y_pred_scaled = predict_with_TabNet(model, X_fold_val_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_fold_val_unscaled = y_fold_val.squeeze()
        y_pred = y_pred.squeeze()
        
        # 計算指標
        mse = mean_squared_error(y_fold_val_unscaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_fold_val_unscaled, y_pred)
        r2 = r2_score(y_fold_val_unscaled, y_pred)
        mape = np.mean(np.abs((y_fold_val_unscaled - y_pred) / y_fold_val_unscaled)) * 100
        
        cv_metrics['mse'].append(mse)
        cv_metrics['rmse'].append(rmse)
        cv_metrics['mae'].append(mae)
        cv_metrics['r2'].append(r2)
        cv_metrics['mape'].append(mape)
        
        print(f"  R²: {r2:.4f}, RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, MAPE: {mape:.2f}%")
        fold_num += 1
    
    # 計算統計信息
    print("\n" + "-"*80)
    print("【交叉驗證結果統計】")
    print("-"*80)
    
    cv_results = {}
    for metric_name, metric_values in cv_metrics.items():
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        cv_results[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'values': metric_values
        }
        
        print(f"{metric_name.upper():<10} Mean: {mean_val:>12.4f}  Std: {std_val:>10.4f}")
    
    return cv_results


def run_xgb_model(prep, train_data, test_data, 
                  n_estimators=200,
                  learning_rate=0.1,
                  max_depth=6,
                  verbose=1):
    """執行 XGBoost 模型訓練和評估"""
    
    print("\n" + "="*80)
    print("【XGBoost 模型訓練】")
    print("="*80)
    
    # 提取特徵和目標變量（使用 preprocessor）
    X_train, y_train = extract_features_and_target(train_data, prep)
    X_test, y_test = extract_features_and_target(test_data, prep)
    
    print(f"\n【數據形狀】")
    print(f"訓練集 X: {X_train.shape}")
    print(f"訓練集 y: {y_train.shape}")
    print(f"測試集 X: {X_test.shape}")
    print(f"測試集 y: {y_test.shape}")
    
    print(f"\n【特徵列表】({len(X_train.columns)} 個特徵)")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i}. {col}")
    
    # 訓練模型
    print("\n" + "-"*80)
    model = train_XGBRegressor(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_test,
        y_valid=y_test,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        random_state=42,
        verbose=verbose,
        use_grid_search=False,
        use_random_search=False,
    )
    print("-"*80)
    
    # 在測試集上進行預測
    y_pred = predict_with_XGB(model, X_test)
    
    # 計算評估指標
    print("\n【測試集評估指標】")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 計算相對誤差和目標變數統計
    y_test_mean = y_test.mean()
    y_test_std = y_test.std()
    y_test_min = y_test.min()
    y_test_max = y_test.max()
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n【預測目標】")
    print(f"目標變數: 發電量(度)/Power Generation(kWh)")
    print(f"實際數值範圍: {y_test_min:,.0f} ~ {y_test_max:,.0f} kWh")
    print(f"平均值: {y_test_mean:,.0f} kWh")
    print(f"標準差: {y_test_std:,.0f} kWh")
    
    print(f"\n【誤差指標】")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} kWh")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} kWh")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    print(f"\n【模型表現】")
    if r2 > 0.9:
        print(f"✓ 優秀: R² = {r2:.4f} (解釋了 {r2*100:.2f}% 的方差)")
    elif r2 > 0.7:
        print(f"✓ 良好: R² = {r2:.4f}")
    else:
        print(f"⚠️  需改進: R² = {r2:.4f}")
    
    print(f"✓ 相對誤差: {mape:.2f}% (MAPE)")
    print(f"✓ 平均絕對誤差佔平均值的比例: {mae/y_test_mean*100:.2f}%")
    
    # 保存 XGB 模型
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'xgb_model'
    
    print(f"\n【保存模型】")
    print(f"✓ XGBoost 模型已保存")
    save_XGB(model, str(model_path))
    
    # 返回結果
    results = {
        'model_name': 'XGBoost',
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }
    
    return results


def run_tabnet_model(prep, train_data, test_data, verbose=1):
    """執行 TabNet 模型訓練和評估"""
    
    print("\n" + "="*80)
    print("【TabNet 模型訓練】")
    print("="*80)
    
    # 提取特徵和目標變量
    X_train, y_train = extract_features_and_target(train_data, prep)
    X_test, y_test = extract_features_and_target(test_data, prep)
    
    print(f"\n【數據形狀】")
    print(f"訓練集 X: {X_train.shape}")
    print(f"訓練集 y: {y_train.shape}")
    print(f"測試集 X: {X_test.shape}")
    print(f"測試集 y: {y_test.shape}")
    
    print(f"\n【特徵列表】({len(X_train.columns)} 個特徵)")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i}. {col}")
    
    # 轉換為 numpy array
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy().reshape(-1, 1)
    y_test_np = y_test.to_numpy().reshape(-1, 1)
    
    # TabNet 需要特徵標準化（深度學習模型對特徵縮放敏感）
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_np = X_scaler.fit_transform(X_train_np)
    X_test_np = X_scaler.transform(X_test_np)
    y_train_np = y_scaler.fit_transform(y_train_np)
    y_test_np_scaled = y_scaler.transform(y_test_np)
    y_test_np_unscaled = y_test_np.copy()  # 保留原始值用於計算指標
    
    # 訓練模型
    print("\n" + "-"*80)
    print("開始訓練 TabNet...（使用標準化後的特徵）")
    model = train_TabNetRegressor(
        X_train=X_train_np,
        y_train=y_train_np,
        X_valid=X_test_np,
        y_valid=y_test_np_scaled,
        cat_idxs=[],  # 沒有分類特徵，所有都是數值
        cat_dims=[],  # 沒有分類特徵維度
        feature_names=X_train.columns.tolist(),
        max_epochs=200,
        patience=20,
        batch_size=64,
        virtual_batch_size=32,
        n_steps=5,
        gamma=2.0,
        lambda_sparse=1e-4,
        verbose=verbose
    )
    print("-"*80)
    
    # 在測試集上進行預測
    y_pred_scaled = predict_with_TabNet(model, X_test_np)
    
    # 將預測值和目標值反轉回原始縮放尺度
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_np = y_test_np_unscaled
    y_pred = y_pred.squeeze()  # Convert from 2D to 1D for sklearn metrics
    y_test_np_1d = y_test_np.squeeze()  # Convert back to 1D for sklearn metrics
    
    # 計算評估指標
    print("\n【測試集評估指標】")
    mse = mean_squared_error(y_test_np_1d, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np_1d, y_pred)
    r2 = r2_score(y_test_np_1d, y_pred)
    
    # 計算相對誤差和目標變數統計
    y_test_mean = y_test_np_1d.mean()
    y_test_std = y_test_np_1d.std()
    y_test_min = y_test_np_1d.min()
    y_test_max = y_test_np_1d.max()
    mape = np.mean(np.abs((y_test_np_1d - y_pred) / y_test_np_1d)) * 100
    
    print(f"\n【預測目標】")
    print(f"目標變數: 發電量(度)/Power Generation(kWh)")
    print(f"實際數值範圍: {y_test_min:,.0f} ~ {y_test_max:,.0f} kWh")
    print(f"平均值: {y_test_mean:,.0f} kWh")
    print(f"標準差: {y_test_std:,.0f} kWh")
    
    print(f"\n【誤差指標】")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} kWh")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} kWh")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    print(f"\n【模型表現】")
    if r2 > 0.9:
        print(f"✓ 優秀: R² = {r2:.4f} (解釋了 {r2*100:.2f}% 的方差)")
    elif r2 > 0.7:
        print(f"✓ 良好: R² = {r2:.4f}")
    else:
        print(f"⚠️  需改進: R² = {r2:.4f}")
    
    print(f"✓ 相對誤差: {mape:.2f}% (MAPE)")
    print(f"✓ 平均絕對誤差佔平均值的比例: {mae/y_test_mean*100:.2f}%")
    
    # 保存 TabNet 模型
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / 'tabnet_model')
    
    print(f"\n【保存模型】")
    print(f"✓ TabNet 模型已保存")
    save_TabNet(model, model_path)
    
    # 返回結果
    results = {
        'model_name': 'TabNet',
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }
    
    return results


def plot_regression_results(model_name, y_test, y_pred, metrics, result_dir):
    """
    生成迴歸模型的常見可視化圖表
    
    包括：
    - 實際 vs 預測散點圖
    - 殘差分佈直方圖
    - 預測 vs 殘差圖
    - 預測 vs 實際值折線圖
    """
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    residuals = y_test - y_pred
    
    # 檢測目標變量類型以調整標籤
    is_power_ratio = '發電比' in str(y_test.name) if hasattr(y_test, 'name') else False
    
    if is_power_ratio:
        y_unit = '(無單位)'
        target_desc = '發電比'
    else:
        y_unit = '(kWh)'
        target_desc = '發電量(度)'
    
    # 建立 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - 迴歸性能可視化', fontsize=16, fontweight='bold')
    
    # 1. 實際 vs 預測散點圖
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美預測')
    ax1.set_xlabel(f'實際值 {y_unit}', fontsize=11)
    ax1.set_ylabel(f'預測值 {y_unit}', fontsize=11)
    ax1.set_title('實際 vs 預測', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 殘差分佈直方圖
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='零誤差')
    ax2.set_xlabel(f'殘差 {y_unit}', fontsize=11)
    ax2.set_ylabel('頻率', fontsize=11)
    ax2.set_title('殘差分佈', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 預測值 vs 殘差圖
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel(f'預測值 {y_unit}', fontsize=11)
    ax3.set_ylabel(f'殘差 {y_unit}', fontsize=11)
    ax3.set_title('預測值 vs 殘差', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能指標文字框
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if is_power_ratio:
        format_str = '{:.4f}'
    else:
        format_str = '{:,.0f}'
    
    metrics_text = f"""
    【模型評估指標】
    
    R² Score:        {metrics['r2']:.4f}
    
    RMSE:            {format_str.format(metrics['rmse'])} {y_unit}
    MAE:             {format_str.format(metrics['mae'])} {y_unit}
    MSE:             {metrics['mse']:.2f}
    
    MAPE:            {metrics['mape']:.2f}%
    
    【數據統計】
    
    測試集樣本數:     {len(y_test)}
    實際值範圍:       {format_str.format(y_test.min())} ~ {format_str.format(y_test.max())} {y_unit}
    實際值平均:       {format_str.format(y_test.mean())} {y_unit}
    實際值標準差:     {format_str.format(y_test.std())} {y_unit}
    
    預測值平均:       {format_str.format(y_pred.mean())} {y_unit}
    預測值標準差:     {format_str.format(y_pred.std())} {y_unit}
    
    【殘差統計】
    
    殘差平均:         {format_str.format(residuals.mean())} {y_unit}
    殘差標準差:       {format_str.format(residuals.std())} {y_unit}
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = result_dir / f'{model_name.lower()}_regression_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {save_path}")
    plt.close()
    
    return save_path


def plot_model_comparison(xgb_results, tabnet_results, result_dir):
    """繪製 XGB 和 TabNet 模型的對比圖"""
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢測目標變量類型以調整標籤
    is_power_ratio = '發電比' in str(xgb_results['y_test'].name) or '發電比' in str(tabnet_results['y_test'].name)
    
    # 準備數據
    models = ['XGBoost', 'TabNet']
    r2_scores = [xgb_results['metrics']['r2'], tabnet_results['metrics']['r2']]
    mae_scores = [xgb_results['metrics']['mae'], tabnet_results['metrics']['mae']]
    rmse_scores = [xgb_results['metrics']['rmse'], tabnet_results['metrics']['rmse']]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('XGBoost vs TabNet 模型性能對比', fontsize=14, fontweight='bold')
    
    # 根據目標變量類型調整標籤
    if is_power_ratio:
        mae_label = 'MAE (發電比誤差)'
        rmse_label = 'RMSE (發電比誤差)'
        mae_offset = max(mae_scores) * 0.05
        rmse_offset = max(rmse_scores) * 0.05
    else:
        mae_label = 'MAE (kWh)'
        rmse_label = 'RMSE (kWh)'
        mae_offset = 5000
        rmse_offset = 10000
    
    # R² 對比
    ax1 = axes[0]
    bars1 = ax1.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² 得分比較（越高越好）', fontsize=12, fontweight='bold')
    ax1.set_ylim([min(r2_scores) - 0.1, 1.0])
    for i, (bar, val) in enumerate(zip(bars1, r2_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MAE 對比
    ax2 = axes[1]
    bars2 = ax2.bar(models, mae_scores, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel(mae_label, fontsize=11)
    ax2.set_title('平均絕對誤差（越低越好）', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars2, mae_scores)):
        if is_power_ratio:
            label = f'{val:.4f}'
        else:
            label = f'{val:,.0f}'
        ax2.text(bar.get_x() + bar.get_width()/2, val + mae_offset, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # RMSE 對比
    ax3 = axes[2]
    bars3 = ax3.bar(models, rmse_scores, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel(rmse_label, fontsize=11)
    ax3.set_title('均方根誤差（越低越好）', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars3, rmse_scores)):
        if is_power_ratio:
            label = f'{val:.4f}'
        else:
            label = f'{val:,.0f}'
        ax3.text(bar.get_x() + bar.get_width()/2, val + rmse_offset, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = result_dir / 'model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {save_path}")
    plt.close()
    
    return save_path


def plot_xgb_feature_importance(xgb_model, feature_names, result_dir):
    """繪製 XGBoost 特徵重要性圖"""    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'Arial Unicode Ms', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 取得特徵重要性
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]  # 取前 15 個
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(indices)), importance[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('特徵重要性', fontsize=11)
    ax.set_title('XGBoost - 前 15 個重要特徵', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = result_dir / 'xgb_feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {save_path}")
    plt.close()
    
    return save_path


def plot_cv_comparison(xgb_cv_results, tabnet_cv_results, result_dir):
    """繪製 K-Fold 交叉驗證結果對比圖"""
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備數據
    metrics_names = ['R² Score', 'RMSE (kWh)', 'MAE (kWh)', 'MAPE (%)']
    metrics_keys = ['r2', 'rmse', 'mae', 'mape']
    models = ['XGBoost', 'TabNet']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Fold 交叉驗證結果對比', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
        ax = axes[idx]
        
        xgb_mean = xgb_cv_results[metric_key]['mean']
        xgb_std = xgb_cv_results[metric_key]['std']
        tabnet_mean = tabnet_cv_results[metric_key]['mean']
        tabnet_std = tabnet_cv_results[metric_key]['std']
        
        x_pos = np.arange(2)
        means = [xgb_mean, tabnet_mean]
        stds = [xgb_std, tabnet_std]
        
        bars = ax.bar(x_pos, means, yerr=stds, color=['#1f77b4', '#ff7f0e'], 
                     alpha=0.7, edgecolor='black', capsize=5, error_kw={'linewidth': 2})
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} 交叉驗證結果', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = mean
            ax.text(bar.get_x() + bar.get_width()/2, height + std + abs(height)*0.05,
                   f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = result_dir / 'cv_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {save_path}")
    plt.close()
    
    return save_path


def compare_models(xgb_results, tabnet_results):
    """比較 XGB 和 TabNet 模型的性能"""
    print("\n" + "="*80)
    print("【模型性能比較】")
    print("="*80)
    
    metrics_to_compare = ['mse', 'rmse', 'mae', 'r2', 'mape']
    
    print(f"\n{'指標':<20} {'XGBoost':<20} {'TabNet':<20}")
    print("-" * 80)
    
    for metric in metrics_to_compare:
        xgb_val = xgb_results['metrics'][metric]
        tabnet_val = tabnet_results['metrics'][metric]
        
        print(f"{metric:<20} {xgb_val:<20.2f} {tabnet_val:<20.2f}")

    return


def run_dimensionality_reduction(mode='remove_date'):
    """
    執行 PCA 維度縮減實驗
    
    測試不同維度數量下的模型性能
    - 測試維度: 原始 (baseline), PCA-5, PCA-10, PCA-15, PCA-20
    - 對比: XGBoost vs TabNet
    - 輸出: 維度縮減效果圖表和性能對比表
    """
    print("\n" + "="*80)
    print("【PCA 維度縮減實驗：維度數量對模型性能的影響】")
    print("="*80)
    
    # 準備數據
    prep, train_data, test_data = prepare_data(mode=mode)
    
    # 提取完整訓練集和測試集
    X_train_full, y_train_full = extract_features_and_target(train_data, prep)
    X_test, y_test = extract_features_and_target(test_data, prep)
    
    print(f"\n【數據配置】")
    print(f"訓練集大小: {X_train_full.shape[0]} 樣本")
    print(f"測試集大小: {X_test.shape[0]} 樣本")
    print(f"原始特徵數量: {X_train_full.shape[1]}")
    
    # 標準化數據（PCA 需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # 定義要測試的維度（不超過原始特徵數量）
    n_features = X_train_full.shape[1]
    # 動態生成維度列表：根據特徵數量調整
    if n_features >= 15:
        n_components_list = [3, 5, 10, 15, n_features]  # 最後一個是原始維度
    elif n_features >= 10:
        n_components_list = [3, 5, 8, n_features]
    else:
        n_components_list = [3, 5, n_features]
    
    # 存儲結果
    results = {
        'n_components': [],
        'xgb_r2': [],
        'xgb_rmse': [],
        'xgb_mae': [],
        'xgb_mape': [],
        'xgb_train_time': [],
        'tabnet_r2': [],
        'tabnet_rmse': [],
        'tabnet_mae': [],
        'tabnet_mape': [],
        'tabnet_train_time': [],
        'explained_variance': [],
        'pca_var_ratio': []
    }
    
    print(f"\n{'維度':<10} {'解釋方差':<15} {'XGBoost R²':<15} {'TabNet R²':<15} {'XGB耗時':<12} {'TabNet耗時':<12}")
    print("-" * 100)
    
    # 對每個維度執行實驗
    for n_comp in n_components_list:
        if n_comp == X_train_full.shape[1]:
            # 原始維度（無PCA）
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
            explained_var = 1.0
            pca_var_list = None
            dim_label = f"原始({n_comp})"
        else:
            # 應用PCA
            pca = PCA(n_components=n_comp, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            explained_var = sum(pca.explained_variance_ratio_)
            pca_var_list = pca.explained_variance_ratio_
            dim_label = f"PCA-{n_comp}"
        
        # ===== XGBoost 訓練 =====
        start_time = time.time()
        xgb_model = train_XGBRegressor(
            X_train=pd.DataFrame(X_train_pca),
            y_train=y_train_full,
            X_valid=pd.DataFrame(X_test_pca),
            y_valid=y_test,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            random_state=42,
            verbose=0,
            use_grid_search=False,
            use_random_search=False,
        )
        xgb_train_time = time.time() - start_time
        
        y_pred_xgb = predict_with_XGB(xgb_model, pd.DataFrame(X_test_pca))
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
        
        # ===== TabNet 訓練 =====
        y_train_np = y_train_full.to_numpy().reshape(-1, 1)
        y_test_np = y_test.to_numpy().reshape(-1, 1)
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_np)
        y_test_scaled = y_scaler.transform(y_test_np)
        
        start_time = time.time()
        tabnet_model = train_TabNetRegressor(
            X_train=X_train_pca,
            y_train=y_train_scaled,
            X_valid=X_test_pca,
            y_valid=y_test_scaled,
            cat_idxs=[],
            cat_dims=[],
            feature_names=[f'PC{i+1}' if n_comp != X_train_full.shape[1] else f'F{i+1}' 
                          for i in range(X_train_pca.shape[1])],
            max_epochs=200,
            patience=20,
            batch_size=32,
            virtual_batch_size=16,
            n_steps=5,
            gamma=2.0,
            lambda_sparse=1e-4,
            verbose=0
        )
        tabnet_train_time = time.time() - start_time
        
        y_pred_tabnet_scaled = predict_with_TabNet(tabnet_model, X_test_pca)
        y_pred_tabnet = y_scaler.inverse_transform(y_pred_tabnet_scaled)
        y_pred_tabnet = y_pred_tabnet.squeeze()
        y_test_unscaled = y_test_np.squeeze()
        
        tabnet_r2 = r2_score(y_test_unscaled, y_pred_tabnet)
        tabnet_rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_tabnet))
        tabnet_mae = mean_absolute_error(y_test_unscaled, y_pred_tabnet)
        tabnet_mape = np.mean(np.abs((y_test_unscaled - y_pred_tabnet) / y_test_unscaled)) * 100
        
        # 存儲結果
        results['n_components'].append(n_comp)
        results['xgb_r2'].append(xgb_r2)
        results['xgb_rmse'].append(xgb_rmse)
        results['xgb_mae'].append(xgb_mae)
        results['xgb_mape'].append(xgb_mape)
        results['xgb_train_time'].append(xgb_train_time)
        results['tabnet_r2'].append(tabnet_r2)
        results['tabnet_rmse'].append(tabnet_rmse)
        results['tabnet_mae'].append(tabnet_mae)
        results['tabnet_mape'].append(tabnet_mape)
        results['tabnet_train_time'].append(tabnet_train_time)
        results['explained_variance'].append(explained_var * 100)
        results['pca_var_ratio'].append(pca_var_list)
        
        # 打印結果
        print(f"{dim_label:<10} {explained_var*100:>13.1f}%  {xgb_r2:>14.4f}  {tabnet_r2:>14.4f}  {xgb_train_time:>11.2f}s  {tabnet_train_time:>11.2f}s")
    
    # 繪製 PCA 維度縮減效果
    print("\n【生成 PCA 維度縮減分析圖表】")
    result_dir = Path(__file__).parent.parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢測目標變量類型
    is_power_ratio = '發電比' in str(y_test.name) if hasattr(y_test, 'name') else False
    target_desc = '發電比' if is_power_ratio else '發電量(度)'
    
    # 維度標籤
    dim_labels = [f"PCA-{n}" if n != X_train_full.shape[1] else f"原始\n({n}維)" 
                  for n in results['n_components']]
    
    # 建立 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'PCA 維度縮減分析：維度數量對模型性能的影響 ({target_desc})', fontsize=16, fontweight='bold')
    
    # 1. R² Score vs 維度
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results['n_components']))
    ax1.plot(x_pos, results['xgb_r2'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax1.plot(x_pos, results['tabnet_r2'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax1.set_xlabel('維度', fontsize=11)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² Score vs 維度數量', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dim_labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    for i, (xgb_val, tabnet_val) in enumerate(zip(results['xgb_r2'], results['tabnet_r2'])):
        ax1.text(i, xgb_val + 0.02, f'{xgb_val:.3f}', ha='center', fontsize=8)
        ax1.text(i, tabnet_val - 0.05, f'{tabnet_val:.3f}', ha='center', fontsize=8)
    
    # 2. 訓練時間對比
    ax2 = axes[0, 1]
    width = 0.35
    ax2.bar(x_pos - width/2, results['xgb_train_time'], width, label='XGBoost', color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.bar(x_pos + width/2, results['tabnet_train_time'], width, label='TabNet', color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('維度', fontsize=11)
    ax2.set_ylabel('訓練時間 (秒)', fontsize=11)
    ax2.set_title('訓練時間 vs 維度數量', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dim_labels, fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. MAE vs 維度
    ax3 = axes[1, 0]
    if is_power_ratio:
        ax3.plot(x_pos, results['xgb_mae'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
        ax3.plot(x_pos, results['tabnet_mae'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
        ax3.set_ylabel('MAE (無單位)', fontsize=11)
        for i, (xgb_val, tabnet_val) in enumerate(zip(results['xgb_mae'], results['tabnet_mae'])):
            ax3.text(i, xgb_val + 0.002, f'{xgb_val:.4f}', ha='center', fontsize=8)
            ax3.text(i, tabnet_val - 0.004, f'{tabnet_val:.4f}', ha='center', fontsize=8)
    else:
        ax3.plot(x_pos, results['xgb_mae'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
        ax3.plot(x_pos, results['tabnet_mae'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
        ax3.set_ylabel('MAE (kWh)', fontsize=11)
        for i, (xgb_val, tabnet_val) in enumerate(zip(results['xgb_mae'], results['tabnet_mae'])):
            ax3.text(i, xgb_val + 5, f'{xgb_val:.0f}', ha='center', fontsize=8)
            ax3.text(i, tabnet_val - 5, f'{tabnet_val:.0f}', ha='center', fontsize=8)
    ax3.set_xlabel('維度', fontsize=11)
    ax3.set_title('平均絕對誤差 (MAE) vs 維度數量', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dim_labels, fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. MAPE vs 維度
    ax4 = axes[1, 1]
    ax4.plot(x_pos, results['xgb_mape'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax4.plot(x_pos, results['tabnet_mape'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax4.set_xlabel('維度', fontsize=11)
    ax4.set_ylabel('MAPE (%)', fontsize=11)
    ax4.set_title('平均絕對百分比誤差 (MAPE) vs 維度數量', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dim_labels, fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    for i, (xgb_val, tabnet_val) in enumerate(zip(results['xgb_mape'], results['tabnet_mape'])):
        ax4.text(i, xgb_val + 0.3, f'{xgb_val:.1f}%', ha='center', fontsize=8)
        ax4.text(i, tabnet_val - 0.5, f'{tabnet_val:.1f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    save_path = result_dir / 'pca_dimensionality_reduction.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ PCA 維度縮減圖表已保存: {save_path}")
    plt.close()
    
    # 繪製 PCA 解釋方差比例
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(dim_labels, results['explained_variance'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('維度', fontsize=11)
    ax.set_ylabel('累積解釋方差比例 (%)', fontsize=11)
    ax.set_title('PCA 累積解釋方差比例', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(results['explained_variance']):
        ax.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_path = result_dir / 'pca_explained_variance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ PCA 解釋方差圖表已保存: {save_path}")
    plt.close()
    
    # 生成分析總結
    print("\n" + "="*80)
    print("【PCA 維度縮減分析結論】")
    print("="*80)
    
    # 找出最優維度
    xgb_max_idx = np.argmax(results['xgb_r2'])
    tabnet_max_idx = np.argmax(results['tabnet_r2'])
    
    print(f"\n【性能對比】")
    print(f"XGBoost 最佳維度: {results['n_components'][xgb_max_idx]} 維，R² = {results['xgb_r2'][xgb_max_idx]:.4f}")
    print(f"TabNet 最佳維度: {results['n_components'][tabnet_max_idx]} 維，R² = {results['tabnet_r2'][tabnet_max_idx]:.4f}")
    
    # 維度縮減帶來的性能變化
    print(f"\n【與原始維度的性能差異】")
    original_idx = results['n_components'].index(X_train_full.shape[1])
    for i, n_comp in enumerate(results['n_components'][:-1]):  # 排除原始維度
        xgb_diff = results['xgb_r2'][i] - results['xgb_r2'][original_idx]
        tabnet_diff = results['tabnet_r2'][i] - results['tabnet_r2'][original_idx]
        xgb_speedup = results['xgb_train_time'][original_idx] / results['xgb_train_time'][i]
        tabnet_speedup = results['tabnet_train_time'][original_idx] / results['tabnet_train_time'][i]
        
        print(f"\nPCA-{n_comp} vs 原始維度:")
        print(f"  XGBoost: R² {xgb_diff:+.4f} (△{xgb_diff/results['xgb_r2'][original_idx]*100:+.2f}%) | "
              f"訓練加速 {xgb_speedup:.2f}x")
        print(f"  TabNet:  R² {tabnet_diff:+.4f} (△{tabnet_diff/results['tabnet_r2'][original_idx]*100:+.2f}%) | "
              f"訓練加速 {tabnet_speedup:.2f}x")
    
    print(f"\n✓ PCA 維度縮減實驗完成！")
    
    return {
        'results': results,
        'original_n_features': X_train_full.shape[1]
    }


def run_learning_curve(mode='remove_date'):
    """
    執行 Learning Curve 實驗
    
    測試不同訓練數據量對模型性能的影響
    - 測試訓練集大小: 10%, 25%, 50%, 75%, 100%
    - 對比: XGBoost vs TabNet
    - 輸出: Learning curve 圖表和性能對比表
    """
    print("\n" + "="*80)
    print("【Learning Curve 實驗：訓練數據量影響分析】")
    print("="*80)
    
    # 準備數據
    prep, train_data, test_data = prepare_data(mode=mode)
    
    # 提取完整訓練集和測試集
    X_train_full, y_train_full = extract_features_and_target(train_data, prep)
    X_test, y_test = extract_features_and_target(test_data, prep)
    
    print(f"\n【數據配置】")
    print(f"完整訓練集大小: {X_train_full.shape[0]} 樣本")
    print(f"測試集大小: {X_test.shape[0]} 樣本")
    print(f"特徵數量: {X_train_full.shape[1]}")
    
    # 定義訓練集大小的百分比
    train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    # 存儲結果
    xgb_metrics = {
        'r2': [],
        'rmse': [],
        'mae': [],
        'mape': [],
        'train_size': []
    }
    
    tabnet_metrics = {
        'r2': [],
        'rmse': [],
        'mae': [],
        'mape': [],
        'train_size': []
    }
    
    print(f"\n{'訓練集比例':<15} {'XGBoost R²':<15} {'TabNet R²':<15} {'XGBoost MAE':<15} {'TabNet MAE':<15}")
    print("-" * 80)
    
    # 對每個訓練集大小執行實驗
    for size_ratio in train_sizes:
        # 計算要使用的訓練集大小
        n_samples = int(len(X_train_full) * size_ratio)
        
        # 截取訓練集
        X_train_subset = X_train_full.iloc[:n_samples]
        y_train_subset = y_train_full.iloc[:n_samples]
        
        actual_ratio = n_samples / len(X_train_full)
        
        # ===== XGBoost =====
        xgb_model = train_XGBRegressor(
            X_train=X_train_subset,
            y_train=y_train_subset,
            X_valid=X_test,
            y_valid=y_test,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            random_state=42,
            verbose=0,
            use_grid_search=False,
            use_random_search=False,
        )
        
        y_pred_xgb = predict_with_XGB(xgb_model, X_test)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
        
        xgb_metrics['r2'].append(xgb_r2)
        xgb_metrics['rmse'].append(xgb_rmse)
        xgb_metrics['mae'].append(xgb_mae)
        xgb_metrics['mape'].append(xgb_mape)
        xgb_metrics['train_size'].append(n_samples)
        
        # ===== TabNet =====
        X_train_np = X_train_subset.to_numpy()
        X_test_np = X_test.to_numpy()
        y_train_np = y_train_subset.to_numpy().reshape(-1, 1)
        y_test_np = y_test.to_numpy().reshape(-1, 1)
        
        # 標準化
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train_np)
        X_test_scaled = X_scaler.transform(X_test_np)
        y_train_scaled = y_scaler.fit_transform(y_train_np)
        y_test_scaled = y_scaler.transform(y_test_np)
        
        tabnet_model = train_TabNetRegressor(
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_valid=X_test_scaled,
            y_valid=y_test_scaled,
            cat_idxs=[],
            cat_dims=[],
            feature_names=X_train_subset.columns.tolist(),
            max_epochs=200,
            patience=20,
            batch_size=32,
            virtual_batch_size=16,
            n_steps=5,
            gamma=2.0,
            lambda_sparse=1e-4,
            verbose=0
        )
        
        y_pred_tabnet_scaled = predict_with_TabNet(tabnet_model, X_test_scaled)
        y_pred_tabnet = y_scaler.inverse_transform(y_pred_tabnet_scaled)
        y_pred_tabnet = y_pred_tabnet.squeeze()
        y_test_unscaled = y_test_np.squeeze()
        
        tabnet_r2 = r2_score(y_test_unscaled, y_pred_tabnet)
        tabnet_rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_tabnet))
        tabnet_mae = mean_absolute_error(y_test_unscaled, y_pred_tabnet)
        tabnet_mape = np.mean(np.abs((y_test_unscaled - y_pred_tabnet) / y_test_unscaled)) * 100
        
        tabnet_metrics['r2'].append(tabnet_r2)
        tabnet_metrics['rmse'].append(tabnet_rmse)
        tabnet_metrics['mae'].append(tabnet_mae)
        tabnet_metrics['mape'].append(tabnet_mape)
        tabnet_metrics['train_size'].append(n_samples)
        
        # 打印結果
        print(f"{actual_ratio*100:>13.0f}%  {xgb_r2:>14.4f}  {tabnet_r2:>14.4f}  {xgb_mae:>14.2f}  {tabnet_mae:>14.2f}")
    
    # 繪製 Learning Curves
    print("\n【生成 Learning Curve 圖表】")
    result_dir = Path(__file__).parent.parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 轉換訓練集大小為百分比顯示
    train_sizes_percent = [size / len(X_train_full) * 100 for size in xgb_metrics['train_size']]
    
    # 檢測目標變量類型（決定單位和標籤）
    is_power_ratio = '發電比' in str(y_test.name) if hasattr(y_test, 'name') else False
    
    if is_power_ratio:
        unit_label = '(無單位)'
        rmse_unit = '(無單位)'
        mae_unit = '(無單位)'
        target_desc = '發電比'
        # 調整文本標籤精度
        rmse_format = '.4f'
        mae_format = '.4f'
        rmse_text_offset = 0.01
        mae_text_offset = 0.01
    else:
        unit_label = '(kWh)'
        rmse_unit = '(kWh)'
        mae_unit = '(kWh)'
        target_desc = '發電量(度)'
        rmse_format = '.0f'
        mae_format = '.0f'
        rmse_text_offset = 5
        mae_text_offset = 5
    
    # 建立 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Learning Curve 分析：訓練數據量對模型性能的影響 ({target_desc})', fontsize=16, fontweight='bold')
    
    # 1. R² Score
    ax1 = axes[0, 0]
    ax1.plot(train_sizes_percent, xgb_metrics['r2'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax1.plot(train_sizes_percent, tabnet_metrics['r2'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax1.set_xlabel('訓練集大小 (%)', fontsize=11)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² Score vs 訓練集大小', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    for i, (x, y_xgb, y_tabnet) in enumerate(zip(train_sizes_percent, xgb_metrics['r2'], tabnet_metrics['r2'])):
        ax1.text(x, y_xgb + 0.02, f'{y_xgb:.3f}', ha='center', fontsize=9)
        ax1.text(x, y_tabnet - 0.05, f'{y_tabnet:.3f}', ha='center', fontsize=9)
    
    # 2. RMSE
    ax2 = axes[0, 1]
    ax2.plot(train_sizes_percent, xgb_metrics['rmse'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax2.plot(train_sizes_percent, tabnet_metrics['rmse'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax2.set_xlabel('訓練集大小 (%)', fontsize=11)
    ax2.set_ylabel(f'RMSE {rmse_unit}', fontsize=11)
    ax2.set_title('RMSE vs 訓練集大小', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    for i, (x, y_xgb, y_tabnet) in enumerate(zip(train_sizes_percent, xgb_metrics['rmse'], tabnet_metrics['rmse'])):
        if is_power_ratio:
            ax2.text(x, y_xgb + rmse_text_offset, f'{y_xgb:.4f}', ha='center', fontsize=9)
            ax2.text(x, y_tabnet - rmse_text_offset*2, f'{y_tabnet:.4f}', ha='center', fontsize=9)
        else:
            ax2.text(x, y_xgb + rmse_text_offset, f'{y_xgb:.0f}', ha='center', fontsize=9)
            ax2.text(x, y_tabnet - rmse_text_offset, f'{y_tabnet:.0f}', ha='center', fontsize=9)
    
    # 3. MAE
    ax3 = axes[1, 0]
    ax3.plot(train_sizes_percent, xgb_metrics['mae'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax3.plot(train_sizes_percent, tabnet_metrics['mae'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax3.set_xlabel('訓練集大小 (%)', fontsize=11)
    ax3.set_ylabel(f'MAE {mae_unit}', fontsize=11)
    ax3.set_title('MAE vs 訓練集大小', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    for i, (x, y_xgb, y_tabnet) in enumerate(zip(train_sizes_percent, xgb_metrics['mae'], tabnet_metrics['mae'])):
        if is_power_ratio:
            ax3.text(x, y_xgb + mae_text_offset, f'{y_xgb:.4f}', ha='center', fontsize=9)
            ax3.text(x, y_tabnet - mae_text_offset*2, f'{y_tabnet:.4f}', ha='center', fontsize=9)
        else:
            ax3.text(x, y_xgb + mae_text_offset, f'{y_xgb:.0f}', ha='center', fontsize=9)
            ax3.text(x, y_tabnet - mae_text_offset, f'{y_tabnet:.0f}', ha='center', fontsize=9)
    
    # 4. MAPE
    ax4 = axes[1, 1]
    ax4.plot(train_sizes_percent, xgb_metrics['mape'], 'o-', linewidth=2, markersize=8, label='XGBoost', color='#1f77b4')
    ax4.plot(train_sizes_percent, tabnet_metrics['mape'], 's-', linewidth=2, markersize=8, label='TabNet', color='#ff7f0e')
    ax4.set_xlabel('訓練集大小 (%)', fontsize=11)
    ax4.set_ylabel('MAPE (%)', fontsize=11)
    ax4.set_title('MAPE vs 訓練集大小', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    for i, (x, y_xgb, y_tabnet) in enumerate(zip(train_sizes_percent, xgb_metrics['mape'], tabnet_metrics['mape'])):
        ax4.text(x, y_xgb + 0.5, f'{y_xgb:.1f}%', ha='center', fontsize=9)
        ax4.text(x, y_tabnet - 1, f'{y_tabnet:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = result_dir / 'learning_curve_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning Curve 圖表已保存: {save_path}")
    plt.close()
    
    # 生成分析總結
    print("\n" + "="*80)
    print("【Learning Curve 分析結論】")
    print("="*80)
    
    # 比較在不同訓練集大小下的性能改進
    xgb_improvement = ((xgb_metrics['r2'][-1] - xgb_metrics['r2'][0]) / abs(xgb_metrics['r2'][0])) * 100
    tabnet_improvement = ((tabnet_metrics['r2'][-1] - tabnet_metrics['r2'][0]) / abs(tabnet_metrics['r2'][0])) * 100
    
    print(f"\n【R² 性能改進 (10% → 100% 訓練數據)】")
    print(f"XGBoost: {xgb_metrics['r2'][0]:.4f} → {xgb_metrics['r2'][-1]:.4f} ({xgb_improvement:+.2f}%)")
    print(f"TabNet:  {tabnet_metrics['r2'][0]:.4f} → {tabnet_metrics['r2'][-1]:.4f} ({tabnet_improvement:+.2f}%)")
    
    # 識別數據飽和點
    print(f"\n【數據量對性能的敏感度】")
    print(f"訓練集大小    XGBoost R²   TabNet R²   性能差異")
    print("-" * 60)
    for i, size_pct in enumerate(train_sizes_percent):
        diff = xgb_metrics['r2'][i] - tabnet_metrics['r2'][i]
        print(f"{size_pct:>6.0f}%        {xgb_metrics['r2'][i]:.4f}       {tabnet_metrics['r2'][i]:.4f}       {diff:+.4f}")
    
    print(f"\n✓ Learning Curve 實驗完成！")
    
    return {
        'xgb_metrics': xgb_metrics,
        'tabnet_metrics': tabnet_metrics,
        'train_sizes_percent': train_sizes_percent
    }


def main(mode='remove_date'):
    """
    主執行函數
    
    Args:
        mode: 預處理模式 ('remove_date' 或 'simplified_data')
    """
    print("\n" + "="*80)
    print(f"太陽能發電量預測 - 模型訓練與評估（模式: {mode}）")
    print("="*80)
    
    # 準備數據（根據指定的模式）
    prep, train_data, test_data = prepare_data(mode=mode)
    
    # 提取訓練數據用於交叉驗證
    X_train_full, y_train_full = extract_features_and_target(train_data, prep)
    
    # ========== 交叉驗證部分 ==========
    print("\n" + "="*80)
    print("【第一部分：K-Fold 交叉驗證】")
    print("="*80)
    
    # XGBoost 交叉驗證
    xgb_cv_results = cross_validate_xgb(
        X_train=X_train_full,
        y_train=y_train_full,
        n_splits=5,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6
    )
    
    # TabNet 交叉驗證
    tabnet_cv_results = cross_validate_tabnet(
        X_full=X_train_full,
        y_full=y_train_full,
        n_splits=5,
        verbose=0
    )
    
    # ========== 最終模型訓練部分（使用全部訓練數據 + 測試集分割） ==========
    print("\n" + "="*80)
    print("【第二部分：最終模型訓練（Train-Test Split）】")
    print("="*80)
    
    # 執行 XGBoost 模型
    xgb_results = run_xgb_model(
        prep=prep,
        train_data=train_data,
        test_data=test_data,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        verbose=1
    )
    
    # 執行 TabNet 模型
    tabnet_results = run_tabnet_model(
        prep=prep,
        train_data=train_data,
        test_data=test_data,
        verbose=1
    )
    
    # 比較模型性能
    compare_models(xgb_results, tabnet_results)
    
    # ========== 交叉驗證結果對比 ==========
    print("\n" + "="*80)
    print("【交叉驗證 vs 測試集性能對比】")
    print("="*80)
    
    print("\n" + "-"*80)
    print("XGBoost 模型：", )
    print("-"*80)
    print(f"{'指標':<15} {'CV Mean':<15} {'CV Std':<15} {'Test Set'}")
    print("-"*80)
    print(f"{'R²':<15} {xgb_cv_results['r2']['mean']:<15.4f} {xgb_cv_results['r2']['std']:<15.4f} {xgb_results['metrics']['r2']:.4f}")
    print(f"{'RMSE':<15} {xgb_cv_results['rmse']['mean']:<15.2f} {xgb_cv_results['rmse']['std']:<15.2f} {xgb_results['metrics']['rmse']:.2f}")
    print(f"{'MAE':<15} {xgb_cv_results['mae']['mean']:<15.2f} {xgb_cv_results['mae']['std']:<15.2f} {xgb_results['metrics']['mae']:.2f}")
    print(f"{'MAPE':<15} {xgb_cv_results['mape']['mean']:<15.2f}% {xgb_cv_results['mape']['std']:<14.2f}% {xgb_results['metrics']['mape']:.2f}%")
    
    print("\n" + "-"*80)
    print("TabNet 模型：")
    print("-"*80)
    print(f"{'指標':<15} {'CV Mean':<15} {'CV Std':<15} {'Test Set'}")
    print("-"*80)
    print(f"{'R²':<15} {tabnet_cv_results['r2']['mean']:<15.4f} {tabnet_cv_results['r2']['std']:<15.4f} {tabnet_results['metrics']['r2']:.4f}")
    print(f"{'RMSE':<15} {tabnet_cv_results['rmse']['mean']:<15.2f} {tabnet_cv_results['rmse']['std']:<15.2f} {tabnet_results['metrics']['rmse']:.2f}")
    print(f"{'MAE':<15} {tabnet_cv_results['mae']['mean']:<15.2f} {tabnet_cv_results['mae']['std']:<15.2f} {tabnet_results['metrics']['mae']:.2f}")
    print(f"{'MAPE':<15} {tabnet_cv_results['mape']['mean']:<15.2f}% {tabnet_cv_results['mape']['std']:<14.2f}% {tabnet_results['metrics']['mape']:.2f}%")
    
    # 生成可視化圖表
    print("\n" + "="*80)
    print("【生成可視化圖表】")
    print("="*80)
    
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # XGBoost 分析圖
    print("\n生成 XGBoost 分析圖...")
    plot_regression_results(
        'XGBoost',
        xgb_results['y_test'],
        xgb_results['y_pred'],
        xgb_results['metrics'],
        result_dir
    )
    
    # TabNet 分析圖
    print("生成 TabNet 分析圖...")
    plot_regression_results(
        'TabNet',
        tabnet_results['y_test'],
        tabnet_results['y_pred'],
        tabnet_results['metrics'],
        result_dir
    )
    
    # 模型對比圖
    print("生成模型對比圖...")
    plot_model_comparison(xgb_results, tabnet_results, result_dir)
    
    # XGBoost 特徵重要性圖
    print("生成 XGBoost 特徵重要性圖...")
    plot_xgb_feature_importance(
        xgb_results['model'],
        xgb_results['X_train'].columns.tolist(),
        result_dir
    )
    
    # 交叉驗證結果可視化
    print("生成交叉驗證結果圖...")
    plot_cv_comparison(xgb_cv_results, tabnet_cv_results, result_dir)
    
    print(f"\n✓ 所有圖表已保存到: {result_dir}")
    
    print("\n" + "="*80)
    print("✓ 所有模型訓練與交叉驗證完成！")
    print("="*80)
    
    return {
        'preprocessor': prep,
        'xgb_results': xgb_results,
        'tabnet_results': tabnet_results,
        'xgb_cv_results': xgb_cv_results,
        'tabnet_cv_results': tabnet_cv_results
    }


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='太陽能發電量預測 - 模型訓練與評估'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['remove_date', 'simplified_data'],
        default='remove_date',
        help='預處理模式：\n'
             '  remove_date: 移除日期欄位，使用平均值填充缺失值（默認）\n'
             '  simplified_data: 簡化模式，移除年度/月份/平均單位發電量，合併發電比，用零值填充'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['all', 'learning_curve', 'pca'],
        default='all',
        help='實驗模式：\n'
             '  all: 執行完整的模型訓練和交叉驗證（默認）\n'
             '  learning_curve: 執行 Learning Curve 實驗，分析訓練數據量的影響\n'
             '  pca: 執行 PCA 維度縮減實驗，分析維度數量的影響'
    )
    
    args = parser.parse_args()
    
    print(f"\n【執行參數】")
    print(f"預處理模式: {args.mode}")
    print(f"實驗模式: {args.experiment}")
    
    if args.experiment == 'learning_curve':
        results = run_learning_curve(mode=args.mode)
    elif args.experiment == 'pca':
        results = run_dimensionality_reduction(mode=args.mode)
    else:
        results = main(mode=args.mode)
