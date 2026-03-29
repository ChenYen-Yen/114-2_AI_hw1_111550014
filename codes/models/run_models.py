import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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


def prepare_data():
    """準備訓練數據"""
    print("【載入資料】")
    
    # 初始化預處理器
    prep = preprocessor()
    
    # 讀取原始數據
    df = prep.load_data()
    
    # 探索數據
    prep.explore_data()
    
    # 清理數據（刪除日期欄位，處理缺失值）
    prep.clean_data(
        drop_columns=None,
        handle_missing='smart',  # 特別處理 T 值
        drop_date_columns=True   # 刪除日期欄位
    )
    
    # 不添加新特徵，保持原始欄位
    print("\n【特徵工程】")
    print("✓ 跳過欄位添加（保持精簡，先用原始資料測試）")
    
    # 準備訓練集和測試集
    train_data, test_data = prep.prepare_train_test(by_time=True)
    
    return prep, train_data, test_data


def extract_features_and_target(data, prep, 
                                 target_col='發電量(度)/Power Generation(kWh)'):
    """
    從數據框中提取特徵和目標變量
    
    Args:
        data: 輸入 DataFrame
        prep: SolarDataPreprocessor 實例
        target_col: 目標欄位名稱
    
    Returns:
        features: 特徵 DataFrame（僅數值列）
        targets: 目標 Series
    """
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
    
    # 建立 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - 迴歸性能可視化', fontsize=16, fontweight='bold')
    
    # 1. 實際 vs 預測散點圖
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美預測')
    ax1.set_xlabel('實際值 (kWh)', fontsize=11)
    ax1.set_ylabel('預測值 (kWh)', fontsize=11)
    ax1.set_title('實際 vs 預測', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 殘差分佈直方圖
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='零誤差')
    ax2.set_xlabel('殘差 (kWh)', fontsize=11)
    ax2.set_ylabel('頻率', fontsize=11)
    ax2.set_title('殘差分佈', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 預測值 vs 殘差圖
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('預測值 (kWh)', fontsize=11)
    ax3.set_ylabel('殘差 (kWh)', fontsize=11)
    ax3.set_title('預測值 vs 殘差', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能指標文字框
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    【模型評估指標】
    
    R² Score:        {metrics['r2']:.4f}
    
    RMSE:            {metrics['rmse']:,.2f} kWh
    MAE:             {metrics['mae']:,.2f} kWh
    MSE:             {metrics['mse']:.2f}
    
    MAPE:            {metrics['mape']:.2f}%
    
    【數據統計】
    
    測試集樣本數:     {len(y_test)}
    實際值範圍:       {y_test.min():,.0f} ~ {y_test.max():,.0f} kWh
    實際值平均:       {y_test.mean():,.0f} kWh
    實際值標準差:     {y_test.std():,.0f} kWh
    
    預測值平均:       {y_pred.mean():,.0f} kWh
    預測值標準差:     {y_pred.std():,.0f} kWh
    
    【殘差統計】
    
    殘差平均:         {residuals.mean():,.0f} kWh
    殘差標準差:       {residuals.std():,.0f} kWh
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
    
    # 準備數據
    models = ['XGBoost', 'TabNet']
    r2_scores = [xgb_results['metrics']['r2'], tabnet_results['metrics']['r2']]
    mae_scores = [xgb_results['metrics']['mae'], tabnet_results['metrics']['mae']]
    rmse_scores = [xgb_results['metrics']['rmse'], tabnet_results['metrics']['rmse']]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('XGBoost vs TabNet 模型性能對比', fontsize=14, fontweight='bold')
    
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
    ax2.set_ylabel('MAE (kWh)', fontsize=11)
    ax2.set_title('平均絕對誤差（越低越好）', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars2, mae_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5000, f'{val:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # RMSE 對比
    ax3 = axes[2]
    bars3 = ax3.bar(models, rmse_scores, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('RMSE (kWh)', fontsize=11)
    ax3.set_title('均方根誤差（越低越好）', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars3, rmse_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 10000, f'{val:,.0f}',
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


def main():
    """主執行函數"""
    print("\n" + "="*80)
    print("太陽能發電量預測 - 模型訓練與評估（包含交叉驗證）")
    print("="*80)
    
    # 準備數據
    prep, train_data, test_data = prepare_data()
    
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
    results = main()
