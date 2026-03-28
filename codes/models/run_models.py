import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
    print("太陽能發電量預測 - 模型訓練與評估")
    print("="*80)
    
    # 準備數據
    prep, train_data, test_data = prepare_data()
    
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
    
    print("\n" + "="*80)
    print("✓ 所有模型訓練完成！")
    print("="*80)
    
    return {
        'preprocessor': prep,
        'xgb_results': xgb_results,
        'tabnet_results': tabnet_results
    }


if __name__ == '__main__':
    results = main()
