"""
繪圖模組：處理所有可視化圖表的生成
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    """繪製 XGBoost 特徵重要性圖"""
    # 設定中文字體
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
