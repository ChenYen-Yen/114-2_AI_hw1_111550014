import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import os

def train_XGBRegressor(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray, y_valid: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0,
    random_state: int = 42,
    verbose: int = 1,
    use_grid_search: bool = False,
    use_random_search: bool = False,
    n_jobs: int = -1,
    **kwargs
) -> xgb.XGBRegressor:
    """
    訓練 XGBoost Regressor 模型。
    
    參數說明：
    X_train, y_train: 訓練資料與標籤（numpy array）
    X_valid, y_valid: 驗證資料與標籤（numpy array）
    n_estimators: 樹的數量
    learning_rate: 學習率（eta）
    max_depth: 樹的最大深度
    min_child_weight: 子樹最小權重
    subsample: 訓練時使用的樣本比例（默認 0.8）
    colsample_bytree: 每棵樹使用的特徵比例
    gamma: 最小損失減少量
    random_state: 隨機種子
    verbose: 是否印出訓練訊息
    use_grid_search: 是否使用GridSearchCV進行超參數調優
    use_random_search: 是否使用RandomizedSearchCV進行隨機搜索
    n_jobs: 平行運算核心數 (-1 為使用全部核心)
    """
    if verbose > 0:
        print(f"Training XGBoost Regressor...")
        if use_grid_search:
            print("Using GridSearchCV for hyperparameter tuning...")
        elif use_random_search:
            print("Using RandomizedSearchCV for hyperparameter tuning...")
        else:
            print(f"Params: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
    
    # 確保輸入是 numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
        
    # 處理 y 的形狀，XGBoost 期望 (n_samples,)
    y_train = y_train.ravel()
    
    if use_grid_search or use_random_search:
        # 定義超參數網格
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.5],
        }
        
        try:
            device = 'gpu'
            xgb.XGBRegressor(device=device, random_state=0)  # 測試 GPU 是否可用
        except:
            device = 'cpu'
        
        base_model = xgb.XGBRegressor(
            random_state=random_state,
            verbosity=1 if verbose > 0 else 0,
            tree_method='hist',
            device=device
        )
        
        if use_grid_search:
            search = GridSearchCV(
                base_model, param_grid, cv=5, n_jobs=n_jobs,
                scoring='neg_mean_squared_error', verbose=1 if verbose > 0 else 0
            )
        else:  # use_random_search
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=20, cv=5, n_jobs=n_jobs,
                scoring='neg_mean_squared_error', verbose=1 if verbose > 0 else 0,
                random_state=random_state
            )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        if verbose > 0:
            print(f"\nBest parameters found: {search.best_params_}")
            print(f"Best CV score (neg_mse): {search.best_score_:.4f}")
    else:
        try:
            device = 'gpu'
            xgb.XGBRegressor(device=device, random_state=0)  # 測試 GPU 是否可用
        except:
            device = 'cpu'
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=random_state,
            verbosity=1 if verbose > 0 else 0,
            tree_method='hist',
            device=device
        )
        
        model.fit(X_train, y_train)
    
    if verbose > 0:
        train_score = model.score(X_train, y_train)
        # 如果有提供驗證集，計算驗證分數
        if X_valid is not None and y_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid.to_numpy()
            if isinstance(y_valid, pd.Series):
                y_valid = y_valid.to_numpy()
            y_valid = y_valid.ravel()
            valid_score = model.score(X_valid, y_valid)
            print(f"XGBoost Training Completed.")
            print(f"Train R2: {train_score:.4f}")
            print(f"Valid R2: {valid_score:.4f}")
        else:
            print(f"XGBoost Training Completed. Train R2: {train_score:.4f}")
    
    return model

def predict_with_XGB(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """
    使用訓練好的 XGBoost 模型進行預測
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        
    return model.predict(X)

def save_XGB(model: xgb.XGBRegressor, save_path: str):
    """
    儲存 XGBoost 模型 (使用 pickle)
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 確保副檔名
    if not save_path.endswith('.pkl'):
        save_path += '.pkl'
        
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

def load_XGB(load_path: str) -> xgb.XGBRegressor:
    """
    載入 XGBoost 模型
    """
    if not load_path.endswith('.pkl'):
        load_path += '.pkl'
        
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
        
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model
