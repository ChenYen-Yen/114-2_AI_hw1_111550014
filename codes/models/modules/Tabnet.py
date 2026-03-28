from typing import Any, Optional, List
import numpy as np
import pandas as pd

try:
	from pytorch_tabnet.tab_model import TabNetRegressor
except Exception as exc:
	raise ImportError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet") from exc

def detect_categorical_columns(df: pd.DataFrame, max_unique_for_cat: int = 20) -> List[str]:
	"""
	自動偵測可能為類別的欄位。
	object/category 欄位視為類別；數值欄位若 unique 值少於 max_unique_for_cat 也視為類別。
	"""
	cats = list(df.select_dtypes(include=["object", "category"]).columns)
	for col in df.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns:
		if df[col].nunique(dropna=True) <= max_unique_for_cat:
			if col not in cats:
				cats.append(col)
	return cats

def get_feature_names(data, target_col=None, feature_names=None):
    """
    取得特徵欄位名稱列表。
    支援：
    - DataFrame + target_col：回傳不含目標欄位的欄位名
    - numpy array + feature_names：直接回傳 feature_names
    - DataFrame（無 target_col）：回傳全部欄位名
    """
    if feature_names is not None:
        return feature_names
    if isinstance(data, pd.DataFrame):
        if target_col is not None:
            if target_col not in data.columns:
                raise ValueError(f"target_col '{target_col}' not found in df columns")
            return [col for col in data.columns if col != target_col]
        else:
            return list(data.columns)
    if isinstance(data, np.ndarray):
        if feature_names is not None:
            return feature_names
        else:
            return [f"feature_{i}" for i in range(data.shape[1])]
    raise ValueError("get_feature_names: 輸入型態不支援")

def train_TabNetRegressor(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray, y_valid: np.ndarray,
    cat_idxs: Optional[list] = None, cat_dims: Optional[list] = None,
    feature_names: Optional[list] = None,
    cat_emb_dim: int = 1,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1024,
    virtual_batch_size: int = 128,
    num_workers: int = 0,
    verbose: int = 1,
    device_name: str = "auto",
    n_steps: int = 3,
    gamma: float = 1.5,
    lambda_sparse: float = 1e-3,
    **kwargs
) -> Any:
    """
    訓練 TabNetRegressor 並回傳模型。
    cat_idxs/cat_dims 可手動指定，若未指定則根據 feature_names 自動偵測。
    參數說明：
    X_train, y_train: 訓練資料與標籤（numpy array）
    X_valid, y_valid: 驗證資料與標籤（numpy array）
    cat_idxs: 類別特徵的欄位索引列表
    cat_dims: 類別特徵的維度列表
    feature_names: 欄位名稱（若要自動偵測 cat_idxs/cat_dims 必須提供）
    cat_emb_dim: 類別特徵嵌入維度（預設 1）
    max_epochs: 最大訓練輪數（預設 100）
    patience: 早停耐心值（預設 10）
    batch_size: 訓練批次大小（預設 1024）
    virtual_batch_size: 虛擬批次大小（預設 128）
    num_workers: 資料載入的工作數量（預設 0）
    verbose: 訓練過程的詳細程度（預設 1）
    device_name: 裝置名稱 (auto/mps/cuda/cpu，預設 auto 會自動偵測 MPS > CUDA > CPU)
    n_steps: 決策步驟數（預設 3，增加複雜度）
    gamma: 特徵重用鬆弛參數（預設 1.5）
    lambda_sparse: 稀疏正則化係數（預設 1e-3）
    **kwargs: 傳遞給 TabNetRegressor 的其他參數
    回傳：訓練好的 TabNetRegressor 模型
    """
    # Auto-detect device if requested
    if device_name == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device_name = "cuda"
            elif torch.backends.mps.is_available():
                # MPS doesn't support torch.sort used in TabNet's sparsemax, force CPU
                device_name = "cpu" # can set PYTORCH_ENABLE_MPS_FALLBACK=1 but will be slower
                # can follow https://github.com/pytorch/pytorch/issues/77764
            else:
                device_name = "cpu"
        except Exception:
            device_name = "cpu"
    
    if cat_idxs is None or cat_dims is None:
        # 自動取得 feature_names
        cols = get_feature_names(X_train, feature_names=feature_names)
        X_df = pd.DataFrame(X_train, columns=cols)
        cat_cols = detect_categorical_columns(X_df)
        cat_idxs = []
        cat_dims = []
        for i, col in enumerate(X_df.columns):
            if col in cat_cols:
                cat_idxs.append(i)
                cat_dims.append(int(X_df[col].nunique()))
    
    reg = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        verbose=verbose,
        device_name=device_name,
        **kwargs
    )
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=num_workers,
    )
    return reg

def predict_with_TabNet(model: TabNetRegressor, X) -> Any:
	"""
	用訓練好的 TabNetRegressor 進行預測。
	"""
	return model.predict(X)

def save_TabNet(model: TabNetRegressor, path: str):
	"""
	儲存 TabNetRegressor 模型到指定路徑。
	"""
	model.save_model(path)

def load_TabNet(path: str) -> TabNetRegressor:
	"""
	從指定路徑載入 TabNetRegressor 模型。
	"""
	model = TabNetRegressor()
	model.load_model(path)
	return model

if __name__ == "__main__": 
    pass