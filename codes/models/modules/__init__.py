"""Models modules for solar power generation prediction."""

from .preprocess import SolarDataPreprocessor
from .XGB import (
    train_XGBRegressor,
    predict_with_XGB,
    save_XGB,
    load_XGB
)
from .Tabnet import (
    train_TabNetRegressor,
    predict_with_TabNet,
    save_TabNet,
    load_TabNet
)

__all__ = [
    'SolarDataPreprocessor',
    'train_XGBRegressor',
    'predict_with_XGB',
    'save_XGB',
    'load_XGB',
    'train_TabNetRegressor',
    'predict_with_TabNet',
    'save_TabNet',
    'load_TabNet',
]
