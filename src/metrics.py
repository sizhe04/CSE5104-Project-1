import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Per assignment: 1 - MSE / Variance(observed)
    var = float(np.var(y_true))
    if var == 0:
        return 0.0
    return 1.0 - mse(y_true, y_pred) / var





