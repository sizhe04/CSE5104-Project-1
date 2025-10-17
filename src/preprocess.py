from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class Scaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        # avoid division by zero
        safe_std = np.where(self.std_ == 0, 1.0, self.std_)
        return (X - self.mean_) / safe_std


def fit_standard_scaler(X: pd.DataFrame) -> Scaler:
    mean = X.mean(axis=0).to_numpy()
    std = X.std(axis=0, ddof=0).to_numpy()
    return Scaler(mean_=mean, std_=std)


def standardize_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Scaler]:
    scaler = fit_standard_scaler(X_train)
    Xtr = scaler.transform(X_train.to_numpy(dtype=float))
    Xte = scaler.transform(X_test.to_numpy(dtype=float))
    return Xtr, Xte, scaler


def as_numpy(X: pd.DataFrame) -> np.ndarray:
    return X.to_numpy(dtype=float)


def log1p_transform(X: pd.DataFrame) -> pd.DataFrame:
    # Clip to avoid negative -> log domain issues if any
    X_clipped = X.copy()
    for c in X_clipped.columns:
        col = X_clipped[c]
        min_val = col.min()
        if min_val < -1.0:
            X_clipped[c] = col - (min_val + 1.0)
    return np.log1p(X_clipped)





