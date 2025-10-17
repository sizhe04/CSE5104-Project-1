import os
from typing import Tuple
import pandas as pd
import numpy as np
import requests


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/Concrete/Concrete_Data.xls"
)
LOCAL_XLS = os.path.join("data", "Concrete_Data.xls")
LOCAL_CSV = os.path.join("data", "Concrete_Data.csv")


def ensure_data_files() -> str:
    os.makedirs("data", exist_ok=True)
    # Prefer CSV if present; otherwise try XLS from UCI (then save CSV for consistency)
    if os.path.exists(LOCAL_CSV):
        return LOCAL_CSV
    if not os.path.exists(LOCAL_XLS):
        try:
            resp = requests.get(UCI_URL, timeout=30)
            resp.raise_for_status()
            with open(LOCAL_XLS, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            raise RuntimeError(
                f"无法下载数据，请手动将数据放到 data/ 目录。原始错误: {e}"
            )
    # Convert to CSV for faster reads
    try:
        df = pd.read_excel(LOCAL_XLS)
    except Exception as e:
        raise RuntimeError(f"读取 Excel 数据失败，请确保文件完整。错误: {e}")
    df.to_csv(LOCAL_CSV, index=False)
    return LOCAL_CSV


def load_data() -> pd.DataFrame:
    path = ensure_data_files()
    df = pd.read_csv(path)
    # Standardize column names to known UCI titles
    # UCI columns: Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate,
    # Fine Aggregate, Age, Concrete compressive strength
    # Some copies may differ slightly in spacing/case; normalize them.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def train_test_split_fixed(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.shape[0] < 1030:
        # Still proceed but warn in return; caller can assert if needed
        pass
    # Assuming row 0 is header; pandas read_csv already treated header
    # Use 0-based indexing for slicing: rows 501-630 inclusive => iloc[501:631]
    test_df = df.iloc[501:631].copy()
    train_df = pd.concat([df.iloc[:501], df.iloc[631:]], axis=0).copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Target exact match or fallback by suffix
    target_candidates = [
        "Concrete compressive strength",
        "Concrete_Compressive_Strength",
        "Strength",
    ]
    target_col = None
    for c in df.columns:
        if c in target_candidates:
            target_col = c
            break
    if target_col is None:
        # Heuristic: last column is target (per UCI file)
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y



