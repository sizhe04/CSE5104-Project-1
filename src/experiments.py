import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_io import load_data, train_test_split_fixed, split_features_target
from .preprocess import standardize_train_test, as_numpy, log1p_transform
from .gd_linear_regression import GDConfig, gradient_descent_fit, predict
from .metrics import mse, variance_explained
from .regression_analysis import ols_pvalues
import statsmodels.api as sm


plt.switch_backend("Agg")


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "loss_curves"), exist_ok=True)


def run_univariate_experiments():
    ensure_dirs()
    df = load_data()
    train_df, test_df = train_test_split_fixed(df)
    Xtr_df, ytr = split_features_target(train_df)
    Xte_df, yte = split_features_target(test_df)

    features = list(Xtr_df.columns)
    results: List[Dict] = []

    # Two settings: standardized (A) and raw (B)
    settings = ["standardized", "raw"]
    for setting in settings:
        if setting == "standardized":
            Xtr, Xte, _ = standardize_train_test(Xtr_df, Xte_df)
        else:
            Xtr = as_numpy(Xtr_df)
            Xte = as_numpy(Xte_df)

        for j, feat in enumerate(features):
            Xtr_j = Xtr[:, [j]]
            Xte_j = Xte[:, [j]]

            best_res = None
            best_cfg = None
            for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
                cfg = GDConfig(learning_rate=lr, max_iters=20000, tol=1e-7, record_every=200)
                res = gradient_descent_fit(Xtr_j, ytr.to_numpy(dtype=float), cfg)
                ytr_pred = predict(Xtr_j, res.weights, res.bias)
                loss = mse(ytr.to_numpy(dtype=float), ytr_pred)
                if best_res is None or loss < mse(ytr.to_numpy(dtype=float), predict(Xtr_j, best_res.weights, best_res.bias)):
                    best_res = res
                    best_cfg = cfg

            # Evaluate best on train/test
            ytr_pred = predict(Xtr_j, best_res.weights, best_res.bias)
            yte_pred = predict(Xte_j, best_res.weights, best_res.bias)
            results.append(
                {
                    "setting": setting,
                    "feature": feat,
                    "lr": best_cfg.learning_rate,
                    "m": float(best_res.weights[0]),
                    "b": float(best_res.bias),
                    "train_mse": mse(ytr.to_numpy(dtype=float), ytr_pred),
                    "train_var_explained": variance_explained(ytr.to_numpy(dtype=float), ytr_pred),
                    "test_mse": mse(yte.to_numpy(dtype=float), yte_pred),
                    "test_var_explained": variance_explained(yte.to_numpy(dtype=float), yte_pred),
                }
            )

    pd.DataFrame(results).to_csv(os.path.join("outputs", "univariate_metrics.csv"), index=False)


def plot_losses(losses: List[float], title: str, path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(losses)), losses, label="loss (MSE)")
    plt.xlabel("record step")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_multivariate_experiments():
    ensure_dirs()
    df = load_data()
    train_df, test_df = train_test_split_fixed(df)
    Xtr_df, ytr = split_features_target(train_df)
    Xte_df, yte = split_features_target(test_df)

    results: List[Dict] = []
    feature_names = list(Xtr_df.columns)

    for setting in ["standardized", "raw"]:
        if setting == "standardized":
            Xtr, Xte, _ = standardize_train_test(Xtr_df, Xte_df)
        else:
            Xtr = as_numpy(Xtr_df)
            Xte = as_numpy(Xte_df)

        best_res = None
        best_cfg = None
        for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
            cfg = GDConfig(learning_rate=lr, max_iters=20000, tol=1e-7, record_every=100)
            res = gradient_descent_fit(Xtr, ytr.to_numpy(dtype=float), cfg)
            ytr_pred = predict(Xtr, res.weights, res.bias)
            loss = mse(ytr.to_numpy(dtype=float), ytr_pred)
            if best_res is None or loss < mse(ytr.to_numpy(dtype=float), predict(Xtr, best_res.weights, best_res.bias)):
                best_res = res
                best_cfg = cfg

        # Evaluate best on train/test
        ytr_pred = predict(Xtr, best_res.weights, best_res.bias)
        yte_pred = predict(Xte, best_res.weights, best_res.bias)
        results.append(
            {
                "setting": setting,
                "lr": best_cfg.learning_rate,
                "train_mse": mse(ytr.to_numpy(dtype=float), ytr_pred),
                "train_var_explained": variance_explained(ytr.to_numpy(dtype=float), ytr_pred),
                "test_mse": mse(yte.to_numpy(dtype=float), yte_pred),
                "test_var_explained": variance_explained(yte.to_numpy(dtype=float), yte_pred),
            }
        )

        # Save loss curve
        plot_losses(
            best_res.losses,
            title=f"Multivariate {setting} (lr={best_cfg.learning_rate})",
            path=os.path.join("outputs", "loss_curves", f"multivariate_{setting}.png"),
        )

        # Save weights and bias
        weights_rows = []
        for fname, w in zip(feature_names, best_res.weights.tolist()):
            weights_rows.append({"feature": fname, "weight": float(w)})
        weights_rows.append({"feature": "__bias__", "weight": float(best_res.bias)})
        pd.DataFrame(weights_rows).to_csv(
            os.path.join("outputs", f"multivariate_weights_{setting}.csv"), index=False
        )

    pd.DataFrame(results).to_csv(os.path.join("outputs", "multivariate_metrics.csv"), index=False)


def run_regression_analysis():
    ensure_dirs()
    df = load_data()
    train_df, test_df = train_test_split_fixed(df)
    Xtr_df, ytr = split_features_target(train_df)
    Xte_df, yte = split_features_target(test_df)

    # Set A: standardized
    X_std, _, _ = standardize_train_test(Xtr_df, Xtr_df)
    pvals_std = ols_pvalues(pd.DataFrame(X_std, columns=Xtr_df.columns), ytr)
    pvals_std.to_csv(os.path.join("outputs", "pvalues_setA.csv"), index=False)

    # Set B: raw
    pvals_raw = ols_pvalues(Xtr_df, ytr)
    pvals_raw.to_csv(os.path.join("outputs", "pvalues_setB.csv"), index=False)

    # OLS weights/metrics for RAW (Part B requirement)
    Xtr_const = sm.add_constant(Xtr_df, has_constant='add')
    Xte_const = sm.add_constant(Xte_df, has_constant='add')
    model_raw = sm.OLS(ytr, Xtr_const)
    res_raw = model_raw.fit()
    params = res_raw.params  # includes const
    rows = []
    for name in params.index:
        if name == 'const':
            rows.append({"feature": "__bias__", "weight": float(params[name])})
        else:
            rows.append({"feature": name, "weight": float(params[name])})
    pd.DataFrame(rows).to_csv(os.path.join("outputs", "ols_weights_raw.csv"), index=False)

    # metrics
    ytr_pred = res_raw.predict(Xtr_const).to_numpy()
    yte_pred = res_raw.predict(Xte_const).to_numpy()
    metrics_raw = {
        "train_mse": mse(ytr.to_numpy(dtype=float), ytr_pred),
        "train_var_explained": variance_explained(ytr.to_numpy(dtype=float), ytr_pred),
        "test_mse": mse(yte.to_numpy(dtype=float), yte_pred),
        "test_var_explained": variance_explained(yte.to_numpy(dtype=float), yte_pred),
    }
    pd.DataFrame([metrics_raw]).to_csv(os.path.join("outputs", "ols_metrics_raw.csv"), index=False)

    # Set C: log1p
    X_log = log1p_transform(Xtr_df)
    pvals_log = ols_pvalues(X_log, ytr)
    pvals_log.to_csv(os.path.join("outputs", "pvalues_setC.csv"), index=False)


