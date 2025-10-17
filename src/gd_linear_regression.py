from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class GDConfig:
    learning_rate: float = 1e-2
    max_iters: int = 20000
    tol: float = 1e-7  # relative improvement tolerance
    record_every: int = 100
    # Backtracking line search to avoid divergence/overflow
    use_backtracking: bool = True
    backtrack_factor: float = 0.5
    backtrack_max_trials: int = 20
    min_step: float = 1e-12


@dataclass
class GDResult:
    weights: np.ndarray
    bias: float
    losses: List[float]


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return X @ weights + bias


def gradient_descent_fit(
    X: np.ndarray, y: np.ndarray, config: GDConfig
) -> GDResult:
    # Ensure float64 for stability
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    losses: List[float] = []

    prev_loss = None
    base_lr = config.learning_rate

    # Initial loss
    y_pred = X @ w + b
    residual = y_pred - y
    loss = float(np.mean(residual * residual))
    if config.record_every:
        losses.append(loss)
    prev_loss = loss

    for it in range(1, config.max_iters + 1):
        # gradients for MSE at current point
        grad_w = (2.0 / n_samples) * (X.T @ residual)
        grad_b = (2.0 / n_samples) * float(np.sum(residual))

        step = base_lr
        accepted = False
        for _ in range(config.backtrack_max_trials if config.use_backtracking else 1):
            w_new = w - step * grad_w
            b_new = b - step * grad_b
            y_pred_new = X @ w_new + b_new
            residual_new = y_pred_new - y
            loss_new = float(np.mean(residual_new * residual_new))

            if not np.isfinite(loss_new):
                # shrink step if overflow/NaN
                if config.use_backtracking and step > config.min_step:
                    step *= config.backtrack_factor
                    continue
                else:
                    break

            # Accept if improvement or equal (non-increase)
            if (not np.isnan(loss_new)) and (loss_new <= loss):
                w, b = w_new, b_new
                y_pred, residual, loss = y_pred_new, residual_new, loss_new
                accepted = True
                break
            else:
                if config.use_backtracking and step > config.min_step:
                    step *= config.backtrack_factor
                else:
                    break

        if not accepted:
            # Could not find a decreasing step; stop to avoid divergence
            break

        # Record losses
        if config.record_every and (it % config.record_every == 0):
            losses.append(loss)

        # relative improvement early stop
        if prev_loss > 0:
            rel_improve = (prev_loss - loss) / prev_loss
            if rel_improve >= 0 and rel_improve < config.tol:
                # Append final snapshot
                if not losses or losses[-1] != loss:
                    losses.append(loss)
                break
        prev_loss = loss

    return GDResult(weights=w, bias=b, losses=losses)


