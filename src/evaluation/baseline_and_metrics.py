"""
baseline_and_metrics.py
========================
The naive baseline every routed model must beat (Charter Objective O3:
"forecast accuracy ... MAPE/RMSE reduction vs. baseline"), plus the shared
evaluation metrics used across all model types so comparisons are apples-
to-apples.
"""

from __future__ import annotations

import numpy as np


class MovingAverageBaseline:
    """Forecasts a flat value equal to the trailing N-day mean of training data."""

    def __init__(self, window: int = 28):
        self.window = window
        self.level: float | None = None

    def fit(self, y: np.ndarray) -> "MovingAverageBaseline":
        tail = y[-self.window:] if len(y) >= self.window else y
        self.level = float(np.mean(tail))
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.level is None:
            raise RuntimeError("call fit() before predict()")
        return np.full(n_periods, self.level)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Standard MAPE, undefined when y_true == 0 (which happens often for
    intermittent SKUs). We mask zero-actual periods out of THIS metric and
    rely on MAE/RMSE (which handle zeros fine) as the primary comparison
    for intermittent SKUs -- documented explicitly, not silently averaged
    over a near-infinite term.
    """
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "n_zero_actual_days": int((y_true == 0).sum()),
        "n_periods": len(y_true),
    }
