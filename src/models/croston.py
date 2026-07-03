"""
croston.py
==========
Croston's method for intermittent / lumpy demand (the C-class, archetype
'intermittent' SKUs validated statistically in Phase 2, Section 4).

Standard demand forecasting (moving average, exponential smoothing, even the
Fourier model in seasonal_model.py) systematically over-forecasts intermittent
series, because long zero-runs drag a naive average toward zero while the
occasional spike is averaged away rather than anticipated. Croston's method
fixes this by decomposing the series into two SEPARATE smoothed estimates:
  - z_hat : the average non-zero demand SIZE (when it occurs)
  - p_hat : the average INTERVAL between non-zero demand occurrences
Forecast = z_hat / p_hat (expected demand per period).

Implemented directly from the published recursions (Croston, 1972; with the
standard alpha=0.1 SES smoothing constant) -- this is a short, well-defined
algorithm and does not require an external library.

EMPIRICAL FINDING (see notebook Section 3): on this project's synthetic data
(moderate intermittency -- mean occurrence_prob 0.15-0.45, lump_size_mean 4.0,
NOT the severe near-all-zero pattern Croston's advantage depends on), Croston
beats a 28-day moving average on only 6/16 intermittent SKUs in-sample, with
no sparsity threshold that predicts which 6. Croston's theoretical edge
requires more extreme intermittency than this generator produced. Rather than
force Croston onto every archetype-'intermittent' SKU regardless of fit, the
pipeline performs TRAINING-WINDOW model selection (see CrostonOrBaseline
below): both candidates are fit and compared on a held-out tail of the
TRAINING data only (never the evaluation holdout), and whichever wins is
used. This is legitimate in-sample selection, not test-set peeking.
"""

from __future__ import annotations

import numpy as np


class CrostonForecaster:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.z_hat: float | None = None  # smoothed non-zero demand size
        self.p_hat: float | None = None  # smoothed inter-demand interval

    def fit(self, y: np.ndarray) -> "CrostonForecaster":
        """y: 1D array of period demand (zeros allowed, this is the point)."""
        y = np.asarray(y, dtype=float)
        nonzero_idx = np.flatnonzero(y > 0)

        if len(nonzero_idx) == 0:
            # No demand observed at all in the training window.
            self.z_hat, self.p_hat = 0.0, 1.0
            return self

        if len(nonzero_idx) == 1:
            # Exactly one observed demand event -- not enough to estimate an
            # interval; treat that single size as the level, interval as 1.
            self.z_hat = float(y[nonzero_idx[0]])
            self.p_hat = 1.0
            return self

        sizes = y[nonzero_idx]
        intervals = np.diff(nonzero_idx)  # gaps between successive non-zero periods
        intervals = np.concatenate([[nonzero_idx[0] + 1], intervals])  # include first gap

        z = sizes[0]
        p = float(intervals[0])
        for i in range(1, len(sizes)):
            z = self.alpha * sizes[i] + (1 - self.alpha) * z
            p = self.alpha * intervals[i] + (1 - self.alpha) * p

        self.z_hat, self.p_hat = float(z), float(p)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        """Croston forecasts a constant expected-demand-per-period rate."""
        if self.z_hat is None or self.p_hat is None:
            raise RuntimeError("call fit() before predict()")
        rate = self.z_hat / max(self.p_hat, 1e-6)
        return np.full(n_periods, rate)


class CrostonOrBaseline:
    """
    Per-SKU model selector for intermittent-archetype SKUs.

    Fits BOTH Croston and a moving-average baseline on a validation split of
    the training data only (last `val_days` of train vs. the rest), compares
    RMSE on that internal split, and keeps whichever wins. Then re-fits the
    winning model type on the FULL training window for the final forecast.
    The evaluation holdout is never touched by this selection step.
    """

    def __init__(self, val_days: int = 60, ma_window: int = 28, alpha: float = 0.1):
        self.val_days = val_days
        self.ma_window = ma_window
        self.alpha = alpha
        self.chosen: str | None = None
        self._model = None

    def fit(self, y: np.ndarray) -> "CrostonOrBaseline":
        from src.evaluation.baseline_and_metrics import MovingAverageBaseline

        y = np.asarray(y, dtype=float)
        if len(y) <= self.val_days + 10:
            # Not enough history to hold out a validation split safely; default to Croston.
            self.chosen = "croston"
            self._model = CrostonForecaster(self.alpha).fit(y)
            return self

        inner_train, inner_val = y[: -self.val_days], y[-self.val_days:]

        cr = CrostonForecaster(self.alpha).fit(inner_train)
        cr_pred = cr.predict(len(inner_val))
        cr_rmse = float(np.sqrt(np.mean((inner_val - cr_pred) ** 2)))

        ma = MovingAverageBaseline(self.ma_window).fit(inner_train)
        ma_pred = ma.predict(len(inner_val))
        ma_rmse = float(np.sqrt(np.mean((inner_val - ma_pred) ** 2)))

        self.chosen = "croston" if cr_rmse <= ma_rmse else "moving_average"
        if self.chosen == "croston":
            self._model = CrostonForecaster(self.alpha).fit(y)
        else:
            self._model = MovingAverageBaseline(self.ma_window).fit(y)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        return self._model.predict(n_periods)
