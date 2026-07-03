"""
seasonal_model.py
==================
Seasonal/trend forecaster for SMOOTH-archetype SKUs.

DESIGN NOTE -- read this before citing "Prophet" in your report:
The build environment has no network access and `prophet` could not be
installed (verified: ModuleNotFoundError). Rather than silently swap in a
different library and call it Prophet, this module implements the same
underlying idea Prophet uses for its seasonality component -- a Fourier
series (sum of sine/cosine harmonics) capturing weekly and annual cycles,
plus a linear trend term and a promo-flag feature -- fit with ordinary
ridge regression (`sklearn.linear_model.Ridge`, confirmed installed).

This is a standard, well-documented technique (Fourier-term regression for
seasonality) and is mathematically close to Prophet's internal seasonality
representation. It is NOT Prophet. State this plainly in the methodology
section: "Prophet was unavailable in the build environment; seasonality
was instead modeled via explicit Fourier regression, a closely related
and transparent alternative."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class FourierSeasonalForecaster:
    """
    Per-SKU model: trend + weekly harmonics + annual harmonics + promo flag.

    y_hat(t) = b0 + b1*t
               + sum_k [a_k*sin(2*pi*k*t/7)  + c_k*cos(2*pi*k*t/7)]   (weekly)
               + sum_k [d_k*sin(2*pi*k*t/365.25) + e_k*cos(2*pi*k*t/365.25)]  (annual)
               + f * promo_flag(t)

    One model is fit per SKU (summed across stores -> network-level daily
    demand for that SKU), which matches how the rest of the pipeline
    evaluates accuracy (per SKU, not per store-SKU pair).
    """

    def __init__(self, n_weekly_harmonics: int = 3, n_annual_harmonics: int = 4,
                 alpha: float = 1.0):
        self.n_weekly = n_weekly_harmonics
        self.n_annual = n_annual_harmonics
        self.alpha = alpha
        self.model: Ridge | None = None
        self.t0: pd.Timestamp | None = None

    def _features(self, dates: pd.Series, promo_flag: np.ndarray) -> np.ndarray:
        t = (dates - self.t0).dt.days.to_numpy(dtype=float)
        cols = [np.ones_like(t), t]  # intercept, linear trend
        for k in range(1, self.n_weekly + 1):
            cols.append(np.sin(2 * np.pi * k * t / 7.0))
            cols.append(np.cos(2 * np.pi * k * t / 7.0))
        for k in range(1, self.n_annual + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        cols.append(promo_flag.astype(float))
        return np.column_stack(cols)

    def fit(self, dates: pd.Series, y: np.ndarray, promo_flag: np.ndarray) -> "FourierSeasonalForecaster":
        self.t0 = dates.min()
        X = self._features(dates, promo_flag)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        return self

    def predict(self, dates: pd.Series, promo_flag: np.ndarray) -> np.ndarray:
        if self.model is None or self.t0 is None:
            raise RuntimeError("call fit() before predict()")
        X = self._features(dates, promo_flag)
        yhat = self.model.predict(X)
        return np.clip(yhat, 0.0, None)  # demand can't be negative
