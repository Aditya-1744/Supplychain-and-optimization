"""
gbm_model.py
============
Gradient boosting forecaster, used for SKUs where promo/price-elasticity
effects are the dominant signal worth a flexible, non-linear model. This is
named directly in the charter and project guide ("Gradient Boosting
(promo/price elasticity)") and uses `sklearn.ensemble.GradientBoostingRegressor`
-- no substitution needed here, unlike the Prophet case in seasonal_model.py.

Features: calendar (day-of-week, month), promo flag, and lag features
(demand 7 and 14 days prior) capturing short-term autocorrelation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class GBMForecaster:
    def __init__(self, n_estimators: int = 150, max_depth: int = 3,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        )
        self._feature_cols = ["dow", "month", "on_promo", "lag7", "lag14"]

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        """df must have columns: date, units, on_promo, sorted by date."""
        out = df.copy()
        out["dow"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        out["on_promo"] = out["on_promo"].astype(int)
        out["lag7"] = out["units"].shift(7)
        out["lag14"] = out["units"].shift(14)
        return out

    def fit(self, df: pd.DataFrame) -> "GBMForecaster":
        feat = self._build_features(df).dropna(subset=["lag7", "lag14"])
        X = feat[self._feature_cols].to_numpy()
        y = feat["units"].to_numpy()
        self.model.fit(X, y)
        # Remember the tail of training data so predict() can build lags
        # that extend into the forecast horizon.
        self._history_tail = df[["date", "units", "on_promo"]].tail(14).copy()
        return self

    def predict(self, future_dates: pd.Series, future_promo: np.ndarray) -> np.ndarray:
        """
        Walk forward day by day so each prediction's lag features can use
        previously PREDICTED values once we run past the training history
        (a standard recursive-forecast pattern for lag-feature models).
        """
        history = self._history_tail.copy()
        preds = []
        for dt, promo in zip(future_dates, future_promo):
            row = pd.DataFrame({"date": [dt], "units": [np.nan], "on_promo": [promo]})
            extended = pd.concat([history, row], ignore_index=True)
            feat = self._build_features(extended).iloc[[-1]]
            X = feat[self._feature_cols].to_numpy()
            yhat = max(0.0, float(self.model.predict(X)[0]))
            preds.append(yhat)
            history = pd.concat(
                [history, pd.DataFrame({"date": [dt], "units": [yhat], "on_promo": [promo]})],
                ignore_index=True,
            ).tail(14)
        return np.array(preds)
