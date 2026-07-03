"""
Unit tests for the Phase 3 routed forecasting pipeline.
Run with: pytest tests/unit/test_forecasting.py -v
"""
import numpy as np
import pandas as pd

try:
    import pytest  # noqa: F401  (not used directly in test bodies; kept for `pytest -v` compatibility)
except ImportError:
    pass  # tests have no pytest-specific features, so this file also runs standalone via plain exec

from src.models.croston import CrostonForecaster, CrostonOrBaseline
from src.models.gbm_model import GBMForecaster
from src.models.router import route_skus
from src.models.seasonal_model import FourierSeasonalForecaster
from src.evaluation.baseline_and_metrics import MovingAverageBaseline, evaluate, mape


def _synthetic_smooth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    promo = rng.random(n) < 0.08
    y = 20 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 7) + np.where(promo, 15, 0)
    y = np.clip(y + rng.normal(0, 1.5, n), 0, None)
    return dates, y, promo


def _synthetic_intermittent(n=400, seed=0, occ_p=0.2, size_mean=4.0):
    rng = np.random.default_rng(seed)
    occurs = rng.random(n) < occ_p
    sizes = rng.poisson(size_mean, n)
    return np.where(occurs, sizes, 0).astype(float)


# --------------------------------------------------------------------- #
# FourierSeasonalForecaster
# --------------------------------------------------------------------- #
def test_fourier_fit_predict_shapes():
    dates, y, promo = _synthetic_smooth()
    train_d, train_y, train_p = dates[:340], y[:340], promo[:340]
    test_d, test_p = dates[340:], promo[340:]

    m = FourierSeasonalForecaster().fit(pd.Series(train_d), train_y, train_p)
    preds = m.predict(pd.Series(test_d), test_p)
    assert len(preds) == len(test_d)
    assert (preds >= 0).all()


def test_fourier_beats_naive_on_seasonal_data():
    dates, y, promo = _synthetic_smooth()
    train_d, train_y, train_p = dates[:340], y[:340], promo[:340]
    test_d, test_y, test_p = dates[340:], y[340:], promo[340:]

    m = FourierSeasonalForecaster().fit(pd.Series(train_d), train_y, train_p)
    preds = m.predict(pd.Series(test_d), test_p)

    base = MovingAverageBaseline(28).fit(train_y)
    base_preds = base.predict(len(test_y))

    fourier_rmse = evaluate(test_y, preds)["RMSE"]
    base_rmse = evaluate(test_y, base_preds)["RMSE"]
    assert fourier_rmse < base_rmse, "Fourier model should beat a flat baseline on clearly seasonal data"


# --------------------------------------------------------------------- #
# Croston + CrostonOrBaseline
# --------------------------------------------------------------------- #
def test_croston_basic_recursion():
    y = np.array([0, 0, 3, 0, 0, 0, 5, 0, 4, 0])
    m = CrostonForecaster().fit(y)
    assert m.z_hat is not None and m.p_hat is not None
    preds = m.predict(5)
    assert len(preds) == 5
    assert (preds >= 0).all()
    assert np.allclose(preds, preds[0])  # Croston forecasts a flat rate


def test_croston_handles_all_zero_series():
    y = np.zeros(50)
    m = CrostonForecaster().fit(y)
    preds = m.predict(10)
    assert (preds == 0).all()


def test_croston_handles_single_nonzero():
    y = np.zeros(50)
    y[10] = 7
    m = CrostonForecaster().fit(y)
    preds = m.predict(5)
    assert preds[0] > 0  # should not crash, should produce a positive rate


def test_croston_or_baseline_selects_without_touching_holdout():
    """The selector must only use the training array passed to fit()."""
    y_train = _synthetic_intermittent(n=300, occ_p=0.2)
    m = CrostonOrBaseline(val_days=60).fit(y_train)
    assert m.chosen in ("croston", "moving_average")
    preds = m.predict(90)
    assert len(preds) == 90
    assert (preds >= 0).all()


def test_croston_or_baseline_short_history_defaults_safely():
    y_train = np.array([0, 2, 0, 0, 3])  # far shorter than val_days
    m = CrostonOrBaseline(val_days=60).fit(y_train)
    assert m.chosen == "croston"
    preds = m.predict(5)
    assert len(preds) == 5


# --------------------------------------------------------------------- #
# GBM
# --------------------------------------------------------------------- #
def test_gbm_fit_predict_shapes_and_nonneg():
    rng = np.random.default_rng(1)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    promo = rng.random(n) < 0.1
    units = np.clip(10 + rng.normal(0, 3, n) + np.where(promo, 8, 0), 0, None)
    df = pd.DataFrame({"date": dates, "units": units, "on_promo": promo})

    train, test = df.iloc[:160], df.iloc[160:]
    m = GBMForecaster().fit(train[["date", "units", "on_promo"]])
    preds = m.predict(test["date"], test["on_promo"].to_numpy())
    assert len(preds) == len(test)
    assert (preds >= 0).all()
    assert not np.isnan(preds).any()


# --------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------- #
def test_router_assigns_known_categories():
    df = pd.DataFrame({
        "sku_id": ["S1", "S2", "S3"],
        "abc_class": ["A", "B", "C"],
        "archetype": ["smooth", "intermittent", "smooth"],
        "xyz_class": ["X", "Z", "Z"],  # S3: smooth label but Z class -> disagreement -> gbm
        "demand_cv": [0.3, 1.4, 1.1],
    })
    routing, disagreements = route_skus(df)
    assert routing.set_index("sku_id").loc["S1", "model"] == "fourier_seasonal"
    assert routing.set_index("sku_id").loc["S2", "model"] == "croston"
    assert routing.set_index("sku_id").loc["S3", "model"] == "gbm"
    assert len(disagreements) == 1
    assert disagreements.iloc[0]["sku_id"] == "S3"


def test_router_no_disagreement_case():
    df = pd.DataFrame({
        "sku_id": ["S1"], "abc_class": ["A"],
        "archetype": ["smooth"], "xyz_class": ["X"], "demand_cv": [0.2],
    })
    _, disagreements = route_skus(df)
    assert len(disagreements) == 0


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def test_mape_masks_zero_actuals():
    y_true = np.array([0, 0, 10, 20])
    y_pred = np.array([1, 1, 11, 18])
    # Only the two nonzero actuals should count.
    result = mape(y_true, y_pred)
    expected = np.mean([abs(11 - 10) / 10, abs(18 - 20) / 20]) * 100
    assert np.isclose(result, expected)


def test_evaluate_returns_expected_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    out = evaluate(y_true, y_pred)
    assert set(out.keys()) == {"MAE", "RMSE", "MAPE", "n_zero_actual_days", "n_periods"}
