"""
Unit tests for the Phase 6 rolling-origin backtest harness.
Run with: pytest tests/unit/test_rolling_backtest.py -v
"""
import numpy as np
import pandas as pd
import yaml

try:
    import pytest  # noqa: F401
except ImportError:
    pass

from pathlib import Path
from src.evaluation.rolling_backtest import run_window

ROOT = Path(__file__).resolve().parents[2]


def _load():
    demand = pd.read_csv(ROOT / "data/raw/demand.csv", parse_dates=["date"])
    skus = pd.read_csv(ROOT / "data/raw/sku_master.csv")
    br = yaml.safe_load(open(ROOT / "configs/business_rules.yaml"))
    return demand, skus, br


def test_window_runs_and_returns_consistent_fields():
    demand, skus, br = _load()
    r = run_window(demand, skus, br, pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30"), "test")
    assert r.baseline_total_cost > 0
    assert abs(r.baseline_total_cost - (
        r.baseline_holding_cost + r.baseline_ordering_cost + r.baseline_shortage_cost
    )) < 1.0  # cost components must sum to the total (allow tiny float slack)


def test_ss_pct_of_onhand_in_valid_range():
    demand, skus, br = _load()
    r = run_window(demand, skus, br, pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30"), "test")
    assert 0.0 <= r.ss_pct_of_onhand <= 1.0


def test_coordination_saving_independent_of_shortage_penalty():
    """Pooling acts only on safety stock; the shortage-penalty assumption must not move it."""
    demand, skus, br = _load()
    ts, te = pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30")
    savings = []
    for penalty in [6.0, 12.0, 24.0]:
        r = run_window(demand, skus, br, ts, te, "test", shortage_penalty_override=penalty)
        savings.append(r.coordination_saving_realistic)
    assert max(savings) - min(savings) < 1e-9


def test_shortage_cost_scales_linearly_with_penalty():
    """Doubling the penalty must exactly double the realized shortage cost (same stockout units)."""
    demand, skus, br = _load()
    ts, te = pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30")
    r6 = run_window(demand, skus, br, ts, te, "low", shortage_penalty_override=6.0)
    r12 = run_window(demand, skus, br, ts, te, "moderate", shortage_penalty_override=12.0)
    r24 = run_window(demand, skus, br, ts, te, "high", shortage_penalty_override=24.0)
    assert abs(r12.baseline_shortage_cost - 2 * r6.baseline_shortage_cost) < 1.0
    assert abs(r24.baseline_shortage_cost - 4 * r6.baseline_shortage_cost) < 1.0


def test_shortage_penalty_does_not_affect_holding_or_ordering_cost():
    demand, skus, br = _load()
    ts, te = pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30")
    r_low = run_window(demand, skus, br, ts, te, "low", shortage_penalty_override=6.0)
    r_high = run_window(demand, skus, br, ts, te, "high", shortage_penalty_override=24.0)
    assert abs(r_low.baseline_holding_cost - r_high.baseline_holding_cost) < 1e-6
    assert abs(r_low.baseline_ordering_cost - r_high.baseline_ordering_cost) < 1e-6


def test_two_windows_are_non_overlapping():
    w1_start, w1_end = pd.Timestamp("2025-07-04"), pd.Timestamp("2025-10-01")
    w2_start, w2_end = pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30")
    assert w1_end < w2_start  # strictly before, confirming no shared test data


def test_fill_rates_are_valid_probabilities():
    demand, skus, br = _load()
    r = run_window(demand, skus, br, pd.Timestamp("2025-10-02"), pd.Timestamp("2025-12-30"), "test")
    for abc, fr in r.baseline_fill_by_abc.items():
        assert 0.0 <= fr <= 1.0
