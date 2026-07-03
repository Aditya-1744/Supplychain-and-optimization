"""
Unit tests for dashboard/utils/dashboard_data.py — the single source of truth
shared by the Streamlit app and the verified HTML preview.
Run with: pytest tests/unit/test_dashboard_data.py -v
"""
try:
    import pytest  # noqa: F401
except ImportError:
    pass

from dashboard.utils.dashboard_data import (
    load_all, compute_kpis, what_if_cost, penalty_ladder_df, correlation_sensitivity_curve,
)


def test_load_all_returns_expected_keys():
    data = load_all()
    assert set(data.keys()) == {
        "baseline", "optimization", "backtest", "business_rules",
        "baseline_policy_df", "comparison_df",
    }


def test_compute_kpis_returns_all_expected_fields():
    data = load_all()
    kpis = compute_kpis(data)
    expected_keys = {
        "total_cost", "holding_cost", "ordering_cost", "shortage_cost",
        "fill_rate_A", "fill_rate_B", "fill_rate_C", "floor_A", "floor_B", "floor_C",
        "safety_stock_pct", "coordination_saving_realistic", "coordination_saving_upper_bound",
        "mean_correlation", "window1_coord_saving", "window2_coord_saving",
        "shortage_penalty_spread_pct",
    }
    assert set(kpis.keys()) == expected_keys


def test_kpi_values_in_sane_ranges():
    data = load_all()
    kpis = compute_kpis(data)
    assert kpis["total_cost"] > 0
    for k in ["fill_rate_A", "fill_rate_B", "fill_rate_C", "safety_stock_pct",
              "coordination_saving_realistic", "mean_correlation"]:
        assert 0.0 <= kpis[k] <= 1.0, f"{k} out of [0,1] range: {kpis[k]}"


def test_fill_rates_meet_their_floors():
    """The persisted baseline result should show every class above its floor."""
    data = load_all()
    kpis = compute_kpis(data)
    assert kpis["fill_rate_A"] >= kpis["floor_A"]
    assert kpis["fill_rate_B"] >= kpis["floor_B"]
    assert kpis["fill_rate_C"] >= kpis["floor_C"]


def test_what_if_cost_matches_locked_ladder_exactly():
    """The slider formula must reproduce Phase 6's saved results at the 3 locked points."""
    data = load_all()
    ladder = data["backtest"]["shortage_penalty_ladder"]
    for row in ladder:
        result = what_if_cost(data, row["penalty"])
        assert abs(result["total_cost"] - row["total_cost"]) < 1.0
        assert abs(result["shortage_cost"] - row["shortage_cost"]) < 1.0


def test_what_if_cost_is_linear_in_penalty():
    """Shortage cost must scale exactly linearly with the penalty (Phase 6 finding)."""
    data = load_all()
    w1 = what_if_cost(data, 10.0)
    w2 = what_if_cost(data, 20.0)
    assert abs(w2["shortage_cost"] - 2 * w1["shortage_cost"]) < 1.0


def test_what_if_cost_holding_and_ordering_constant():
    """Holding/ordering must NOT change with the shortage-penalty slider."""
    data = load_all()
    w_low = what_if_cost(data, 5.0)
    w_high = what_if_cost(data, 50.0)
    assert w_low["holding_cost"] == w_high["holding_cost"]
    assert w_low["ordering_cost"] == w_high["ordering_cost"]


def test_what_if_cost_zero_penalty_means_zero_shortage_cost():
    data = load_all()
    w = what_if_cost(data, 0.0)
    assert w["shortage_cost"] == 0.0


def test_penalty_ladder_df_sorted_and_complete():
    data = load_all()
    df = penalty_ladder_df(data)
    assert len(df) == 3
    assert list(df["penalty"]) == sorted(df["penalty"])


def test_correlation_curve_monotonic_decreasing():
    """Pooling benefit must fall as correlation rises (matches Phase 5's proven property)."""
    curve = correlation_sensitivity_curve()
    vals = curve["saving_pct"].to_numpy()
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def test_correlation_curve_endpoints():
    curve = correlation_sensitivity_curve()
    assert curve.iloc[0]["saving_pct"] > 30  # rho=0 -> large benefit
    assert abs(curve.iloc[-1]["saving_pct"]) < 1e-6  # rho=1 -> zero benefit


def test_load_all_raises_clearly_if_files_missing(tmp_path=None):
    """Confirm the error path gives an actionable message rather than a bare crash."""
    import dashboard.utils.dashboard_data as dd
    original = dd.PROCESSED
    dd.PROCESSED = dd.REPO_ROOT / "definitely_does_not_exist"
    try:
        raised = False
        try:
            dd.load_all()
        except FileNotFoundError as e:
            raised = True
            assert "requires these phase outputs" in str(e)
        assert raised
    finally:
        dd.PROCESSED = original
