"""
Unit tests for src/models/inference.py — the live UI-to-model bridge.
Run with: pytest tests/unit/test_inference.py -v
"""
import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pass

from src.models.inference import (
    DemandProfile, synthesize_history, forecast_live, simulate_policy_live, routing_sandbox,
)


def _smooth_profile(**overrides):
    base = dict(sku_id="T1", abc_class="A", archetype="smooth", xyz_class="X",
                demand_cv=0.3, promo_uplift=2.0, base_daily_demand=20, n_history_days=200)
    base.update(overrides)
    return DemandProfile(**base)


def test_synthesize_history_smooth_nonnegative_and_shaped():
    h = synthesize_history(_smooth_profile())
    assert set(h.columns) == {"date", "units", "on_promo"}
    assert len(h) == 200
    assert (h["units"] >= 0).all()


def test_synthesize_history_intermittent_has_real_zero_fraction():
    h = synthesize_history(_smooth_profile(archetype="intermittent", xyz_class="Z",
                                            demand_cv=1.5, base_daily_demand=5))
    assert (h["units"] == 0).mean() > 0.2


def test_forecast_live_routes_smooth_to_fourier():
    r = forecast_live(_smooth_profile(), horizon_days=10)
    assert r.routed_model == "fourier_seasonal"
    assert r.is_disagreement is False
    assert len(r.forecast_values) == 10
    assert (r.forecast_values >= 0).all()


def test_forecast_live_routes_intermittent_to_croston():
    r = forecast_live(_smooth_profile(archetype="intermittent", xyz_class="Z", demand_cv=1.2), horizon_days=10)
    assert r.routed_model == "croston"
    assert r.chosen_submodel in ("croston", "moving_average")


def test_forecast_live_disagreement_routes_to_gbm():
    """The exact case the user asked about: smooth archetype + Z-class XYZ -> GBM."""
    r = forecast_live(_smooth_profile(xyz_class="Z", demand_cv=0.9), horizon_days=10)
    assert r.routed_model == "gbm"
    assert r.is_disagreement is True


def test_forecast_live_forecast_dates_continue_from_history():
    r = forecast_live(_smooth_profile(), horizon_days=10)
    assert r.forecast_dates.min() > r.history["date"].max()


def test_simulate_policy_live_returns_consistent_des_result():
    demand = np.full(90, 10.0)
    sim = simulate_policy_live(demand, abc_class="B", unit_cost=10.0, lead_time=2,
                               holding_cost_rate=0.25, shortage_penalty=12.0)
    assert sim.order_up_to > sim.reorder_point
    assert len(sim.sim_result.daily_on_hand) == 90
    # Cost components must sum to the total (same invariant tested in Phase 4).
    r = sim.sim_result
    assert abs(r.total_cost - (r.total_holding_cost + r.total_ordering_cost + r.total_shortage_cost)) < 1e-6


def test_simulate_policy_live_higher_service_class_gets_more_safety_stock():
    """Class A (98% floor) should carry more safety stock than Class C (90%) for identical demand."""
    demand = np.full(90, 10.0) + np.random.default_rng(0).normal(0, 2, 90).clip(0)
    sim_a = simulate_policy_live(demand, abc_class="A", unit_cost=10.0, lead_time=2,
                                 holding_cost_rate=0.25, shortage_penalty=12.0)
    sim_c = simulate_policy_live(demand, abc_class="C", unit_cost=10.0, lead_time=2,
                                 holding_cost_rate=0.25, shortage_penalty=12.0)
    assert sim_a.safety_stock > sim_c.safety_stock


def test_routing_sandbox_matches_real_router_disagreement_case():
    res = routing_sandbox(archetype="smooth", xyz_class="Z", demand_cv=0.85)
    assert res.routed_model == "gbm"
    assert res.is_disagreement is True
    assert "DISAGREEMENT" in res.explanation


def test_routing_sandbox_smooth_x_class_routes_fourier():
    res = routing_sandbox(archetype="smooth", xyz_class="X", demand_cv=0.2)
    assert res.routed_model == "fourier_seasonal"
    assert res.is_disagreement is False


def test_routing_sandbox_intermittent_routes_croston_regardless_of_xyz():
    res = routing_sandbox(archetype="intermittent", xyz_class="Y", demand_cv=0.6)
    assert res.routed_model == "croston"
