"""
Unit tests for the Phase 4 baseline policy: inventory math + DES engine.
Run with: pytest tests/unit/test_baseline_policy.py -v
"""
import numpy as np

try:
    import pytest  # noqa: F401  (no pytest-specific features used; kept for `pytest -v`)
except ImportError:
    pass

from src.optimization.policy_math import (
    service_z, safety_stock, eoq, reorder_point, order_up_to, apply_moq_and_pack,
)
from src.optimization.des_engine import simulate_sS


# ----------------------------- policy math ------------------------------ #
def test_service_z_monotonic():
    # Higher service level -> higher z-score.
    assert service_z(0.90) < service_z(0.95) < service_z(0.98)
    assert abs(service_z(0.98) - 2.0537) < 1e-3  # known value


def test_safety_stock_scales_with_sqrt_lead_time():
    ss1 = safety_stock(z=2.0, sigma_error=5.0, lead_time=1)
    ss4 = safety_stock(z=2.0, sigma_error=5.0, lead_time=4)
    assert abs(ss4 / ss1 - 2.0) < 1e-9  # sqrt(4)/sqrt(1) = 2


def test_eoq_known_value():
    # D=1000, K=50, h=2 -> sqrt(2*1000*50/2) = sqrt(50000) ~ 223.6
    assert abs(eoq(1000, 50, 2) - 223.6068) < 1e-3


def test_eoq_guards_zero_inputs():
    assert eoq(0, 50, 2) == 0.0
    assert eoq(1000, 50, 0) == 0.0


def test_reorder_and_order_up_to():
    s = reorder_point(mean_daily_demand=10, lead_time=2, ss=5)
    assert s == 25  # 10*2 + 5
    S = order_up_to(s, order_qty=40)
    assert S == 65


def test_moq_and_pack_rounding():
    assert apply_moq_and_pack(0, moq=10, pack_size=5) == 0       # no order
    assert apply_moq_and_pack(3, moq=10, pack_size=5) == 10      # below MOQ -> MOQ, rounded to pack
    assert apply_moq_and_pack(12, moq=10, pack_size=5) == 15     # 12 -> next pack multiple
    assert apply_moq_and_pack(20, moq=10, pack_size=5) == 20     # already a pack multiple


# ------------------------------ DES engine ------------------------------ #
def _const_demand(n=90, d=5):
    return np.full(n, float(d))


def test_des_unit_conservation():
    """Demanded = fulfilled + short, always."""
    demand = _const_demand()
    r = simulate_sS(demand, reorder_point=20, order_up_to=60, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5)
    assert r.units_demanded == r.units_fulfilled + r.units_short


def test_des_total_cost_is_sum_of_parts():
    demand = _const_demand()
    r = simulate_sS(demand, reorder_point=20, order_up_to=60, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5)
    assert abs(r.total_cost - (r.total_holding_cost + r.total_ordering_cost + r.total_shortage_cost)) < 1e-6


def test_des_high_stock_no_shortage():
    """A generous order-up-to relative to demand should avoid stockouts."""
    demand = _const_demand(n=90, d=5)
    r = simulate_sS(demand, reorder_point=40, order_up_to=200, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5)
    assert r.units_short == 0
    assert r.fill_rate == 1.0


def test_des_zero_replenishment_causes_stockout():
    """If we never reorder (S below demand-over-leadtime), stock runs out."""
    demand = _const_demand(n=90, d=10)
    # reorder point negative so no order ever triggers; start with small stock
    r = simulate_sS(demand, reorder_point=-1, order_up_to=0, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5,
                    initial_on_hand=20)
    assert r.units_short > 0
    assert r.fill_rate < 1.0


def test_des_fill_rate_bounds():
    demand = _const_demand()
    r = simulate_sS(demand, reorder_point=20, order_up_to=60, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5)
    assert 0.0 <= r.fill_rate <= 1.0


def test_des_holding_cost_nonnegative():
    demand = _const_demand()
    r = simulate_sS(demand, reorder_point=20, order_up_to=60, lead_time=2,
                    unit_cost=10, holding_rate_annual=0.25, ordering_cost=75,
                    shortage_penalty_per_unit=12, moq=10, pack_size=5)
    assert r.total_holding_cost >= 0
    assert r.avg_on_hand >= 0
