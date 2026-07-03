"""
Unit tests for the Phase 5 multi-echelon optimization module.
Run with: pytest tests/unit/test_multi_echelon.py -v
"""
import math

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pass

from src.optimization.multi_echelon import (
    pooled_safety_stock, decentralized_safety_stock, decompose_sku, optimize_order_quantity,
)
from src.optimization.policy_math import service_z, eoq


def test_pooling_reduces_safety_stock_when_independent():
    """Independent pooling must be strictly less than the decentralized sum."""
    z = service_z(0.95)
    sig = [3.0, 2.0, 1.0]
    dec = decentralized_safety_stock(z, lead_time=2, store_sigmas=sig)
    pooled = pooled_safety_stock(z, lead_time=2, store_sigmas=sig, corr_matrix=None)
    assert pooled < dec


def test_independent_pooling_matches_sqrt_law():
    """Independent pooled sigma should equal sqrt(sum of squares)."""
    z = service_z(0.95)
    sig = [3.0, 4.0]  # sqrt(9+16)=5
    pooled = pooled_safety_stock(z, lead_time=1, store_sigmas=sig, corr_matrix=None)
    assert abs(pooled - z * 5.0) < 1e-9


def test_perfect_correlation_gives_no_pooling_benefit():
    """At rho=1, pooled safety stock equals the decentralized sum (no benefit)."""
    z = service_z(0.95)
    sig = [3.0, 2.0, 1.0]
    C = np.ones((3, 3))  # all correlations = 1
    dec = decentralized_safety_stock(z, lead_time=2, store_sigmas=sig)
    pooled = pooled_safety_stock(z, lead_time=2, store_sigmas=sig, corr_matrix=C)
    assert abs(pooled - dec) < 1e-6


def test_pooling_benefit_monotonic_in_correlation():
    """Higher correlation -> smaller pooling benefit (pooled SS rises toward the sum)."""
    z = service_z(0.95)
    sig = [3.0, 2.0, 1.0]
    prev = None
    for rho in [0.0, 0.3, 0.6, 0.9, 1.0]:
        C = np.full((3, 3), rho)
        np.fill_diagonal(C, 1.0)
        pooled = pooled_safety_stock(z, 2, sig, corr_matrix=C)
        if prev is not None:
            assert pooled >= prev - 1e-9  # non-decreasing in rho
        prev = pooled


def test_decompose_orders_regimes_correctly():
    """For positive correlation < 1: independent <= correlated <= decentralized."""
    C = np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [0.6, 0.6, 1.0]])
    d = decompose_sku("SKU1", "A", [3.0, 2.0, 1.0], service_level=0.98, lead_time=2, corr_matrix=C)
    assert d.ss_pooled_independent <= d.ss_pooled_correlated <= d.ss_decentralized


def test_optimize_order_quantity_recovers_eoq():
    """With no extra constraints, the numerical optimum should match closed-form EOQ."""
    D, K, h = 2000.0, 75.0, 2.5
    numeric = optimize_order_quantity(D, K, h)
    closed = eoq(D, K, h)
    assert abs(numeric - closed) / closed < 0.02  # within 2%


def test_optimize_order_quantity_respects_cap():
    """A max_q cap below the unconstrained optimum should bind."""
    D, K, h = 2000.0, 75.0, 0.5  # large EOQ
    capped = optimize_order_quantity(D, K, h, max_q=50.0)
    assert capped <= 50.0 + 1e-6


def test_zero_demand_returns_zero_quantity():
    assert optimize_order_quantity(0, 75, 2.5) == 0.0
