"""
multi_echelon.py
================
Coordinated multi-echelon inventory optimization for the 3-echelon network
(Factory -> DC -> 3 Stores), and the savings DECOMPOSITION that attributes
the win to its two distinct causes.

Approach (confirmed with stakeholder):
  * ANALYTICAL multi-echelon model (closed-form risk-pooling), with order
    quantities optimized via scipy.optimize. Chosen over PuLP because the
    pooling benefit is intrinsically NONLINEAR (square-root law), which a
    linear program cannot represent without linearization that would destroy
    the very effect being measured. (PuLP was also unavailable in the build
    environment.)
  * Cross-store demand assumed INDEPENDENT for the clean upper-bound pooling
    figure, but the REALISTIC correlated figure is computed too and LEADS the
    reporting, because measured cross-store correlation (~0.68) makes the
    independent number an optimistic ~4x overstatement.

THE SAVINGS DECOMPOSITION (baseline -> optimized, two separable steps):
  Step A  "floor right-sizing": keep safety stock DECENTRALIZED at the stores,
          but size it to actually hit the ABC service floor instead of the
          baseline's incidental over-service. Isolates the EOQ/floor batch
          effect found in Phase 4.
  Step B  "coordination": pool the safety stock at the DC. The ADDITIONAL
          saving from A to B is the genuine multi-echelon benefit.

Risk-pooling core (the square-root law):
  decentralized SS_total = z * sqrt(L) * sum_i(sigma_i)
  pooled (independent)    = z * sqrt(L) * sqrt(sum_i(sigma_i^2))
  pooled (correlated)     = z * sqrt(L) * sqrt(sigma' . Sigma_corr . sigma)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .policy_math import service_z


@dataclass
class EchelonDecomposition:
    """Per-SKU safety-stock figures across the three policy regimes."""
    sku_id: str
    abc_class: str
    ss_decentralized: float       # baseline-style: sum of per-store SS at floor
    ss_pooled_independent: float  # pooled at DC, independence assumption (upper bound)
    ss_pooled_correlated: float   # pooled at DC, measured correlation (realistic)
    store_sigmas: list[float]


def pooled_safety_stock(
    z: float,
    lead_time: float,
    store_sigmas: list[float],
    corr_matrix: np.ndarray | None = None,
) -> float:
    """
    Pooled safety stock under either independence (corr_matrix=None) or a
    supplied correlation matrix. Both use the same z and lead time.
    """
    sig = np.asarray(store_sigmas, dtype=float)
    if corr_matrix is None:
        pooled_sigma = math.sqrt(float(np.sum(sig ** 2)))         # independent case
    else:
        # Variance of summed (pooled) demand = sum_ij rho_ij * sigma_i * sigma_j.
        cov = corr_matrix * np.outer(sig, sig)
        pooled_sigma = math.sqrt(float(np.sum(cov)))
    return float(z * pooled_sigma * math.sqrt(lead_time))


def decentralized_safety_stock(z: float, lead_time: float, store_sigmas: list[float]) -> float:
    """Sum of independent per-store safety stocks (no pooling)."""
    sig = np.asarray(store_sigmas, dtype=float)
    return float(z * math.sqrt(lead_time) * np.sum(sig))


def decompose_sku(
    sku_id: str,
    abc_class: str,
    store_sigmas: list[float],
    service_level: float,
    lead_time: float,
    corr_matrix: np.ndarray | None,
) -> EchelonDecomposition:
    """Compute the three safety-stock regimes for one SKU."""
    z = service_z(service_level)
    ss_dec = decentralized_safety_stock(z, lead_time, store_sigmas)
    ss_indep = pooled_safety_stock(z, lead_time, store_sigmas, corr_matrix=None)
    ss_corr = pooled_safety_stock(z, lead_time, store_sigmas, corr_matrix=corr_matrix) \
        if corr_matrix is not None else ss_indep
    return EchelonDecomposition(
        sku_id=sku_id, abc_class=abc_class,
        ss_decentralized=ss_dec,
        ss_pooled_independent=ss_indep,
        ss_pooled_correlated=ss_corr,
        store_sigmas=list(store_sigmas),
    )


def optimize_order_quantity(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
    max_q: float | None = None,
) -> float:
    """
    Optimal order quantity minimizing ordering + holding cost per year. With no
    extra constraints this reproduces the EOQ closed form, but solving it via
    scipy.optimize.minimize_scalar lets Phase 5 add nonlinear terms or caps
    (e.g. a max_q from warehouse-capacity considerations) that pure EOQ can't
    express -- which is the point of using a numerical optimizer here.

    total annual cost(Q) = D/Q * K + Q/2 * h
    """
    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return 0.0

    def total_cost(Q):
        if Q <= 0:
            return np.inf
        return (annual_demand / Q) * ordering_cost + (Q / 2.0) * holding_cost_per_unit

    upper = max_q if max_q is not None else annual_demand  # sane search ceiling
    res = minimize_scalar(total_cost, bounds=(1.0, max(2.0, upper)), method="bounded")
    return float(res.x)
