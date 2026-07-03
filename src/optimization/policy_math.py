"""
policy_math.py
==============
Closed-form inventory-policy quantities for the DECENTRALIZED baseline
(Charter Section 5). Every node computes these independently, ignoring the
rest of the network -- that lack of coordination is the whole point of the
baseline, and is exactly what Phase 5's multi-echelon optimizer will improve on.

Formulas (all standard inventory theory):

  Safety stock      SS = z * sigma_d * sqrt(L)
                    z      : service-level z-score (from the ABC service floor)
                    sigma_d: std of per-period DEMAND ERROR (forecast residual),
                             not raw demand -- this credits well-forecast SKUs
                             with needing less buffer.
                    L      : lead time in periods (deterministic, Charter assump #1)

  EOQ               Q* = sqrt(2 * D * K / h)
                    D : annual demand, K : ordering cost, h : annual holding cost/unit

  Reorder point     s = mu_d * L + SS         (demand over lead time + safety stock)
  Order-up-to       S = s + Q*                 (classic (s, S) relationship)

z-scores come from the inverse normal CDF of the service floor.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def service_z(service_level: float) -> float:
    """Inverse-normal z-score for a target cycle service level (e.g. 0.98 -> 2.054)."""
    return float(norm.ppf(service_level))


def safety_stock(z: float, sigma_error: float, lead_time: float) -> float:
    """SS = z * sigma_error * sqrt(L). sigma_error is the forecast-residual std."""
    return float(z * sigma_error * math.sqrt(lead_time))


def eoq(annual_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> float:
    """Economic order quantity. Guards against zero/negative demand or holding cost."""
    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return 0.0
    return float(math.sqrt(2.0 * annual_demand * ordering_cost / holding_cost_per_unit))


def reorder_point(mean_daily_demand: float, lead_time: float, ss: float) -> float:
    """s = expected demand over the lead time + safety stock."""
    return float(mean_daily_demand * lead_time + ss)


def order_up_to(reorder_pt: float, order_qty: float) -> float:
    """S = s + Q*."""
    return float(reorder_pt + order_qty)


def apply_moq_and_pack(qty: float, moq: int, pack_size: int) -> int:
    """
    Round an order quantity up to satisfy the minimum-order-quantity and
    pack-size constraints (Charter Section 3 constraints). Orders are placed
    in whole packs, and never below the MOQ when an order is placed at all.
    """
    if qty <= 0:
        return 0
    qty = max(qty, moq)
    packs = math.ceil(qty / pack_size)
    return int(packs * pack_size)
