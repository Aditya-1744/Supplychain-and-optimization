"""
des_engine.py
=============
Discrete-event inventory simulation for a single (SKU, node) under an
(s, S) policy with deterministic lead time.

DESIGN NOTE -- read before citing "SimPy" in your report:
SimPy could not be installed in the build environment (no network access;
confirmed). This engine implements the SAME discrete-event behaviour SimPy
would model -- a day-by-day event loop tracking inventory position, orders
in transit subject to a lead-time delay, receipts, demand fulfilment, and
unmet demand (stockouts) -- using a plain NumPy/pandas loop so it actually
runs and can be verified here. State this plainly in the methodology section:
"a discrete-event inventory simulation; SimPy was unavailable in the build
environment, so the event loop is implemented directly." The costing logic
is identical to what a SimPy version would produce.

This same engine is reused in Phase 5 (to evaluate the optimized policy) and
Phase 6 (to run the shortage-penalty sensitivity scenarios) -- it is the core
simulation harness for the back half of the project, not throwaway scaffolding.

Events per simulated day, in order:
  1. RECEIVE any order whose arrival day == today (lead-time delay elapsed).
  2. OBSERVE demand for the day.
  3. FULFIL demand from on-hand stock; shortfall is a stockout (lost sales,
     not backordered -- a standard and conservative retail assumption).
  4. REVIEW inventory position (on-hand + on-order); if at or below reorder
     point s, place an order to bring position up to S (respecting MOQ/pack).
  5. ACCRUE end-of-day on-hand holding cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimResult:
    total_holding_cost: float
    total_ordering_cost: float
    total_shortage_cost: float
    total_cost: float
    units_demanded: int
    units_fulfilled: int
    units_short: int
    n_orders: int
    avg_on_hand: float
    fill_rate: float
    daily_on_hand: np.ndarray = field(repr=False, default=None)


def simulate_sS(
    demand: np.ndarray,
    reorder_point: float,
    order_up_to: float,
    lead_time: int,
    unit_cost: float,
    holding_rate_annual: float,
    ordering_cost: float,
    shortage_penalty_per_unit: float,
    moq: int,
    pack_size: int,
    initial_on_hand: float | None = None,
) -> SimResult:
    """
    Run the (s, S) policy over `demand` (1D array of per-day demand) and return
    costed results. Lead time is in days. Holding cost is charged per unit of
    end-of-day on-hand inventory at the daily rate (annual rate / 365).
    """
    from .policy_math import apply_moq_and_pack

    n = len(demand)
    daily_holding_rate = holding_rate_annual / 365.0

    # Start with enough stock to cover the order-up-to level (warm start avoids
    # a spurious stockout burst on day 1 that would unfairly penalise the policy).
    on_hand = float(order_up_to if initial_on_hand is None else initial_on_hand)
    pipeline = {}  # arrival_day -> quantity in transit

    total_holding = total_ordering = total_shortage = 0.0
    units_demanded = units_fulfilled = units_short = 0
    n_orders = 0
    daily_on_hand = np.empty(n)

    for day in range(n):
        # 1. Receive arrivals scheduled for today.
        if day in pipeline:
            on_hand += pipeline.pop(day)

        # 2/3. Demand and fulfilment (lost sales for any shortfall).
        d = float(demand[day])
        fulfilled = min(on_hand, d)
        short = d - fulfilled
        on_hand -= fulfilled
        units_demanded += int(round(d))
        units_fulfilled += int(round(fulfilled))
        units_short += int(round(short))
        total_shortage += short * shortage_penalty_per_unit

        # 4. Review inventory position and reorder if needed.
        on_order = sum(pipeline.values())
        position = on_hand + on_order
        if position <= reorder_point:
            raw_qty = order_up_to - position
            qty = apply_moq_and_pack(raw_qty, moq, pack_size)
            if qty > 0:
                arrival = day + lead_time
                pipeline[arrival] = pipeline.get(arrival, 0) + qty
                total_ordering += ordering_cost
                n_orders += 1

        # 5. End-of-day holding cost on remaining on-hand stock.
        total_holding += on_hand * unit_cost * daily_holding_rate
        daily_on_hand[day] = on_hand

    total_cost = total_holding + total_ordering + total_shortage
    fill_rate = units_fulfilled / units_demanded if units_demanded > 0 else 1.0

    return SimResult(
        total_holding_cost=total_holding,
        total_ordering_cost=total_ordering,
        total_shortage_cost=total_shortage,
        total_cost=total_cost,
        units_demanded=units_demanded,
        units_fulfilled=units_fulfilled,
        units_short=units_short,
        n_orders=n_orders,
        avg_on_hand=float(daily_on_hand.mean()),
        fill_rate=fill_rate,
        daily_on_hand=daily_on_hand,
    )
