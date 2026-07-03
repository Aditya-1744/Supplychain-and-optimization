"""
rolling_backtest.py
====================
Generalizes the Phase 3 (forecast) -> Phase 4 (baseline) -> Phase 5 (pooling)
pipeline to run over an ARBITRARY (train_end, test_start, test_end) window,
so the same logic can be re-run on multiple rolling-origin windows without
duplicating it. This is the Phase 6 rigor check: does the Phase 5 finding
(pooling bounded by safety-stock share) hold up on a different period, or
was it specific to the one window used in Phases 4-5?

Each call to `run_window` reproduces, for that window only:
  - forecast residual std per SKU (for safety stock)
  - decentralized baseline cost (DES-simulated)
  - safety-stock pooling benefit (realistic + independence upper bound)
  - the inventory-composition ratio (safety stock as % of on-hand)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.router import route_skus
from src.models.seasonal_model import FourierSeasonalForecaster
from src.models.gbm_model import GBMForecaster
from src.models.croston import CrostonOrBaseline
from src.optimization.policy_math import service_z, safety_stock, eoq, reorder_point, order_up_to
from src.optimization.multi_echelon import decompose_sku
from src.optimization.des_engine import simulate_sS


@dataclass
class WindowResult:
    label: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_days: int
    baseline_total_cost: float
    baseline_holding_cost: float
    baseline_ordering_cost: float
    baseline_shortage_cost: float
    baseline_fill_by_abc: dict
    ss_units: float
    onhand_units: float
    ss_pct_of_onhand: float
    coordination_saving_realistic: float
    coordination_saving_upper_bound: float
    mean_cross_store_corr: float


def run_window(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    br: dict,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    label: str,
    shortage_penalty_override: float | None = None,
) -> WindowResult:
    """Run the full forecast -> baseline -> pooling pipeline for one window."""
    costs, floors = dict(br["costs"]), br["service_floors"]
    if shortage_penalty_override is not None:
        costs["shortage_penalty_per_unit"] = shortage_penalty_override

    LT_STORE = br["assumptions"]["lead_time_days"]["dc_to_store"]
    MOQ, PACK = br["constraints"]["min_order_qty"], br["constraints"]["pack_size"]
    stores = {s["id"]: s for s in br["network"]["stores"]}
    store_ids = list(stores.keys())

    full_idx = pd.date_range(demand.date.min(), test_end, freq="D")
    routing, _ = route_skus(skus)
    sku_info = skus.set_index("sku_id")
    daily_h = costs["holding_rate_annual"] / 365.0
    train_days = (full_idx < test_start).sum()

    net_daily = (
        demand[demand.date <= test_end]
        .groupby(["sku_id", "date"], observed=True)
        .agg(units=("units", "sum"), on_promo=("on_promo", "max"))
        .reset_index().sort_values(["sku_id", "date"])
    )

    def resid_std_net(sku_id):
        sub = net_daily[net_daily.sku_id == sku_id].set_index("date").reindex(full_idx)
        sub["units"] = sub["units"].fillna(0)
        sub["on_promo"] = sub["on_promo"].fillna(False)
        sub = sub.reset_index().rename(columns={"index": "date"})
        tr = sub[sub.date < test_start]
        te = sub[(sub.date >= test_start) & (sub.date <= test_end)]
        model = routing.set_index("sku_id").loc[sku_id, "model"]
        yv = te["units"].to_numpy()
        if model == "fourier_seasonal":
            m = FourierSeasonalForecaster().fit(tr["date"], tr["units"].to_numpy(), tr["on_promo"].to_numpy())
            pred = m.predict(te["date"], te["on_promo"].to_numpy())
        elif model == "gbm":
            m = GBMForecaster().fit(tr[["date", "units", "on_promo"]])
            pred = m.predict(te["date"], te["on_promo"].to_numpy())
        else:
            m = CrostonOrBaseline().fit(tr["units"].to_numpy())
            pred = m.predict(len(yv))
        return float(np.std(yv - pred)), tr["units"].to_numpy()

    resid_net, net_mean = {}, {}
    for sid in routing.sku_id:
        std, tr_units = resid_std_net(sid)
        resid_net[sid] = std
        net_mean[sid] = float(np.mean(tr_units)) if len(tr_units) else 0.0

    # --- Decentralized baseline: simulate every store-SKU over this window ---
    def store_series(store_id, sku_id):
        sub = (demand[(demand.store_id == store_id) & (demand.sku_id == sku_id) & (demand.date <= test_end)]
               .groupby("date")["units"].sum().reindex(full_idx).fillna(0))
        return sub.to_numpy()

    share = {st: stores[st]["size_multiplier"] / sum(s["size_multiplier"] for s in stores.values()) for st in store_ids}
    n_train = train_days

    total_cost = total_holding = total_ordering = total_shortage = 0.0
    fill_by_abc = {"A": [], "B": [], "C": []}
    ss_rows = []
    for st in store_ids:
        for sid in routing.sku_id:
            abc = sku_info.loc[sid, "abc_class"]
            unit_cost = float(sku_info.loc[sid, "unit_cost"])
            z = service_z(floors[abc])
            series = store_series(st, sid)
            train_d, sim_d = series[:n_train], series[n_train:]
            mean_daily = float(train_d.mean()) if len(train_d) else 0.0
            sigma_err = resid_net[sid] * share[st]

            h = unit_cost * costs["holding_rate_annual"]
            ss = safety_stock(z, sigma_err, LT_STORE)
            Q = eoq(mean_daily * 365.0, costs["ordering_cost"], h)
            s = reorder_point(mean_daily, LT_STORE, ss)
            S = order_up_to(s, Q)

            r = simulate_sS(sim_d, s, S, LT_STORE, unit_cost, costs["holding_rate_annual"],
                            costs["ordering_cost"], costs["shortage_penalty_per_unit"], MOQ, PACK)
            total_cost += r.total_cost
            total_holding += r.total_holding_cost
            total_ordering += r.total_ordering_cost
            total_shortage += r.total_shortage_cost
            fill_by_abc[abc].append(r.fill_rate)
            ss_rows.append({"safety_stock": ss, "avg_on_hand": r.avg_on_hand})

    ss_df = pd.DataFrame(ss_rows)
    ss_units = ss_df["safety_stock"].sum()
    onhand_units = ss_df["avg_on_hand"].sum()

    # --- Pooling (coordination) benefit, same window ---
    piv = (demand[(demand.date >= demand.date.min()) & (demand.date <= test_end)]
           .groupby(["date", "store_id"])["units"].sum().unstack("store_id")[store_ids])
    corr_mat = piv.corr().to_numpy()
    mean_rho = corr_mat[np.triu_indices(len(store_ids), 1)].mean()

    rows = []
    for sid in routing.sku_id:
        abc = sku_info.loc[sid, "abc_class"]
        unit_cost = float(sku_info.loc[sid, "unit_cost"])
        sig = [resid_net[sid] * share[st] for st in store_ids]
        d = decompose_sku(sid, abc, sig, floors[abc], LT_STORE, corr_mat)
        hcf = unit_cost * daily_h * (test_end - test_start).days
        rows.append({"dec": d.ss_decentralized * hcf, "corr": d.ss_pooled_correlated * hcf,
                     "indep": d.ss_pooled_independent * hcf})
    P = pd.DataFrame(rows)
    # NOTE: use bracket access, not attribute access -- "corr" collides with
    # DataFrame.corr() and silently returns the method instead of the column.
    dec_cost = P["dec"].sum()
    corr_cost = P["corr"].sum()
    indep_cost = P["indep"].sum()

    return WindowResult(
        label=label, test_start=test_start, test_end=test_end, train_days=int(train_days),
        baseline_total_cost=total_cost, baseline_holding_cost=total_holding,
        baseline_ordering_cost=total_ordering, baseline_shortage_cost=total_shortage,
        baseline_fill_by_abc={k: float(np.mean(v)) for k, v in fill_by_abc.items()},
        ss_units=float(ss_units), onhand_units=float(onhand_units),
        ss_pct_of_onhand=float(ss_units / onhand_units) if onhand_units else 0.0,
        coordination_saving_realistic=float(1 - corr_cost / dec_cost) if dec_cost else 0.0,
        coordination_saving_upper_bound=float(1 - indep_cost / dec_cost) if dec_cost else 0.0,
        mean_cross_store_corr=float(mean_rho),
    )
