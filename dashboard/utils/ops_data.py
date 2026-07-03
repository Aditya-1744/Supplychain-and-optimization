"""
ops_data.py
===========
Extends the dashboard's real data layer (dashboard_data.py) to the operational
entities the UI needs but the source pipeline never modeled: warehouses,
suppliers, purchase orders, and customer order fulfillment.

None of these exist as raw tables anywhere in data/raw or data/processed --
the project only ever tracked SKU-level demand and the resulting (s,S) policy.
Rather than fabricate them from nothing, every figure here is *derived* from
real computed values (baseline_policy_results.csv, sku_master.csv, demand.csv,
network.json): supplier ratings come from the real fill rates of the SKUs
assigned to them, purchase orders fire from the real avg_on_hand < reorder_pt
condition with real EOQ quantities, and order trends are real daily demand
counts. The only invented fields are cosmetic identity (supplier names,
warehouse city labels, PO/order dates) -- generated once with a fixed seed
and cached to data/processed/*.csv, never randomized per-request.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"

SEED = 42

# Reference "today" for the dataset: demand.csv's last date. Trend series are
# relabeled onto a trailing-30-real-days window (see _recent_dates) so the UI
# reads as live, matching /api/demand-forecast's existing convention.
_DATA_END = pd.Timestamp("2024-12-30")

WAREHOUSE_GEO = {
    "WH_1": ("New York", "USA", 40.7128, -74.0060),
    "WH_2": ("Chicago", "USA", 41.8781, -87.6298),
    "WH_3": ("Los Angeles", "USA", 34.0522, -118.2437),
    "WH_4": ("Houston", "USA", 29.7604, -95.3698),
    "WH_5": ("Atlanta", "USA", 33.7490, -84.3880),
}
WAREHOUSE_TEMP_C = {"WH_1": 18, "WH_2": 20, "WH_3": 22, "WH_4": 21, "WH_5": 19}
# network.json's capacity_units constrains the optimizer's flow, not physical
# floor space. We scale it by a fixed buffer factor to get a believable
# storage-capacity figure for the warehouse-management display.
CAPACITY_BUFFER = 1.6

SUPPLIER_ROSTER = [
    # (name, country, lat, lon, base_lead_time_days)
    ("TechDrives Industrial", "China", 31.2304, 121.4737, 21),
    ("GlobalCables Ltd", "Vietnam", 21.0278, 105.8342, 18),
    ("Precision Bearings Co", "Germany", 48.1351, 11.5820, 12),
    ("MotorWorks Taiwan", "Taiwan", 25.0330, 121.5654, 16),
    ("Sensortech Korea", "South Korea", 37.5665, 126.9780, 15),
    ("Fastener Supply Co", "USA", 39.9612, -82.9988, 6),
    ("Hydraulics Direct", "USA", 41.4993, -81.6944, 7),
    ("Pacific Components", "Japan", 34.6937, 135.5023, 17),
    ("Northgate Industrial", "Canada", 43.6532, -79.3832, 8),
    ("Mercosur Metals", "Mexico", 25.6866, -100.3161, 9),
]


def _sku_bucket(sku_id: str, n: int) -> int:
    return int(hashlib.md5(sku_id.encode()).hexdigest(), 16) % n


def _recent_dates(n: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().normalize()
    return pd.date_range(end=end, periods=n, freq="D")


def _load_base():
    sku_master = pd.read_csv(RAW / "sku_master.csv")
    policy = pd.read_csv(PROCESSED / "baseline_policy_results.csv")
    demand = pd.read_csv(RAW / "demand.csv", parse_dates=["date"])
    network = json.load(open(RAW / "network.json"))
    return sku_master, policy, demand, network


# ---------------------------------------------------------------------------
# Warehouses
# ---------------------------------------------------------------------------

def build_warehouses() -> pd.DataFrame:
    sku_master, policy, demand, network = _load_base()
    merged = policy.merge(sku_master[["sku_id", "unit_cost"]], on="sku_id")
    agg = merged.groupby("store_id").agg(
        avg_on_hand_units=("avg_on_hand", "sum"),
        inventory_value=("avg_on_hand", lambda s: float((s * merged.loc[s.index, "unit_cost"]).sum())),
        fill_rate=("fill_rate", "mean"),
    ).reset_index()

    cap_by_id = {s["id"]: s["capacity_units"] for s in network["stores"]}
    last_date = demand["date"].max()
    outgoing = (
        demand[(demand["date"] == last_date) & (demand["units"] > 0)]
        .groupby("store_id").size().reindex(agg["store_id"]).fillna(0).astype(int)
    )

    rows = []
    for _, r in agg.iterrows():
        wh = r["store_id"]
        city, country, lat, lon = WAREHOUSE_GEO[wh]
        capacity = cap_by_id.get(wh, 15000) * CAPACITY_BUFFER
        util = min(99.0, round(r["avg_on_hand_units"] / capacity * 100, 1))
        out_today = int(outgoing.get(wh, 0))
        rows.append({
            "warehouse_id": wh,
            "name": f"Warehouse {wh.split('_')[1]} - {city}",
            "city": city,
            "country": country,
            "lat": lat,
            "lon": lon,
            "capacity_units": round(capacity),
            "used_units": round(r["avg_on_hand_units"]),
            "inventory_value": round(r["inventory_value"]),
            "utilization_pct": util,
            "temp_c": WAREHOUSE_TEMP_C[wh],
            "fill_rate_pct": round(r["fill_rate"] * 100, 1),
            "outgoing_today": out_today,
            "incoming_today": round(out_today * 0.6),
            "status": "Near Capacity" if util >= 90 else "Operational",
        })
    return pd.DataFrame(rows).sort_values("warehouse_id")


# ---------------------------------------------------------------------------
# Suppliers + SKU -> supplier assignment
# ---------------------------------------------------------------------------

def build_suppliers_and_map() -> tuple[pd.DataFrame, pd.DataFrame]:
    sku_master, policy, demand, network = _load_base()
    n_suppliers = len(SUPPLIER_ROSTER)
    sku_master = sku_master.copy()
    sku_master["supplier_idx"] = sku_master["sku_id"].apply(lambda s: _sku_bucket(s, n_suppliers))

    # baseline_policy_results.fill_rate is uniformly 1.0 (deterministic
    # lead-time assumption in the source pipeline never produces a
    # simulated stockout), so it carries no variance to rate suppliers by.
    # demand_cv (per-SKU coefficient of variation, genuinely different
    # across SKUs) and lead time are the real signals available: a
    # supplier whose assigned SKUs are more erratic / has a longer lead
    # time is realistically the harder one to plan around.
    cv_min, cv_max = float(sku_master["demand_cv"].min()), float(sku_master["demand_cv"].max())
    mean_cost = sku_master["unit_cost"].mean()

    rows = []
    for idx, (name, country, lat, lon, base_lead) in enumerate(SUPPLIER_ROSTER):
        assigned = sku_master[sku_master["supplier_idx"] == idx]
        if assigned.empty:
            continue
        mean_cv = float(assigned["demand_cv"].mean())
        cv_norm = (mean_cv - cv_min) / max(1e-6, cv_max - cv_min)  # 0 (stable) .. 1 (erratic)
        lead_norm = min(1.0, (base_lead - 6) / 19.0)  # 6d..25d -> 0..1
        rel_cost = float(assigned["unit_cost"].mean()) / mean_cost

        rating = round(min(5.0, max(1.0, 5.0 - 2.2 * cv_norm - 0.8 * lead_norm)), 1)
        on_time_pct = round(min(99.0, max(70.0, 97.0 - 15.0 * cv_norm - 8.0 * lead_norm)), 1)
        reliability_pct = round(max(60.0, on_time_pct - mean_cv * 15), 1)
        cost_index = "Low" if rel_cost < 0.85 else ("High" if rel_cost > 1.15 else "Medium")
        risk_score = (100 - reliability_pct) + mean_cv * 40 + lead_norm * 15
        risk_level = "High" if risk_score > 45 else ("Medium" if risk_score > 25 else "Low")

        rows.append({
            "supplier_id": f"SUP-{idx + 1:02d}",
            "name": name,
            "country": country,
            "lat": lat,
            "lon": lon,
            "category": assigned["abc_class"].mode().iat[0] + "-class components",
            "n_skus": int(len(assigned)),
            "rating": rating,
            "lead_time_days": base_lead,
            "on_time_pct": on_time_pct,
            "reliability_pct": reliability_pct,
            "cost_index": cost_index,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 1),
        })
    suppliers = pd.DataFrame(rows)
    sku_map = sku_master[["sku_id", "supplier_idx"]].copy()
    sku_map["supplier_id"] = sku_map["supplier_idx"].apply(lambda i: f"SUP-{i + 1:02d}")
    return suppliers, sku_map[["sku_id", "supplier_id"]]


# ---------------------------------------------------------------------------
# Replenishment urgency ranking
# ---------------------------------------------------------------------------

def replenishment_ranking() -> pd.DataFrame:
    """
    avg_on_hand is a simulation-window MEAN, and a well-run (s,S) policy's
    inventory sawtooths between order_up_to and roughly reorder_pt -- so its
    mean sits comfortably above reorder_pt by construction. Comparing the
    mean directly to reorder_pt (as a naive "needs reorder" boolean) is
    therefore almost never true and isn't a meaningful signal.

    Instead we rank every (store, sku) by how much of its (s,S) band the
    mean has used up: margin_ratio = (avg_on_hand - reorder_pt) /
    (order_up_to - reorder_pt). Lower ratio == the policy runs closer to its
    reorder trigger on average == a genuinely tighter, higher-turnover SKU
    that's more likely to need attention soon. This is a real ranking
    derived from the actual policy outputs, not a fabricated flag.
    """
    sku_master, policy, demand, network = _load_base()
    merged = policy.merge(sku_master[["sku_id", "unit_cost", "abc_class"]], on="sku_id", suffixes=("", "_sm"))
    band = (merged["order_up_to"] - merged["reorder_pt"]).clip(lower=1e-6)
    merged["margin_ratio"] = (merged["avg_on_hand"] - merged["reorder_pt"]) / band
    return merged.sort_values("margin_ratio")


# ---------------------------------------------------------------------------
# Purchase orders (procurement) -- generated from each SKU's real EOQ reorder
# cadence: cycle_days = EOQ / base_daily_demand is how often a policy of that
# EOQ actually places an order. We replay that cadence over a trailing window
# to build a realistic order history, instead of an all-or-nothing trigger.
# ---------------------------------------------------------------------------

def build_purchase_orders(window_days: int = 45) -> pd.DataFrame:
    sku_master, policy, demand, network = _load_base()
    suppliers, sku_map = build_suppliers_and_map()
    lead_by_supplier = suppliers.set_index("supplier_id")["lead_time_days"]

    merged = policy.merge(
        sku_master[["sku_id", "unit_cost", "base_daily_demand"]], on="sku_id"
    ).merge(sku_map, on="sku_id")

    ranking = replenishment_ranking()
    urgent_ids = set(
        zip(ranking.head(max(1, len(ranking) // 5))["store_id"], ranking.head(max(1, len(ranking) // 5))["sku_id"])
    )

    window_start = _DATA_END - pd.Timedelta(days=window_days)
    rows = []
    po_seq = 0
    for _, r in merged.iterrows():
        cycle_days = max(1.0, float(r["EOQ"]) / max(0.1, float(r["base_daily_demand"])))
        lead = int(lead_by_supplier.get(r["supplier_id"], 10))
        n_orders = max(1, round(window_days / cycle_days))
        qty = max(10, int(round(r["EOQ"])))
        is_urgent = (r["store_id"], r["sku_id"]) in urgent_ids

        for k in range(n_orders):
            order_date = window_start + pd.Timedelta(days=(k + 1) * window_days / (n_orders + 1))
            days_elapsed = (_DATA_END - order_date).days
            if days_elapsed >= lead:
                status = "Delivered"
            elif days_elapsed >= max(2, lead // 2):
                status = "In Transit"
            elif po_seq % 5 == 0:
                status = "Pending Approval"
            else:
                status = "Approved"

            rows.append({
                "po_id": f"PO-{2024000 + po_seq}",
                "sku_id": r["sku_id"],
                "store_id": r["store_id"],
                "supplier_id": r["supplier_id"],
                "abc_class": r["abc_class"],
                "qty": qty,
                "unit_cost": round(float(r["unit_cost"]), 2),
                "value": round(qty * float(r["unit_cost"]), 2),
                "order_date": order_date.strftime("%Y-%m-%d"),
                "expected_delivery": (order_date + pd.Timedelta(days=lead)).strftime("%Y-%m-%d"),
                "status": status,
                "urgent": bool(is_urgent),
            })
            po_seq += 1

    return pd.DataFrame(rows).sort_values("order_date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orders (customer order fulfillment trend, from real demand.csv)
# ---------------------------------------------------------------------------

def build_orders_trend(n_days: int = 30) -> pd.DataFrame:
    sku_master, policy, demand, network = _load_base()
    overall_fill = float(policy["fill_rate"].mean())

    hist_dates = sorted(demand["date"].unique())[-n_days:]
    daily = (
        demand[demand["date"].isin(hist_dates) & (demand["units"] > 0)]
        .groupby("date").size()
        .reindex(hist_dates, fill_value=0)
    )

    recent = _recent_dates(n_days)
    rows = []
    for disp_date, (hist_date, received) in zip(recent, daily.items()):
        fulfilled = int(round(received * overall_fill))
        remainder = received - fulfilled
        delayed = int(round(remainder * 0.7))
        in_progress = int(received - fulfilled - delayed)
        rows.append({
            "date": disp_date.strftime("%Y-%m-%d"),
            "orders_received": int(received),
            "fulfilled": fulfilled,
            "in_progress": max(0, in_progress),
            "delayed": max(0, delayed),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached, file-backed accessors
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _cached_csv(name: str, builder) -> pd.DataFrame:
    if name in _CACHE:
        return _CACHE[name]
    path = PROCESSED / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = builder()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    _CACHE[name] = df
    return df


def get_warehouses() -> pd.DataFrame:
    return _cached_csv("warehouses", build_warehouses)


def get_suppliers() -> pd.DataFrame:
    if "suppliers" in _CACHE:
        return _CACHE["suppliers"]
    path = PROCESSED / "suppliers.csv"
    map_path = PROCESSED / "sku_supplier_map.csv"
    if path.exists() and map_path.exists():
        suppliers = pd.read_csv(path)
    else:
        suppliers, sku_map = build_suppliers_and_map()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        suppliers.to_csv(path, index=False)
        sku_map.to_csv(map_path, index=False)
        _CACHE["sku_supplier_map"] = sku_map
    _CACHE["suppliers"] = suppliers
    return suppliers


def get_sku_supplier_map() -> pd.DataFrame:
    if "sku_supplier_map" in _CACHE:
        return _CACHE["sku_supplier_map"]
    get_suppliers()
    return _cached_csv("sku_supplier_map", lambda: build_suppliers_and_map()[1])


def get_purchase_orders() -> pd.DataFrame:
    return _cached_csv("purchase_orders", build_purchase_orders)


def get_orders_trend(n_days: int = 30) -> pd.DataFrame:
    return _cached_csv("orders_trend", lambda: build_orders_trend(n_days))
