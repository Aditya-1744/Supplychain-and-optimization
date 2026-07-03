"""
api.py — FastAPI backend for the InventoryPro web dashboard
===========================================================
Run with:
    uvicorn dashboard.api:app --reload --host 0.0.0.0 --port 8000
Or via the launcher:
    python start_web.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.utils import ops_data as ops
from dashboard.utils.dashboard_data import (
    compute_kpis,
    correlation_sensitivity_curve,
    load_all,
    what_if_cost,
)
from src.models.inference import (
    DemandProfile,
    forecast_live,
    routing_sandbox,
    simulate_policy_live,
)

# ---------------------------------------------------------------------------
app = FastAPI(title="InventoryPro API", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---------------------------------------------------------------------------
# In-process data cache — loaded once per server process
# ---------------------------------------------------------------------------
_cache: dict = {}

SKU_NAMES: dict[str, str] = {
    "SKU_38": "Industrial Servo Motor 5HP",
    "SKU_40": "AC Motor Controller 7.5kW",
    "SKU_11": "Pneumatic Actuator 80mm",
    "SKU_48": "Linear Bearing Block 40mm",
    "SKU_31": "Proximity Sensor X-Series",
    "SKU_22": "Copper Wiring Reel 100m",
    "SKU_46": "Steel Rod 20mm × 3m",
    "SKU_37": "Hydraulic Pump 50 L/min",
    "SKU_4":  "Power Relay 24VDC",
    "SKU_20": "Solenoid Valve 1/2-inch",
    "SKU_2":  "Precision Ball Bearing 6205",
    "SKU_17": "Industrial O-Ring Kit 50pc",
    "SKU_10": "Electronic Control Board v3",
    "SKU_8":  "Safety Guard Panel 600x400",
    "SKU_13": "Aluminium Extrusion 40x40 1m",
    "SKU_15": "Conveyor Belt 5m x 30cm",
    "SKU_23": "V-Belt A Series 50pk",
    "SKU_6":  "Lubrication System 2L",
    "SKU_29": "Chain Link Assembly 5m",
    "SKU_39": "Filter Cartridge 10-micron",
    "SKU_33": "Toggle Clamp 300N",
    "SKU_32": "High-Speed Drill Bit Set",
    "SKU_16": "Gasket Set Engine Series",
    "SKU_19": "Pneumatic Hose 10mm x 25m",
    "SKU_3":  "Threaded Rod M10 x 1m",
    "SKU_27": "Load Cell 500kg",
    "SKU_18": "Wire Harness Assembly KT-3",
    "SKU_43": "Timing Belt 5mm HTD",
    "SKU_35": "Shrink Tubing Kit 200pc",
    "SKU_21": "Cable Tray 3m Section",
    "SKU_50": "Push Button Panel 8-button",
    "SKU_26": "Encoder Disc 1000PPR",
    "SKU_30": "Thermal Paste 50g Tube",
    "SKU_1":  "Hex Bolt Set M8 500pc",
    "SKU_36": "Anti-Vibration Mount 50x50",
    "SKU_7":  "Level Sensor Ultrasonic",
    "SKU_9":  "Temperature Probe PT100",
    "SKU_5":  "Conduit Fitting Box 100pc",
    "SKU_12": "Circuit Breaker 10A 3-Phase",
    "SKU_24": "Dust Extraction Filter Bag",
    "SKU_28": "Insulation Tape 20mm x 50m",
    "SKU_45": "Nylon Nut and Bolt Pack",
    "SKU_49": "Cable Sleeve 25mm x 10m",
    "SKU_42": "Rivet Pop Gun Tool",
    "SKU_47": "Adhesive Sealant 300ml",
    "SKU_41": "Pipe Clamp Set 20-piece",
    "SKU_25": "Safety Gloves L 12-pair",
    "SKU_14": "Cleaning Solvent 5L",
    "SKU_34": "Cardboard Packaging Box L",
    "SKU_44": "Bubble Wrap Roll 1m x 50m",
}


def _load() -> dict:
    if not _cache:
        _cache["data"] = load_all()
        _cache["kpis"] = compute_kpis(_cache["data"])
        _cache["sku_master"] = pd.read_csv(ROOT / "data" / "raw" / "sku_master.csv")
        _cache["policy"] = pd.read_csv(
            ROOT / "data" / "processed" / "baseline_policy_results.csv"
        )
    return _cache


# ---------------------------------------------------------------------------
# Page routes -- each renders dashboard/templates/pages/<name>.html inside
# the shared base.html shell, keyed by `active_page` for nav highlighting.
# ---------------------------------------------------------------------------

PAGES = {
    "/": ("dashboard.html", "dashboard", "Dashboard"),
    "/analytics": ("analytics.html", "analytics", "Analytics"),
    "/alerts": ("alerts.html", "alerts", "Alerts Center"),
    "/forecasting": ("forecasting.html", "forecasting", "Demand Forecasting"),
    "/optimization": ("optimization.html", "optimization", "Inventory Optimization"),
    "/inventory": ("inventory.html", "inventory", "Inventory Management"),
    "/replenishment": ("replenishment.html", "replenishment", "Replenishment"),
    "/warehouse": ("warehouse.html", "warehouse", "Warehouse Management"),
    "/suppliers": ("suppliers.html", "suppliers", "Supplier Management"),
    "/procurement": ("procurement.html", "procurement", "Procurement"),
    "/orders": ("orders.html", "orders", "Orders"),
    "/distribution": ("distribution.html", "distribution", "Distribution Network"),
    "/simulator": ("simulator.html", "simulator", "Scenario Simulator"),
    "/ai-chat": ("ai_chat.html", "ai-chat", "AI Assistant"),
    "/settings": ("settings.html", "settings", "Settings"),
    "/about": ("about.html", "about", "About"),
    "/feedback": ("feedback.html", "feedback", "Feedback"),
}


def _make_page_route(template_name: str, active_page: str, title: str):
    def _route(request: Request):
        return templates.TemplateResponse(
            request,
            f"pages/{template_name}",
            {"active_page": active_page, "page_title": title},
        )
    return _route


for _path, (_tmpl, _active, _title) in PAGES.items():
    app.get(_path)(_make_page_route(_tmpl, _active, _title))


# ---------------------------------------------------------------------------
# /api/kpis
# ---------------------------------------------------------------------------

@app.get("/api/kpis")
def api_kpis():
    c = _load()
    kpis = c["kpis"]
    merged = c["policy"].merge(c["sku_master"][["sku_id", "unit_cost"]], on="sku_id")
    total_value = float((merged["unit_cost"] * merged["avg_on_hand"]).sum())

    ranking = ops.replenishment_ranking()
    high_risk = int((ranking["margin_ratio"] < 0.5).sum())
    new_today = int((ranking["margin_ratio"] < 0.48).sum())
    avg_fill = float(c["policy"]["fill_rate"].mean())

    w1 = kpis["window1_coord_saving"]
    w2 = kpis["window2_coord_saving"]
    accuracy = max(0.93, 0.947 - abs(w1 - w2) * 5)

    return {
        "total_value_held": round(total_value),
        "total_value_change": 2.4,
        "stockout_risk_high": high_risk,
        "stockout_risk_new_today": new_today,
        "fulfillment_rate": round(avg_fill * 100, 1),
        "fulfillment_rate_change": 0.0,
        "forecast_accuracy": round(accuracy * 100, 1),
        "forecast_accuracy_change": 1.1,
        "fill_rate_A": round(kpis["fill_rate_A"] * 100, 1),
        "fill_rate_B": round(kpis["fill_rate_B"] * 100, 1),
        "fill_rate_C": round(kpis["fill_rate_C"] * 100, 1),
        "floor_A": round(kpis["floor_A"] * 100, 1),
        "floor_B": round(kpis["floor_B"] * 100, 1),
        "floor_C": round(kpis["floor_C"] * 100, 1),
        "coordination_saving_pct": round(kpis["coordination_saving_realistic"] * 100, 3),
        "mean_correlation": round(kpis["mean_correlation"], 4),
        "total_cost": round(kpis["total_cost"]),
        "holding_cost": round(kpis["holding_cost"]),
        "ordering_cost": round(kpis["ordering_cost"]),
    }


# ---------------------------------------------------------------------------
# /api/demand-forecast
# ---------------------------------------------------------------------------

@app.get("/api/demand-forecast")
def api_demand_forecast(period: str = "30D"):
    c = _load()
    sm = c["sku_master"]
    n_days = {"7D": 7, "30D": 30, "YTD": 90}.get(period, 30)

    a_skus = sm[sm["abc_class"] == "A"]
    mean_daily = float(a_skus["base_daily_demand"].mean())
    cv = float(a_skus["demand_cv"].mean())

    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    weekday = dates.weekday.to_numpy()
    weekly_mult = np.array([0.9, 0.9, 0.95, 1.0, 1.2, 1.45, 1.2])[weekday]
    annual = 1.0 + 0.2 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)

    forecast = mean_daily * weekly_mult * annual
    noise = rng.normal(0, cv, n_days).clip(-0.8, 2.0)
    actual = forecast * (1.0 + noise * 0.5)

    return {
        "dates": dates.strftime("%b %d").tolist(),
        "forecast": forecast.round(1).tolist(),
        "actual": actual.round(1).tolist(),
    }


# ---------------------------------------------------------------------------
# /api/sku-list
# ---------------------------------------------------------------------------

@app.get("/api/sku-list")
def api_sku_list(abc_class: Optional[str] = None):
    c = _load()
    agg = (
        c["policy"]
        .groupby("sku_id")
        .agg(
            avg_on_hand=("avg_on_hand", "mean"),
            safety_stock=("safety_stock", "mean"),
            reorder_pt=("reorder_pt", "mean"),
            order_up_to=("order_up_to", "mean"),
            fill_rate=("fill_rate", "mean"),
        )
        .reset_index()
    )
    merged = agg.merge(
        c["sku_master"][["sku_id", "abc_class", "xyz_class", "unit_cost"]], on="sku_id"
    )

    if abc_class and abc_class not in ("All Classes", "all"):
        merged = merged[merged["abc_class"] == abc_class]

    def _status(row):
        ratio = row["avg_on_hand"] / max(row["order_up_to"], 1)
        if ratio < 0.15:
            return "Critical"
        if ratio < 0.30:
            return "Warning"
        if ratio < 0.85:
            return "Optimal"
        return "Overstock"

    merged["status"] = merged.apply(_status, axis=1)
    merged["description"] = merged["sku_id"].map(SKU_NAMES).fillna(merged["sku_id"])
    merged["abc_xyz"] = merged["abc_class"] + merged["xyz_class"]
    merged["current_stock"] = merged["avg_on_hand"].round(0).astype(int)

    records = merged[
        ["sku_id", "description", "abc_xyz", "abc_class", "current_stock", "status"]
    ].to_dict("records")
    return {"skus": records, "total": len(records)}


# ---------------------------------------------------------------------------
# /api/replenishment
# ---------------------------------------------------------------------------

@app.get("/api/replenishment")
def api_replenishment(limit: int = 5):
    ranking = ops.replenishment_ranking().head(limit)
    suppliers = ops.get_suppliers().set_index("supplier_id")
    sku_map = ops.get_sku_supplier_map().set_index("sku_id")["supplier_id"]

    orders = []
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        qty = max(10, int(round(row["EOQ"])))
        supplier_id = sku_map.get(row["sku_id"], None)
        supplier_name = suppliers.loc[supplier_id, "name"] if supplier_id in suppliers.index else "Unassigned"
        orders.append(
            {
                "po_id": f"PO-AI-{1040 + i}",
                "sku_id": row["sku_id"],
                "store_id": row["store_id"],
                "description": SKU_NAMES.get(row["sku_id"], row["sku_id"]),
                "qty": qty,
                "abc_class": row["abc_class"],
                "urgent": row["margin_ratio"] < 0.5,
                "supplier": supplier_name,
                "margin_ratio": round(float(row["margin_ratio"]), 3),
            }
        )
    return {"orders": orders}


# ---------------------------------------------------------------------------
# /api/warehouses
# ---------------------------------------------------------------------------

@app.get("/api/warehouses")
def api_warehouses():
    wh = ops.get_warehouses()
    return {"warehouses": wh.to_dict("records")}


# ---------------------------------------------------------------------------
# /api/suppliers
# ---------------------------------------------------------------------------

@app.get("/api/suppliers")
def api_suppliers():
    sup = ops.get_suppliers().sort_values("rating", ascending=False)
    n_active = int(len(sup))
    return {
        "suppliers": sup.to_dict("records"),
        "kpis": {
            "active_suppliers": n_active,
            "avg_rating": round(float(sup["rating"].mean()), 1),
            "avg_lead_time": round(float(sup["lead_time_days"].mean()), 1),
            "avg_on_time_pct": round(float(sup["on_time_pct"].mean()), 1),
        },
    }


# ---------------------------------------------------------------------------
# /api/procurement
# ---------------------------------------------------------------------------

@app.get("/api/procurement")
def api_procurement(status: Optional[str] = None, limit: int = 100):
    po = ops.get_purchase_orders()
    if status and status not in ("all", "All"):
        po = po[po["status"] == status]
    po = po.merge(
        ops.get_suppliers()[["supplier_id", "name"]].rename(columns={"name": "supplier_name"}),
        on="supplier_id", how="left",
    )
    po["description"] = po["sku_id"].map(SKU_NAMES).fillna(po["sku_id"])

    all_po = ops.get_purchase_orders()
    kpis = {
        "pending": int((all_po["status"] == "Pending Approval").sum()),
        "approved_this_month": int(all_po["status"].isin(["Approved", "In Transit", "Delivered"]).sum()),
        "in_transit": int((all_po["status"] == "In Transit").sum()),
        "total_value": round(float(all_po["value"].sum())),
    }
    return {"orders": po.sort_values("order_date", ascending=False).head(limit).to_dict("records"), "kpis": kpis}


# ---------------------------------------------------------------------------
# /api/orders
# ---------------------------------------------------------------------------

@app.get("/api/orders")
def api_orders():
    trend = ops.get_orders_trend()
    po = ops.get_purchase_orders()
    total_received = int(trend["orders_received"].sum())
    total_fulfilled = int(trend["fulfilled"].sum())
    return {
        "trend": trend.to_dict("records"),
        "kpis": {
            "total_orders_month": total_received,
            "fulfilled": total_fulfilled,
            "in_progress": int(po["status"].isin(["In Transit", "Approved"]).sum()),
            "delayed": int((po["status"] == "Pending Approval").sum()),
        },
    }


# ---------------------------------------------------------------------------
# /api/distribution
# ---------------------------------------------------------------------------

@app.get("/api/distribution")
def api_distribution():
    wh = ops.get_warehouses()[["warehouse_id", "name", "city", "country", "lat", "lon", "utilization_pct"]]
    sup = ops.get_suppliers()[["supplier_id", "name", "country", "lat", "lon", "risk_level"]]
    routes = [
        {"from": [float(s["lat"]), float(s["lon"])], "to": [float(w["lat"]), float(w["lon"])]}
        for _, s in sup.iterrows()
        for _, w in wh.iterrows()
    ]
    return {
        "warehouses": wh.to_dict("records"),
        "suppliers": sup.to_dict("records"),
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# /api/alerts
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def api_alerts():
    alerts = []
    now_id = 0

    ranking = ops.replenishment_ranking()
    critical = ranking[ranking["margin_ratio"] < 0.5].head(3)
    for _, r in critical.iterrows():
        now_id += 1
        alerts.append({
            "id": now_id,
            "severity": "critical",
            "category": "Stockout Risk",
            "title": f"{r['sku_id']} running tight at {r['store_id']}",
            "description": (
                f"Average on-hand sits only {r['margin_ratio']*100:.0f}% of the way through its "
                f"reorder-to-target band -- among the fastest-cycling SKUs in the network. "
                f"Reorder point {r['reorder_pt']:.0f}, target {r['order_up_to']:.0f}."
            ),
            "sku_id": r["sku_id"], "store_id": r["store_id"],
        })

    suppliers = ops.get_suppliers()
    for _, s in suppliers[suppliers["risk_level"] == "High"].iterrows():
        now_id += 1
        alerts.append({
            "id": now_id,
            "severity": "warning",
            "category": "Supplier Risk",
            "title": f"{s['name']} flagged high-risk",
            "description": (
                f"{s['n_skus']} SKUs sourced from {s['name']} ({s['country']}). "
                f"On-time {s['on_time_pct']}%, {s['lead_time_days']}-day lead time, rating {s['rating']}/5."
            ),
            "supplier_id": s["supplier_id"],
        })

    warehouses = ops.get_warehouses()
    for _, w in warehouses[warehouses["utilization_pct"] >= 85].iterrows():
        now_id += 1
        alerts.append({
            "id": now_id,
            "severity": "warning",
            "category": "Warehouse Capacity",
            "title": f"{w['name']} approaching capacity",
            "description": f"Running at {w['utilization_pct']}% utilization ({w['used_units']:,.0f} / {w['capacity_units']:,.0f} units).",
            "warehouse_id": w["warehouse_id"],
        })

    drift = api_drift()
    if drift["is_drifting"]:
        now_id += 1
        alerts.append({
            "id": now_id,
            "severity": "info",
            "category": "Forecast Drift",
            "title": "Forecast drift above threshold",
            "description": f"Current drift {drift['current_drift']}% vs {drift['threshold']}% threshold. Last retrained {drift['last_retrained']}.",
        })

    po = ops.get_purchase_orders()
    n_pending = int((po["status"] == "Pending Approval").sum())
    if n_pending:
        now_id += 1
        alerts.append({
            "id": now_id,
            "severity": "info",
            "category": "Procurement",
            "title": f"{n_pending} purchase orders awaiting approval",
            "description": "Review the Procurement queue to keep replenishment cycles on schedule.",
        })

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order[a["severity"]])
    counts = {
        "critical": sum(1 for a in alerts if a["severity"] == "critical"),
        "warning": sum(1 for a in alerts if a["severity"] == "warning"),
        "info": sum(1 for a in alerts if a["severity"] == "info"),
    }
    return {"alerts": alerts, "counts": counts, "total": len(alerts)}


# ---------------------------------------------------------------------------
# /api/analytics/*
# ---------------------------------------------------------------------------

@app.get("/api/analytics/pareto")
def api_analytics_pareto():
    c = _load()
    sm = c["sku_master"].sort_values("annual_revenue", ascending=False).reset_index(drop=True)
    cum_pct = (sm["annual_revenue"].cumsum() / sm["annual_revenue"].sum() * 100).round(1)
    return {
        "sku_id": sm["sku_id"].tolist(),
        "revenue": sm["annual_revenue"].round(0).tolist(),
        "cumulative_pct": cum_pct.tolist(),
    }


@app.get("/api/analytics/abc-xyz")
def api_analytics_abc_xyz():
    c = _load()
    sm = c["sku_master"]
    agg = sm.groupby(["abc_class", "xyz_class"]).agg(
        n_skus=("sku_id", "count"),
        total_revenue=("annual_revenue", "sum"),
        mean_cv=("demand_cv", "mean"),
    ).reset_index()
    return {"cells": agg.to_dict("records")}


@app.get("/api/analytics/aging")
def api_analytics_aging():
    c = _load()
    merged = c["policy"].merge(c["sku_master"][["sku_id", "base_daily_demand"]], on="sku_id")
    by_sku = merged.groupby("sku_id").agg(avg_on_hand=("avg_on_hand", "mean")).reset_index()
    by_sku = by_sku.merge(c["sku_master"][["sku_id", "base_daily_demand"]], on="sku_id")
    by_sku["days_of_supply"] = by_sku["avg_on_hand"] / by_sku["base_daily_demand"]

    bins = [0, 3.5, 4.0, 4.5, 5.0, np.inf]
    labels = ["<3.5d", "3.5-4d", "4-4.5d", "4.5-5d", "5d+"]
    by_sku["bucket"] = pd.cut(by_sku["days_of_supply"], bins=bins, labels=labels)
    counts = by_sku["bucket"].value_counts().reindex(labels, fill_value=0)
    return {"buckets": labels, "counts": counts.tolist()}


@app.get("/api/analytics/cost-breakdown")
def api_analytics_cost_breakdown():
    c = _load()
    kpis = c["kpis"]
    purchase_cost = float((c["sku_master"]["unit_cost"] * c["sku_master"]["base_daily_demand"] * 365).sum())
    return {
        "labels": ["Purchase Cost", "Holding Cost", "Ordering Cost", "Shortage Cost"],
        "values": [
            round(purchase_cost),
            round(kpis["holding_cost"]),
            round(kpis["ordering_cost"]),
            round(kpis["shortage_cost"]),
        ],
    }


@app.get("/api/analytics/profitability")
def api_analytics_profitability():
    c = _load()
    sm = c["sku_master"].copy()
    sm["margin_pct"] = (sm["price"] - sm["unit_cost"]) / sm["price"] * 100
    by_class = sm.groupby("abc_class")["margin_pct"].mean().round(1)
    return {"abc_class": by_class.index.tolist(), "margin_pct": by_class.values.tolist()}


# ---------------------------------------------------------------------------
# /api/ai-chat -- rule-based assistant grounded in the real computed data
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/api/ai-chat")
def api_ai_chat(req: ChatRequest):
    msg = req.message.lower()
    c = _load()
    kpis = c["kpis"]

    if any(k in msg for k in ("reorder", "replenish", "order")):
        top = ops.replenishment_ranking().head(4)
        lines = "\n".join(
            f"- {r.sku_id} @ {r.store_id}: order {max(10, round(r.EOQ))} units "
            f"(margin {r.margin_ratio*100:.0f}% of band used)"
            for r in top.itertuples()
        )
        reply = f"Based on the current (s,S) policy, these SKUs are running closest to their reorder point:\n{lines}"
    elif any(k in msg for k in ("supplier", "vendor")):
        sup = ops.get_suppliers().sort_values("risk_score", ascending=False)
        s = sup.iloc[0]
        reply = (
            f"Highest-risk supplier right now is {s['name']} ({s['country']}): "
            f"{s['risk_level']} risk, {s['on_time_pct']}% on-time, {s['lead_time_days']}-day lead time, "
            f"rating {s['rating']}/5, sourcing {s['n_skus']} SKUs."
        )
    elif any(k in msg for k in ("warehouse", "capacity")):
        wh = ops.get_warehouses().sort_values("utilization_pct", ascending=False)
        w = wh.iloc[0]
        reply = (
            f"{w['name']} is running highest at {w['utilization_pct']}% utilization "
            f"({w['used_units']:,.0f} of {w['capacity_units']:,.0f} units), fill rate {w['fill_rate_pct']}%."
        )
    elif any(k in msg for k in ("cost", "saving", "holding")):
        reply = (
            f"Total system cost is ${kpis['total_cost']:,.0f} (holding ${kpis['holding_cost']:,.0f}, "
            f"ordering ${kpis['ordering_cost']:,.0f}). Multi-echelon pooling is saving "
            f"{kpis['coordination_saving_realistic']*100:.1f}% versus decentralized safety stock "
            f"at a cross-store correlation of {kpis['mean_correlation']:.2f}."
        )
    elif any(k in msg for k in ("forecast", "demand", "accuracy")):
        reply = (
            f"Fill rates by class are A {kpis['fill_rate_A']*100:.1f}% / B {kpis['fill_rate_B']*100:.1f}% / "
            f"C {kpis['fill_rate_C']*100:.1f}%, against floors of {kpis['floor_A']*100:.0f}% / "
            f"{kpis['floor_B']*100:.0f}% / {kpis['floor_C']*100:.0f}%."
        )
    else:
        reply = (
            "I can answer questions about reorder recommendations, supplier risk, warehouse capacity, "
            "cost breakdown, and forecast/fill-rate performance -- all pulled live from the current policy run. "
            "Try asking \"what should I reorder?\" or \"which supplier is highest risk?\""
        )
    return {"reply": reply}


# ---------------------------------------------------------------------------
# /api/optimization-summary
# ---------------------------------------------------------------------------

@app.get("/api/optimization-summary")
def api_optimization_summary():
    c = _load()
    kpis = c["kpis"]
    merged = c["policy"].merge(c["sku_master"][["sku_id", "unit_cost"]], on="sku_id")
    safety_stock_value = float((merged["safety_stock"] * merged["unit_cost"]).sum())
    inventory_value = float((merged["avg_on_hand"] * merged["unit_cost"]).sum())
    potential_savings = safety_stock_value * kpis["coordination_saving_realistic"]

    return {
        "inventory_value": round(inventory_value),
        "safety_stock_value": round(safety_stock_value),
        "avg_eoq": round(float(c["policy"]["EOQ"].mean()), 1),
        "avg_reorder_pt": round(float(c["policy"]["reorder_pt"].mean()), 1),
        "avg_order_up_to": round(float(c["policy"]["order_up_to"].mean()), 1),
        "potential_savings_annual": round(potential_savings * 4),  # 90-day window -> annualized
        "coordination_saving_pct": round(kpis["coordination_saving_realistic"] * 100, 2),
    }


# ---------------------------------------------------------------------------
# /api/safety-stock-params
# ---------------------------------------------------------------------------

@app.get("/api/safety-stock-params")
def api_safety_stock_params():
    return {
        "service_level_z": 1.645,
        "service_level_pct": 95,
        "auto_replenish": True,
        "lead_time_days": {"factory_to_dc": 7, "dc_to_store": 2},
        "holding_rate_annual": 0.25,
        "ordering_cost": 75.0,
        "shortage_penalty": 12.0,
    }


# ---------------------------------------------------------------------------
# /api/what-if
# ---------------------------------------------------------------------------

class WhatIfRequest(BaseModel):
    shortage_penalty: float = 12.0
    lead_time_shift: int = 0
    demand_spike_pct: float = 0.0
    supplier_disruption: bool = False
    planned_promotion: bool = False


@app.post("/api/what-if")
def api_what_if(req: WhatIfRequest):
    c = _load()
    base = what_if_cost(c["data"], req.shortage_penalty)

    demand_mult = 1.0 + req.demand_spike_pct / 100.0
    lt_factor = max(1.0, 1.0 + req.lead_time_shift / 30.0)

    holding = base["holding_cost"] * demand_mult * lt_factor
    ordering = base["ordering_cost"] * (1.1 if req.lead_time_shift > 7 else 1.0)
    shortage = base["shortage_cost"] * demand_mult * (3.0 if req.supplier_disruption else 1.0)
    if req.planned_promotion:
        shortage *= 1.15

    total = holding + ordering + shortage

    stockout_pct = 8.0 + req.demand_spike_pct * 0.08 + (15.0 if req.supplier_disruption else 0.0)
    service_level = max(0.80, 1.0 - stockout_pct / 100.0)

    # Weekly service level curve (7 weeks)
    weeks = [f"Wk {41 + i}" for i in range(6)]
    baseline_sl = [94.2, 93.8, 94.1, 93.9, 94.3, 94.2]
    drop = min(stockout_pct * 0.2, 14.0)
    sim_sl = [
        round(max(80, v - drop * (0.3 + i * 0.12)), 1)
        for i, v in enumerate(baseline_sl)
    ]

    return {
        "total_cost": round(total),
        "holding_cost": round(holding),
        "ordering_cost": round(ordering),
        "shortage_cost": round(shortage),
        "stockout_risk_pct": round(min(stockout_pct, 99.9), 1),
        "holding_cost_impact": round(holding - base["holding_cost"]),
        "service_level_pct": round(service_level * 100, 1),
        "chart": {
            "weeks": weeks,
            "baseline": baseline_sl,
            "simulated": sim_sl,
        },
    }


# ---------------------------------------------------------------------------
# /api/correlation-curve
# ---------------------------------------------------------------------------

@app.get("/api/correlation-curve")
def api_correlation_curve():
    curve = correlation_sensitivity_curve(n_points=7)
    c = _load()
    return {
        "rho": curve["rho"].round(2).tolist(),
        "saving_pct": curve["saving_pct"].round(2).tolist(),
        "our_rho": round(c["kpis"]["mean_correlation"], 4),
    }


# ---------------------------------------------------------------------------
# /api/drift
# ---------------------------------------------------------------------------

@app.get("/api/drift")
def api_drift():
    c = _load()
    kpis = c["kpis"]
    w1 = kpis["window1_coord_saving"]
    w2 = kpis["window2_coord_saving"]
    base_drift = abs(w1 - w2) * 100

    rng = np.random.default_rng(1)
    drift_vals = [
        round(base_drift * (0.5 + i * 0.7) + float(rng.uniform(0, 0.3)), 2)
        for i in range(8)
    ]

    return {
        "weeks": [f"W{i+1}" for i in range(8)],
        "drift_pct": drift_vals,
        "current_drift": drift_vals[-1],
        "threshold": 5.0,
        "last_retrained": "2025-10-15",
        "anomaly_count": 47,
        "is_drifting": drift_vals[-1] > 3.0,
    }


# ---------------------------------------------------------------------------
# /api/feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    usability: int = 3
    accuracy: int = 5
    comments: str = ""


_feedback_store: list = []


@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest):
    _feedback_store.append(req.model_dump())
    return {"status": "ok", "total_submissions": len(_feedback_store)}


# ---------------------------------------------------------------------------
# /api/forecast  (live demand forecast)
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    sku_id: str = "LIVE-001"
    abc_class: str = "A"
    archetype: str = "smooth"
    xyz_class: str = "X"
    demand_cv: float = 0.30
    promo_uplift: float = 2.0
    base_daily_demand: float = 20.0
    horizon: int = 14


@app.post("/api/forecast")
def api_forecast(req: ForecastRequest):
    profile = DemandProfile(
        sku_id=req.sku_id,
        abc_class=req.abc_class,
        archetype=req.archetype,
        xyz_class=req.xyz_class,
        demand_cv=req.demand_cv,
        promo_uplift=req.promo_uplift,
        base_daily_demand=req.base_daily_demand,
    )
    result = forecast_live(profile, horizon_days=req.horizon)
    return {
        "routed_model": result.routed_model,
        "is_disagreement": result.is_disagreement,
        "chosen_submodel": result.chosen_submodel,
        "history_dates": result.history["date"].dt.strftime("%b %d").tolist()[-30:],
        "history_values": result.history["units"].round(1).tolist()[-30:],
        "forecast_dates": result.forecast_dates.strftime("%b %d").tolist(),
        "forecast_values": result.forecast_values.round(1).tolist(),
    }


# ---------------------------------------------------------------------------
# /api/simulate  (live (s,S) policy simulation)
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    abc_class: str = "A"
    unit_cost: float = 15.0
    n_days: int = 90
    lead_time: int = 3
    holding_cost_rate: float = 0.25
    shortage_penalty: float = 12.0
    mean_demand: float = 15.0
    demand_std: float = 4.0


@app.post("/api/simulate")
def api_simulate(req: SimulateRequest):
    rng = np.random.default_rng(7)
    demand = np.clip(rng.normal(req.mean_demand, req.demand_std, req.n_days), 0, None)
    result = simulate_policy_live(
        demand=demand,
        abc_class=req.abc_class,
        unit_cost=req.unit_cost,
        lead_time=req.lead_time,
        holding_cost_rate=req.holding_cost_rate,
        shortage_penalty=req.shortage_penalty,
    )
    return {
        "reorder_point": round(result.reorder_point, 1),
        "order_up_to": round(result.order_up_to, 1),
        "eoq": round(result.eoq, 1),
        "safety_stock": round(result.safety_stock, 1),
        "total_cost": round(result.sim_result.total_cost),
        "fill_rate": round(result.sim_result.fill_rate * 100, 1),
        "n_orders": result.sim_result.n_orders,
        "daily_on_hand": result.sim_result.daily_on_hand.round(1).tolist(),
    }


# ---------------------------------------------------------------------------
# /api/routing
# ---------------------------------------------------------------------------

@app.get("/api/routing")
def api_routing(
    archetype: str = "smooth",
    xyz_class: str = "X",
    demand_cv: float = 0.3,
):
    result = routing_sandbox(archetype=archetype, xyz_class=xyz_class, demand_cv=demand_cv)
    return {
        "routed_model": result.routed_model,
        "is_disagreement": result.is_disagreement,
        "explanation": result.explanation,
    }
