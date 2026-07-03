"""
dashboard_data.py
==================
Single source of truth for everything the Phase 7 dashboard displays.
Both the Streamlit app (app.py) and the verified static HTML preview consume
this module, so the two artifacts can never silently disagree -- if this
module is correct (and it is unit-tested), both deliverables are correct.

Loads the persisted results from Phases 4-6 (data/processed/*.json, *.csv)
rather than re-running any pipeline, so the dashboard is fast and always
reflects exactly what those phases verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # dashboard/utils/this_file.py -> repo root
PROCESSED = REPO_ROOT / "data" / "processed"
CONFIGS = REPO_ROOT / "configs"


def load_all() -> dict:
    """Load every persisted result file the dashboard needs. Raises clearly if any is missing."""
    required = {
        "baseline_summary": PROCESSED / "baseline_summary.json",
        "optimization_summary": PROCESSED / "optimization_summary.json",
        "backtest_summary": PROCESSED / "backtest_summary.json",
        "business_rules": CONFIGS / "business_rules.yaml",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dashboard requires these phase outputs to exist first: " + ", ".join(missing)
        )

    data = {}
    data["baseline"] = json.load(open(required["baseline_summary"]))
    data["optimization"] = json.load(open(required["optimization_summary"]))
    data["backtest"] = json.load(open(required["backtest_summary"]))
    data["business_rules"] = yaml.safe_load(open(required["business_rules"]))
    data["baseline_policy_df"] = pd.read_csv(PROCESSED / "baseline_policy_results.csv")
    data["comparison_df"] = pd.read_csv(PROCESSED / "final_comparison_table.csv")
    return data


def compute_kpis(data: dict) -> dict:
    """Derive the headline KPI numbers the dashboard's top row displays."""
    b = data["baseline"]
    o = data["optimization"]
    bt = data["backtest"]

    fill_by_abc = b["fill_rate_by_abc"]
    floors = data["business_rules"]["service_floors"]

    return {
        "total_cost": b["total_cost"],
        "holding_cost": b["holding_cost"],
        "ordering_cost": b["ordering_cost"],
        "shortage_cost": b["shortage_cost"],
        "fill_rate_A": fill_by_abc["A"],
        "fill_rate_B": fill_by_abc["B"],
        "fill_rate_C": fill_by_abc["C"],
        "floor_A": floors["A"],
        "floor_B": floors["B"],
        "floor_C": floors["C"],
        "safety_stock_pct": o["safety_stock_pct_of_onhand"],
        "coordination_saving_realistic": o["coordination_saving_realistic"],
        "coordination_saving_upper_bound": o["coordination_saving_upper_bound"],
        "mean_correlation": o["mean_cross_store_correlation"],
        "window1_coord_saving": bt["window1"]["coord_saving"],
        "window2_coord_saving": bt["window2"]["coord_saving"],
        "shortage_penalty_spread_pct": bt["total_cost_spread_pct"],
    }


def what_if_cost(data: dict, shortage_penalty: float) -> dict:
    """
    Re-derive total cost under a custom shortage-penalty value, using the
    EXACT linear relationship verified in Phase 6 (test_rolling_backtest.py:
    test_shortage_cost_scales_linearly_with_penalty). This lets the dashboard
    slider answer instantly with a formula instead of re-running the full
    simulation pipeline on every drag event.
    """
    ladder = data["backtest"]["shortage_penalty_ladder"]
    moderate = next(r for r in ladder if r["scenario"] == "moderate")
    base_penalty = moderate["penalty"]
    base_shortage = moderate["shortage_cost"]

    scaled_shortage = base_shortage * (shortage_penalty / base_penalty)
    new_total = moderate["holding_cost"] + moderate["ordering_cost"] + scaled_shortage

    return {
        "shortage_penalty": shortage_penalty,
        "shortage_cost": scaled_shortage,
        "holding_cost": moderate["holding_cost"],
        "ordering_cost": moderate["ordering_cost"],
        "total_cost": new_total,
    }


def penalty_ladder_df(data: dict) -> pd.DataFrame:
    """The 3-scenario ladder as a DataFrame, for charting."""
    return pd.DataFrame(data["backtest"]["shortage_penalty_ladder"]).sort_values("penalty")


def correlation_sensitivity_curve(n_points: int = 21) -> pd.DataFrame:
    """
    Re-derive the rho -> pooling-saving curve from Phase 5's verified math
    (src/optimization/multi_echelon.py), for the dashboard's explanatory chart.
    Uses representative sigma proportions; the SHAPE of this curve is what
    matters for the explanation, and Phase 5 proved (test_multi_echelon.py)
    the ratio is invariant to the sigma scale chosen.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from src.optimization.multi_echelon import decentralized_safety_stock, pooled_safety_stock
    from src.optimization.policy_math import service_z

    z = service_z(0.95)
    sig = [1.0, 0.7, 0.45]  # representative store-size proportions (Charter network)
    rhos = np.linspace(0, 1, n_points)
    savings = []
    for rho in rhos:
        C = np.full((3, 3), rho)
        np.fill_diagonal(C, 1.0)
        dec = decentralized_safety_stock(z, 2, sig)
        pooled = pooled_safety_stock(z, 2, sig, corr_matrix=C)
        savings.append(1 - pooled / dec)
    return pd.DataFrame({"rho": rhos, "saving_pct": [s * 100 for s in savings]})
