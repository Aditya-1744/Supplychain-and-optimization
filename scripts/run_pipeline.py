"""
run_pipeline.py
===============
Single end-to-end pipeline runner. Executes every analysis phase in order and
writes all of data/processed/, with NO Jupyter dependency -- it calls the same
importable functions in src/ that the notebooks call, so the notebooks remain
useful for narrative/exploration while THIS script is the reproducible,
automatable path the dashboard and CI depend on.

This is the fix for the two most common failure modes:
  * "the dashboard crashes on a fresh checkout" -- because data/processed/* is
    gitignored and nothing regenerated it. Run this once and it's populated.
  * "phases fail when run out of order" -- notebooks each need the prior phase's
    outputs; this script runs them in the correct order in one process.

Usage:
    python scripts/run_pipeline.py            # run everything
    python scripts/run_pipeline.py --force    # regenerate even if outputs exist
    python scripts/run_pipeline.py --quiet     # less console output

Idempotent: safe to re-run. Skips regeneration if all outputs already exist
(unless --force), so the launcher can call it cheaply on every startup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.router import route_skus
from src.models.seasonal_model import FourierSeasonalForecaster
from src.models.gbm_model import GBMForecaster
from src.models.croston import CrostonOrBaseline
from src.optimization.policy_math import service_z, safety_stock, eoq, reorder_point, order_up_to
from src.optimization.multi_echelon import decompose_sku
from src.optimization.des_engine import simulate_sS
from src.evaluation.rolling_backtest import run_window

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"

REQUIRED_OUTPUTS = [
    PROCESSED / "baseline_summary.json",
    PROCESSED / "baseline_policy_results.csv",
    PROCESSED / "optimization_summary.json",
    PROCESSED / "backtest_summary.json",
    PROCESSED / "final_comparison_table.csv",
]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


# --------------------------------------------------------------------- #
# Phase 1: synthetic data (delegates to the existing generator script)
# --------------------------------------------------------------------- #
def ensure_raw_data(force: bool, quiet: bool) -> None:
    needed = [RAW / "demand.csv", RAW / "sku_master.csv"]
    if not force and all(p.exists() for p in needed):
        log("  [1/4] raw data present -- skipping generation", quiet)
        return
    log("  [1/4] generating synthetic data (Phase 1)...", quiet)
    # Import and call the generator's main() directly rather than shelling out,
    # so a failure surfaces as a real traceback instead of a non-zero exit code.
    import importlib.util
    spec = importlib.util.spec_from_file_location("import_kaggle_data", ROOT / "scripts" / "import_kaggle_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# --------------------------------------------------------------------- #
# Phases 4-6: baseline, optimization, backtest -- all via run_window()
# --------------------------------------------------------------------- #
def run_analysis(quiet: bool) -> None:
    demand = pd.read_csv(RAW / "demand.csv", parse_dates=["date"])
    skus = pd.read_csv(RAW / "sku_master.csv")
    br = yaml.safe_load(open(CONFIGS / "business_rules.yaml"))
    mp = yaml.safe_load(open(CONFIGS / "model_params.yaml"))
    holdout = int(mp["simulation"]["holdout_days"])

    # Two non-overlapping rolling-origin windows (Phase 6 design).
    last = demand["date"].max()
    w2_start = last - pd.Timedelta(days=holdout - 1)
    w1_end = w2_start - pd.Timedelta(days=1)
    w1_start = w1_end - pd.Timedelta(days=holdout - 1)

    log("  [2/4] running baseline + optimization on the headline window (Phases 4-5)...", quiet)
    w2 = run_window(demand, skus, br, w2_start, last, "Window 2 (headline)")

    log("  [3/4] running the rolling-origin backtest window (Phase 6)...", quiet)
    w1 = run_window(demand, skus, br, w1_start, w1_end, "Window 1")

    log("  [4/4] running the shortage-penalty sensitivity ladder...", quiet)
    ladder_rows = []
    for scenario, penalty in br["costs"]["shortage_penalty_scenarios"].items():
        r = run_window(demand, skus, br, w2_start, last, scenario, shortage_penalty_override=penalty)
        ladder_rows.append({
            "scenario": scenario, "penalty": penalty,
            "total_cost": r.baseline_total_cost, "holding_cost": r.baseline_holding_cost,
            "ordering_cost": r.baseline_ordering_cost, "shortage_cost": r.baseline_shortage_cost,
        })

    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ---- baseline_summary.json (Phase 4) ----
    baseline_summary = {
        "total_cost": w2.baseline_total_cost,
        "holding_cost": w2.baseline_holding_cost,
        "ordering_cost": w2.baseline_ordering_cost,
        "shortage_cost": w2.baseline_shortage_cost,
        "fill_rate_by_abc": w2.baseline_fill_by_abc,
        "holdout_days": holdout,
    }
    (PROCESSED / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2))

    # ---- baseline_policy_results.csv (per store-SKU detail, recomputed) ----
    _write_baseline_policy_csv(demand, skus, br, w2_start, last, holdout)

    # ---- optimization_summary.json (Phase 5) ----
    ladder_df = pd.DataFrame(ladder_rows)
    optimization_summary = {
        "safety_stock_pct_of_onhand": w2.ss_pct_of_onhand,
        "coordination_saving_realistic": w2.coordination_saving_realistic,
        "coordination_saving_upper_bound": w2.coordination_saving_upper_bound,
        "mean_cross_store_correlation": w2.mean_cross_store_corr,
        # Derived dollar figures the dashboard/report cite:
        "ss_holding_cost_decentralized": None,  # filled by detailed pooling pass below
        "ss_holding_cost_pooled_realistic": None,
        "ss_holding_cost_pooled_independent": None,
        "coordination_saving_pct_of_total_holding": None,
    }
    _fill_pooling_dollars(optimization_summary, demand, skus, br, w2_start, last, holdout)
    (PROCESSED / "optimization_summary.json").write_text(json.dumps(optimization_summary, indent=2))

    # ---- backtest_summary.json (Phase 6) ----
    moderate_total = ladder_df.loc[ladder_df.scenario == "moderate", "total_cost"].iloc[0]
    spread = (ladder_df.total_cost.max() - ladder_df.total_cost.min()) / moderate_total
    backtest_summary = {
        "window1": {"coord_saving": w1.coordination_saving_realistic, "ss_pct": w1.ss_pct_of_onhand},
        "window2": {"coord_saving": w2.coordination_saving_realistic, "ss_pct": w2.ss_pct_of_onhand},
        "shortage_penalty_ladder": ladder_rows,
        "total_cost_spread_pct": float(spread),
    }
    (PROCESSED / "backtest_summary.json").write_text(json.dumps(backtest_summary, indent=2))

    # ---- final_comparison_table.csv ----
    comparison = pd.DataFrame({
        "Metric": [
            "Total cost (90d, moderate)", "Holding cost", "Ordering cost", "Shortage cost",
            "Safety stock (% of on-hand)", "Coordination saving (realistic)",
            "Coordination saving (upper bound)", "Cross-window stability",
            "Shortage-penalty cost spread",
        ],
        "Result": [
            f"${w2.baseline_total_cost:,.0f}", f"${w2.baseline_holding_cost:,.0f}",
            f"${w2.baseline_ordering_cost:,.0f}", f"${w2.baseline_shortage_cost:,.0f}",
            f"{w2.ss_pct_of_onhand:.1%}", f"{w2.coordination_saving_realistic:.1%}",
            f"{w2.coordination_saving_upper_bound:.1%} (not headline)",
            f"{w1.coordination_saving_realistic:.1%} vs {w2.coordination_saving_realistic:.1%}",
            f"{spread:.1%} across the penalty ladder",
        ],
    })
    comparison.to_csv(PROCESSED / "final_comparison_table.csv", index=False)


def _write_baseline_policy_csv(demand, skus, br, test_start, test_end, holdout):
    """Per store-SKU policy detail (the granular Phase 4 output the dashboard reads)."""
    costs, floors = br["costs"], br["service_floors"]
    LT = br["assumptions"]["lead_time_days"]["dc_to_store"]
    MOQ, PACK = br["constraints"]["min_order_qty"], br["constraints"]["pack_size"]
    stores = {s["id"]: s for s in br["network"]["stores"]}
    store_ids = list(stores.keys())
    routing, _ = route_skus(skus)
    sku_info = skus.set_index("sku_id")
    full_idx = pd.date_range(demand.date.min(), test_end, freq="D")
    n_train = (full_idx < test_start).sum()

    net_daily = (demand.groupby(["sku_id", "date"], observed=True)
                 .agg(units=("units", "sum"), on_promo=("on_promo", "max")).reset_index())

    def resid_std(sid):
        sub = net_daily[net_daily.sku_id == sid].set_index("date").reindex(full_idx)
        sub["units"] = sub["units"].fillna(0); sub["on_promo"] = sub["on_promo"].fillna(False)
        sub = sub.reset_index().rename(columns={"index": "date"})
        tr, te = sub[sub.date < test_start], sub[(sub.date >= test_start)]
        model = routing.set_index("sku_id").loc[sid, "model"]
        yv = te["units"].to_numpy()
        if model == "fourier_seasonal":
            m = FourierSeasonalForecaster().fit(tr["date"], tr["units"].to_numpy(), tr["on_promo"].to_numpy())
            pred = m.predict(te["date"], te["on_promo"].to_numpy())
        elif model == "gbm":
            m = GBMForecaster().fit(tr[["date", "units", "on_promo"]]); pred = m.predict(te["date"], te["on_promo"].to_numpy())
        else:
            m = CrostonOrBaseline().fit(tr["units"].to_numpy()); pred = m.predict(len(yv))
        return float(np.std(yv - pred))

    resid = {sid: resid_std(sid) for sid in routing.sku_id}
    share = {st: stores[st]["size_multiplier"] / sum(s["size_multiplier"] for s in stores.values()) for st in store_ids}

    rows = []
    for st in store_ids:
        for sid in routing.sku_id:
            abc = sku_info.loc[sid, "abc_class"]; unit_cost = float(sku_info.loc[sid, "unit_cost"])
            z = service_z(floors[abc])
            series = (demand[(demand.store_id == st) & (demand.sku_id == sid)]
                      .groupby("date")["units"].sum().reindex(full_idx).fillna(0).to_numpy())
            train_d, sim_d = series[:n_train], series[n_train:]
            mean_daily = float(train_d.mean())
            h = unit_cost * costs["holding_rate_annual"]
            ss = safety_stock(z, resid[sid] * share[st], LT)
            Q = eoq(mean_daily * 365.0, costs["ordering_cost"], h)
            s = reorder_point(mean_daily, LT, ss); S = order_up_to(s, Q)
            r = simulate_sS(sim_d, s, S, LT, unit_cost, costs["holding_rate_annual"],
                            costs["ordering_cost"], costs["shortage_penalty_per_unit"], MOQ, PACK)
            rows.append({"store_id": st, "sku_id": sid, "abc_class": abc, "safety_stock": ss,
                         "reorder_pt": s, "order_up_to": S, "EOQ": Q,
                         "holding_cost": r.total_holding_cost, "ordering_cost": r.total_ordering_cost,
                         "shortage_cost": r.total_shortage_cost, "total_cost": r.total_cost,
                         "fill_rate": r.fill_rate, "avg_on_hand": r.avg_on_hand})
    pd.DataFrame(rows).to_csv(PROCESSED / "baseline_policy_results.csv", index=False)


def _fill_pooling_dollars(summary, demand, skus, br, test_start, test_end, holdout):
    """Compute the three-regime safety-stock dollar figures for the optimization summary."""
    costs, floors = br["costs"], br["service_floors"]
    LT = br["assumptions"]["lead_time_days"]["dc_to_store"]
    stores = {s["id"]: s for s in br["network"]["stores"]}
    store_ids = list(stores.keys())
    routing, _ = route_skus(skus)
    sku_info = skus.set_index("sku_id")
    full_idx = pd.date_range(demand.date.min(), test_end, freq="D")
    daily_h = costs["holding_rate_annual"] / 365.0

    piv = demand.groupby(["date", "store_id"])["units"].sum().unstack("store_id")[store_ids]
    CORR = piv.corr().to_numpy()

    net_daily = (demand.groupby(["sku_id", "date"], observed=True)
                 .agg(units=("units", "sum"), on_promo=("on_promo", "max")).reset_index())

    def resid_std(sid):
        sub = net_daily[net_daily.sku_id == sid].set_index("date").reindex(full_idx)
        sub["units"] = sub["units"].fillna(0); sub["on_promo"] = sub["on_promo"].fillna(False)
        sub = sub.reset_index().rename(columns={"index": "date"})
        tr, te = sub[sub.date < test_start], sub[sub.date >= test_start]
        model = routing.set_index("sku_id").loc[sid, "model"]; yv = te["units"].to_numpy()
        if model == "fourier_seasonal":
            m = FourierSeasonalForecaster().fit(tr["date"], tr["units"].to_numpy(), tr["on_promo"].to_numpy())
            pred = m.predict(te["date"], te["on_promo"].to_numpy())
        elif model == "gbm":
            m = GBMForecaster().fit(tr[["date", "units", "on_promo"]]); pred = m.predict(te["date"], te["on_promo"].to_numpy())
        else:
            m = CrostonOrBaseline().fit(tr["units"].to_numpy()); pred = m.predict(len(yv))
        return float(np.std(yv - pred))

    resid = {sid: resid_std(sid) for sid in routing.sku_id}
    share = {st: stores[st]["size_multiplier"] / sum(s["size_multiplier"] for s in stores.values()) for st in store_ids}

    dec = corr = indep = 0.0
    for sid in routing.sku_id:
        abc = sku_info.loc[sid, "abc_class"]; uc = float(sku_info.loc[sid, "unit_cost"])
        sig = [resid[sid] * share[st] for st in store_ids]
        d = decompose_sku(sid, abc, sig, floors[abc], LT, CORR)
        hcf = uc * daily_h * holdout
        dec += d.ss_decentralized * hcf; corr += d.ss_pooled_correlated * hcf; indep += d.ss_pooled_independent * hcf

    baseline = json.loads((PROCESSED / "baseline_summary.json").read_text())
    summary["ss_holding_cost_decentralized"] = dec
    summary["ss_holding_cost_pooled_realistic"] = corr
    summary["ss_holding_cost_pooled_independent"] = indep
    summary["coordination_saving_pct_of_total_holding"] = (dec - corr) / baseline["holding_cost"]


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #
def main(force: bool = False, quiet: bool = False) -> None:
    t0 = time.time()
    if not force and all(p.exists() for p in REQUIRED_OUTPUTS):
        log("All pipeline outputs already present. Use --force to regenerate.", quiet)
        return

    log("Running full pipeline (Phases 1-6)...", quiet)
    ensure_raw_data(force, quiet)
    run_analysis(quiet)

    missing = [p.name for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        raise RuntimeError(f"Pipeline finished but these outputs are missing: {missing}")
    log(f"\nPipeline complete in {time.time()-t0:.1f}s. All outputs written to data/processed/.", quiet)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the full supply-chain optimization pipeline.")
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    ap.add_argument("--quiet", action="store_true", help="reduce console output")
    args = ap.parse_args()
    main(force=args.force, quiet=args.quiet)
