"""
router.py
=========
Decides which forecasting model each SKU gets, per the rule confirmed with
the user: ARCHETYPE is the primary routing signal (it's what Phase 2
statistically validated via the Welch t-test on zero-day fraction), with
XYZ class used as an independent CROSS-CHECK, not a silent override.

Routing rule:
    archetype == 'intermittent'  -> Croston's method
    archetype == 'smooth'        -> Fourier seasonal model (default)
    archetype == 'smooth' AND xyz_class == 'Z'
                                  -> flagged disagreement; route to GBM
                                     instead (GBM handles irregular patterns
                                     more robustly than a pure seasonal model)

Disagreements (archetype says one thing, XYZ says another) are collected
and surfaced explicitly -- on Phase 1's actual generated data there were
zero such cases (verified), but the logic must not silently assume that
holds for every future re-generation with a different seed.
"""

from __future__ import annotations

import pandas as pd


def route_skus(sku_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (routing_table, disagreements).
    routing_table has columns: sku_id, abc_class, archetype, xyz_class, model
    disagreements is a subset of routing_table flagged as edge cases.
    """
    df = sku_master[["sku_id", "abc_class", "archetype", "xyz_class", "demand_cv"]].copy()

    def assign(row) -> str:
        if row["archetype"] == "intermittent":
            return "croston"
        # archetype == 'smooth'
        if row["xyz_class"] == "Z":
            return "gbm"  # disagreement case: smooth label but erratic CV
        return "fourier_seasonal"

    df["model"] = df.apply(assign, axis=1)

    disagreement_mask = (
        ((df["archetype"] == "smooth") & (df["xyz_class"] == "Z"))
        | ((df["archetype"] == "intermittent") & (df["xyz_class"] == "X"))
    )
    disagreements = df.loc[disagreement_mask].copy()
    return df, disagreements
