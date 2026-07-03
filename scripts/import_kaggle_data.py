"""
import_kaggle_data.py
=====================
Downloads the Kaggle dataset and transforms it into the format expected by the project.
Replaces the old synthetic generation pipeline.

Outputs:
  - data/raw/demand.csv
  - data/raw/sku_master.csv
  - data/raw/network.json
"""

import json
import sys
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
CONFIGS = ROOT / "configs"


def main():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("ziya07/high-dimensional-supply-chain-inventory-dataset")
    csv_path = Path(path) / "supply_chain_dataset1.csv"

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # 1. Transform to demand.csv
    # Expected columns: date, store_id, sku_id, units, on_promo
    demand = df[["Date", "Warehouse_ID", "SKU_ID", "Units_Sold", "Promotion_Flag"]].copy()
    demand.columns = ["date", "store_id", "sku_id", "units", "on_promo"]
    demand["date"] = pd.to_datetime(demand["date"])
    demand["on_promo"] = demand["on_promo"].astype(bool)

    # 2. Extract sku_master.csv
    sku_stats = df.groupby("SKU_ID").agg(
        Units_Sold=("Units_Sold", "sum"),
        Unit_Cost=("Unit_Cost", "mean"),
        Unit_Price=("Unit_Price", "mean")
    ).reset_index()

    days = demand["date"].nunique()
    years = days / 365.25 if days > 0 else 1.0

    sku_master = pd.DataFrame()
    sku_master["sku_id"] = sku_stats["SKU_ID"]
    sku_master["unit_cost"] = sku_stats["Unit_Cost"].round(2)
    sku_master["price"] = sku_stats["Unit_Price"].round(2)
    sku_master["base_daily_demand"] = (sku_stats["Units_Sold"] / days).round(3)
    sku_master["annual_revenue"] = (sku_stats["Units_Sold"] * sku_master["price"] / years).round(2)

    # ABC Classification
    with open(CONFIGS / "business_rules.yaml") as fh:
        br = yaml.safe_load(fh)

    sku_master = sku_master.sort_values("annual_revenue", ascending=False).reset_index(drop=True)
    cum_share = sku_master["annual_revenue"].cumsum() / sku_master["annual_revenue"].sum()

    abc_cuts = br["skus"]["abc_cutpoints"]
    sku_master["abc_class"] = np.where(cum_share <= abc_cuts["A"], "A",
                                       np.where(cum_share <= abc_cuts["B"], "B", "C"))

    # XYZ Classification
    daily = demand.groupby(["sku_id", "date"], observed=True)["units"].sum().reset_index()
    stats = daily.groupby("sku_id", observed=True)["units"].agg(["mean", "std"]).reset_index()
    stats["demand_cv"] = (stats["std"] / stats["mean"].replace(0, np.nan)).fillna(0.0)

    xyz_cuts = br["skus"]["xyz_cutpoints"]
    stats["xyz_class"] = np.where(stats["demand_cv"] < float(xyz_cuts["X"]), "X",
                                  np.where(stats["demand_cv"] < float(xyz_cuts["Y"]), "Y", "Z"))

    sku_master = sku_master.merge(stats[["sku_id", "demand_cv", "xyz_class"]], on="sku_id", how="left")
    sku_master["demand_cv"] = sku_master["demand_cv"].round(3)

    # Archetype Classification
    zero_frac = demand.assign(is_zero=demand["units"].eq(0)).groupby("sku_id", observed=True)["is_zero"].mean().reset_index()
    sku_master = sku_master.merge(zero_frac, on="sku_id", how="left")
    sku_master["archetype"] = np.where(sku_master["is_zero"] > 0.3, "intermittent", "smooth")
    sku_master = sku_master.drop(columns=["is_zero"])

    # Pipeline Fallbacks
    sku_master["trend_growth"] = 0.0
    sku_master["annual_amplitude"] = 0.0
    sku_master["season_phase"] = 0.0
    sku_master["occurrence_prob"] = 0.5
    sku_master["lump_size_mean"] = sku_master["base_daily_demand"]

    # 3. Network JSON & Business Rules Update
    wh_ids = df["Warehouse_ID"].unique()
    stores = []
    total_demand = demand["units"].sum()
    for wh in wh_ids:
        wh_demand = demand[demand["store_id"] == wh]["units"].sum()
        stores.append({
            "id": str(wh),
            "name": f"Kaggle {wh}",
            "size_multiplier": float(wh_demand / total_demand * len(wh_ids)) if total_demand > 0 else 1.0,
            "capacity_units": 15000
        })

    network_data = {
        "factory_id": "F1",
        "dc_id": "DC1",
        "dc_capacity": 60000,
        "stores": stores
    }

    # Update business_rules.yaml
    br["network"]["stores"] = stores
    br["skus"]["n_skus"] = len(sku_master)
    with open(CONFIGS / "business_rules.yaml", "w") as fh:
        yaml.safe_dump(br, fh, sort_keys=False)

    # 4. Save
    RAW.mkdir(parents=True, exist_ok=True)
    demand.to_csv(RAW / "demand.csv", index=False)
    sku_master.to_csv(RAW / "sku_master.csv", index=False)
    with open(RAW / "network.json", "w") as fh:
        json.dump(network_data, fh, indent=2)

    print(f"Imported {len(demand)} demand records.")
    print(f"Imported {len(sku_master)} SKUs.")
    print(f"Imported {len(stores)} Warehouses.")
    print(f"Done. Saved to {RAW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
