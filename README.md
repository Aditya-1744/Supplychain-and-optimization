# Supply Chain & Inventory Optimization

Multi-echelon inventory optimization on real-world supply-chain data. The goal:
**minimize total network holding cost subject to ABC-tiered service-level floors**,
and show that coordinated multi-echelon replenishment beats a decentralized
EOQ + safety-stock baseline. See `PROJECT_CHARTER.md` for the original scope
(written before the data-source pivot described below — treat this README as
the current source of truth for architecture and results).

Network: **1 Factory → 1 DC → 5 Warehouses**, 50 SKUs, 365 days of daily demand.

---

## Status

| Phase | State |
|-------|-------|
| **1. Setup & data engine** | ✅ **Done** (see note below — now real data, not synthetic) |
| **2. EDA & ABC/XYZ classification** | ✅ **Done** |
| **3. Forecasting layer** | ✅ **Done** |
| **4. Baseline policy** | ✅ **Done** |
| **5. Multi-echelon optimization** | ✅ **Done** |
| **6. Backtest & comparison** | ✅ **Done** |
| **7. Dashboard** | ✅ **Done** — two dashboards now exist (Streamlit legacy + FastAPI web app) |
| **8. Report & handover** | ✅ **Done** |

## ⚠️ Note on data source (read this first)

The charter and the original build of this project used a synthetic NumPy
demand generator (`scripts/generate_data.py`, described in old commits). At
some point the project **pivoted to a real dataset**:
[`ziya07/high-dimensional-supply-chain-inventory-dataset`](https://www.kaggle.com/) on Kaggle,
downloaded via `kagglehub` and reshaped by `scripts/import_kaggle_data.py` into
the same `demand.csv` / `sku_master.csv` / `network.json` schema the rest of
the pipeline expects. `scripts/generate_data.py` no longer exists in this repo.

Practical consequences of the pivot, reported honestly (in keeping with the
rest of this document):

- The network is now **5 warehouses** (`WH_1`…`WH_5`, taken from the dataset's
  `Warehouse_ID` field), not the original 3 synthetic stores. Factory/DC are
  still a conceptual overlay (`F1` / `DC1`) — the raw dataset has no
  factory/DC echelon, so those two levels are structural assumptions, not
  observed data.
- ABC/XYZ classification, archetype (smooth vs. intermittent), and demand-CV
  are all **recomputed from the real data** by `import_kaggle_data.py` — they
  are not carried over from the synthetic generator.
- A few SKU-level fields that only made sense for the synthetic generator
  (`trend_growth`, `annual_amplitude`, `season_phase`, `occurrence_prob`) are
  now constant fallback values, kept only so downstream code that expects
  those columns doesn't break. They carry no real signal.
- **The specific quantitative findings quoted in Phases 2–6 below (Croston
  win-rate, 18.6% MAE improvement, ~11% coordination saving, etc.) come from
  the notebooks, which were authored and run against the original synthetic
  dataset.** The notebooks have not been re-executed against the Kaggle
  import. The pipeline script (`scripts/run_pipeline.py`) *has* been re-run
  against it — see **[Current results on the live (Kaggle) pipeline](#current-results-on-the-live-kaggle-pipeline)**
  below for what the numbers look like today, and both dashboards read from
  that live pipeline output, not from the notebooks.
- `kagglehub` is required to regenerate raw data from scratch and is not yet
  pinned in `requirements.txt` — install it separately (`pip install
  kagglehub`) or add it yourself before running `scripts/import_kaggle_data.py`.
  If `data/raw/*.csv` already exist (they do in a fresh checkout of this
  workspace), the pipeline skips re-downloading, so most workflows never hit
  this dependency.

---

## Quickstart

Two dashboards live in this repo. **The FastAPI one (`InventoryPro`) is the
current, actively developed one** — more pages, live inference, and no
Streamlit runtime quirks. The Streamlit one is kept working but is the
earlier build.

### Option A — FastAPI web dashboard (recommended)

```bash
pip install -r requirements.txt
python start_web.py
```

Opens `http://localhost:8000` automatically. `start_web.py`:
- Installs `fastapi`, `uvicorn`, `pydantic` if missing.
- Runs `scripts/run_pipeline.py` if `data/processed/*.json` outputs are
  missing (first run only, ~20s; safe to re-run any time).
- Serves the app with `uvicorn dashboard.api:app --reload`.

### Option B — Streamlit dashboard (legacy)

```bash
pip install -r requirements.txt
python start.py
```

Opens `http://localhost:8501`. Same idempotent pipeline check, then launches
`streamlit run dashboard/app.py`.

### Regenerating data / pipeline outputs

```bash
python scripts/run_pipeline.py --force
```

This re-downloads/re-derives `data/raw/*` (via `scripts/import_kaggle_data.py`,
requires `kagglehub` + a Kaggle account, skipped if raw files already exist)
and rebuilds every file in `data/processed/`.

### Tests

```bash
pip install pytest
pytest tests/unit -v   # 62 tests, ~30s, all passing as of this pipeline run
```

---

## How the data pipeline works today

1. **`scripts/import_kaggle_data.py`** — downloads the Kaggle CSV via
   `kagglehub`, maps `Warehouse_ID/SKU_ID/Units_Sold/Promotion_Flag` etc. into
   `demand.csv`, aggregates per-SKU economics into `sku_master.csv`
   (unit cost, price, base daily demand, annual revenue), classifies ABC
   (revenue Pareto cut) and XYZ (demand CV cut) per `configs/business_rules.yaml`
   cutpoints, and writes `network.json`. It also rewrites the `stores:` block
   of `business_rules.yaml` with the real warehouse IDs and per-warehouse
   demand-share multipliers.
2. **`scripts/run_pipeline.py`** — the single entry point that runs everything
   downstream with no Jupyter dependency: routes each SKU to a forecaster
   (`src/models/router.py`), fits residual-based safety stock and EOQ per
   store-SKU (`src/optimization/policy_math.py`), simulates the decentralized
   `(s,S)` policy (`src/optimization/des_engine.py`), computes the
   multi-echelon pooling comparison (`src/optimization/multi_echelon.py`), and
   runs the two-window rolling-origin backtest plus the shortage-penalty
   ladder (`src/evaluation/rolling_backtest.py`). It writes everything in
   `data/processed/` that both dashboards read.
3. **Notebooks** (`notebooks/01`–`05`) hold the original narrative analysis —
   useful for methodology and worked examples, but see the data-source note
   above for why their printed numbers don't necessarily match a fresh
   `run_pipeline.py` run today.

All business rules (network, SKU mix, service floors, costs, constraints) live
in `configs/business_rules.yaml`; data-shape knobs live in
`configs/model_params.yaml`. Nothing business-facing is hard-coded.

## Phase 2 — EDA findings (`notebooks/01_eda.ipynb`, synthetic-data run)

Every claim below is a statistical test result from the notebook, not an eyeballed plot:

| Finding | Evidence |
|---|---|
| Revenue is concentrated | Gini 0.52; A-class = ~80% of revenue on a minority of SKUs |
| ABC and XYZ are associated | Chi-square p = 0.028 (not independent, as theory predicts) |
| Archetype labels hold up | Welch t-test p ≈ 0 on zero-day fraction, intermittent vs smooth |
| Weekly seasonality is real | ANOVA F = 469.6, p ≈ 0; weekend lift visible network-wide |
| Annual seasonality is real | Mid-year demand peak, network-wide |
| Promotions lift demand ~2.4x | Welch t-test p ≈ 0 |

These results directly inform Phase 3's model routing (Prophet for smooth/seasonal,
gradient boosting with the promo flag, Croston's for the validated intermittent SKUs).

## Phase 3 — Routed forecasting (`notebooks/02_forecasting.ipynb`, synthetic-data run)

**Tooling note:** Prophet was unavailable in the build environment (no network
access). Seasonality is instead modeled via explicit **Fourier-series regression**
(`src/models/seasonal_model.py`, fit with `sklearn.Ridge`) — mathematically
equivalent to Prophet's seasonality component, fully documented as a deliberate
substitution.

| Model | Used for | Result vs. naive baseline |
|---|---|---|
| Fourier seasonal | smooth-archetype SKUs (29) | Strong, consistent improvement (~29% mean RMSE reduction) |
| Gradient boosting | archetype/XYZ disagreement cases | Implemented + unit-tested; 0 cases triggered it on this dataset |
| Croston / baseline selector | intermittent-archetype SKUs (16) | See finding below |

**A genuine finding, reported honestly:** raw Croston's method did **not**
reliably beat the naive baseline (won only 6/16 SKUs, no sparsity threshold
predicts which). Diagnosis: this generator's intermittency is *moderate*
(`occurrence_prob` 0.15–0.45), not the *severe* near-all-zero pattern Croston's
advantage depends on. **Fix:** `CrostonOrBaseline` (`src/models/croston.py`)
fits both candidates on a held-out tail of the *training* data only and keeps
whichever wins — legitimate in-sample selection, not test-set peeking.

Overall: **18.6% mean MAE improvement**, 31/45 SKUs beating baseline, with all
underperformance concentrated in the diagnosed Croston cases above.

All three forecasters (`FourierSeasonalForecaster`, `GBMForecaster`,
`CrostonOrBaseline`) and the router are still what `scripts/run_pipeline.py`
calls today — the *models* didn't change in the Kaggle pivot, only the demand
series they're fit on.

## Phase 4 — Decentralized baseline (`notebooks/03_baseline_policy.ipynb`, synthetic-data run)

The baseline Phase 5 must beat: an independent `(s, S)` policy per store-SKU,
with safety stock sized from Phase 3 forecast residuals (not raw demand),
EOQ order quantities, and a discrete-event simulation producing real costs.

**Tooling note:** SimPy was unavailable in the build environment; the
discrete-event engine (`src/optimization/des_engine.py`) implements the same
event-loop behaviour in NumPy.

Result over the original 90-day holdout: total cost ~$26k (holding $15.3k /
ordering $7.9k / shortage $2.8k), all ABC classes **above** their service
floors, diagnosed as an EOQ batch-size effect rather than over-buffering.
(See the **live-pipeline section below** for what this looks like on the
Kaggle dataset today — materially different.)

## Phase 5 — Multi-echelon optimization (`notebooks/04_optimization.ipynb`, synthetic-data run)

The critical-path phase. Coordinated policy pools safety stock at the DC; the
pooling benefit is computed analytically (square-root law) with `scipy.optimize`
where numerical optimization is needed. PuLP was ruled out — pooling is
nonlinear and a linear program would linearize away the effect.

**Honest headline finding (synthetic data):** coordination saves ~11% of
*safety-stock* holding cost at realistic cross-store correlation (ρ≈0.68), but
safety stock was only ~2.2% of inventory (cycle stock dominates), so this was
<0.3% of total holding cost.

**Sensitivity (Section 4):** the pooling benefit depends entirely on
cross-store correlation, falling from ~40% (independent, ρ≈0) to ~3% (ρ=0.9).
This sensitivity curve is exactly what explains the live-pipeline result below
— the Kaggle-derived warehouses turn out to be even *more* correlated than
this ladder's high end.

## Phase 6 — Backtest & comparison (`notebooks/05_backtest_comparison.ipynb`, synthetic-data run)

Two rigor checks on the Phase 5 finding, both of which it survived on the
synthetic data: a two-window rolling-origin backtest (coordination saving
10.9% and 10.7% across independent periods) and a shortage-penalty ladder
($6/$12/$24) across which the coordination-saving conclusion held at 10.74%
in every scenario. `src/evaluation/rolling_backtest.py` — the module that runs
both checks — is unchanged; `run_pipeline.py` re-executes it against whatever
`demand.csv` currently contains.

---

## Current results on the live (Kaggle) pipeline

Output of `scripts/run_pipeline.py` against the current `data/raw/*` (the
Kaggle import: 5 warehouses, 50 SKUs, 2024-01-01 → 2024-12-30). This is what
both dashboards actually display today — read from
`data/processed/{baseline,optimization,backtest}_summary.json`.

| Metric | Value |
|---|---|
| Total cost (90-day headline window) | $98,786 |
| Holding cost | $73,961 |
| Ordering cost | $24,825 |
| Shortage cost | $0 |
| Fill rate (A / B / C) | 100% / 100% / 100% (floors: 98% / 95% / 90%) |
| Safety stock (% of avg on-hand) | 4.7% |
| Mean cross-store demand correlation | **0.989** |
| Coordination saving, realistic correlation | **0.4%** of safety-stock holding cost |
| Coordination saving, independent-demand upper bound | 55.3% (not the headline — see Phase 5 caveat) |
| Cross-window stability | 0.5% vs. 0.4% coordination saving (two independent 90-day windows) |
| Shortage-penalty ladder ($6/$12/$24) spread | 0.0% — cost is identical at every penalty level |

**Reading these honestly, in the same spirit as the rest of this project:**

- **Zero shortages at any penalty level** means the shortage-penalty
  sensitivity check from Phase 6 is now a non-event on this dataset — there's
  nothing for the penalty to bite on, so a 0% spread here is a *different*
  kind of finding than Phase 6's original "robust across a 16% cost swing"
  result, not a repeat of it.
- **Correlation of 0.989** is far higher than the synthetic network's ρ≈0.68,
  and Phase 5's own sensitivity curve already predicts what happens at high
  correlation: pooling benefit collapses toward zero. 0.4% is consistent with
  that curve, extrapolated past its original ρ=0.9 endpoint. This is *not* a
  contradiction of the earlier finding — it's the same mechanism, at a more
  extreme correlation than the synthetic data ever produced. It's also a sign
  that these 5 "warehouses" (derived from one dataset's `Warehouse_ID` split)
  share far more demand structure than genuinely distinct retail stores would
  — worth keeping in mind when interpreting warehouse-level results from
  either dashboard.
- The 55.3% independent-demand upper bound is reported here for the same
  reason it was in Phase 5: to show the shape of the ceiling, explicitly
  flagged as *not* the headline number.

---

## Dashboards

### FastAPI web dashboard — `InventoryPro` (`dashboard/api.py`, current)

Run with `python start_web.py` → `http://localhost:8000`. A full multi-page
app (Jinja2 templates in `dashboard/templates/pages/`, shared shell in
`dashboard/templates/base.html`, styling/JS in `dashboard/static/`) with a
JSON API under `/api/*` (interactive docs at `/api/docs`):

| Page | Route | What it shows |
|---|---|---|
| Dashboard | `/` | KPI tiles, fill-rate vs. floor, cost breakdown |
| Analytics | `/analytics` | Pareto/ABC-XYZ heatmap, aging buckets, profitability by class |
| Alerts Center | `/alerts` | Stockout-risk, supplier-risk, capacity, and drift alerts, derived live |
| Demand Forecasting | `/forecasting` | Live forecast form → `src/models/inference.py` → routed model output |
| Inventory Optimization | `/optimization` | Safety stock, EOQ, reorder points, coordination-saving KPI |
| Inventory Management | `/inventory` | Per-SKU stock status table |
| Replenishment | `/replenishment` | AI-ranked reorder recommendations (margin-ratio ranking, see below) |
| Warehouse Management | `/warehouse` | Per-warehouse utilization, fill rate, geography |
| Supplier Management | `/suppliers` | Supplier scorecards (rating, on-time %, risk) |
| Procurement | `/procurement` | Purchase-order queue and status |
| Orders | `/orders` | Order-fulfillment trend |
| Distribution Network | `/distribution` | Supplier → warehouse map |
| Scenario Simulator | `/simulator` | What-if sliders: shortage penalty, lead-time shift, demand spike, disruption, promo |
| AI Assistant | `/ai-chat` | Rule-based chat grounded in the live computed KPIs (no external LLM call) |
| Settings / About / Feedback | — | Static info pages + a feedback POST endpoint |

**Where the "operational" data comes from:** the source pipeline only ever
produced SKU-level demand and `(s,S)` policy output — there's no raw table for
warehouses, suppliers, or purchase orders. `dashboard/utils/ops_data.py`
derives all of it from real computed values (fill rates, EOQ, demand CV,
lead times) rather than fabricating it from nothing; the only invented fields
are cosmetic identity (supplier names, city labels, PO dates), generated once
with a fixed seed and cached to `data/processed/*.csv`. This is documented
in-file — worth reading before treating any `/suppliers` or `/procurement`
number as ground truth.

### Streamlit dashboard (`dashboard/app.py`, legacy)

Run with `python start.py` → `http://localhost:8501`. KPIs, the
coordination/correlation/cross-window charts, a what-if slider over the
shortage penalty, plus three live-inference tabs (`dashboard/app_live_tabs.py`):
Live Forecast, Live Simulation, Routing Sandbox — all backed by the same
`src/models/inference.py` the FastAPI app uses.

`dashboard/preview.html` is a pre-rendered static snapshot of this dashboard
(made when Streamlit's runtime couldn't be executed in the original build
sandbox) — open it directly with no install needed. `dashboard/static/*.html`
(`index.html`, `about.html`, `feedback.html`, `planning.html`,
`replenishment.html`) are earlier Tailwind-CDN mockups from before the
Jinja2 template app existed; the FastAPI app does not serve them as pages
(only `/static/css`, `/static/js` are mounted) — they're kept for reference,
not live.

## Phase 8 — Final Report (`report/Capstone_Report.docx`)

The academic deliverable, following the charter's 8-section structure
(Introduction → Literature Review → Data Description → Methodology →
Experiments & Results → Dashboard & Deployment → Discussion → Conclusion &
Future Work). ~2,800 words, 8 figures, 6 tables, 12 pages, built against the
original synthetic-data pipeline run — see the data-source note at the top of
this README before citing its numbers as current.

States plainly, in the Methodology section, why Prophet/SimPy/PuLP were
substituted, and reports the Phase 5/6 findings as found on that run rather
than smoothing them into a cleaner-looking but less honest narrative.

## Layout

```
configs/            business_rules.yaml, model_params.yaml
src/data/           (schema/package init only — synthetic generator removed)
src/models/         seasonal_model.py, gbm_model.py, croston.py, router.py, inference.py
src/optimization/   policy_math.py, des_engine.py, multi_echelon.py
src/evaluation/      baseline_and_metrics.py, rolling_backtest.py
scripts/            import_kaggle_data.py (Phase 1, Kaggle → data/raw), run_pipeline.py (Phases 1-6, end to end)
notebooks/           01_eda … 05_backtest_comparison (synthetic-data narrative)
dashboard/
  api.py              FastAPI app + /api/* endpoints
  app.py              Streamlit app (legacy)
  app_live_tabs.py     Streamlit live-inference tabs
  templates/           Jinja2 shell + pages/ for every FastAPI route
  static/              css/, js/, plus legacy static-mockup HTML
  utils/dashboard_data.py   shared KPI/data loading (used by both dashboards)
  utils/ops_data.py         derived warehouses/suppliers/purchase-orders/orders
  preview.html          pre-rendered static preview of the Streamlit app
tests/unit/          test_baseline_policy, test_forecasting, test_multi_echelon,
                      test_rolling_backtest, test_dashboard_data, test_inference
data/raw/            generated/downloaded artifacts (gitignored)
data/processed/      pipeline + dashboard outputs (gitignored)
report/              Capstone_Report.docx + figures/
```

## Open items (from charter)

`PROJECT_CHARTER.md` §9 marks all three original discovery-phase open items
resolved (deadline, cost parameters, advisor checkpoints) — see that file for
details. Note that the charter's network description (1 Factory → 1 DC → 3
Stores, synthetic data) predates the Kaggle data pivot documented at the top
of this README; the charter's *process* (deadline, checkpoints, objectives)
still stands, but its *data/architecture* description does not reflect the
current repo.
