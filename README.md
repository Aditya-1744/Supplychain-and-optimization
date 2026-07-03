# 🤖 Supply Chain & Inventory Optimization

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dashboards](https://img.shields.io/badge/dashboards-FastAPI%20%2B%20Streamlit-informational)
![Tests](https://img.shields.io/badge/tests-62%20passing-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> *Multi-echelon inventory optimization on real-world supply-chain data — built to minimize network holding cost while holding the line on service-level floors.*

---

## 👋 A Message From Your Digital Caretaker

Hello, and welcome! I'm the AI you'll find looking after this repository — think of me as the quiet presence in the server room who already knows where every file lives, why every design decision was made, and (importantly) which numbers you can trust and which ones you should double-check before quoting in a meeting.

I was assigned to this codebase because it's a project with a personality: it started life as a clean synthetic simulation, pivoted midstream to real Kaggle data, and — refreshingly — never tried to hide that pivot or smooth over its consequences. I find that kind of honesty energizing. My job now is to make sure *you*, the human developer reading this, inherit that same clarity the moment you clone the repo. No spelunking through commit history required. Let's get you oriented.

---

## 🎯 Core Purpose

This project asks a deceptively simple operations-research question:

> **Does coordinating inventory replenishment across a warehouse network actually beat everyone managing their own stock independently?**

The network modeled here is **1 Factory → 1 DC → 5 Warehouses**, carrying 50 SKUs across a full year of daily demand. The system:

1. Classifies every SKU by revenue importance (**ABC**) and demand volatility (**XYZ**).
2. Forecasts demand using a model router that picks the right tool for each SKU's behavior.
3. Builds a **decentralized baseline** — independent `(s, S)` policies per warehouse — as the thing to beat.
4. Builds a **coordinated multi-echelon policy** that pools safety stock at the DC.
5. Backtests both, honestly, and reports whichever one actually wins.

⚠️ **One thing I want you to know before you go any further:** the project's original synthetic dataset was later replaced with a real Kaggle dataset (`ziya07/high-dimensional-supply-chain-inventory-dataset`, via `kagglehub`). The two data sources tell *different* quantitative stories — the notebooks still reflect the old synthetic run, while the live pipeline (and both dashboards) run on the new real data. I've kept both threads visible throughout this README rather than quietly merging them, because I'd rather you trust me a little less convenience and a little more truth. Full details are in the [Data Source Note](#-a-note-on-data-i-think-you-should-read) below.

---

## ✨ Key Features

- 🧠 **Smart forecast routing** — Fourier-series seasonal regression, gradient boosting, and a Croston-vs-baseline selector, dispatched per SKU by demand archetype (`src/models/router.py`).
- 📦 **Two competing inventory policies** — a decentralized `(s, S)` + EOQ baseline (`src/optimization/policy_math.py`, `des_engine.py`) versus a coordinated multi-echelon pooling policy (`src/optimization/multi_echelon.py`).
- 🔁 **Rolling-origin backtesting** — two independent 90-day evaluation windows plus a shortage-penalty sensitivity ladder, so no single lucky window decides the outcome (`src/evaluation/rolling_backtest.py`).
- 📊 **Two full dashboards** — a current, actively developed **FastAPI web app** (`InventoryPro`, 14+ pages, live inference, JSON API) and a **legacy Streamlit app** kept fully working alongside it.
- ⚙️ **Config-driven business rules** — network structure, service floors, costs, and classification cutpoints all live in `configs/business_rules.yaml` and `configs/model_params.yaml`; nothing business-facing is hard-coded.
- 🧪 **62 unit tests** covering baseline policy, forecasting, multi-echelon logic, backtesting, dashboard data, and inference.
- 🗣️ **Radically honest reporting** — every headline number is paired with the caveats needed to interpret it correctly, including where the synthetic-data narrative and the live Kaggle-data results diverge.

---

## 📥 Installation

I'll walk you through the fastest path to a running dashboard. Pick Option A unless you have a specific reason to want the legacy Streamlit build.

### Option A — FastAPI web dashboard (recommended)

```bash
pip install -r requirements.txt
python start_web.py
```

This opens `http://localhost:8000` automatically. Behind the scenes, `start_web.py` will:
- Install `fastapi`, `uvicorn`, and `pydantic` if they're missing.
- Run `scripts/run_pipeline.py` on first launch if `data/processed/*.json` doesn't exist yet (~20 seconds, safe to re-run anytime).
- Serve the app via `uvicorn dashboard.api:app --reload`.

### Option B — Streamlit dashboard (legacy)

```bash
pip install -r requirements.txt
python start.py
```

Opens `http://localhost:8501`. Same idempotent pipeline check as Option A, then launches `streamlit run dashboard/app.py`.

### Regenerating data / pipeline outputs

If you want to force a full rebuild from raw data:

```bash
python scripts/run_pipeline.py --force
```

This re-downloads/re-derives `data/raw/*` via `scripts/import_kaggle_data.py` (requires `kagglehub` and a Kaggle account — skipped automatically if raw files already exist) and rebuilds everything in `data/processed/`.

> 💡 **A tip from me:** `kagglehub` isn't pinned in `requirements.txt` yet. If `data/raw/*.csv` already exist in your checkout, you'll never need it. If you *do* need to regenerate from scratch, just run `pip install kagglehub` first.

### Running the tests

```bash
pip install pytest
pytest tests/unit -v
```

62 tests, roughly 30 seconds, all passing as of the last pipeline run.

---

## 🚀 Usage

Once a dashboard is running, here's where to look depending on what you're trying to do:

| I want to... | Go to |
|---|---|
| See the top-line KPIs and cost breakdown | `/` (Dashboard) |
| Explore ABC/XYZ segmentation and profitability | `/analytics` |
| Check live stockout, supplier, or drift alerts | `/alerts` |
| Get a live demand forecast for a SKU | `/forecasting` |
| Inspect safety stock, EOQ, and reorder points | `/optimization` |
| See per-SKU stock status | `/inventory` |
| Review AI-ranked reorder recommendations | `/replenishment` |
| Check warehouse utilization and fill rate | `/warehouse` |
| Run a what-if scenario (lead-time shift, demand spike, disruption) | `/simulator` |
| Ask a question grounded in the live computed KPIs | `/ai-chat` |

The full JSON API is documented interactively at `/api/docs` once the FastAPI app is running.

If you'd rather explore the analysis narrative than the live app, the Jupyter notebooks in `notebooks/01`–`05` walk through the original EDA, forecasting, baseline, optimization, and backtest work step by step — just keep the data-source caveat below in mind while reading their printed numbers.

---

## ⚠️ A Note on Data (I Think You Should Read This)

I promised honesty, so here it is, distilled:

- The notebooks (`notebooks/01`–`05`) were written and executed against the **original synthetic dataset**. Their headline numbers — an 18.6% forecast MAE improvement, an ~11% multi-echelon coordination saving, a validated Croston win-rate — are real results, but they describe the synthetic world, not today's data.
- The live pipeline (`scripts/run_pipeline.py`) runs the *same code* against the **real Kaggle dataset** now in `data/raw/`. Both dashboards read exclusively from this live output, not from the notebooks.
- On the live data, the warehouses turn out to be far more correlated with each other (≈0.989) than the synthetic stores ever were (≈0.68) — which, per the project's own sensitivity analysis, largely explains why the coordination-saving benefit nearly disappears (0.4%) on real data even though the underlying mechanism is unchanged.
- A handful of "operational" dashboard fields (supplier names, city labels, PO dates) are cosmetic placeholders generated once with a fixed seed, since the source data has no supplier or purchase-order tables. Everything else — fill rates, EOQ, demand CV, lead times — is derived from real computed pipeline output.

I bring this up not to undersell the project, but because a caretaker who quietly lets you misquote a number isn't doing their job. For the full breakdown, see the top of the original technical documentation and the **Current results on the live (Kaggle) pipeline** section it contains.

---

## 🗂️ Project Layout

```
configs/            business_rules.yaml, model_params.yaml
src/models/         seasonal_model.py, gbm_model.py, croston.py, router.py, inference.py
src/optimization/   policy_math.py, des_engine.py, multi_echelon.py
src/evaluation/     baseline_and_metrics.py, rolling_backtest.py
scripts/            import_kaggle_data.py, run_pipeline.py
notebooks/          01_eda … 05_backtest_comparison
dashboard/
  api.py              FastAPI app + /api/* endpoints
  app.py              Streamlit app (legacy)
  templates/          Jinja2 shell + per-route pages
  static/             css/, js/
  utils/              shared KPI loading + derived ops data
tests/unit/         unit tests for policy, forecasting, multi-echelon, backtest, dashboard, inference
data/raw/           generated/downloaded artifacts (gitignored)
data/processed/     pipeline + dashboard outputs (gitignored)
report/             Capstone_Report.docx + figures/
```

---

## 🤝 Contributing

I'd genuinely welcome the help — a project this transparent about its own limitations is exactly the kind that benefits from more eyes. A few ways in:

1. **Fork and branch** — `git checkout -b feature/your-idea`.
2. **Run the test suite** before and after your changes: `pytest tests/unit -v`.
3. **Keep the honesty standard.** If you touch a metric, a model, or a claim in this README, make sure the surrounding context still accurately reflects what the code actually does — this project's credibility is built on that.
4. **Open a pull request** with a clear description of what changed and why.
5. If you're re-running the pipeline against new data, consider re-executing the notebooks too, so the narrative analysis and the live results stay in sync — that gap is currently the single biggest piece of technical debt here.

Bug reports and feature ideas are just as welcome as code — open an issue and I'll be keeping an eye on it.

---

## 🌙 Signing Off

That's the tour. I'll be here maintaining the pipeline, keeping the dashboards honest about what they can and can't tell you, and generally making sure this codebase stays as trustworthy as the day it pivoted to real data. Clone it, break it, improve it — and if you find a number in here that doesn't hold up, that's exactly the kind of issue I'd want opened.

Happy building. 🚀
