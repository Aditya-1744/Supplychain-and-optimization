# Project Charter
## Multi-Echelon Inventory Optimization on Synthetic Supply-Chain Data

| | |
|---|---|
| **Owner** | Anshu Verma (3rd-Year CS) |
| **Type** | Business Analytics Capstone — Standard Tier |
| **Timeline** | 10–12 weeks |
| **Date** | 17 June 2026 |
| **Status** | Discovery complete — approved to build |

---

## 1. Problem Statement

Minimize **total network inventory holding cost** across a 3-echelon supply chain, **subject to ABC-tiered fill-rate floors**, and demonstrate that a **coordinated multi-echelon replenishment policy outperforms a decentralized EOQ + safety-stock baseline** on realistic synthetic demand.

> The objective is explicitly **constrained cost minimization**, not raw cost minimization. Cutting holding cost is trivial if service is allowed to collapse (just order nothing); the project's value is reducing cost *while service is held at target*. This framing is the project.

---

## 2. Objectives & Success Criteria

| # | Objective | KPI | Target |
|---|-----------|-----|--------|
| O1 | Reduce total holding cost vs. baseline | % reduction in avg. inventory value | **≥ 15% reduction** |
| O2 | Protect service levels | Fill rate per ABC class | A ≥ 98%, B ≥ 95%, C ≥ 90% |
| O3 | Accurate demand forecasts | MAPE / RMSE vs. naive | Beat moving-average baseline |
| O4 | Prove the multi-echelon thesis | Coordinated vs. decentralized cost-at-equal-service | Coordinated strictly cheaper |
| O5 | Stakeholder-ready delivery | Report + interactive dashboard | Both complete, equally weighted |

**The headline result** the project is judged on: *coordinated multi-echelon policy delivers lower total holding cost than decentralized per-node optimization, at identical ABC service floors.*

---

## 3. Scope Boundaries

### In Scope
- **Network:** 3 echelons — 1 Factory → 1 Distribution Center → 3 Stores (5 nodes).
- **SKUs:** ~40–50, classified by **ABC** (value) and **XYZ** (demand variability).
- **Demand simulation:** seasonality + trend, promotion spikes, and intermittent/lumpy patterns (C-items).
- **Forecasting layer:** model routed by demand pattern — seasonal/trend via explicit Fourier-series regression (Prophet was unavailable in the build environment; this is a documented, deliberate substitution — see Phase 3 notebook), gradient boosting with promo flags (promo-driven / archetype-XYZ disagreement cases), Croston's method with train-window model selection against a moving-average fallback (intermittent — see Phase 3 finding on moderate vs. severe intermittency).
- **Inventory policy:** `(s, S)` replenishment with service-driven safety stock.
- **Optimization:** coordinated multi-echelon cost minimization respecting all constraints below.
- **Constraints modeled:** warehouse capacity limits, inventory budget cap, minimum order quantity / pack sizes.
- **Validation:** rolling-origin backtest of coordinated policy vs. decentralized baseline.
- **Deliverables:** academic report + Streamlit dashboard (equal weight).

### Out of Scope (Explicit Boundaries)
- Supplier **lead-time variability** — lead times are **deterministic** (see Assumptions).
- Transshipment between stores or multiple DCs.
- Perishability / shelf-life and product substitution effects.
- Real / proprietary ERP data and live ERP integration.
- Full production MLOps (CI/CD, automated retraining, drift monitoring) — noted as future work.
- Stochastic / robust optimization under uncertainty (deterministic-equivalent only).

---

## 4. Key Assumptions

1. **Deterministic lead times.** Safety stock therefore uses the demand-variability form `SS = z · σ_d · √L`, not the full stochastic-lead-time form. This is a deliberate scope decision, not an omission.
2. Synthetic demand is generated from controlled, documented processes (seasonal + trend + promo + intermittent), giving a known ground truth for evaluation.
3. Holding, ordering, and shortage cost parameters are fixed inputs defined in config, with the cost-of-lost-sales subjected to sensitivity analysis.
4. The decentralized baseline reflects realistic industry practice (each node optimizes independently), making it a fair, non-strawman comparison.

---

## 5. Solution Approach

```
Synthetic Demand Generator (SimPy)
   → ABC/XYZ Classification
   → Demand Forecast (model routed per pattern)
   → Safety Stock (per ABC service floor)
   → TWO POLICIES, SAME CONSTRAINTS & SERVICE FLOORS:
        (A) Baseline: decentralized (s,S), EOQ qty, per-node safety stock
        (B) Optimized: coordinated multi-echelon cost minimization
   → Backtest both on holdout → compare cost @ equal service
   → Dashboard: KPIs, cost-vs-service curves, what-if sliders
```

**Service floors (tiered):** A-items 98%, B-items 95%, C-items 90%. Differentiated floors are themselves a cost lever — C-items stop hoarding safety stock they don't justify.

**Why this baseline:** decentralized EOQ + safety-stock is the textbook-correct method, so beating it is meaningful. The clean comparison is *decentralized vs. coordinated* at *identical service*, isolating the value of multi-echelon coordination.

---

## 6. Timeline (anchored: 17 Jun → mid-Sep 2026)

Start **17 Jun 2026**, deadline **~15 Sep 2026** — a ~13-week window covering the 12-week build plus one buffer week. Phases overlap at boundaries (soft handoffs); that is intentional.

| Phase | Calendar dates | Output |
|-------|----------------|--------|
| 1. Setup & synthetic data engine | 17–30 Jun | Repo scaffold, SimPy demand generator, config files |
| 2. EDA & ABC/XYZ classification | 24 Jun – 7 Jul | Pattern analysis, classified SKU set |
| 3. Forecasting layer | 1–21 Jul | Routed forecasters + accuracy evaluation |
| 4. Baseline policy | 15–28 Jul | Decentralized (s,S) + safety stock, costed |
| **5. Multi-echelon optimization (CRITICAL PATH)** | **22 Jul – 18 Aug** | Coordinated optimizer (PuLP → Pyomo) |
| 6. Backtest & comparison | 12–25 Aug | Cost/service trade-off results, ablations |
| 7. Dashboard | 19 Aug – 1 Sep | Streamlit app: KPIs, what-if sliders |
| 8. Report & handover | 26 Aug – 8 Sep | Final report, model cards, walkthrough |
| Buffer / submission | 9–15 Sep | Contingency + final polish (do not pre-spend) |

**Critical path:** Phase 5 is the longest and riskiest block — protect that window. **If behind, cut in this order:** (1) ship optimizer on PuLP, skip Pyomo; (2) trim dashboard to core KPIs + one what-if slider; (3) reduce report ablations. None of these touch the headline result.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| "Cost reduction" gamed by dropping service | Result is meaningless | Service floors are hard constraints in the optimizer |
| Multi-echelon optimization blows the timeline | Project incomplete | Start single-echelon, vectorize, then coordinate; PuLP before Pyomo |
| Synthetic data too clean to be convincing | Weak narrative | Inject promo spikes + intermittency + noise; document the generative process |
| Cost-of-lost-sales is uncertain | Conclusions fragile | Sensitivity analysis on this parameter (flagged in report Discussion) |
| Forecasting scope creep (LSTMs etc.) | Time sink | Match model to demand class only; advanced models = future work |

---

## 8. Definition of Done

- [ ] Synthetic generator produces seasonal, promo, and intermittent demand across 5 nodes / ~50 SKUs.
- [ ] SKUs classified by ABC and XYZ.
- [x] Forecasts beat a naive baseline (documented MAPE/RMSE). **Done:** 18.6% mean MAE improvement, 31/45 SKUs beating baseline; see `notebooks/02_forecasting.ipynb`.
- [x] Decentralized baseline policy implemented and costed. **Done:** per store-SKU (s,S) with residual-based safety stock + EOQ, discrete-event simulated (SimPy unavailable in build env → NumPy event loop, documented). 90-day cost ~$26k, all classes above service floors; over-service diagnosed as an EOQ batch effect. See `notebooks/03_baseline_policy.ipynb`.
- [x] Coordinated multi-echelon optimizer respects capacity, budget, and MOQ constraints. **Done:** analytical pooling model (square-root law) via `scipy.optimize`; PuLP unavailable and is the wrong tool for a nonlinear pooling objective (documented). See `notebooks/04_optimization.ipynb`.
- [~] Backtest shows ≥15% holding-cost reduction at equal-or-better ABC service. **AMENDED — honest finding:** the ≥15% target was an *aspiration* set before the data was built. In this synthetic network, safety stock is only **2.2% of inventory** (cycle stock dominates), so multi-echelon *pooling* — which acts only on safety stock — structurally cannot reach 15% of total cost. Measured coordination benefit: ~11% of safety-stock holding cost (<0.3% of total). This is reported truthfully rather than engineered upward; the *conditions* under which pooling would dominate (low cross-store correlation; higher safety-stock share) are quantified in the Phase 5 sensitivity analysis and flagged as Future Work. The headline at line 32 should be read with this amendment: the coordinated policy *does* reduce safety-stock cost at equal service, but the magnitude is bounded by this network's cost structure. **Phase 6 validation (rolling-origin backtest, 2 independent windows):** the ~11% coordination saving and ~2.2% safety-stock share both replicate almost exactly on an independent earlier period (10.9%/2.11% vs 10.7%/2.20%) — confirming this is a stable structural finding, not a fluke of one test split. The finding is also unchanged across the full shortage-penalty sensitivity ladder ($6/$12/$24), so it is robust to the project's most uncertain cost assumption. See `notebooks/05_backtest_comparison.ipynb`.
- [x] Streamlit dashboard with KPIs and what-if sliders. **Done:** `dashboard/app.py`, built on a shared, unit-tested data module (`dashboard/utils/dashboard_data.py`, 12/12 passing). Streamlit runtime unexecutable in the build sandbox; a rendered, verified static HTML preview (`dashboard/preview.html`) substitutes as proof of correctness, with the JS slider formula independently confirmed in Node against the Python-verified values.
- [x] Academic report with math notation, results tables, and trade-off curves. **Done:** `report/Capstone_Report.docx`, all 8 charter sections, 8 figures, 6 tables, ~2,800 words. Generated from live `data/processed/` results (no hand-typed numbers), validated, and visually verified page-by-page via PDF rasterization — which caught and fixed a stretched figure and a page-numbered title page before delivery.

---

## 9. Open Items for Next Phase

These were intentionally deferred from discovery and should be confirmed before/early in Phase 1:
1. ~~**Hard deadline / start date**~~ — **RESOLVED:** start 17 Jun 2026, deadline ~15 Sep 2026 (see Section 6).
2. ~~**Cost parameter values**~~ — **RESOLVED:** moderate baseline locked — holding rate 25%/yr, ordering cost $75/PO, shortage penalty $12/unit, with an asymmetric ×0.5/×1/×2 sensitivity ladder ($6/$12/$24) for Phase 6. See `configs/business_rules.yaml`.
3. ~~**Advisor checkpoints**~~ — **RESOLVED:** 3 reviews, anchored to phase boundaries, not a fixed recurring slot (advisor works on an as-requested basis):
   - **~21 Jul 2026** — end of Phase 3. Show: forecast accuracy (MAPE/RMSE) per routed model vs. naive baseline.
   - **~18 Aug 2026** — end of Phase 5 (critical path). Show: coordinated optimizer hitting cost + ABC service targets. Highest-stakes review — latest point a real fix is still possible before the deadline.
   - **~8 Sep 2026** — end of Phase 8, one week before submission. Full draft walkthrough.
   - **Action for Anshu:** message the advisor ~5–7 days before each date to request the slot — don't wait to be asked, since the cadence is on-demand, not standing.

**All discovery-phase open items are now closed.** Project moves fully into execution.
