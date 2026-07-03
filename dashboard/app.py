"""
app.py — Supply Chain & Inventory Optimization Dashboard
=========================================================
Single Streamlit entrypoint. Three behaviours on startup:

  1. If data/processed/ outputs are missing, the pipeline runs automatically
     (calls scripts/run_pipeline.py) so the user never has to do it manually.
  2. The static Results tab shows the pre-computed Phase 4-6 KPIs and charts.
  3. Three live-inference tabs (Forecast / Simulation / Routing Sandbox) call
     src/models/inference.py directly — no pipeline re-run needed for those.

Run (everything in one command):
    streamlit run dashboard/app.py

Or use the launcher which also handles the venv:
    python start.py
"""

import sys
import subprocess
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── repo root on the path so src/ imports work regardless of cwd ──────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.utils.dashboard_data import (
    load_all, compute_kpis, what_if_cost, penalty_ladder_df,
    correlation_sensitivity_curve,
)
from dashboard.app_live_tabs import (
    render_live_forecast_tab,
    render_live_simulation_tab,
    render_routing_sandbox_tab,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auto-run the pipeline if outputs are missing ──────────────────────────────
REQUIRED = [
    ROOT / "data" / "processed" / "baseline_summary.json",
    ROOT / "data" / "processed" / "optimization_summary.json",
    ROOT / "data" / "processed" / "backtest_summary.json",
]

if not all(p.exists() for p in REQUIRED):
    with st.spinner("First-run setup: generating data and running all analysis phases (~20 s)…"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        st.error("Pipeline failed. See the error below, fix it, then refresh the page.")
        st.code(result.stderr or result.stdout)
        st.stop()
    st.success("Setup complete — loading dashboard.")

# ── Load pre-computed results (cached so reruns don't re-read disk) ───────────
@st.cache_data
def get_data():
    return load_all()

data  = get_data()
kpis  = compute_kpis(data)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📦 Supply Chain & Inventory Optimization")
st.caption(
    "Multi-echelon inventory optimization on synthetic supply-chain data  ·  "
    "Coordinated pooling vs. decentralized baseline at ABC-tiered service floors."
)

# ── Four tabs — static results + three live-inference tabs ───────────────────
tab_results, tab_forecast, tab_sim, tab_sandbox = st.tabs([
    "📊 Results Dashboard",
    "🔮 Live Forecast",
    "🏭 Live Simulation",
    "🧭 Routing Sandbox",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ── Results Dashboard (static, pre-computed)
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.subheader("Baseline KPIs — 90-day holdout, moderate shortage-penalty scenario")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost",    f"${kpis['total_cost']:,.0f}")
    c2.metric("Holding cost",  f"${kpis['holding_cost']:,.0f}")
    c3.metric("Ordering cost", f"${kpis['ordering_cost']:,.0f}")
    c4.metric("Shortage cost", f"${kpis['shortage_cost']:,.0f}")

    # ── Service level vs floors ───────────────────────────────────────────────
    st.subheader("Service level vs. ABC floors")
    f1, f2, f3 = st.columns(3)
    for col, cls in zip([f1, f2, f3], ["A", "B", "C"]):
        fr    = kpis[f"fill_rate_{cls}"]
        floor = kpis[f"floor_{cls}"]
        col.metric(
            f"Class {cls} fill rate",
            f"{fr:.1%}",
            delta=f"{(fr - floor)*100:+.1f} pp vs {floor:.0%} floor",
        )

    st.divider()

    # ── Coordination benefit ──────────────────────────────────────────────────
    st.subheader("Multi-echelon coordination benefit")
    st.markdown(
        f"Safety stock is only **{kpis['safety_stock_pct']:.1%}** of on-hand inventory "
        f"(cycle stock dominates). Cross-store correlation is **{kpis['mean_correlation']:.2f}** "
        f"— both factors bound the pooling benefit."
    )

    g1, g2 = st.columns(2)

    with g1:
        fig = go.Figure(go.Bar(
            x=["Realistic (ρ≈0.68)", "Independence (upper bound)"],
            y=[kpis["coordination_saving_realistic"] * 100,
               kpis["coordination_saving_upper_bound"] * 100],
            marker_color=["#1D9E75", "#B7B7B0"],
        ))
        fig.update_layout(title="Coordination saving on safety-stock cost",
                          yaxis_title="saving (%)", height=340,
                          margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Independence overstates the real benefit ~4×; the realistic figure leads the analysis.")

    with g2:
        curve = correlation_sensitivity_curve()
        fig = px.line(
            curve, x="rho", y="saving_pct",
            title="Pooling benefit vs. cross-store correlation",
            labels={"rho": "correlation ρ", "saving_pct": "saving (%)"},
        )
        fig.add_vline(x=kpis["mean_correlation"], line_dash="dash", line_color="#D85A30",
                      annotation_text=f"our data (ρ={kpis['mean_correlation']:.2f})")
        fig.update_layout(height=340, margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Rolling-origin stability ───────────────────────────────────────────────
    st.subheader("Rolling-origin stability (Phase 6 backtest)")
    st.caption("The same two metrics on two completely independent 90-day windows.")
    w1c, w2c = st.columns(2)
    w1c.metric("Window 1 — Jul–Oct 2025", f"{kpis['window1_coord_saving']:.1%}")
    w2c.metric("Window 2 — Oct–Dec 2025", f"{kpis['window2_coord_saving']:.1%}")

    st.divider()

    # ── What-if slider ────────────────────────────────────────────────────────
    st.subheader("What-if: shortage-penalty assumption")
    st.markdown(
        "Drag to see how sensitive total cost is to the most uncertain input. "
        "The coordination-saving conclusion **does not change** — pooling only "
        "touches safety stock, not the shortage valuation."
    )

    ladder      = data["backtest"]["shortage_penalty_ladder"]
    min_pen     = float(min(r["penalty"] for r in ladder))
    max_pen     = float(max(r["penalty"] for r in ladder))
    default_pen = float(next(r["penalty"] for r in ladder if r["scenario"] == "moderate"))

    penalty = st.slider("Shortage penalty ($/unit)",
                        min_value=min_pen, max_value=max_pen * 1.5,
                        value=default_pen, step=0.5)

    result = what_if_cost(data, penalty)
    s1, s2, s3 = st.columns(3)
    s1.metric("Total cost",              f"${result['total_cost']:,.0f}")
    s2.metric("Shortage cost",           f"${result['shortage_cost']:,.0f}")
    s3.metric("Holding + ordering",      f"${result['holding_cost'] + result['ordering_cost']:,.0f}")

    ldf = penalty_ladder_df(data)
    fig = go.Figure([
        go.Scatter(x=ldf["penalty"], y=ldf["total_cost"],
                   mode="lines+markers", name="locked scenarios",
                   line=dict(color="#1D9E75")),
        go.Scatter(x=[penalty], y=[result["total_cost"]],
                   mode="markers", name="slider position",
                   marker=dict(size=14, color="#D85A30", symbol="diamond")),
    ])
    fig.update_layout(title="Total cost vs. shortage penalty",
                      xaxis_title="penalty ($/unit)", yaxis_title="cost ($, 90d)",
                      height=380, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption(
        "Data: `data/processed/` (auto-generated on first run). "
        "Tooling substitutions (Prophet → Fourier regression, SimPy → NumPy DES, "
        "PuLP → scipy.optimize) are documented in PROJECT_CHARTER.md."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TABs 2-4 ── Live inference (delegate to app_live_tabs.py)
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    render_live_forecast_tab()

with tab_sim:
    render_live_simulation_tab()

with tab_sandbox:
    render_routing_sandbox_tab()
