"""
app_live_tabs.py
=================
NEW interactive sections for app.py, added as additional tabs alongside the
existing static results dashboard (Section 0-7 in the original app.py, which
reads data/processed/*.json and is UNCHANGED -- this file is purely additive).

Three new tabs, each backed by src/models/inference.py:
  Tab A: Live Forecast      -- form -> synthesize history -> route -> forecast
  Tab B: Live Simulation    -- form -> (s, S) policy -> live DES trajectory
  Tab C: Routing Sandbox    -- toggles -> instant routing decision, no form
                                needed (cheap enough to run on every interaction)

INTEGRATION: import `render_live_tabs()` into the existing app.py and call it
inside a new st.tabs(...) entry. See the bottom of this file for the exact
one-line wiring into the existing dashboard.

STATE MANAGEMENT PATTERN:
  st.form + st.form_submit_button prevents the (s, S) simulation and the
  forecast from re-running on every widget interaction -- only on explicit
  submit. Results are written to st.session_state so they persist and
  re-render across reruns triggered by OTHER widgets (e.g. switching tabs),
  rather than vanishing until the form is resubmitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.models.inference import (
    DemandProfile, forecast_live, simulate_policy_live, routing_sandbox,
)


# ======================================================================= #
# TAB A: Live Forecast
# ======================================================================= #
def render_live_forecast_tab() -> None:
    st.subheader("Live demand forecast")
    st.caption(
        "Describe a SKU's demand profile below. Since no historical data exists for a "
        "SKU you're describing live, a synthetic history is generated from your inputs "
        "first (clearly labeled below), then routed through the real production router "
        "(`route_skus`) and forecast with whichever real model class it lands on."
    )

    # --- st.form: nothing below re-runs until Submit is pressed ---------- #
    with st.form("live_forecast_form"):
        c1, c2, c3 = st.columns(3)
        sku_id = c1.text_input("SKU ID", value="LIVE-001")
        abc_class = c2.selectbox("ABC class", ["A", "B", "C"], index=0)
        archetype = c3.selectbox("Demand archetype", ["smooth", "intermittent"], index=0)

        c4, c5, c6 = st.columns(3)
        xyz_class = c4.selectbox("XYZ volatility class", ["X", "Y", "Z"], index=0)
        demand_cv = c5.slider("Demand CV (coefficient of variation)", 0.05, 2.0, 0.30, 0.05)
        promo_uplift = c6.slider("Promo uplift multiplier", 1.0, 4.0, 2.0, 0.1)

        c7, c8 = st.columns(2)
        base_demand = c7.number_input("Base daily demand (units)", min_value=1.0, value=20.0, step=1.0)
        horizon = c8.slider("Forecast horizon (days)", 7, 60, 14)

        submitted = st.form_submit_button("Run live forecast", type="primary")

    # --- On submit: call inference.py, stash result in session_state ----- #
    if submitted:
        profile = DemandProfile(
            sku_id=sku_id, abc_class=abc_class, archetype=archetype, xyz_class=xyz_class,
            demand_cv=demand_cv, promo_uplift=promo_uplift, base_daily_demand=base_demand,
        )
        with st.spinner("Synthesizing history, routing, and forecasting..."):
            st.session_state["live_forecast_result"] = forecast_live(profile, horizon_days=horizon)
            st.session_state["live_forecast_profile"] = profile

    # --- Render from session_state (persists across reruns/tab switches) - #
    result = st.session_state.get("live_forecast_result")
    if result is None:
        st.info("Fill in the form above and press **Run live forecast** to see results.")
        return

    badge = "🔀 DISAGREEMENT — routed to GBM" if result.is_disagreement else f"routed to **{result.routed_model}**"
    st.success(f"Model routing decision: {badge}")
    if result.chosen_submodel:
        st.caption(f"Croston/baseline selector chose: **{result.chosen_submodel}** (train-window comparison; see CrostonOrBaseline).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.history["date"], y=result.history["units"], mode="lines",
        name="synthesized history", line=dict(color="#888780", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=result.forecast_dates, y=result.forecast_values, mode="lines+markers",
        name=f"live forecast ({result.routed_model})", line=dict(color="#1D9E75", width=2),
    ))
    fig.update_layout(
        title="Synthesized history (gray) + live forecast (green)",
        xaxis_title="date", yaxis_title="units", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ The gray line is **synthesized**, not real historical sales — there is no "
        "history for a SKU described live. It exists only to give the real forecasting "
        "models something to fit on, so the routing/forecast logic itself is exercised "
        "exactly as in production."
    )


# ======================================================================= #
# TAB B: Live Multi-Echelon (s, S) Simulation
# ======================================================================= #
def render_live_simulation_tab() -> None:
    st.subheader("Live (s, S) inventory simulation")
    st.caption(
        "Runs the same NumPy discrete-event simulation engine used in Phases 4-6 "
        "(`src/optimization/des_engine.py::simulate_sS`), live, against your parameters."
    )

    with st.form("live_simulation_form"):
        c1, c2, c3 = st.columns(3)
        abc_class = c1.selectbox("ABC class (sets the service floor)", ["A", "B", "C"], index=0, key="sim_abc")
        unit_cost = c2.number_input("Unit cost ($)", min_value=0.1, value=15.0, step=0.5)
        n_days = c3.slider("Simulation length (days)", 30, 180, 90)

        c4, c5, c6 = st.columns(3)
        lead_time = c4.slider("Lead time (days)", 1, 14, 3)
        holding_cost_rate = c5.slider("Holding cost rate (annual, % of unit cost)", 0.05, 0.50, 0.25, 0.01)
        shortage_penalty = c6.slider("Shortage penalty ($/unit)", 1.0, 50.0, 12.0, 1.0)

        c7, c8 = st.columns(2)
        mean_demand = c7.number_input("Mean daily demand for this simulation", min_value=1.0, value=15.0)
        demand_std = c8.number_input("Daily demand std dev", min_value=0.0, value=4.0)

        submitted = st.form_submit_button("Run live simulation", type="primary")

    if submitted:
        rng = np.random.default_rng(7)
        demand = np.clip(rng.normal(mean_demand, demand_std, n_days), 0, None)
        with st.spinner("Computing (s, S) and running the discrete-event simulation..."):
            st.session_state["live_sim_result"] = simulate_policy_live(
                demand=demand, abc_class=abc_class, unit_cost=unit_cost, lead_time=lead_time,
                holding_cost_rate=holding_cost_rate, shortage_penalty=shortage_penalty,
            )

    sim = st.session_state.get("live_sim_result")
    if sim is None:
        st.info("Set parameters above and press **Run live simulation** to see the trajectory.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reorder point (s)", f"{sim.reorder_point:.0f}")
    m2.metric("Order-up-to (S)", f"{sim.order_up_to:.0f}")
    m3.metric("Total cost", f"${sim.sim_result.total_cost:,.0f}")
    m4.metric("Fill rate", f"{sim.sim_result.fill_rate:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sim.sim_result.daily_on_hand, mode="lines",
                             name="on-hand inventory", line=dict(color="#1D9E75", width=1.5)))
    fig.add_hline(y=sim.reorder_point, line_dash="dash", line_color="#D85A30",
                  annotation_text="reorder point s")
    fig.add_hline(y=sim.order_up_to, line_dash="dot", line_color="#888780",
                  annotation_text="order-up-to S")
    fig.update_layout(title="Live (s, S) policy trajectory", xaxis_title="day",
                      yaxis_title="units on hand", height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full cost breakdown"):
        st.write(f"Holding: ${sim.sim_result.total_holding_cost:,.0f}")
        st.write(f"Ordering: ${sim.sim_result.total_ordering_cost:,.0f}")
        st.write(f"Shortage: ${sim.sim_result.total_shortage_cost:,.0f}")
        st.write(f"Orders placed: {sim.sim_result.n_orders}")


# ======================================================================= #
# TAB C: SKU Routing Sandbox
# ======================================================================= #
def render_routing_sandbox_tab() -> None:
    st.subheader("SKU routing sandbox")
    st.caption(
        "Tweak the toggles below to see exactly how the REAL, unmodified `route_skus` "
        "function (src/models/router.py) assigns a model. No form/submit needed here — "
        "routing is cheap enough to recompute on every interaction."
    )

    c1, c2, c3 = st.columns(3)
    archetype = c1.radio("Demand archetype", ["smooth", "intermittent"], horizontal=True)
    xyz_class = c2.radio("XYZ volatility class", ["X", "Y", "Z"], horizontal=True)
    demand_cv = c3.slider("Demand CV", 0.0, 2.0, 0.3, 0.05, key="sandbox_cv")

    result = routing_sandbox(archetype=archetype, xyz_class=xyz_class, demand_cv=demand_cv)

    if result.is_disagreement:
        st.error(f"**{result.routed_model.upper()}** — {result.explanation}")
    else:
        st.success(f"**{result.routed_model.upper()}** — {result.explanation}")

    # A small static reference table so the user can see the full rule at a glance.
    st.markdown("##### Full routing rule (for reference)")
    ref = pd.DataFrame([
        {"archetype": "intermittent", "xyz_class": "any", "→ model": "croston", "disagreement?": "no"},
        {"archetype": "smooth", "xyz_class": "X or Y", "→ model": "fourier_seasonal", "disagreement?": "no"},
        {"archetype": "smooth", "xyz_class": "Z", "→ model": "gbm", "disagreement?": "YES — smooth label but erratic CV"},
    ])
    st.dataframe(ref, use_container_width=True, hide_index=True)


# ======================================================================= #
# Wiring into the existing app.py
# ======================================================================= #
def render_live_tabs() -> None:
    """
    Call this from the existing app.py to add the three new tabs. Example
    integration (add near the top of the existing dashboard's body, after
    the existing st.title()/st.caption() calls):

        from dashboard.app_live_tabs import render_live_tabs
        ...
        static_tab, forecast_tab, sim_tab, sandbox_tab = st.tabs(
            ["📊 Results Dashboard", "🔮 Live Forecast", "🏭 Live Simulation", "🧭 Routing Sandbox"]
        )
        with static_tab:
            ... existing app.py KPI/chart code unchanged, indented one level ...
        with forecast_tab:
            render_live_forecast_tab()
        with sim_tab:
            render_live_simulation_tab()
        with sandbox_tab:
            render_routing_sandbox_tab()
    """
    forecast_tab, sim_tab, sandbox_tab = st.tabs(
        ["🔮 Live Forecast", "🏭 Live Simulation", "🧭 Routing Sandbox"]
    )
    with forecast_tab:
        render_live_forecast_tab()
    with sim_tab:
        render_live_simulation_tab()
    with sandbox_tab:
        render_routing_sandbox_tab()
