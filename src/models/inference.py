"""
inference.py
============
Controller layer bridging the Streamlit frontend to the existing modeling
engine (src/models/*, src/optimization/*). This is the ONLY new module;
it does not reimplement any model -- it calls the real, unmodified classes
(FourierSeasonalForecaster, GBMForecaster, CrostonOrBaseline, route_skus,
simulate_sS) with parameters supplied live from the UI, instead of those
classes only ever being invoked from inside a notebook against a CSV.

Three responsibilities, one function each, used by three different dashboard
tabs:

  1. synthesize_history()  -- a user typing "SKU id / ABC / archetype / XYZ /
     CV / promo uplift" into a form has described a DEMAND PROFILE, not a
     time series. The existing forecasters need a dated history to fit on.
     This function generates that history live, reusing the SAME generative
     logic as Phase 1's DemandGenerator (Poisson-around-a-seasonal-mean for
     smooth, occurrence/lump-size for intermittent), just parameterized from
     the form instead of from model_params.yaml. This is explicit, not a
     silent substitution: the UI labels the resulting chart "synthesized
     history," not "actual history."

  2. forecast_live()        -- routes the synthesized SKU through route_skus
     (the real production router) and fits/predicts with whichever real
     model class it lands on. No new forecasting logic; this only orchestrates
     existing classes.

  3. simulate_policy_live() -- computes (s, S) from user-supplied lead time /
     holding rate / shortage penalty via the existing policy_math functions,
     then runs the existing simulate_sS DES engine and returns the full
     day-by-day trajectory for plotting.

  4. routing_sandbox()      -- a thin, read-only wrapper around route_skus
     for the "tweak demand_cv / toggle xyz_class, see what happens" sandbox.
     Calls the real router on a synthetic one-row DataFrame; does not
     reimplement the routing rule, so the sandbox can never drift from
     production behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.router import route_skus
from src.models.seasonal_model import FourierSeasonalForecaster
from src.models.gbm_model import GBMForecaster
from src.models.croston import CrostonOrBaseline
from src.optimization.policy_math import service_z, safety_stock, eoq, reorder_point, order_up_to
from src.optimization.des_engine import simulate_sS, SimResult


# --------------------------------------------------------------------- #
# 1. Synthesize a history from a live demand profile
# --------------------------------------------------------------------- #
@dataclass
class DemandProfile:
    """The shape a live user-submitted form takes once parsed."""
    sku_id: str
    abc_class: str          # "A" | "B" | "C"
    archetype: str          # "smooth" | "intermittent"
    xyz_class: str          # "X" | "Y" | "Z"
    demand_cv: float        # coefficient of variation, used for the smooth path's noise level
    promo_uplift: float     # demand multiplier during promo windows, e.g. 1.5-3.0
    base_daily_demand: float = 15.0
    n_history_days: int = 365
    seed: int = 42


def synthesize_history(profile: DemandProfile) -> pd.DataFrame:
    """
    Build a synthetic dated demand history matching the live profile, using
    the same two archetypes Phase 1's DemandGenerator implements. Returns a
    DataFrame with columns: date, units, on_promo.

    This is a STAND-IN for real historical data the live form doesn't supply.
    The dashboard must label any chart built from this "synthesized," not
    "actual," history -- see app.py's st.caption() at the call site.
    """
    rng = np.random.default_rng(profile.seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=profile.n_history_days, freq="D")
    n = len(dates)

    # Promo windows: a handful of multi-day uplift periods, same mechanism as Phase 1.
    on_promo = np.zeros(n, dtype=bool)
    n_promo_events = rng.poisson(profile.n_history_days / 60)  # roughly one every ~2 months
    for _ in range(int(n_promo_events)):
        start = rng.integers(0, n)
        dur = rng.integers(3, 8)
        on_promo[start:min(start + dur, n)] = True

    if profile.archetype == "intermittent":
        # Occurrence/lump-size process. Higher demand_cv -> sparser occurrence,
        # so a Z-class (erratic) profile produces visibly lumpier data than an
        # X-class one, even within the same "intermittent" archetype.
        occurrence_prob = float(np.clip(0.5 - 0.15 * profile.demand_cv, 0.05, 0.5))
        lump_size_mean = max(profile.base_daily_demand, 1.0)
        occurs = rng.random(n) < np.where(on_promo, min(occurrence_prob * 1.6, 0.95), occurrence_prob)
        sizes = rng.poisson(np.where(on_promo, lump_size_mean * profile.promo_uplift, lump_size_mean), n)
        units = np.where(occurs, sizes, 0)
    else:
        # Smooth path: Poisson around a seasonal mean level, with demand_cv
        # controlling how much weekly/random noise is layered on top.
        weekday = dates.weekday.to_numpy()
        weekly_mult = np.array([0.9, 0.9, 0.95, 1.0, 1.2, 1.45, 1.2])[weekday]
        annual = 1.0 + 0.2 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
        promo_mult = np.where(on_promo, profile.promo_uplift, 1.0)
        noise_scale = 1.0 + rng.normal(0, profile.demand_cv, n).clip(-0.8, 2.0)
        level = profile.base_daily_demand * weekly_mult * annual * promo_mult * np.clip(noise_scale, 0.05, None)
        units = rng.poisson(np.clip(level, 0.0, None))

    return pd.DataFrame({"date": dates, "units": units.astype(float), "on_promo": on_promo})


# --------------------------------------------------------------------- #
# 2. Live forecast: route + fit + predict, using the REAL model classes
# --------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    routed_model: str                  # which model route_skus chose
    is_disagreement: bool              # whether archetype/XYZ disagreed
    history: pd.DataFrame              # the synthesized history (for plotting)
    forecast_dates: pd.DatetimeIndex
    forecast_values: np.ndarray
    chosen_submodel: str | None = None  # only set when routed_model == "croston"


def forecast_live(profile: DemandProfile, horizon_days: int = 30) -> ForecastResult:
    """
    End-to-end live forecast: synthesize a history for this profile, route it
    through the REAL route_skus function, fit whichever real model class it
    lands on, and predict `horizon_days` forward.
    """
    history = synthesize_history(profile)

    # route_skus expects a SKU-master-shaped frame; build the one-row version.
    sku_row = pd.DataFrame([{
        "sku_id": profile.sku_id,
        "abc_class": profile.abc_class,
        "archetype": profile.archetype,
        "xyz_class": profile.xyz_class,
        "demand_cv": profile.demand_cv,
    }])
    routing, disagreements = route_skus(sku_row)
    routed_model = routing.iloc[0]["model"]
    is_disagreement = profile.sku_id in set(disagreements["sku_id"]) if len(disagreements) else False

    future_dates = pd.date_range(
        start=history["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D"
    )
    # A flat promo assumption for the forecast horizon (no promo scheduled);
    # this keeps the live demo simple and avoids guessing future promo dates.
    future_promo = np.zeros(horizon_days, dtype=bool)

    chosen_submodel = None
    if routed_model == "fourier_seasonal":
        model = FourierSeasonalForecaster().fit(
            history["date"], history["units"].to_numpy(), history["on_promo"].to_numpy()
        )
        forecast_values = model.predict(pd.Series(future_dates), future_promo)

    elif routed_model == "gbm":
        model = GBMForecaster().fit(history[["date", "units", "on_promo"]])
        forecast_values = model.predict(pd.Series(future_dates), future_promo)

    elif routed_model == "croston":
        model = CrostonOrBaseline().fit(history["units"].to_numpy())
        forecast_values = model.predict(horizon_days)
        chosen_submodel = model.chosen

    else:
        raise ValueError(f"Unrecognized route: {routed_model}")

    return ForecastResult(
        routed_model=routed_model,
        is_disagreement=is_disagreement,
        history=history,
        forecast_dates=future_dates,
        forecast_values=forecast_values,
        chosen_submodel=chosen_submodel,
    )


# --------------------------------------------------------------------- #
# 3. Live (s, S) policy simulation, using the REAL DES engine
# --------------------------------------------------------------------- #
@dataclass
class PolicySimResult:
    reorder_point: float
    order_up_to: float
    eoq: float
    safety_stock: float
    sim_result: SimResult              # the real SimResult from simulate_sS
    demand_used: np.ndarray


def simulate_policy_live(
    demand: np.ndarray,
    abc_class: str,
    unit_cost: float,
    lead_time: int,
    holding_cost_rate: float,
    shortage_penalty: float,
    ordering_cost: float = 75.0,
    moq: int = 10,
    pack_size: int = 5,
) -> PolicySimResult:
    """
    Compute (s, S) from live parameters via the existing policy_math
    functions, then run the existing simulate_sS DES engine and return the
    full trajectory. `demand` is the array the policy will be simulated
    against (e.g. a forecast, or a synthesized history's tail).
    """
    floors = {"A": 0.98, "B": 0.95, "C": 0.90}
    z = service_z(floors.get(abc_class, 0.95))

    mean_daily = float(np.mean(demand))
    sigma_d = float(np.std(demand))
    h = unit_cost * holding_cost_rate

    ss = safety_stock(z, sigma_d, lead_time)
    Q = eoq(mean_daily * 365.0, ordering_cost, h)
    s = reorder_point(mean_daily, lead_time, ss)
    S = order_up_to(s, Q)

    result = simulate_sS(
        demand=demand, reorder_point=s, order_up_to=S, lead_time=lead_time,
        unit_cost=unit_cost, holding_rate_annual=holding_cost_rate,
        ordering_cost=ordering_cost, shortage_penalty_per_unit=shortage_penalty,
        moq=moq, pack_size=pack_size,
    )

    return PolicySimResult(
        reorder_point=s, order_up_to=S, eoq=Q, safety_stock=ss,
        sim_result=result, demand_used=demand,
    )


# --------------------------------------------------------------------- #
# 4. Routing sandbox: a read-only, real call to route_skus
# --------------------------------------------------------------------- #
@dataclass
class RoutingSandboxResult:
    routed_model: str
    is_disagreement: bool
    explanation: str


def routing_sandbox(archetype: str, xyz_class: str, demand_cv: float, abc_class: str = "B") -> RoutingSandboxResult:
    """
    Calls the REAL, unmodified route_skus on a synthetic one-row DataFrame so
    the sandbox demonstrates production routing behaviour exactly, with no
    separate "demo" rule that could drift from it.
    """
    sku_row = pd.DataFrame([{
        "sku_id": "SANDBOX",
        "abc_class": abc_class,
        "archetype": archetype,
        "xyz_class": xyz_class,
        "demand_cv": demand_cv,
    }])
    routing, disagreements = route_skus(sku_row)
    routed_model = routing.iloc[0]["model"]
    is_disagreement = len(disagreements) > 0

    if is_disagreement:
        explanation = (
            f"DISAGREEMENT: archetype='{archetype}' but xyz_class='{xyz_class}'. "
            f"Per the production rule in src/models/router.py, a 'smooth' SKU "
            f"landing in erratic Z-class is flagged and routed to GBM instead of "
            f"the default Fourier model, since GBM handles irregular patterns "
            f"more robustly than a pure seasonal model."
        )
    elif archetype == "intermittent":
        explanation = "archetype='intermittent' -> routed to Croston (with train-window fallback to a moving-average baseline; see CrostonOrBaseline)."
    else:
        explanation = f"archetype='{archetype}', xyz_class='{xyz_class}' agree -> default Fourier seasonal route."

    return RoutingSandboxResult(routed_model=routed_model, is_disagreement=is_disagreement, explanation=explanation)
