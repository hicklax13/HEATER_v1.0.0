"""Standings — Projected standings via Monte Carlo and composite power rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.database import init_db
from src.ui_shared import inject_custom_css, render_context_card, render_context_columns, render_page_layout

try:
    from src.standings_projection import (
        generate_round_robin_schedule,  # noqa: F401
        simulate_season,
    )

    STANDINGS_AVAILABLE = True
except ImportError:
    STANDINGS_AVAILABLE = False

try:
    from src.power_rankings import bootstrap_confidence_interval, compute_power_rankings

    POWER_RANKINGS_AVAILABLE = True
except ImportError:
    POWER_RANKINGS_AVAILABLE = False

st.set_page_config(page_title="Heater | Standings", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()

render_page_layout("STANDINGS", banner_teaser="Projected standings and power rankings", banner_icon="standings")

if not STANDINGS_AVAILABLE and not POWER_RANKINGS_AVAILABLE:
    st.warning(
        "Standings and power rankings modules are not available. "
        "Ensure src/standings_projection.py and src/power_rankings.py exist."
    )
    st.stop()

# Shared demo team name list
team_names_list = [f"Team {i + 1}" for i in range(12)]

ctx, main = render_context_columns()

with ctx:
    if STANDINGS_AVAILABLE:
        render_context_card(
            "Simulation Controls",
            "<p>Configure and run the Monte Carlo season simulation.</p>",
        )
        n_sims = st.slider(
            "Number of Simulations",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            key="standings_sims",
        )

        if st.button("Run Simulation", type="primary", key="run_standings_sim"):
            # Build demo team data — weekly per-game rates for 12 categories
            rng = np.random.default_rng(42)
            demo_teams: dict[str, dict[str, float]] = {}
            for name in team_names_list:
                demo_teams[name] = {
                    "R": 4.5 + rng.normal(0, 1.0),
                    "HR": 1.5 + rng.normal(0, 0.5),
                    "RBI": 4.5 + rng.normal(0, 1.0),
                    "SB": 0.8 + rng.normal(0, 0.3),
                    "AVG": 0.260 + rng.normal(0, 0.010),
                    "OBP": 0.330 + rng.normal(0, 0.012),
                    "W": 0.9 + rng.normal(0, 0.2),
                    "L": 0.9 + rng.normal(0, 0.2),
                    "SV": 0.4 + rng.normal(0, 0.2),
                    "K": 7.5 + rng.normal(0, 1.5),
                    "ERA": 4.00 + rng.normal(0, 0.5),
                    "WHIP": 1.25 + rng.normal(0, 0.08),
                }
            with st.spinner("Simulating season..."):
                try:
                    df = simulate_season(demo_teams, n_sims=n_sims, seed=42)
                    st.session_state["standings_result"] = df
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

with main:
    tab1, tab2 = st.tabs(["Projected Standings", "Power Rankings"])

    with tab1:
        if not STANDINGS_AVAILABLE:
            st.info("Standings projection module not available.")
        else:
            st.markdown("Simulates head-to-head matchups across a full season using Monte Carlo methods.")

            if "standings_result" in st.session_state:
                df = st.session_state["standings_result"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Configure simulation settings in the panel on the left, then click Run Simulation.")

    with tab2:
        if not POWER_RANKINGS_AVAILABLE:
            st.info("Power rankings module not available.")
        else:
            st.markdown("Composite team rankings based on 5 weighted factors.")

            # Generate demo power rankings data
            rng_pr = np.random.default_rng(42)
            demo_power_data: list[dict] = []
            for name in team_names_list:
                rq = max(0.1, min(1.0, 0.5 + rng_pr.normal(0, 0.15)))
                cb = max(0.1, min(1.0, 0.6 + rng_pr.normal(0, 0.12)))
                ss = max(0.1, min(1.0, 0.5 + rng_pr.normal(0, 0.10)))
                ie = max(0.0, min(0.5, 0.1 + rng_pr.normal(0, 0.08)))
                demo_power_data.append(
                    {
                        "team_name": name,
                        "roster_quality": round(rq, 3),
                        "category_balance": round(cb, 3),
                        "schedule_strength": round(ss, 3),
                        "injury_exposure": round(ie, 3),
                        "momentum": 1.0,
                    }
                )

            try:
                pr_df = compute_power_rankings(demo_power_data)

                # Add bootstrap confidence intervals
                cis: list[dict[str, float]] = []
                for _, row in pr_df.iterrows():
                    try:
                        p5, p95 = bootstrap_confidence_interval(row["power_rating"])
                        cis.append({"p5": p5, "p95": p95})
                    except Exception:
                        cis.append({"p5": 0.0, "p95": 0.0})

                ci_df = pd.DataFrame(cis)
                pr_df["ci_low"] = ci_df["p5"].values
                pr_df["ci_high"] = ci_df["p95"].values

                st.dataframe(pr_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Power rankings computation failed: {e}")
