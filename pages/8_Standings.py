"""Standings — Projected standings via Monte Carlo and composite power rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings
from src.ui_shared import (
    inject_custom_css,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
)

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
            # Try to build team_totals from real league standings data
            standings_df = load_league_standings()
            team_totals: dict[str, dict[str, float]] = {}

            if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
                # Pivot long-form standings table into {team: {category: total}}
                for _, row in standings_df.iterrows():
                    t = str(row["team_name"])
                    c = str(row["category"])
                    v = float(row.get("total", 0.0) or 0.0)
                    team_totals.setdefault(t, {})[c] = v

            if not team_totals:
                # Fall back to synthetic demo data
                rng = np.random.default_rng(42)
                for name in team_names_list:
                    team_totals[name] = {
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
                st.session_state["standings_demo_mode"] = True
            else:
                st.session_state["standings_demo_mode"] = False

            with st.spinner("Simulating season..."):
                try:
                    df = simulate_season(team_totals, n_sims=n_sims, seed=42)
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

            # Show current real standings if available
            standings_df = load_league_standings()
            if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
                st.markdown("**Current League Standings**")
                pivot = (
                    standings_df.pivot_table(
                        index="team_name",
                        columns="category",
                        values="total",
                        aggfunc="first",
                    )
                    .reset_index()
                    .rename(columns={"team_name": "Team"})
                )
                render_compact_table(pivot)
            else:
                st.info("No live league standings data available. Connect your Yahoo league to populate real data.")

            if "standings_result" in st.session_state:
                st.markdown("**Projected Season Results (Monte Carlo Simulation)**")
                if st.session_state.get("standings_demo_mode"):
                    st.caption(
                        "Simulation ran on synthetic demo data — connect your Yahoo league for real projections."
                    )
                sim_df = st.session_state["standings_result"]
                render_compact_table(sim_df)
            else:
                st.info("Configure simulation settings in the panel on the left, then click Run Simulation.")

    with tab2:
        if not POWER_RANKINGS_AVAILABLE:
            st.info("Power rankings module not available.")
        else:
            st.markdown("Composite team rankings based on 5 weighted factors.")

            # Build power data from real league data when available
            rosters_df = load_league_rosters()
            standings_df_pr = load_league_standings()
            using_real_data = False
            power_data: list[dict] = []

            if not rosters_df.empty and "team_name" in rosters_df.columns:
                using_real_data = True
                # Derive roster-size-based quality proxy per team
                roster_sizes = rosters_df.groupby("team_name")["player_id"].count()
                max_size = roster_sizes.max() or 1

                # Category balance from standings: normalize spread across categories
                cat_totals: dict[str, dict[str, float]] = {}
                if not standings_df_pr.empty and "team_name" in standings_df_pr.columns:
                    for _, row in standings_df_pr.iterrows():
                        t = str(row["team_name"])
                        c = str(row["category"])
                        v = float(row.get("total", 0.0) or 0.0)
                        cat_totals.setdefault(t, {})[c] = v

                # Collect all team names from rosters (and standings)
                team_names_real = sorted(set(rosters_df["team_name"].unique().tolist()) | set(cat_totals.keys()))

                # Compute category balance: fraction of cats where team is above league median
                all_cat_values: dict[str, list[float]] = {}
                for t_totals in cat_totals.values():
                    for cat, val in t_totals.items():
                        all_cat_values.setdefault(cat, []).append(val)
                cat_medians = {c: float(np.median(vals)) for c, vals in all_cat_values.items() if vals}
                # Inverse cats: lower is better
                inverse_cats = {"ERA", "WHIP", "L"}

                for team in team_names_real:
                    rq = float(roster_sizes.get(team, 0)) / float(max_size)
                    rq = max(0.1, min(1.0, rq))

                    # Category balance: fraction of cats where team beats median
                    t_cats = cat_totals.get(team, {})
                    if t_cats and cat_medians:
                        beats = 0
                        total_cats = 0
                        for cat, median in cat_medians.items():
                            val = t_cats.get(cat)
                            if val is not None:
                                total_cats += 1
                                if cat in inverse_cats:
                                    if val <= median:
                                        beats += 1
                                else:
                                    if val >= median:
                                        beats += 1
                        cb = float(beats) / float(total_cats) if total_cats > 0 else 0.5
                    else:
                        cb = 0.5

                    power_data.append(
                        {
                            "team_name": team,
                            "roster_quality": round(rq, 3),
                            "category_balance": round(max(0.1, min(1.0, cb)), 3),
                            "schedule_strength": None,
                            "injury_exposure": None,
                            "momentum": None,
                        }
                    )

            if not power_data:
                # Fall back to synthetic demo data
                rng_pr = np.random.default_rng(42)
                for name in team_names_list:
                    rq = max(0.1, min(1.0, 0.5 + rng_pr.normal(0, 0.15)))
                    cb = max(0.1, min(1.0, 0.6 + rng_pr.normal(0, 0.12)))
                    ss = max(0.1, min(1.0, 0.5 + rng_pr.normal(0, 0.10)))
                    ie = max(0.0, min(0.5, 0.1 + rng_pr.normal(0, 0.08)))
                    mom = max(0.5, min(2.0, 1.0 + rng_pr.normal(0, 0.2)))
                    power_data.append(
                        {
                            "team_name": name,
                            "roster_quality": round(rq, 3),
                            "category_balance": round(cb, 3),
                            "schedule_strength": round(ss, 3),
                            "injury_exposure": round(ie, 3),
                            "momentum": round(mom, 3),
                        }
                    )

            if not using_real_data:
                st.caption("Showing synthetic demo data — connect your Yahoo league for real power rankings.")

            # Analytics transparency badge (data quality from bootstrap)
            from src.data_bootstrap import get_bootstrap_context

            _boot_ctx = get_bootstrap_context()
            if _boot_ctx:
                from src.ui_analytics_badge import render_analytics_badge

                render_analytics_badge(_boot_ctx)

            try:
                pr_df = compute_power_rankings(power_data)

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

                # Show "N/A" for unavailable components instead of hiding them
                for col in ["schedule_strength", "injury_exposure", "momentum"]:
                    if col in pr_df.columns:
                        pr_df[col] = pr_df[col].apply(
                            lambda v: "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else v
                        )

                render_compact_table(pr_df)
            except Exception as e:
                st.error(f"Power rankings computation failed: {e}")
