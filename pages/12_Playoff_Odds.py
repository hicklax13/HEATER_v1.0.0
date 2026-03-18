"""Playoff Odds — Monte Carlo simulation of remaining H2H season."""

import logging
import time
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings, load_player_pool
from src.ui_shared import T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

try:
    from src.playoff_sim import simulate_season

    _HAS_PLAYOFF_SIM = True
except ImportError:
    _HAS_PLAYOFF_SIM = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Playoff Odds", page_icon="", layout="wide")

init_db()
inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>PLAYOFF ODDS</span></div></div>',
    unsafe_allow_html=True,
)

if not _HAS_PLAYOFF_SIM:
    st.error("Playoff simulation module not available.")
    st.stop()

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Load standings and rosters
standings = load_league_standings()
rosters = load_league_rosters()

if standings.empty or rosters.empty:
    st.warning("Playoff odds require both standings and roster data. Connect your Yahoo league in Connect League.")
    st.stop()

# Build team totals
all_team_totals: dict[str, dict[str, float]] = {}
if "category" in standings.columns:
    for _, row in standings.iterrows():
        team = str(row.get("team_name", ""))
        cat = str(row.get("category", "")).strip()
        val = float(row.get("total", 0))
        all_team_totals.setdefault(team, {})[cat] = val

if len(all_team_totals) < 2:
    st.warning("Not enough team data to simulate a season.")
    st.stop()

# Build league_rosters dict
league_rosters_dict: dict[str, list[int]] = {}
for _, row in rosters.iterrows():
    team = str(row.get("team_name", ""))
    pid = row.get("player_id")
    if team and pid:
        league_rosters_dict.setdefault(team, []).append(int(pid))

# Get user team
user_teams = rosters[rosters["is_user_team"] == 1]
user_team_name = str(user_teams.iloc[0]["team_name"]) if not user_teams.empty else None

# Weeks remaining
now = datetime.now(UTC)
season_end = datetime(now.year, 10, 1, tzinfo=UTC)
weeks_remaining = max(1, int((season_end - now).days / 7))

# Simulation controls
st.markdown(f"**Weeks Remaining:** {weeks_remaining} | **Teams:** {len(all_team_totals)}")

n_sims = st.slider("Simulations:", min_value=100, max_value=2000, value=500, step=100, key="playoff_sims")

if st.button("Simulate Season", type="primary", key="run_playoff_sim"):
    progress = st.progress(0, text="Simulating season...")
    try:
        progress.progress(20, text=f"Running {n_sims} simulations...")
        results = simulate_season(
            all_team_totals=all_team_totals,
            league_rosters=league_rosters_dict,
            player_pool=pool,
            weeks_remaining=weeks_remaining,
            n_sims=n_sims,
            config=config,
        )
        progress.progress(100, text="Simulation complete!")
        time.sleep(0.3)
        progress.empty()
        st.session_state["playoff_results"] = results
    except Exception as e:
        progress.empty()
        logger.exception("Playoff simulation failed")
        st.error(f"Simulation failed: {e}")

results = st.session_state.get("playoff_results")
if results:
    # User's playoff probability
    if user_team_name and user_team_name in results:
        user_result = results[user_team_name]
        playoff_prob = user_result.get("playoff_prob", 0.5)

        if playoff_prob > 0.6:
            prob_color = T["green"]
        elif playoff_prob > 0.4:
            prob_color = T["hot"]
        else:
            prob_color = T["danger"]

        st.markdown(
            f'<div class="glass" style="text-align:center;padding:24px;margin:16px 0;'
            f'border:2px solid {prob_color};">'
            f'<div style="font-family:Bebas Neue,sans-serif;font-size:48px;color:{prob_color};">'
            f"{playoff_prob:.0%}</div>"
            f'<div style="color:{T["tx2"]};font-size:16px;">Playoff Probability</div></div>',
            unsafe_allow_html=True,
        )

    # Projected standings table
    st.markdown("### Projected Standings")
    standings_rows = []
    for team, data in sorted(results.items(), key=lambda x: -x[1].get("playoff_prob", 0)):
        avg_w = data.get("avg_wins", 0)
        avg_l = data.get("avg_losses", 0)
        pp = data.get("playoff_prob", 0)
        is_user = team == user_team_name

        standings_rows.append(
            {
                "Team": f"** {team} **" if is_user else team,
                "Avg W": f"{avg_w:.1f}",
                "Avg L": f"{avg_l:.1f}",
                "Playoff %": f"{pp:.0%}",
            }
        )

    render_styled_table(pd.DataFrame(standings_rows))

    # Rank distribution for user's team
    if user_team_name and user_team_name in results:
        rank_dist = results[user_team_name].get("rank_distribution", [])
        if rank_dist:
            st.markdown("### Your Rank Distribution")
            st.caption("Percentage of simulations where you finish at each position.")
            total = sum(rank_dist)
            if total > 0:
                dist_rows = []
                for i, count in enumerate(rank_dist):
                    pct = count / total * 100
                    if pct > 0.5:  # Only show ranks with >0.5% probability
                        dist_rows.append(
                            {
                                "Rank": f"{i + 1}",
                                "Probability": f"{pct:.1f}%",
                                "Bar": "|" * int(pct / 2),
                            }
                        )
                if dist_rows:
                    render_styled_table(pd.DataFrame(dist_rows))
else:
    st.info("Click 'Simulate Season' to run the playoff odds simulation.")
