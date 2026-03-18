"""Weekly Recap — Review matchup results and plan next week's moves."""

import logging

import pandas as pd
import streamlit as st

from src.database import (
    init_db,
    load_league_rosters,
    load_league_standings,
    load_player_pool,
)
from src.league_manager import get_team_roster
from src.ui_shared import T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig, SGPCalculator

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Weekly Recap", page_icon="", layout="wide")

init_db()
inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>WEEKLY RECAP</span></div></div>',
    unsafe_allow_html=True,
)

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
    st.warning("Weekly Recap requires standings and roster data. Connect your Yahoo league in Connect League.")
    st.stop()

# Get user team
user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified.")
    st.stop()

user_team_name = str(user_teams.iloc[0]["team_name"])

# Build team totals from standings
all_team_totals: dict[str, dict[str, float]] = {}
if "category" in standings.columns:
    for _, row in standings.iterrows():
        team = str(row.get("team_name", ""))
        cat = str(row.get("category", "")).strip()
        val = float(row.get("total", 0))
        all_team_totals.setdefault(team, {})[cat] = val

my_totals = all_team_totals.get(user_team_name, {})

if not my_totals:
    st.warning("No standings data for your team.")
    st.stop()

# ── Opponent Selection ──────────────────────────────────────────────────

other_teams = sorted([t for t in all_team_totals.keys() if t != user_team_name])
if not other_teams:
    st.info("No opponent data available.")
    st.stop()

opponent = st.selectbox("Select this week's opponent:", other_teams, key="recap_opponent")
opp_totals = all_team_totals.get(opponent, {})

# ── Category Comparison ─────────────────────────────────────────────────

st.markdown(f"### {user_team_name} vs {opponent}")

my_wins = 0
opp_wins = 0
ties = 0
cat_rows = []

for cat in config.all_categories:
    my_val = my_totals.get(cat, 0)
    opp_val = opp_totals.get(cat, 0)

    is_inverse = cat in config.inverse_stats
    is_rate = cat in config.rate_stats

    # Determine winner
    if is_inverse:
        if my_val < opp_val:
            result = "WIN"
            my_wins += 1
        elif my_val > opp_val:
            result = "LOSS"
            opp_wins += 1
        else:
            result = "TIE"
            ties += 1
    else:
        if my_val > opp_val:
            result = "WIN"
            my_wins += 1
        elif my_val < opp_val:
            result = "LOSS"
            opp_wins += 1
        else:
            result = "TIE"
            ties += 1

    # Format values
    if is_rate:
        my_str = f"{my_val:.3f}"
        opp_str = f"{opp_val:.3f}"
        margin = f"{abs(my_val - opp_val):.3f}"
    else:
        my_str = f"{my_val:.0f}"
        opp_str = f"{opp_val:.0f}"
        margin = f"{abs(my_val - opp_val):.0f}"

    cat_rows.append(
        {
            "Category": cat,
            "You": my_str,
            "Opponent": opp_str,
            "Margin": margin,
            "Result": result,
        }
    )

# Score banner
if my_wins > opp_wins:
    score_color = T["green"]
    verdict = "WIN"
elif my_wins < opp_wins:
    score_color = T["danger"]
    verdict = "LOSS"
else:
    score_color = T["hot"]
    verdict = "TIE"

recap_html = (
    f'<div class="glass" style="text-align:center;padding:20px;margin:12px 0;'
    f'border:2px solid {score_color};">'
    f'<span style="font-family:Bebas Neue,sans-serif;font-size:32px;color:{score_color};">'
    f"{verdict}: {user_team_name} {my_wins} - {opp_wins} {opponent}</span>"
    f'<div style="color:{T["tx2"]};font-size:14px;margin-top:4px;">'
    f"Categories Won: {my_wins} | Lost: {opp_wins} | Tied: {ties}</div></div>"
)
st.markdown(recap_html, unsafe_allow_html=True)

render_styled_table(pd.DataFrame(cat_rows))

# ── Top Contributors / Underperformers ──────────────────────────────────

st.markdown("### Key Players")

user_roster = get_team_roster(user_team_name)
if not user_roster.empty:
    sgp_calc = SGPCalculator(config)

    player_sgps = []
    for _, player in user_roster.iterrows():
        try:
            total = sgp_calc.total_sgp(player)
            player_sgps.append(
                {
                    "Player": str(player.get("name", "Unknown")),
                    "Positions": str(player.get("positions", "")),
                    "Total SGP": total,
                }
            )
        except Exception:
            continue

    if player_sgps:
        sgp_df = pd.DataFrame(player_sgps).sort_values("Total SGP", ascending=False)

        col_top, col_bottom = st.columns(2)
        with col_top:
            st.markdown(
                f'<div style="color:{T["green"]};font-weight:700;">Top Contributors</div>', unsafe_allow_html=True
            )
            top_3 = sgp_df.head(3).copy()
            top_3["Total SGP"] = top_3["Total SGP"].map(lambda x: f"{x:.2f}")
            render_styled_table(top_3)

        with col_bottom:
            st.markdown(
                f'<div style="color:{T["danger"]};font-weight:700;">Lowest Producers</div>', unsafe_allow_html=True
            )
            bottom_3 = sgp_df.tail(3).copy()
            bottom_3["Total SGP"] = bottom_3["Total SGP"].map(lambda x: f"{x:.2f}")
            render_styled_table(bottom_3)

# ── Suggested Moves ─────────────────────────────────────────────────────

st.markdown("### Suggested Moves for Next Week")

# Identify weak categories (losses)
losses = [r["Category"] for r in cat_rows if r["Result"] == "LOSS"]
close_losses = [r for r in cat_rows if r["Result"] == "LOSS"]

if losses:
    st.markdown("**Categories to target:** " + ", ".join(losses))
    st.caption(
        "Check the Waiver Wire page for add/drop recommendations targeting these categories. "
        "Visit the Weekly Dashboard for start/sit decisions and streaming pitcher suggestions."
    )
else:
    st.success("You won all categories — no urgent moves needed. Focus on maintaining your lead.")
