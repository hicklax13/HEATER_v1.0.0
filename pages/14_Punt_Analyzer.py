"""Punt Analyzer — Simulate different category punt strategies."""

import logging
import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings, load_player_pool
from src.ui_shared import inject_custom_css, render_styled_table
from src.valuation import LeagueConfig, SGPCalculator

_HAS_CATEGORY_ANALYSIS = True
try:
    from src.engine.portfolio.category_analysis import category_gap_analysis  # noqa: F401
except ImportError:
    _HAS_CATEGORY_ANALYSIS = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Punt Analyzer", page_icon="", layout="wide")

init_db()
inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>PUNT STRATEGY SIMULATOR</span></div></div>',
    unsafe_allow_html=True,
)

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# ── Category Selection ──────────────────────────────────────────────────────

st.markdown(
    "Select categories to **punt** (ignore). The tool will recompute player values "
    "and show how your roster and available players change in value."
)

all_cats = config.all_categories
punt_cats = st.multiselect(
    "Categories to punt:",
    options=all_cats,
    default=[],
    key="punt_cats",
    help="Select categories you want to intentionally concede in your H2H matchups.",
)

if not punt_cats:
    st.info("Select one or more categories above to see the punt analysis.")
    st.stop()

# ── Compute Original Values ─────────────────────────────────────────────────

progress = st.progress(0, text="Computing original player values...")

try:
    # Original values
    progress.progress(20, text="Computing baseline values...")
    sgp_orig = SGPCalculator(config)

    # Punt-adjusted values: zero out punted category denominators
    punt_config = LeagueConfig()
    # Set punted category weights to effectively zero by using huge denominators
    for cat in punt_cats:
        punt_config.sgp_denominators[cat] = 999999.0

    progress.progress(50, text="Computing punt-adjusted values...")
    sgp_punt = SGPCalculator(punt_config)

    # Compute total SGP for each player under both strategies
    orig_sgps = []
    punt_sgps = []
    for _, player in pool.iterrows():
        orig_sgps.append(sgp_orig.total_sgp(player))
        punt_sgps.append(sgp_punt.total_sgp(player))

    pool = pool.copy()
    pool["original_sgp"] = orig_sgps
    pool["punt_sgp"] = punt_sgps
    pool["value_change"] = pool["punt_sgp"] - pool["original_sgp"]

    progress.progress(100, text="Analysis complete!")
except Exception as e:
    progress.empty()
    logger.exception("Punt analysis failed")
    st.error(f"Analysis failed: {e}")
    st.stop()

time.sleep(0.3)
progress.empty()

# ── Summary ─────────────────────────────────────────────────────────────────

st.markdown(f"### Punting: {', '.join(punt_cats)}")

# Show which categories remain active
active_cats = [c for c in all_cats if c not in punt_cats]
active_str = ", ".join(active_cats)
st.markdown(f"**Active categories ({len(active_cats)}):** {active_str}")
st.markdown(f"**Punted categories ({len(punt_cats)}):** {', '.join(punt_cats)}")

# ── Biggest Winners (value increases under punt) ────────────────────────────

st.markdown("### Biggest Value Gainers Under Punt Strategy")
st.caption("Players whose value increases most when punted categories are removed.")

gainers = pool.nlargest(15, "value_change")[
    ["player_name", "positions", "team", "original_sgp", "punt_sgp", "value_change"]
].copy()
gainers["original_sgp"] = gainers["original_sgp"].map(lambda x: f"{x:.2f}")
gainers["punt_sgp"] = gainers["punt_sgp"].map(lambda x: f"{x:.2f}")
gainers["value_change"] = gainers["value_change"].map(lambda x: f"{x:+.2f}")
gainers = gainers.rename(
    columns={
        "player_name": "Player",
        "positions": "Pos",
        "team": "Team",
        "original_sgp": "Original SGP",
        "punt_sgp": "Punt SGP",
        "value_change": "Change",
    }
)
render_styled_table(gainers)

# ── Biggest Losers (value decreases under punt) ────────────────────────────

st.markdown("### Biggest Value Losers Under Punt Strategy")
st.caption("Players whose value decreases when punted categories are removed — potential trade targets.")

losers = pool.nsmallest(15, "value_change")[
    ["player_name", "positions", "team", "original_sgp", "punt_sgp", "value_change"]
].copy()
losers["original_sgp"] = losers["original_sgp"].map(lambda x: f"{x:.2f}")
losers["punt_sgp"] = losers["punt_sgp"].map(lambda x: f"{x:.2f}")
losers["value_change"] = losers["value_change"].map(lambda x: f"{x:+.2f}")
losers = losers.rename(
    columns={
        "player_name": "Player",
        "positions": "Pos",
        "team": "Team",
        "original_sgp": "Original SGP",
        "punt_sgp": "Punt SGP",
        "value_change": "Change",
    }
)
render_styled_table(losers)

# ── Standings Impact ────────────────────────────────────────────────────────

if _HAS_CATEGORY_ANALYSIS:
    standings = load_league_standings()
    rosters = load_league_rosters()

    if not standings.empty and not rosters.empty:
        user_teams = rosters[rosters["is_user_team"] == 1]
        if not user_teams.empty:
            user_team_name = str(user_teams.iloc[0]["team_name"])

            all_team_totals: dict[str, dict[str, float]] = {}
            if "category" in standings.columns:
                for _, row in standings.iterrows():
                    team = str(row.get("team_name", ""))
                    cat = str(row.get("category", "")).strip()
                    val = float(row.get("total", 0))
                    all_team_totals.setdefault(team, {})[cat] = val

            user_totals = all_team_totals.get(user_team_name, {})

            if user_totals and all_team_totals:
                st.markdown("### Standings Impact")
                st.caption("How your current standings ranks change when focusing only on non-punted categories.")

                impact_rows = []
                for cat in all_cats:
                    rank_vals = sorted(
                        [t.get(cat, 0) for t in all_team_totals.values()],
                        reverse=(cat not in config.inverse_stats),
                    )
                    my_val = user_totals.get(cat, 0)
                    my_rank = 1
                    for v in rank_vals:
                        if cat in config.inverse_stats:
                            if v < my_val:
                                my_rank += 1
                        else:
                            if v > my_val:
                                my_rank += 1

                    status = "PUNT" if cat in punt_cats else "ACTIVE"
                    impact_rows.append(
                        {
                            "Category": cat,
                            "Your Total": f"{my_val:.3f}" if cat in config.rate_stats else f"{my_val:.0f}",
                            "Rank": f"{my_rank}/12",
                            "Status": status,
                        }
                    )

                render_styled_table(pd.DataFrame(impact_rows))

                # Compute effective standings points (only from active categories)
                active_points = sum(13 - int(r["Rank"].split("/")[0]) for r in impact_rows if r["Status"] == "ACTIVE")
                punt_points = sum(13 - int(r["Rank"].split("/")[0]) for r in impact_rows if r["Status"] == "PUNT")
                total_points = active_points + punt_points

                st.metric(
                    "Standings Points from Active Categories",
                    f"{active_points}",
                    help=f"Total points: {total_points} (active: {active_points}, punted: {punt_points})",
                )
