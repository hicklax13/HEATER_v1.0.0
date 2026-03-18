"""Trade Values — Universal trade value chart (0-100) for all players."""

import logging
import math
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings, load_player_pool
from src.trade_value import (
    TIER_COLORS,
    compute_contextual_values,
    compute_trade_values,
    filter_by_position,
)
from src.ui_shared import METRIC_TOOLTIPS, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Trade Values", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>TRADE VALUE CHART</span></div></div>',
    unsafe_allow_html=True,
)

# ── Load Player Pool ──────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# ── Compute Weeks Remaining ───────────────────────────────────────
# MLB season ~26 weeks, ends approximately October 1.
season_end = datetime(2026, 10, 1, tzinfo=UTC)
now = datetime.now(UTC)
weeks_remaining = max(1, math.ceil((season_end - now).days / 7))
weeks_remaining = min(weeks_remaining, 26)

# ── Load Standings (long format) ─────────────────────────────────

standings = load_league_standings()
all_team_totals: dict[str, dict[str, float]] = {}
if not standings.empty and "category" in standings.columns:
    for _, srow in standings.iterrows():
        tname = srow.get("team_name", "")
        cat = str(srow.get("category", "")).strip()
        all_team_totals.setdefault(tname, {})[cat] = float(srow.get("total", 0))

# ── Detect User Team ─────────────────────────────────────────────

user_team_name = None
user_totals: dict[str, float] = {}
rosters = load_league_rosters()
if not rosters.empty:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if not user_teams.empty:
        user_team_name = user_teams.iloc[0]["team_name"]
        user_totals = all_team_totals.get(user_team_name, {})

# ── Compute Trade Values ─────────────────────────────────────────

try:
    trade_values = compute_trade_values(pool, config, standings if not standings.empty else None, weeks_remaining)
except Exception as e:
    logger.exception("Failed to compute trade values")
    st.error(f"Error computing trade values: {e}")
    st.stop()

if trade_values.empty:
    st.info("No trade values computed. Check that player data is loaded.")
    st.stop()

# ── Contextual Values (if standings available) ────────────────────

has_context = bool(user_team_name and user_totals and all_team_totals)
contextual_df = None
if has_context:
    try:
        contextual_df = compute_contextual_values(trade_values, user_totals, all_team_totals, user_team_name, config)
    except Exception as e:
        logger.warning("Contextual values unavailable: %s", e)
        contextual_df = None

# ── Controls Row ──────────────────────────────────────────────────

ctrl_col1, ctrl_col2 = st.columns([2, 1])

with ctrl_col1:
    search_query = st.text_input("Search players", placeholder="Type a player name...", key="tv_search")

with ctrl_col2:
    show_contextual = False
    if has_context and contextual_df is not None:
        view_mode = st.toggle("My Team Needs", value=False, key="tv_contextual_toggle")
        show_contextual = view_mode
    elif not has_context:
        st.caption("Connect Yahoo for team-need values")

# ── Position Pill Filters ─────────────────────────────────────────

positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
pill_cols = st.columns(len(positions))
pos_filter = st.session_state.get("tv_pos_filter", "All")

for i, pos in enumerate(positions):
    with pill_cols[i]:
        btn_type = "primary" if pos_filter == pos else "secondary"
        if st.button(pos, key=f"tv_pill_{pos}", type=btn_type, width="stretch"):
            st.session_state.tv_pos_filter = pos
            st.rerun()

# ── Apply Filters ─────────────────────────────────────────────────

active_df = contextual_df if show_contextual and contextual_df is not None else trade_values

# Position filter
filtered = filter_by_position(active_df, pos_filter)

# Search filter
if search_query:
    name_col = "name" if "name" in filtered.columns else "player_name"
    mask = filtered[name_col].str.contains(search_query, case=False, na=False)
    filtered = filtered[mask].copy()

if filtered.empty:
    st.info("No players match the current filters.")
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────

value_col = "contextual_value" if show_contextual and "contextual_value" in filtered.columns else "trade_value"
tier_col = "contextual_tier" if show_contextual and "contextual_tier" in filtered.columns else "tier"

total_count = len(filtered)
elite_count = len(filtered[filtered[tier_col] == "Elite"])
star_count = len(filtered[filtered[tier_col] == "Star"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Players", f"{total_count}")
m2.metric("Elite Tier", f"{elite_count}")
m3.metric("Star Tier", f"{star_count}")
m4.metric("Weeks Remaining", f"{weeks_remaining}")

if not all_team_totals:
    st.info(
        "No league standings loaded. Showing universal values with default weights. "
        "Connect your Yahoo league for standings-adjusted values."
    )

# ── Render Tier Sections ──────────────────────────────────────────

tier_order = ["Elite", "Star", "Solid Starter", "Flex", "Replacement"]

for tier_name in tier_order:
    tier_players = filtered[filtered[tier_col] == tier_name].copy()
    if tier_players.empty:
        continue

    tier_color = TIER_COLORS.get(tier_name, "#666666")

    st.markdown(
        f'<div style="margin-top:24px;margin-bottom:8px;padding:8px 16px;'
        f"background:linear-gradient(135deg, {tier_color}, {tier_color}dd);"
        f'border-radius:8px;display:inline-block;">'
        f'<span style="color:#ffffff;font-weight:700;font-size:16px;'
        f'letter-spacing:0.5px;font-family:Figtree,sans-serif;">'
        f"{tier_name} ({len(tier_players)})</span></div>",
        unsafe_allow_html=True,
    )

    # Build display dataframe
    name_col = "name" if "name" in tier_players.columns else "player_name"
    display = pd.DataFrame()
    display["Rank"] = tier_players["rank"].values
    display["Player"] = tier_players[name_col].values
    display["Pos"] = tier_players["positions"].values
    display["Team"] = tier_players["team"].values
    display["Trade Value"] = tier_players[value_col].apply(lambda v: f"{int(round(v))}").values
    display["Dollar Value"] = tier_players["dollar_value"].apply(lambda v: f"${v:.2f}").values

    render_styled_table(display, max_height=400)

# ── Tooltips ──────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    METRIC_TOOLTIPS.get(
        "sgp",
        "Standings Gained Points measure how much a player's stats move your team up in each category.",
    )
)
st.caption(
    "Trade Value (0-100) combines Standings Gained Points surplus with G-Score "
    "H2H variance adjustment. Higher values indicate more tradeable assets. "
    "Dollar values represent auction-equivalent pricing."
)
if show_contextual:
    st.caption(
        "My Team Needs mode adjusts values based on your category gaps. "
        "Players who fill weak categories are boosted; redundant strengths are discounted."
    )

# ── Positional Scarcity Waterfall ──────────────────────────────────────

with st.expander("Positional Scarcity Chart", expanded=False):
    st.caption("Shows the value drop-off at each position. Steeper drop = scarcer position.")

    try:
        from src.valuation import SGPCalculator, compute_replacement_levels

        sgp_calc = SGPCalculator(config)
        rep_levels = compute_replacement_levels(pool, config, sgp_calc)

        positions_chart = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
        scarcity_rows = []

        for pos in positions_chart:
            pos_players = pool[pool["positions"].apply(lambda p: pos in str(p).split(",") if pd.notna(p) else False)]
            if pos_players.empty:
                continue

            # Compute SGP for each player at this position
            sgps = []
            for _, p in pos_players.iterrows():
                sgps.append(sgp_calc.total_sgp(p))

            sgps_sorted = sorted(sgps, reverse=True)
            top_5_avg = sum(sgps_sorted[:5]) / min(5, len(sgps_sorted)) if sgps_sorted else 0
            top_10_avg = sum(sgps_sorted[:10]) / min(10, len(sgps_sorted)) if sgps_sorted else 0
            rep_sgp = rep_levels.get(pos, {}).get("total", 0) if isinstance(rep_levels.get(pos), dict) else 0

            # Scarcity score = drop-off steepness
            if len(sgps_sorted) >= 3:
                scarcity_score = (sgps_sorted[0] - sgps_sorted[min(2, len(sgps_sorted) - 1)]) / max(sgps_sorted[0], 0.1)
            else:
                scarcity_score = 0

            scarcity_rows.append(
                {
                    "Position": pos,
                    "Players": len(pos_players),
                    "Top 5 Avg SGP": f"{top_5_avg:.1f}",
                    "Top 10 Avg SGP": f"{top_10_avg:.1f}",
                    "Drop-off (1 to 3)": f"{scarcity_score:.2f}",
                }
            )

        if scarcity_rows:
            render_styled_table(pd.DataFrame(scarcity_rows))
            st.caption(
                "Drop-off measures how quickly value falls from the top player to the 3rd-best. "
                "Higher drop-off = scarcer position (C, SS typically highest)."
            )
    except Exception as e:
        st.error(f"Scarcity analysis failed: {e}")
