"""Free Agents — Rank available players by marginal value to your team."""

import logging

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_player_pool
from src.in_season import rank_free_agents
from src.league_manager import get_free_agents, get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, inject_custom_css, render_theme_toggle
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Free Agents", page_icon="🏷️", layout="wide")

init_db()

inject_custom_css()
render_theme_toggle()

st.title("🏷️ Free Agent Rankings")

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Get user roster
rosters = load_league_rosters()
if rosters.empty:
    st.warning("No league data loaded. Import your league rosters in the main app (Setup Step 3).")
    st.stop()
else:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if user_teams.empty:
        st.warning("No user team identified.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        user_roster = get_team_roster(user_team_name)
        user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

        # Get free agents
        fa_pool = get_free_agents(pool)
        if fa_pool.empty:
            st.info("No free agents available (all players are rostered).")
        else:
            # Filters
            col1, col2 = st.columns([1, 2])
            with col1:
                positions = sorted(
                    set(
                        pos.strip()
                        for poslist in fa_pool["positions"].dropna()
                        for pos in str(poslist).split(",")
                        if pos.strip()
                    )
                )
                pos_filter = st.selectbox("Position Filter", ["All"] + positions)

            if pos_filter != "All":
                fa_pool = fa_pool[fa_pool["positions"].str.contains(pos_filter, na=False)]

            if fa_pool.empty:
                st.info(f"No free agents at position: {pos_filter}")
            else:
                # Rank
                with st.spinner("Computing marginal values..."):
                    try:
                        ranked = rank_free_agents(user_roster_ids, fa_pool, pool, config)
                    except Exception as e:
                        logger.exception("Failed to rank free agents")
                        st.error(f"Error computing free agent rankings: {e}")
                        ranked = pd.DataFrame()

                if ranked.empty:
                    st.info("No ranked free agents found.")
                else:
                    st.subheader(f"Top Free Agents ({len(ranked)} available)")
                    st.dataframe(
                        ranked[["player_name", "positions", "marginal_value", "best_category", "best_cat_impact"]],
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(METRIC_TOOLTIPS["marginal_value"])
