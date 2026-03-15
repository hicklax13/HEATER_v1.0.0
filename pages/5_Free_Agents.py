"""Free Agents — Rank available players by marginal value to your team."""

import logging
import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_player_pool
from src.in_season import rank_free_agents
from src.league_manager import get_free_agents, get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Free Agents", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>FREE AGENT RANKINGS</span></div></div>',
    unsafe_allow_html=True,
)

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Get user roster
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_fa"):
            client = st.session_state.get("yahoo_client")
            if client:
                progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                try:
                    progress.progress(30, text="Fetching league standings...")
                    sync_result = client.sync_to_db()
                    progress.progress(100, text="Sync complete!")
                    standings_count = sync_result.get("standings", 0) if sync_result else 0
                    rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                    if rosters_count > 0:
                        st.success(f"Synced {rosters_count} roster entries and {standings_count} standing entries.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(
                            f"Sync completed but Yahoo returned no roster data "
                            f"(standings: {standings_count}). This may mean the league "
                            f"season hasn't started yet on Yahoo, or rosters haven't been set."
                        )
                except Exception as e:
                    progress.empty()
                    st.error(f"Sync failed: {e}")
            else:
                st.error("Yahoo client not found in session. Return to Connect League and reconnect.")
    else:
        st.warning(
            "No league data loaded. Connect your Yahoo league in Connect League, or league data will load automatically on next app launch."
        )
    st.stop()
else:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if user_teams.empty:
        st.warning("No user team identified.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        user_roster = get_team_roster(user_team_name)
        # Remap roster IDs to player pool IDs via name matching
        # (league_rosters may use Yahoo IDs while player_pool uses MLB Stats API IDs)
        name_to_pool_id = dict(zip(pool["name"], pool["player_id"])) if not pool.empty else {}
        user_roster_ids = []
        if not user_roster.empty and "name" in user_roster.columns:
            for _, r in user_roster.iterrows():
                pname = r.get("name", "")
                pool_pid = name_to_pool_id.get(pname)
                if pool_pid is not None:
                    user_roster_ids.append(pool_pid)
                else:
                    user_roster_ids.append(r["player_id"])
        else:
            user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

        # Get free agents
        fa_pool = get_free_agents(pool)
        if fa_pool.empty:
            st.info("No free agents available (all players are rostered).")
        else:
            # Position filter pills
            positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
            pill_cols = st.columns(len(positions))
            pos_filter = st.session_state.get("fa_pos_filter", "All")

            for i, pos in enumerate(positions):
                with pill_cols[i]:
                    btn_type = "primary" if pos_filter == pos else "secondary"
                    if st.button(pos, key=f"fa_pill_{pos}", type=btn_type, width="stretch"):
                        st.session_state.fa_pos_filter = pos
                        st.rerun()

            if pos_filter != "All":
                fa_pool = fa_pool[fa_pool["positions"].str.contains(pos_filter, na=False)]

            if fa_pool.empty:
                st.info(f"No free agents at position: {pos_filter}")
            else:
                # Rank
                fa_progress = st.progress(0, text="Computing marginal values for free agents...")
                try:
                    fa_progress.progress(20, text="Evaluating category-need weighting...")
                    ranked = rank_free_agents(user_roster_ids, fa_pool, pool, config)
                    fa_progress.progress(100, text="Free agent rankings complete!")
                except Exception as e:
                    logger.exception("Failed to rank free agents")
                    st.error(f"Error computing free agent rankings: {e}")
                    ranked = pd.DataFrame()
                time.sleep(0.3)
                fa_progress.empty()

                if ranked.empty:
                    st.info("No ranked free agents found.")
                else:
                    st.subheader(f"Top Free Agents ({len(ranked)} available)")
                    display_df = ranked[
                        ["player_name", "positions", "marginal_value", "best_category", "best_cat_impact"]
                    ].copy()
                    display_df["marginal_value"] = display_df["marginal_value"].map(lambda x: f"{x:.2f}")
                    display_df["best_cat_impact"] = display_df["best_cat_impact"].map(lambda x: f"{x:.2f}")
                    display_df = display_df.rename(
                        columns={
                            "player_name": "Player",
                            "positions": "Position",
                            "marginal_value": "Marginal Value",
                            "best_category": "Best Category",
                            "best_cat_impact": "Category Impact",
                        }
                    )
                    render_styled_table(display_df)
                    st.caption(METRIC_TOOLTIPS["marginal_value"])
