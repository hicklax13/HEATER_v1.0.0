"""Free Agents — Rank available players by marginal value to your team."""

import streamlit as st
import pandas as pd

from src.database import init_db, load_player_pool, load_league_rosters
from src.in_season import rank_free_agents
from src.league_manager import get_team_roster, get_free_agents
from src.valuation import LeagueConfig
from src.ui_shared import THEME, T

st.set_page_config(page_title="Free Agents", page_icon="🏷️", layout="wide")

init_db()

st.markdown(
    f"""<style>
    .stApp {{ background-color: {T['bg']}; }}
    h1, h2, h3 {{ color: {T['amber']}; font-family: 'Oswald', sans-serif; }}
    </style>""",
    unsafe_allow_html=True,
)

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
    st.warning("No league rosters loaded. Import your league data first.")
    st.stop()

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified.")
    st.stop()

user_team_name = user_teams.iloc[0]["team_name"]
user_roster = get_team_roster(user_team_name)
user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

# Get free agents
fa_pool = get_free_agents(pool)
if fa_pool.empty:
    st.info("No free agents available (all players are rostered).")
    st.stop()

# Filters
col1, col2 = st.columns([1, 2])
with col1:
    positions = sorted(set(
        pos for poslist in fa_pool["positions"].dropna()
        for pos in str(poslist).split(",")
    ))
    pos_filter = st.selectbox("Position Filter", ["All"] + positions)

if pos_filter != "All":
    fa_pool = fa_pool[fa_pool["positions"].str.contains(pos_filter, na=False)]

if fa_pool.empty:
    st.info(f"No free agents at position: {pos_filter}")
    st.stop()

# Rank
with st.spinner("Computing marginal values..."):
    ranked = rank_free_agents(user_roster_ids, fa_pool, pool, config)

if ranked.empty:
    st.info("No ranked free agents found.")
else:
    st.subheader(f"Top Free Agents ({len(ranked)} available)")
    st.dataframe(
        ranked[["player_name", "positions", "marginal_value", "best_category", "best_cat_impact"]],
        use_container_width=True,
        hide_index=True,
    )
