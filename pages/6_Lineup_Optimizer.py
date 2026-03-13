"""Lineup Optimizer — Start/sit decisions using LP solver."""

import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, inject_custom_css, render_theme_toggle

try:
    from src.lineup_optimizer import LineupOptimizer

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

st.set_page_config(page_title="Lineup Optimizer", page_icon="", layout="wide")

init_db()

inject_custom_css()
render_theme_toggle()

st.title("Lineup Optimizer")

# ── Load user team ────────────────────────────────────────────────
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_lineup"):
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

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified in roster data.")
    st.stop()

user_team_name = user_teams.iloc[0]["team_name"]
st.markdown(f"**Team:** {user_team_name}")

roster = get_team_roster(user_team_name)
if roster is None or roster.empty:
    st.info("Your roster is empty. Import roster data first.")
    st.stop()

# ── Build optimizer config ────────────────────────────────────────
# Normalize player_name → name FIRST (before filling missing columns)
if "player_name" in roster.columns and "name" not in roster.columns:
    roster = roster.rename(columns={"player_name": "name"})

# Ensure required columns exist for the optimizer
required_cols = ["player_id", "name", "positions", "is_hitter"]
stat_cols = ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]
for col in required_cols + stat_cols:
    if col not in roster.columns:
        roster[col] = 0

# Build league config dict for optimizer
config = {
    "hitting_categories": ["r", "hr", "rbi", "sb", "avg"],
    "pitching_categories": ["w", "sv", "k", "era", "whip"],
    "roster_slots": {
        "C": 1,
        "1B": 1,
        "2B": 1,
        "3B": 1,
        "SS": 1,
        "OF": 3,
        "Util": 2,
        "SP": 2,
        "RP": 2,
        "P": 4,
    },
}

if not OPTIMIZER_AVAILABLE:
    st.error("Lineup Optimizer requires PuLP. Install it with: `pip install pulp`")
    st.stop()

optimizer = LineupOptimizer(roster, config)

# ── Optimize button ───────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    optimize_clicked = st.button("Optimize Lineup", type="primary")

if optimize_clicked:
    lp_progress = st.progress(0, text="Running Linear Programming solver...")
    lp_progress.progress(20, text="Building constraint matrix for roster slots...")
    result = optimizer.optimize_lineup()
    lp_progress.progress(100, text="Optimization complete!")
    time.sleep(0.3)
    lp_progress.empty()

    if not result or not result.get("lineup"):
        st.warning("Could not produce a valid lineup. Check your roster data.")
    else:
        st.success("Optimal lineup found!")

        # Display lineup
        lineup = result["lineup"]
        lineup_data = []
        for slot, player_name in sorted(lineup.items()):
            lineup_data.append({"Slot": slot, "Player": player_name})

        st.subheader("Recommended Lineup")
        st.dataframe(
            pd.DataFrame(lineup_data),
            hide_index=True,
            width="stretch",
        )

        # Display projected stats if available
        if "projected_stats" in result and result["projected_stats"]:
            st.subheader("Projected Category Totals")
            stats_df = pd.DataFrame([result["projected_stats"]])
            st.dataframe(stats_df, hide_index=True, width="stretch")

# ── Category Targeting ────────────────────────────────────────────
st.divider()
st.subheader("Category Targeting")

standings = load_league_standings()
if standings.empty:
    st.info(
        "Import league standings to see category targeting recommendations. "
        "The optimizer will identify where small gains yield the biggest standings jumps."
    )
else:
    weights = optimizer.category_targeting(standings, user_team_name)
    if weights:
        cat_full_names = {
            "r": "Runs",
            "hr": "Home Runs",
            "rbi": "Runs Batted In",
            "sb": "Stolen Bases",
            "avg": "Batting Average",
            "w": "Wins",
            "sv": "Saves",
            "k": "Strikeouts",
            "era": "Earned Run Average",
            "whip": "Walks + Hits per Inning Pitched",
        }
        st.markdown("**Priority categories** (higher weight = bigger standings impact):")
        weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for cat, weight in weights_sorted:
            bar_len = int(weight * 20)
            icon = (
                PAGE_ICONS["fire"]
                if weight > 1.5
                else PAGE_ICONS["trending_up"]
                if weight > 1.0
                else PAGE_ICONS["minus"]
            )
            display_name = cat_full_names.get(cat, cat.upper())
            st.markdown(
                f"{icon} **{display_name}**: {'█' * bar_len}{'░' * (20 - bar_len)} ({weight:.2f}x)",
                unsafe_allow_html=True,
            )
        st.caption(METRIC_TOOLTIPS["cat_targeting"])
    else:
        st.info("Could not compute targeting weights from standings data.")

# ── Current Roster Overview ───────────────────────────────────────
st.divider()
st.subheader("Current Roster")

# Add health info to roster overview
health_dict = {}
try:
    from src.database import get_connection

    conn = get_connection()
    try:
        injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
    finally:
        conn.close()

    if not injury_df.empty:
        health_badges = []
        for _, row in roster.iterrows():
            pid = row.get("player_id")
            pi = injury_df[injury_df["player_id"] == pid]
            if not pi.empty:
                hs = compute_health_score(pi["games_played"].tolist(), pi["games_available"].tolist())
                icon, _ = get_injury_badge(hs)
                health_badges.append(icon)
                health_dict[pid] = hs
            else:
                health_badges.append(
                    '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px;background:#84cc16;"></span>'
                )
                health_dict[pid] = 1.0
        roster["Health"] = health_badges
except Exception:
    pass

# Apply health penalty to LP objective
HEALTH_PENALTY_WEIGHT = 0.15
if health_dict and "projected_sgp" in roster.columns:
    roster["health_adjusted_sgp"] = roster.apply(
        lambda r: r["projected_sgp"] * (1.0 - HEALTH_PENALTY_WEIGHT * (1.0 - health_dict.get(r.get("player_id"), 1.0))),
        axis=1,
    )
    st.caption("Health-adjusted Standings Gained Points applies a 15% penalty weight based on injury history risk.")

# Two-start SP detection via MLB schedule
try:
    from datetime import UTC, datetime, timedelta

    import statsapi

    today = datetime.now(UTC)
    end = today + timedelta(days=7)
    schedule = statsapi.schedule(start_date=today.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))

    team_game_counts = {}
    for game in schedule:
        for team_key in ["home_name", "away_name"]:
            team_name = game.get(team_key, "")
            team_game_counts[team_name] = team_game_counts.get(team_name, 0) + 1

    # Flag SPs whose team has 2+ games in the period
    two_start_sps = []
    for _, row in roster.iterrows():
        if not row.get("is_hitter", True) and "SP" in str(row.get("positions", "")):
            team = row.get("team", "")
            if team_game_counts.get(team, 0) >= 2:
                two_start_sps.append(row.get("name", row.get("player_name", "")))

    if two_start_sps:
        st.info(f"Potential two-start Starting Pitchers this week: {', '.join(two_start_sps)}")
except Exception:
    pass  # Graceful degradation when MLB Stats API unavailable

display_cols = ["name", "positions", "is_hitter"] + [c for c in stat_cols if c in roster.columns]
if "Health" in roster.columns:
    display_cols = ["name", "positions", "is_hitter", "Health"] + [c for c in stat_cols if c in roster.columns]
st.dataframe(
    roster[[c for c in display_cols if c in roster.columns]],
    hide_index=True,
    width="stretch",
)
