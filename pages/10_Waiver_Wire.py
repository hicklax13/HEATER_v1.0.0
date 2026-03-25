"""Waiver Wire — Add/drop recommendations based on your roster needs."""

import logging
import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_player_pool
from src.league_manager import get_team_roster
from src.ui_shared import (
    THEME,
    inject_custom_css,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
)
from src.valuation import LeagueConfig

try:
    from src.waiver_wire import compute_add_drop_recommendations

    WAIVER_WIRE_AVAILABLE = True
except ImportError:
    WAIVER_WIRE_AVAILABLE = False

logger = logging.getLogger(__name__)

T = THEME

st.set_page_config(
    page_title="Heater | Waiver Wire",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()

inject_custom_css()

render_page_layout(
    "WAIVER WIRE",
    banner_teaser="Add/drop recommendations based on your roster needs",
    banner_icon="free_agents",
)

if not WAIVER_WIRE_AVAILABLE:
    st.error("The waiver wire module could not be loaded. Ensure all required dependencies are installed.")
    st.stop()

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
# Restore the 'name' column alias so waiver_wire internals can find it
pool["name"] = pool["player_name"]
config = LeagueConfig()

# ── Roster check ──────────────────────────────────────────────────────────────
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_waiver"):
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
            "No league data loaded. Connect your Yahoo league in Connect League, "
            "or league data will load automatically on next app launch."
        )
    st.stop()

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified.")
    st.stop()

user_team_name = user_teams.iloc[0]["team_name"]
if isinstance(user_team_name, bytes):
    user_team_name = user_team_name.decode("utf-8", errors="replace")

user_roster = get_team_roster(user_team_name)

# Remap roster IDs to player pool IDs via name matching
name_to_pool_id = dict(zip(pool["player_name"], pool["player_id"])) if not pool.empty else {}
user_roster_ids: list[int] = []
if not user_roster.empty and "name" in user_roster.columns:
    for _, r in user_roster.iterrows():
        pname = r.get("name", "")
        pool_pid = name_to_pool_id.get(pname)
        if pool_pid is not None:
            user_roster_ids.append(int(pool_pid))
        else:
            raw_pid = r.get("player_id")
            if raw_pid is not None:
                user_roster_ids.append(int(raw_pid))
else:
    user_roster_ids = user_roster["player_id"].astype(int).tolist() if not user_roster.empty else []

if not user_roster_ids:
    st.warning("Your roster appears to be empty. No waiver wire recommendations can be generated.")
    st.stop()

# ── Roster summary stats for context panel ───────────────────────────────────
num_cols_coerce = [
    "r",
    "hr",
    "rbi",
    "sb",
    "ab",
    "h",
    "bb",
    "hbp",
    "sf",
    "w",
    "l",
    "sv",
    "k",
    "ip",
    "er",
    "bb_allowed",
    "h_allowed",
]
for c in num_cols_coerce:
    if c in user_roster.columns:
        user_roster[c] = pd.to_numeric(user_roster[c], errors="coerce").fillna(0)

hitters_df = user_roster[user_roster["is_hitter"] == 1] if "is_hitter" in user_roster.columns else pd.DataFrame()
pitchers_df = user_roster[user_roster["is_hitter"] == 0] if "is_hitter" in user_roster.columns else pd.DataFrame()
n_hitters = len(hitters_df)
n_pitchers = len(pitchers_df)

# ── Position filter for waiver wire recommendations ──────────────────────────
POSITIONS = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]

ctx, main = render_context_columns()

# ── Context panel ─────────────────────────────────────────────────────────────
with ctx:
    # Roster summary card
    roster_html = (
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Total Players</span>'
        f'<span style="font-weight:600;color:{T["tx"]};">{len(user_roster_ids)}</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Hitters</span>'
        f'<span style="font-weight:600;color:{T["tx"]};">{n_hitters}</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Pitchers</span>'
        f'<span style="font-weight:600;color:{T["tx"]};">{n_pitchers}</span></div>'
    )
    render_context_card("Roster Summary", roster_html)

    # Position filter
    pos_filter = st.session_state.get("waiver_pos_filter", "All")
    render_context_card(
        "Filter by Position",
        f'<div style="font-size:12px;color:{T["tx2"]};font-family:Figtree,sans-serif;">'
        f"Select a position to narrow recommendations to players at that spot.</div>",
    )
    for pos in POSITIONS:
        btn_type = "primary" if pos_filter == pos else "secondary"
        if st.button(pos, key=f"waiver_pill_{pos}", type=btn_type, use_container_width=True):
            st.session_state.waiver_pos_filter = pos
            st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
with main:
    waiver_progress = st.progress(0, text="Analyzing category gaps and free agent pool...")
    recommendations: list[dict] = []
    try:
        waiver_progress.progress(20, text="Scoring drop candidates by roster impact...")
        waiver_progress.progress(50, text="Computing net swap values for top candidates...")
        from src.validation.dynamic_context import compute_weeks_remaining

        recommendations = compute_add_drop_recommendations(
            user_roster_ids=user_roster_ids,
            player_pool=pool,
            config=config,
            weeks_remaining=compute_weeks_remaining(),
            max_moves=5,
            max_fa_candidates=30,
            max_drop_candidates=8,
        )
        waiver_progress.progress(100, text="Waiver wire analysis complete!")
    except Exception as e:
        logger.exception("Failed to compute waiver wire recommendations")
        waiver_progress.empty()
        st.error(f"Error computing waiver wire recommendations: {e}")
        recommendations = []
    time.sleep(0.3)
    waiver_progress.empty()

    # Analytics transparency badge (data quality from bootstrap)
    from src.data_bootstrap import get_bootstrap_context

    _boot_ctx = get_bootstrap_context()
    if _boot_ctx:
        from src.ui_analytics_badge import render_analytics_badge

        render_analytics_badge(_boot_ctx)

    # Apply position filter
    pos_filter = st.session_state.get("waiver_pos_filter", "All")
    if pos_filter != "All" and recommendations:
        recommendations = [rec for rec in recommendations if pos_filter in str(rec.get("add_positions", "")).split(",")]

    # ── Recommended Adds table ────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-head">Recommended Adds</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
        f'font-family:Figtree,sans-serif;">'
        f"Free agents ranked by net Standings Gained Points improvement "
        f"after accounting for the recommended drop. Only positive-value swaps are shown.</div>",
        unsafe_allow_html=True,
    )

    # Build mlb_id lookup from pool for headshot rendering
    _pid_to_mlb = {}
    if "mlb_id" in pool.columns and "player_id" in pool.columns:
        _pid_to_mlb = dict(zip(pool["player_id"], pool["mlb_id"]))

    if not recommendations:
        st.info(
            "No waiver wire add recommendations found. "
            "This may mean the free agent pool is empty, your roster is already optimal, "
            "or no free agent improves your team's Standings Gained Points total."
        )
    else:
        adds_rows = []
        for rec in recommendations:
            sustainability_pct = int(rec.get("sustainability_score", 0.5) * 100)
            entry = {
                "Add": rec.get("add_name", ""),
                "Position": rec.get("add_positions", ""),
                "Net SGP Delta": f"{rec.get('net_sgp_delta', 0):.3f}",
                "Sustainability": f"{sustainability_pct}%",
                "Drop": rec.get("drop_name", ""),
            }
            if _pid_to_mlb:
                entry["mlb_id"] = _pid_to_mlb.get(rec.get("add_player_id"))
            adds_rows.append(entry)
        adds_df = pd.DataFrame(adds_rows)
        render_compact_table(adds_df, highlight_cols=["Net SGP Delta"])

        # Reasoning expanders
        for i, rec in enumerate(recommendations):
            reasons = rec.get("reasoning", [])
            if reasons:
                with st.expander(
                    f"Why: Add {rec.get('add_name', '')} / Drop {rec.get('drop_name', '')}",
                    expanded=False,
                ):
                    for reason in reasons:
                        st.markdown(
                            f'<div style="font-size:13px;color:{T["tx"]};'
                            f'padding:2px 0;font-family:Figtree,sans-serif;">'
                            f"{reason}</div>",
                            unsafe_allow_html=True,
                        )

        # Player card selector for add candidates
        add_names = [rec.get("add_name", "") for rec in recommendations]
        add_ids_raw = [rec.get("add_player_id") for rec in recommendations]
        add_ids = [int(pid) for pid in add_ids_raw if pid is not None]
        if add_names and add_ids:
            render_player_select(add_names, add_ids, key_suffix="waiver_adds")

    # ── Recommended Drops table ───────────────────────────────────────────────
    st.markdown(
        '<div class="sec-head" style="margin-top:24px;">Recommended Drops</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
        f'font-family:Figtree,sans-serif;">'
        f"Roster players paired with a recommended add, ordered by how little "
        f"Standings Gained Points they contribute to your lineup totals.</div>",
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.info("No drop recommendations available.")
    else:
        drops_rows = []
        for rec in recommendations:
            category_impact = rec.get("category_impact", {})
            # Show the most-impacted category from this swap
            if category_impact:
                top_cat = max(category_impact, key=lambda c: abs(category_impact[c]))
                top_delta = category_impact[top_cat]
                impact_str = f"{top_cat}: {top_delta:+.3f}"
            else:
                impact_str = ""

            entry = {
                "Drop": rec.get("drop_name", ""),
                "Position": rec.get("drop_positions", ""),
                "Replaced By": rec.get("add_name", ""),
                "Top Category Impact": impact_str,
            }
            if _pid_to_mlb:
                entry["mlb_id"] = _pid_to_mlb.get(rec.get("drop_player_id"))
            drops_rows.append(entry)
        drops_df = pd.DataFrame(drops_rows)
        render_compact_table(drops_df, highlight_cols=["Top Category Impact"])

        # Player card selector for drop candidates
        drop_names = [rec.get("drop_name", "") for rec in recommendations]
        drop_ids_raw = [rec.get("drop_player_id") for rec in recommendations]
        drop_ids = [int(pid) for pid in drop_ids_raw if pid is not None]
        if drop_names and drop_ids:
            render_player_select(drop_names, drop_ids, key_suffix="waiver_drops")
