"""Waiver Wire — Intelligent add/drop recommendations powered by LP-verified swap analysis."""

import logging
import time
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings, load_player_pool
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig
from src.waiver_wire import classify_category_priority, compute_add_drop_recommendations

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Waiver Wire", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>WAIVER WIRE</span></div></div>',
    unsafe_allow_html=True,
)

# ── Load player pool ─────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# ── Compute weeks remaining from current date ────────────────────────
# MLB regular season typically ends late September
SEASON_END = datetime(2026, 9, 27, tzinfo=UTC)
now = datetime.now(UTC)
weeks_remaining = max(1, int((SEASON_END - now).days / 7))

# ── Load rosters ─────────────────────────────────────────────────────

rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_ww"):
            client = st.session_state.get("yahoo_client")
            if client:
                progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                try:
                    progress.progress(30, text="Fetching league standings...")
                    sync_result = client.sync_to_db()
                    progress.progress(100, text="Sync complete!")
                    rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                    standings_count = sync_result.get("standings", 0) if sync_result else 0
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
user_roster = get_team_roster(user_team_name)

# Remap roster IDs to player pool IDs via name matching
name_to_pool_id = dict(zip(pool["player_name"], pool["player_id"])) if not pool.empty else {}
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

if not user_roster_ids:
    st.warning("User roster is empty. Sync your league data first.")
    st.stop()

# ── Load standings for category priority ─────────────────────────────

standings = load_league_standings()
all_team_totals: dict[str, dict[str, float]] = {}
if not standings.empty and "category" in standings.columns:
    for _, row in standings.iterrows():
        team = row["team_name"]
        cat = str(row["category"]).strip()
        all_team_totals.setdefault(team, {})[cat] = float(row.get("total", 0))

# Build user totals from standings or roster
user_totals: dict[str, float] = all_team_totals.get(user_team_name, {})

# ── Category Priority Banner ─────────────────────────────────────────

if all_team_totals and user_totals:
    st.subheader("Category Priorities")
    try:
        priorities = classify_category_priority(user_totals, all_team_totals, user_team_name, weeks_remaining, config)
    except Exception as e:
        logger.exception("Failed to classify category priorities")
        priorities = {}

    if priorities:
        pills_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">'
        for cat in config.all_categories:
            tier = priorities.get(cat, "ATTACK")
            if tier == "ATTACK":
                bg = T["green"]
            elif tier == "DEFEND":
                bg = T["sky"]
            else:
                bg = "#6b7280"
            pills_html += (
                f'<span style="background:{bg};color:#ffffff;padding:6px 14px;'
                f"border-radius:20px;font-weight:700;font-size:0.85rem;"
                f'letter-spacing:0.5px;">{cat}: {tier}</span>'
            )
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)
        st.caption(
            "ATTACK: winnable category, target adds here. "
            "DEFEND: protect your lead, avoid drops here. "
            "IGNORE: punt or dominant, low priority."
        )
else:
    st.info(
        "Category priorities require league standings data. "
        "Connect your Yahoo league or import standings for priority analysis."
    )

# ── Settings Controls ────────────────────────────────────────────────

st.markdown("---")
ctrl_c1, ctrl_c2 = st.columns(2)
with ctrl_c1:
    max_moves = st.number_input(
        "Maximum Moves",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of add/drop pairs to recommend.",
    )
with ctrl_c2:
    st.markdown(f"**Weeks Remaining:** {weeks_remaining}")

# ── Position Filter Pills ────────────────────────────────────────────

positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
pill_cols = st.columns(len(positions))
pos_filter = st.session_state.get("ww_pos_filter", "All")

for i, pos in enumerate(positions):
    with pill_cols[i]:
        btn_type = "primary" if pos_filter == pos else "secondary"
        if st.button(pos, key=f"ww_pill_{pos}", type=btn_type, width="stretch"):
            st.session_state.ww_pos_filter = pos
            st.rerun()

# ── Compute Recommendations ──────────────────────────────────────────

st.markdown("---")
st.subheader("Recommended Moves")

standings_totals = all_team_totals if all_team_totals else None

ww_progress = st.progress(0, text="Analyzing waiver wire...")
try:
    ww_progress.progress(10, text="Scanning free agent pool...")
    recommendations = compute_add_drop_recommendations(
        user_roster_ids=user_roster_ids,
        player_pool=pool,
        config=config,
        standings_totals=standings_totals,
        user_team_name=user_team_name,
        weeks_remaining=weeks_remaining,
        max_moves=int(max_moves),
    )
    ww_progress.progress(100, text="Waiver wire analysis complete!")
except Exception as e:
    logger.exception("Failed to compute add/drop recommendations")
    st.error(f"Error computing recommendations: {e}")
    recommendations = []

time.sleep(0.3)
ww_progress.empty()

# ── Apply position filter ────────────────────────────────────────────

if pos_filter != "All" and recommendations:
    recommendations = [rec for rec in recommendations if pos_filter in str(rec.get("add_positions", "")).split(",")]

# ── Display Recommendations ──────────────────────────────────────────

if not recommendations:
    st.info("No recommendations available. Your roster may already be optimized, or no positive swaps exist.")
else:
    for idx, rec in enumerate(recommendations):
        add_name = rec.get("add_name", "Unknown")
        add_pos = rec.get("add_positions", "")
        drop_name = rec.get("drop_name", "Unknown")
        drop_pos = rec.get("drop_positions", "")
        net_sgp = rec.get("net_sgp_delta", 0.0)
        sust = rec.get("sustainability_score", 0.5)
        cat_impact = rec.get("category_impact", {})
        reasoning = rec.get("reasoning", [])

        # Color coding
        sgp_color = T["green"] if net_sgp > 0 else "#dc2626"
        if sust > 0.6:
            sust_color = T["green"]
        elif sust > 0.3:
            sust_color = "#d97706"
        else:
            sust_color = "#dc2626"

        # Top 3 improved categories
        sorted_cats = sorted(cat_impact.items(), key=lambda x: x[1], reverse=True)
        top_cats = [(c, v) for c, v in sorted_cats if v > 0][:3]
        top_cats_html = ""
        if top_cats:
            cat_parts = []
            for cat_name, cat_val in top_cats:
                cat_parts.append(
                    f'<span style="background:{T["green"]};color:#fff;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.8rem;font-weight:600;">'
                    f"{cat_name} +{cat_val:.2f}</span>"
                )
            top_cats_html = " ".join(cat_parts)

        # Reasoning bullets
        reasoning_html = ""
        if reasoning:
            reasoning_items = "".join(f"<li>{r}</li>" for r in reasoning)
            reasoning_html = (
                f'<ul style="margin:6px 0 0 16px;padding:0;font-size:0.85rem;color:#374151;">{reasoning_items}</ul>'
            )

        # Glass card
        card_html = f"""
        <div style="background:rgba(255,255,255,0.75);backdrop-filter:blur(12px);
                    border:1px solid rgba(0,0,0,0.08);border-radius:14px;
                    padding:20px 24px;margin-bottom:16px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                <span style="font-size:1.15rem;font-weight:800;color:#1a1a2e;">
                    #{idx + 1}
                </span>
                <span style="font-size:1.05rem;font-weight:700;color:{T["green"]};">
                    Add {add_name}
                </span>
                <span style="font-size:0.85rem;color:#6b7280;">({add_pos})</span>
                <span style="font-size:1.05rem;color:#9ca3af;font-weight:700;">
                    &rarr;
                </span>
                <span style="font-size:1.05rem;font-weight:700;color:#dc2626;">
                    Drop {drop_name}
                </span>
                <span style="font-size:0.85rem;color:#6b7280;">({drop_pos})</span>
            </div>
            <div style="display:flex;gap:24px;align-items:center;margin-bottom:10px;">
                <div>
                    <span style="font-size:0.78rem;color:#6b7280;font-weight:600;">
                        Net Standings Gained Points Delta
                    </span><br/>
                    <span style="font-size:1.2rem;font-weight:800;color:{sgp_color};">
                        {"+" if net_sgp > 0 else ""}{net_sgp:.2f}
                    </span>
                </div>
                <div>
                    <span style="font-size:0.78rem;color:#6b7280;font-weight:600;">
                        Sustainability Score
                    </span><br/>
                    <span style="font-size:1.2rem;font-weight:800;color:{sust_color};">
                        {sust:.2f}
                    </span>
                </div>
                <div>
                    <span style="font-size:0.78rem;color:#6b7280;font-weight:600;">
                        Top Category Gains
                    </span><br/>
                    {top_cats_html if top_cats_html else '<span style="font-size:0.85rem;color:#9ca3af;">None</span>'}
                </div>
            </div>
            {reasoning_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # ── Summary Table ────────────────────────────────────────────────
    st.subheader("Recommendations Summary")
    summary_data = []
    for rec in recommendations:
        summary_data.append(
            {
                "Add": rec.get("add_name", ""),
                "Position": rec.get("add_positions", ""),
                "Drop": rec.get("drop_name", ""),
                "Net Standings Gained Points": f"{rec.get('net_sgp_delta', 0.0):+.2f}",
                "Sustainability": f"{rec.get('sustainability_score', 0.0):.2f}",
            }
        )
    summary_df = pd.DataFrame(summary_data)
    render_styled_table(summary_df)
    st.caption(METRIC_TOOLTIPS.get("sgp", ""))
