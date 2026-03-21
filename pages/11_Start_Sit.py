"""Start/Sit Advisor — Weekly lineup decisions based on matchup analysis."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)

# ── Optional advisor import ───────────────────────────────────────────

START_SIT_AVAILABLE = False
try:
    from src.start_sit import start_sit_recommendation

    START_SIT_AVAILABLE = True
except Exception:  # pragma: no cover
    logger.warning("start_sit module unavailable", exc_info=True)

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heater | Start/Sit Advisor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()

render_page_layout(
    "START/SIT ADVISOR",
    banner_teaser="Weekly lineup recommendations based on matchup analysis",
    banner_icon="lineup_optimizer",
)

# ── Import guard ──────────────────────────────────────────────────────

if not START_SIT_AVAILABLE:
    st.error(
        "The start/sit advisor module could not be loaded. "
        "Check that src/start_sit.py and its dependencies are installed."
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded. Run the bootstrap to fetch projections.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

rosters = load_league_rosters()

# Determine user team
user_team_name: str | None = None
user_roster: pd.DataFrame = pd.DataFrame()
user_player_ids: list[int] = []

if not rosters.empty:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if not user_teams.empty:
        user_team_name = str(user_teams.iloc[0]["team_name"])
        user_roster = get_team_roster(user_team_name)
        if not user_roster.empty:
            user_player_ids = user_roster["player_id"].tolist()

# ── Layout ────────────────────────────────────────────────────────────

ctx, main = render_context_columns()

# ── Context panel ─────────────────────────────────────────────────────

with ctx:
    # Matchup state info card
    matchup_state_label = "close"
    matchup_note = "No weekly totals provided. Using balanced (close) matchup strategy."

    if user_team_name:
        render_context_card(
            "Your Team",
            f'<p style="margin:0;font-size:13px;color:{THEME["tx1"]};">'
            f"{user_team_name}</p>"
            f'<p style="margin:4px 0 0;font-size:12px;color:{THEME["tx2"]};">'
            f"{len(user_player_ids)} rostered players loaded</p>",
        )
    else:
        render_context_card(
            "League Data",
            f'<p style="margin:0;font-size:12px;color:{THEME["tx2"]};">'
            "No league connected. Connect your Yahoo league to enable "
            "matchup-aware recommendations.</p>",
        )

    render_context_card(
        "Matchup State",
        f'<p style="margin:0;font-size:13px;color:{THEME["tx1"]};">'
        f"Strategy: <strong>{matchup_state_label.upper()}</strong></p>"
        f'<p style="margin:4px 0 0;font-size:12px;color:{THEME["tx2"]};">'
        f"{matchup_note}</p>",
    )

    render_context_card(
        "How It Works",
        f'<p style="margin:0;font-size:12px;color:{THEME["tx2"]};line-height:1.5;">'
        "Layer 1: H2H-weighted weekly projection score.<br>"
        "Layer 2: Risk adjustment for matchup context.<br>"
        "Layer 3: Per-category Standings Gained Points impact.</p>",
    )

# ── Main panel ────────────────────────────────────────────────────────

with main:
    st.subheader("Compare Players for a Roster Slot")

    if not START_SIT_AVAILABLE:
        st.error("Start/Sit advisor is unavailable.")
        st.stop()

    # Build player name list from pool (prefer roster players first)
    all_names: list[str] = []
    all_ids: list[int] = []

    if not pool.empty:
        # Put rostered players at top of list for convenience
        if user_player_ids:
            rostered = pool[pool["player_id"].isin(user_player_ids)].copy()
            others = pool[~pool["player_id"].isin(user_player_ids)].copy()
            sorted_pool = pd.concat([rostered, others], ignore_index=True)
        else:
            sorted_pool = pool.copy()

        all_names = sorted_pool["player_name"].tolist()
        all_ids = sorted_pool["player_id"].tolist()

    st.markdown(
        f'<p style="font-size:13px;color:{THEME["tx2"]};margin-bottom:4px;">'
        "Select 2 to 4 players competing for the same roster slot. "
        "The advisor will rank them using a 3-layer decision model.</p>",
        unsafe_allow_html=True,
    )

    # Multi-select for players to compare
    selected_names = st.multiselect(
        "Players to compare",
        options=all_names,
        default=[],
        max_selections=4,
        key="start_sit_player_select",
        help="Select 2 to 4 players competing for the same slot.",
        placeholder="Choose 2 to 4 players...",
    )

    if not selected_names:
        st.info(
            "Select at least 2 players above to receive a start/sit recommendation. "
            "Rostered players appear at the top of the list."
        )
    elif len(selected_names) == 1:
        st.warning("Select at least 2 players to compare. Only 1 player chosen.")
    else:
        # Map selected names to player IDs
        name_to_id = dict(zip(all_names, all_ids))
        selected_ids = [int(name_to_id[n]) for n in selected_names if n in name_to_id]

        if len(selected_ids) < 2:
            st.warning("Could not resolve player IDs for the selected players.")
        else:
            # Run the advisor
            with st.spinner("Computing start/sit recommendation..."):
                try:
                    result = start_sit_recommendation(
                        player_ids=selected_ids,
                        player_pool=pool.rename(columns={"player_name": "name"}),
                        config=config,
                        weekly_schedule=None,
                        park_factors=None,
                        my_weekly_totals=None,
                        opp_weekly_totals=None,
                        standings=None,
                        team_name=user_team_name,
                    )
                except Exception as exc:
                    logger.exception("start_sit_recommendation failed")
                    st.error(f"Recommendation engine error: {exc}")
                    result = None

            if result is None or not result.get("players"):
                st.warning("No recommendation could be generated for the selected players.")
            else:
                players_list = result["players"]
                rec_id = result.get("recommendation")
                confidence = result.get("confidence", 0.0)
                confidence_label = result.get("confidence_label", "Toss-up")

                # ── Summary banner ────────────────────────────────────
                if rec_id is not None:
                    rec_player = next((p for p in players_list if p["player_id"] == rec_id), None)
                    rec_name = rec_player["name"] if rec_player else "Unknown"

                    label_color = (
                        THEME["green"]
                        if confidence_label == "Clear Start"
                        else THEME["warn"]
                        if confidence_label == "Lean Start"
                        else THEME["tx2"]
                    )

                    st.markdown(
                        f'<div style="'
                        f"background:{THEME['card']};"
                        f"border:1px solid {THEME['card_h']};"
                        f"border-left:4px solid {label_color};"
                        f"border-radius:8px;"
                        f"padding:14px 18px;"
                        f"margin-bottom:16px;"
                        f'">'
                        f'<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
                        f"color:{THEME['tx2']};text-transform:uppercase;margin-bottom:4px;"
                        f'">Recommendation</div>'
                        f'<div style="font-size:20px;font-weight:700;color:{THEME["tx1"]};">'
                        f"Start {rec_name}"
                        f"</div>"
                        f'<div style="font-size:13px;color:{label_color};font-weight:600;margin-top:4px;">'
                        f"{confidence_label} &mdash; {confidence * 100:.0f}% confidence"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Recommendations table ─────────────────────────────
                st.markdown(
                    f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                    f"color:{THEME['tx2']};text-transform:uppercase;"
                    f'margin:0 0 6px;">Player Rankings</p>',
                    unsafe_allow_html=True,
                )

                # Build display DataFrame with color-coded recommendation column
                rows = []
                row_classes: dict[int, str] = {}
                for rank, p in enumerate(players_list):
                    is_rec = p["player_id"] == rec_id
                    rec_label = "START" if is_rec else "SIT"
                    top_cat = ""
                    if p.get("category_impact"):
                        sorted_cats = sorted(
                            p["category_impact"].items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )
                        if sorted_cats:
                            top_cat = sorted_cats[0][0]

                    rows.append(
                        {
                            "Player": p["name"],
                            "Decision": rec_label,
                            "Score": f"{p['start_score']:.3f}",
                            "Floor": f"{p['floor']:.3f}",
                            "Ceiling": f"{p['ceiling']:.3f}",
                            "Top Category": top_cat.upper() if top_cat else "",
                        }
                    )
                    # row_classes uses positional index
                    row_classes[rank] = "row-start" if is_rec else "row-sit"

                display_df = pd.DataFrame(rows)

                # Inject row coloring CSS
                st.markdown(
                    f"<style>"
                    f"tr.row-start td {{ background-color:{THEME['green']}22 !important; }}"
                    f"tr.row-start .col-name {{ color:{THEME['green']} !important; font-weight:700 !important; }}"
                    f"tr.row-sit td {{ background-color:{THEME['danger']}18 !important; }}"
                    f"tr.row-sit .col-name {{ color:{THEME['danger']} !important; }}"
                    f"</style>",
                    unsafe_allow_html=True,
                )

                render_compact_table(
                    display_df,
                    highlight_cols=["Score", "Floor", "Ceiling"],
                    row_classes=row_classes,
                )

                # ── Per-player reasoning ──────────────────────────────
                st.markdown(
                    f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                    f"color:{THEME['tx2']};text-transform:uppercase;"
                    f'margin:16px 0 6px;">Decision Reasoning</p>',
                    unsafe_allow_html=True,
                )

                for p in players_list:
                    is_rec = p["player_id"] == rec_id
                    badge_color = THEME["green"] if is_rec else THEME["danger"]
                    badge_label = "START" if is_rec else "SIT"
                    reasons = p.get("reasoning", [])
                    reasons_html = "".join(
                        f'<li style="margin-bottom:4px;font-size:13px;color:{THEME["tx1"]};">{r}</li>' for r in reasons
                    )

                    st.markdown(
                        f'<div style="'
                        f"background:{THEME['card']};"
                        f"border:1px solid {THEME['card_h']};"
                        f"border-radius:6px;"
                        f"padding:12px 16px;"
                        f"margin-bottom:10px;"
                        f'">'
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                        f'<span style="font-weight:700;font-size:14px;color:{THEME["tx1"]};">'
                        f"{p['name']}"
                        f"</span>"
                        f'<span style="'
                        f"background:{badge_color}22;"
                        f"color:{badge_color};"
                        f"font-size:10px;font-weight:700;letter-spacing:1px;"
                        f"padding:2px 8px;border-radius:4px;"
                        f'">{badge_label}</span>'
                        f"</div>"
                        f'<ul style="margin:0;padding-left:18px;">{reasons_html}</ul>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Player card selector ──────────────────────────────
                rec_names = [p["name"] for p in players_list]
                rec_ids = [p["player_id"] for p in players_list]

                if rec_ids:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                        f"color:{THEME['tx2']};text-transform:uppercase;"
                        f'margin:16px 0 4px;">Player Details</p>',
                        unsafe_allow_html=True,
                    )
                    render_player_select(
                        rec_names,
                        rec_ids,
                        key_suffix="startsit",
                    )

    # ── No roster guidance ────────────────────────────────────────────
    if rosters.empty:
        st.markdown(
            f'<div style="'
            f"background:{THEME['warn']}18;"
            f"border:1px solid {THEME['warn']}44;"
            f"border-radius:6px;"
            f"padding:12px 16px;"
            f"margin-top:16px;"
            f'">'
            f'<p style="margin:0;font-size:13px;color:{THEME["tx1"]};">'
            f"No league roster data found. Connect your Yahoo league for matchup-aware "
            f"recommendations. The advisor still works with any player from the pool.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
