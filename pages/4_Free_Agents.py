"""Free Agents — Unified free agent browsing, add/drop recommendations, and streaming targets."""

import logging

import pandas as pd
import streamlit as st

from src.database import get_connection, init_db, load_player_pool
from src.in_season import rank_free_agents
from src.league_manager import get_free_agents, get_team_roster
from src.ui_shared import (
    METRIC_TOOLTIPS,
    THEME,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
    render_sortable_table,
)
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

try:
    from src.waiver_wire import compute_add_drop_recommendations

    WAIVER_WIRE_AVAILABLE = True
except ImportError:
    WAIVER_WIRE_AVAILABLE = False

logger = logging.getLogger(__name__)

T = THEME

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heater | Free Agents",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()
page_timer_start()

# ── Data loading ──────────────────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
# Preserve 'name' alias for waiver_wire internals
pool["name"] = pool["player_name"]
config = LeagueConfig()

# ── Roster check ──────────────────────────────────────────────────────────────

yds = get_yahoo_data_service()
rosters = yds.get_rosters()
if rosters.empty:
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

# ── Remap roster IDs to player pool IDs via name matching ─────────────────────

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
    st.warning("Your roster appears to be empty. No free agent analysis can be generated.")
    st.stop()

# ── Get free agents — via YahooDataService (Yahoo-first with DB fallback) ─────

fa_pool = yds.get_free_agents()
# Cross-reference with player pool to get projection data
if not fa_pool.empty and not pool.empty:
    from src.live_stats import match_player_id

    yahoo_pids = set()
    for _, row in fa_pool.iterrows():
        pid = match_player_id(row.get("player_name", ""), row.get("team", ""))
        if pid is not None:
            yahoo_pids.add(pid)
    if yahoo_pids:
        fa_pool = pool[pool["player_id"].isin(yahoo_pids)].copy()
    else:
        fa_pool = get_free_agents(pool)

if fa_pool.empty:
    st.info("No free agents available (all players are rostered).")
    st.stop()

# ── Compute add/drop recommendations ─────────────────────────────────────────

recommendations: list[dict] = []
if WAIVER_WIRE_AVAILABLE:
    try:
        from src.validation.dynamic_context import compute_weeks_remaining

        recommendations = compute_add_drop_recommendations(
            user_roster_ids=user_roster_ids,
            player_pool=pool,
            config=config,
            weeks_remaining=compute_weeks_remaining(),
            max_moves=5,
            max_fa_candidates=100,
            max_drop_candidates=8,
            user_team_name=user_team_name,
        )
    except Exception as e:
        logger.exception("Failed to compute waiver wire recommendations")
        recommendations = []

# ── Compute FA rankings ───────────────────────────────────────────────────────

ranked_fas = pd.DataFrame()
try:
    ranked_fas = rank_free_agents(user_roster_ids, fa_pool, pool, config)
except Exception as e:
    logger.exception("Failed to rank free agents")

# ── Roster summary stats for context panel ────────────────────────────────────

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

# ── Build mlb_id lookup from pool for headshot rendering ──────────────────────

_pid_to_mlb = {}
if "mlb_id" in pool.columns and "player_id" in pool.columns:
    _pid_to_mlb = dict(zip(pool["player_id"], pool["mlb_id"]))

# ── Banner: top pickup recommendation ─────────────────────────────────────────

_banner_teaser = "Top free agent pickups by marginal value"
if recommendations:
    _top_rec = recommendations[0]
    _top_add = _top_rec.get("add_name", "Unknown")
    _top_drop = _top_rec.get("drop_name", "Unknown")
    _top_delta = _top_rec.get("net_sgp_delta", 0)
    _banner_teaser = f"Top pickup: Add {_top_add} (drop {_top_drop}) for +{_top_delta:.2f} Standings Gained Points"

render_page_layout("FREE AGENTS", banner_teaser=_banner_teaser, banner_icon="free_agents")

# ── Position filter (shared across all sections) ─────────────────────────────

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

    # Position filter pills
    pos_filter = st.session_state.get("fa_merged_pos_filter", "All")
    render_context_card(
        "Filter by Position",
        f'<div style="font-size:12px;color:{T["tx2"]};font-family:Figtree,sans-serif;">'
        f"Filters all sections below to the selected position.</div>",
    )
    for pos in POSITIONS:
        btn_type = "primary" if pos_filter == pos else "secondary"
        if st.button(pos, key=f"fa_merged_pill_{pos}", type=btn_type, use_container_width=True):
            st.session_state.fa_merged_pos_filter = pos
            st.rerun()

    # Data freshness status
    _freshness = yds.get_data_freshness()
    _roster_fresh = _freshness.get("rosters", "Unknown")
    _fa_fresh = _freshness.get("free_agents", "Unknown")
    _connected = yds.is_connected()
    _sync_status = "Connected" if _connected else "Not connected"
    _sync_color = T["ok"] if _connected else T["tx2"]
    sync_html = (
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Yahoo Status</span>'
        f'<span style="font-weight:600;color:{_sync_color};">{_sync_status}</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Rosters</span>'
        f'<span style="font-weight:600;color:{T["tx"]};">{_roster_fresh}</span></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
        f'<span style="color:{T["tx2"]};">Free Agents</span>'
        f'<span style="font-weight:600;color:{T["tx"]};">{_fa_fresh}</span></div>'
    )
    render_context_card("Data Freshness", sync_html)

# ── Helper: apply position filter to a DataFrame ─────────────────────────────


def _apply_pos_filter(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Filter a DataFrame by position using the 'positions' column."""
    if pos == "All" or df.empty:
        return df
    col = "positions" if "positions" in df.columns else "Position"
    if col not in df.columns:
        return df
    return df[df[col].apply(lambda p: pos in str(p).split(",") if pd.notna(p) else False)]


def _apply_pos_filter_recs(recs: list[dict], pos: str) -> list[dict]:
    """Filter recommendation dicts by add_positions."""
    if pos == "All":
        return recs
    return [r for r in recs if pos in [p.strip() for p in str(r.get("add_positions", "")).split(",")]]


# ── Main content ──────────────────────────────────────────────────────────────

with main:
    # Analytics transparency badge
    try:
        from src.data_bootstrap import get_bootstrap_context

        _boot_ctx = get_bootstrap_context()
        if _boot_ctx:
            from src.ui_analytics_badge import render_analytics_badge

            render_analytics_badge(_boot_ctx)
    except Exception:
        pass

    pos_filter = st.session_state.get("fa_merged_pos_filter", "All")

    # ── Section 1: Recommended Adds/Drops ─────────────────────────────────────

    st.markdown(
        '<div class="sec-head">Recommended Adds/Drops</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
        f'font-family:Figtree,sans-serif;">'
        f"Free agents ranked by net Standings Gained Points improvement "
        f"after accounting for the recommended drop. Only positive-value swaps are shown.</div>",
        unsafe_allow_html=True,
    )

    filtered_recs = _apply_pos_filter_recs(recommendations, pos_filter)

    if not filtered_recs:
        if not WAIVER_WIRE_AVAILABLE:
            st.error("The waiver wire module could not be loaded. Ensure all required dependencies are installed.")
        else:
            st.info(
                "No add/drop recommendations found. "
                "This may mean the free agent pool is empty, your roster is already optimal, "
                "or no free agent improves your team's Standings Gained Points total."
            )
    else:
        adds_rows = []
        for rec in filtered_recs:
            sustainability_pct = int(rec.get("sustainability_score", 0.5) * 100)
            entry = {
                "Add": rec.get("add_name", ""),
                "Position": rec.get("add_positions", ""),
                "Net Standings Gained Points Delta": f"{rec.get('net_sgp_delta', 0):.2f}",
                "Sustainability": f"{sustainability_pct}%",
                "Drop": rec.get("drop_name", ""),
            }
            if _pid_to_mlb:
                entry["mlb_id"] = _pid_to_mlb.get(rec.get("add_player_id"))
            adds_rows.append(entry)
        adds_df = pd.DataFrame(adds_rows)
        render_compact_table(adds_df, highlight_cols=["Net Standings Gained Points Delta"])

        # Reasoning expanders
        for i, rec in enumerate(filtered_recs):
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
        add_names = [rec.get("add_name", "") for rec in filtered_recs]
        add_ids_raw = [rec.get("add_player_id") for rec in filtered_recs]
        add_ids = [int(pid) for pid in add_ids_raw if pid is not None]
        if add_names and add_ids:
            render_player_select(add_names, add_ids, key_suffix="fa_merged_adds")

    # ── Section 2: This Week's Streams ────────────────────────────────────────

    try:
        from src.waiver_wire import recommend_streams

        _opp_profile = None
        try:
            from src.opponent_intel import get_current_opponent

            _opp_profile = get_current_opponent()
        except Exception:
            pass

        # Exclude ALL rostered players (all 12 teams), not just user's team
        from src.database import get_all_rostered_player_ids

        _all_rostered = get_all_rostered_player_ids(rosters)
        _fa_for_streams = pool[~pool["player_id"].isin(_all_rostered)]
        if pos_filter != "All":
            _fa_for_streams = _apply_pos_filter(_fa_for_streams, pos_filter)

        _stream_recs = recommend_streams(
            fa_pool=_fa_for_streams,
            player_pool=pool,
            user_roster_ids=user_roster_ids,
            opponent_profile=_opp_profile,
        )
        if _stream_recs:
            st.markdown(
                '<div class="sec-head" style="margin-top:24px;">This Week\'s Streams</div>',
                unsafe_allow_html=True,
            )
            _opp_label = f" (vs {_opp_profile['name']})" if _opp_profile else ""
            st.markdown(
                f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
                f'font-family:Figtree,sans-serif;">'
                f"Matchup-aware streaming targets{_opp_label}. Budget: 5 streaming adds per week.</div>",
                unsafe_allow_html=True,
            )
            _stream_rows = []
            for sr in _stream_recs:
                _stream_rows.append(
                    {
                        "Player": sr["player_name"],
                        "Position": sr["positions"],
                        "Type": sr["stream_type"],
                        "Reasoning": sr["reasoning"],
                    }
                )
            _stream_df = pd.DataFrame(_stream_rows)
            render_compact_table(_stream_df)
    except Exception:
        pass  # Non-fatal — streaming is supplementary

    # ── Section 3: All Free Agents ────────────────────────────────────────────

    st.markdown(
        '<div class="sec-head" style="margin-top:24px;">All Free Agents</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
        f'font-family:Figtree,sans-serif;">'
        f"Full ranked free agent list with projected impact on your roster. "
        f"The Impact column shows the Standings Gained Points delta if this player "
        f"replaced the weakest player at the same position on your roster.</div>",
        unsafe_allow_html=True,
    )

    _display_fas = _apply_pos_filter(ranked_fas, pos_filter) if not ranked_fas.empty else ranked_fas

    if _display_fas.empty:
        if pos_filter != "All":
            st.info(f"No ranked free agents at position: {pos_filter}")
        else:
            st.info("No ranked free agents found.")
    else:
        # Compute the "Impact" column: SGP delta if FA replaced weakest roster player
        # at the same position
        _user_pool = pool[pool["player_id"].isin(user_roster_ids)].copy()

        # Pre-compute marginal values for roster players using rank_free_agents
        # so Impact can compare like-for-like SGP units
        _roster_ranked = None
        try:
            _roster_ranked = rank_free_agents([], _user_pool, pool, config)
        except Exception:
            pass

        def _compute_impact(fa_row):
            """Compute SGP delta for adding this FA and dropping weakest same-position player.

            Both sides use marginal_value from rank_free_agents so units are
            comparable (SGP points, not raw counting stats).
            """
            fa_positions = str(fa_row.get("positions", "")) if pd.notna(fa_row.get("positions")) else ""
            fa_pos_list = [p.strip() for p in fa_positions.split(",") if p.strip()]
            if not fa_pos_list:
                return 0.0

            fa_marginal = fa_row.get("marginal_value", 0.0)
            if pd.isna(fa_marginal):
                fa_marginal = 0.0

            # Find weakest roster player at overlapping position using ranked data
            source = _roster_ranked if _roster_ranked is not None and not _roster_ranked.empty else _user_pool
            candidates = source[
                source["positions"].apply(
                    lambda p: any(fp in str(p).split(",") for fp in fa_pos_list) if pd.notna(p) else False
                )
            ]
            if candidates.empty:
                return float(fa_marginal)  # No one to compare — pure add value

            if "marginal_value" in candidates.columns:
                weakest_val = float(candidates["marginal_value"].min())
            else:
                weakest_val = 0.0

            return float(fa_marginal) - weakest_val

        _display_fas = _display_fas.copy()
        _display_fas["impact"] = _display_fas.apply(_compute_impact, axis=1)

        # Merge mlb_id from player pool for headshot rendering
        if "mlb_id" not in _display_fas.columns and "player_id" in _display_fas.columns:
            _id_to_mlb = pool[["player_id", "mlb_id"]].drop_duplicates(subset="player_id")
            _display_fas = _display_fas.merge(_id_to_mlb, on="player_id", how="left")

        show_cols = ["player_name", "positions", "marginal_value", "impact", "best_category", "best_cat_impact"]
        if "mlb_id" in _display_fas.columns:
            show_cols.append("mlb_id")
        # Only include columns that exist
        show_cols = [c for c in show_cols if c in _display_fas.columns]
        display_fa_df = _display_fas[show_cols].copy()

        # Format numeric columns
        for col in ["marginal_value", "impact", "best_cat_impact"]:
            if col in display_fa_df.columns:
                display_fa_df[col] = display_fa_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")

        display_fa_df = display_fa_df.rename(
            columns={
                "player_name": "Player",
                "positions": "Position",
                "marginal_value": "Marginal Value",
                "impact": "Impact",
                "best_category": "Best Category",
                "best_cat_impact": "Category Impact",
            }
        )

        st.markdown(
            f'<div style="font-size:11px;color:{T["tx2"]};margin-bottom:4px;'
            f'font-family:Figtree,sans-serif;">'
            f"Showing {len(display_fa_df)} free agents. Click column headers to sort.</div>",
            unsafe_allow_html=True,
        )
        render_sortable_table(display_fa_df, height=450, key="fa_merged_all_fas")
        st.caption(METRIC_TOOLTIPS["marginal_value"])

        # Player card selector
        if "player_id" in _display_fas.columns:
            render_player_select(
                display_fa_df["Player"].tolist(),
                _display_fas["player_id"].tolist(),
                key_suffix="fa_merged_browse",
            )

    # ── Section 4: Recommended Drops ──────────────────────────────────────────

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

    if not filtered_recs:
        st.info("No drop recommendations available.")
    else:
        # AVIS Rule #6: Protect IL stashes (Bieber, Strider, AND players returning within 2 weeks)
        protected_names = {"Shane Bieber", "Spencer Strider"}  # Per AVIS Section 7

        # Dynamic IL return-date check: protect players returning within 14 days
        try:
            _il_conn = get_connection()
            try:
                _il_news = pd.read_sql_query(
                    "SELECT player_name, il_status, headline FROM player_news "
                    "WHERE news_type = 'injury' ORDER BY fetched_at DESC",
                    _il_conn,
                )
            finally:
                _il_conn.close()
            if not _il_news.empty:
                for _, _inj in _il_news.iterrows():
                    _il_st = str(_inj.get("il_status", "") or "")
                    # Protect players on 10-day IL (likely returning soon)
                    if "10-day" in _il_st.lower() or "day-to-day" in _il_st.lower():
                        _pname = str(_inj.get("player_name", ""))
                        if _pname:
                            protected_names.add(_pname)
        except Exception:
            pass  # Non-fatal: fall back to hardcoded list

        drops_rows = []
        for rec in filtered_recs:
            drop_name = rec.get("drop_name", "")
            if drop_name in protected_names:
                st.info(
                    f"IL STASH PROTECTED: {drop_name} — excluded from drop candidates "
                    f"per AVIS rules (returning soon or per AVIS Section 7)."
                )
                continue

            category_impact = rec.get("category_impact", {})
            # Show the most-impacted category from this swap
            if category_impact:
                top_cat = max(category_impact, key=lambda c: abs(category_impact[c]))
                top_delta = category_impact[top_cat]
                impact_str = f"{top_cat}: {top_delta:+.2f}"
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
        if not drops_df.empty:
            render_compact_table(drops_df, highlight_cols=["Top Category Impact"])

            # Player card selector for drop candidates (exclude IL-protected)
            drop_names = [r["Drop"] for r in drops_rows]
            drop_ids = [
                int(rec.get("drop_player_id"))
                for rec in filtered_recs
                if rec.get("drop_name", "") not in protected_names and rec.get("drop_player_id") is not None
            ]
            if drop_names and drop_ids:
                render_player_select(drop_names, drop_ids, key_suffix="fa_merged_drops")
        else:
            st.info("All drop candidates are protected by IL stash rules.")

page_timer_footer("Free Agents")
