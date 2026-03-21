"""Matchup Planner — Weekly per-game matchup ratings for your roster."""

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

try:
    from src.matchup_planner import compute_weekly_matchup_ratings

    MATCHUP_PLANNER_AVAILABLE = True
except Exception:
    MATCHUP_PLANNER_AVAILABLE = False

try:
    from src.optimizer.matchup_adjustments import get_weekly_schedule

    SCHEDULE_AVAILABLE = True
except Exception:
    SCHEDULE_AVAILABLE = False

try:
    from src.data_bootstrap import PARK_FACTORS

    PARK_FACTORS_AVAILABLE = True
except Exception:
    PARK_FACTORS_AVAILABLE = False
    PARK_FACTORS = {}

logger = logging.getLogger(__name__)

T = THEME

# ── Tier color map ────────────────────────────────────────────────────
_TIER_COLORS: dict[str, str] = {
    "smash": T["green"],
    "favorable": T["sky"],
    "neutral": T["tx2"],
    "unfavorable": T["warn"],
    "avoid": T["danger"],
}

_TIER_LABELS: dict[str, str] = {
    "smash": "Smash",
    "favorable": "Favorable",
    "neutral": "Neutral",
    "unfavorable": "Unfavorable",
    "avoid": "Avoid",
}

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heater | Matchup Planner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()

render_page_layout(
    "MATCHUP PLANNER",
    banner_teaser="Weekly matchup ratings with per-game analysis",
    banner_icon="calendar",
)

# ── Guard: matchup planner module ─────────────────────────────────────

if not MATCHUP_PLANNER_AVAILABLE:
    st.error(
        "The matchup planner module could not be loaded. "
        "Ensure all dependencies are installed: pip install -r requirements.txt"
    )
    st.stop()

# ── Load player pool ──────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded. Run load_sample_data.py or bootstrap from the main app.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# ── Load league rosters ───────────────────────────────────────────────

rosters = load_league_rosters()

# ── Helper: build tier badge HTML ─────────────────────────────────────


def _tier_badge(tier: str) -> str:
    """Return an inline HTML badge for a matchup tier."""
    color = _TIER_COLORS.get(tier, T["tx2"])
    label = _TIER_LABELS.get(tier, tier.capitalize())
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f"background:{color};color:#ffffff;font-size:11px;font-weight:700;"
        f'letter-spacing:0.4px;">{label}</span>'
    )


# ── Helper: build per-game detail HTML ───────────────────────────────


def _games_detail_html(games: list[dict]) -> str:
    """Render a compact list of per-game details as HTML."""
    if not games:
        return '<span style="color:#6b7280;font-size:12px;">No games scheduled</span>'
    parts = []
    for g in games:
        date = str(g.get("game_date", ""))
        opp = str(g.get("opponent", ""))
        raw = g.get("raw_score", 0.0)
        home_away = g.get("home_away", 1.0)
        loc = "Home" if home_away and home_away > 1.0 else "Away"
        pf = g.get("park_factor", 1.0)
        parts.append(
            f'<span style="font-size:11px;color:{T["tx2"]};">'
            f"{date} vs {opp} ({loc}, Park Factor: {float(pf):.2f}, Score: {float(raw):.3f})"
            f"</span>"
        )
    return "<br>".join(parts)


# ── Layout ────────────────────────────────────────────────────────────

ctx, main = render_context_columns()

# ── Context panel ─────────────────────────────────────────────────────

with ctx:
    # Week selector: how many days ahead to look
    days_ahead = st.selectbox(
        "Days to look ahead",
        options=[7, 10, 14],
        index=0,
        key="matchup_days_ahead",
        help="Number of days to include in the matchup window.",
    )

    # Player type filter
    player_type = st.radio(
        "Player type",
        options=["All", "Hitters", "Pitchers"],
        index=0,
        key="matchup_player_type",
    )

    # Team selector (if league data is available)
    team_name = None
    roster_df: pd.DataFrame = pd.DataFrame()

    if not rosters.empty:
        team_names = sorted(rosters["team_name"].unique().tolist())
        user_rows = rosters[rosters["is_user_team"] == 1]
        default_team = user_rows.iloc[0]["team_name"] if not user_rows.empty else team_names[0]
        default_idx = team_names.index(default_team) if default_team in team_names else 0

        team_name = st.selectbox(
            "Team",
            options=team_names,
            index=default_idx,
            key="matchup_team_select",
        )

        render_context_card(
            "Team",
            f'<div style="font-size:13px;color:{T["tx"]};">{team_name}</div>',
        )
    else:
        st.info("No league roster data loaded. Using full player pool for demonstration.")
        render_context_card(
            "Data Source",
            f'<div style="font-size:12px;color:{T["tx2"]};">Showing all pooled players (no league connected).</div>',
        )

    # Tier legend
    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'<span style="font-size:12px;color:{T["tx"]};">{label}</span>'
        f"</div>"
        for tier, color in _TIER_COLORS.items()
        for label in [_TIER_LABELS[tier]]
    )
    render_context_card("Rating Tiers", legend_rows)

# ── Main panel ────────────────────────────────────────────────────────

with main:
    # Build roster DataFrame for the planner
    if team_name and not rosters.empty:
        try:
            roster_df = get_team_roster(team_name)
        except Exception as exc:
            logger.exception("Failed to load team roster for %s", team_name)
            st.error(f"Could not load roster for {team_name}: {exc}")
            roster_df = pd.DataFrame()
    else:
        # Fallback: use the top 23 players from the pool as a demo roster
        roster_df = pool.head(23).rename(columns={"player_name": "name"})

    if roster_df is None or roster_df.empty:
        st.warning("No roster data found for the selected team.")
        st.stop()

    # Normalize column names: get_team_roster returns "name"; pool uses "player_name"
    if "player_name" in roster_df.columns and "name" not in roster_df.columns:
        roster_df = roster_df.rename(columns={"player_name": "name"})

    # Apply player-type filter before running the planner
    filtered_roster = roster_df.copy()
    if player_type == "Hitters":
        filtered_roster = filtered_roster[filtered_roster["is_hitter"] == 1]
    elif player_type == "Pitchers":
        filtered_roster = filtered_roster[filtered_roster["is_hitter"] == 0]

    if filtered_roster.empty:
        st.info(f"No {player_type.lower()} found on this roster.")
        st.stop()

    # Fetch weekly schedule
    weekly_schedule = None
    if SCHEDULE_AVAILABLE:
        with st.spinner("Fetching MLB schedule..."):
            try:
                weekly_schedule = get_weekly_schedule(days_ahead=int(days_ahead))
            except Exception as exc:
                logger.warning("Could not fetch schedule: %s", exc)
                weekly_schedule = None

    if not weekly_schedule:
        st.info(
            "No live schedule data available. Matchup ratings are computed "
            "from projection data only (games count will be 0 for all players). "
            "Connect to the internet and ensure statsapi is installed for live schedules."
        )

    # Run the matchup planner
    park_factors = PARK_FACTORS if PARK_FACTORS_AVAILABLE else {}

    with st.spinner("Computing matchup ratings..."):
        try:
            ratings_df = compute_weekly_matchup_ratings(
                roster=filtered_roster,
                weekly_schedule=weekly_schedule,
                park_factors=park_factors or None,
            )
        except Exception as exc:
            logger.exception("Matchup planner failed")
            st.error(f"Error computing matchup ratings: {exc}")
            ratings_df = pd.DataFrame()

    if ratings_df is None or ratings_df.empty:
        st.info("No matchup ratings could be computed. Check that the roster and schedule data are available.")
        st.stop()

    # Sort by rating descending
    ratings_df = ratings_df.sort_values("weekly_matchup_rating", ascending=False).reset_index(drop=True)

    # ── Summary metrics ───────────────────────────────────────────────

    n_smash = int((ratings_df["matchup_tier"] == "smash").sum())
    n_avoid = int((ratings_df["matchup_tier"] == "avoid").sum())
    avg_games = float(ratings_df["games_count"].mean()) if "games_count" in ratings_df.columns else 0.0
    avg_rating = (
        float(ratings_df["weekly_matchup_rating"].mean()) if "weekly_matchup_rating" in ratings_df.columns else 0.0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players Rated", len(ratings_df))
    m2.metric("Smash Matchups", n_smash)
    m3.metric("Avoid Matchups", n_avoid)
    m4.metric("Average Games", f"{avg_games:.1f}")

    st.markdown("---")

    # ── Tab view: Summary | Per-Game Detail ───────────────────────────

    tab_summary, tab_detail, tab_hitters, tab_pitchers = st.tabs(
        ["Summary", "Per-Game Detail", "Hitters Only", "Pitchers Only"]
    )

    # Build display DataFrame (shared across tabs)
    def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
        """Build a flat display DataFrame from ratings output."""
        rows = []
        for _, row in df.iterrows():
            tier = str(row.get("matchup_tier", "neutral"))
            rows.append(
                {
                    "Player": str(row.get("name", "")),
                    "Position": str(row.get("positions", "")),
                    "Type": "Hitter" if row.get("is_hitter", True) else "Pitcher",
                    "Games": int(row.get("games_count", 0)),
                    "Rating": f"{float(row.get('weekly_matchup_rating', 0.0)):.1f}",
                    "Tier": _TIER_LABELS.get(tier, tier.capitalize()),
                }
            )
        return pd.DataFrame(rows)

    def _tier_row_classes(df: pd.DataFrame) -> dict[int, str]:
        """Map row index to a CSS row class based on matchup tier."""
        cls_map = {
            "smash": "row-start",
            "favorable": "",
            "neutral": "",
            "unfavorable": "row-bench",
            "avoid": "row-bench",
        }
        result = {}
        for i, (_, row) in enumerate(df.iterrows()):
            tier = str(row.get("matchup_tier", "neutral"))
            cls = cls_map.get(tier, "")
            if cls:
                result[i] = cls
        return result

    with tab_summary:
        st.subheader(f"Weekly Matchup Summary — {len(ratings_df)} players")
        display_df = _build_display_df(ratings_df)
        row_classes = _tier_row_classes(ratings_df)
        render_compact_table(display_df, highlight_cols=["Rating", "Games"], row_classes=row_classes)

        # Player card selector
        if "player_id" in ratings_df.columns:
            render_player_select(
                display_df["Player"].tolist(),
                ratings_df["player_id"].tolist(),
                key_suffix="matchup_summary",
            )

    with tab_detail:
        st.subheader("Per-Game Matchup Detail")
        st.caption(
            "Each player's individual game matchups for the week. "
            "Park factor and home/away status are factored into every game score."
        )

        for _, row in ratings_df.iterrows():
            name = str(row.get("name", ""))
            tier = str(row.get("matchup_tier", "neutral"))
            rating = float(row.get("weekly_matchup_rating", 0.0))
            games = row.get("games", []) or []
            games_count = int(row.get("games_count", 0))
            positions = str(row.get("positions", ""))
            is_hitter = bool(row.get("is_hitter", True))
            player_type_label = "Hitter" if is_hitter else "Pitcher"

            tier_color = _TIER_COLORS.get(tier, T["tx2"])
            tier_label = _TIER_LABELS.get(tier, tier.capitalize())

            header_html = (
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'border-left:4px solid {tier_color};padding-left:10px;margin-bottom:4px;">'
                f'<span style="font-weight:700;font-size:14px;color:{T["tx"]};">{name}</span>'
                f'<span style="font-size:12px;color:{T["tx2"]};">{positions} &bull; {player_type_label}</span>'
                f'<span style="font-size:12px;color:{T["tx2"]};">{games_count} game(s)</span>'
                f'<span style="font-size:12px;font-weight:700;color:{tier_color};">'
                f"Rating: {rating:.1f} &bull; {tier_label}"
                f"</span>"
                f"</div>"
            )
            games_html = _games_detail_html(games)

            with st.expander(name, expanded=False):
                st.markdown(header_html, unsafe_allow_html=True)
                st.markdown(games_html, unsafe_allow_html=True)

                # Projected stat adjustments for hitters
                proj = row.get("projected_stats_adjusted") or {}
                if proj and is_hitter:
                    proj_parts = []
                    stat_labels = {"hr": "Home Runs", "r": "Runs", "rbi": "Runs Batted In", "sb": "Stolen Bases"}
                    for stat_key, stat_label in stat_labels.items():
                        val = proj.get(stat_key)
                        if val is not None:
                            proj_parts.append(f"{stat_label}: {float(val):.2f}")
                    if proj_parts:
                        st.caption("Park-adjusted weekly projections: " + " | ".join(proj_parts))

    with tab_hitters:
        hitter_df = ratings_df[ratings_df["is_hitter"] == True].copy()  # noqa: E712
        if hitter_df.empty:
            st.info("No hitters found in current selection.")
        else:
            st.subheader(f"Hitters — {len(hitter_df)} players")
            hitter_display = _build_display_df(hitter_df)
            hitter_classes = _tier_row_classes(hitter_df)
            render_compact_table(hitter_display, highlight_cols=["Rating", "Games"], row_classes=hitter_classes)
            if "player_id" in hitter_df.columns:
                render_player_select(
                    hitter_display["Player"].tolist(),
                    hitter_df["player_id"].tolist(),
                    key_suffix="matchup_hitters",
                )

    with tab_pitchers:
        pitcher_df = ratings_df[ratings_df["is_hitter"] == False].copy()  # noqa: E712
        if pitcher_df.empty:
            st.info("No pitchers found in current selection.")
        else:
            st.subheader(f"Pitchers — {len(pitcher_df)} players")
            pitcher_display = _build_display_df(pitcher_df)
            pitcher_classes = _tier_row_classes(pitcher_df)
            render_compact_table(pitcher_display, highlight_cols=["Rating", "Games"], row_classes=pitcher_classes)
            if "player_id" in pitcher_df.columns:
                render_player_select(
                    pitcher_display["Player"].tolist(),
                    pitcher_df["player_id"].tolist(),
                    key_suffix="matchup_pitchers",
                )

    # ── Tier color legend inline ──────────────────────────────────────
    st.markdown("---")
    legend_items = " &nbsp;|&nbsp; ".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{color};display:inline-block;"></span>'
        f'<span style="font-size:12px;color:{T["tx2"]};">{_TIER_LABELS[tier]}: '
        f"{'80th+ percentile' if tier == 'smash' else '60-80th' if tier == 'favorable' else '40-60th' if tier == 'neutral' else '20-40th' if tier == 'unfavorable' else 'below 20th'}"
        f"</span></span>"
        for tier, color in _TIER_COLORS.items()
    )
    st.markdown(
        f'<div style="font-size:12px;color:{T["tx2"]};padding:4px 0;">{legend_items}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ratings are computed using a multiplicative model: base weighted on-base average (wOBA) "
        "or expected fielding-independent pitching (xFIP), scaled by park factor, platoon adjustment, "
        "home/away advantage, and games count. Percentile ranks are computed within hitters and pitchers separately."
    )
