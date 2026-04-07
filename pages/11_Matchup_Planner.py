"""Matchup Planner — Weekly per-game matchup ratings + category win probabilities."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.league_manager import get_team_roster
from src.ui_shared import (
    THEME,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
)
from src.yahoo_data_service import get_yahoo_data_service

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

try:
    from src.standings_engine import (
        ALL_CATEGORIES,
        compute_category_win_probabilities,
        find_user_opponent,
    )
    from src.valuation import LeagueConfig

    STANDINGS_ENGINE_AVAILABLE = True
except Exception:
    STANDINGS_ENGINE_AVAILABLE = False
    ALL_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]

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

# ── Hit/pitch category sets ──────────────────────────────────────────
_HIT_CATS = {"R", "HR", "RBI", "SB", "AVG", "OBP"}
_PITCH_CATS = {"W", "L", "SV", "K", "ERA", "WHIP"}

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heater | Matchup Planner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()
page_timer_start()

# ── Load player pool ──────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded. Run load_sample_data.py or bootstrap from the main app.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# ── Load league rosters ───────────────────────────────────────────────

yds = get_yahoo_data_service()
rosters = yds.get_rosters()

# ── Get current matchup + schedule for category probabilities ─────────

matchup_data = yds.get_matchup()
current_week = matchup_data.get("week", 1) if matchup_data else 1

# Attempt to load the full league schedule for week navigation
full_schedule: dict[int, list[tuple[str, str]]] = {}
try:
    full_schedule = yds.get_full_league_schedule() or {}
except Exception:
    logger.debug("Could not load full league schedule", exc_info=True)

# Determine user team name
user_team_name: str | None = None
if not rosters.empty:
    user_rows = rosters[rosters["is_user_team"] == 1]
    if not user_rows.empty:
        user_team_name = str(user_rows.iloc[0]["team_name"])

# ── Compute category win probabilities if possible ────────────────────

win_prob_data: dict | None = None
selected_week = st.session_state.get("matchup_week", current_week)


def _compute_win_probs(week: int) -> dict | None:
    """Compute win probabilities for the given week."""
    if not STANDINGS_ENGINE_AVAILABLE:
        return None
    if not user_team_name or rosters.empty:
        return None

    # Find opponent for the selected week
    opp_name = None
    if full_schedule:
        opp_name = find_user_opponent(full_schedule, week, user_team_name)
    # Fallback: current matchup opponent
    if not opp_name and matchup_data and week == current_week:
        opp_name = matchup_data.get("opp_name")
    if not opp_name:
        return None

    # Get roster IDs
    user_roster = rosters[rosters["team_name"] == user_team_name]
    opp_roster = rosters[rosters["team_name"] == opp_name]
    if user_roster.empty or opp_roster.empty:
        return None

    user_ids = user_roster["player_id"].dropna().astype(int).tolist()
    opp_ids = opp_roster["player_id"].dropna().astype(int).tolist()

    if not user_ids or not opp_ids:
        return None

    config = LeagueConfig()
    weeks_played = max(0, week - 1)
    weeks_remaining = max(1, 24 - week + 1)

    try:
        result = compute_category_win_probabilities(
            user_roster_ids=user_ids,
            opp_roster_ids=opp_ids,
            player_pool=pool,
            config=config,
            weeks_played=weeks_played,
            weeks_remaining=weeks_remaining,
            n_sims=10000,
            seed=42,
        )
        result["opponent"] = opp_name
        return result
    except Exception:
        logger.exception("Failed to compute category win probabilities")
        return None


# ── Build recommendation banner ──────────────────────────────────────

banner_teaser = "Weekly matchup ratings with per-game analysis"
if matchup_data:
    opp = matchup_data.get("opp_name", "Unknown")
    w = matchup_data.get("wins", 0)
    lo = matchup_data.get("losses", 0)
    t_cnt = matchup_data.get("ties", 0)
    banner_teaser = f"Week {current_week} vs {opp}: currently {w}-{lo}-{t_cnt} in categories"

# Pre-compute for current week to enhance the banner
win_prob_data = _compute_win_probs(selected_week)
if win_prob_data and selected_week == current_week:
    wp = win_prob_data.get("overall_win_pct", 0)
    proj = win_prob_data.get("projected_score", {})
    pw = proj.get("W", 0)
    pl = proj.get("L", 0)
    opp_name_display = win_prob_data.get("opponent", "opponent")
    banner_teaser = (
        f"Week {current_week} vs {opp_name_display}: {wp * 100:.0f}% chance to win (projected {pw:.0f}-{pl:.0f})"
    )
    # Add top cats
    cats = win_prob_data.get("categories", [])
    if cats:
        sorted_cats = sorted(cats, key=lambda c: c["win_pct"], reverse=True)
        best = sorted_cats[:2]
        best_str = ", ".join(f"{c['name']} ({c['win_pct'] * 100:.0f}%)" for c in best)
        tossups = [c for c in sorted_cats if 0.45 <= c["win_pct"] <= 0.55][:2]
        toss_str = ", ".join(f"{c['name']} ({c['win_pct'] * 100:.0f}%)" for c in tossups)
        banner_teaser += f". Best odds: {best_str}"
        if toss_str:
            banner_teaser += f". Toss-ups: {toss_str}"

render_page_layout(
    "MATCHUP PLANNER",
    banner_teaser=banner_teaser,
    banner_icon="calendar",
)

# ── Guard: matchup planner module ─────────────────────────────────────

if not MATCHUP_PLANNER_AVAILABLE:
    st.error(
        "The matchup planner module could not be loaded. "
        "Ensure all dependencies are installed: pip install -r requirements.txt"
    )
    st.stop()


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
            f"{date} vs {opp} ({loc}, Park Factor: {float(pf):.2f}, Score: {float(raw):.2f})"
            f"</span>"
        )
    return "<br>".join(parts)


# ── Helper: build category probability bar HTML ──────────────────────


def _prob_bar_color(pct: float) -> str:
    """Return bar color based on win probability percentage."""
    if pct >= 70:
        return T["green"]
    elif pct >= 55:
        return T["sky"]
    elif pct >= 45:
        return T["warn"]
    else:
        return T["danger"]


def _cat_header_color(cat: str) -> str:
    """Return category header color: blue for hitting, red for pitching."""
    if cat in _HIT_CATS:
        return T["sky"]
    return T["primary"]


def _build_category_prob_html(cat_data: list[dict]) -> str:
    """Build full HTML for category probability bars.

    Each entry has: name, user_proj, opp_proj, win_pct, confidence, is_inverse.
    """
    rows: list[str] = []
    for cat in cat_data:
        name = cat["name"]
        pct = cat["win_pct"] * 100
        user_proj = cat["user_proj"]
        opp_proj = cat["opp_proj"]
        confidence = cat.get("confidence", "low")
        bar_color = _prob_bar_color(pct)
        header_color = _cat_header_color(name)
        is_inverse = cat.get("is_inverse", False)
        direction_hint = " (lower is better)" if is_inverse else ""

        # Confidence badge
        conf_opacity = "1.0" if confidence == "high" else "0.7" if confidence == "medium" else "0.5"

        # Format projection values
        if name in ("AVG", "OBP"):
            user_str = f"{user_proj:.3f}"
            opp_str = f"{opp_proj:.3f}"
        elif name in ("ERA", "WHIP"):
            user_str = f"{user_proj:.2f}"
            opp_str = f"{opp_proj:.2f}"
        else:
            user_str = f"{user_proj:.1f}"
            opp_str = f"{opp_proj:.1f}"

        row_html = f"""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 0;
                    border-bottom:1px solid rgba(0,0,0,0.06);">
            <div style="width:55px;font-weight:700;font-size:13px;
                        color:{header_color};flex-shrink:0;">{name}</div>
            <div style="flex:1;position:relative;height:22px;
                        background:rgba(0,0,0,0.05);border-radius:11px;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;height:100%;
                            width:{max(2, min(pct, 100)):.1f}%;
                            background:{bar_color};border-radius:11px;
                            transition:width 0.3s ease;"></div>
                <div style="position:absolute;top:0;left:0;width:100%;height:100%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:11px;font-weight:700;color:{T["tx"]};
                            text-shadow:0 0 3px rgba(255,255,255,0.8);">
                    {pct:.0f}%
                </div>
            </div>
            <div style="width:110px;font-size:11px;color:{T["tx2"]};
                        text-align:right;flex-shrink:0;opacity:{conf_opacity};">
                {user_str} vs {opp_str}{direction_hint}
            </div>
        </div>
        """
        rows.append(row_html)

    return "\n".join(rows)


def _build_win_prob_context_html(prob_data: dict) -> str:
    """Build context card HTML for overall win probability with stacked bar."""
    win_pct = prob_data.get("overall_win_pct", 0) * 100
    tie_pct = prob_data.get("overall_tie_pct", 0) * 100
    loss_pct = prob_data.get("overall_loss_pct", 0) * 100
    proj = prob_data.get("projected_score", {})
    pw = proj.get("W", 0)
    pl = proj.get("L", 0)

    return f"""
    <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;font-size:12px;
                    margin-bottom:4px;color:{T["tx"]};">
            <span style="color:{T["green"]};font-weight:700;">Win {win_pct:.0f}%</span>
            <span style="color:{T["tx2"]};font-weight:600;">Tie {tie_pct:.0f}%</span>
            <span style="color:{T["danger"]};font-weight:700;">Loss {loss_pct:.0f}%</span>
        </div>
        <div style="display:flex;height:14px;border-radius:7px;overflow:hidden;
                    background:rgba(0,0,0,0.05);">
            <div style="width:{win_pct:.1f}%;background:{T["green"]};"></div>
            <div style="width:{tie_pct:.1f}%;background:{T["tx2"]};opacity:0.4;"></div>
            <div style="width:{loss_pct:.1f}%;background:{T["danger"]};"></div>
        </div>
        <div style="font-size:11px;color:{T["tx2"]};margin-top:4px;text-align:center;">
            Projected: {pw:.0f}-{pl:.0f}
        </div>
    </div>
    """


# ── Layout ────────────────────────────────────────────────────────────

ctx, main = render_context_columns()

# ── Context panel ─────────────────────────────────────────────────────

with ctx:
    # Week navigator
    if "matchup_week" not in st.session_state:
        st.session_state["matchup_week"] = current_week

    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    with nav_c1:
        if st.button("◀", key="week_prev", help="Previous week"):
            if st.session_state["matchup_week"] > 1:
                st.session_state["matchup_week"] -= 1
                st.rerun()
    with nav_c2:
        sel_wk = st.session_state["matchup_week"]
        if sel_wk == current_week:
            week_label = "Current"
        elif sel_wk < current_week:
            week_label = "Past"
        else:
            week_label = "Future"
        st.markdown(
            f'<div style="text-align:center;font-size:14px;font-weight:700;'
            f'color:{T["tx"]};padding:4px 0;">Week {sel_wk}'
            f'<br><span style="font-size:11px;font-weight:400;color:{T["tx2"]};">'
            f"{week_label}</span></div>",
            unsafe_allow_html=True,
        )
    with nav_c3:
        if st.button("▶", key="week_next", help="Next week"):
            if st.session_state["matchup_week"] < 24:
                st.session_state["matchup_week"] += 1
                st.rerun()

    selected_week = st.session_state["matchup_week"]

    # Recompute win probs if week changed from the pre-computed one
    if win_prob_data is None or win_prob_data.get("_week") != selected_week:
        win_prob_data = _compute_win_probs(selected_week)
        if win_prob_data:
            win_prob_data["_week"] = selected_week

    # Show opponent for selected week
    opp_display = "Unknown"
    if win_prob_data:
        opp_display = win_prob_data.get("opponent", "Unknown")
    elif matchup_data and selected_week == current_week:
        opp_display = matchup_data.get("opp_name", "Unknown")
    elif full_schedule and user_team_name:
        found_opp = (
            find_user_opponent(full_schedule, selected_week, user_team_name) if STANDINGS_ENGINE_AVAILABLE else None
        )
        if found_opp:
            opp_display = found_opp

    render_context_card(
        "Opponent",
        f'<div style="font-size:13px;color:{T["tx"]};font-weight:600;">{opp_display}</div>',
    )

    # Win probability context card
    if win_prob_data:
        render_context_card("Win Probability", _build_win_prob_context_html(win_prob_data))

    st.markdown("---")

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

    # Opponent weakness card from MatchupContextService
    try:
        from src.matchup_context import get_matchup_context

        _ctx = get_matchup_context()
        _opp = _ctx.get_opponent_context()
        if _opp.get("name") and _opp["name"] != "Unknown":
            _weak = _opp.get("weaknesses", [])
            _strong = _opp.get("strengths", [])
            _opp_html = f'<div style="font-size:12px;color:{THEME["tx"]};">'
            _opp_html += f"<b>{_opp['name']}</b>"
            if _opp.get("tier"):
                _opp_html += f" ({_opp['tier']})"
            _opp_html += "</div>"
            if _weak:
                weak_str = ", ".join(str(w) for w in _weak[:4])
                _opp_html += (
                    f'<div style="font-size:11px;color:{THEME["danger"]};margin-top:4px;">Weak: {weak_str}</div>'
                )
            if _strong:
                strong_str = ", ".join(str(s) for s in _strong[:4])
                _opp_html += (
                    f'<div style="font-size:11px;color:{THEME["green"]};margin-top:2px;">Strong: {strong_str}</div>'
                )
            render_context_card("Opponent Weakness", _opp_html)
    except Exception:
        pass  # Graceful fallback

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

    # Fetch weekly schedule (cached for 1 hour — schedule doesn't change frequently)
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_schedule(_days: int):
        try:
            return get_weekly_schedule(days_ahead=_days)
        except Exception as exc:
            logger.warning("Could not fetch schedule: %s", exc)
            return None

    weekly_schedule = None
    if SCHEDULE_AVAILABLE:
        with st.spinner("Fetching MLB schedule..."):
            weekly_schedule = _cached_schedule(int(days_ahead))

    if not weekly_schedule:
        schedule_warning = (
            "No live schedule data available. Matchup ratings are computed "
            "from projection data only (games count will be 0 for all players). "
            "Connect to the internet and ensure statsapi is installed for live schedules."
        )

    # Run the matchup planner
    park_factors = PARK_FACTORS if PARK_FACTORS_AVAILABLE else {}

    # Fetch team strength data from MatchupContextService for pitcher matchup ratings
    team_batting_stats = None
    try:
        from src.matchup_context import get_matchup_context

        ctx = get_matchup_context()
        team_batting_stats = ctx.get_all_team_strengths()
    except Exception:
        pass  # Graceful fallback — ratings use defaults without team strength

    with st.spinner("Computing matchup ratings..."):
        try:
            ratings_df = compute_weekly_matchup_ratings(
                roster=filtered_roster,
                weekly_schedule=weekly_schedule,
                park_factors=park_factors or None,
                team_batting_stats=team_batting_stats,
            )
        except Exception as exc:
            logger.exception("Matchup planner failed")
            st.error(f"Error computing matchup ratings: {exc}")
            ratings_df = pd.DataFrame()

    # Check if we have valid ratings data for player-matchup tabs
    has_player_ratings = True
    if ratings_df is None or ratings_df.empty:
        has_player_ratings = False
    elif "games_count" in ratings_df.columns and ratings_df["games_count"].sum() == 0:
        has_player_ratings = False

    if has_player_ratings:
        # Merge mlb_id from pool for headshot rendering
        if "mlb_id" not in ratings_df.columns and "player_id" in ratings_df.columns:
            _id_to_mlb = pool[["player_id", "mlb_id"]].drop_duplicates(subset="player_id")
            ratings_df = ratings_df.merge(_id_to_mlb, on="player_id", how="left")

        # Sort by rating descending
        ratings_df = ratings_df.sort_values("weekly_matchup_rating", ascending=False).reset_index(drop=True)

    # ── Summary metrics ───────────────────────────────────────────────

    if has_player_ratings:
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
        m4.metric("Average Games", f"{avg_games:.2f}")

    st.markdown("---")

    # ── Tab view: Category Probabilities | Player Matchups | Per-Game Detail | Hitters | Pitchers
    tab_probs, tab_summary, tab_detail, tab_hitters, tab_pitchers = st.tabs(
        ["Category Probabilities", "Player Matchups", "Per-Game Detail", "Hitters Only", "Pitchers Only"]
    )

    # ── Tab: Category Probabilities ──────────────────────────────────

    with tab_probs:
        selected_week = st.session_state.get("matchup_week", current_week)
        is_past_week = selected_week < current_week

        if is_past_week and matchup_data and selected_week == current_week - 1:
            # Past week — show actual results hint
            st.subheader(f"Week {selected_week} — Past Results")
            st.info(
                "Past week results are shown from Yahoo live data when available. "
                "Category probabilities are only shown for current and future weeks."
            )
        elif is_past_week:
            st.subheader(f"Week {selected_week} — Past Week")
            st.info("This is a past week. Category probabilities are projections for current and future weeks only.")

        if not is_past_week and win_prob_data:
            cats = win_prob_data.get("categories", [])
            opp_name_disp = win_prob_data.get("opponent", "opponent")
            overall_wp = win_prob_data.get("overall_win_pct", 0) * 100

            st.subheader(f"Week {selected_week} vs {opp_name_disp}")

            # Overall summary line
            proj_score = win_prob_data.get("projected_score", {})
            st.markdown(
                f'<div style="font-size:14px;color:{T["tx"]};margin-bottom:12px;">'
                f'Overall win probability: <strong style="color:{T["green"] if overall_wp >= 55 else T["danger"] if overall_wp < 45 else T["warn"]};">'
                f"{overall_wp:.0f}%</strong>"
                f" &mdash; Projected score: {proj_score.get('W', 0):.0f}-{proj_score.get('L', 0):.0f}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Sort categories by win probability descending
            sorted_cats = sorted(cats, key=lambda c: c["win_pct"], reverse=True)

            # Build and render bars
            html = _build_category_prob_html(sorted_cats)
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.6);border-radius:12px;'
                f'padding:12px 16px;border:1px solid rgba(0,0,0,0.06);">{html}</div>',
                unsafe_allow_html=True,
            )

            # Legend for bar colors
            st.markdown(
                f'<div style="font-size:11px;color:{T["tx2"]};margin-top:8px;padding:4px 0;">'
                f'<span style="color:{T["green"]};">70%+ Strong advantage</span> &nbsp;|&nbsp; '
                f'<span style="color:{T["sky"]};">55-70% Lean advantage</span> &nbsp;|&nbsp; '
                f'<span style="color:{T["warn"]};">45-55% Toss-up</span> &nbsp;|&nbsp; '
                f'<span style="color:{T["danger"]};">Below 45% Disadvantage</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                "Based on current rosters and remaining-season projections. "
                "Probabilities computed via Gaussian copula with 10,000 simulations."
            )

        elif not is_past_week and not win_prob_data:
            st.subheader(f"Week {selected_week} Category Probabilities")
            if not STANDINGS_ENGINE_AVAILABLE:
                st.info(
                    "The standings engine module is not available. "
                    "Ensure src/standings_engine.py is present and all dependencies are installed."
                )
            elif not user_team_name:
                st.info(
                    "Connect your Yahoo league to see category win probabilities. "
                    "The Category Probabilities tab requires league roster data."
                )
            else:
                st.info(
                    "Could not compute category win probabilities for this week. "
                    "This may happen if the opponent cannot be identified or roster data is missing."
                )

    # ── Build display DataFrame (shared across player tabs) ───────────

    def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
        """Build a flat display DataFrame from ratings output."""
        has_mlb_id = "mlb_id" in df.columns
        rows = []
        for _, row in df.iterrows():
            tier = str(row.get("matchup_tier", "neutral"))
            entry = {
                "Player": str(row.get("name", "")),
                "Position": str(row.get("positions", "")),
                "Type": "Hitter" if row.get("is_hitter", True) else "Pitcher",
                "Games": int(row.get("games_count", 0)),
                "Rating": f"{float(row.get('weekly_matchup_rating', 0.0)):.2f}",
                "Tier": _TIER_LABELS.get(tier, tier.capitalize()),
            }
            if has_mlb_id:
                entry["mlb_id"] = row.get("mlb_id")
            rows.append(entry)
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

    # ── Tab: Player Matchups (was "Summary") ─────────────────────────

    with tab_summary:
        if not has_player_ratings:
            st.info(
                "No matchup ratings could be computed. Check that the roster and "
                "schedule data are available. Try running the app bootstrap or "
                "syncing Yahoo data to refresh schedule information."
            )
        else:
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
        if not has_player_ratings:
            st.info("No matchup ratings available for per-game detail.")
        else:
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
                    f"Rating: {rating:.2f} &bull; {tier_label}"
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
        if not has_player_ratings:
            st.info("No matchup ratings available for hitters.")
        else:
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
        if not has_player_ratings:
            st.info("No matchup ratings available for pitchers.")
        else:
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

page_timer_footer("Matchup Planner")
