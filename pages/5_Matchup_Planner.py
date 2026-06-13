"""Matchup Planner — Weekly per-game matchup ratings + category win probabilities."""

from __future__ import annotations

import html as _html
import logging

import pandas as pd
import streamlit as st

from src.ai.chat import render_chat_widget
from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.league_manager import get_team_roster
from src.ui_shared import (
    THEME,
    _headshot_img_html,
    build_eyebrow_html,
    build_heatbar_html,
    build_panel_html,
    format_stat,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_empty_state,
    render_matchup_ticker,
    render_page_header,
    render_player_select,
    render_reco_banner,
    team_logo_url,
)
from src.usage import log_page_view
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

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Matchup Planner",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

init_db()
inject_custom_css()
require_auth()
require_page_enabled("page:5_Matchup_Planner")
log_page_view("Matchup Planner")
render_chat_widget("Matchup Planner")
page_timer_start()

# ── Load player pool ──────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def _get_player_pool():
    """Return the enriched player pool, cached for 5 minutes per the Yahoo TTL."""
    return load_player_pool()


pool = _get_player_pool()
if pool.empty:
    st.warning("No player data loaded. Run load_sample_data.py or bootstrap from the main app.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# ── Load league rosters ───────────────────────────────────────────────

yds = get_yahoo_data_service()
rosters = yds.get_rosters()

# ── Get current matchup + schedule for category probabilities ─────────

matchup_data = yds.get_matchup()
if matchup_data and matchup_data.get("week"):
    current_week = int(matchup_data["week"])
else:
    # No live Yahoo matchup in this session (read-only member, or between weeks):
    # derive the current week from the league calendar instead of defaulting to 1
    # (which mislabeled today's week as "Future" for every member).
    try:
        from src.opponent_intel import get_week_number

        current_week = get_week_number()
    except Exception:
        current_week = 1

# Attempt to load the full league schedule for week navigation
full_schedule: dict[int, list[tuple[str, str]]] = {}
try:
    full_schedule = yds.get_full_league_schedule() or {}
except Exception:
    logger.debug("Could not load full league schedule", exc_info=True)

# Determine user team name
user_team_name: str | None = resolve_viewer_team_name(rosters)
if user_team_name is not None:
    user_team_name = str(user_team_name)

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
    # 2026-05-17 Section 3 D5: source from LeagueConfig.season_weeks
    # (was hardcoded 26 in Section 2 L10 fix; now derived from config).
    #
    # Intentionally NOT the canonical src.league_rules.weeks_remaining():
    # that helper is CALENDAR-based (weeks left from *today*), whereas the
    # Matchup Planner analyzes an arbitrary selected matchup `week` and needs
    # the horizon relative to THAT week's index (weeks remaining including the
    # analyzed week). Swapping in the calendar helper would ignore `week` and
    # mis-horizon any matchup other than the current one. Deep-audit punchlist
    # Task 10 step 5 prescribed the swap here; closed as wontfix-by-design
    # because the semantics differ (week-index-relative vs calendar-from-today).
    weeks_remaining = max(1, config.season_weeks - week + 1)

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

render_page_header(
    "Matchup Planner",
    eyebrow="THIS WEEK",
    fig="FIG.05 — MATCHUP GRID",
)
render_reco_banner(banner_teaser, "", "calendar")
render_matchup_ticker()

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


def _opp_logo_html(opp_abbr: str, size: int = 14) -> str:
    """Return an inline MLB team logo <img> for an opponent abbreviation."""
    if not opp_abbr:
        return ""
    return (
        f'<img src="{team_logo_url(opp_abbr)}" alt="" '
        f'style="width:{size}px;height:{size}px;vertical-align:-3px;margin:0 4px 0 2px;" />'
    )


def _games_detail_html(games: list[dict]) -> str:
    """Render per-game details as instrument cards (accent-left + date eyebrow)."""
    if not games:
        return '<span style="color:var(--fp-tx-muted);font-size:12px;">No games scheduled</span>'
    parts = []
    for g in games:
        date = str(g.get("game_date", ""))
        opp = str(g.get("opponent", ""))
        raw = g.get("raw_score", 0.0)
        home_away = g.get("home_away", 1.0)
        loc = "Home" if home_away and home_away > 1.0 else "Away"
        pf = g.get("park_factor", 1.0)
        _opp_logo = _opp_logo_html(opp)
        parts.append(
            '<div class="instr-panel accent-left" style="padding:8px 14px;border-radius:10px;">'
            f'<div class="t-eyebrow" style="margin-bottom:2px;">{_html.escape(date)}</div>'
            '<span style="font-size:11px;color:var(--fp-tx-muted);display:inline-flex;'
            'align-items:center;gap:2px;">vs '
            f'{_opp_logo}<b style="color:var(--fp-tx);">{_html.escape(opp)}</b> '
            f'<span style="color:var(--fp-tx-subtle);">({loc} · PF {float(pf):.2f} · '
            f"Score {float(raw):.2f})</span></span></div>"
        )
    return '<div style="display:flex;flex-direction:column;gap:6px;">' + "".join(parts) + "</div>"


def _section_label(text: str, *, fig: str = "") -> str:
    """Return an instrument-style eyebrow section label (orange bar + Archivo)."""
    fig_html = ""
    if fig:
        fig_html = (
            '<span style="font-family:var(--font-mono);font-weight:500;font-size:10px;'
            f'letter-spacing:.12em;color:var(--fp-tx-muted);">{_html.escape(str(fig))}</span>'
        )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'margin:2px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--fp-divider);">'
        '<div style="display:flex;align-items:center;gap:9px;">'
        '<span style="width:3px;height:14px;background:var(--fp-primary);border-radius:2px;'
        'box-shadow:0 0 8px rgba(255,109,0,.5);"></span>'
        f"{build_eyebrow_html(text)}</div>{fig_html}</div>"
    )


# ── Helper: build category probability bar HTML ──────────────────────


def _cat_header_color(cat: str) -> str:
    """Return category header color: blue for hitting, red for pitching."""
    if cat in _HIT_CATS:
        return T["sky"]
    return T["primary"]


def _build_category_prob_html(cat_data: list[dict]) -> str:
    """Build full HTML for category probability heat bars.

    Each entry has: name, user_proj, opp_proj, win_pct, confidence, is_inverse.
    """
    rows: list[str] = []
    for cat in cat_data:
        name = cat["name"]
        pct = cat["win_pct"] * 100
        user_proj = cat["user_proj"]
        opp_proj = cat["opp_proj"]
        confidence = cat.get("confidence", "low")
        header_color = _cat_header_color(name)
        is_inverse = cat.get("is_inverse", False)
        direction_hint = " (lower is better)" if is_inverse else ""

        # Confidence badge
        conf_opacity = "1.0" if confidence == "high" else "0.7" if confidence == "medium" else "0.5"

        # Format projection values
        if name in ("AVG", "OBP") or name in ("ERA", "WHIP"):
            user_str = format_stat(user_proj, name)
            opp_str = format_stat(opp_proj, name)
        else:
            user_str = f"{user_proj:.1f}"
            opp_str = f"{opp_proj:.1f}"

        # Canonical Combustion heat bar: orange gradient when favored (>=50%),
        # steel when trailing. Mono percentage figure sits beside the track.
        _heatbar = build_heatbar_html(pct, win=(pct >= 50))
        _pct_color = "var(--fp-primary)" if pct >= 50 else "var(--fp-cold)"

        # Single-line HTML, joined with "" (F-VIS-1, 2026-06-02 browser
        # walkthrough). A multi-line pretty-printed f-string broke rendering:
        # Streamlit's markdown treats lines indented 4+ spaces as a code block,
        # and the blank line between rows terminated the first HTML block — so
        # the 2nd+ category bars rendered as escaped raw HTML. Keeping each row on
        # one logical line with no leading indentation, and joining with "", makes
        # the whole bar set one uninterrupted HTML block.
        row_html = (
            f'<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--fp-divider);">'
            f'<div style="width:55px;font-family:var(--font-display);font-weight:800;font-size:13px;color:{header_color};flex-shrink:0;">{name}</div>'
            f'<div style="flex:1;">{_heatbar}</div>'
            f'<div style="width:46px;flex-shrink:0;text-align:right;font-family:var(--font-mono);font-weight:600;font-size:12px;color:{_pct_color};">{pct:.0f}%</div>'
            f'<div style="width:130px;font-family:var(--font-mono);font-size:10.5px;color:{T["tx2"]};text-align:right;flex-shrink:0;opacity:{conf_opacity};letter-spacing:.02em;">{user_str} vs {opp_str}{direction_hint}</div>'
            f"</div>"
        )
        rows.append(row_html)

    return "".join(rows)


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
        f'<div style="font-size:13px;color:{T["tx"]};font-weight:600;">{_html.escape(str(opp_display))}</div>',
    )

    # Win probability context card
    if win_prob_data:
        render_context_card("Win Probability", _build_win_prob_context_html(win_prob_data))

    st.markdown('<div class="hr-fade"></div>', unsafe_allow_html=True)

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
        _viewer_team = resolve_viewer_team_name(rosters)
        default_team = _viewer_team if _viewer_team in team_names else team_names[0]
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
        render_empty_state(
            "No league roster data loaded",
            "Using the full player pool for demonstration.",
            icon_key="users",
        )
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
        # Resolve the SELECTED team's opponent (defaults to the viewer's own team)
        # instead of the global/admin opponent, so each member sees the right
        # matchup on this card. (2026-06-01 audit.)
        _opp_name = None
        if STANDINGS_ENGINE_AVAILABLE and team_name and full_schedule:
            try:
                _opp_name = find_user_opponent(full_schedule, selected_week, team_name)
            except Exception:
                _opp_name = None
        _opp = _ctx.get_opponent_context(week=selected_week, opponent_name=_opp_name)
        if _opp.get("name") and _opp["name"] != "Unknown":
            _weak = _opp.get("weaknesses", [])
            _strong = _opp.get("strengths", [])
            _opp_html = f'<div style="font-size:12px;color:{THEME["tx"]};">'
            _opp_html += f"<b>{_html.escape(str(_opp['name']))}</b>"
            if _opp.get("tier"):
                _opp_html += f" ({_html.escape(str(_opp['tier']))})"
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
        # Fallback: use the top 23 players from the pool as a demo roster.
        # (BUG-024) Surface a banner so the user knows the displayed
        # matchup ratings are against demo data, not their real team.
        st.warning(
            "No team selected or roster not yet loaded — showing **demo data** "
            "based on the top 23 players from the pool. Matchup ratings below "
            "are illustrative only. Connect to Yahoo or pick a team to see "
            "your real roster."
        )
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
        render_empty_state(f"No {player_type.lower()} found on this roster", icon_key="users")
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
    if (
        ratings_df is None
        or ratings_df.empty
        or "games_count" in ratings_df.columns
        and ratings_df["games_count"].sum() == 0
    ):
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

    st.markdown('<div class="hr-fade"></div>', unsafe_allow_html=True)

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
            render_empty_state(
                "Past week",
                "Past week results are shown from Yahoo live data when available. "
                "Category probabilities are only shown for current and future weeks.",
                icon_key="calendar",
            )
        elif is_past_week:
            st.subheader(f"Week {selected_week} — Past Week")
            render_empty_state(
                "Past week",
                "Category probabilities are projections for current and future weeks only.",
                icon_key="calendar",
            )

        if not is_past_week and win_prob_data:
            cats = win_prob_data.get("categories", [])
            opp_name_disp = win_prob_data.get("opponent", "opponent")
            overall_wp = win_prob_data.get("overall_win_pct", 0) * 100

            st.subheader(f"Week {selected_week} vs {opp_name_disp}")

            # Overall summary line — gradient hero numeral on the headline figure
            proj_score = win_prob_data.get("projected_score", {})
            st.markdown(
                f'<div style="font-size:14px;color:{T["tx"]};margin-bottom:12px;'
                f'display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">'
                f"<span>Overall win probability:</span>"
                f'<strong class="hero-num" style="font-size:34px;line-height:1;">{overall_wp:.0f}%</strong>'
                f"<span>&mdash; Projected score: {proj_score.get('W', 0):.0f}-{proj_score.get('L', 0):.0f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Sort categories by win probability descending
            sorted_cats = sorted(cats, key=lambda c: c["win_pct"], reverse=True)

            # Build and render heat bars inside an instrument panel (corner
            # ticks + Archivo header + orange accent), matching the dossier look.
            html = _build_category_prob_html(sorted_cats)
            st.markdown(
                build_panel_html(
                    "Category Outlook",
                    html,
                    fig_label="FIG.02 · WIN PROBABILITY BY CAT",
                    accent="top",
                ),
                unsafe_allow_html=True,
            )

            # Legend for heat-bar colors
            st.markdown(
                '<div style="font-size:11px;color:var(--fp-tx-muted);margin-top:8px;padding:4px 0;">'
                '<span style="color:var(--fp-primary);font-weight:600;">Orange — favored (50%+)</span> &nbsp;|&nbsp; '
                '<span style="color:var(--fp-cold);font-weight:600;">Steel — trailing (below 50%)</span>'
                "</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                "Based on current rosters and remaining-season projections. "
                "Probabilities computed via Gaussian copula with 10,000 simulations."
            )

        elif not is_past_week and not win_prob_data:
            st.subheader(f"Week {selected_week} Category Probabilities")
            if not STANDINGS_ENGINE_AVAILABLE:
                render_empty_state(
                    "Standings engine unavailable",
                    "Ensure src/standings_engine.py is present and all dependencies are installed.",
                    icon_key="warning",
                )
            elif not user_team_name:
                render_empty_state(
                    "Connect your Yahoo league to see category win probabilities",
                    "The Category Probabilities tab requires league roster data.",
                    icon_key="users",
                )
            else:
                render_empty_state(
                    "Could not compute category win probabilities for this week",
                    "This may happen if the opponent cannot be identified or roster data is missing.",
                    icon_key="calendar",
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
            render_empty_state(
                "No matchup ratings could be computed",
                "Check that the roster and schedule data are available. Try running "
                "the app bootstrap or syncing Yahoo data to refresh schedule information.",
            )
        else:
            st.markdown(
                _section_label("Weekly Matchup Summary", fig=f"∑ {len(ratings_df)} PLAYERS"), unsafe_allow_html=True
            )
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
            render_empty_state("No matchup ratings available for per-game detail")
        else:
            st.markdown(_section_label("Per-Game Matchup Detail", fig="PARK · PLATOON"), unsafe_allow_html=True)
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

                _det_headshot = _headshot_img_html(row.get("mlb_id"), size=26)
                _det_team = str(row.get("team", "") or "")
                _det_team_logo = _opp_logo_html(_det_team, size=15) if _det_team else ""
                _safe_name = _html.escape(name)
                _safe_pos = _html.escape(positions)
                header_html = (
                    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
                    f'border-left:4px solid {tier_color};padding:2px 0 2px 11px;margin-bottom:6px;">'
                    f'<span style="display:inline-flex;align-items:center;">{_det_headshot}'
                    f'<span style="font-family:var(--font-display);font-weight:800;font-size:14px;'
                    f'color:var(--fp-tx);">{_safe_name}</span></span>'
                    f"{_det_team_logo}"
                    f'<span style="font-family:var(--font-mono);font-size:11px;color:var(--fp-tx-muted);'
                    f'letter-spacing:.03em;">{_safe_pos} · {player_type_label} · {games_count} GM</span>'
                    f'<span style="margin-left:auto;font-family:var(--font-display);font-size:12px;'
                    f'font-weight:800;color:{tier_color};">RATING {rating:.2f} · {tier_label.upper()}</span>'
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
            render_empty_state("No matchup ratings available for hitters")
        else:
            hitter_df = ratings_df[ratings_df["is_hitter"] == True].copy()  # noqa: E712
            if hitter_df.empty:
                render_empty_state("No hitters found in current selection", icon_key="users")
            else:
                st.markdown(_section_label("Hitters", fig=f"∑ {len(hitter_df)}"), unsafe_allow_html=True)
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
            render_empty_state("No matchup ratings available for pitchers")
        else:
            pitcher_df = ratings_df[ratings_df["is_hitter"] == False].copy()  # noqa: E712
            if pitcher_df.empty:
                render_empty_state("No pitchers found in current selection", icon_key="users")
            else:
                st.markdown(_section_label("Pitchers", fig=f"∑ {len(pitcher_df)}"), unsafe_allow_html=True)
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
    st.markdown('<div class="hr-fade"></div>', unsafe_allow_html=True)
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
render_feedback_widget("Matchup Planner")
