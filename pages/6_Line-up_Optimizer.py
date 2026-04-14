"""Lineup — Merged optimizer + start/sit advisor with 6-tab interface.

Combines the LP optimizer pipeline (11 modules) with the 3-layer start/sit
decision model into a single unified lineup management page:

  1. Optimize: mode selector, alpha slider, LP solve, starting lineup + bench
  2. Start/Sit: per-slot comparison with bench alternatives and free agent upgrades
  3. Category Analysis: non-linear SGP weights, maximin, standings position
  4. Head-to-Head: per-category win probabilities against weekly opponent
  5. Streaming: pitcher streaming candidates + two-start SP detection
  6. Roster: full roster with health badges and player cards
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

from src.database import (
    coerce_numeric_df,
    get_connection,
    init_db,
    load_player_pool,
)
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_free_agents, get_team_roster
from src.standings_utils import get_all_team_totals
from src.ui_shared import (
    METRIC_TOOLTIPS,
    PAGE_ICONS,
    THEME,
    format_stat,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
    render_styled_table,
)
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

logger = logging.getLogger(__name__)

T = THEME

# ── Optional module imports ──────────────────────────────────────────

# Pipeline (primary optimizer)
try:
    from src.optimizer.pipeline import LineupOptimizerPipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Fallback: basic LP optimizer
try:
    from src.lineup_optimizer import LineupOptimizer

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# Head-to-Head engine
try:
    from src.optimizer.h2h_engine import (
        compute_h2h_category_weights,
        estimate_h2h_win_probability,
    )

    H2H_AVAILABLE = True
except ImportError:
    H2H_AVAILABLE = False

# Non-linear SGP weights
try:
    from src.optimizer.sgp_theory import compute_nonlinear_weights

    SGP_AVAILABLE = True
except ImportError:
    SGP_AVAILABLE = False

# Matchup adjustments
try:
    from src.optimizer.matchup_adjustments import get_weekly_schedule

    MATCHUP_AVAILABLE = True
except ImportError:
    MATCHUP_AVAILABLE = False

# Park factors
try:
    from src.data_bootstrap import PARK_FACTORS

    PARK_FACTORS_AVAILABLE = True
except ImportError:
    PARK_FACTORS: dict[str, float] = {}
    PARK_FACTORS_AVAILABLE = False

# Start/sit advisor
START_SIT_AVAILABLE = False
try:
    from src.start_sit import classify_matchup_state, start_sit_recommendation

    START_SIT_AVAILABLE = True
except Exception:
    logger.warning("start_sit module unavailable", exc_info=True)

# IP tracker
IP_TRACKER_AVAILABLE = False
try:
    from src.ip_tracker import compute_weekly_ip_projection, get_days_remaining_in_week

    IP_TRACKER_AVAILABLE = True
except Exception:
    pass

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heater | Line-up Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()
page_timer_start()

render_page_layout(
    "LINE-UP OPTIMIZER",
    banner_teaser="Optimize your weekly lineup and make start/sit decisions",
    banner_icon="lineup_optimizer",
)

# ── Category display names ───────────────────────────────────────────

CAT_DISPLAY_NAMES: dict[str, str] = {
    "r": "Runs",
    "hr": "Home Runs",
    "rbi": "Runs Batted In",
    "sb": "Stolen Bases",
    "avg": "Batting Average",
    "obp": "On-Base Percentage",
    "w": "Wins",
    "l": "Losses",
    "sv": "Saves",
    "k": "Strikeouts",
    "era": "Earned Run Average",
    "whip": "Walks + Hits per Inning Pitched",
}

ALL_CATS = list(CAT_DISPLAY_NAMES.keys())


# ── Load user team ───────────────────────────────────────────────────

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
    st.warning("No user team identified in roster data.")
    st.stop()

user_team_name = str(user_teams.iloc[0]["team_name"])
_display_team_name = "".join(c for c in user_team_name if ord(c) < 0x10000).strip()

roster = get_team_roster(user_team_name)
if roster is None or roster.empty:
    st.info("Your roster is empty. Import roster data first.")
    st.stop()

# User player IDs for start/sit filtering
user_player_ids: list[int] = roster["player_id"].tolist() if "player_id" in roster.columns else []

# ── Normalize columns ────────────────────────────────────────────────

if "name" in roster.columns and "player_name" not in roster.columns:
    roster = roster.rename(columns={"name": "player_name"})

required_cols = ["player_id", "player_name", "positions", "is_hitter"]
stat_cols = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
component_cols = ["ip", "h", "ab", "er", "bb_allowed", "h_allowed"]
for col in required_cols + stat_cols + component_cols:
    if col not in roster.columns:
        roster[col] = 0

# Coerce numeric columns (Python 3.13+ SQLite may return bytes)
for col in stat_cols + component_cols:
    if col in roster.columns:
        roster[col] = pd.to_numeric(roster[col], errors="coerce").fillna(0)


# ── Projection fallback: load from player pool if stats are all zeros ──

pool = load_player_pool()
if not pool.empty:
    pool = pool.rename(columns={"name": "player_name"})

all_stat_cols = stat_cols + component_cols
numeric_stats = roster[all_stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
if numeric_stats.sum().sum() == 0:
    st.warning(
        "Player projections are missing — the optimizer cannot differentiate starters from bench. "
        "Loading projection data from the player pool as a fallback."
    )
    if not pool.empty:
        available = [c for c in all_stat_cols if c in pool.columns]
        fallback = pool[["player_id"] + available].copy()
        roster = roster.drop(columns=[c for c in available if c in roster.columns], errors="ignore")
        roster = roster.merge(fallback, on="player_id", how="left")
        for c in all_stat_cols:
            if c in roster.columns:
                roster[c] = pd.to_numeric(roster[c], errors="coerce").fillna(0)

# ── Projected IP lookup (for IP budget tracker) ─────────────────────
# The roster "ip" column contains ACTUAL IP pitched so far this season.
# For weekly IP projection we need full-season PROJECTED IP from the player pool.
_proj_ip_lookup: dict[int, float] = {}
if not pool.empty and "ip" in pool.columns:
    for _, _pp in pool.iterrows():
        _pid = _pp.get("player_id")
        _pip = float(_pp.get("ip", 0) or 0)
        if _pid is not None and _pip > 0:
            _proj_ip_lookup[_pid] = _pip


# ── League config ────────────────────────────────────────────────────

league_config = LeagueConfig()

optimizer_config = {
    "hitting_categories": ["r", "hr", "rbi", "sb", "avg", "obp"],
    "pitching_categories": ["w", "l", "sv", "k", "era", "whip"],
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

if not OPTIMIZER_AVAILABLE and not PIPELINE_AVAILABLE:
    st.error("Lineup Optimizer requires PuLP. Install it with: `pip install pulp`")
    st.stop()


# ── Apply health-based stat discount BEFORE optimization ─────────────
# Uses get_health_adjusted_pool() which accounts for both injury history
# AND current IL/DTD status (IL15=0.84x, IL60=0.55x, DTD=0.95x, NA excluded).

HEALTH_PENALTY_WEIGHT = 0.15
health_dict: dict[int, float] = {}
try:
    from src.trade_intelligence import get_health_adjusted_pool as _get_hap

    roster = _get_hap(roster)
    # Populate health_dict from the adjusted pool for display use
    for _, _row in roster.iterrows():
        _pid = _row.get("player_id")
        if _pid is not None:
            health_dict[_pid] = float(_row.get("health_score", 1.0) or 1.0)
except ImportError:
    # Fallback: raw injury history when trade_intelligence unavailable
    try:
        conn = get_connection()
        try:
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
            injury_df = coerce_numeric_df(injury_df)
        finally:
            conn.close()

        if not injury_df.empty:
            counting_cols = ["r", "hr", "rbi", "sb", "w", "l", "sv", "k"]
            for col in counting_cols:
                if col in roster.columns:
                    roster[col] = pd.to_numeric(roster[col], errors="coerce").astype(float)
            for idx, row in roster.iterrows():
                pid = row.get("player_id")
                pi = injury_df[injury_df["player_id"] == pid]
                if not pi.empty:
                    hs = compute_health_score(pi["games_played"].tolist(), pi["games_available"].tolist())
                else:
                    hs = 1.0
                health_dict[pid] = hs
                penalty = HEALTH_PENALTY_WEIGHT * (1.0 - hs)
                if penalty > 0:
                    for col in counting_cols:
                        if col in roster.columns:
                            roster.at[idx, col] = float(roster.at[idx, col]) * (1.0 - penalty)
    except Exception:
        pass  # Graceful degradation when injury data unavailable
except Exception:
    pass  # Graceful degradation


# ── Load standings and schedule ──────────────────────────────────────

standings = yds.get_standings()

team_totals = get_all_team_totals()
my_totals: dict[str, float] = team_totals.get(user_team_name, {})

# Opponent weekly totals (league median proxy)
opp_weekly_totals: dict[str, float] | None = None
if team_totals and my_totals:
    _all_cats: set[str] = set()
    for _tt in team_totals.values():
        _all_cats.update(_tt.keys())
    _median_totals: dict[str, float] = {}
    for _cat in _all_cats:
        _vals = [tt.get(_cat, 0.0) for tt in team_totals.values() if _cat in tt]
        if _vals:
            _vals.sort()
            _mid = len(_vals) // 2
            _median_totals[_cat] = _vals[_mid]
    if _median_totals:
        opp_weekly_totals = _median_totals

# Fetch weekly schedule (cached in session state)
week_schedule: list = []
if MATCHUP_AVAILABLE:
    if "lineup_week_schedule" not in st.session_state:
        try:
            st.session_state["lineup_week_schedule"] = get_weekly_schedule(days_ahead=7)
        except Exception:
            st.session_state["lineup_week_schedule"] = []
    week_schedule = st.session_state.get("lineup_week_schedule", [])


# ── IP budget tracking ───────────────────────────────────────────────

ip_budget_html = ""
if IP_TRACKER_AVAILABLE and not roster.empty:
    try:
        _pitcher_data = []
        for _, _p in roster.iterrows():
            if _p.get("is_hitter") == 0 or any(
                pos.strip() in ("P", "SP", "RP") for pos in str(_p.get("positions", "")).upper().split(",")
            ):
                _pitcher_data.append(
                    {
                        "name": _p.get("player_name", ""),
                        "ip": _proj_ip_lookup.get(_p.get("player_id"), float(_p.get("ip", 0) or 0)),
                        "positions": str(_p.get("positions", "")),
                        "status": str(_p.get("status", "active")),
                        "is_starter": "SP" in str(_p.get("positions", "")).upper(),
                    }
                )
        if _pitcher_data:
            _ip_result = compute_weekly_ip_projection(_pitcher_data, get_days_remaining_in_week())
            _ip_color_map = {"safe": "#2d6a4f", "warning": "#ff9f1c", "danger": "#e63946"}
            ip_budget_html = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
                f'<span style="font-size:11px;color:{T["tx2"]};">Pace</span>'
                f'<span style="font-size:13px;font-weight:bold;font-family:IBM Plex Mono,monospace;color:'
                f'{_ip_color_map.get(_ip_result["status"], T["tx2"])}">'
                f"{_ip_result['projected_ip']} / {_ip_result['ip_needed']:.0f} "
                f"({_ip_result['ip_pace']:.0f}%)</span></div>"
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:11px;color:{T["tx2"]};">Status</span>'
                f'<span style="font-size:13px;font-weight:bold;font-family:IBM Plex Mono,monospace;">'
                f"{_ip_result['message']}</span></div>"
            )
    except Exception:
        pass  # Non-fatal


# ── Load live matchup data early (before matchup state classification) ──
# Try YDS first, then fall back to the matchup ticker's session cache
# (which uses the raw Yahoo client directly and is often populated when YDS isn't)

_live_matchup_early = yds.get_matchup()
if _live_matchup_early is None:
    _live_matchup_early = st.session_state.get("_matchup_ticker_data")
_pre_live_opp_name = ""
_pre_live_my: dict[str, float] = {}
_pre_live_opp: dict[str, float] = {}
_live_losing_cats: list[str] = []
_live_winning_cats: list[str] = []
_live_tied_cats: list[str] = []
try:
    if _live_matchup_early and "categories" in _live_matchup_early:
        _pre_live_opp_name = _live_matchup_early.get("opp_name", "")
        for _mc in _live_matchup_early["categories"]:
            _cat = str(_mc.get("cat", "")).lower()
            if _cat:
                _pre_live_my[_cat] = float(_mc.get("you", 0) or 0)
                _pre_live_opp[_cat] = float(_mc.get("opp", 0) or 0)
            _result = str(_mc.get("result", "")).upper()
            _cat_display = str(_mc.get("cat", ""))
            if _result == "LOSS":
                _live_losing_cats.append(_cat_display)
            elif _result == "WIN":
                _live_winning_cats.append(_cat_display)
            elif _result == "TIE":
                _live_tied_cats.append(_cat_display)
        # Override my_totals/opp_weekly_totals with live data
        if _pre_live_my:
            my_totals = _pre_live_my
            opp_weekly_totals = _pre_live_opp
        # When standings are empty, populate team_totals from live matchup
        # so H2H tab and opponent selector have data to work with
        if not team_totals and _pre_live_opp_name and _pre_live_opp:
            team_totals[_pre_live_opp_name] = _pre_live_opp
            if user_team_name and _pre_live_my:
                team_totals[user_team_name] = _pre_live_my
except Exception:
    pass  # Non-fatal: fall back to standings-derived matchup state


# ── Matchup state classification ─────────────────────────────────────

matchup_state_label = "close"
matchup_note = "No weekly totals available. Using balanced (close) matchup strategy."
if START_SIT_AVAILABLE and my_totals and opp_weekly_totals:
    try:
        matchup_state_label = classify_matchup_state(my_totals, opp_weekly_totals, league_config)
        if _pre_live_my:
            matchup_note = "Matchup state from live Yahoo H2H matchup data."
        else:
            matchup_note = "Matchup state derived from league standings. Opponent approximated by league median."
    except Exception:
        pass


# ── 3-zone hybrid layout ────────────────────────────────────────────

ctx, main = render_context_columns()


# ── Context panel ────────────────────────────────────────────────────

with ctx:
    # Team info — show active roster size (excluding IL/NA/DTD)
    _IL_STATUSES_CTX = {"il10", "il15", "il60", "il", "na", "not active", "dl", "dtd", "day-to-day"}
    _active_count = len(user_player_ids)
    _il_count = 0
    if "status" in roster.columns:
        _il_count = int(roster["status"].apply(lambda s: str(s or "").strip().lower() in _IL_STATUSES_CTX).sum())
        _active_count = len(user_player_ids) - _il_count
    _roster_detail = f"{_active_count} active"
    if _il_count > 0:
        _roster_detail += f" + {_il_count} IL"
    render_context_card(
        "Team",
        f'<div class="context-stat-row"><span class="context-stat-label">Team</span>'
        f'<span class="context-stat-value">{_display_team_name}</span></div>'
        f'<div class="context-stat-row"><span class="context-stat-label">Roster</span>'
        f'<span class="context-stat-value">{_roster_detail}</span></div>',
    )

    # Optimizer settings
    st.markdown(
        f'<p style="font-family:Bebas Neue,sans-serif;font-size:10px;text-transform:uppercase;'
        f"letter-spacing:2px;color:{T['tx2']};margin:12px 0 6px;padding-bottom:4px;"
        f'border-bottom:1px solid {T["border"]};">Optimization Settings</p>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Optimization Mode",
        options=["quick", "standard", "full"],
        index=1,
        format_func=lambda m: {
            "quick": "Quick (<1s)",
            "standard": "Standard (2-3s)",
            "full": "Full (5-10s)",
        }[m],
        help=(
            "Quick: Enhanced projections + mean-variance only. "
            "Standard: 200 scenarios + CVaR + matchup adjustments. "
            "Full: Stochastic MIP + maximin + streaming + multi-period."
        ),
        key="lineup_mode",
    )
    # Auto-compute weeks remaining from current date (season is 24 weeks)
    try:
        from datetime import UTC as _utc
        from datetime import datetime as _dt

        # MLB 2026 fantasy Week 1 started March 23, 2026
        _SEASON_START = _dt(2026, 3, 23, tzinfo=_utc)
        _TOTAL_WEEKS = 24
        _weeks_elapsed = max(0, (_dt.now(_utc) - _SEASON_START).days // 7)
        _auto_weeks = max(1, _TOTAL_WEEKS - _weeks_elapsed)
    except Exception:
        _auto_weeks = 16
    weeks_remaining = _auto_weeks
    alpha = 0.5  # Default; auto-overridden per scope when optimization runs
    risk_aversion = st.slider(
        "Risk Aversion",
        min_value=0.0,
        max_value=0.50,
        value=0.15,
        step=0.05,
        help="Higher = prefer consistent players over boom/bust. 0 = risk neutral.",
        key="lineup_risk",
    )

    # Matchup state display — live W/L/T already loaded above
    if _live_losing_cats or _live_winning_cats or _live_tied_cats:
        _w = len(_live_winning_cats)
        _l = len(_live_losing_cats)
        _t = len(_live_tied_cats)
        _live_state_label = f"{_w}-{_l}-{_t}"
        # Build losing categories with gap values from live matchup data
        _losing_with_gaps = []
        if _live_matchup_early and "categories" in _live_matchup_early:
            _inverse = {"era", "whip", "l"}
            for _mc in _live_matchup_early["categories"]:
                if str(_mc.get("result", "")).upper() == "LOSS":
                    _cat_name = str(_mc.get("cat", ""))
                    try:
                        _you = float(_mc.get("you", 0) or 0)
                        _opp = float(_mc.get("opp", 0) or 0)
                        _cat_lower = _cat_name.lower()
                        _rate_stats = {"avg", "obp", "era", "whip"}
                        # For inverse stats, gap = your value - opp value (positive = bad)
                        if _cat_lower in _inverse:
                            _gap = _you - _opp
                            _gap_str = f"+{_gap:.2f}" if _cat_lower in ("era", "whip") else f"+{_gap:.0f}"
                        elif _cat_lower in _rate_stats:
                            _gap = _you - _opp
                            _gap_str = f"{_gap:+.3f}"
                        else:
                            _gap = _you - _opp
                            _gap_str = f"{_gap:.0f}" if abs(_gap) > 1 else f"{_gap:.0f}"
                        _losing_with_gaps.append(f"{_cat_name} ({_gap_str})")
                    except (ValueError, TypeError):
                        _losing_with_gaps.append(_cat_name)
        if not _losing_with_gaps:
            _losing_with_gaps = _live_losing_cats if _live_losing_cats else ["None"]
        _losing_str = ", ".join(_losing_with_gaps)
        # Combined W-L-T + strategy on one line (remove redundant strategy row)
        _matchup_state_html = (
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
            f'<span style="font-size:11px;color:{T["tx2"]};">Matchup</span>'
            f'<span style="font-size:13px;font-weight:bold;font-family:IBM Plex Mono,monospace;">'
            f"{_live_state_label} ({matchup_state_label.upper()})</span></div>"
            f'<p style="margin:4px 0 0;font-size:11px;color:{T["tx2"]};">'
            f"Losing: {_losing_str}</p>"
        )
    else:
        _matchup_state_html = (
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
            f'<span style="font-size:11px;color:{T["tx2"]};">Strategy</span>'
            f'<span style="font-size:13px;font-weight:bold;font-family:IBM Plex Mono,monospace;">{matchup_state_label.upper()}</span></div>'
            f'<p style="margin:4px 0 0;font-size:11px;color:{T["tx2"]};">'
            f"{matchup_note}</p>"
        )
    render_context_card("Matchup State", _matchup_state_html)

    # IP budget
    if ip_budget_html:
        render_context_card("Innings Pitched Budget", ip_budget_html)

    # Opponent selector — use live matchup data when available
    opponent_names = [t for t in team_totals if t != user_team_name]
    selected_opponent = None
    opp_totals: dict[str, float] = {}

    # Apply pre-loaded live matchup data to my_totals and opp_totals
    if _pre_live_my:
        my_totals = _pre_live_my
        opp_totals = _pre_live_opp
        selected_opponent = _pre_live_opp_name

    # Opponent selector — only show when no live matchup (need manual selection)
    if not selected_opponent and opponent_names:
        selected_opponent = st.selectbox(
            "This Week's Opponent",
            options=["(None - Season-Long Only)"] + sorted(opponent_names),
            index=0,
            help="Select opponent to enable matchup-specific optimization.",
            key="lineup_opponent",
        )
        if selected_opponent and selected_opponent != "(None - Season-Long Only)":
            opp_totals = team_totals.get(selected_opponent, {})
        else:
            selected_opponent = None

    if selected_opponent and opp_totals and my_totals and H2H_AVAILABLE:
        try:
            _h2h_quick = estimate_h2h_win_probability(my_totals, opp_totals)
            _win_prob = _h2h_quick.get("overall_win_prob", 0.5)
            _exp_wins = _h2h_quick.get("expected_wins", 5.0)
            _status = "Favored" if _exp_wins > 5.5 else "Underdog" if _exp_wins < 4.5 else "Toss-up"
            render_context_card(
                "Win Probability",
                f'<div class="context-stat-row">'
                f'<span class="context-stat-label">Win Probability</span>'
                f'<span class="context-stat-value">{_win_prob:.1%}</span></div>'
                f'<div class="context-stat-row">'
                f'<span class="context-stat-label">Expected Wins</span>'
                f'<span class="context-stat-value">{_exp_wins:.1f} / 10</span></div>'
                f'<div class="context-stat-row">'
                f'<span class="context-stat-label">Outlook</span>'
                f'<span class="context-stat-value">{_status}</span></div>',
            )
        except Exception:
            pass

    # Unified Data Freshness panel (optimizer sources + Yahoo status)
    _freshness_ctx = st.session_state.get("optimizer_context")
    _opt_timestamps = (
        _freshness_ctx.data_timestamps
        if _freshness_ctx and hasattr(_freshness_ctx, "data_timestamps") and _freshness_ctx.data_timestamps
        else {}
    )
    # Show Yahoo connection status + optimizer data sources
    try:
        from src.yahoo_data_service import get_yahoo_data_service as _get_yds_fresh

        _yds_fresh = _get_yds_fresh()
        _yahoo_connected = _yds_fresh.is_connected()
    except Exception:
        _yahoo_connected = False

    _status_colors = {"fresh": "#4CAF50", "stale": "#FF9800", "unknown": "#9E9E9E"}

    _freshness_html = (
        f'<div class="context-stat-row" style="margin-bottom:6px;">'
        f'<span class="context-stat-label">Yahoo</span>'
        f'<span class="context-stat-value" style="color:{"#4CAF50" if _yahoo_connected else "#9E9E9E"}">'
        f"{'Connected' if _yahoo_connected else 'Offline'}</span></div>"
    )

    if _opt_timestamps:
        for _src_name, _src_info in _opt_timestamps.items():
            _f_status = _src_info.get("status", "unknown")
            _f_color = _status_colors.get(_f_status, "#9E9E9E")
            _f_label = _src_name.replace("_", " ").title()
            _f_source = _src_info.get("source_label", "")
            _f_as_of = _src_info.get("data_as_of", "")
            _f_ts = _src_info.get("timestamp", "")
            # Format timestamp to readable local time
            _f_time_str = ""
            if _f_ts:
                try:
                    from datetime import datetime as _dt_fresh

                    _parsed = _dt_fresh.fromisoformat(_f_ts)
                    _local = _parsed.astimezone()
                    _f_time_str = _local.strftime("%I:%M %p").lstrip("0")
                except Exception:
                    _f_time_str = _src_info.get("age", "")
            _detail = _f_as_of if _f_as_of else ""
            if _f_source:
                _detail = f"{_f_source}: {_detail}" if _detail else _f_source
            _freshness_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'padding:2px 0;border-bottom:1px solid {T["border"]}33;">'
                f'<span style="font-size:11px;color:{T["tx2"]};max-width:55%;">{_f_label}</span>'
                f'<span style="font-size:11px;font-family:IBM Plex Mono,monospace;color:{_f_color};">'
                f"{_f_time_str}</span></div>"
            )
            if _detail:
                _freshness_html += (
                    f'<div style="font-size:9px;color:{T["tx2"]};opacity:0.7;padding:0 0 3px;'
                    f'margin-top:-1px;">{_detail}</div>'
                )
    else:
        # Before optimizer runs: show Yahoo data freshness with timestamps from refresh_log
        try:
            from src.database import get_connection as _get_conn_fresh

            _conn_f = _get_conn_fresh()
            try:
                _refresh_rows = _conn_f.execute(
                    "SELECT source, last_refresh, status FROM refresh_log ORDER BY last_refresh DESC"
                ).fetchall()
            finally:
                _conn_f.close()

            _source_labels = {
                "players": ("Players", "MLB Stats API", "40-man rosters"),
                "projections": ("Projections", "Blended (Steamer/ZiPS/DC)", "ROS projections"),
                "live_stats": ("Live Stats", "MLB Stats API", "2026 season stats"),
                "depth_charts": ("Depth Charts", "FanGraphs", "Playing time estimates"),
                "yahoo": ("Yahoo Sync", "Yahoo Fantasy API", "Rosters + standings"),
                "news": ("News", "ESPN + RotoWire + MLB", "Player news/injuries"),
                "adp": ("ADP/ECR", "FanGraphs + FantasyPros", "Rankings consensus"),
                "ecr": ("ECR", "Multi-source", "Expert rankings"),
                "prospects": ("Prospects", "FanGraphs + MiLB", "Prospect rankings"),
                "park_factors": ("Park Factors", "pybaseball", "Ballpark adjustments"),
                "deduplication": ("Deduplication", "Local DB", "Player ID merge"),
            }
            for _src, _ts, _status in _refresh_rows:
                _label_info = _source_labels.get(_src, (_src.replace("_", " ").title(), "", ""))
                _f_label, _f_source, _f_as_of = _label_info
                _f_time_str = ""
                if _ts:
                    try:
                        from datetime import datetime as _dt_ref

                        _parsed_ref = _dt_ref.fromisoformat(str(_ts))
                        _local_ref = _parsed_ref.astimezone()
                        _f_time_str = _local_ref.strftime("%I:%M %p").lstrip("0")
                    except Exception:
                        _f_time_str = str(_ts)[:16]
                _f_color = "#4CAF50" if _status == "success" else "#FF9800"
                _freshness_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'padding:2px 0;border-bottom:1px solid {T["border"]}33;">'
                    f'<span style="font-size:11px;color:{T["tx2"]};max-width:55%;">{_f_label}</span>'
                    f'<span style="font-size:11px;font-family:IBM Plex Mono,monospace;color:{_f_color};">'
                    f"{_f_time_str}</span></div>"
                )
                _detail = f"{_f_source}: {_f_as_of}" if _f_source and _f_as_of else _f_source or _f_as_of
                if _detail:
                    _freshness_html += (
                        f'<div style="font-size:9px;color:{T["tx2"]};opacity:0.7;padding:0 0 3px;'
                        f'margin-top:-1px;">{_detail}</div>'
                    )
        except Exception:
            pass

    render_context_card("Data Freshness", _freshness_html)


# ── Main content panel ───────────────────────────────────────────────

with main:
    # S2: H2H tab removed — duplicate of Matchup Planner Category Probabilities.
    (
        tab_optimize,
        tab_startsit,
        tab_analysis,
        tab_streaming,
    ) = st.tabs(
        [
            "Optimizer",
            "Start/Sit",
            "Category Analysis",
            "Streaming",
        ]
    )

    # ================================================================
    # TAB 1: OPTIMIZE
    # ================================================================

    with tab_optimize:
        # ── Scope selector ────────────────────────────────────────────
        _opt_scope = st.radio(
            "Optimization Scope",
            ["Today", "Rest of Week", "Rest of Season"],
            horizontal=True,
            key="optimizer_scope",
            help=(
                "**Today:** Daily Category Value (DCV) engine — who to start today based on "
                "confirmed lineups, park factors, and matchup urgency. "
                "**Rest of Week:** LP solver with remaining-games scaling and two-start premiums. "
                "**Rest of Season:** LP solver with full projections and schedule strength."
            ),
        )
        _scope_key = {"Today": "today", "Rest of Week": "rest_of_week", "Rest of Season": "rest_of_season"}[_opt_scope]

        optimize_clicked = st.button("Optimize Lineup", type="primary", key="lineup_run_opt")

        if optimize_clicked:
            # Auto-compute alpha per scope (replaces manual slider)
            if _scope_key == "rest_of_week":
                alpha = 1.0  # Pure H2H for weekly matchup
            elif _scope_key == "rest_of_season":
                alpha = max(0.0, 1.0 - (weeks_remaining / 24.0))
            # "today" scope: alpha unused by DCV engine

            progress_bar = st.progress(0, text="Syncing roster and building shared data context...")

            # Force roster refresh to ensure current players only
            try:
                yds.get_rosters(force_refresh=True)
                roster = get_team_roster(user_team_name)
                if "name" in roster.columns and "player_name" not in roster.columns:
                    roster = roster.rename(columns={"name": "player_name"})
                for _sc in ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip", "ip", "pa"]:
                    if _sc in roster.columns:
                        roster[_sc] = pd.to_numeric(roster[_sc], errors="coerce").fillna(0)
            except Exception:
                pass

            # ── Build shared optimizer context (single source of truth) ──
            _opt_ctx = None
            try:
                from src.optimizer.shared_data_layer import build_optimizer_context

                progress_bar.progress(10, text="Loading matchup data, schedule, and player intelligence...")
                _opt_ctx = build_optimizer_context(
                    scope=_scope_key,
                    yds=yds,
                    config=league_config,
                    weeks_remaining=weeks_remaining,
                    park_factors=PARK_FACTORS if PARK_FACTORS else None,
                    user_team_name=user_team_name,
                    roster=roster,
                )
                st.session_state["optimizer_context"] = _opt_ctx
            except Exception as _ctx_err:
                import logging as _log

                _log.getLogger(__name__).warning("Failed to build optimizer context: %s", _ctx_err)

            # Fetch today's MLB schedule for game-today annotations
            _today_teams_playing: set[str] = set()
            try:
                from datetime import datetime, timedelta, timezone

                import statsapi

                _FULL_TO_ABBR: dict[str, str] = {
                    "Arizona Diamondbacks": "ARI",
                    "Atlanta Braves": "ATL",
                    "Baltimore Orioles": "BAL",
                    "Boston Red Sox": "BOS",
                    "Chicago Cubs": "CHC",
                    "Chicago White Sox": "CWS",
                    "Cincinnati Reds": "CIN",
                    "Cleveland Guardians": "CLE",
                    "Colorado Rockies": "COL",
                    "Detroit Tigers": "DET",
                    "Houston Astros": "HOU",
                    "Kansas City Royals": "KC",
                    "Los Angeles Angels": "LAA",
                    "Los Angeles Dodgers": "LAD",
                    "Miami Marlins": "MIA",
                    "Milwaukee Brewers": "MIL",
                    "Minnesota Twins": "MIN",
                    "New York Mets": "NYM",
                    "New York Yankees": "NYY",
                    "Athletics": "ATH",
                    "Oakland Athletics": "OAK",
                    "Philadelphia Phillies": "PHI",
                    "Pittsburgh Pirates": "PIT",
                    "San Diego Padres": "SD",
                    "San Francisco Giants": "SF",
                    "Seattle Mariners": "SEA",
                    "St. Louis Cardinals": "STL",
                    "Tampa Bay Rays": "TB",
                    "Texas Rangers": "TEX",
                    "Toronto Blue Jays": "TOR",
                    "Washington Nationals": "WSH",
                }
                # Target date: today or tomorrow if all games final
                from src.game_day import get_target_game_date

                _opt_target_date = get_target_game_date()
                st.session_state["_optimizer_target_date"] = _opt_target_date
                _today_sched = statsapi.schedule(date=_opt_target_date)

                for _g in _today_sched:
                    for _side in ("home_name", "away_name"):
                        _raw = str(_g.get(_side, ""))
                        _abbr = _FULL_TO_ABBR.get(_raw, "")
                        if _abbr:
                            _today_teams_playing.add(_abbr)
                        if _raw:
                            _today_teams_playing.add(_raw)
            except Exception:
                pass

            # ── Route to appropriate engine based on scope ────────────
            if _scope_key == "today":
                # ── TODAY: DCV engine ─────────────────────────────────
                progress_bar.progress(30, text="Computing Daily Category Values...")
                try:
                    from src.optimizer.daily_optimizer import build_daily_dcv_table

                    _dcv_sched = None
                    try:
                        import statsapi

                        # Use the same target date (auto-advanced if today's games are final)
                        _dcv_target = st.session_state.get("_optimizer_target_date", "")
                        if not _dcv_target:
                            from datetime import datetime, timedelta, timezone

                            _ET2 = timezone(timedelta(hours=-4))
                            _dcv_target = datetime.now(_ET2).strftime("%Y-%m-%d")
                        _dcv_sched = statsapi.schedule(date=_dcv_target)
                    except Exception:
                        _dcv_sched = []

                    # Pass shared context enrichments into DCV
                    _dcv_urgency = _opt_ctx.urgency_weights if _opt_ctx else None
                    _dcv_lineups = _opt_ctx.confirmed_lineups if _opt_ctx else None
                    _dcv_form = _opt_ctx.recent_form if _opt_ctx else None
                    _dcv_rate_modes = _dcv_urgency.get("rate_modes") if _dcv_urgency else None

                    dcv = build_daily_dcv_table(
                        roster=roster if not roster.empty else pool,
                        matchup=yds.get_matchup(),
                        schedule_today=_dcv_sched if _dcv_sched else None,
                        park_factors=PARK_FACTORS if PARK_FACTORS else {},
                        urgency_weights=_dcv_urgency,
                        confirmed_lineups=_dcv_lineups,
                        recent_form=_dcv_form,
                        rate_modes=_dcv_rate_modes,
                    )

                    progress_bar.progress(100, text="Daily optimization complete!")
                    time.sleep(0.3)
                    progress_bar.empty()

                    # Store as result in compatible format for display
                    st.session_state["lineup_optimizer_result"] = {
                        "dcv_table": dcv,
                        "scope": "today",
                        "lineup": None,
                        "category_weights": _opt_ctx.category_weights if _opt_ctx else {},
                        "h2h_analysis": None,
                        "streaming_suggestions": None,
                        "risk_metrics": None,
                        "maximin_comparison": None,
                        "recommendations": [],
                        "timing": {"total": 0},
                        "mode": "dcv",
                        "matchup_adjusted": True,
                    }
                    st.session_state["lineup_today_teams"] = _today_teams_playing

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Daily optimization failed: {e}")

            else:
                # ── REST OF WEEK / REST OF SEASON: LP pipeline ────────
                # Use shared context weights instead of monkey-patching
                _urgency_weights: dict[str, float] = {}
                if _opt_ctx and _opt_ctx.urgency_weights:
                    # Extract inner urgency dict — compute_urgency_weights() returns
                    # {"urgency": {cat: float}, "rate_modes": {...}, "summary": {...}}
                    _urgency_weights = (
                        _opt_ctx.urgency_weights.get("urgency", {})
                        if isinstance(_opt_ctx.urgency_weights, dict)
                        else {}
                    )

                # IP-aware pitching weight adjustment
                _ip_boost = 1.0
                try:
                    if IP_TRACKER_AVAILABLE:
                        from src.ip_tracker import compute_weekly_ip_projection, get_days_remaining_in_week

                        _pitchers = []
                        for _, _pr in (
                            roster[roster.get("is_hitter", 1) == 0].iterrows() if "is_hitter" in roster.columns else []
                        ):
                            _pitchers.append(
                                {
                                    "ip": _proj_ip_lookup.get(_pr.get("player_id"), float(_pr.get("ip", 0) or 0)),
                                    "positions": str(_pr.get("positions", "")),
                                    "status": str(_pr.get("status", "active")),
                                    "is_starter": "SP" in str(_pr.get("positions", "")).upper(),
                                }
                            )
                        if _pitchers:
                            _ip_result = compute_weekly_ip_projection(_pitchers, get_days_remaining_in_week())
                            if _ip_result.get("status") == "danger":
                                _ip_boost = 1.5
                            elif _ip_result.get("status") == "safe":
                                _ip_boost = 0.7
                except Exception:
                    pass

                if PIPELINE_AVAILABLE:
                    progress_bar.progress(20, text=f"Running {mode} optimization pipeline ({_opt_scope})...")

                    pipeline = LineupOptimizerPipeline(
                        roster,
                        mode=mode,
                        alpha=alpha,
                        weeks_remaining=weeks_remaining,
                        config=league_config,
                    )

                    pipeline._preset = dict(pipeline._preset)
                    pipeline._preset["risk_aversion"] = risk_aversion

                    # Wire urgency weights + IP boost via monkey-patch
                    try:
                        if _urgency_weights or _ip_boost != 1.0:
                            _pitching_cats = {"w", "l", "sv", "k", "era", "whip"}
                            _urgency_mults: dict[str, float] = {}
                            for _ucat, _urg in _urgency_weights.items():
                                _ucat_lower = _ucat.lower()
                                if _urg > 0.6:
                                    _mult = 1.5
                                elif _urg > 0.4:
                                    _mult = 1.2
                                elif _urg > 0.25:
                                    _mult = 1.0
                                else:
                                    _mult = 0.6
                                if _ucat_lower in _pitching_cats:
                                    _mult *= _ip_boost
                                _urgency_mults[_ucat_lower] = _mult

                            if _urgency_mults:
                                _original_compute = pipeline._compute_category_weights

                                def _urgency_wrapped_compute(*args, **kwargs):
                                    base_weights = _original_compute(*args, **kwargs)
                                    adjusted = {}
                                    for cat, base_w in base_weights.items():
                                        cat_key = cat.lower()
                                        if cat_key in _urgency_mults:
                                            adjusted[cat] = base_w * _urgency_mults[cat_key]
                                        else:
                                            adjusted[cat] = base_w
                                    return adjusted

                                pipeline._compute_category_weights = _urgency_wrapped_compute
                    except Exception:
                        pass

                    progress_bar.progress(40, text="Computing category weights and projections...")

                    _fa_for_streaming = pd.DataFrame()
                    if mode == "full":
                        try:
                            _fa_for_streaming = yds.get_free_agents(max_players=500)
                        except Exception:
                            pass

                    result = pipeline.optimize(
                        standings=standings if not standings.empty else None,
                        team_name=user_team_name,
                        h2h_opponent_totals=opp_totals if opp_totals else None,
                        my_totals=my_totals if my_totals else None,
                        week_schedule=week_schedule if week_schedule else None,
                        park_factors=PARK_FACTORS if PARK_FACTORS else None,
                        free_agents=_fa_for_streaming if not _fa_for_streaming.empty else None,
                    )

                    progress_bar.progress(100, text="Optimization complete!")
                    time.sleep(0.3)
                    progress_bar.empty()

                    result["scope"] = _scope_key
                    st.session_state["lineup_optimizer_result"] = result
                    st.session_state["lineup_today_teams"] = _today_teams_playing

                else:
                    # Fallback to basic optimizer
                    progress_bar.progress(20, text="Building constraint matrix...")
                    optimizer = LineupOptimizer(roster, league_config)
                    basic_result = optimizer.optimize_lineup(
                        category_weights=_opt_ctx.category_weights if _opt_ctx else None,
                    )
                    progress_bar.progress(100, text="Optimization complete!")
                    time.sleep(0.3)
                    progress_bar.empty()

                    result = {
                        "lineup": basic_result,
                        "scope": _scope_key,
                        "category_weights": {},
                        "h2h_analysis": None,
                        "streaming_suggestions": None,
                        "risk_metrics": None,
                        "maximin_comparison": None,
                        "recommendations": [],
                        "timing": {"total": 0},
                        "mode": "basic",
                        "matchup_adjusted": False,
                    }
                    st.session_state["lineup_optimizer_result"] = result

        # Display results from session state
        result = st.session_state.get("lineup_optimizer_result")

        if result and result.get("scope") == "today" and result.get("dcv_table") is not None:
            # ── DCV "Today" results display ───────────────────────────
            dcv = result["dcv_table"]
            if dcv.empty:
                st.info("Could not compute Daily Category Values. Check that roster data is loaded.")
            else:
                _target_date_display = st.session_state.get("_optimizer_target_date", "")
                if _target_date_display:
                    try:
                        from datetime import datetime as _dt_disp

                        _td_parsed = _dt_disp.strptime(_target_date_display, "%Y-%m-%d")
                        _target_label = _td_parsed.strftime("%A %m/%d")
                    except Exception:
                        _target_label = _target_date_display
                else:
                    _target_label = "Today"
                st.success(f"Lineup optimized for {_target_label} (DCV engine, matchup-adjusted)")

                # Show W-L-T summary from actual Yahoo matchup results
                _ctx_disp = st.session_state.get("optimizer_context")
                _urg_summary = (
                    _ctx_disp.urgency_weights.get("summary", {})
                    if _ctx_disp and isinstance(_ctx_disp.urgency_weights, dict)
                    else {}
                )
                _winning = _urg_summary.get("winning", [])
                _losing = _urg_summary.get("losing", [])
                _tied = _urg_summary.get("tied", [])
                if _winning or _losing or _tied:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Winning", len(_winning))
                        if _winning:
                            st.caption(", ".join(_winning))
                    with col2:
                        st.metric("Losing", len(_losing))
                        if _losing:
                            st.caption(", ".join(_losing))
                    with col3:
                        st.metric("Tied", len(_tied))
                        if _tied:
                            st.caption(", ".join(_tied))

                # ── Merge Yahoo slot assignment + eligible positions from roster ──
                if "selected_position" not in dcv.columns and "player_id" in dcv.columns:
                    _merge_cols = ["player_id", "selected_position"]
                    if "roster_slot" in roster.columns:
                        _merge_cols.append("roster_slot")
                    _sp_lookup = roster[_merge_cols].drop_duplicates("player_id")
                    dcv = dcv.merge(_sp_lookup, on="player_id", how="left")
                    dcv["selected_position"] = dcv["selected_position"].fillna("").astype(str).str.strip()
                    dcv.loc[dcv["selected_position"].isin(["", "None", "none"]), "selected_position"] = "BN"
                    if "roster_slot" in dcv.columns:
                        dcv["roster_slot"] = dcv["roster_slot"].fillna("").astype(str).str.strip()

                # ── Greedy slot assignment for START/BENCH decisions ──
                eligible = (
                    dcv[
                        (dcv.get("volume_factor", pd.Series(dtype=float)) > 0)
                        & (dcv.get("health_factor", pd.Series(dtype=float)) > 0)
                    ].copy()
                    if "volume_factor" in dcv.columns and "health_factor" in dcv.columns
                    else dcv.copy()
                )

                _SLOT_ORDER = [
                    ("C", 1),
                    ("1B", 1),
                    ("2B", 1),
                    ("3B", 1),
                    ("SS", 1),
                    ("OF", 3),
                    ("SP", 2),
                    ("RP", 2),
                    ("P", 4),
                    ("Util", 2),
                ]
                _OF_POSITIONS = {"LF", "CF", "RF", "OF"}
                _PITCHER_POSITIONS = {"SP", "RP", "P"}

                if "total_dcv" in eligible.columns:
                    _sorted = eligible.sort_values("total_dcv", ascending=False)
                else:
                    _sorted = eligible

                _assigned_idx: set[int] = set()
                _starter_indices: list[int] = []

                def _player_fits_slot(row, slot: str) -> bool:
                    pos_str = str(row.get("positions", "") or "").upper()
                    pos_set = {p.strip() for p in pos_str.replace("/", ",").split(",") if p.strip()}
                    if slot == "C":
                        return "C" in pos_set
                    if slot == "OF":
                        return bool(pos_set & _OF_POSITIONS)
                    if slot == "SP":
                        return "SP" in pos_set
                    if slot == "RP":
                        return "RP" in pos_set
                    if slot == "P":
                        return bool(pos_set & _PITCHER_POSITIONS)
                    if slot == "Util":
                        return bool(pos_set - _PITCHER_POSITIONS)
                    return slot in pos_set

                for slot, count in _SLOT_ORDER:
                    filled = 0
                    for idx, row in _sorted.iterrows():
                        if filled >= count:
                            break
                        if idx in _assigned_idx:
                            continue
                        if _player_fits_slot(row, slot):
                            _assigned_idx.add(idx)
                            _starter_indices.append(idx)
                            filled += 1

                # ── Decision column on full DCV table ──
                dcv["Decision"] = "BENCH"
                dcv.loc[dcv.index.isin(set(_starter_indices)), "Decision"] = "START"
                _il_slots = {"IL", "IL+", "NA"}
                if "selected_position" in dcv.columns:
                    dcv.loc[dcv["selected_position"].isin(_il_slots), "Decision"] = "IL"
                if "health_factor" in dcv.columns:
                    dcv.loc[
                        (dcv["health_factor"] == 0) & (~dcv["Decision"].isin(["IL"])),
                        "Decision",
                    ] = "IL"

                # ── Post-process: bench SPs when rate stats are abandoned ──
                # When both ERA and WHIP are unrecoverable, pure SPs with
                # negative DCV only add IP damage. Bench them; keep RPs for saves.
                if _dcv_rate_modes:
                    _era_abn = _dcv_rate_modes.get("ERA") == "abandon"
                    _whip_abn = _dcv_rate_modes.get("WHIP") == "abandon"
                    if _era_abn and _whip_abn:
                        for idx, row in dcv.iterrows():
                            if (
                                row.get("Decision") == "START"
                                and not bool(row.get("is_hitter", 1))
                                and "SP" in str(row.get("positions", "")).upper()
                                and "RP" not in str(row.get("positions", "")).upper()
                                and float(row.get("total_dcv", 0)) < 0
                            ):
                                dcv.at[idx, "Decision"] = "BENCH"

                # ── Split batters / pitchers ──
                _is_hitter_col = dcv.get("is_hitter", pd.Series(dtype=float))
                batters_dcv = dcv[_is_hitter_col == 1].copy()
                pitchers_dcv = dcv[_is_hitter_col == 0].copy()

                # ── Sort by Yahoo slot order, then DCV within same slot ──
                _BATTER_SLOT_ORDER = {
                    "C": 0,
                    "1B": 1,
                    "2B": 2,
                    "3B": 3,
                    "SS": 4,
                    "OF": 5,
                    "LF": 5,
                    "CF": 5,
                    "RF": 5,
                    "Util": 6,
                    "DH": 6,
                    "BN": 7,
                    "IL": 8,
                    "IL+": 8,
                    "NA": 9,
                }
                _PITCHER_SLOT_ORDER = {
                    "SP": 0,
                    "RP": 1,
                    "P": 2,
                    "BN": 3,
                    "IL": 4,
                    "IL+": 4,
                    "NA": 5,
                }

                def _slot_sort_key(df_in, order_map):
                    default = max(order_map.values()) + 1
                    return df_in["selected_position"].map(lambda s: order_map.get(s, default))

                # Sort: START on top, then BENCH, then IL — within each group by slot order
                _DECISION_ORDER = {"START": 0, "BENCH": 1, "IL": 2}

                if not batters_dcv.empty and "selected_position" in batters_dcv.columns:
                    batters_dcv["_slot_sort"] = _slot_sort_key(batters_dcv, _BATTER_SLOT_ORDER)
                    batters_dcv["_dec_sort"] = batters_dcv["Decision"].map(_DECISION_ORDER).fillna(2)
                    batters_dcv = batters_dcv.sort_values(
                        ["_dec_sort", "_slot_sort", "total_dcv"], ascending=[True, True, False]
                    ).drop(columns=["_slot_sort", "_dec_sort"])
                if not pitchers_dcv.empty and "selected_position" in pitchers_dcv.columns:
                    pitchers_dcv["_slot_sort"] = _slot_sort_key(pitchers_dcv, _PITCHER_SLOT_ORDER)
                    pitchers_dcv["_dec_sort"] = pitchers_dcv["Decision"].map(_DECISION_ORDER).fillna(2)
                    pitchers_dcv = pitchers_dcv.sort_values(
                        ["_dec_sort", "_slot_sort", "total_dcv"], ascending=[True, True, False]
                    ).drop(columns=["_slot_sort", "_dec_sort"])

                # ── Build display DataFrames ──
                # Use roster_slot (Yahoo eligible positions) if available, else positions
                _pos_col = "roster_slot" if "roster_slot" in dcv.columns else "positions"
                _display_cols = [
                    "selected_position",
                    "name",
                    _pos_col,
                    "team",
                    "total_dcv",
                    "matchup_mult",
                    "Decision",
                ]
                _col_rename = {
                    "selected_position": "Slot",
                    "name": "Player",
                    _pos_col: "Position Eligibility",
                    "team": "Team",
                    "total_dcv": "DCV Score",
                    "matchup_mult": "Matchup",
                }

                def _build_dcv_display(df_in):
                    disp = df_in[[c for c in _display_cols if c in df_in.columns]].copy()
                    disp = disp.rename(columns=_col_rename).reset_index(drop=True)
                    return disp

                batters_display = _build_dcv_display(batters_dcv)
                pitchers_display = _build_dcv_display(pitchers_dcv)

                # ── Row styling by decision ──
                def _decision_row_classes(df_in):
                    classes = {}
                    for i in range(len(df_in)):
                        decision = df_in.iloc[i].get("Decision", "BENCH")
                        if decision == "START":
                            classes[i] = "row-start"
                        elif decision == "IL":
                            classes[i] = "row-il"
                        else:
                            classes[i] = "row-bench"
                    return classes

                st.markdown(
                    f"<style>"
                    f"tr.row-start td {{ background-color:{T['green']}12 !important; }}"
                    f"tr.row-start td.col-name {{ color:{T['green']} !important; font-weight:700 !important; }}"
                    f"tr.row-bench td {{ background-color: rgba(230,57,70,0.08) !important; }}"
                    f"tr.row-bench td.col-name {{ color:{T['danger']} !important; }}"
                    f"tr.row-il td {{ background-color: rgba(158,158,158,0.06) !important; opacity:0.5; }}"
                    f"</style>",
                    unsafe_allow_html=True,
                )

                # ── Batters table ──
                st.markdown(
                    f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                    f"color:{T['tx2']};text-transform:uppercase;"
                    f'margin:0 0 6px;">Batters</p>',
                    unsafe_allow_html=True,
                )
                if not batters_display.empty:
                    render_compact_table(batters_display, row_classes=_decision_row_classes(batters_display))

                # ── Pitchers table ──
                st.markdown(
                    f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                    f"color:{T['tx2']};text-transform:uppercase;"
                    f'margin:16px 0 6px;">Pitchers</p>',
                    unsafe_allow_html=True,
                )
                if not pitchers_display.empty:
                    render_compact_table(pitchers_display, row_classes=_decision_row_classes(pitchers_display))

                st.caption(
                    "DCV = Daily Category Value. Computed from Bayesian-blended projections "
                    "* matchup factors * H2H category urgency * confirmed lineups. "
                    "Higher = more valuable today."
                )

                # FA recommendations for Today scope
                _ctx_fa = st.session_state.get("optimizer_context")
                if _ctx_fa:
                    try:
                        from src.optimizer.fa_recommender import recommend_fa_moves

                        _fa_moves = recommend_fa_moves(_ctx_fa, max_moves=3)
                        if _fa_moves:
                            st.divider()
                            st.markdown(
                                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                                f"color:{T['tx2']};text-transform:uppercase;"
                                f'margin:0 0 6px;">Recommended Free Agent Moves</p>',
                                unsafe_allow_html=True,
                            )
                            for _mv in _fa_moves:
                                _add = _mv.get("add_name", "?")
                                _drop = _mv.get("drop_name", "?")
                                _delta = _mv.get("net_sgp_delta", 0)
                                _cats = _mv.get("urgency_categories", [])
                                _warn = _mv.get("news_warning")
                                _reasons = _mv.get("reasoning", [])
                                _color = T["green"] if _delta > 0 else T["danger"]
                                st.markdown(
                                    f'<div style="border-left:3px solid {_color};padding:8px 12px;margin:6px 0;'
                                    f'background:{T["bg"]};border-radius:4px;">'
                                    f"<b>Add</b> {_add} / <b>Drop</b> {_drop} "
                                    f'<span style="color:{_color};font-weight:700;">'
                                    f"({_delta:+.2f} Standings Gained Points)</span>"
                                    f"{'<br>Helps: ' + ', '.join(_cats) if _cats else ''}"
                                    f"{'<br><span style=' + chr(34) + 'color:' + T['danger'] + chr(34) + '>' + _warn + '</span>' if _warn else ''}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                if _reasons:
                                    st.caption(" | ".join(_reasons))
                    except Exception:
                        pass

        elif result:
            lineup = result.get("lineup", {})

            if not lineup or not lineup.get("assignments"):
                st.warning(
                    "Could not produce a valid lineup. This usually means player projection "
                    "data is missing or all zeros. Try syncing your Yahoo league data or "
                    "running the data bootstrap to refresh projections."
                )
            else:
                # Success banner
                mode_label = result.get("mode", "standard").title()
                timing_total = result.get("timing", {}).get("total", 0)
                matchup_flag = " | Matchup-adjusted" if result.get("matchup_adjusted") else ""
                st.success(f"Optimal lineup found ({mode_label} mode, {timing_total:.2f}s{matchup_flag})")

                # Analytics transparency badge
                if "analytics_context" in result:
                    try:
                        from src.ui_analytics_badge import render_analytics_badge

                        render_analytics_badge(result["analytics_context"])
                    except Exception:
                        pass

                # Recommended lineup table — sorted by Yahoo slot order
                st.markdown(
                    f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                    f"color:{T['tx2']};text-transform:uppercase;"
                    f'margin:0 0 6px;">Starting Lineup</p>',
                    unsafe_allow_html=True,
                )
                assignments = lineup["assignments"]
                _pid_to_mlb_lineup = {}
                if "mlb_id" in roster.columns and "player_id" in roster.columns:
                    _pid_to_mlb_lineup = dict(zip(roster["player_id"], roster["mlb_id"]))

                # Build team lookup for game-today annotation
                _pid_to_team: dict[int, str] = {}
                if "player_id" in roster.columns and "team" in roster.columns:
                    _pid_to_team = dict(zip(roster["player_id"], roster["team"].astype(str)))
                _saved_today_teams: set[str] = st.session_state.get("lineup_today_teams", set())

                # Build the full expected slot template from ROSTER_SLOTS
                # so all 18 starter slots appear even if unfilled.
                from src.lineup_optimizer import ROSTER_SLOTS as _LP_SLOTS
                from src.ui_shared import SLOT_ORDER_HITTERS, SLOT_ORDER_PITCHERS

                _ALL_SLOTS_ORDERED = SLOT_ORDER_HITTERS + SLOT_ORDER_PITCHERS

                # Expand slots: OF(3) → OF, OF, OF; P(4) → P, P, P, P; etc.
                _expected_slots: list[str] = []
                for _es in _ALL_SLOTS_ORDERED:
                    _count = _LP_SLOTS.get(_es, (0, []))[0]
                    if _count > 0:
                        _expected_slots.extend([_es] * _count)

                # Sort assignments by slot order, then greedily match to expected slots
                _slot_order_map = {s: i for i, s in enumerate(_ALL_SLOTS_ORDERED)}

                def _slot_sort_key(entry):
                    return _slot_order_map.get(entry.get("slot", ""), 99)

                assignments_sorted = sorted(assignments, key=_slot_sort_key)

                # Match assignments to expected slot positions
                # Track how many of each slot type have been filled
                _slot_fill_counts: dict[str, int] = {}
                _assigned_to_expected: list[dict | None] = [None] * len(_expected_slots)
                for entry in assignments_sorted:
                    slot = entry.get("slot", "")
                    _slot_fill_counts.setdefault(slot, 0)
                    # Find the next unfilled expected slot of this type
                    for ei, es in enumerate(_expected_slots):
                        if es == slot and _assigned_to_expected[ei] is None:
                            _assigned_to_expected[ei] = entry
                            _slot_fill_counts[slot] += 1
                            break

                # Helper: check if a player's team plays today
                def _has_game_today(pid: int | None) -> str:
                    if not _saved_today_teams or pid is None:
                        return ""
                    team = _pid_to_team.get(pid, "")
                    if not team or team in ("", "MLB", "None"):
                        return ""
                    # Direct lookup: _saved_today_teams has both abbreviations and full names
                    if team in _saved_today_teams or team.upper() in {t.upper() for t in _saved_today_teams}:
                        return "Yes"
                    return "No game"

                lineup_data = []
                for ei, es in enumerate(_expected_slots):
                    entry = _assigned_to_expected[ei]
                    if entry:
                        player = entry.get("player_name", "")
                        pid = entry.get("player_id")
                        hs = health_dict.get(pid, 1.0) if pid else 1.0
                        badge, label = get_injury_badge(hs)
                        lineup_data.append(
                            {
                                "Slot": es,
                                "Player": player,
                                "Status": "START",
                                "Health": f"{badge} {label}",
                                "Today": _has_game_today(pid),
                                "mlb_id": _pid_to_mlb_lineup.get(pid),
                            }
                        )
                    else:
                        lineup_data.append(
                            {
                                "Slot": es,
                                "Player": "(empty)",
                                "Status": "EMPTY",
                                "Health": "",
                                "Today": "",
                                "mlb_id": None,
                            }
                        )

                # Inject row styling: green for START, amber for EMPTY
                _start_row_classes: dict[int, str] = {}
                for _ri, _rd in enumerate(lineup_data):
                    if _rd["Status"] == "START":
                        _start_row_classes[_ri] = "row-start"
                    elif _rd["Status"] == "EMPTY":
                        _start_row_classes[_ri] = "row-empty"
                st.markdown(
                    f"<style>"
                    f"tr.row-start td {{ background-color:{T['green']}15 !important; }}"
                    f"tr.row-bench td {{ background-color:{T['danger']}10 !important; }}"
                    f"tr.row-empty td {{ background-color:{T['amber']}15 !important; color:{T['tx2']} !important; font-style:italic !important; }}"
                    f"</style>",
                    unsafe_allow_html=True,
                )
                render_compact_table(
                    pd.DataFrame(lineup_data),
                    show_avatars=True,
                    health_col="Health",
                    row_classes=_start_row_classes,
                )

                # Projected stats
                if lineup.get("projected_stats"):
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                        f"color:{T['tx2']};text-transform:uppercase;"
                        f'margin:16px 0 6px;">Projected Weekly Category Totals</p>',
                        unsafe_allow_html=True,
                    )
                    proj = lineup["projected_stats"]
                    # Scale counting stats to weekly (24-week fantasy season).
                    # Rate stats (AVG, OBP, ERA, WHIP) don't scale.
                    WEEKS_IN_SEASON = 24.0
                    _rate_stats = {"avg", "obp", "era", "whip"}
                    weekly_proj: dict[str, float] = {}
                    for cat, val in proj.items():
                        if cat.lower() in _rate_stats:
                            weekly_proj[cat] = val
                        else:
                            weekly_proj[cat] = val / WEEKS_IN_SEASON

                    display_proj = {}
                    for cat in ALL_CATS:
                        if cat in weekly_proj:
                            if cat in ("avg", "obp"):
                                display_proj[CAT_DISPLAY_NAMES[cat]] = format_stat(weekly_proj[cat], cat.upper())
                            elif cat in ("era", "whip"):
                                display_proj[CAT_DISPLAY_NAMES[cat]] = format_stat(weekly_proj[cat], cat.upper())
                            else:
                                display_proj[CAT_DISPLAY_NAMES[cat]] = f"{weekly_proj[cat]:.1f}"
                    render_styled_table(pd.DataFrame([display_proj]))

                # Risk metrics
                risk_metrics = result.get("risk_metrics")
                if risk_metrics:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                        f"color:{T['tx2']};text-transform:uppercase;"
                        f'margin:16px 0 6px;">Risk Analysis</p>',
                        unsafe_allow_html=True,
                    )
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    with rc1:
                        st.metric("Expected Value", f"{risk_metrics.get('mean', 0):.2f}")
                    with rc2:
                        st.metric("Standard Deviation", f"{risk_metrics.get('std', 0):.2f}")
                    with rc3:
                        st.metric("Value at Risk (5th)", f"{risk_metrics.get('var_5', 0):.2f}")
                    with rc4:
                        st.metric("CVaR (Tail Risk)", f"{risk_metrics.get('cvar_5', 0):.2f}")
                    st.caption(
                        "Value at Risk is the 5th percentile outcome. CVaR is the average of the worst 5% of scenarios."
                    )

                # Recommendations
                recs = result.get("recommendations", [])
                if recs:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                        f"color:{T['tx2']};text-transform:uppercase;"
                        f'margin:16px 0 6px;">Recommendations</p>',
                        unsafe_allow_html=True,
                    )
                    for rec in recs:
                        st.markdown(
                            f"{PAGE_ICONS['trending_up']} {rec}",
                            unsafe_allow_html=True,
                        )

                # Bench players — clearly labeled as BENCH
                bench = lineup.get("bench", [])
                if bench:
                    st.markdown(
                        f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                        f"color:{T['tx2']};text-transform:uppercase;"
                        f'margin:16px 0 6px;">Bench (Sit)</p>',
                        unsafe_allow_html=True,
                    )
                    name_col = "player_name" if "player_name" in roster.columns else "name"
                    _pos_lookup = dict(
                        zip(
                            roster[name_col].astype(str),
                            roster["positions"].astype(str),
                        )
                    )
                    bench_data = []
                    for entry in bench:
                        if isinstance(entry, dict):
                            pname = entry.get("player_name", "")
                            bench_data.append(
                                {
                                    "Player": pname,
                                    "Position": entry.get("positions", _pos_lookup.get(pname, "")),
                                    "Status": "BENCH",
                                }
                            )
                        else:
                            pname = str(entry)
                            bench_data.append(
                                {
                                    "Player": pname,
                                    "Position": _pos_lookup.get(pname, ""),
                                    "Status": "BENCH",
                                }
                            )
                    if bench_data:
                        _bench_row_classes = {i: "row-bench" for i in range(len(bench_data))}
                        render_compact_table(pd.DataFrame(bench_data), row_classes=_bench_row_classes)

                # Timing breakdown
                timing = result.get("timing", {})
                if len(timing) > 1:
                    with st.expander("Performance Breakdown"):
                        timing_data = [
                            {
                                "Stage": k.replace("_", " ").title(),
                                "Time (s)": f"{v:.2f}",
                            }
                            for k, v in timing.items()
                        ]
                        render_styled_table(pd.DataFrame(timing_data))

                # ── FA Recommendations (post-optimization) ────────────
                _ctx_fa_lp = st.session_state.get("optimizer_context")
                if _ctx_fa_lp:
                    try:
                        from src.optimizer.fa_recommender import recommend_fa_moves

                        _fa_moves_lp = recommend_fa_moves(_ctx_fa_lp, max_moves=3)
                        if _fa_moves_lp:
                            st.divider()
                            st.markdown(
                                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                                f"color:{T['tx2']};text-transform:uppercase;"
                                f'margin:0 0 6px;">Recommended Free Agent Moves</p>',
                                unsafe_allow_html=True,
                            )
                            for _mv in _fa_moves_lp:
                                _add = _mv.get("add_name", "?")
                                _drop = _mv.get("drop_name", "?")
                                _delta = _mv.get("net_sgp_delta", 0)
                                _cats = _mv.get("urgency_categories", [])
                                _warn = _mv.get("news_warning")
                                _reasons = _mv.get("reasoning", [])
                                _color = T["green"] if _delta > 0 else T["danger"]
                                st.markdown(
                                    f'<div style="border-left:3px solid {_color};padding:8px 12px;margin:6px 0;'
                                    f'background:{T["bg"]};border-radius:4px;">'
                                    f"<b>Add</b> {_add} / <b>Drop</b> {_drop} "
                                    f'<span style="color:{_color};font-weight:700;">'
                                    f"({_delta:+.2f} Standings Gained Points)</span>"
                                    f"{'<br>Helps: ' + ', '.join(_cats) if _cats else ''}"
                                    f"{'<br><span style=' + chr(34) + 'color:' + T['danger'] + chr(34) + '>' + _warn + '</span>' if _warn else ''}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                if _reasons:
                                    st.caption(" | ".join(_reasons))
                    except Exception:
                        pass
        else:
            st.info("Click **Optimize Lineup** to generate your optimal start/sit decisions.")

    # ================================================================
    # TAB 2: START/SIT
    # ================================================================

    with tab_startsit:
        if not START_SIT_AVAILABLE:
            st.error(
                "The start/sit advisor module could not be loaded. "
                "Check that src/start_sit.py and its dependencies are installed."
            )
        else:
            # Auto-population from optimizer result
            opt_result = st.session_state.get("lineup_optimizer_result")
            opt_lineup = (opt_result.get("lineup") or {}) if opt_result else {}
            opt_assignments = opt_lineup.get("assignments", [])
            opt_bench = opt_lineup.get("bench", [])

            # Build per-slot data from optimizer
            _starter_ids: set[int] = set()
            _bench_ids: set[int] = set()
            _slot_map: dict[str, dict] = {}

            if opt_assignments:
                for entry in opt_assignments:
                    pid = entry.get("player_id")
                    slot = entry.get("slot", "")
                    if pid:
                        _starter_ids.add(pid)
                        _slot_map[slot] = entry

            if opt_bench:
                for entry in opt_bench:
                    if isinstance(entry, dict):
                        pid = entry.get("player_id")
                        if pid:
                            _bench_ids.add(pid)
                    else:
                        # Bench entry is a name string — look up ID
                        name_col = "player_name" if "player_name" in roster.columns else "name"
                        match = roster[roster[name_col] == str(entry)]
                        if not match.empty:
                            _bench_ids.add(int(match.iloc[0]["player_id"]))

            # Build bench player roster rows for position matching
            bench_roster = roster[roster["player_id"].isin(_bench_ids)].copy()

            # Load free agents for upgrade suggestions
            fa_pool = pd.DataFrame()
            if not pool.empty:
                try:
                    fa_pool = get_free_agents(pool.rename(columns={"player_name": "name"}))
                    if not fa_pool.empty and "name" in fa_pool.columns:
                        fa_pool = fa_pool.rename(columns={"name": "player_name"})
                except Exception:
                    fa_pool = pd.DataFrame()

            if opt_assignments:
                st.markdown(
                    f'<p style="font-size:13px;color:{T["tx2"]};margin-bottom:8px;">'
                    "The optimizer has assigned starters to each slot. For each slot below, "
                    "compare the starter against bench alternatives and top free agents at "
                    "that position. Select a slot to run the start/sit advisor.</p>",
                    unsafe_allow_html=True,
                )

                # Build slot options for selection
                slot_options = []
                for entry in opt_assignments:
                    slot = entry.get("slot", "")
                    pname = entry.get("player_name", "")
                    slot_options.append(f"{slot}: {pname}")

                selected_slot_label = st.selectbox(
                    "Select Roster Slot to Analyze",
                    options=slot_options,
                    index=0,
                    key="lineup_startsit_slot",
                )

                if selected_slot_label:
                    try:
                        slot_idx = slot_options.index(selected_slot_label)
                    except ValueError:
                        slot_idx = 0
                    starter_entry = opt_assignments[slot_idx]
                    starter_pid = starter_entry.get("player_id")
                    starter_name = starter_entry.get("player_name", "")
                    slot_name = starter_entry.get("slot", "")

                    # Determine the position(s) for this slot
                    # Map slot names to position filters
                    _slot_to_pos = {
                        "C": "C",
                        "1B": "1B",
                        "2B": "2B",
                        "3B": "3B",
                        "SS": "SS",
                        "OF": "OF",
                        "LF": "OF",
                        "CF": "OF",
                        "RF": "OF",
                        "Util": None,  # Any hitter
                        "SP": "SP",
                        "RP": "RP",
                        "P": None,  # Any pitcher
                    }
                    slot_pos = _slot_to_pos.get(slot_name)

                    # Find bench players eligible for this slot
                    bench_at_pos = []
                    if not bench_roster.empty:
                        for _, bp in bench_roster.iterrows():
                            bp_positions = str(bp.get("positions", ""))
                            if slot_pos is None:
                                # Util/P — any player of matching type
                                if slot_name == "Util" and bp.get("is_hitter", 1):
                                    bench_at_pos.append(int(bp["player_id"]))
                                elif slot_name == "P" and not bp.get("is_hitter", 1):
                                    bench_at_pos.append(int(bp["player_id"]))
                            elif slot_pos in bp_positions:
                                bench_at_pos.append(int(bp["player_id"]))

                    # Find top free agents at this position (ranked by marginal SGP)
                    fa_at_pos_ids: list[int] = []
                    if not fa_pool.empty and slot_pos:
                        fa_filtered = fa_pool[
                            fa_pool["positions"].astype(str).str.contains(slot_pos, case=False, na=False)
                        ].copy()
                        # Prefer marginal_value (SGP-based) over raw counting stats
                        if "marginal_value" in fa_filtered.columns:
                            fa_filtered["_sort_val"] = pd.to_numeric(
                                fa_filtered["marginal_value"], errors="coerce"
                            ).fillna(0)
                        elif "pick_score" in fa_filtered.columns:
                            fa_filtered["_sort_val"] = pd.to_numeric(fa_filtered["pick_score"], errors="coerce").fillna(
                                0
                            )
                        else:
                            # Fallback: sum relevant counting stats as proxy
                            _hit_cols = ["r", "hr", "rbi", "sb"]
                            _pit_cols = ["k", "sv", "w"]
                            _proxy_cols = _hit_cols if slot_pos not in ("SP", "RP") else _pit_cols
                            fa_filtered["_sort_val"] = sum(
                                pd.to_numeric(fa_filtered[c], errors="coerce").fillna(0)
                                for c in _proxy_cols
                                if c in fa_filtered.columns
                            )
                        fa_filtered = fa_filtered.nlargest(3, "_sort_val")
                        fa_at_pos_ids = fa_filtered["player_id"].tolist()
                    elif not fa_pool.empty and slot_name == "Util":
                        # For Util, show top hitters by SGP or composite stats
                        if "is_hitter" in fa_pool.columns:
                            fa_hitters = fa_pool[fa_pool["is_hitter"].fillna(0).astype(bool)].copy()
                        else:
                            fa_hitters = pd.DataFrame()
                        if not fa_hitters.empty:
                            if "marginal_value" in fa_hitters.columns:
                                fa_hitters["_sort_val"] = pd.to_numeric(
                                    fa_hitters["marginal_value"], errors="coerce"
                                ).fillna(0)
                            elif "pick_score" in fa_hitters.columns:
                                fa_hitters["_sort_val"] = pd.to_numeric(
                                    fa_hitters["pick_score"], errors="coerce"
                                ).fillna(0)
                            else:
                                fa_hitters["_sort_val"] = sum(
                                    pd.to_numeric(fa_hitters[c], errors="coerce").fillna(0)
                                    for c in ["r", "hr", "rbi", "sb"]
                                    if c in fa_hitters.columns
                                )
                            fa_hitters = fa_hitters.nlargest(3, "_sort_val")
                            fa_at_pos_ids = fa_hitters["player_id"].tolist()

                    # Combine player IDs for comparison
                    compare_ids = [starter_pid] + bench_at_pos[:3] + fa_at_pos_ids[:3]
                    compare_ids = [int(pid) for pid in compare_ids if pid is not None]
                    # Deduplicate while preserving order
                    seen: set[int] = set()
                    unique_ids: list[int] = []
                    for pid in compare_ids:
                        if pid not in seen:
                            seen.add(pid)
                            unique_ids.append(pid)
                    compare_ids = unique_ids

                    if len(compare_ids) < 2:
                        st.info(
                            f"No bench or free agent alternatives found for the "
                            f"{slot_name} slot. The current starter ({starter_name}) "
                            f"is the only option."
                        )
                    else:
                        # Label the candidates
                        _pid_to_source: dict[int, str] = {}
                        _pid_to_source[starter_pid] = "Starter"
                        for pid in bench_at_pos:
                            _pid_to_source[pid] = "Bench"
                        for pid in fa_at_pos_ids:
                            _pid_to_source[pid] = "Free Agent"

                        st.markdown(
                            f'<div style="background:{T["card"]};'
                            f"border:1px solid {T['card_h']};"
                            f'border-radius:6px;padding:10px 14px;margin-bottom:12px;">'
                            f'<span style="font-size:11px;font-weight:700;'
                            f"letter-spacing:1px;color:{T['tx2']};"
                            f'text-transform:uppercase;">Slot: {slot_name}</span>'
                            f'<span style="font-size:14px;font-weight:700;'
                            f'color:{T["tx"]};margin-left:12px;">'
                            f"{starter_name}</span>"
                            f'<span style="font-size:12px;color:{T["tx2"]};">'
                            f" vs {len(compare_ids) - 1} alternative(s)</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Build pool for start_sit (needs "name" column)
                        _ss_pool = pool.rename(columns={"player_name": "name"})

                        # Also merge FA data if FA IDs are in the comparison
                        if fa_at_pos_ids and not fa_pool.empty:
                            _fa_for_merge = fa_pool.rename(columns={"player_name": "name"})
                            _existing_ids = set(_ss_pool["player_id"].values)
                            _new_fa = _fa_for_merge[~_fa_for_merge["player_id"].isin(_existing_ids)]
                            if not _new_fa.empty:
                                _ss_pool = pd.concat([_ss_pool, _new_fa], ignore_index=True)

                        # Fetch recent form and weather from MatchupContextService
                        _ss_recent_form = None
                        _ss_weather = None
                        try:
                            from src.matchup_context import get_matchup_context

                            _ss_ctx = get_matchup_context()
                            _ss_recent_form = {}
                            for _pid in compare_ids:
                                _p_row = _ss_pool[_ss_pool["player_id"] == _pid]
                                if not _p_row.empty:
                                    _mlb = _p_row.iloc[0].get("mlb_id")
                                    if _mlb is not None:
                                        try:
                                            _ss_recent_form[_pid] = _ss_ctx.get_player_form(int(_mlb))
                                        except (ValueError, TypeError):
                                            pass
                            _ss_weather = {}
                            for _pid in compare_ids:
                                _p_row = _ss_pool[_ss_pool["player_id"] == _pid]
                                if not _p_row.empty:
                                    _team = str(_p_row.iloc[0].get("team", ""))
                                    if _team and _team not in _ss_weather:
                                        _ss_weather[_team] = _ss_ctx.get_weather(_team)
                        except Exception:
                            pass  # Graceful fallback — start/sit works without form/weather

                        with st.spinner("Computing start/sit recommendation..."):
                            try:
                                ss_result = start_sit_recommendation(
                                    player_ids=compare_ids,
                                    player_pool=_ss_pool,
                                    config=league_config,
                                    weekly_schedule=week_schedule if week_schedule else None,
                                    park_factors=(PARK_FACTORS if PARK_FACTORS else None),
                                    my_weekly_totals=my_totals if my_totals else None,
                                    opp_weekly_totals=opp_weekly_totals,
                                    standings=(standings if not standings.empty else None),
                                    team_name=user_team_name,
                                    recent_form=_ss_recent_form if _ss_recent_form else None,
                                    weather=_ss_weather if _ss_weather else None,
                                )
                            except Exception as exc:
                                logger.exception("start_sit_recommendation failed")
                                st.error(f"Recommendation engine error: {exc}")
                                ss_result = None

                        if ss_result and ss_result.get("players"):
                            players_list = ss_result["players"]
                            rec_id = ss_result.get("recommendation")
                            confidence = ss_result.get("confidence", 0.0)
                            confidence_label = ss_result.get("confidence_label", "Toss-up")

                            # Summary banner
                            if rec_id is not None:
                                rec_player = next(
                                    (p for p in players_list if p["player_id"] == rec_id),
                                    None,
                                )
                                rec_name = rec_player["name"] if rec_player else "Unknown"
                                label_color = (
                                    T["green"]
                                    if confidence_label == "Clear Start"
                                    else T["warn"]
                                    if confidence_label == "Lean Start"
                                    else T["tx2"]
                                )
                                st.markdown(
                                    f'<div style="'
                                    f"background:{T['card']};"
                                    f"border:1px solid {T['card_h']};"
                                    f"border-left:4px solid {label_color};"
                                    f"border-radius:8px;padding:14px 18px;"
                                    f'margin-bottom:16px;">'
                                    f'<div style="font-size:11px;font-weight:700;'
                                    f"letter-spacing:1px;color:{T['tx2']};"
                                    f'text-transform:uppercase;margin-bottom:4px;">'
                                    f"Recommendation</div>"
                                    f'<div style="font-size:20px;font-weight:700;'
                                    f'color:{T["tx"]};">'
                                    f"Start {rec_name}</div>"
                                    f'<div style="font-size:13px;color:{label_color};'
                                    f'font-weight:600;margin-top:4px;">'
                                    f"{confidence_label} &mdash; "
                                    f"{confidence * 100:.0f}% confidence</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                            # Rankings table
                            st.markdown(
                                f'<p style="font-size:12px;font-weight:700;'
                                f"letter-spacing:1px;color:{T['tx2']};"
                                f'text-transform:uppercase;margin:0 0 6px;">'
                                f"Player Rankings</p>",
                                unsafe_allow_html=True,
                            )

                            _pid_to_mlb_ss = {}
                            if "mlb_id" in pool.columns and "player_id" in pool.columns:
                                _pid_to_mlb_ss = dict(zip(pool["player_id"], pool["mlb_id"]))

                            rows = []
                            row_classes: dict[int, str] = {}
                            for rank, p in enumerate(players_list):
                                is_rec = p["player_id"] == rec_id
                                source = _pid_to_source.get(p["player_id"], "")
                                # Clear decision label: START for recommended, source-aware for others
                                if is_rec:
                                    rec_label = "START"
                                elif source == "Free Agent":
                                    rec_label = "FA OPTION"
                                elif source == "Bench":
                                    rec_label = "BENCH ALT"
                                else:
                                    rec_label = "SIT"
                                top_cat = ""
                                if p.get("category_impact"):
                                    sorted_cats = sorted(
                                        p["category_impact"].items(),
                                        key=lambda x: abs(x[1]),
                                        reverse=True,
                                    )
                                    if sorted_cats:
                                        top_cat = sorted_cats[0][0]

                                entry = {
                                    "Player": p["name"],
                                    "Source": source,
                                    "Decision": rec_label,
                                    "Score": f"{p['start_score']:.2f}",
                                    "Floor": f"{p['floor']:.2f}",
                                    "Ceiling": f"{p['ceiling']:.2f}",
                                    "Top Category": (top_cat.upper() if top_cat else ""),
                                }
                                if _pid_to_mlb_ss:
                                    entry["mlb_id"] = _pid_to_mlb_ss.get(p["player_id"])
                                rows.append(entry)
                                row_classes[rank] = "row-start" if is_rec else "row-sit"

                            display_df = pd.DataFrame(rows)

                            st.markdown(
                                f"<style>"
                                f"tr.row-start td {{ background-color:"
                                f"{T['green']}22 !important; }}"
                                f"tr.row-start .col-name {{ color:"
                                f"{T['green']} !important; "
                                f"font-weight:700 !important; }}"
                                f"tr.row-sit td {{ background-color:"
                                f"{T['danger']}18 !important; }}"
                                f"tr.row-sit .col-name {{ color:"
                                f"{T['danger']} !important; }}"
                                f"</style>",
                                unsafe_allow_html=True,
                            )

                            render_compact_table(
                                display_df,
                                highlight_cols=[
                                    "Score",
                                    "Floor",
                                    "Ceiling",
                                ],
                                row_classes=row_classes,
                            )

                            # Per-player reasoning
                            st.markdown(
                                f'<p style="font-size:12px;font-weight:700;'
                                f"letter-spacing:1px;color:{T['tx2']};"
                                f'text-transform:uppercase;margin:16px 0 6px;">'
                                f"Decision Reasoning</p>",
                                unsafe_allow_html=True,
                            )

                            for p in players_list:
                                is_rec = p["player_id"] == rec_id
                                badge_color = T["green"] if is_rec else T["danger"]
                                badge_label = "START" if is_rec else "SIT"
                                reasons = p.get("reasoning", [])
                                reasons_html = "".join(
                                    f'<li style="margin-bottom:4px;font-size:13px;color:{T["tx"]};">{r}</li>'
                                    for r in reasons
                                )

                                st.markdown(
                                    f'<div style="'
                                    f"background:{T['card']};"
                                    f"border:1px solid {T['card_h']};"
                                    f"border-radius:6px;"
                                    f"padding:12px 16px;"
                                    f'margin-bottom:10px;">'
                                    f'<div style="display:flex;'
                                    f"align-items:center;gap:10px;"
                                    f'margin-bottom:8px;">'
                                    f'<span style="font-weight:700;'
                                    f"font-size:14px;"
                                    f'color:{T["tx"]};">'
                                    f"{p['name']}</span>"
                                    f'<span style="'
                                    f"background:{badge_color}22;"
                                    f"color:{badge_color};"
                                    f"font-size:10px;font-weight:700;"
                                    f"letter-spacing:1px;"
                                    f"padding:2px 8px;"
                                    f'border-radius:4px;">'
                                    f"{badge_label}</span>"
                                    f"</div>"
                                    f'<ul style="margin:0;padding-left:18px;">'
                                    f"{reasons_html}</ul>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        else:
                            st.warning("No recommendation could be generated for the selected players.")

            else:
                # Manual comparison mode (no optimizer result)
                st.markdown(
                    f'<p style="font-size:13px;color:{T["tx2"]};margin-bottom:4px;">'
                    "Run the optimizer first to auto-populate slot comparisons, "
                    "or select 2 to 4 players below to compare manually. "
                    "The advisor ranks them using a 3-layer decision model.</p>",
                    unsafe_allow_html=True,
                )

                # Build player name list (rostered first, deduplicated)
                all_names: list[str] = []
                all_ids: list[int] = []
                if not pool.empty:
                    _deduped_pool = pool.drop_duplicates(subset=["player_id"], keep="first")
                    if user_player_ids:
                        rostered = _deduped_pool[_deduped_pool["player_id"].isin(user_player_ids)].copy()
                        others = _deduped_pool[~_deduped_pool["player_id"].isin(user_player_ids)].copy()
                        sorted_pool = pd.concat([rostered, others], ignore_index=True)
                    else:
                        sorted_pool = _deduped_pool.copy()
                    all_names = sorted_pool["player_name"].tolist()
                    all_ids = sorted_pool["player_id"].tolist()

                selected_names = st.multiselect(
                    "Players to compare",
                    options=all_names,
                    default=[],
                    max_selections=4,
                    key="lineup_startsit_manual_select",
                    help="Select 2 to 4 players competing for the same slot.",
                    placeholder="Choose 2 to 4 players...",
                )

                if not selected_names:
                    st.info(
                        "Select at least 2 players above to receive a "
                        "start/sit recommendation. Rostered players appear "
                        "at the top of the list."
                    )
                elif len(selected_names) == 1:
                    st.warning("Select at least 2 players to compare. Only 1 player chosen.")
                else:
                    name_to_id = dict(zip(all_names, all_ids))
                    selected_ids = [int(name_to_id[n]) for n in selected_names if n in name_to_id]

                    if len(selected_ids) < 2:
                        st.warning("Could not resolve player IDs for the selected players.")
                    else:
                        # Fetch recent form for manual comparison
                        _m_form = None
                        try:
                            from src.matchup_context import get_matchup_context

                            _m_ctx = get_matchup_context()
                            _m_form = {}
                            _m_pool = pool.rename(columns={"player_name": "name"})
                            for _mid in selected_ids:
                                _mr = _m_pool[_m_pool["player_id"] == _mid]
                                if not _mr.empty:
                                    _mm = _mr.iloc[0].get("mlb_id")
                                    if _mm is not None:
                                        try:
                                            _m_form[_mid] = _m_ctx.get_player_form(int(_mm))
                                        except (ValueError, TypeError):
                                            pass
                        except Exception:
                            _m_pool = pool.rename(columns={"player_name": "name"})

                        with st.spinner("Computing start/sit recommendation..."):
                            try:
                                manual_result = start_sit_recommendation(
                                    player_ids=selected_ids,
                                    player_pool=_m_pool
                                    if "_m_pool" in dir()
                                    else pool.rename(columns={"player_name": "name"}),
                                    config=league_config,
                                    weekly_schedule=week_schedule if week_schedule else None,
                                    park_factors=(PARK_FACTORS if PARK_FACTORS else None),
                                    my_weekly_totals=(my_totals if my_totals else None),
                                    opp_weekly_totals=opp_weekly_totals,
                                    standings=(standings if not standings.empty else None),
                                    team_name=user_team_name,
                                    recent_form=_m_form if _m_form else None,
                                )
                            except Exception as exc:
                                logger.exception("start_sit_recommendation failed")
                                st.error(f"Recommendation engine error: {exc}")
                                manual_result = None

                        # Analytics badge
                        try:
                            from src.data_bootstrap import get_bootstrap_context

                            _boot_ctx = get_bootstrap_context()
                            if _boot_ctx:
                                from src.ui_analytics_badge import (
                                    render_analytics_badge,
                                )

                                render_analytics_badge(_boot_ctx)
                        except Exception:
                            pass

                        if manual_result and manual_result.get("players"):
                            players_list = manual_result["players"]
                            rec_id = manual_result.get("recommendation")
                            confidence = manual_result.get("confidence", 0.0)
                            confidence_label = manual_result.get("confidence_label", "Toss-up")

                            # Summary banner
                            if rec_id is not None:
                                rec_player = next(
                                    (p for p in players_list if p["player_id"] == rec_id),
                                    None,
                                )
                                rec_name = rec_player["name"] if rec_player else "Unknown"
                                label_color = (
                                    T["green"]
                                    if confidence_label == "Clear Start"
                                    else T["warn"]
                                    if confidence_label == "Lean Start"
                                    else T["tx2"]
                                )
                                st.markdown(
                                    f'<div style="'
                                    f"background:{T['card']};"
                                    f"border:1px solid {T['card_h']};"
                                    f"border-left:4px solid "
                                    f"{label_color};"
                                    f"border-radius:8px;"
                                    f"padding:14px 18px;"
                                    f'margin-bottom:16px;">'
                                    f'<div style="font-size:11px;'
                                    f"font-weight:700;"
                                    f"letter-spacing:1px;"
                                    f"color:{T['tx2']};"
                                    f"text-transform:uppercase;"
                                    f'margin-bottom:4px;">'
                                    f"Recommendation</div>"
                                    f'<div style="font-size:20px;'
                                    f"font-weight:700;"
                                    f'color:{T["tx"]};">'
                                    f"Start {rec_name}</div>"
                                    f'<div style="font-size:13px;'
                                    f"color:{label_color};"
                                    f"font-weight:600;"
                                    f'margin-top:4px;">'
                                    f"{confidence_label} &mdash; "
                                    f"{confidence * 100:.0f}% "
                                    f"confidence</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                            # Rankings table
                            _pid_to_mlb_m = {}
                            if "mlb_id" in pool.columns and "player_id" in pool.columns:
                                _pid_to_mlb_m = dict(zip(pool["player_id"], pool["mlb_id"]))

                            m_rows = []
                            m_row_classes: dict[int, str] = {}
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

                                entry = {
                                    "Player": p["name"],
                                    "Decision": rec_label,
                                    "Score": f"{p['start_score']:.2f}",
                                    "Floor": f"{p['floor']:.2f}",
                                    "Ceiling": f"{p['ceiling']:.2f}",
                                    "Top Category": (top_cat.upper() if top_cat else ""),
                                }
                                if _pid_to_mlb_m:
                                    entry["mlb_id"] = _pid_to_mlb_m.get(p["player_id"])
                                m_rows.append(entry)
                                m_row_classes[rank] = "row-start" if is_rec else "row-sit"

                            m_display_df = pd.DataFrame(m_rows)

                            st.markdown(
                                f"<style>"
                                f"tr.row-start td {{ "
                                f"background-color:{T['green']}22 "
                                f"!important; }}"
                                f"tr.row-start .col-name {{ "
                                f"color:{T['green']} !important; "
                                f"font-weight:700 !important; }}"
                                f"tr.row-sit td {{ "
                                f"background-color:{T['danger']}18 "
                                f"!important; }}"
                                f"tr.row-sit .col-name {{ "
                                f"color:{T['danger']} !important; }}"
                                f"</style>",
                                unsafe_allow_html=True,
                            )

                            render_compact_table(
                                m_display_df,
                                highlight_cols=[
                                    "Score",
                                    "Floor",
                                    "Ceiling",
                                ],
                                row_classes=m_row_classes,
                            )

                            # Per-player reasoning
                            for p in players_list:
                                is_rec = p["player_id"] == rec_id
                                badge_color = T["green"] if is_rec else T["danger"]
                                badge_label = "START" if is_rec else "SIT"
                                reasons = p.get("reasoning", [])
                                reasons_html = "".join(
                                    f'<li style="margin-bottom:4px;font-size:13px;color:{T["tx"]};">{r}</li>'
                                    for r in reasons
                                )
                                st.markdown(
                                    f'<div style="'
                                    f"background:{T['card']};"
                                    f"border:1px solid "
                                    f"{T['card_h']};"
                                    f"border-radius:6px;"
                                    f"padding:12px 16px;"
                                    f'margin-bottom:10px;">'
                                    f'<div style="display:flex;'
                                    f"align-items:center;"
                                    f"gap:10px;"
                                    f'margin-bottom:8px;">'
                                    f'<span style="'
                                    f"font-weight:700;"
                                    f"font-size:14px;"
                                    f'color:{T["tx"]};">'
                                    f"{p['name']}</span>"
                                    f'<span style="'
                                    f"background:"
                                    f"{badge_color}22;"
                                    f"color:{badge_color};"
                                    f"font-size:10px;"
                                    f"font-weight:700;"
                                    f"letter-spacing:1px;"
                                    f"padding:2px 8px;"
                                    f'border-radius:4px;">'
                                    f"{badge_label}</span>"
                                    f"</div>"
                                    f'<ul style="margin:0;'
                                    f'padding-left:18px;">'
                                    f"{reasons_html}</ul>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                            # Player card selector
                            rec_names = [p["name"] for p in players_list]
                            rec_ids = [p["player_id"] for p in players_list]
                            if rec_ids:
                                render_player_select(
                                    rec_names,
                                    rec_ids,
                                    key_suffix="lineup_startsit",
                                )

                        else:
                            st.warning("No recommendation could be generated.")

    # ================================================================
    # TAB 3: CATEGORY ANALYSIS
    # ================================================================

    with tab_analysis:
        result = st.session_state.get("lineup_optimizer_result")
        weights = result.get("category_weights", {}) if result else {}

        # Use shared context weights for consistency (same weights optimizer used)
        if not weights:
            _analysis_ctx = st.session_state.get("optimizer_context")
            if _analysis_ctx and hasattr(_analysis_ctx, "category_weights") and _analysis_ctx.category_weights:
                weights = dict(_analysis_ctx.category_weights)

        if not weights and not standings.empty:
            if SGP_AVAILABLE:
                try:
                    weights = compute_nonlinear_weights(standings, user_team_name)
                except Exception:
                    pass

        if weights:
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:0 0 6px;">Category Weight Distribution</p>',
                unsafe_allow_html=True,
            )
            st.markdown("**Priority categories** (higher weight = bigger marginal standings impact):")
            weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for cat, weight in weights_sorted:
                bar_len = int(min(weight, 3.0) / 3.0 * 20)
                icon = (
                    PAGE_ICONS["fire"]
                    if weight > 1.5
                    else PAGE_ICONS["trending_up"]
                    if weight > 1.0
                    else PAGE_ICONS["minus"]
                )
                display_name = CAT_DISPLAY_NAMES.get(cat, cat.upper())
                st.markdown(
                    f"{icon} **{display_name}**: {'█' * bar_len}{'░' * (20 - bar_len)} ({weight:.2f}x)",
                    unsafe_allow_html=True,
                )
            st.caption(METRIC_TOOLTIPS.get("cat_targeting", ""))
        elif standings.empty:
            st.info(
                "Import league standings to see category targeting recommendations. "
                "The optimizer will identify where small gains yield the biggest standings jumps."
            )
        else:
            st.info("Category weights will appear after running the optimizer.")

        # Maximin comparison
        maximin = result.get("maximin_comparison") if result else None
        if maximin:
            st.divider()
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:0 0 6px;">Maximin (Balanced) Lineup Comparison</p>',
                unsafe_allow_html=True,
            )
            z_val = maximin.get("z_value", 0)
            st.metric("Worst-Category Floor (z-score)", f"{z_val:.2f}")

            maximin_assignments = maximin.get("assignments", {})
            if maximin_assignments:
                maximin_data = [{"Player": name} for name, val in maximin_assignments.items() if val == 1]
                if maximin_data:
                    render_styled_table(pd.DataFrame(maximin_data))
            st.caption(
                "The maximin lineup maximizes the WORST category instead of the total. "
                "Compare with your main lineup to see if you are sacrificing too much balance."
            )

        # Standings position for each category
        if not standings.empty and my_totals:
            st.divider()
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:0 0 6px;">Current Standings Position by Category</p>',
                unsafe_allow_html=True,
            )
            position_rows = []
            for cat in ALL_CATS:
                my_val = my_totals.get(cat, 0)
                is_inverse = cat in ("l", "era", "whip")
                rank = 1
                for tname, ttotals in team_totals.items():
                    if tname == user_team_name:
                        continue
                    opp_val = ttotals.get(cat, 0)
                    if is_inverse:
                        if opp_val < my_val:
                            rank += 1
                    else:
                        if opp_val > my_val:
                            rank += 1

                if cat in ("avg", "obp"):
                    val_fmt = format_stat(my_val, cat.upper())
                elif cat in ("era", "whip"):
                    val_fmt = format_stat(my_val, cat.upper())
                else:
                    val_fmt = f"{my_val:.0f}"
                rank_label = f"{rank}" + ("st" if rank == 1 else "nd" if rank == 2 else "rd" if rank == 3 else "th")

                position_rows.append(
                    {
                        "Category": CAT_DISPLAY_NAMES.get(cat, cat.upper()),
                        "Your Total": val_fmt,
                        "Rank": rank_label,
                        "Points": str(13 - rank),
                    }
                )
            render_styled_table(pd.DataFrame(position_rows))

    # S2: H2H tab removed — was duplicate of Matchup Planner.
    # See pages/11_Matchup_Planner.py for Category Probabilities.
    if False:  # pragma: no cover — dead code preserved for reference
        # Build H2H data from multiple sources (priority order):
        # 1. Shared optimizer context (if optimizer has been run)
        # 2. Context panel selections (my_totals, opp_totals, selected_opponent)
        # 3. Live matchup data (fresh fetch)
        _h2h_ctx = st.session_state.get("optimizer_context")
        if _h2h_ctx and hasattr(_h2h_ctx, "my_totals") and _h2h_ctx.my_totals:
            _h2h_my = dict(_h2h_ctx.my_totals)
            _h2h_opp = dict(_h2h_ctx.opp_totals)
            _h2h_opponent_name = _h2h_ctx.opponent_name or ""
        else:
            _h2h_my = dict(my_totals) if my_totals else {}
            _h2h_opp = dict(opp_totals) if opp_totals else {}
            _h2h_opponent_name = selected_opponent or ""

        # Fallback: if still missing, fetch live matchup directly
        if not _h2h_my or not _h2h_opp:
            try:
                _h2h_matchup = yds.get_matchup(force_refresh=True)
                if _h2h_matchup and "categories" in _h2h_matchup:
                    _h2h_opponent_name = _h2h_matchup.get("opp_name", _h2h_opponent_name)
                    for _mc in _h2h_matchup["categories"]:
                        _cat = str(_mc.get("cat", "")).lower()
                        if _cat:
                            try:
                                _h2h_my[_cat] = float(_mc.get("you", 0) or 0)
                                _h2h_opp[_cat] = float(_mc.get("opp", 0) or 0)
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.warning("H2H live matchup fallback failed: %s", e)

        # Build opponent_names from live data if standings were empty
        _h2h_opponent_names = opponent_names
        if not _h2h_opponent_names and _h2h_opponent_name:
            _h2h_opponent_names = [_h2h_opponent_name]

        if not H2H_AVAILABLE:
            st.info("Head-to-Head analysis module not available. Ensure scipy is installed.")
        elif not _h2h_opponent_name or not _h2h_opp:
            if not _h2h_opponent_names:
                st.info(
                    "No matchup data available. Connect your Yahoo league and ensure "
                    "an active matchup exists to see Head-to-Head analysis."
                )
            else:
                st.info(
                    "Select a Head-to-Head opponent in the settings panel to see "
                    "per-category win probabilities and matchup-specific strategy."
                )
        elif not _h2h_my:
            st.info("No category totals available. Sync your Yahoo league to get matchup data.")
        else:
            _opp_display = "".join(c for c in _h2h_opponent_name if ord(c) < 0x10000).strip()
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:0 0 6px;">Matchup: {_display_team_name} vs {_opp_display}</p>',
                unsafe_allow_html=True,
            )

            h2h_result = estimate_h2h_win_probability(_h2h_my, _h2h_opp)
            per_cat = h2h_result.get("per_category", {})
            exp_wins = h2h_result.get("expected_wins", 5.0)
            overall_prob = h2h_result.get("overall_win_prob", 0.5)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Win Probability", f"{overall_prob:.1%}")
            with mc2:
                st.metric("Expected Category Wins", f"{exp_wins:.2f} / 10")
            with mc3:
                status = "Favored" if exp_wins > 5.5 else "Underdog" if exp_wins < 4.5 else "Toss-up"
                st.metric("Matchup Status", status)

            # Per-category breakdown
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:16px 0 6px;">Category-by-Category Breakdown</p>',
                unsafe_allow_html=True,
            )
            cat_rows = []
            for cat in ALL_CATS:
                p_win = per_cat.get(cat, 0.5)
                my_val = _h2h_my.get(cat, 0)
                opp_val = _h2h_opp.get(cat, 0)

                if p_win >= 0.65:
                    verdict = "Likely Win"
                elif p_win >= 0.50:
                    verdict = "Lean Win"
                elif p_win >= 0.35:
                    verdict = "Lean Loss"
                else:
                    verdict = "Likely Loss"

                bar_len = int(p_win * 20)
                bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)

                if cat in ("avg", "obp"):
                    you_fmt = format_stat(my_val, cat.upper())
                    opp_fmt = format_stat(opp_val, cat.upper())
                elif cat in ("era", "whip"):
                    you_fmt = format_stat(my_val, cat.upper())
                    opp_fmt = format_stat(opp_val, cat.upper())
                else:
                    you_fmt = f"{my_val:.0f}"
                    opp_fmt = f"{opp_val:.0f}"

                cat_rows.append(
                    {
                        "Category": CAT_DISPLAY_NAMES.get(cat, cat.upper()),
                        "You": you_fmt,
                        "Opponent": opp_fmt,
                        "Win %": f"{p_win:.1%}",
                        "Confidence": bar,
                        "Verdict": verdict,
                    }
                )

            render_styled_table(pd.DataFrame(cat_rows))

            # Focus areas
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:16px 0 6px;">Optimal Focus Areas</p>',
                unsafe_allow_html=True,
            )
            h2h_weights = compute_h2h_category_weights(_h2h_my, _h2h_opp)
            if h2h_weights:
                focus_sorted = sorted(h2h_weights.items(), key=lambda x: x[1], reverse=True)
                for cat, weight in focus_sorted[:5]:
                    icon = (
                        PAGE_ICONS["fire"]
                        if weight > 1.5
                        else PAGE_ICONS["trending_up"]
                        if weight > 1.0
                        else PAGE_ICONS["minus"]
                    )
                    display_name = CAT_DISPLAY_NAMES.get(cat, cat.upper())
                    bar_len = int(min(weight, 3.0) / 3.0 * 20)
                    st.markdown(
                        f"{icon} **{display_name}**: {'█' * bar_len}{'░' * (20 - bar_len)} ({weight:.2f}x weight)",
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Categories near the toss-up point get higher weight. "
                    "Small improvements there have the biggest impact on winning."
                )

    # ================================================================
    # TAB 4: STREAMING
    # ================================================================

    with tab_streaming:
        # ── Shared data: fetch weekly schedule + FA pitchers ──────────
        _STREAM_TEAM_ABBREVS: dict[str, str] = {
            "Arizona Diamondbacks": "ARI",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KC",
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Oakland Athletics": "OAK",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD",
            "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TB",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH",
        }
        _STREAM_ABBREV_TO_FULL = {v: k for k, v in _STREAM_TEAM_ABBREVS.items()}

        # IL/DTD/NA statuses to exclude
        _IL_EXCLUDE = {
            "il10",
            "il15",
            "il60",
            "il",
            "na",
            "not active",
            "dl",
            "dtd",
            "day-to-day",
            "minors",
            "out",
            "suspended",
        }

        # Fetch the weekly schedule (shared across all sections)
        _stream_schedule: list[dict] = []
        _stream_team_game_counts: dict[str, int] = {}
        _stream_team_games_by_date: dict[str, dict[str, list[str]]] = {}
        _stream_target_date = ""
        try:
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            from datetime import timezone as _tz

            import statsapi

            _ET_STREAM = _tz(_td(hours=-4))
            _stream_today = _dt.now(_ET_STREAM)
            _stream_end = _stream_today + _td(days=7)
            _stream_target_date = _stream_today.strftime("%Y-%m-%d")

            # Try using get_target_game_date for smarter date targeting
            try:
                from src.game_day import get_target_game_date

                _stream_target_date = get_target_game_date()
            except Exception:
                pass

            _raw_sched = statsapi.schedule(
                start_date=_stream_today.strftime("%Y-%m-%d"),
                end_date=_stream_end.strftime("%Y-%m-%d"),
            )
            for _g in _raw_sched:
                _gd = _g.get("game_date", "")
                _home_full = _g.get("home_name", "")
                _away_full = _g.get("away_name", "")
                _home_abbr = _STREAM_TEAM_ABBREVS.get(_home_full, "")
                _away_abbr = _STREAM_TEAM_ABBREVS.get(_away_full, "")

                _stream_schedule.append(
                    {
                        "game_date": _gd,
                        "home_name": _home_full,
                        "away_name": _away_full,
                        "home_abbr": _home_abbr,
                        "away_abbr": _away_abbr,
                        "home_probable_pitcher": _g.get("home_probable_pitcher", ""),
                        "away_probable_pitcher": _g.get("away_probable_pitcher", ""),
                    }
                )
                for _abbr in [_home_abbr, _away_abbr]:
                    if _abbr:
                        _stream_team_game_counts[_abbr] = _stream_team_game_counts.get(_abbr, 0) + 1
                        _stream_team_games_by_date.setdefault(_abbr, {}).setdefault(_gd, [])
                        # Store opponent for this date
                        _opp = _away_abbr if _abbr == _home_abbr else _home_abbr
                        _stream_team_games_by_date[_abbr][_gd].append(_opp)
        except Exception as _sched_exc:
            logger.debug("Streaming schedule fetch failed: %s", _sched_exc)

        # Build probable pitcher schedule: pitcher_name -> [{date, opponent, is_home}]
        _pitcher_schedule: dict[str, list[dict]] = {}
        _pitcher_teams: dict[str, str] = {}
        for _g in _stream_schedule:
            _gd = _g["game_date"]
            for _side, _opp_side, _is_home in [
                ("home_abbr", "away_abbr", True),
                ("away_abbr", "home_abbr", False),
            ]:
                _pp_name = _g.get(
                    "home_probable_pitcher" if _is_home else "away_probable_pitcher",
                    "",
                )
                if _pp_name:
                    _pitcher_schedule.setdefault(_pp_name, []).append(
                        {
                            "date": _gd,
                            "opponent": _g[_opp_side],
                            "is_home": _is_home,
                        }
                    )
                    _pitcher_teams[_pp_name] = _g[_side]

        # Load FA pitcher pool
        _fa_sp_pool = pd.DataFrame()
        try:
            _fa_raw = yds.get_free_agents(max_players=500)
            if not _fa_raw.empty:
                if "name" in _fa_raw.columns and "player_name" not in _fa_raw.columns:
                    _fa_raw = _fa_raw.rename(columns={"name": "player_name"})
                # Filter to pitchers only (SP or RP in positions)
                _has_pos = "positions" in _fa_raw.columns
                if _has_pos:
                    _fa_sp_pool = _fa_raw[_fa_raw["positions"].str.contains("SP|RP", case=False, na=False)].copy()
                else:
                    # Fallback: use is_hitter flag
                    _fa_sp_pool = _fa_raw[_fa_raw.get("is_hitter", 1) == 0].copy()
        except Exception as _fa_exc:
            logger.debug("FA pool fetch for streaming failed: %s", _fa_exc)

        # If Yahoo FA pool is empty, fall back to player pool unrostered pitchers
        if _fa_sp_pool.empty and not pool.empty:
            try:
                _rostered_ids = set(roster["player_id"].tolist()) if "player_id" in roster.columns else set()
                _pool_pitchers = pool[pool.get("is_hitter", 1) == 0].copy()
                if not _pool_pitchers.empty and "player_id" in _pool_pitchers.columns:
                    _fa_sp_pool = _pool_pitchers[~_pool_pitchers["player_id"].isin(_rostered_ids)].copy()
            except Exception:
                pass

        # Merge projection stats from player pool onto FA pool
        if not _fa_sp_pool.empty and not pool.empty:
            try:
                _proj_cols = ["player_id", "era", "whip", "k", "w", "ip", "sv", "k_bb_pct", "xfip"]
                _avail_proj = [c for c in _proj_cols if c in pool.columns]
                if "player_id" in _avail_proj:
                    _proj_merge = pool[_avail_proj].copy()
                    # Drop existing stat columns in FA pool before merge to prefer projections
                    _drop_cols = [c for c in _avail_proj if c != "player_id" and c in _fa_sp_pool.columns]
                    if _drop_cols:
                        _fa_sp_pool = _fa_sp_pool.drop(columns=_drop_cols, errors="ignore")
                    _fa_sp_pool = _fa_sp_pool.merge(_proj_merge, on="player_id", how="left")
            except Exception:
                pass

        # Coerce numeric stat columns
        for _sc in ["era", "whip", "k", "w", "ip", "sv", "k_bb_pct", "xfip"]:
            if _sc in _fa_sp_pool.columns:
                _fa_sp_pool[_sc] = pd.to_numeric(_fa_sp_pool[_sc], errors="coerce").fillna(0)

        # ──────────────────────────────────────────────────────────────
        # SECTION 1: Daily Streaming Picks
        # ──────────────────────────────────────────────────────────────
        st.markdown(
            f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
            f"color:{T['tx2']};text-transform:uppercase;"
            f'margin:0 0 6px;">Streaming Pitcher Picks</p>',
            unsafe_allow_html=True,
        )

        if _stream_target_date:
            st.caption(f"Target date: **{_stream_target_date}** | Schedule window: next 7 days")

        _stream_picks: list[dict] = []
        if not _fa_sp_pool.empty and _stream_schedule:
            # Build name->player_id lookup from FA pool
            _fa_name_col = "player_name" if "player_name" in _fa_sp_pool.columns else "name"
            _fa_name_set = (
                set(_fa_sp_pool[_fa_name_col].str.strip().tolist()) if _fa_name_col in _fa_sp_pool.columns else set()
            )

            # Score each FA pitcher based on: schedule, projections, matchup quality
            for _, _fp in _fa_sp_pool.iterrows():
                _fp_name = str(_fp.get("player_name", _fp.get("name", ""))).strip()
                _fp_team = str(_fp.get("team", "")).strip()
                _fp_positions = str(_fp.get("positions", ""))
                _is_sp = "SP" in _fp_positions

                # Projected stats
                _fp_k = float(_fp.get("k", 0) or 0)
                _fp_era = float(_fp.get("era", 0) or 0)
                _fp_whip = float(_fp.get("whip", 0) or 0)
                _fp_ip = float(_fp.get("ip", 0) or 0)
                _fp_w = float(_fp.get("w", 0) or 0)
                _fp_sv = float(_fp.get("sv", 0) or 0)

                # Skip pitchers with no meaningful projections
                if _fp_ip <= 0 and _fp_k <= 0 and _fp_sv <= 0:
                    continue

                # Per-IP rates for scoring
                _k_per_ip = (_fp_k / _fp_ip) if _fp_ip > 0 else 0
                _k_per_start = _k_per_ip * 6.0  # approximate per-start K

                # Check if this pitcher is a probable starter in the schedule
                _starts_this_week: list[dict] = []
                _next_start_date = ""
                _next_opp = ""
                if _fp_name in _pitcher_schedule:
                    _starts_this_week = _pitcher_schedule[_fp_name]
                    if _starts_this_week:
                        _starts_this_week.sort(key=lambda x: x["date"])
                        _next_start_date = _starts_this_week[0]["date"]
                        _next_opp = _starts_this_week[0]["opponent"]

                # Calculate team games this week
                _team_games = _stream_team_game_counts.get(_fp_team, 0)

                # ── Streaming score ──
                # Base: K rate (higher = better)
                _score = _k_per_start * 1.5

                # Win upside
                _score += _fp_w * 0.3

                # ERA/WHIP penalty (lower is better; penalize bad ratios)
                if _fp_era > 0:
                    _era_penalty = max(0, (_fp_era - 3.50)) * 1.2
                    _score -= _era_penalty
                if _fp_whip > 0:
                    _whip_penalty = max(0, (_fp_whip - 1.15)) * 3.0
                    _score -= _whip_penalty

                # Two-start bonus for SP
                _num_starts = len(_starts_this_week)
                if _is_sp and _num_starts >= 2:
                    _score += 3.0
                elif _is_sp and _num_starts == 1:
                    _score += 1.0

                # Known start bonus (probable pitcher listed)
                if _next_start_date:
                    _score += 1.5

                # Schedule volume bonus (team has many games)
                if _team_games >= 7:
                    _score += 0.5

                # Saves bonus for RP closers
                if _fp_sv > 0 and "RP" in _fp_positions:
                    _score += _fp_sv * 0.4

                # Matchup quality: check opponent batting strength
                if _next_opp:
                    try:
                        from src.game_day import get_team_strength

                        _opp_strength = get_team_strength(_next_opp)
                        _opp_wrc = float(_opp_strength.get("wrc_plus", 100) or 100)
                        # Lower wRC+ = weaker opponent = better matchup
                        if _opp_wrc < 95:
                            _score += 1.5  # favorable
                            _matchup_label = "Easy"
                        elif _opp_wrc > 108:
                            _score -= 1.0  # tough
                            _matchup_label = "Tough"
                        else:
                            _matchup_label = "Avg"
                    except Exception:
                        _matchup_label = "--"
                else:
                    _matchup_label = "--"

                # Stream type label
                if _is_sp and _num_starts >= 2:
                    _stream_type = "2-Start SP"
                elif _is_sp:
                    _stream_type = "SP Stream"
                elif _fp_sv > 0:
                    _stream_type = "Closer"
                else:
                    _stream_type = "RP Stream"

                _stream_picks.append(
                    {
                        "player_name": _fp_name,
                        "team": _fp_team,
                        "type": _stream_type,
                        "next_start": _next_start_date if _next_start_date else "--",
                        "opponent": _next_opp if _next_opp else "--",
                        "num_starts": _num_starts,
                        "proj_k": _fp_k,
                        "proj_era": _fp_era,
                        "proj_whip": _fp_whip,
                        "proj_ip": _fp_ip,
                        "proj_sv": _fp_sv,
                        "matchup": _matchup_label,
                        "score": round(_score, 2),
                    }
                )

            # Sort by score descending, take top 15
            _stream_picks.sort(key=lambda x: x["score"], reverse=True)
            _stream_picks = _stream_picks[:15]

        if _stream_picks:
            _picks_display = []
            for _sp in _stream_picks:
                _row_data: dict = {
                    "Pitcher": _sp["player_name"],
                    "Team": _sp["team"],
                    "Type": _sp["type"],
                    "Next Start": _sp["next_start"],
                    "vs": _sp["opponent"],
                    "Matchup": _sp["matchup"],
                    "Starts": _sp["num_starts"] if _sp["num_starts"] > 0 else "--",
                }
                # Show relevant stat based on type
                if "Closer" in _sp["type"]:
                    _row_data["Proj SV"] = f"{_sp['proj_sv']:.0f}"
                    _row_data["ERA"] = format_stat(_sp["proj_era"], "era") if _sp["proj_era"] > 0 else "--"
                else:
                    _row_data["Proj K"] = f"{_sp['proj_k']:.0f}"
                    _row_data["ERA"] = format_stat(_sp["proj_era"], "era") if _sp["proj_era"] > 0 else "--"
                    _row_data["WHIP"] = format_stat(_sp["proj_whip"], "whip") if _sp["proj_whip"] > 0 else "--"
                _row_data["Score"] = f"{_sp['score']:.1f}"
                _picks_display.append(_row_data)

            render_styled_table(pd.DataFrame(_picks_display))
            st.caption(
                "Score combines projected K rate, ERA/WHIP quality, two-start bonus, "
                "and opponent strength (wRC+). Higher is better."
            )
        elif not _stream_schedule:
            st.info("Unable to load MLB schedule. Check your internet connection.")
        elif _fa_sp_pool.empty:
            st.info("No free agent pitchers available. Connect to Yahoo or ensure projection data is loaded.")
        else:
            st.info("No streaming candidates found for this week.")

        # ──────────────────────────────────────────────────────────────
        # SECTION 2: Two-Start Starting Pitchers (your roster)
        # ──────────────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
            f"color:{T['tx2']};text-transform:uppercase;"
            f'margin:0 0 6px;">Two-Start Pitchers (Your Roster)</p>',
            unsafe_allow_html=True,
        )

        two_start_sps: list[dict] = []
        _stream_ctx = st.session_state.get("optimizer_context")
        if _stream_ctx and _stream_ctx.two_start_pitchers:
            # Build display from shared context (pre-computed, consistent with optimizer)
            _name_col = "player_name" if "player_name" in roster.columns else "name"
            _pid_name = (
                dict(zip(roster.get("player_id", []), roster.get(_name_col, [])))
                if "player_id" in roster.columns
                else {}
            )
            _pid_team = (
                dict(zip(roster.get("player_id", []), roster.get("team", []))) if "player_id" in roster.columns else {}
            )
            _pid_status = (
                dict(zip(roster.get("player_id", []), roster.get("status", [])))
                if "player_id" in roster.columns and "status" in roster.columns
                else {}
            )
            _remaining = _stream_ctx.remaining_games_this_week or {}
            for _ts_pid in _stream_ctx.two_start_pitchers:
                _ts_status = str(_pid_status.get(_ts_pid, "")).strip().lower()
                if _ts_status in _IL_EXCLUDE:
                    continue
                _ts_name = _pid_name.get(_ts_pid, f"Player {_ts_pid}")
                _ts_team = str(_pid_team.get(_ts_pid, ""))
                _ts_games = _remaining.get(_ts_team, 0)

                # Look up projected stats from player pool
                _ts_era = "--"
                _ts_k = "--"
                _ts_whip = "--"
                if not pool.empty:
                    _ts_match = pool[pool["player_id"] == _ts_pid]
                    if not _ts_match.empty:
                        _ts_row = _ts_match.iloc[0]
                        _era_val = float(_ts_row.get("era", 0) or 0)
                        _k_val = float(_ts_row.get("k", 0) or 0)
                        _whip_val = float(_ts_row.get("whip", 0) or 0)
                        _ts_era = format_stat(_era_val, "era") if _era_val > 0 else "--"
                        _ts_k = f"{_k_val:.0f}" if _k_val > 0 else "--"
                        _ts_whip = format_stat(_whip_val, "whip") if _whip_val > 0 else "--"

                # Get scheduled opponents
                _ts_opps = ""
                if _ts_name in _pitcher_schedule:
                    _ts_starts = _pitcher_schedule[_ts_name]
                    _opp_parts = []
                    for _s in sorted(_ts_starts, key=lambda x: x["date"]):
                        _loc = "vs" if _s["is_home"] else "@"
                        _opp_parts.append(f"{_s['date'][-5:]} {_loc} {_s['opponent']}")
                    _ts_opps = ", ".join(_opp_parts)

                two_start_sps.append(
                    {
                        "Pitcher": _ts_name,
                        "Team": _ts_team,
                        "Games": _ts_games,
                        "Proj ERA": _ts_era,
                        "Proj K": _ts_k,
                        "WHIP": _ts_whip,
                        "Matchups": _ts_opps if _ts_opps else "--",
                    }
                )
        else:
            # Fallback: compute inline from schedule
            TWO_START_THRESHOLD = 7
            for _, _r in roster.iterrows():
                if not _r.get("is_hitter", True) and "SP" in str(_r.get("positions", "")):
                    _row_status = str(_r.get("status", "")).strip().lower()
                    if _row_status in _IL_EXCLUDE:
                        continue
                    _r_team = str(_r.get("team", "")).strip()
                    _r_games = _stream_team_game_counts.get(_r_team, 0)
                    _r_name = str(_r.get("player_name", _r.get("name", ""))).strip()
                    if _r_games >= TWO_START_THRESHOLD:
                        # Look up projected stats
                        _r_era = "--"
                        _r_k = "--"
                        _r_whip = "--"
                        _r_pid = _r.get("player_id")
                        if not pool.empty and _r_pid is not None:
                            _r_match = pool[pool["player_id"] == _r_pid]
                            if not _r_match.empty:
                                _r_row = _r_match.iloc[0]
                                _era_v = float(_r_row.get("era", 0) or 0)
                                _k_v = float(_r_row.get("k", 0) or 0)
                                _whip_v = float(_r_row.get("whip", 0) or 0)
                                _r_era = format_stat(_era_v, "era") if _era_v > 0 else "--"
                                _r_k = f"{_k_v:.0f}" if _k_v > 0 else "--"
                                _r_whip = format_stat(_whip_v, "whip") if _whip_v > 0 else "--"

                        # Get scheduled matchups
                        _r_opps = ""
                        if _r_name in _pitcher_schedule:
                            _r_starts = _pitcher_schedule[_r_name]
                            _opp_parts = []
                            for _s in sorted(_r_starts, key=lambda x: x["date"]):
                                _loc = "vs" if _s["is_home"] else "@"
                                _opp_parts.append(f"{_s['date'][-5:]} {_loc} {_s['opponent']}")
                            _r_opps = ", ".join(_opp_parts)

                        two_start_sps.append(
                            {
                                "Pitcher": _r_name,
                                "Team": _r_team,
                                "Games": _r_games,
                                "Proj ERA": _r_era,
                                "Proj K": _r_k,
                                "WHIP": _r_whip,
                                "Matchups": _r_opps if _r_opps else "--",
                            }
                        )

        if two_start_sps:
            st.markdown(
                f"{PAGE_ICONS['trending_up']} **{len(two_start_sps)} potential two-start pitcher(s) this week:**",
                unsafe_allow_html=True,
            )
            render_styled_table(pd.DataFrame(two_start_sps))
            st.caption(
                "Starting pitchers on teams with 7+ games this week likely get two starts. "
                "Two starts double counting stat contributions (K, W) but also double "
                "rate-stat exposure (ERA, WHIP)."
            )
        else:
            st.info("No two-start starting pitchers detected for your roster this week.")

        # ──────────────────────────────────────────────────────────────
        # SECTION 3: Weekly Streaming Calendar
        # ──────────────────────────────────────────────────────────────
        if _stream_picks and _stream_schedule:
            st.divider()
            st.markdown(
                f'<p style="font-size:12px;font-weight:700;letter-spacing:1px;'
                f"color:{T['tx2']};text-transform:uppercase;"
                f'margin:0 0 6px;">Weekly Streaming Calendar</p>',
                unsafe_allow_html=True,
            )

            # Collect all unique dates in the schedule window
            _all_dates = sorted({g["game_date"] for g in _stream_schedule if g["game_date"]})
            _all_dates = _all_dates[:7]  # Cap at 7 days

            if _all_dates:
                # Build calendar: top streaming picks with their start dates
                _cal_pitchers = [p for p in _stream_picks if p["num_starts"] > 0 and "SP" in p.get("type", "")][:10]

                if _cal_pitchers:
                    _cal_rows = []
                    for _cp in _cal_pitchers:
                        _cp_name = _cp["player_name"]
                        _cp_row: dict = {"Pitcher": _cp_name, "Team": _cp["team"]}

                        # Map this pitcher's starts to dates
                        _cp_starts = _pitcher_schedule.get(_cp_name, [])
                        _cp_start_map: dict[str, str] = {}
                        for _s in _cp_starts:
                            _loc = "vs" if _s["is_home"] else "@"
                            _cp_start_map[_s["date"]] = f"{_loc} {_s['opponent']}"

                        for _d in _all_dates:
                            _day_label = _d[-5:]  # MM-DD
                            if _d in _cp_start_map:
                                _cp_row[_day_label] = _cp_start_map[_d]
                            else:
                                _cp_row[_day_label] = ""

                        _cal_rows.append(_cp_row)

                    if _cal_rows:
                        render_styled_table(pd.DataFrame(_cal_rows))
                        st.caption(
                            "Shows scheduled starts for top streaming SP candidates. "
                            "Blank cells = no start that day. "
                            "'vs' = home, '@' = away."
                        )
                else:
                    st.caption("No probable starters identified in schedule for streaming candidates.")

page_timer_footer("Lineup")
