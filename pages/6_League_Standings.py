"""League Standings -- Live H2H standings, MC season projections, and team strength."""

from __future__ import annotations

import html as _html
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.ai.chat import render_chat_widget
from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import (
    init_db,
    load_league_records,
    load_league_schedule_full,
    load_player_pool,
)
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.standings_utils import (
    close_battle_categories,
    filter_standings_to_valid_teams,
    get_all_team_totals,
)
from src.ui_shared import (
    THEME,
    build_compact_table_html,
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
    render_data_freshness_card,
    render_empty_state,
    render_matchup_ticker,
    render_page_header,
    render_reco_banner,
)
from src.usage import log_page_view
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

logger = logging.getLogger(__name__)

T = THEME

# ── Optional imports ──────────────────────────────────────────────────

try:
    from src.standings_engine import (
        compute_magic_numbers,
        compute_team_strength_profiles,
        simulate_season_enhanced,
    )

    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

try:
    from src.standings_projection import simulate_season

    LEGACY_SIM_AVAILABLE = True
except ImportError:
    LEGACY_SIM_AVAILABLE = False

try:
    from src.power_rankings import compute_power_rankings  # noqa: F401

    POWER_RANKINGS_AVAILABLE = True
except ImportError:
    POWER_RANKINGS_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────

# 2026-05-19 D6: snapshot rate_stats from LeagueConfig.
# 2026-05-19 D4: snapshot inverse_stats from LeagueConfig.
# MS-C1 (2026-06-07): total weeks sourced from LeagueConfig (canonical 26),
# mirroring pages/2_Line-up_Optimizer.py's _LC_W().season_weeks pattern.
from src.valuation import LeagueConfig as _LC_FOR_CATS  # noqa: E402

_SEASON_START = datetime(2026, 3, 25, tzinfo=UTC)
_TOTAL_WEEKS = _LC_FOR_CATS().season_weeks
_PLAYOFF_SPOTS = 4

_HIT_CATS = {"R", "HR", "RBI", "SB", "AVG", "OBP"}
_PIT_CATS = {"W", "L", "SV", "K", "ERA", "WHIP"}

_RATE_STATS = set(_LC_FOR_CATS().rate_stats)
_INVERSE_CATS = set(_LC_FOR_CATS().inverse_stats)
_CAT_ORDER = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]


# ── Helper: user team name ────────────────────────────────────────────


def _get_user_team_name(rosters) -> str | None:
    """The current viewer's team — canonical resolver (auth-aware under MULTI_USER).

    Pass the standings/roster frame so a Yahoo team name carrying a leading
    emoji/whitespace reconciles to the env-seeded admin assignment (e.g.
    "Team Hickey"); a no-arg resolve skips reconciliation and breaks every
    downstream ``== user_team`` match -- the 2026-06-05 "Your Position: Team not
    found in standings" bug. Required arg so a frame-less call fails loudly.
    """
    return resolve_viewer_team_name(rosters)


# ── Helper: ordinal suffix ────────────────────────────────────────────


def _ordinal(n: int) -> str:
    """Return ordinal string for a rank (1 -> '1st', 2 -> '2nd', etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Helper: compute projected team totals from rosters ────────────────


def _compute_projected_team_totals() -> dict[str, dict[str, float]]:
    """Compute projected weekly category averages per team.

    Adapter around the canonical ``standings_utils.get_all_team_totals`` so
    downstream consumers (``simulate_season_enhanced``, ``compute_team_strength_profiles``)
    keep their per-week-totals contract. Counting stats are scaled by 1/26 to
    convert season totals to per-matchup-week averages; rate stats (AVG/OBP/ERA/WHIP)
    are passed through unchanged.

    When Yahoo standings are connected, season-to-date totals are projected to
    a 26-week pace using ``current_week / weeks_played`` as a proxy for full-season
    rate; otherwise the projection-based fallback fires using full-season blended
    projections from the ``projections`` table.
    """
    # Build league_rosters dict {team_name: [player_ids]} for the projection fallback.
    _yds = get_yahoo_data_service()
    rosters_df = _yds.get_rosters()
    league_rosters_dict: dict[str, list[int]] = {}
    if not rosters_df.empty and "team_name" in rosters_df.columns and "player_id" in rosters_df.columns:
        for team, group in rosters_df.groupby("team_name"):
            pids = [int(pid) for pid in group["player_id"].dropna().tolist()]
            league_rosters_dict[str(team)] = pids

    pool = load_player_pool()
    if not pool.empty and "name" in pool.columns:
        pool = pool.rename(columns={"name": "player_name"})

    season_totals = get_all_team_totals(
        league_rosters=league_rosters_dict if league_rosters_dict else None,
        player_pool=pool if not pool.empty else None,
    )

    if not season_totals:
        return {}

    weeks = 26.0
    counting_cats = {"R", "HR", "RBI", "SB", "W", "L", "SV", "K"}
    # 2026-05-19 D6: reuse module-level _RATE_STATS (already snapshot from LeagueConfig).
    rate_cats = _RATE_STATS
    weekly_totals: dict[str, dict[str, float]] = {}
    for team, cat_map in season_totals.items():
        per_week: dict[str, float] = {}
        for cat, val in cat_map.items():
            if cat in counting_cats:
                per_week[cat] = float(val) / weeks
            elif cat in rate_cats:
                per_week[cat] = float(val)
        # Backfill defaults if a rate stat is missing
        per_week.setdefault("AVG", 0.250)
        per_week.setdefault("OBP", 0.330)
        per_week.setdefault("ERA", 4.00)
        per_week.setdefault("WHIP", 1.25)
        weekly_totals[str(team)] = per_week

    return weekly_totals


# ── Helper: build reco banner text ────────────────────────────────────


def _build_banner_teaser(matchup: dict | None) -> str:
    """Build a one-line recommendation banner from matchup data."""
    if not matchup:
        return "Connect your Yahoo league to see live matchup analysis."
    opp = matchup.get("opp_name", "opponent")
    w = matchup.get("wins", 0)
    l = matchup.get("losses", 0)
    t = matchup.get("ties", 0)
    cats = matchup.get("categories", [])

    # Close categories (small margin), excluding any unnamed entries so the
    # banner never renders "Close battles: , , ." (2026-06-05 cosmetic fix).
    close_cats = close_battle_categories(cats)

    teaser = f"This week vs {opp}: "
    if w > l:
        teaser += f"leading {w}-{l}-{t} in categories."
    elif l > w:
        teaser += f"trailing {w}-{l}-{t} in categories."
    else:
        teaser += f"tied {w}-{l}-{t} in categories."

    if close_cats:
        teaser += f" Close battles: {', '.join(close_cats[:3])}."

    return teaser


# ── Helper: rank badge HTML ───────────────────────────────────────────


def _rank_badge(rank: int) -> str:
    """Return an HTML span with color-coded rank badge."""
    if 1 <= rank <= 4:
        bg = T["green"]
        color = T["ink"]
    elif 5 <= rank <= 8:
        bg = T["sky"]
        color = T["ink"]
    else:
        bg = T["danger"]
        color = T["ink"]
    return (
        f'<span style="display:inline-block;min-width:24px;padding:2px 6px;'
        f"border-radius:4px;background:{bg} !important;color:{color} !important;"
        f'font-weight:600;font-size:11px;text-align:center;">{rank}</span>'
    )


# ══════════════════════════════════════════════════════════════════════
# Page Setup
# ══════════════════════════════════════════════════════════════════════

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | League Standings",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

init_db()
inject_custom_css()
require_auth()
require_page_enabled("page:6_League_Standings")
log_page_view("League Standings")
render_chat_widget("League Standings")
page_timer_start()

# ── Data loading ──────────────────────────────────────────────────────

yds = get_yahoo_data_service()
matchup = yds.get_matchup()
# DB-only: Yahoo API surface doesn't expose historical league records.
records_df = load_league_records()
user_team = _get_user_team_name(records_df)

# ── Banner ────────────────────────────────────────────────────────────

banner_teaser = _build_banner_teaser(matchup)
render_page_header(
    "League Standings",
    eyebrow="LEAGUE",
    fig="FIG.06 — STANDINGS BOARD",
)
render_reco_banner(banner_teaser, "", "league_standings")
render_matchup_ticker()


def _section_label(text: str, *, fig: str = "") -> None:
    """Render an instrument-style eyebrow section label (orange bar + Archivo)."""
    fig_html = ""
    if fig:
        fig_html = (
            '<span style="font-family:var(--font-mono);font-weight:500;font-size:10px;'
            f'letter-spacing:.12em;color:var(--fp-tx-muted);">{_html.escape(str(fig))}</span>'
        )
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'margin:6px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--fp-divider);">'
        '<div style="display:flex;align-items:center;gap:9px;">'
        '<span style="width:3px;height:14px;background:var(--fp-primary);border-radius:2px;'
        'box-shadow:0 0 8px rgba(255,109,0,.5);"></span>'
        f"{build_eyebrow_html(text)}</div>{fig_html}</div>",
        unsafe_allow_html=True,
    )


# ── Layout ────────────────────────────────────────────────────────────

ctx, main = render_context_columns()

# ── Context panel ─────────────────────────────────────────────────────

with ctx:
    # Card 1: YOUR POSITION
    if not records_df.empty and user_team:
        user_row = records_df[records_df["team_name"] == user_team]
        if not user_row.empty:
            u = user_row.iloc[0]
            rank = int(u.get("rank", 0))
            wins = int(u.get("wins", 0))
            losses = int(u.get("losses", 0))
            ties = int(u.get("ties", 0))
            win_pct = float(u.get("win_pct", 0.0))

            # Games back from 1st
            first_row = records_df[records_df["rank"] == 1]
            if not first_row.empty:
                first_wins = int(first_row.iloc[0]["wins"])
                gb_first = first_wins - wins
            else:
                gb_first = 0

            # Games back from playoff line (4th) or ahead of 5th
            if rank <= _PLAYOFF_SPOTS:
                fifth_row = records_df[records_df["rank"] == _PLAYOFF_SPOTS + 1]
                if not fifth_row.empty:
                    fifth_wins = int(fifth_row.iloc[0]["wins"])
                    gb_playoff = wins - fifth_wins
                    playoff_label = f"+{gb_playoff} ahead of {_ordinal(_PLAYOFF_SPOTS + 1)}"
                else:
                    playoff_label = "In playoff position"
            else:
                fourth_row = records_df[records_df["rank"] == _PLAYOFF_SPOTS]
                if not fourth_row.empty:
                    fourth_wins = int(fourth_row.iloc[0]["wins"])
                    gb_playoff_val = fourth_wins - wins
                    playoff_label = f"{gb_playoff_val} GB from {_ordinal(_PLAYOFF_SPOTS)}"
                else:
                    playoff_label = "Outside playoffs"

            # Streak chip — movement signal from existing records data (hot =
            # winning streak, steel = losing streak). Text only, no arrows.
            _streak = str(u.get("streak", "") or "").strip()
            _streak_chip_html = ""
            if _streak:
                _streak_cls = "hot" if _streak.upper().startswith("W") else "cold"
                _streak_chip_html = (
                    f'<p style="margin:4px 0 2px;"><span class="chip {_streak_cls}">'
                    f"{_html.escape(_streak)} STREAK</span></p>"
                )

            render_context_card(
                "YOUR POSITION",
                f'<p class="hero-num" style="font-size:28px;margin:0;">'
                f"{_ordinal(rank)}</p>"
                f'<p style="margin:4px 0;font-size:13px;color:{T["tx2"]};">'
                f"{wins}-{losses}-{ties} ({win_pct:.3f})</p>"
                f"{_streak_chip_html}"
                f'<p style="margin:2px 0;font-size:12px;color:{T["tx2"]};">'
                f"GB from 1st: {gb_first}</p>"
                f'<p style="margin:2px 0;font-size:12px;color:{T["tx2"]};">'
                f"{playoff_label}</p>",
            )
        else:
            render_context_card(
                "YOUR POSITION",
                '<p style="color:var(--fp-tx-muted);font-size:12px;">Team not found in standings.</p>',
            )
    else:
        render_context_card(
            "YOUR POSITION",
            '<p style="color:var(--fp-tx-muted);font-size:12px;">Connect Yahoo to see your position.</p>',
        )

    # Card 2: THIS WEEK
    if matchup:
        opp = matchup.get("opp_name", "Unknown")
        mw = matchup.get("wins", 0)
        ml = matchup.get("losses", 0)
        mt = matchup.get("ties", 0)
        if mw > ml:
            score_color = T["green"]
        elif ml > mw:
            score_color = T["danger"]
        else:
            score_color = T["tx2"]
        render_context_card(
            "THIS WEEK",
            f'<p style="margin:0;font-size:13px;color:{T["tx2"]};">vs {opp}</p>'
            f'<p style="font-size:20px;font-weight:700;margin:4px 0;color:{score_color};">'
            f"{mw}-{ml}-{mt}</p>",
        )
    else:
        render_context_card(
            "THIS WEEK", '<p style="color:var(--fp-tx-muted);font-size:12px;">No matchup data available.</p>'
        )

    # Card 3: PLAYOFF ODDS
    sim_result = st.session_state.get("standings_sim_result")
    if sim_result and user_team:
        playoff_pct = sim_result.get("playoff_probability", {}).get(user_team, 0.0)
        magic_nums = sim_result.get("magic_numbers", {})
        magic = magic_nums.get(user_team) if magic_nums else None
        if magic == 0:
            magic_label = "CLINCHED"
        elif magic is None:
            magic_label = "--"
        else:
            magic_label = str(magic)

        pct_color = T["green"] if playoff_pct >= 0.5 else T["danger"]
        render_context_card(
            "PLAYOFF ODDS",
            f'<p style="font-size:22px;font-weight:700;margin:0;color:{pct_color};">'
            f"{playoff_pct * 100:.0f}%</p>"
            f'<p style="margin:4px 0;font-size:12px;color:{T["tx2"]};">'
            f"Magic number: {magic_label}</p>",
        )
    else:
        render_context_card(
            "PLAYOFF ODDS",
            '<p style="color:var(--fp-tx-muted);font-size:12px;">Run projections to see playoff odds.</p>',
        )

    # Card 4: Data freshness
    render_data_freshness_card()


# ══════════════════════════════════════════════════════════════════════
# Main Content — Tabs
# ══════════════════════════════════════════════════════════════════════


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for a number (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        return "th"
    last = n % 10
    if last == 1:
        return "st"
    if last == 2:
        return "nd"
    if last == 3:
        return "rd"
    return "th"


def _render_playoff_odds_tab() -> None:
    """Render the playoff-odds Monte Carlo simulator (was Playoff_Odds page).

    2026-05-18 Section 5: merged from the standalone Playoff_Odds page.
    Self-contained — loads its own data via cached helpers.
    """
    import time as _time

    from src.database import coerce_numeric_df as _coerce

    try:
        from src.playoff_sim import (
            simulate_season as _simulate_season_playoffs,
        )
    except ImportError:
        st.error("Playoff simulation module not available (src/playoff_sim.py missing).")
        return

    _po_pool = load_player_pool()
    if _po_pool.empty:
        st.warning("No player pool loaded.")
        return
    _po_pool = _po_pool.rename(columns={"name": "player_name"})
    _po_config = LeagueConfig()
    _po_yds = get_yahoo_data_service()

    _po_standings = _po_yds.get_standings()
    _po_all_team_totals: dict[str, dict[str, float]] = {}
    if not _po_standings.empty and "category" in _po_standings.columns:
        for _, srow in _po_standings.iterrows():
            tname = str(srow["team_name"])
            cat = str(srow["category"]).strip()
            _po_all_team_totals.setdefault(tname, {})[cat] = float(srow.get("total", 0) or 0)

    _po_rosters = _po_yds.get_rosters()
    if _po_rosters.empty:
        render_empty_state("No league rosters loaded", "Connect your Yahoo league first.", icon_key="users")
        return
    user_team_name = resolve_viewer_team_name(_po_rosters)
    if not user_team_name:
        render_empty_state("No user team identified in roster data", icon_key="users")
        return
    user_team_name = str(user_team_name)

    _po_rosters = _coerce(_po_rosters)
    league_rosters_dict: dict[str, list[int]] = {}
    for _, row in _po_rosters.iterrows():
        team = str(row["team_name"])
        pid = int(row["player_id"])
        league_rosters_dict.setdefault(team, []).append(pid)

    all_team_totals = _po_all_team_totals
    pool = _po_pool
    config = _po_config

    if not all_team_totals:
        render_empty_state(
            "No standings data loaded",
            "Connect your Yahoo league or load standings data to run playoff odds.",
            icon_key="playoff_odds",
        )
        return

    team_names_p = sorted(league_rosters_dict.keys())
    if len(team_names_p) < 2:
        st.warning("Need at least 2 teams with roster data.")
        return

    # 2026-05-19 Section 5: use canonical weeks_remaining (replaces
    # estimate_weeks_remaining from playoff_sim — same semantic, single source).
    from src.league_rules import weeks_remaining as _weeks_remaining

    weeks_default = _weeks_remaining()
    ctrl1, ctrl2, _ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        weeks_remaining_p = st.number_input(
            "Weeks remaining",
            min_value=1,
            max_value=config.season_weeks,
            value=weeks_default,
            step=1,
            key="playoff_weeks_remaining",
        )
    with ctrl2:
        n_sims_p = st.selectbox(
            "Simulations",
            options=[200, 500, 1000, 2000],
            index=1,
            format_func=lambda x: f"{x:,}",
            key="playoff_n_sims",
        )

    if st.button("Simulate Season", type="primary", key="playoff_simulate_btn"):
        progress_bar = st.progress(0, text="Running playoff simulations...")

        def _update_progress(frac: float) -> None:
            pct = int(frac * 100)
            progress_bar.progress(frac, text=f"Simulating season... {pct}%")

        start_time = _time.time()
        try:
            results_p = _simulate_season_playoffs(
                all_team_totals=all_team_totals,
                league_rosters=league_rosters_dict,
                player_pool=pool,
                weeks_remaining=int(weeks_remaining_p),
                n_sims=int(n_sims_p),
                config=config,
                on_progress=_update_progress,
            )
            elapsed = _time.time() - start_time
            progress_bar.progress(1.0, text=f"Complete in {elapsed:.1f}s")
            _time.sleep(0.3)
            progress_bar.empty()
            st.session_state["playoff_sim_results"] = results_p
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Simulation failed: {exc}")

    results_p = st.session_state.get("playoff_sim_results")
    if results_p is None:
        render_empty_state(
            "No simulation yet",
            "Press Simulate Season to run the playoff odds simulator.",
            icon_key="playoff_odds",
        )
        return
    if not results_p:
        st.warning("Simulation returned no results. Check that roster data is available for all teams.")
        return

    user_result = results_p.get(user_team_name)
    if user_result:
        prob_pct = user_result["playoff_prob"] * 100.0
        prob_color = (
            T["green"]
            if prob_pct >= 60
            else T.get("hot", T["primary"])
            if prob_pct >= 40
            else T.get("primary", T["primary"])
        )
        prob_label = "Strong" if prob_pct >= 60 else "Moderate" if prob_pct >= 40 else "At Risk"
        _po_hero_body = (
            f'<div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;">'
            f"<div>"
            f'<div style="font-family:var(--font-display);font-size:56px;font-weight:800;'
            f'font-variant-numeric:tabular-nums;color:{prob_color};line-height:1;">{prob_pct:.1f}%</div>'
            f'<div style="font-family:var(--font-body);font-size:14px;color:{T["tx2"]};margin-top:4px;">'
            f"{prob_label} -- {user_team_name}</div></div>"
            f'<div style="margin-left:auto;text-align:right;">'
            f'<div class="t-label">Projected Record</div>'
            f'<div style="font-family:var(--font-display);font-size:28px;font-weight:700;'
            f'font-variant-numeric:tabular-nums;color:{T["tx"]};">'
            f"{user_result['avg_wins']:.1f} - {user_result['avg_losses']:.1f}</div></div></div>"
        )
        st.markdown(
            build_panel_html(
                "Your Playoff Probability",
                _po_hero_body,
                fig_label="FIG.03 · PLAYOFF ODDS",
                accent="top",
            ),
            unsafe_allow_html=True,
        )

    # Projected standings table
    rows = []
    for team in team_names_p:
        r = results_p.get(team)
        if r is None:
            continue
        rank_dist = r["rank_distribution"]
        most_likely_rank = rank_dist.index(max(rank_dist)) + 1
        rows.append(
            {
                "Team": team,
                "Avg Wins": r["avg_wins"],
                "Avg Losses": r["avg_losses"],
                "Playoff %": round(r["playoff_prob"] * 100, 1),
                "Most Likely Finish": most_likely_rank,
            }
        )
    if rows:
        standings_p = pd.DataFrame(rows).sort_values("Playoff %", ascending=False).reset_index(drop=True)
        standings_p.index = standings_p.index + 1
        standings_p.index.name = "Rank"

        def _highlight_user(row: pd.Series) -> list[str]:
            if row["Team"] == user_team_name:
                return ["background-color: rgba(255, 109, 0, 0.08); font-weight: 700;"] * len(row)
            return [""] * len(row)

        styled = standings_p.style.apply(_highlight_user, axis=1).format(
            {"Playoff %": "{:.1f}%", "Avg Wins": "{:.1f}", "Avg Losses": "{:.1f}"}
        )
        st.dataframe(styled, width="stretch", height=min(460, 38 + 35 * len(standings_p)))

    # Finish distribution for user
    if user_result:
        rank_dist = user_result["rank_distribution"]
        total_sims = sum(rank_dist) or 1
        dist_rows = []
        for idx, count in enumerate(rank_dist):
            rank = idx + 1
            pct = count / total_sims * 100
            if pct < 0.5:
                continue
            suffix = _ordinal_suffix(rank)
            dist_rows.append({"Finish": f"{rank}{suffix}", "Probability": f"{pct:.1f}%", "Simulations": count})
        if dist_rows:
            st.markdown("**Your Finish Distribution**")
            render_compact_table(pd.DataFrame(dist_rows))


with main:
    # 2026-05-18 Section 5: Playoff Odds page folded in as 3rd tab.
    tab1, tab2, tab_playoffs = st.tabs(["Current Standings", "Season Projections", "Playoff Odds"])

    # ── Tab 1: Current Standings ──────────────────────────────────────
    with tab1:
        has_records = not records_df.empty
        if has_records:
            # Section A: H2H Record Table (instrument panel, FIG.01)
            display_rows: list[dict] = []
            for _, row in records_df.iterrows():
                tn = str(row["team_name"])
                w = int(row.get("wins", 0))
                l = int(row.get("losses", 0))
                t = int(row.get("ties", 0))
                wp = float(row.get("win_pct", 0.0))
                streak = str(row.get("streak", "") or "")
                rk = int(row.get("rank", 0))

                # Games back from 1st
                first_w = (
                    int(records_df[records_df["rank"] == 1].iloc[0]["wins"])
                    if not records_df[records_df["rank"] == 1].empty
                    else w
                )
                gb = first_w - w
                gb_str = "--" if gb == 0 else str(gb)

                display_rows.append(
                    {
                        "Rank": rk,
                        "Team": tn,
                        "W": w,
                        "L": l,
                        "T": t,
                        "Win%": format_stat(wp, "AVG"),
                        "GB": gb_str,
                        "Streak": streak,
                    }
                )

            record_df = pd.DataFrame(display_rows)

            # Build custom row classes: highlight user team, playoff cutoff
            row_classes: dict[int, str] = {}
            for idx, r in enumerate(display_rows):
                if user_team and r["Team"] == user_team:
                    row_classes[idx] = "row-start"  # user team highlight

            # Build HTML with custom playoff cutoff line
            html = build_compact_table_html(record_df, row_classes=row_classes)

            # Inject dashed red border between rank 4 and 5 (playoff cutoff)
            # We inject CSS targeting the table row
            cutoff_css = f"""
            <style>
            .standings-cutoff tr:nth-child({_PLAYOFF_SPOTS + 1}) td {{
                border-top: 2px dashed {T["danger"]} !important;
            }}
            </style>
            """
            html = html.replace(
                'class="compact-table-wrap"',
                'class="compact-table-wrap standings-cutoff"',
            )
            st.markdown(
                cutoff_css
                + build_panel_html(
                    "Head-to-Head Record",
                    html,
                    fig_label="FIG.01 · CURRENT STANDINGS",
                    accent="top",
                ),
                unsafe_allow_html=True,
            )

        # Section B: Category Standings Grid
        standings_df = yds.get_standings()
        if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
            # Filter to scoring categories only
            config = LeagueConfig()
            scoring_cats = set(c.upper() for c in config.all_categories)
            cat_standings = standings_df[standings_df["category"].str.upper().isin(scoring_cats)].copy()

            # MS-C5: drop ghost teams (present in the standings cache but not on
            # current rosters) so per-category ranks never include a renamed/
            # abandoned team. Rosters unavailable -> no filtering.
            _rosters_for_filter = yds.get_rosters()
            _valid_teams = (
                set(_rosters_for_filter["team_name"].astype(str).unique())
                if not _rosters_for_filter.empty and "team_name" in _rosters_for_filter.columns
                else None
            )
            cat_standings = filter_standings_to_valid_teams(cat_standings, _valid_teams)

            if not cat_standings.empty:
                _section_label("Category Standings", fig="12 CAT · RANKS")

                # Compute per-category ranks: for each category rank teams by total
                # (inverse stats: lower is better)
                teams_in_standings = sorted(cat_standings["team_name"].unique().tolist())
                cat_rank_data: dict[str, dict[str, int]] = {}  # {team: {cat: rank}}

                for cat in _CAT_ORDER:
                    cat_rows = cat_standings[cat_standings["category"].str.upper() == cat]
                    if cat_rows.empty:
                        continue
                    cat_vals = []
                    for _, r in cat_rows.iterrows():
                        cat_vals.append((str(r["team_name"]), float(r.get("total", 0.0) or 0.0)))

                    # Sort: inverse stats ascending (lower=better), others descending
                    if cat in _INVERSE_CATS:
                        cat_vals.sort(key=lambda x: x[1])
                    else:
                        cat_vals.sort(key=lambda x: -x[1])

                    for rank_idx, (tn, _) in enumerate(cat_vals, start=1):
                        cat_rank_data.setdefault(tn, {})[cat] = rank_idx

                # Build display grid with rank badges
                grid_rows: list[dict] = []
                for tn in teams_in_standings:
                    row_data: dict[str, str] = {"Team": tn}
                    for cat in _CAT_ORDER:
                        rank_val = cat_rank_data.get(tn, {}).get(cat, 0)
                        if rank_val > 0:
                            row_data[cat] = _rank_badge(rank_val)
                        else:
                            row_data[cat] = "--"
                    grid_rows.append(row_data)

                grid_df = pd.DataFrame(grid_rows)

                # Build highlight_cols: hit categories with th-hit, pitch with th-pit
                highlight_cols: dict[str, str] = {}
                for cat in _CAT_ORDER:
                    if cat in _HIT_CATS:
                        highlight_cols[cat] = "th-hit"
                    elif cat in _PIT_CATS:
                        highlight_cols[cat] = "th-pit"

                # Row classes: highlight user team
                grid_row_classes: dict[int, str] = {}
                for idx, r in enumerate(grid_rows):
                    if user_team and r["Team"] == user_team:
                        grid_row_classes[idx] = "row-start"

                grid_html = build_compact_table_html(
                    grid_df,
                    highlight_cols=highlight_cols,
                    row_classes=grid_row_classes,
                    html_cols=set(_CAT_ORDER),
                )
                st.markdown(grid_html, unsafe_allow_html=True)

        elif not has_records:
            render_empty_state(
                "No standings yet",
                "Connect your Yahoo league to see live standings. Run the app bootstrap first.",
                icon_key="league_standings",
            )

    # ── Tab 2: Season Projections ─────────────────────────────────────
    with tab2:
        # Auto-run simulation on page load if not cached
        _sim_cache_key = "standings_sim_result"
        _sim_ran = _sim_cache_key in st.session_state

        # Compute current week
        _weeks_played = max(1.0, (datetime.now(UTC) - _SEASON_START).days / 7.0)
        _current_week = max(1, int(_weeks_played))

        if not _sim_ran:
            # Try to get data and run simulation automatically
            team_totals = _compute_projected_team_totals()

            # Get current W-L-T records for enhanced simulation
            current_standings: dict[str, dict[str, int]] = {}
            if not records_df.empty:
                for _, row in records_df.iterrows():
                    tn = str(row["team_name"])
                    current_standings[tn] = {
                        "W": int(row.get("wins", 0)),
                        "L": int(row.get("losses", 0)),
                        "T": int(row.get("ties", 0)),
                    }

            # Get full schedule
            # DB-only: Yahoo API surface doesn't expose the full historical schedule.
            full_schedule = load_league_schedule_full()
            if not full_schedule:
                # Try fetching from Yahoo
                try:
                    full_schedule = yds.get_full_league_schedule()
                except Exception:
                    full_schedule = {}

            if team_totals and ENGINE_AVAILABLE:
                # Use enhanced simulation with actual schedule
                with st.spinner("Simulating remaining season…"):
                    try:
                        sim_result = simulate_season_enhanced(
                            current_standings=current_standings
                            if current_standings
                            else {t: {"W": 0, "L": 0, "T": 0} for t in team_totals},
                            team_weekly_totals=team_totals,
                            full_schedule=full_schedule
                            if full_schedule
                            else {
                                w: [
                                    (list(team_totals.keys())[i], list(team_totals.keys())[(i + w) % len(team_totals)])
                                    for i in range(0, len(team_totals), 2)
                                ]
                                for w in range(1, _TOTAL_WEEKS + 1)
                            },
                            current_week=_current_week,
                            n_sims=500,
                            seed=42,
                            playoff_spots=_PLAYOFF_SPOTS,
                        )
                        # Add magic numbers
                        remaining = _TOTAL_WEEKS - _current_week + 1
                        current_wins = {t: rec["W"] for t, rec in (current_standings or {}).items()}
                        if not current_wins and team_totals:
                            current_wins = {t: 0 for t in team_totals}
                        magic_nums = compute_magic_numbers(
                            current_wins=current_wins,
                            remaining_matchups=remaining,
                            playoff_spots=_PLAYOFF_SPOTS,
                        )
                        sim_result["magic_numbers"] = magic_nums
                        st.session_state[_sim_cache_key] = sim_result
                    except Exception as e:
                        logger.error("Enhanced simulation failed: %s", e, exc_info=True)
                        st.warning(f"Season simulation encountered an issue: {e}")

            elif team_totals and LEGACY_SIM_AVAILABLE:
                # Fallback to legacy simulation
                with st.spinner("Simulating remaining season…"):
                    try:
                        legacy_df = simulate_season(team_totals, n_sims=500, seed=42)
                        st.session_state["standings_legacy_result"] = legacy_df
                    except Exception as e:
                        st.warning(f"Legacy simulation failed: {e}")

        # Display simulation results
        sim_result = st.session_state.get(_sim_cache_key)

        if sim_result:
            # Lead section: projected standings table (instrument panel, FIG.02)
            proj_records = sim_result.get("projected_records", {})
            playoff_probs = sim_result.get("playoff_probability", {})
            magic_nums = sim_result.get("magic_numbers", {})
            sos_data = sim_result.get("strength_of_schedule", {})
            ci_data = sim_result.get("confidence_intervals", {})

            # Sort by projected wins descending
            sorted_teams = sorted(
                proj_records.keys(),
                key=lambda t: proj_records[t].get("win_pct", 0),
                reverse=True,
            )

            proj_rows: list[dict] = []
            for rank_idx, tn in enumerate(sorted_teams, start=1):
                rec = proj_records[tn]
                pp = playoff_probs.get(tn, 0.0)
                mn = magic_nums.get(tn)
                ss = sos_data.get(tn, 0.5)

                if mn == 0:
                    magic_str = "CLINCHED"
                elif mn is None:
                    magic_str = "--"
                else:
                    magic_str = str(mn)

                proj_rows.append(
                    {
                        "Rank": rank_idx,
                        "Team": tn,
                        "Proj W": rec.get("W", 0),
                        "Proj L": rec.get("L", 0),
                        "Proj T": rec.get("T", 0),
                        "Win%": format_stat(rec.get("win_pct", 0), "AVG"),
                        "Playoff%": f"{pp * 100:.0f}%",
                        "Magic#": magic_str,
                        "SOS": f"{ss:.3f}",
                    }
                )

            proj_df = pd.DataFrame(proj_rows)

            # Row classes for user highlight and playoff cutoff
            proj_row_classes: dict[int, str] = {}
            for idx, r in enumerate(proj_rows):
                if user_team and r["Team"] == user_team:
                    proj_row_classes[idx] = "row-start"

            proj_html = build_compact_table_html(proj_df, row_classes=proj_row_classes)

            # Playoff cutoff line
            proj_cutoff_css = f"""
            <style>
            .proj-cutoff tr:nth-child({_PLAYOFF_SPOTS + 1}) td {{
                border-top: 2px dashed {T["danger"]} !important;
            }}
            </style>
            """
            proj_html = proj_html.replace(
                'class="compact-table-wrap"',
                'class="compact-table-wrap proj-cutoff"',
            )
            st.markdown(
                proj_cutoff_css
                + build_panel_html(
                    "Projected Final Standings",
                    proj_html,
                    fig_label="FIG.02 · SEASON PROJECTIONS",
                    accent="top",
                ),
                unsafe_allow_html=True,
            )

            # ── Playoff-odds heat-bar strip (instrument panel) ──
            # A dossier-style visualization of each team's MC playoff probability:
            # orange fill above the 4-spot line, steel below. The user's team is
            # accent-highlighted. Plays the role the escape-safe table can't.
            _odds_rows = ""
            for tn in sorted_teams:
                _pp = float(playoff_probs.get(tn, 0.0))
                _pct = max(0.0, min(100.0, _pp * 100.0))
                _is_user = bool(user_team and tn == user_team)
                _name_color = "var(--fp-primary)" if _is_user else "var(--fp-tx)"
                _pct_color = "var(--fp-primary)" if _pp >= 0.5 else "var(--fp-tx-muted)"
                _row_bg = "background:rgba(255,109,0,.06);" if _is_user else ""
                _odds_rows += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:5px 6px;'
                    f'border-bottom:1px solid var(--fp-divider);{_row_bg}border-radius:4px;">'
                    f'<div style="width:150px;flex-shrink:0;font-family:var(--font-display);'
                    f"font-weight:{'800' if _is_user else '700'};font-size:12px;color:{_name_color};"
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_html.escape(str(tn))}</div>'
                    f'<div style="flex:1;">{build_heatbar_html(_pct, win=(_pp >= 0.5))}</div>'
                    f'<div style="width:42px;flex-shrink:0;text-align:right;font-family:var(--font-mono);'
                    f'font-weight:600;font-size:12px;color:{_pct_color};">{_pct:.0f}%</div>'
                    "</div>"
                )
            st.markdown(
                build_panel_html("Playoff Odds", _odds_rows, fig_label=f"TOP {_PLAYOFF_SPOTS} ADVANCE"),
                unsafe_allow_html=True,
            )

            # Section B: Team Strength Profiles (expandable)
            with st.expander("Team Strength Profiles", expanded=False):
                if ENGINE_AVAILABLE:
                    try:
                        team_totals_for_strength = _compute_projected_team_totals()
                        # DB-only: Yahoo API surface doesn't expose the full historical schedule.
                        full_schedule = load_league_schedule_full()
                        if not full_schedule:
                            try:
                                full_schedule = yds.get_full_league_schedule()
                            except Exception:
                                full_schedule = {}

                        profiles = compute_team_strength_profiles(
                            team_weekly_totals=team_totals_for_strength,
                            full_schedule=full_schedule if full_schedule else None,
                            current_week=_current_week,
                        )

                        if profiles:
                            strength_rows: list[dict] = []
                            for p in sorted(profiles, key=lambda x: x.get("rank", 99)):
                                ie = p.get("injury_exposure")
                                mom = p.get("momentum")
                                strength_rows.append(
                                    {
                                        "Rank": p.get("rank", 0),
                                        "Team": p.get("team_name", ""),
                                        "Power": f"{p.get('power_rating', 0):.1f}",
                                        "Roster": f"{p.get('roster_quality', 0):.3f}",
                                        "Balance": f"{p.get('category_balance', 0):.3f}",
                                        "SOS": f"{p.get('schedule_strength', 0.5):.3f}",
                                        "Injury": f"{ie:.3f}"
                                        if ie is not None and not (isinstance(ie, float) and np.isnan(ie))
                                        else "N/A",
                                        "Momentum": f"{mom:.2f}"
                                        if mom is not None and not (isinstance(mom, float) and np.isnan(mom))
                                        else "N/A",
                                        "CI": f"{p.get('ci_low', 0):.1f}-{p.get('ci_high', 0):.1f}",
                                    }
                                )

                            strength_df = pd.DataFrame(strength_rows)

                            # Highlight user team
                            str_row_classes: dict[int, str] = {}
                            for idx, r in enumerate(strength_rows):
                                if user_team and r["Team"] == user_team:
                                    str_row_classes[idx] = "row-start"

                            render_compact_table(strength_df, row_classes=str_row_classes)
                    except Exception as e:
                        st.warning(f"Team strength profiles unavailable: {e}")
                elif POWER_RANKINGS_AVAILABLE:
                    st.info("Enhanced standings engine not available. Using basic power rankings.")
                else:
                    st.info("Power rankings module not available.")

            # Section C: Scenario Explorer
            _section_label("Scenario Explorer", fig="WHAT-IF")
            st.caption("What if you go ___-___-___ this week?")

            sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1])
            with sc1:
                scenario_w = st.number_input("W", min_value=0, max_value=12, value=6, key="scenario_w")
            with sc2:
                scenario_l = st.number_input("L", min_value=0, max_value=12, value=6, key="scenario_l")
            with sc3:
                scenario_t = st.number_input("T", min_value=0, max_value=12, value=0, key="scenario_t")
            with sc4:
                total_cats = scenario_w + scenario_l + scenario_t
                if total_cats != 12:
                    st.warning(f"Sum = {total_cats} (must be 12)")
                run_scenario = st.button(
                    "Re-simulate", type="primary", key="run_scenario_btn", disabled=(total_cats != 12)
                )

            if run_scenario and total_cats == 12 and user_team:
                # Modify user's record for this week and re-simulate
                modified_standings = {}
                if not records_df.empty:
                    for _, row in records_df.iterrows():
                        tn = str(row["team_name"])
                        modified_standings[tn] = {
                            "W": int(row.get("wins", 0)),
                            "L": int(row.get("losses", 0)),
                            "T": int(row.get("ties", 0)),
                        }
                else:
                    team_totals_tmp = _compute_projected_team_totals()
                    modified_standings = {t: {"W": 0, "L": 0, "T": 0} for t in team_totals_tmp}

                if user_team in modified_standings:
                    # Award scenario W-L-T to user
                    if scenario_w > scenario_l:
                        modified_standings[user_team]["W"] += 1
                    elif scenario_l > scenario_w:
                        modified_standings[user_team]["L"] += 1
                    else:
                        modified_standings[user_team]["T"] += 1

                    team_totals_for_scenario = _compute_projected_team_totals()
                    # DB-only: Yahoo API surface doesn't expose the full historical schedule.
                    full_schedule_sc = load_league_schedule_full()
                    if not full_schedule_sc:
                        try:
                            full_schedule_sc = yds.get_full_league_schedule()
                        except Exception:
                            full_schedule_sc = {}

                    if team_totals_for_scenario and ENGINE_AVAILABLE:
                        with st.spinner("Re-simulating with scenario..."):
                            try:
                                scenario_result = simulate_season_enhanced(
                                    current_standings=modified_standings,
                                    team_weekly_totals=team_totals_for_scenario,
                                    full_schedule=full_schedule_sc
                                    if full_schedule_sc
                                    else {
                                        w: [
                                            (
                                                list(team_totals_for_scenario.keys())[i],
                                                list(team_totals_for_scenario.keys())[
                                                    (i + w) % len(team_totals_for_scenario)
                                                ],
                                            )
                                            for i in range(0, len(team_totals_for_scenario), 2)
                                        ]
                                        for w in range(1, _TOTAL_WEEKS + 1)
                                    },
                                    current_week=_current_week + 1,  # simulate from next week
                                    n_sims=500,
                                    seed=42,
                                    playoff_spots=_PLAYOFF_SPOTS,
                                )
                                # Compare playoff odds
                                old_pct = sim_result.get("playoff_probability", {}).get(user_team, 0.0)
                                new_pct = scenario_result.get("playoff_probability", {}).get(user_team, 0.0)
                                delta = (new_pct - old_pct) * 100
                                delta_color = T["green"] if delta > 0 else T["danger"] if delta < 0 else T["tx2"]
                                delta_sign = "+" if delta > 0 else ""

                                st.markdown(
                                    f'<div style="padding:12px;background:{T["card"]};border-radius:8px;'
                                    f'border:1px solid {T["border"]};margin:8px 0;">'
                                    f'<span style="font-size:14px;color:{T["tx"]};">Your playoff odds change from '
                                    f"<strong>{old_pct * 100:.0f}%</strong> to "
                                    f"<strong>{new_pct * 100:.0f}%</strong> "
                                    f'(<span style="color:{delta_color};font-weight:700;">{delta_sign}{delta:.0f}%</span>)</span></div>',
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.warning(f"Scenario simulation failed: {e}")
                    else:
                        render_empty_state("Projection data not available for scenario analysis")
                else:
                    st.warning("Your team not found in standings data.")

        elif "standings_legacy_result" in st.session_state:
            st.markdown("**Projected Season Results (Legacy Simulation)**")
            st.caption("Using basic round-robin simulation. Enhanced engine not available.")
            render_compact_table(st.session_state["standings_legacy_result"])

        else:
            render_empty_state(
                "No roster data for projections",
                "Connect your Yahoo league and run bootstrap.",
                icon_key="bar_chart",
            )

    # ── Tab 3: Playoff Odds (was pages/7_Playoff_Odds.py) ────────────
    with tab_playoffs:
        _render_playoff_odds_tab()


page_timer_footer("League Standings")
render_feedback_widget("League Standings")
