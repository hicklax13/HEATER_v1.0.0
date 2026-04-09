"""League Standings -- Live H2H standings, MC season projections, and team strength."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.database import (
    get_connection,
    init_db,
    load_league_records,
    load_league_schedule_full,
)
from src.ui_shared import (
    THEME,
    build_compact_table_html,
    format_stat,  # noqa: F401
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_data_freshness_card,
    render_page_layout,
)
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

_SEASON_START = datetime(2026, 3, 25, tzinfo=UTC)
_TOTAL_WEEKS = 24
_PLAYOFF_SPOTS = 4

_HIT_CATS = {"R", "HR", "RBI", "SB", "AVG", "OBP"}
_PIT_CATS = {"W", "L", "SV", "K", "ERA", "WHIP"}
_RATE_STATS = {"AVG", "OBP", "ERA", "WHIP"}
_INVERSE_CATS = {"L", "ERA", "WHIP"}
_CAT_ORDER = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]


# ── Helper: user team name ────────────────────────────────────────────


def _get_user_team_name() -> str | None:
    """Retrieve the authenticated user's team name from the DB."""
    try:
        conn = get_connection()
        try:
            row = conn.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


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
    """Compute projected weekly category averages per team from roster projections."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT lr.team_name, p.is_hitter,
                   COALESCE(proj.pa, 0) as pa, COALESCE(proj.ab, 0) as ab,
                   COALESCE(proj.h, 0) as h, COALESCE(proj.r, 0) as r,
                   COALESCE(proj.hr, 0) as hr, COALESCE(proj.rbi, 0) as rbi,
                   COALESCE(proj.sb, 0) as sb,
                   COALESCE(proj.bb, 0) as bb, COALESCE(proj.hbp, 0) as hbp,
                   COALESCE(proj.sf, 0) as sf,
                   COALESCE(proj.ip, 0) as ip, COALESCE(proj.w, 0) as w,
                   COALESCE(proj.l, 0) as l, COALESCE(proj.sv, 0) as sv,
                   COALESCE(proj.k, 0) as k,
                   COALESCE(proj.er, 0) as er,
                   COALESCE(proj.bb_allowed, 0) as bb_allowed,
                   COALESCE(proj.h_allowed, 0) as h_allowed
            FROM league_rosters lr
            JOIN players p ON lr.player_id = p.player_id
            LEFT JOIN projections proj ON p.player_id = proj.player_id
                AND proj.system = 'blended'
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return {}

    weeks = 26.0
    team_totals: dict[str, dict[str, float]] = {}

    for team, group in df.groupby("team_name"):
        hitters = group[group["is_hitter"] == 1]
        pitchers = group[group["is_hitter"] == 0]

        total_r = float(hitters["r"].sum()) / weeks
        total_hr = float(hitters["hr"].sum()) / weeks
        total_rbi = float(hitters["rbi"].sum()) / weeks
        total_sb = float(hitters["sb"].sum()) / weeks

        total_ab = float(hitters["ab"].sum())
        total_h = float(hitters["h"].sum())
        total_bb_h = float(hitters["bb"].sum())
        total_hbp = float(hitters["hbp"].sum())
        total_sf = float(hitters["sf"].sum())

        avg = total_h / total_ab if total_ab > 0 else 0.250
        obp_denom = total_ab + total_bb_h + total_hbp + total_sf
        obp = (total_h + total_bb_h + total_hbp) / obp_denom if obp_denom > 0 else 0.330

        total_w = float(pitchers["w"].sum()) / weeks
        total_l = float(pitchers["l"].sum()) / weeks
        total_sv = float(pitchers["sv"].sum()) / weeks
        total_k = float(pitchers["k"].sum()) / weeks

        total_ip = float(pitchers["ip"].sum())
        total_er = float(pitchers["er"].sum())
        total_bb_p = float(pitchers["bb_allowed"].sum())
        total_h_p = float(pitchers["h_allowed"].sum())

        era = (total_er * 9) / total_ip if total_ip > 0 else 4.00
        whip = (total_bb_p + total_h_p) / total_ip if total_ip > 0 else 1.25

        team_totals[str(team)] = {
            "R": total_r,
            "HR": total_hr,
            "RBI": total_rbi,
            "SB": total_sb,
            "AVG": avg,
            "OBP": obp,
            "W": total_w,
            "L": total_l,
            "SV": total_sv,
            "K": total_k,
            "ERA": era,
            "WHIP": whip,
        }

    return team_totals


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

    # Determine close categories (where diff is small)
    close_cats: list[str] = []
    for c in cats:
        name = c.get("name", "")
        user_v = float(c.get("user_val", 0))
        opp_v = float(c.get("opp_val", 0))
        if abs(user_v - opp_v) < max(0.01 * max(abs(user_v), abs(opp_v), 1), 0.001):
            close_cats.append(name)

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

st.set_page_config(
    page_title="Heater | League Standings",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()
inject_custom_css()
page_timer_start()

# ── Data loading ──────────────────────────────────────────────────────

yds = get_yahoo_data_service()
matchup = yds.get_matchup()
records_df = load_league_records()
user_team = _get_user_team_name()

# ── Banner ────────────────────────────────────────────────────────────

banner_teaser = _build_banner_teaser(matchup)
render_page_layout(
    "LEAGUE STANDINGS",
    banner_teaser=banner_teaser,
    banner_icon="league_standings",
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

            render_context_card(
                "YOUR POSITION",
                f'<p style="font-size:22px;font-weight:700;margin:0;color:{T["tx"]};">'
                f"{_ordinal(rank)}</p>"
                f'<p style="margin:4px 0;font-size:13px;color:{T["tx2"]};">'
                f"{wins}-{losses}-{ties} ({win_pct:.3f})</p>"
                f'<p style="margin:2px 0;font-size:12px;color:{T["tx2"]};">'
                f"GB from 1st: {gb_first}</p>"
                f'<p style="margin:2px 0;font-size:12px;color:{T["tx2"]};">'
                f"{playoff_label}</p>",
            )
        else:
            render_context_card(
                "YOUR POSITION", '<p style="color:#6b7280;font-size:12px;">Team not found in standings.</p>'
            )
    else:
        render_context_card(
            "YOUR POSITION", '<p style="color:#6b7280;font-size:12px;">Connect Yahoo to see your position.</p>'
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
        render_context_card("THIS WEEK", '<p style="color:#6b7280;font-size:12px;">No matchup data available.</p>')

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
            "PLAYOFF ODDS", '<p style="color:#6b7280;font-size:12px;">Run projections to see playoff odds.</p>'
        )

    # Card 4: Data freshness
    render_data_freshness_card()


# ══════════════════════════════════════════════════════════════════════
# Main Content — Tabs
# ══════════════════════════════════════════════════════════════════════

with main:
    tab1, tab2 = st.tabs(["Current Standings", "Season Projections"])

    # ── Tab 1: Current Standings ──────────────────────────────────────
    with tab1:
        has_records = not records_df.empty
        if has_records:
            # Section A: H2H Record Table
            st.markdown("**Head-to-Head Record**")

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
                        "Win%": f"{wp:.3f}",
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
            st.markdown(cutoff_css + html, unsafe_allow_html=True)

        # Section B: Category Standings Grid
        standings_df = yds.get_standings()
        if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
            # Filter to scoring categories only
            config = LeagueConfig()
            scoring_cats = set(c.upper() for c in config.all_categories)
            cat_standings = standings_df[standings_df["category"].str.upper().isin(scoring_cats)].copy()

            if not cat_standings.empty:
                st.markdown("**Category Standings**")

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
                )
                st.markdown(grid_html, unsafe_allow_html=True)

        elif not has_records:
            st.info("Connect your Yahoo league to see live standings. Run the app bootstrap first.")

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
            full_schedule = load_league_schedule_full()
            if not full_schedule:
                # Try fetching from Yahoo
                try:
                    full_schedule = yds.get_full_league_schedule()
                except Exception:
                    full_schedule = {}

            if team_totals and ENGINE_AVAILABLE:
                # Use enhanced simulation with actual schedule
                with st.spinner("Running season simulation..."):
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
                with st.spinner("Running season simulation..."):
                    try:
                        legacy_df = simulate_season(team_totals, n_sims=500, seed=42)
                        st.session_state["standings_legacy_result"] = legacy_df
                    except Exception as e:
                        st.warning(f"Legacy simulation failed: {e}")

        # Display simulation results
        sim_result = st.session_state.get(_sim_cache_key)

        if sim_result:
            st.markdown("**Projected Final Standings**")

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
                        "Win%": f"{rec.get('win_pct', 0):.3f}",
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
            st.markdown(proj_cutoff_css + proj_html, unsafe_allow_html=True)

            # Section B: Team Strength Profiles (expandable)
            with st.expander("Team Strength Profiles", expanded=False):
                if ENGINE_AVAILABLE:
                    try:
                        team_totals_for_strength = _compute_projected_team_totals()
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
            st.markdown("**Scenario Explorer**")
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
                        st.info("Projection data not available for scenario analysis.")
                else:
                    st.warning("Your team not found in standings data.")

        elif "standings_legacy_result" in st.session_state:
            st.markdown("**Projected Season Results (Legacy Simulation)**")
            st.caption("Using basic round-robin simulation. Enhanced engine not available.")
            render_compact_table(st.session_state["standings_legacy_result"])

        else:
            st.info("No roster data available for projections. Connect your Yahoo league and run bootstrap.")

page_timer_footer("League Standings")
