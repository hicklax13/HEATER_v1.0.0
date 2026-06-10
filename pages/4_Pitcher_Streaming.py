"""Pitcher Streaming — matchup-specific stream finder for any date this week.

Design: docs/superpowers/specs/2026-06-09-pitcher-streaming-analyzer-design.md.
All scoring arrives from src/optimizer/stream_analyzer.py (Stream Score board)
and src/optimizer/fa_recommender.py (today's add/drop swaps) — this page
renders engine output and never computes a score itself (guarded by
tests/test_stream_page_no_inline_scoring.py).
"""

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.game_day import get_target_game_date
from src.league_manager import get_team_roster
from src.league_rules import WEEKLY_TRANSACTION_LIMIT
from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.optimizer.fa_recommender import recommend_streaming_moves
from src.optimizer.stream_analyzer import build_stream_board
from src.ui_shared import (
    THEME,
    format_stat,
    inject_custom_css,
    no_league_data_message,
    page_timer_footer,
    page_timer_start,
    render_empty_state,
    render_page_header,
    render_sortable_table,
)
from src.usage import log_page_view
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

logger = logging.getLogger(__name__)

T = THEME

# ── Page setup ────────────────────────────────────────────────────────────────

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Pitcher Streaming",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

init_db()
inject_custom_css()
require_auth()
require_page_enabled("page:4_Pitcher_Streaming")
log_page_view("Pitcher Streaming")
page_timer_start()

# ── Data loading ──────────────────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()
pool = pool.rename(columns={"name": "player_name"})

config = LeagueConfig()
yds = get_yahoo_data_service()
rosters = yds.get_rosters()

if rosters.empty:
    render_empty_state(
        "No league data yet",
        no_league_data_message(yds.data_unavailable_reason()),
        icon_key="users",
    )
    st.stop()

user_team_name = resolve_viewer_team_name(rosters)
if not user_team_name:
    st.warning("No user team identified.")
    st.stop()

user_roster = get_team_roster(user_team_name)


def _build_ctx():
    """Build (and session-cache) the shared optimizer context.

    scope="today" so recommend_streaming_moves can produce swap recs; the
    board itself works for future dates off the same context. A failed build
    degrades to a pool-only context with a visible banner (the FA-page
    visible-fallback rule — never silently).
    """
    cached = st.session_state.get("stream_page_ctx")
    if cached is not None:
        return cached
    try:
        from src.optimizer.shared_data_layer import build_optimizer_context
        from src.validation.dynamic_context import compute_weeks_remaining

        with st.spinner("Loading matchup, schedule, and player intelligence..."):
            ctx = build_optimizer_context(
                scope="today",
                yds=yds,
                config=config,
                weeks_remaining=compute_weeks_remaining(),
                user_team_name=user_team_name,
                roster=user_roster,
                level_filter="MLB only",
            )
        st.session_state["stream_page_ctx"] = ctx
        return ctx
    except Exception as ctx_err:
        logger.exception("Pitcher Streaming: optimizer context build failed")
        st.warning(
            "Live matchup context unavailable "
            f"({type(ctx_err).__name__}: {ctx_err}) — showing a schedule-only "
            "board without matchup urgency, drop suggestions, or roster filtering."
        )
        name_to_pool_id = dict(zip(pool["player_name"], pool["player_id"]))
        rostered_ids = {
            int(pid)
            for pid in (name_to_pool_id.get(str(n)) for n in rosters.get("name", pd.Series(dtype=str)))
            if pid is not None
        }
        return SimpleNamespace(
            player_pool=pool,
            user_roster_ids=[],
            league_rostered_ids=rostered_ids,
            team_strength={},
            park_factors={},
            weather={},
            two_start_pitchers=[],
            recent_form={},
            todays_schedule=[],
            config=config,
            category_weights=None,
            adds_remaining_this_week=None,
            urgency_weights={},
            scope="today",
        )


ctx = _build_ctx()


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_schedule_cached(date_str: str) -> list[dict]:
    """One day's MLB schedule (probables + status), cached 15 minutes."""
    try:
        import statsapi

        return statsapi.schedule(start_date=date_str, end_date=date_str)
    except Exception:
        logger.warning("Pitcher Streaming: schedule fetch failed for %s", date_str, exc_info=True)
        return []


# ── Header ───────────────────────────────────────────────────────────────────

render_page_header("Pitcher Streaming", eyebrow="DAILY", fig="FIG.4 — PITCHER STREAMING")

tab_finder, tab_microscope, tab_planner, tab_record = st.tabs(
    ["Stream Finder", "Matchup Microscope", "Week Planner", "Track Record"]
)

# ── Tab 1: Stream Finder ─────────────────────────────────────────────────────

with tab_finder:
    target_default = get_target_game_date()
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    date_options: list[str] = []
    base = datetime.strptime(target_default, "%Y-%m-%d")
    for offset in range(8):
        date_options.append((base + timedelta(days=offset)).strftime("%Y-%m-%d"))

    def _date_label(d: str) -> str:
        dt = datetime.strptime(d, "%Y-%m-%d")
        suffix = " (today)" if d == today_str else ""
        return dt.strftime("%a %b %d") + suffix

    ctrl_date, ctrl_mine, ctrl_locked = st.columns([2, 1, 1])
    with ctrl_date:
        target_date = st.selectbox("Stream date", date_options, format_func=_date_label, key="stream_date")
    with ctrl_mine:
        include_rostered = st.checkbox(
            "Include my SPs",
            value=False,
            help="Compare your own probable starters against the wire on this date.",
        )
    with ctrl_locked:
        show_locked = st.checkbox(
            "Show locked starts",
            value=True,
            help="Keep rows whose game is live or final visible (non-actionable).",
        )

    # Budget / pacing strip
    adds_remaining = getattr(ctx, "adds_remaining_this_week", None)
    ip_target = float(CONSTANTS_REGISTRY["stream_ip_target"].value)
    losing = []
    tied = []
    try:
        summary = (getattr(ctx, "urgency_weights", None) or {}).get("summary", {})
        losing = list(summary.get("losing", []))
        tied = list(summary.get("tied", []))
    except Exception:
        pass
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Adds remaining this week",
            f"{adds_remaining} / {WEEKLY_TRANSACTION_LIMIT}" if adds_remaining is not None else "—",
        )
    with m2:
        st.metric("Weekly IP target", f"{ip_target:.0f} IP")
    with m3:
        st.metric(
            "Cats in play",
            ", ".join(losing + tied) if (losing or tied) else "—",
            help="Losing/tied categories this matchup — streams that help these matter most.",
        )

    schedule = None if target_date == today_str else _fetch_schedule_cached(target_date)
    board = build_stream_board(ctx, target_date, schedule=schedule, include_rostered=include_rostered)
    board_full = board  # unfiltered reference for the Microscope tab

    if board.empty:
        render_empty_state(
            "No streamable starts",
            "No probable starters found in the player pool for this date — "
            "probables may not be posted yet (typically 1-5 days out).",
            icon_key="baseball",
        )
    else:
        if not show_locked:
            board = board[board["actionable"]].reset_index(drop=True)

        display = pd.DataFrame(
            {
                "Pitcher": board["player_name"],
                "Tm": board["team"],
                "Opp": [("vs " if home else "@ ") + opp for home, opp in zip(board["is_home"], board["opponent"])],
                "Status": board["status"],
                "Conf": board["confidence"],
                "GS": board["num_starts"],
                "Score": board["stream_score"],
                "Net SGP": [format_stat(v, "SGP") for v in board["net_sgp"]],
                "Opp wRC+": board["opp_wrc_plus"].round(0).astype(int),
                "Opp K%": board["opp_k_pct"].round(1),
                "Park": board["park_factor"].round(2),
                "xIP": board["expected_ip"].round(1),
                "xK": board["expected_k"].round(1),
                "xER": board["expected_er"].round(1),
                "W%": (board["win_probability"] * 100).round(0).astype(int),
                "Own%": board["percent_owned"].fillna(0).round(0).astype(int),
                "Risk": [", ".join(f) for f in board["risk_flags"]],
            }
        )
        if board["split_source"].eq("overall").all():
            st.caption("Opponent wRC+/K% are overall team rates (vs-handedness splits unavailable).")
        render_sortable_table(display, height=520, key="stream_board")

        # Why-expanders for the top actionable candidates
        top = board[board["actionable"]].head(5)
        if not top.empty:
            st.markdown("**Why these scores**")
            for _, row in top.iterrows():
                title = (
                    f"{row['player_name']} ({row['team']}) "
                    f"{'vs' if row['is_home'] else '@'} {row['opponent']} — "
                    f"score {row['stream_score']}"
                )
                with st.expander(title):
                    comps = row["components"]
                    comp_df = pd.DataFrame(
                        {
                            "Component": list(comps.keys()),
                            "Value (-1 to +1)": list(comps.values()),
                            "Weight": [CONSTANTS_REGISTRY[f"stream_score_w_{name}"].value for name in comps.keys()],
                        }
                    )
                    st.dataframe(comp_df, hide_index=True)
                    if row["risk_flags"]:
                        st.caption("Risk flags: " + ", ".join(row["risk_flags"]))
                    st.caption(
                        f"Expected line: {row['expected_ip']:.1f} IP, "
                        f"{row['expected_k']:.1f} K, {row['expected_er']:.1f} ER, "
                        f"{row['win_probability']:.0%} win"
                    )

    # ── Matchup impact: this week's matchup with vs without the stream ──
    st.markdown("---")
    st.markdown("**Matchup impact (with vs without the stream)**")
    sel_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    now_utc = datetime.now(UTC).date()
    week_monday = now_utc - timedelta(days=now_utc.weekday())
    week_sunday = week_monday + timedelta(days=6)
    in_matchup_week = week_monday <= sel_date <= week_sunday
    my_totals = getattr(ctx, "my_totals", None) or {}
    opp_totals = getattr(ctx, "opp_totals", None) or {}

    if not in_matchup_week:
        st.caption(
            "This date falls outside the current matchup week (Mon-Sun) — "
            "next week's opponent isn't known yet, so no with/without "
            "projection is possible."
        )
    elif not my_totals or not opp_totals:
        st.caption("Live matchup totals unavailable — connect Yahoo to project impact.")
    elif board.empty or not board["actionable"].any():
        st.caption("No actionable streams to project for this date.")
    else:
        from src.optimizer.stream_analyzer import compute_matchup_impact

        opp_label = getattr(ctx, "opponent_name", "") or "opponent"
        impact_rows = []
        for _, row in board[board["actionable"]].head(5).iterrows():
            impact = compute_matchup_impact(
                my_totals,
                opp_totals,
                {
                    "ip": row["expected_ip"],
                    "k": row["expected_k"],
                    "er": row["expected_er"],
                    "win_prob": row["win_probability"],
                },
                pitcher_whip=row.get("whip") or 0.0,
                config=config,
                num_starts=int(row["num_starts"]),
            )
            if impact is None:
                continue
            movers = sorted(impact["per_cat"].items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True)
            mover_txt = ", ".join(
                f"{cat.upper()} {d['delta']:+.0%}" for cat, d in movers[:3] if abs(d["delta"]) >= 0.005
            )
            impact_rows.append(
                {
                    "Pitcher": row["player_name"],
                    "vs": row["opponent"],
                    "Exp cat wins": f"{impact['expected_wins_delta']:+.2f}",
                    "Matchup win%": f"{impact['overall_win_prob_delta']:+.1%}",
                    "Biggest movers": mover_txt or "negligible",
                }
            )
        if impact_rows:
            st.caption(
                f"Projected change to this week's matchup vs {opp_label} if you "
                "add each pitcher for this start (same category engine as the "
                "Lineup Optimizer; drop-side cost shown in the swap table below)."
            )
            st.dataframe(pd.DataFrame(impact_rows), hide_index=True)
        else:
            st.caption("Matchup impact could not be computed for these candidates.")

    # Swap recommendations — same-day matchup state required
    st.markdown("---")
    st.markdown("**Suggested swap (today)**")
    if target_date != today_str:
        st.caption(
            "Swap suggestions need same-day matchup state (lineup locks, "
            "in-play categories) — pick today's date to see add/drop pairs."
        )
    elif getattr(ctx, "scope", "") != "today" or not getattr(ctx, "user_roster_ids", None):
        st.caption("Swap suggestions unavailable without live matchup context.")
    else:
        try:
            moves = recommend_streaming_moves(ctx)
            pitcher_moves = moves.get("pitchers", [])
            if not pitcher_moves:
                note = (moves.get("diagnostics") or {}).get("note", "")
                st.caption(
                    "No pitcher stream clears the engine's net-SGP and "
                    "category-protection bars today." + (f" ({note})" if note else "")
                )
            else:
                swap_df = pd.DataFrame(
                    {
                        "Add": [m.get("add_name", "") for m in pitcher_moves],
                        "Drop": [m.get("drop_name", "") for m in pitcher_moves],
                        "Net SGP": [format_stat(m.get("net_sgp", 0.0), "SGP") for m in pitcher_moves],
                        "Helps": [
                            ", ".join(m.get("target_cats_helped") or m.get("helps") or []) for m in pitcher_moves
                        ],
                    }
                )
                st.dataframe(swap_df, hide_index=True)
        except Exception as swap_err:
            logger.exception("Pitcher Streaming: recommend_streaming_moves failed")
            st.warning(
                f"Swap engine unavailable ({type(swap_err).__name__}: {swap_err}) — board scores above are unaffected."
            )

# ── Tab 2: Matchup Microscope ────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _history_cached(mlb_id: int, opp_team: str | None, last_n: int) -> pd.DataFrame:
    from src.optimizer.stream_analyzer import get_pitcher_vs_team_history

    return get_pitcher_vs_team_history(mlb_id, opp_team=opp_team, last_n=last_n)


with tab_microscope:
    if board_full.empty:
        st.info("Pick a date with streamable starts on the Stream Finder tab first.")
    else:
        options = list(board_full.index)
        pick = st.selectbox(
            "Candidate",
            options,
            format_func=lambda i: (
                f"{board_full.loc[i, 'player_name']} ({board_full.loc[i, 'team']}) "
                f"{'vs' if board_full.loc[i, 'is_home'] else '@'} {board_full.loc[i, 'opponent']}"
            ),
            key="microscope_pick",
        )
        sel = board_full.loc[pick]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Stream Score", sel["stream_score"])
        with c2:
            st.metric("Net SGP", format_stat(sel["net_sgp"], "SGP"))
        with c3:
            st.metric("Matchup (0-10)", sel["matchup_score"])
        with c4:
            st.metric("Status", f"{sel['status']} / {sel['confidence']}")

        env_bits = [
            f"Venue {sel['venue']} (park {sel['park_factor']:.2f})",
            f"Opp wRC+ {sel['opp_wrc_plus']:.0f}, K% {sel['opp_k_pct']:.1f}",
        ]
        if sel["risk_flags"]:
            env_bits.append("Risk: " + ", ".join(sel["risk_flags"]))
        st.caption(" | ".join(env_bits))
        if sel["split_source"] == "overall":
            st.caption("Opponent rates are overall splits (vs-handedness data unavailable).")

        comp_df = pd.DataFrame(
            {
                "Component": list(sel["components"].keys()),
                "Value (-1 to +1)": list(sel["components"].values()),
                "Weight": [CONSTANTS_REGISTRY[f"stream_score_w_{n}"].value for n in sel["components"]],
            }
        )
        st.dataframe(comp_df, hide_index=True)

        # ── Opposing lineup + PvB exposure ───────────────────────────────
        st.markdown("**Opposing lineup**")
        confirmed = (getattr(ctx, "confirmed_lineups", None) or {}).get(sel["opponent"]) or []
        if not confirmed:
            st.caption(
                f"No confirmed lineup posted for {sel['opponent']} yet "
                "(lineups typically post 1-2 hours before first pitch)."
            )
        else:
            name_to_pool = dict(zip(pool["player_name"], pool["player_id"]))
            bats_by_name = dict(zip(pool["player_name"], pool.get("bats", "")))
            batter_ids = [int(name_to_pool[n]) for n in confirmed if name_to_pool.get(n) is not None]
            lineup_df = pd.DataFrame(
                {
                    "Order": range(1, len(confirmed) + 1),
                    "Batter": confirmed,
                    "Bats": [bats_by_name.get(n, "") or "" for n in confirmed],
                }
            )
            st.dataframe(lineup_df, hide_index=True)
            try:
                from src.optimizer.stream_analyzer import compute_lineup_exposure

                exposure = compute_lineup_exposure(int(sel["player_id"]), batter_ids, pool)
                if exposure is not None:
                    st.caption(
                        f"Regressed lineup wOBA vs league average: {exposure:+.3f} "
                        "(positive = dangerous lineup for this pitcher; "
                        "PvB samples shrunk to 60-PA stabilization)"
                    )
            except Exception:
                logger.exception("Pitcher Streaming: lineup exposure failed")

        # ── Pitcher game-log history ─────────────────────────────────────
        st.markdown("**Recent starts**")
        mlb_id = sel.get("mlb_id")
        if not mlb_id or pd.isna(mlb_id):
            st.caption("No MLB id on file for this pitcher — game logs unavailable.")
        elif st.button("Load game logs", key="load_history"):
            hist = _history_cached(int(mlb_id), None, 10)
            if hist.empty:
                st.caption("No game logs returned (statsapi unavailable or no starts).")
            else:
                from src.optimizer.stream_analyzer import aggregate_pitcher_history

                hist_disp = hist.copy()
                hist_disp["ip"] = hist_disp["ip"].round(1)
                st.dataframe(
                    hist_disp[["date", "opponent", "is_home", "ip", "k", "er", "bb", "h"]],
                    hide_index=True,
                )
                agg_all = aggregate_pitcher_history(hist)
                vs_opp = hist[hist["opponent"] == sel["opponent"]]
                agg_opp = aggregate_pitcher_history(vs_opp)
                a1, a2 = st.columns(2)
                with a1:
                    st.metric(
                        f"Last {agg_all.get('games', 0)} starts",
                        f"{format_stat(agg_all.get('era', 0.0), 'ERA')} ERA, "
                        f"{format_stat(agg_all.get('whip', 0.0), 'WHIP')} WHIP, "
                        f"{agg_all.get('k', 0):.0f} K",
                    )
                with a2:
                    if agg_opp:
                        st.metric(
                            f"vs {sel['opponent']} ({agg_opp['games']} starts)",
                            f"{format_stat(agg_opp.get('era', 0.0), 'ERA')} ERA, "
                            f"{format_stat(agg_opp.get('whip', 0.0), 'WHIP')} WHIP, "
                            f"{agg_opp.get('k', 0):.0f} K",
                        )
                    else:
                        st.metric(f"vs {sel['opponent']}", "no recent starts")

# ── Tab 3: Week Planner ──────────────────────────────────────────────────────


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_week_schedule_cached(start: str, end: str) -> list[dict]:
    try:
        import statsapi

        return statsapi.schedule(start_date=start, end_date=end)
    except Exception:
        logger.warning("Pitcher Streaming: week schedule fetch failed", exc_info=True)
        return []


with tab_planner:
    st.caption(
        "Greedy stream sequence for the next 7 days under your remaining add "
        "budget. Advisory only — FCFS waivers mean nothing is reserved; the "
        "plan re-computes on every visit."
    )
    week_start = datetime.now(UTC).strftime("%Y-%m-%d")
    week_end = (datetime.now(UTC) + timedelta(days=6)).strftime("%Y-%m-%d")
    week_schedule = _fetch_week_schedule_cached(week_start, week_end)

    from src.optimizer.stream_analyzer import build_week_plan

    plan_result = build_week_plan(ctx, schedule=week_schedule)
    plan = plan_result["plan"]
    summary = plan_result["summary"]

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Planned streams", f"{summary['n_planned']} / {summary['max_adds']} adds")
    with p2:
        st.metric("IP added", f"{summary['ip_added']:.1f}")
    with p3:
        st.metric("K added", f"{summary['k_added']:.0f}")
    with p4:
        st.metric("Net SGP", format_stat(summary["net_sgp_total"], "SGP"))
    st.caption(
        f"Weekly pacing: {summary['ip_target']:.0f} IP target, {summary['ip_floor']:.0f} IP Yahoo forfeit floor."
    )

    if not plan:
        render_empty_state(
            "No positive-value streams in the window",
            "Either probables aren't posted yet, the add budget is exhausted, "
            "or no FA start clears a positive net SGP.",
            icon_key="baseball",
        )
    else:
        plan_df = pd.DataFrame(
            {
                "Date": [e["game_date"] for e in plan],
                "Add": [e["player_name"] for e in plan],
                "Tm": [e["team"] for e in plan],
                "Opp": [e["opponent"] for e in plan],
                "GS": [e["num_starts"] for e in plan],
                "Conf": [e["confidence"] for e in plan],
                "Score": [e["stream_score"] for e in plan],
                "Net SGP": [format_stat(e["net_value"], "SGP") for e in plan],
                "xIP": [round(e["expected_ip"] * e["num_starts"], 1) for e in plan],
                "xK": [round(e["expected_k"] * e["num_starts"], 1) for e in plan],
                "Risk": [", ".join(e["risk_flags"]) for e in plan],
            }
        )
        st.dataframe(plan_df, hide_index=True)
        sequence = " → ".join(f"{e['game_date'][5:]}: {e['player_name']}" for e in plan)
        st.caption(f"Sequence: {sequence}")

# ── Tab 4: Track Record ──────────────────────────────────────────────────────

with tab_record:
    past_options = [(datetime.now(UTC) - timedelta(days=off)).strftime("%Y-%m-%d") for off in range(1, 15)]
    replay_date = st.selectbox(
        "Replay date",
        past_options,
        format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%a %b %d"),
        key="replay_date",
    )

    if st.button("Run replay", key="run_replay"):
        from src.optimizer.stream_analyzer import replay_stream_date

        with st.spinner("Scoring the historical board and fetching box scores..."):
            replay = replay_stream_date(ctx, replay_date, schedule=_fetch_schedule_cached(replay_date))
        if replay["proxy_caveat"]:
            st.caption(
                "Replay scores use CURRENT projections and form as a proxy — "
                "HEATER does not store point-in-time projections. Matchup facts "
                "(opponent, park, schedule) are historically exact; treat the "
                "form component as approximate."
            )
        if replay["board_then"].empty:
            render_empty_state(
                "Nothing to replay",
                "No probable starters matched the player pool for this date.",
                icon_key="baseball",
            )
        elif replay["actuals"].empty:
            st.caption(
                "Board scored, but no actual game logs matched this date (statsapi unavailable or starts were skipped)."
            )
            st.dataframe(
                replay["board_then"][["player_name", "team", "opponent", "stream_score", "expected_k"]],
                hide_index=True,
            )
        else:
            summary = replay["summary"]
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric(
                    "Top picks graded",
                    f"{summary['games']} starts",
                )
            with r2:
                st.metric(
                    "Actual line",
                    f"{format_stat(summary['era'], 'ERA')} ERA / {format_stat(summary['whip'], 'WHIP')} WHIP",
                )
            with r3:
                st.metric("Quality-start rate", f"{summary['qs_rate']:.0%}")
            with r4:
                st.metric("K vs expected", f"{summary['k_delta_vs_expected']:+.1f}")
            actuals_disp = replay["actuals"].rename(
                columns={
                    "player_name": "Pitcher",
                    "team": "Tm",
                    "opponent": "Opp",
                    "stream_score_then": "Score",
                    "expected_k": "xK",
                    "actual_ip": "IP",
                    "actual_k": "K",
                    "actual_er": "ER",
                    "actual_w": "W",
                    "quality_start": "QS",
                }
            )
            st.dataframe(
                actuals_disp[["Pitcher", "Tm", "Opp", "Score", "xK", "IP", "K", "ER", "W", "QS"]],
                hide_index=True,
            )

    # ── My streams this season ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("**My pitcher adds this season**")
    try:
        txns = yds.get_transactions()
        if txns is None or txns.empty:
            st.caption("No transaction history available.")
        else:
            mine = txns.copy()
            if "team_name" in mine.columns:
                mine = mine[mine["team_name"] == user_team_name]
            if "type" in mine.columns:
                mine = mine[mine["type"] == "add"]
            pitcher_names = set(pool.loc[pool.get("is_hitter", 1) == 0, "player_name"].astype(str))
            name_col = "player_name" if "player_name" in mine.columns else "name"
            if name_col in mine.columns:
                mine = mine[mine[name_col].astype(str).isin(pitcher_names)]
            if mine.empty:
                st.caption("No pitcher adds recorded for your team yet.")
            else:
                show_cols = [c for c in [name_col, "timestamp", "type"] if c in mine.columns]
                st.dataframe(mine[show_cols].reset_index(drop=True), hide_index=True)
    except Exception:
        logger.exception("Pitcher Streaming: transactions table failed")
        st.caption("Transaction history unavailable.")

page_timer_footer("Pitcher Streaming")
render_feedback_widget("Pitcher Streaming")
