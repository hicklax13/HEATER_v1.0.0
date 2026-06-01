"""Trade Analyzer — Evaluate trade proposals with Phase 1 engine pipeline.

Uses the new engine modules for marginal SGP elasticity, punt detection,
LP lineup optimization, and deterministic grading (A+ to F).
Falls back to the legacy analyzer if the engine is unavailable.
"""

import logging
import time

import pandas as pd
import streamlit as st

from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import coerce_numeric_df, init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.injury_model import get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import (
    METRIC_TOOLTIPS,
    PAGE_ICONS,
    T,
    format_stat,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_data_freshness_card,
    render_page_layout,
    render_player_select,
    render_styled_table,
)
from src.usage import log_page_view
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility
from src.yahoo_data_service import get_yahoo_data_service

try:
    from src.trade_intelligence import apply_scarcity_flags, get_health_adjusted_pool

    TRADE_INTEL_AVAILABLE = True
except ImportError:
    TRADE_INTEL_AVAILABLE = False

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Trade Analyzer", page_icon="", layout="wide", initial_sidebar_state="collapsed"
    )


def _standings_data_state() -> str:
    """Return the quality state of the league_standings table.

    Returns:
        "empty"    — table has no rows at all.
        "all_zero" — table has rows but every 'total' value is zero or null
                     (week 1 before any matchup has completed).
        "ok"       — table has meaningful non-zero data.
    """
    try:
        yds = get_yahoo_data_service()
        standings = yds.get_standings()
        if standings.empty:
            return "empty"
        totals = standings.get("total", pd.Series(dtype=float))
        totals = pd.to_numeric(totals, errors="coerce").fillna(0)
        if (totals == 0).all():
            return "all_zero"
        return "ok"
    except Exception:
        return "empty"


init_db()

inject_custom_css()
require_auth()
require_page_enabled("page:11_Trade_Analyzer")
log_page_view("Trade Analyzer")
page_timer_start()

render_page_layout("TRADE ANALYZER", banner_teaser="Analyze a trade below", banner_icon="trade")

# Load data
pool = load_player_pool()
if pool.empty:
    st.warning("No player data. Load sample data or import projections first.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# Apply trade intelligence: health-adjust projections and add scarcity flags
if TRADE_INTEL_AVAILABLE:
    try:
        pool = get_health_adjusted_pool(pool)
        pool = apply_scarcity_flags(pool)
    except Exception:
        pass  # Graceful degradation — use raw pool if intelligence fails

config = LeagueConfig()


# Get user team roster
yds = get_yahoo_data_service()
rosters = yds.get_rosters()
if rosters.empty:
    st.warning(
        "No league data loaded. Connect your Yahoo league in Connect League, "
        "or league data will load automatically on next app launch."
    )
    st.stop()
else:
    user_team_name = resolve_viewer_team_name(rosters)
    if not user_team_name:
        st.warning("No user team identified.")
        st.stop()
    else:
        user_team_name = str(user_team_name)
        user_roster = get_team_roster(user_team_name)

        # Player selection — build name-to-id mapping from both pool AND roster
        # so that roster players missing from player_pool still appear.
        player_names = pool[["player_id", "player_name"]].drop_duplicates()
        name_to_id = dict(zip(player_names["player_name"], player_names["player_id"]))

        # Remap roster IDs to player_pool IDs via name matching.
        # Yahoo roster uses Yahoo Fantasy IDs; the pool uses MLB Stats API IDs.
        # They share player names, so we map: roster name → pool player_id.
        user_roster_ids = []
        if not user_roster.empty and "name" in user_roster.columns:
            for _, r in user_roster.iterrows():
                pname = r.get("name", "")
                pool_pid = name_to_id.get(pname)
                if pool_pid is not None:
                    user_roster_ids.append(pool_pid)
                else:
                    # Fallback: keep the roster ID (player may lack projections)
                    user_roster_ids.append(r["player_id"])
        else:
            user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

        # Also add roster player names (roster JOIN players gives "name")
        if not user_roster.empty and "name" in user_roster.columns:
            for _, r in user_roster.iterrows():
                pname = r.get("name", "")
                pid = r.get("player_id")
                if pname and pid and pname not in name_to_id:
                    name_to_id[pname] = pid

        ctx, main = render_context_columns()

        with ctx:
            render_context_card(
                "Trade Status",
                '<div style="font-size:12px;color:#6b7280;">Select players to analyze a trade</div>',
            )

            # Recent transactions card
            try:
                _txns = yds.get_transactions()
                if not _txns.empty:
                    _txn_rows = ""
                    _txn_limit = min(8, len(_txns))
                    for _ti in range(_txn_limit):
                        _txr = _txns.iloc[_ti]
                        _txtype = str(_txr.get("type", "")).replace("_", " ").title()
                        _txplayer = str(_txr.get("player_name", ""))[:20]
                        _txdest = str(_txr.get("team_to", ""))[:15]
                        _txlabel = f"{_txtype} → {_txdest}" if _txdest else _txtype
                        _txn_rows += (
                            f'<div style="padding:2px 0;font-size:11px!important;'
                            f'font-family:IBM Plex Mono,monospace!important">'
                            f'<span style="color:{T["tx"]}!important;font-weight:600!important">'
                            f"{_txplayer}</span> "
                            f'<span style="color:{T["tx2"]}!important">{_txlabel}</span></div>'
                        )
                    render_context_card("Recent Transactions", _txn_rows)
            except Exception:
                pass  # Transactions are optional — don't break the page

            # Data freshness card
            render_data_freshness_card()

        with main:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("You Give")
                # Use the actual roster for the dropdown — player_pool may not
                # contain all roster players (only those with projections).
                if not user_roster.empty and "name" in user_roster.columns:
                    give_options = sorted(user_roster["name"].dropna().unique().tolist())
                else:
                    give_options = sorted(pool[pool["player_id"].isin(user_roster_ids)]["player_name"].tolist())
                giving_names = st.multiselect(
                    "Select players to trade away",
                    options=give_options,
                    key="giving",
                )
            with col2:
                st.subheader("You Receive")
                # Build tradeable universe: players rostered by OTHER teams
                # (not FAs — you trade with opponents, not the waiver wire)
                other_team_pids = set()
                all_teams = rosters["team_name"].unique()
                for tn in all_teams:
                    if str(tn) == str(user_team_name):
                        continue
                    team_rows = rosters[rosters["team_name"] == tn]
                    for _, tr in team_rows.iterrows():
                        pname = tr.get("player_name") or tr.get("name", "")
                        matched = name_to_id.get(pname)
                        if matched is not None:
                            other_team_pids.add(matched)
                        else:
                            other_team_pids.add(tr.get("player_id"))

                if other_team_pids:
                    other_players = pool[pool["player_id"].isin(other_team_pids)]
                else:
                    # Fallback: show only players on OTHER teams' rosters
                    # (you can only trade for players someone owns)
                    from src.database import get_all_rostered_player_ids

                    _all_rostered = get_all_rostered_player_ids(rosters)
                    _other_team_ids = _all_rostered - set(user_roster_ids)
                    if _other_team_ids:
                        other_players = pool[pool["player_id"].isin(_other_team_ids)]
                    else:
                        # No rosters loaded at all — show top players as last resort
                        other_players = pool[~pool["player_id"].isin(user_roster_ids)].head(500)

                receive_options = sorted(other_players["player_name"].dropna().unique().tolist())
                receiving_names = st.multiselect(
                    "Select players to receive",
                    options=receive_options,
                    key="receiving",
                )

            if st.button("Analyze Trade", type="primary", width="stretch"):
                if not giving_names or not receiving_names:
                    st.error("Select at least one player on each side.")
                else:
                    giving_ids = [name_to_id[n] for n in giving_names if n in name_to_id]
                    receiving_ids = [name_to_id[n] for n in receiving_names if n in name_to_id]

                    if not giving_ids or not receiving_ids:
                        st.error("One or more selected players could not be matched. Please reselect.")
                        st.stop()

                    trade_progress = st.progress(0, text="Computing category impacts...")

                    # Try the new Phase 1 engine first, fall back to legacy
                    try:
                        from src.engine.output.trade_evaluator import evaluate_trade

                        trade_progress.progress(20, text="Running marginal elasticity analysis...")
                        trade_progress.progress(40, text="Computing category gaps and punt detection...")
                        trade_progress.progress(60, text="Optimizing lineup assignments...")

                        from src.validation.dynamic_context import compute_weeks_remaining

                        # 2026-05-23: load Yahoo schedule + league rosters + current
                        # wins for Features 2 (weekly matrix) + 3 (playoff sim).
                        # Best-effort — if any source fails we still produce the
                        # core trade evaluation; the affected feature is skipped.
                        _weekly_schedule: dict[int, str] | None = None
                        _league_rosters: dict[str, list[int]] | None = None
                        _current_wins: dict[str, int] | None = None
                        # Full Schedule Phase (2026-05-24): also load
                        # full league schedule so playoff sim can use
                        # per-team-per-week opponent matchups instead of
                        # Binomial(N, avg_p) approximation.
                        _full_league_schedule = None
                        try:
                            from src.database import (
                                get_connection,
                                load_league_schedule,
                                load_league_schedule_full,
                                load_league_standings,
                            )

                            _weekly_schedule = load_league_schedule() or None
                            _full_league_schedule = load_league_schedule_full() or None
                            _conn = get_connection()
                            try:
                                import pandas as pd

                                _lr_df = pd.read_sql_query(
                                    "SELECT team_name, player_id FROM league_rosters WHERE player_id IS NOT NULL",
                                    _conn,
                                )
                            finally:
                                _conn.close()
                            if not _lr_df.empty:
                                _league_rosters = {}
                                for _tn, _grp in _lr_df.groupby("team_name"):
                                    _league_rosters[str(_tn)] = _grp["player_id"].astype(int).tolist()
                            _standings = load_league_standings()
                            if not _standings.empty:
                                _wins_rows = _standings[_standings["category"] == "WINS"]
                                if not _wins_rows.empty:
                                    _current_wins = {
                                        str(r["team_name"]): int(r["total"]) for _, r in _wins_rows.iterrows()
                                    }
                        except Exception as _ctx_exc:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Trade Analyzer: failed to load schedule/rosters/wins "
                                "for Features 2/3 wiring (%s) — proceeding without",
                                _ctx_exc,
                            )

                        # Both Features 2 + 3 require schedule + rosters; F3 also
                        # needs current_wins + user_team_name. Wire what we have.
                        _enable_wm = bool(_weekly_schedule and _league_rosters)
                        _enable_ps = bool(_weekly_schedule and _league_rosters and _current_wins and user_team_name)

                        result = evaluate_trade(
                            giving_ids=giving_ids,
                            receiving_ids=receiving_ids,
                            user_roster_ids=user_roster_ids,
                            player_pool=pool,
                            config=config,
                            user_team_name=user_team_name,
                            weeks_remaining=compute_weeks_remaining(),
                            enable_weekly_matrix=_enable_wm,
                            enable_playoff_sim=_enable_ps,
                            weekly_schedule=_weekly_schedule,
                            league_rosters=_league_rosters,
                            current_wins=_current_wins,
                            playoff_n_sims=20_000,
                            full_league_schedule=_full_league_schedule,
                        )
                        engine_used = "phase1"
                    except Exception as e:
                        import logging
                        import traceback

                        _tb = traceback.format_exc()
                        logging.getLogger(__name__).warning(
                            "Phase 1 engine failed, falling back to legacy: %s\n%s", e, _tb
                        )
                        st.warning(
                            f"Phase 1 engine failed ({type(e).__name__}: {e}), using legacy analyzer. Details in console."
                        )
                        # Fall back to legacy analyzer
                        from src.in_season import analyze_trade

                        trade_progress.progress(40, text="Running Monte Carlo simulation (200 iterations)...")
                        result = analyze_trade(
                            giving_ids=giving_ids,
                            receiving_ids=receiving_ids,
                            user_roster_ids=user_roster_ids,
                            player_pool=pool,
                            config=config,
                        )
                        engine_used = "legacy"

                    trade_progress.progress(100, text="Trade analysis complete!")
                    time.sleep(0.3)
                    trade_progress.empty()

                    # Update context panel with trade summary
                    with ctx:
                        verdict_color = T["ok"] if result["verdict"] == "ACCEPT" else T["danger"]
                        grade_val = result.get("grade", "")
                        grade_display = f'<div style="font-size:28px;font-family:Bebas Neue,sans-serif;color:{verdict_color};letter-spacing:2px;">{result["verdict"]}</div>'
                        if grade_val:
                            grade_display += f'<div style="font-size:20px;font-family:Bebas Neue,sans-serif;color:{verdict_color};">{grade_val}</div>'
                        grade_display += f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">{result["confidence_pct"]:.2f}% confidence</div>'
                        render_context_card("Trade Verdict", grade_display)

                        if engine_used == "phase1" and result.get("punt_categories"):
                            punt_list = "".join(
                                f'<div style="font-size:12px;color:#6b7280;padding:2px 0;">{cat}</div>'
                                for cat in result["punt_categories"]
                            )
                            render_context_card(
                                "Punted Categories",
                                f'<div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Zero-weighted in evaluation</div>{punt_list}',
                            )

                        surplus = (
                            result.get("surplus_sgp") if engine_used == "phase1" else result.get("total_sgp_change", 0)
                        )
                        surplus_label = (
                            "Surplus Standings Gained Points"
                            if engine_used == "phase1"
                            else "Total Standings Gained Points Change"
                        )
                        surplus_color = T["ok"] if (surplus or 0) >= 0 else T["danger"]
                        render_context_card(
                            surplus_label,
                            f'<div style="font-size:20px;font-family:Bebas Neue,sans-serif;color:{surplus_color};">{surplus:+.2f}</div>',
                        )

                    # ── Feature 3 (2026-05-23): Playoff + Championship probability ──
                    # Per the Enhanced Trade Engine report Q(a), this is the engine's
                    # PRIMARY objective. Render ABOVE the SGP-grade verdict so the user
                    # sees title-odds impact first, with SGP as the interpretable diagnostic.
                    if "playoff_sim" in result:
                        ps = result["playoff_sim"]
                        before = ps.get("before", {})
                        after = ps.get("after", {})
                        d_play = result.get("delta_playoff_prob", 0.0)
                        d_champ = result.get("delta_champ_prob", 0.0)

                        st.markdown(
                            '<div style="font-family:Bebas Neue,sans-serif;font-size:18px;'
                            'color:#9ca3af;letter-spacing:2px;margin-top:8px;">'
                            "PRIMARY OBJECTIVE — Δ TITLE ODDS</div>",
                            unsafe_allow_html=True,
                        )
                        pc1, pc2, pc3 = st.columns(3)
                        pc1.metric(
                            "P(make playoffs / top-4)",
                            f"{after.get('playoff_prob', 0):.1%}",
                            delta=f"{d_play:+.2%}",
                            delta_color="normal",
                            help=(
                                "Monte-Carlo simulated probability of finishing top-4. "
                                "Delta = after-trade minus before-trade. Negative = trade "
                                "hurts your playoff odds. Report Section B.10 + Q(a)."
                            ),
                        )
                        pc2.metric(
                            "P(win championship)",
                            f"{after.get('champ_prob', 0):.1%}",
                            delta=f"{d_champ:+.2%}",
                            delta_color="normal",
                            help=(
                                "Joint probability of (a) finishing top-4 AND (b) winning "
                                "the 2-round bracket (1v4, 2v3, re-seeded final). "
                                "20,000 MC sims with paired-MC discipline."
                            ),
                        )
                        pc3.metric(
                            "E[additional regular-season wins]",
                            f"{after.get('mean_regular_season_wins', 0):.1f}",
                            delta=f"{after.get('mean_regular_season_wins', 0) - before.get('mean_regular_season_wins', 0):+.1f}",
                            delta_color="normal",
                            help=(
                                "Expected matchup wins over remaining weeks (per the schedule-aware weekly H2H matrix)."
                            ),
                        )

                        # ── CARA mean-CVaR utility (report Section B.9) ──
                        # Risk-adjusted utility on per-sim Δchamp_prob deltas.
                        # CARA = E[Δchamp] - λ/2 × Var[Δchamp] with λ=0.15.
                        # CVaR_20 = average outcome in worst 20% of sims.
                        # Together they answer: "Is the expected gain worth
                        # the downside risk?"
                        _cara = ps.get("cara_utility")
                        _cvar20 = ps.get("cvar20_champ")
                        _var_champ = ps.get("var_champ")
                        _lambda = ps.get("lambda_risk_aversion", 0.15)
                        if _cara is not None and _cvar20 is not None:
                            cara1, cara2, cara3 = st.columns(3)
                            cara1.metric(
                                "CARA utility (risk-adjusted)",
                                f"{_cara:+.4f}",
                                help=(
                                    f"E[Δchamp] − (λ/2) × Var[Δchamp] with λ={_lambda}. "
                                    f"Penalizes high-variance trades. > 0 → accept under "
                                    f"CARA preferences; < 0 → variance penalty exceeds "
                                    f"expected gain. Report Section B.9."
                                ),
                            )
                            cara2.metric(
                                "CVaR₂₀ (worst-20% Δchamp)",
                                f"{_cvar20:+.2%}",
                                help=(
                                    "Conditional Value-at-Risk at 20%: average "
                                    "championship-prob delta across the WORST 20% "
                                    "of simulated futures. Negative = trade carries "
                                    "real downside in unlucky scenarios. Report B.9."
                                ),
                            )
                            cara3.metric(
                                "Var[Δchamp]",
                                f"{_var_champ:.4f}" if _var_champ is not None else "—",
                                help=(
                                    "Variance of per-sim championship-prob delta. "
                                    "0 = trade always produces identical bracket "
                                    "outcome; >0 = bracket outcome varies across sims."
                                ),
                            )

                            # λ-sensitivity sweep (report Sections H.10 / C.7).
                            # Shows how the risk-adjusted utility shifts with the
                            # manager's risk stance: 0.05 = risk-seeking (must
                            # gamble for a playoff push), 0.30 = risk-averse
                            # (protecting a strong position).
                            _sweep = ps.get("cara_utility_sweep") or {}
                            if _sweep:
                                _parts = []
                                _labels = {0.05: "risk-seeking", 0.15: "central", 0.30: "risk-averse"}
                                for _lam in sorted(_sweep):
                                    _parts.append(f"λ={_lam} ({_labels.get(_lam, '')}): **{_sweep[_lam]:+.4f}**")
                                st.caption("CARA utility across risk-aversion λ — " + " · ".join(_parts))

                    # Verdict banner
                    if result["verdict"] == "ACCEPT":
                        color = T["ok"]
                        icon = PAGE_ICONS["accept"]
                    else:
                        color = T["danger"]
                        icon = PAGE_ICONS["reject"]

                    # Show trade grade if available (Phase 1 engine)
                    grade_html = ""
                    if "grade" in result:
                        grade = result["grade"]
                        # Color the grade based on quality
                        if grade.startswith("A"):
                            grade_color = T["ok"]
                        elif grade.startswith("B"):
                            grade_color = T["sky"]
                        elif grade.startswith("C"):
                            grade_color = T["hot"]
                        else:
                            grade_color = T["danger"]
                        grade_html = (
                            f'<span style="font-family:Bebas Neue,sans-serif;font-size:36px;'
                            f"color:{grade_color};letter-spacing:3px;margin-left:16px;"
                            f'font-weight:bold;">{grade}</span>'
                        )

                    st.markdown(
                        f'<div class="glass" style="border:2px solid {color};'
                        f"padding:20px;text-align:center;margin:16px 0;"
                        f'animation:slideUp 0.4s ease-out both;">'
                        f"{icon}"
                        f'<span style="font-family:Bebas Neue,sans-serif;font-size:28px;color:{color};'
                        f'letter-spacing:2px;margin-left:12px;">{result["verdict"]}</span>'
                        f"{grade_html}"
                        f'<span style="color:{T["tx2"]};margin-left:12px;font-size:18px;">'
                        f"{result['confidence_pct']:.2f}% confidence</span></div>",
                        unsafe_allow_html=True,
                    )
                    verdict_key = "trade_verdict" if engine_used == "phase1" else "trade_verdict_legacy"
                    st.caption(METRIC_TOOLTIPS[verdict_key])

                    # ── Feature 1 (2026-05-23): IP-floor status indicator ──
                    # When projected weekly IP drops below the 20 IP/week Yahoo floor,
                    # surface it as an inline warning. Adapts text based on whether the
                    # trade caused the shortfall (delta > 0) or it pre-existed (delta == 0).
                    _ip_detail = result.get("ip_floor_detail", {}) or {}
                    if _ip_detail.get("below_floor"):
                        _post_ip = _ip_detail.get("after_weekly_ip", 0.0)
                        _floor = _ip_detail.get("threshold_ip_per_week", 20.0)
                        _delta_pen = _ip_detail.get("delta_penalty", 0.0)
                        _wks_rem = _ip_detail.get("weeks_remaining", "?")
                        if _delta_pen > 0:
                            st.error(
                                f"⚠ **IP floor breach (trade-caused)** — post-trade "
                                f"weekly IP **{_post_ip:.1f}** below the {_floor:.0f} IP/week "
                                f"Yahoo floor. SGP penalty of **{_delta_pen:.2f}** applied. "
                                f"At this rate, Yahoo will convert pitching-cat shortfall to "
                                f"losses. (Using {_wks_rem} weeks remaining.)"
                            )
                        else:
                            st.warning(
                                f"⚠ **IP floor warning (pre-existing)** — weekly IP "
                                f"**{_post_ip:.1f}** is already below the {_floor:.0f} IP/week "
                                f"Yahoo floor. This trade doesn't worsen the situation, but "
                                f"your pitching cats are at risk of converting to losses. "
                                f"Consider adding pitching volume via FA pickups."
                            )

                    # ── Secondary valuation diagnostics (report B.8 / B.5-C.2 / B.7 / H.2) ──
                    # These do NOT drive the grade — they are the report's
                    # sanity-check + transparency outputs alongside the primary
                    # roster-context SGP surplus.
                    if engine_used == "phase1":
                        _vorp = result.get("delta_vorp_prp")
                        _gscore = result.get("delta_g_score")
                        _spec_pen = result.get("specialist_cap_penalty", 0.0) or 0.0
                        _drl_chain = result.get("drl_replacement_chain") or []
                        if _vorp is not None or _gscore is not None or _spec_pen or _drl_chain:
                            with st.expander("Secondary diagnostics (VORP · G-score · DRL · specialist cap)"):
                                dcol1, dcol2 = st.columns(2)
                                dcol1.metric(
                                    "Δ VORP (league-wide)",
                                    f"{_vorp:+.2f}" if _vorp is not None else "—",
                                    help=(
                                        "Secondary league-wide fairness check (report B.8): "
                                        "Σ VORP(receiving) − Σ VORP(giving) using standard "
                                        "positional replacement levels. Independent of YOUR "
                                        "roster — answers 'is this trade fair league-wide?'"
                                    ),
                                )
                                dcol2.metric(
                                    "Δ G-score (variance-aware)",
                                    f"{_gscore:+.2f}" if _gscore is not None else "—",
                                    help=(
                                        "Rosenof G-score delta (report B.5/C.2): like a "
                                        "z-score delta but with the team-level weekly-variance "
                                        "term, so volatile boom-bust profiles are discounted "
                                        "relative to steady producers."
                                    ),
                                )
                                if _spec_pen > 0:
                                    _capped = (result.get("specialist_cap_detail", {}) or {}).get("capped", {})
                                    _names = ", ".join(_capped.keys())
                                    st.warning(
                                        f"Specialist cap (report H.2): **{_spec_pen:.2f}** SGP of "
                                        f"single-category credit removed for {_names} — a single "
                                        f"player shouldn't be valued on more than 25% of one "
                                        f"category's standings range."
                                    )
                                if _drl_chain:
                                    st.caption("**Replacement chain** (the bench/FA moves this trade triggers):")
                                    for _entry in _drl_chain:
                                        _verb = "Pick up" if _entry["action"] == "pickup" else "Drop"
                                        st.caption(
                                            f"• {_verb} **{_entry['name']}** "
                                            f"({_entry['positions']}, {_entry['sgp']:+.2f} SGP, "
                                            f"via {_entry['source']})"
                                        )

                    # Warn when standings data is absent or not yet meaningful
                    _standings_state = _standings_data_state()
                    if engine_used == "phase1":
                        if _standings_state == "all_zero":
                            st.caption(
                                "League standings have not yet populated. Trade analysis is using "
                                "league-average category weights instead of standings-based weights. "
                                "Results will be more accurate after week 1."
                            )
                        elif _standings_state == "empty" or not result.get("category_analysis"):
                            st.warning(
                                "No league standings data available. All categories are weighted equally, "
                                "so the analysis cannot account for your team's strategic position "
                                "(punt detection, marginal elasticity). Sync your Yahoo league for "
                                "a more accurate evaluation."
                            )

                    # Analytics transparency badge (Phase 2 wiring)
                    if engine_used == "phase1" and "analytics_context" in result:
                        from src.ui_analytics_badge import render_analytics_badge

                        render_analytics_badge(result["analytics_context"])

                    # Metrics row — different layout for Phase 1 vs legacy
                    if engine_used == "phase1":
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric(
                            "Trade Grade",
                            result.get("grade", "N/A"),
                        )
                        col2.metric(
                            "Surplus Standings Gained Points",
                            format_stat(result.get("surplus_sgp", 0), "SGP"),
                            help=METRIC_TOOLTIPS["sgp"],
                        )
                        # Roster move indicator (drop or pickup for uneven trades)
                        roster_move = "None"
                        if result.get("drop_candidate"):
                            roster_move = f"Drop {result['drop_candidate']}"
                        elif result.get("fa_pickup"):
                            roster_move = f"Add {result['fa_pickup']}"
                        col3.metric(
                            "Roster Move",
                            roster_move,
                            help=METRIC_TOOLTIPS.get("roster_move", ""),
                        )
                        col4.metric(
                            "Replacement Penalty",
                            format_stat(result.get("replacement_penalty", 0), "SGP"),
                            help=METRIC_TOOLTIPS["replacement_penalty"],
                        )
                        col5.metric(
                            "Punted Categories",
                            str(len(result.get("punt_categories", []))),
                            help="Categories where gaining standings positions is impossible. "
                            "These are zero-weighted in the trade evaluation.",
                        )

                        # Show punted categories if any
                        if result.get("punt_categories"):
                            punt_text = ", ".join(result["punt_categories"])
                            st.info(f"Punted categories (zero-weighted): {punt_text}")

                        # ── Three-horizon split (report Q(b)) ──
                        # Per Enhanced Trade Engine report Q(b): the
                        # decision-relevant horizons are pre-deadline,
                        # regular-season-finish, and playoff-window. Slice
                        # the weekly matrix by these horizons so the user
                        # can see asymmetric impact (e.g., trade hurts you
                        # for 12 regular weeks but helps the playoff window).
                        _wm = result.get("weekly_matrix")
                        if _wm and "summary" in _wm and not _wm["summary"].empty:
                            _summary_for_horizons = _wm["summary"]
                            # Yahoo H2H Cats standard: trade deadline week ~19,
                            # regular season ends week ~24, playoffs weeks 25-26.
                            # Use the user's actual schedule weeks (clamped).
                            _weeks_in_sched = list(_summary_for_horizons.index)
                            _TRADE_DEADLINE_WEEK = 19
                            _REGULAR_END_WEEK = 24
                            _pre_deadline = [w for w in _weeks_in_sched if w <= _TRADE_DEADLINE_WEEK]
                            _reg_season = [w for w in _weeks_in_sched if w <= _REGULAR_END_WEEK]
                            _playoff = [w for w in _weeks_in_sched if w > _REGULAR_END_WEEK]

                            def _sum_delta(weeks: list) -> float:
                                if not weeks:
                                    return 0.0
                                return float(_summary_for_horizons.loc[weeks, "delta_cat_wins"].sum())

                            _pd_delta = _sum_delta(_pre_deadline)
                            _rs_delta = _sum_delta(_reg_season)
                            _po_delta = _sum_delta(_playoff)

                            st.markdown(
                                '<div style="font-family:Bebas Neue,sans-serif;font-size:16px;'
                                'color:#9ca3af;letter-spacing:2px;margin-top:8px;">'
                                "THREE-HORIZON IMPACT (report Q(b))</div>",
                                unsafe_allow_html=True,
                            )
                            h1, h2, h3 = st.columns(3)
                            h1.metric(
                                f"Pre-trade-deadline ({len(_pre_deadline)} wks)",
                                "Σ Δ cat-wins",
                                delta=f"{_pd_delta:+.2f}",
                                delta_color="normal",
                                help=(
                                    f"Cumulative cat-wins delta across weeks "
                                    f"{_pre_deadline[0] if _pre_deadline else '?'}-"
                                    f"{_pre_deadline[-1] if _pre_deadline else '?'} "
                                    f"(through Yahoo trade deadline ~week 19). After this "
                                    f"window you cannot reverse the trade if it goes poorly."
                                ),
                            )
                            h2.metric(
                                f"Full regular season ({len(_reg_season)} wks)",
                                "Σ Δ cat-wins",
                                delta=f"{_rs_delta:+.2f}",
                                delta_color="normal",
                                help=(
                                    "Cumulative cat-wins delta across all remaining "
                                    "regular-season weeks (through ~week 24)."
                                ),
                            )
                            if _playoff:
                                h3.metric(
                                    f"Playoff window ({len(_playoff)} wks)",
                                    "Σ Δ cat-wins",
                                    delta=f"{_po_delta:+.2f}",
                                    delta_color="normal",
                                    help=(
                                        f"Cumulative cat-wins delta across playoff "
                                        f"weeks {_playoff[0]}-{_playoff[-1]}. "
                                        f"Conditional on making the playoffs."
                                    ),
                                )
                            else:
                                h3.metric(
                                    "Playoff window",
                                    "Not in schedule",
                                    help=(
                                        "Yahoo schedule cache doesn't yet include "
                                        "playoff weeks (typically populated late season)."
                                    ),
                                )

                        # ── Feature 2 (2026-05-23): Weekly H2H matrix ──
                        # Per report Section B.5 — for each remaining week + cat,
                        # P(win) before vs after the trade. Negative delta = trade
                        # hurts you that week. Surfaces playoff-window asymmetry
                        # the SGP scalar can't see.
                        if _wm and "summary" in _wm:
                            with st.expander(
                                f"📅 Weekly H2H impact (Feature 2 — {len(_wm['summary'])} weeks)",
                                expanded=False,
                            ):
                                st.caption(
                                    f"Per-week expected category wins (of 12) before and after the trade, "
                                    f"against the actual Yahoo schedule. Method: {_wm.get('method', '?')}. "
                                    f"Use the heatmap below to spot playoff-window vs regular-season asymmetry."
                                )
                                _summary = _wm["summary"].copy()
                                # Round for display
                                if "expected_cat_wins_before" in _summary.columns:
                                    _summary["expected_cat_wins_before"] = _summary["expected_cat_wins_before"].round(2)
                                    _summary["expected_cat_wins_after"] = _summary["expected_cat_wins_after"].round(2)
                                    _summary["delta_cat_wins"] = _summary["delta_cat_wins"].round(2)
                                st.dataframe(_summary, use_container_width=True)

                                # Heatmap of the delta matrix (weeks × cats)
                                try:
                                    import plotly.express as px

                                    _delta_df = _wm["delta"]
                                    fig_hm = px.imshow(
                                        _delta_df.astype(float),
                                        labels=dict(x="Category", y="Week", color="Δ win probability"),
                                        x=list(_delta_df.columns),
                                        y=list(_delta_df.index),
                                        color_continuous_scale="RdYlGn",
                                        color_continuous_midpoint=0,
                                        aspect="auto",
                                    )
                                    fig_hm.update_layout(
                                        height=480,
                                        margin=dict(l=40, r=40, t=30, b=40),
                                        template="plotly_white",
                                    )
                                    st.plotly_chart(fig_hm, use_container_width=True)
                                    st.caption(
                                        "Green = trade helps you in that week+category. "
                                        "Red = trade hurts. Look for asymmetry between "
                                        "regular-season weeks and the playoff window (weeks 24-26)."
                                    )
                                except ImportError:
                                    pass  # plotly not available — table alone is fine

                                # Worst/best week callouts
                                _sorted = _summary.sort_values("delta_cat_wins")
                                _worst = _sorted.iloc[0]
                                _best = _sorted.iloc[-1]
                                wcol, bcol = st.columns(2)
                                wcol.metric(
                                    f"Worst week impact (vs {_worst.get('opponent', '?')})",
                                    f"week {_worst.name}",
                                    delta=f"{_worst['delta_cat_wins']:+.2f} cat-wins",
                                    delta_color="normal",
                                )
                                bcol.metric(
                                    f"Best week impact (vs {_best.get('opponent', '?')})",
                                    f"week {_best.name}",
                                    delta=f"{_best['delta_cat_wins']:+.2f} cat-wins",
                                    delta_color="normal",
                                )
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Total Standings Gained Points Change",
                            format_stat(result["total_sgp_change"], "SGP"),
                            help=METRIC_TOOLTIPS["sgp"],
                        )
                        col2.metric(
                            "Monte Carlo Mean",
                            format_stat(result["mc_mean"], "SGP"),
                            help=METRIC_TOOLTIPS["mc_mean"],
                        )
                        col3.metric(
                            "Monte Carlo Standard Deviation",
                            f"{result['mc_std']:.2f}",
                            help=METRIC_TOOLTIPS["mc_std"],
                        )

                    # ── Acceptance Analysis Panel ───────────────────────────
                    # Shows behavioral intelligence: will the opponent accept?
                    try:
                        from src.trade_finder import (
                            acceptance_label,
                            compute_adp_fairness,
                            estimate_acceptance_probability,
                        )

                        st.subheader("Acceptance Analysis")
                        st.caption(
                            "Behavioral model: estimates whether the opponent would accept this trade "
                            "based on draft capital, expert rankings, loss aversion, and opponent needs."
                        )

                        # Compute ADP fairness for each pair (average across all give/recv pairs)
                        adp_scores = []
                        for gid in giving_ids:
                            for rid in receiving_ids:
                                adp_scores.append(compute_adp_fairness(gid, rid, pool))
                        avg_adp_fairness = sum(adp_scores) / len(adp_scores) if adp_scores else 0.5

                        # ECR fairness — use pool (Unified Services: load_player_pool enriches with consensus_rank)
                        ecr_ranks_local = {}
                        try:
                            if "consensus_rank" in pool.columns:
                                _ecr_pool = pool[["player_id", "consensus_rank"]].dropna(subset=["consensus_rank"])
                                ecr_ranks_local = dict(
                                    zip(
                                        _ecr_pool["player_id"].astype(int),
                                        _ecr_pool["consensus_rank"].astype(int),
                                    )
                                )
                        except Exception:
                            pass

                        ecr_scores = []
                        for gid in giving_ids:
                            for rid in receiving_ids:
                                g_ecr = ecr_ranks_local.get(gid)
                                r_ecr = ecr_ranks_local.get(rid)
                                if g_ecr and r_ecr:
                                    ecr_scores.append((min(g_ecr, r_ecr) / max(g_ecr, r_ecr, 1)) ** 0.5)
                        avg_ecr_fairness = sum(ecr_scores) / len(ecr_scores) if ecr_scores else 0.5

                        # Opponent needs (if we can identify the opponent)
                        opp_need_match = 0.5
                        opp_willingness = 0.5
                        opp_rank = None
                        try:
                            from src.opponent_trade_analysis import (
                                compute_opponent_needs,
                                get_opponent_archetype,
                            )

                            # Try to identify opponent from receiving players' team
                            _recv_teams = set()
                            for rid in receiving_ids:
                                _rp = pool[pool["player_id"] == rid]
                                if not _rp.empty:
                                    _rt = rosters[rosters["player_id"] == rid]
                                    if _rt.empty:
                                        # Try name matching (Yahoo IDs vs MLB Stats API IDs)
                                        _pname = _rp.iloc[0].get("player_name", _rp.iloc[0].get("name", ""))
                                        if _pname:
                                            _name_col = (
                                                rosters["name"]
                                                if "name" in rosters.columns
                                                else rosters.get("player_name", pd.Series(dtype=str))
                                            )
                                            _rt = rosters[_name_col.str.strip() == str(_pname).strip()]
                                    if not _rt.empty:
                                        _recv_teams.add(_rt.iloc[0].get("team_name", ""))

                            if _recv_teams:
                                opp_team = list(_recv_teams)[0]
                                from src.standings_utils import get_all_team_totals

                                standings = yds.get_standings()
                                all_team_totals = get_all_team_totals()
                                if all_team_totals:
                                    opp_needs = compute_opponent_needs(opp_team, all_team_totals)
                                    arch = get_opponent_archetype(opp_team)
                                    opp_willingness = arch.get("trade_willingness", 0.5)

                                    # Compute opponent rank
                                    opp_standings = standings[standings["team_name"] == opp_team]
                                    if not opp_standings.empty:
                                        opp_rank = int(opp_standings.iloc[0].get("rank", 6))

                                    # Compute need match
                                    opp_weak = [c for c, info in opp_needs.items() if info.get("rank", 6) >= 8]
                                    if opp_weak:
                                        helps = 0
                                        for gid in giving_ids:
                                            gp = pool[pool["player_id"] == gid]
                                            if not gp.empty:
                                                for c in opp_weak:
                                                    if float(gp.iloc[0].get(c.lower(), 0) or 0) > 0:
                                                        helps += 1
                                        opp_need_match = helps / (len(opp_weak) * len(giving_ids)) if opp_weak else 0.5
                        except Exception:
                            pass

                        # Compute acceptance probability
                        user_sgp = result.get("surplus_sgp", result.get("total_sgp_change", 0))
                        opp_sgp = -user_sgp * 0.5  # Rough estimate: opponent loses ~half what user gains
                        need_match = min(1.0, max(0.0, (opp_sgp + 1.0) / 2.0))

                        p_accept = estimate_acceptance_probability(
                            user_gain_sgp=user_sgp,
                            opponent_gain_sgp=opp_sgp,
                            need_match_score=need_match,
                            adp_fairness=avg_adp_fairness,
                            opponent_need_match=opp_need_match,
                            opponent_standings_rank=opp_rank,
                            opponent_trade_willingness=opp_willingness,
                        )

                        # Display metrics
                        ac1, ac2, ac3, ac4 = st.columns(4)

                        ac1.metric(
                            "Acceptance Probability",
                            f"{p_accept:.0%}",
                            help="Estimated probability the opponent accepts. Uses loss aversion (1.8x), "
                            "ADP fairness, ECR rankings, opponent needs, and archetype willingness.",
                        )
                        ac2.metric(
                            "ADP Fairness",
                            f"{avg_adp_fairness:.0%}",
                            help="How well draft positions match. 100% = same draft round. "
                            "Uses league draft data with generic ADP fallback.",
                        )
                        ac3.metric(
                            "ECR Fairness",
                            f"{avg_ecr_fairness:.0%}",
                            help="Expert Consensus Ranking alignment. 100% = same rank. "
                            "Based on 7-source Trimmed Borda Count consensus.",
                        )
                        ac4.metric(
                            "Acceptance Tier",
                            acceptance_label(p_accept),
                            help="High (60%+), Medium (30-60%), Low (<30%)",
                        )

                        # Show ADP detail per player pair
                        if len(giving_ids) <= 3 and len(receiving_ids) <= 3:
                            adp_detail = []
                            for gid in giving_ids:
                                gp = pool[pool["player_id"] == gid]
                                gname = gp.iloc[0].get("player_name", "?") if not gp.empty else "?"
                                g_ecr = ecr_ranks_local.get(gid, "N/A")
                                for rid in receiving_ids:
                                    rp = pool[pool["player_id"] == rid]
                                    rname = rp.iloc[0].get("player_name", "?") if not rp.empty else "?"
                                    r_ecr = ecr_ranks_local.get(rid, "N/A")
                                    af = compute_adp_fairness(gid, rid, pool)
                                    adp_detail.append(
                                        {
                                            "Give": gname,
                                            "Give ECR": f"#{g_ecr}" if isinstance(g_ecr, int) else g_ecr,
                                            "Receive": rname,
                                            "Recv ECR": f"#{r_ecr}" if isinstance(r_ecr, int) else r_ecr,
                                            "ADP Fair": f"{af:.0%}",
                                        }
                                    )
                            if adp_detail:
                                with st.expander("ADP and ECR Detail", expanded=False):
                                    st.dataframe(pd.DataFrame(adp_detail), hide_index=True, width="stretch")

                    except ImportError:
                        pass  # trade_finder not available — skip acceptance panel
                    except Exception as _acc_err:
                        logging.getLogger(__name__).warning("Acceptance analysis failed: %s", _acc_err)

                    # Category impact table — enhanced for Phase 1
                    st.subheader("Category Impact")
                    if engine_used == "phase1" and result.get("category_analysis"):
                        impact_rows = []
                        for cat, sgp_val in result["category_impact"].items():
                            analysis = result["category_analysis"].get(cat, {})
                            rank = analysis.get("rank", "-")
                            is_punt = analysis.get("is_punt", False)
                            gap = analysis.get("gap_to_next", 0)
                            status = "PUNT" if is_punt else f"Rank {rank}"

                            impact_rows.append(
                                {
                                    "Category": cat,
                                    "Standings Gained Points Change": format_stat(sgp_val, "SGP"),
                                    "Your Rank": status,
                                    "Gap to Next": f"{gap:.2f}" if not is_punt else "-",
                                }
                            )
                        render_compact_table(pd.DataFrame(impact_rows))
                    else:
                        impact_df = pd.DataFrame(
                            [
                                {"Category": cat, "Standings Gained Points Change": format_stat(val, "SGP")}
                                for cat, val in result["category_impact"].items()
                            ]
                        )
                        render_compact_table(impact_df)

                    # Category Replacement Difficulty
                    if engine_used == "phase1" and result.get("replacement_detail"):
                        detail = result["replacement_detail"]
                        # Only show section if there's at least one non-skipped category with a penalty
                        active_entries = {
                            cat: d for cat, d in detail.items() if not d.get("skipped") and d.get("sgp_penalty", 0) > 0
                        }
                        if active_entries:
                            st.subheader("Category Replacement Difficulty")
                            st.caption(METRIC_TOOLTIPS["replacement_penalty"])
                            repl_rows = []
                            for cat, d in detail.items():
                                if d.get("skipped"):
                                    continue
                                if d.get("sgp_penalty", 0) <= 0:
                                    continue
                                repl_rows.append(
                                    {
                                        "Category": cat,
                                        "Raw Loss": f"{d['raw_loss']:.2f}",
                                        "Best Free Agent": f"{d.get('best_fa_name', 'N/A')} ({d['best_fa']:.2f})",
                                        "Unrecoverable Gap": f"{d['unrecoverable']:.2f}",
                                        "Standings Gained Points Penalty": format_stat(d["sgp_penalty"], "SGP"),
                                    }
                                )
                            if repl_rows:
                                render_styled_table(pd.DataFrame(repl_rows))

                    # Risk flags
                    if result["risk_flags"]:
                        st.subheader("Risk Flags")
                        for flag in result["risk_flags"]:
                            st.warning(flag)

                    # Show headshots + injury badges for traded players
                    from src.ui_shared import _headshot_img_html

                    trade_col1, trade_col2 = st.columns(2)
                    with trade_col1:
                        st.markdown("**Giving:**")
                        for name in giving_names:
                            p = pool[pool["player_name"] == name]
                            if not p.empty:
                                pid = p.iloc[0]["player_id"]
                                mid = p.iloc[0].get("mlb_id")
                                hs = float(p.iloc[0].get("health_score", 0.85) or 0.85)
                                badge_icon, label = get_injury_badge(hs)
                                shot = _headshot_img_html(mid, size=28)
                                st.markdown(f"{shot}{badge_icon} {name} — {label}", unsafe_allow_html=True)
                    with trade_col2:
                        st.markdown("**Receiving:**")
                        for name in receiving_names:
                            p = pool[pool["player_name"] == name]
                            if not p.empty:
                                pid = p.iloc[0]["player_id"]
                                mid = p.iloc[0].get("mlb_id")
                                hs = float(p.iloc[0].get("health_score", 0.85) or 0.85)
                                badge_icon, label = get_injury_badge(hs)
                                shot = _headshot_img_html(mid, size=28)
                                st.markdown(f"{shot}{badge_icon} {name} — {label}", unsafe_allow_html=True)

                    # Player card selector for traded players
                    _trade_all_names = list(giving_names) + list(receiving_names)
                    _trade_all_ids = []
                    for _tname in _trade_all_names:
                        _tp = pool[pool["player_name"] == _tname]
                        _trade_all_ids.append(int(_tp.iloc[0]["player_id"]) if not _tp.empty else 0)
                    if any(pid != 0 for pid in _trade_all_ids):
                        render_player_select(_trade_all_names, _trade_all_ids, key_suffix="trade")

                    # P10/P90 risk assessment for traded players.
                    # Per Unified Services: source player IDs from the pool (load_player_pool).
                    # Per-system projections must still come from the projections table because
                    # the pool only carries the blended projection; volatility = std across systems.
                    try:
                        from src.database import get_connection

                        # Scope per-system fetch to traded player IDs (sourced from pool above)
                        _vol_ids = [pid for pid in _trade_all_ids if pid != 0]
                        conn = get_connection()
                        try:
                            systems = {}
                            if _vol_ids:
                                _placeholders = ",".join("?" * len(_vol_ids))
                                for sys_name in ["steamer", "zips", "depthcharts"]:
                                    df = pd.read_sql_query(
                                        f"SELECT * FROM projections WHERE system = ? AND player_id IN ({_placeholders})",
                                        conn,
                                        params=(sys_name, *_vol_ids),
                                    )
                                    if not df.empty:
                                        df = coerce_numeric_df(df)
                                        systems[sys_name] = df
                        finally:
                            conn.close()

                        if len(systems) >= 2:
                            vol = compute_projection_volatility(systems)
                            vol = add_process_risk(vol)
                            pct = compute_percentile_projections(base=pool, volatility=vol)
                            p10_df, p90_df = pct.get(10), pct.get(90)
                            if p10_df is not None and p90_df is not None:
                                st.subheader("Upside/Downside Risk")
                                all_names = list(giving_names) + list(receiving_names)
                                risk_rows = []
                                for name in all_names:
                                    p10_row = (
                                        p10_df[p10_df["player_name"] == name]
                                        if "player_name" in p10_df.columns
                                        else pd.DataFrame()
                                    )
                                    p90_row = (
                                        p90_df[p90_df["player_name"] == name]
                                        if "player_name" in p90_df.columns
                                        else pd.DataFrame()
                                    )
                                    if not p10_row.empty and not p90_row.empty:
                                        # Use HR as a representative counting stat for upside/downside display
                                        p10_hr = p10_row.iloc[0].get("hr", 0)
                                        p90_hr = p90_row.iloc[0].get("hr", 0)
                                        p10_avg = p10_row.iloc[0].get("avg", 0)
                                        p90_avg = p90_row.iloc[0].get("avg", 0)
                                        risk_rows.append(
                                            {
                                                "Player": name,
                                                "10th Percentile Home Runs (Floor)": f"{p10_hr:.2f}",
                                                "90th Percentile Home Runs (Ceiling)": f"{p90_hr:.2f}",
                                                "10th Percentile Batting Average (Floor)": f"{p10_avg:.3f}",
                                                "90th Percentile Batting Average (Ceiling)": f"{p90_avg:.3f}",
                                            }
                                        )
                                if risk_rows:
                                    render_styled_table(pd.DataFrame(risk_rows))
                                    st.caption(METRIC_TOOLTIPS["p10_p90"])
                    except Exception:
                        pass  # Graceful degradation

page_timer_footer("Trade Analyzer")
render_feedback_widget("Trade Analyzer")
