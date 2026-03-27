"""Trade Analyzer — Evaluate trade proposals with Phase 1 engine pipeline.

Uses the new engine modules for marginal SGP elasticity, punt detection,
LP lineup optimization, and deterministic grading (A+ to F).
Falls back to the legacy analyzer if the engine is unavailable.
"""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, init_db, load_league_rosters, load_league_standings, load_player_pool
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import (
    METRIC_TOOLTIPS,
    PAGE_ICONS,
    T,
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
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

st.set_page_config(page_title="Heater | Trade Analyzer", page_icon="", layout="wide", initial_sidebar_state="collapsed")


def _standings_data_state() -> str:
    """Return the quality state of the league_standings table.

    Returns:
        "empty"    — table has no rows at all.
        "all_zero" — table has rows but every 'total' value is zero or null
                     (week 1 before any matchup has completed).
        "ok"       — table has meaningful non-zero data.
    """
    try:
        standings = load_league_standings()
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
page_timer_start()

render_page_layout("TRADE ANALYZER", banner_teaser="Analyze a trade below", banner_icon="trade")

# Load data
pool = load_player_pool()
if pool.empty:
    st.warning("No player data. Load sample data or import projections first.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Load health scores from injury history
health_dict = {}
try:
    from src.database import get_connection

    conn = get_connection()
    try:
        injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        injury_df = coerce_numeric_df(injury_df)
    finally:
        conn.close()
    if not injury_df.empty and "player_id" in injury_df.columns:
        for pid, group in injury_df.groupby("player_id"):
            gp = group["games_played"].tolist()
            ga = group["games_available"].tolist()
            health_dict[pid] = compute_health_score(gp, ga)
except Exception:
    pass

# Get user team roster
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_trade"):
            client = st.session_state.get("yahoo_client")
            if client:
                progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                try:
                    progress.progress(30, text="Fetching league standings...")
                    sync_result = client.sync_to_db()
                    progress.progress(100, text="Sync complete!")
                    standings_count = sync_result.get("standings", 0) if sync_result else 0
                    rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                    if rosters_count > 0:
                        st.success(f"Synced {rosters_count} roster entries and {standings_count} standing entries.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(
                            f"Sync completed but Yahoo returned no roster data "
                            f"(standings: {standings_count}). This may mean the league "
                            f"season hasn't started yet on Yahoo, or rosters haven't been set."
                        )
                except Exception as e:
                    progress.empty()
                    st.error(f"Sync failed: {e}")
            else:
                st.error("Yahoo client not found in session. Return to Connect League and reconnect.")
    else:
        st.warning(
            "No league data loaded. Connect your Yahoo league in Connect League, or league data will load automatically on next app launch."
        )
    st.stop()
else:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if user_teams.empty:
        st.warning("No user team identified.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
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
                    # Fallback: top 500 players by projected value (not all 9K)
                    stat_cols = ["hr", "r", "rbi", "sb", "w", "sv", "k"]
                    available_cols = [c for c in stat_cols if c in pool.columns]
                    if available_cols:
                        pool_sorted = pool.copy()
                        pool_sorted["_sort_val"] = pool_sorted[available_cols].sum(axis=1)
                        pool_sorted = pool_sorted.sort_values("_sort_val", ascending=False)
                        other_players = pool_sorted[~pool_sorted["player_id"].isin(user_roster_ids)].head(500)
                    else:
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

                        result = evaluate_trade(
                            giving_ids=giving_ids,
                            receiving_ids=receiving_ids,
                            user_roster_ids=user_roster_ids,
                            player_pool=pool,
                            config=config,
                            user_team_name=user_team_name,
                            weeks_remaining=compute_weeks_remaining(),
                        )
                        engine_used = "phase1"
                    except Exception:
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
                            f"{result.get('surplus_sgp', 0):+.2f}",
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
                            f"{result.get('replacement_penalty', 0):+.2f}",
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
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Total Standings Gained Points Change",
                            f"{result['total_sgp_change']:+.2f}",
                            help=METRIC_TOOLTIPS["sgp"],
                        )
                        col2.metric(
                            "Monte Carlo Mean",
                            f"{result['mc_mean']:+.2f}",
                            help=METRIC_TOOLTIPS["mc_mean"],
                        )
                        col3.metric(
                            "Monte Carlo Standard Deviation",
                            f"{result['mc_std']:.2f}",
                            help=METRIC_TOOLTIPS["mc_std"],
                        )

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
                                    "Standings Gained Points Change": f"{sgp_val:+.2f}",
                                    "Your Rank": status,
                                    "Gap to Next": f"{gap:.2f}" if not is_punt else "-",
                                }
                            )
                        render_compact_table(pd.DataFrame(impact_rows))
                    else:
                        impact_df = pd.DataFrame(
                            [
                                {"Category": cat, "Standings Gained Points Change": f"{val:+.2f}"}
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
                                        "Standings Gained Points Penalty": f"{d['sgp_penalty']:+.2f}",
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
                                hs = health_dict.get(pid, 0.85)
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
                                hs = health_dict.get(pid, 0.85)
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

                    # P10/P90 risk assessment for traded players
                    try:
                        from src.database import get_connection

                        conn = get_connection()
                        try:
                            systems = {}
                            for sys_name in ["steamer", "zips", "depthcharts"]:
                                df = pd.read_sql_query(
                                    "SELECT * FROM projections WHERE system = ?",
                                    conn,
                                    params=(sys_name,),
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
                                                "10th Percentile Batting Average (Floor)": f"{p10_avg:.2f}",
                                                "90th Percentile Batting Average (Ceiling)": f"{p90_avg:.2f}",
                                            }
                                        )
                                if risk_rows:
                                    render_styled_table(pd.DataFrame(risk_rows))
                                    st.caption(METRIC_TOOLTIPS["p10_p90"])
                    except Exception:
                        pass  # Graceful degradation

page_timer_footer("Trade Analyzer")
