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
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

try:
    from src.trade_finder import find_trade_opportunities

    _HAS_TRADE_FINDER = True
except ImportError:
    _HAS_TRADE_FINDER = False

try:
    from src.trade_value import compute_trade_values

    _HAS_TRADE_VALUE = True
except ImportError:
    _HAS_TRADE_VALUE = False

try:
    from src.trade_finder import estimate_acceptance_probability

    _HAS_ACCEPTANCE = True
except ImportError:
    _HAS_ACCEPTANCE = False

st.set_page_config(page_title="Heater | Trade Analyzer", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>TRADE ANALYZER</span></div></div>',
    unsafe_allow_html=True,
)

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
            other_players = pool[~pool["player_id"].isin(user_roster_ids)]
            receiving_names = st.multiselect(
                "Select players to receive",
                options=sorted(other_players["player_name"].tolist()),
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

                # Deep analysis toggle (Phases 2-6)
                deep_analysis = st.checkbox(
                    "Enable Deep Analysis (Monte Carlo + Game Theory)",
                    key="deep_analysis",
                    help="Activates Phases 2-6: stochastic MC simulation, signal intelligence, "
                    "context engine, game theory (opponent modeling, adverse selection). Takes 5-15s.",
                )

                trade_progress = st.progress(0, text="Computing category impacts...")

                # Try the new Phase 1 engine first, fall back to legacy
                try:
                    from src.engine.output.trade_evaluator import evaluate_trade

                    trade_progress.progress(20, text="Running marginal elasticity analysis...")
                    trade_progress.progress(40, text="Computing category gaps and punt detection...")
                    trade_progress.progress(60, text="Optimizing lineup assignments...")

                    result = evaluate_trade(
                        giving_ids=giving_ids,
                        receiving_ids=receiving_ids,
                        user_roster_ids=user_roster_ids,
                        player_pool=pool,
                        config=config,
                        user_team_name=user_team_name,
                        weeks_remaining=16,
                        enable_mc=deep_analysis,
                        enable_context=True,
                        enable_game_theory=deep_analysis,
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
                    f"{result['confidence_pct']:.1f}% confidence</span></div>",
                    unsafe_allow_html=True,
                )
                verdict_key = "trade_verdict" if engine_used == "phase1" else "trade_verdict_legacy"
                st.caption(METRIC_TOOLTIPS[verdict_key])

                # Warning when no standings data is loaded
                if engine_used == "phase1" and not result.get("category_analysis"):
                    st.warning(
                        "No league standings data available. All categories are weighted equally, "
                        "so the analysis cannot account for your team's strategic position "
                        "(punt detection, marginal elasticity). Sync your Yahoo league for "
                        "a more accurate evaluation."
                    )

                # Metrics row — different layout for Phase 1 vs legacy
                if engine_used == "phase1":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric(
                        "Trade Grade",
                        result.get("grade", "N/A"),
                    )
                    col2.metric(
                        "Surplus Standings Gained Points",
                        f"{result.get('surplus_sgp', 0):+.3f}",
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
                        f"{result.get('replacement_penalty', 0):+.3f}",
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
                        f"{result['total_sgp_change']:+.3f}",
                        help=METRIC_TOOLTIPS["sgp"],
                    )
                    col2.metric(
                        "Monte Carlo Mean",
                        f"{result['mc_mean']:+.3f}",
                        help=METRIC_TOOLTIPS["mc_mean"],
                    )
                    col3.metric(
                        "Monte Carlo Standard Deviation",
                        f"{result['mc_std']:.3f}",
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
                                "Standings Gained Points Change": f"{sgp_val:+.3f}",
                                "Your Rank": status,
                                "Gap to Next": f"{gap:.1f}" if not is_punt else "-",
                            }
                        )
                    render_styled_table(pd.DataFrame(impact_rows))
                else:
                    impact_df = pd.DataFrame(
                        [
                            {"Category": cat, "Standings Gained Points Change": f"{val:+.3f}"}
                            for cat, val in result["category_impact"].items()
                        ]
                    )
                    render_styled_table(impact_df)

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
                                    "Raw Loss": f"{d['raw_loss']:.1f}",
                                    "Best Free Agent": f"{d.get('best_fa_name', 'N/A')} ({d['best_fa']:.1f})",
                                    "Unrecoverable Gap": f"{d['unrecoverable']:.1f}",
                                    "Standings Gained Points Penalty": f"{d['sgp_penalty']:+.3f}",
                                }
                            )
                        if repl_rows:
                            render_styled_table(pd.DataFrame(repl_rows))

                # Risk flags
                if result["risk_flags"]:
                    st.subheader("Risk Flags")
                    for flag in result["risk_flags"]:
                        st.warning(flag)

                # Show injury badges for traded players
                trade_col1, trade_col2 = st.columns(2)
                with trade_col1:
                    st.markdown("**Giving:**")
                    for name in giving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            pid = p.iloc[0]["player_id"]
                            hs = health_dict.get(pid, 0.85)
                            badge_icon, label = get_injury_badge(hs)
                            st.markdown(f"{badge_icon} {name} — {label}", unsafe_allow_html=True)
                with trade_col2:
                    st.markdown("**Receiving:**")
                    for name in receiving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            pid = p.iloc[0]["player_id"]
                            hs = health_dict.get(pid, 0.85)
                            badge_icon, label = get_injury_badge(hs)
                            st.markdown(f"{badge_icon} {name} — {label}", unsafe_allow_html=True)

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
                                            "10th Percentile Home Runs (Floor)": f"{p10_hr:.0f}",
                                            "90th Percentile Home Runs (Ceiling)": f"{p90_hr:.0f}",
                                            "10th Percentile Batting Average (Floor)": f"{p10_avg:.3f}",
                                            "90th Percentile Batting Average (Ceiling)": f"{p90_avg:.3f}",
                                        }
                                    )
                            if risk_rows:
                                render_styled_table(pd.DataFrame(risk_rows))
                                st.caption(METRIC_TOOLTIPS["p10_p90"])
                except Exception:
                    pass  # Graceful degradation

                # ── Deep Analysis Results (Phases 2-6) ──────────────────────
                if deep_analysis and engine_used == "phase1":
                    with st.expander("Deep Analysis Results", expanded=True):
                        # MC metrics
                        mc_mean = result.get("mc_mean")
                        if mc_mean is not None:
                            st.markdown("#### Monte Carlo Simulation")
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("MC Mean SGP", f"{mc_mean:+.3f}")
                            mc2.metric("P(Positive)", f"{result.get('prob_positive', 0):.0%}")
                            mc3.metric("Sharpe Ratio", f"{result.get('sharpe', 0):.2f}")
                            mc4.metric("MC Std Dev", f"{result.get('mc_std', 0):.3f}")

                            # VaR/CVaR
                            var5 = result.get("var5") or result.get("mc_p5")
                            cvar5 = result.get("cvar5")
                            if var5 is not None:
                                v1, v2 = st.columns(2)
                                v1.metric("Value at Risk (5%)", f"{var5:+.3f}", help="Worst-case SGP at 5th percentile")
                                if cvar5 is not None:
                                    v2.metric(
                                        "Conditional VaR (5%)",
                                        f"{cvar5:+.3f}",
                                        help="Expected loss in worst 5% of scenarios",
                                    )

                        # Concentration risk
                        conc_before = result.get("concentration_hhi_before")
                        if conc_before is not None:
                            st.markdown("#### Concentration Risk")
                            h1, h2, h3 = st.columns(3)
                            h1.metric("HHI Before", f"{conc_before:.3f}")
                            h2.metric("HHI After", f"{result.get('concentration_hhi_after', 0):.3f}")
                            h3.metric("Penalty", f"{result.get('concentration_penalty', 0):+.3f} SGP")

                        # Adverse selection
                        adv = result.get("adverse_selection", {})
                        if adv and adv.get("discount_factor") is not None:
                            st.markdown("#### Adverse Selection Analysis")
                            a1, a2 = st.columns(2)
                            a1.metric("Discount Factor", f"{adv['discount_factor']:.2f}")
                            a2.metric("Risk Level", adv.get("risk_level", "unknown"))

                        # Sensitivity
                        sens = result.get("sensitivity_report", {})
                        if sens:
                            st.markdown("#### Trade Sensitivity")
                            s1, s2, s3 = st.columns(3)
                            s1.metric("Biggest Driver", sens.get("biggest_driver", "—"))
                            s2.metric("Biggest Drag", sens.get("biggest_drag", "—"))
                            s3.metric("Vulnerability", sens.get("vulnerability", "—"))

        # ── Trade Finder Section ────────────────────────────────────────
        if _HAS_TRADE_FINDER:
            with st.expander("Trade Finder — Scan for Opportunities", expanded=False):
                standings = load_league_standings()
                all_team_totals: dict[str, dict[str, float]] = {}
                if not standings.empty and "category" in standings.columns:
                    for _, srow in standings.iterrows():
                        t_name = str(srow.get("team_name", ""))
                        cat_name = str(srow.get("category", "")).strip()
                        all_team_totals.setdefault(t_name, {})[cat_name] = float(srow.get("total", 0))

                # Build league_rosters dict
                league_rosters_dict: dict[str, list[int]] = {}
                all_rosters = load_league_rosters()
                if not all_rosters.empty:
                    for _, rr in all_rosters.iterrows():
                        t_name = str(rr.get("team_name", ""))
                        rpid = rr.get("player_id")
                        if t_name and rpid:
                            league_rosters_dict.setdefault(t_name, []).append(int(rpid))

                if not league_rosters_dict or not all_team_totals:
                    st.warning("Trade Finder requires league rosters and standings. Connect your Yahoo league.")
                elif st.button("Scan for Trade Opportunities", type="primary", key="find_trades"):
                    find_progress = st.progress(0, text="Scanning rosters for complementary trades...")
                    try:
                        find_progress.progress(30, text="Computing team complementarity...")
                        opportunities = find_trade_opportunities(
                            user_roster_ids=user_roster_ids,
                            player_pool=pool,
                            config=config,
                            all_team_totals=all_team_totals,
                            user_team_name=user_team_name,
                            league_rosters=league_rosters_dict,
                        )
                        find_progress.progress(100, text="Done!")
                        time.sleep(0.3)
                        find_progress.empty()
                        st.session_state["trade_finder_results"] = opportunities
                    except Exception as e:
                        find_progress.empty()
                        st.error(f"Trade finder failed: {e}")

                finder_results = st.session_state.get("trade_finder_results", [])
                if finder_results:
                    st.markdown(f"**{len(finder_results)} trade opportunities found**")
                    for i, opp in enumerate(finder_results[:15]):
                        giving = ", ".join(opp.get("giving_names", []))
                        receiving = ", ".join(opp.get("receiving_names", []))
                        gain = opp.get("user_sgp_gain", 0)
                        p_accept = opp.get("acceptance_probability", 0)
                        opp_team = opp.get("opponent_team", "?")

                        gain_color = T["green"] if gain > 0 else T["danger"]
                        st.markdown(
                            f'<div class="glass" style="padding:12px;margin:8px 0;'
                            f'border-left:4px solid {gain_color};">'
                            f"<strong>Send</strong> {giving} <strong>to {opp_team} for</strong> {receiving}"
                            f'<br><span style="color:{gain_color};">SGP Gain: {gain:+.2f}</span>'
                            f" | Acceptance: {p_accept:.0%}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # ── "What Can I Get For X?" Tool ────────────────────────────────
        if _HAS_TRADE_VALUE:
            with st.expander("What Can I Get For X?", expanded=False):
                st.caption("Select a player to trade away and see value-matched return candidates across the league.")

                if not user_roster.empty and "name" in user_roster.columns:
                    trade_away = st.selectbox(
                        "Player to trade away:",
                        options=sorted(user_roster["name"].dropna().tolist()),
                        key="wcigfx_player",
                    )

                    if st.button("Find Return Candidates", key="wcigfx_go"):
                        try:
                            tv_df = compute_trade_values(pool, config)
                            # Get selected player's trade value
                            player_tv = tv_df[tv_df["name"] == trade_away]
                            if player_tv.empty:
                                st.warning(f"Could not find trade value for {trade_away}.")
                            else:
                                my_value = float(player_tv.iloc[0].get("trade_value", 50))
                                my_tier = str(player_tv.iloc[0].get("tier", ""))
                                st.markdown(
                                    f"**{trade_away}** — Trade Value: **{my_value:.0f}** ({my_tier}). "
                                    f"Searching for returns worth {max(0, my_value - 10):.0f}-{min(100, my_value + 10):.0f}."
                                )

                                # Find value-matched players NOT on user's roster
                                margin = 10
                                matches = tv_df[
                                    (tv_df["trade_value"] >= my_value - margin)
                                    & (tv_df["trade_value"] <= my_value + margin)
                                    & (~tv_df["player_id"].isin(user_roster_ids))
                                    & (tv_df["name"] != trade_away)
                                ].sort_values("trade_value", ascending=False)

                                if matches.empty:
                                    st.info("No value-matched return candidates found.")
                                else:
                                    match_rows = []
                                    for _, m in matches.head(20).iterrows():
                                        p_accept = 0.5
                                        if _HAS_ACCEPTANCE:
                                            try:
                                                user_gain = float(m.get("total_sgp", 0)) - float(
                                                    player_tv.iloc[0].get("total_sgp", 0)
                                                )
                                                opp_gain = -user_gain
                                                p_accept = estimate_acceptance_probability(user_gain, opp_gain)
                                            except Exception:
                                                p_accept = 0.5

                                        match_rows.append(
                                            {
                                                "Player": str(m.get("name", "?")),
                                                "Team": str(m.get("team", "?")),
                                                "Pos": str(m.get("positions", "?")),
                                                "Trade Value": f"{m.get('trade_value', 0):.0f}",
                                                "Tier": str(m.get("tier", "")),
                                                "Accept %": f"{p_accept:.0%}",
                                            }
                                        )
                                    render_styled_table(pd.DataFrame(match_rows))
                        except Exception as e:
                            st.error(f"Value lookup failed: {e}")
                else:
                    st.info("Load roster data to use this tool.")

        # ── Multi-Team Trade Builder ────────────────────────────────────
        with st.expander("3-Team Trade Builder", expanded=False):
            st.caption(
                "Evaluate a 3-team trade by analyzing each bilateral leg independently. "
                "All legs must be positive for the trade to be recommended."
            )

            all_rosters_df = load_league_rosters()
            if all_rosters_df.empty:
                st.warning("League rosters required for multi-team trades. Connect your Yahoo league.")
            else:
                team_names_all = sorted(all_rosters_df["team_name"].dropna().unique().tolist())

                mt_col1, mt_col2, mt_col3 = st.columns(3)
                with mt_col1:
                    st.markdown(f"**Team A: {user_team_name}**")
                    mt_a_give = st.multiselect(
                        "Team A gives:",
                        options=sorted(user_roster["name"].dropna().tolist())
                        if not user_roster.empty and "name" in user_roster.columns
                        else [],
                        key="mt_a_give",
                    )

                with mt_col2:
                    mt_team_b = st.selectbox(
                        "Team B:",
                        options=[t for t in team_names_all if t != user_team_name],
                        key="mt_team_b",
                    )
                    team_b_roster = get_team_roster(mt_team_b)
                    mt_b_give = st.multiselect(
                        "Team B gives:",
                        options=sorted(team_b_roster["name"].dropna().tolist())
                        if not team_b_roster.empty and "name" in team_b_roster.columns
                        else [],
                        key="mt_b_give",
                    )

                with mt_col3:
                    remaining_teams = [t for t in team_names_all if t != user_team_name and t != mt_team_b]
                    mt_team_c = st.selectbox(
                        "Team C:", options=remaining_teams if remaining_teams else ["—"], key="mt_team_c"
                    )
                    if mt_team_c != "—":
                        team_c_roster = get_team_roster(mt_team_c)
                        mt_c_give = st.multiselect(
                            "Team C gives:",
                            options=sorted(team_c_roster["name"].dropna().tolist())
                            if not team_c_roster.empty and "name" in team_c_roster.columns
                            else [],
                            key="mt_c_give",
                        )
                    else:
                        mt_c_give = []

                st.markdown("**Trade Flow:** A gives → B, B gives → C, C gives → A")

                if st.button("Evaluate 3-Team Trade", key="mt_evaluate"):
                    if not mt_a_give or not mt_b_give or not mt_c_give:
                        st.error("Each team must give at least one player.")
                    else:
                        try:
                            from src.engine.output.trade_evaluator import evaluate_trade as mt_evaluate

                            # Resolve IDs
                            mt_a_ids = [name_to_id[n] for n in mt_a_give if n in name_to_id]
                            mt_b_ids = [name_to_id.get(n) or 0 for n in mt_b_give]
                            mt_c_ids = [name_to_id.get(n) or 0 for n in mt_c_give]

                            # Build roster ID lists for teams B and C
                            b_roster_ids = team_b_roster["player_id"].tolist() if not team_b_roster.empty else []
                            c_roster_ids = (
                                team_c_roster["player_id"].tolist()
                                if not team_c_roster.empty and mt_team_c != "—"
                                else []
                            )

                            # Leg 1: A gives to B, A receives from C
                            leg1 = mt_evaluate(
                                giving_ids=mt_a_ids,
                                receiving_ids=mt_c_ids,
                                user_roster_ids=user_roster_ids,
                                player_pool=pool,
                                config=config,
                                enable_mc=False,
                                enable_context=False,
                                enable_game_theory=False,
                            )
                            # Leg 2: B gives to C, B receives from A
                            leg2 = mt_evaluate(
                                giving_ids=mt_b_ids,
                                receiving_ids=mt_a_ids,
                                user_roster_ids=b_roster_ids,
                                player_pool=pool,
                                config=config,
                                enable_mc=False,
                                enable_context=False,
                                enable_game_theory=False,
                            )
                            # Leg 3: C gives to A, C receives from B
                            leg3 = mt_evaluate(
                                giving_ids=mt_c_ids,
                                receiving_ids=mt_b_ids,
                                user_roster_ids=c_roster_ids,
                                player_pool=pool,
                                config=config,
                                enable_mc=False,
                                enable_context=False,
                                enable_game_theory=False,
                            )

                            # Display results
                            r1, r2, r3 = st.columns(3)
                            for col, leg, team_label in [
                                (r1, leg1, user_team_name),
                                (r2, leg2, mt_team_b),
                                (r3, leg3, mt_team_c),
                            ]:
                                grade = leg.get("grade", "?")
                                surplus = leg.get("surplus_sgp", 0)
                                if grade.startswith("A") or grade.startswith("B"):
                                    gc = T["green"]
                                elif grade.startswith("C"):
                                    gc = T["hot"]
                                else:
                                    gc = T["danger"]
                                col.markdown(
                                    f'<div style="text-align:center;padding:12px;border:2px solid {gc};border-radius:8px;">'
                                    f'<div style="font-size:12px;color:{T["tx2"]};">{team_label}</div>'
                                    f'<div style="font-size:32px;font-weight:900;color:{gc};">{grade}</div>'
                                    f'<div style="font-size:14px;">SGP: {surplus:+.2f}</div></div>',
                                    unsafe_allow_html=True,
                                )

                            # Combined verdict
                            all_positive = all(leg.get("surplus_sgp", 0) > -0.5 for leg in [leg1, leg2, leg3])
                            if all_positive:
                                st.success("All teams benefit from this trade. Recommended to proceed.")
                            else:
                                losers = []
                                for leg, name in [(leg1, user_team_name), (leg2, mt_team_b), (leg3, mt_team_c)]:
                                    if leg.get("surplus_sgp", 0) <= -0.5:
                                        losers.append(name)
                                st.warning(f"Trade disadvantages: {', '.join(losers)}. Adjust terms.")

                        except Exception as e:
                            st.error(f"Multi-team evaluation failed: {e}")
