"""Trade Analyzer — Evaluate trade proposals with Phase 1 engine pipeline.

Uses the new engine modules for marginal SGP elasticity, punt detection,
LP lineup optimization, and deterministic grading (A+ to F).
Falls back to the legacy analyzer if the engine is unavailable.
"""

import time

import pandas as pd
import streamlit as st

from src.database import (
    coerce_numeric_df,
    init_db,
    load_league_rosters,
    load_league_standings,
    load_player_pool,
)
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, T, inject_custom_css, render_styled_table
from src.valuation import (
    LeagueConfig,
    add_process_risk,
    compute_percentile_projections,
    compute_projection_volatility,
)

# Optional imports — graceful fallback when modules unavailable
_HAS_TRADE_FINDER = False
try:
    from src.trade_finder import find_complementary_teams, find_trade_opportunities

    _HAS_TRADE_FINDER = True
except ImportError:
    pass

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

# Load standings for deep analysis + trade finder
standings = load_league_standings()
all_team_totals: dict[str, dict[str, float]] = {}
if not standings.empty and "category" in standings.columns:
    for _, srow in standings.iterrows():
        team = srow["team_name"]
        cat = str(srow["category"]).strip()
        all_team_totals.setdefault(team, {})[cat] = float(srow.get("total", 0))

# Load health scores from injury history
health_dict: dict = {}
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
            "No league data loaded. Connect your Yahoo league in Connect League, "
            "or league data will load automatically on next app launch."
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
        user_roster_ids: list = []
        if not user_roster.empty and "name" in user_roster.columns:
            for _, r in user_roster.iterrows():
                pname = r.get("name", "")
                pool_pid = name_to_id.get(pname)
                if pool_pid is not None:
                    user_roster_ids.append(pool_pid)
                else:
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

        # ── Tabs: Build Trade | Find Trades ─────────────────────────────
        tab_build, tab_find = st.tabs(["Build Trade", "Find Trades"])

        # ════════════════════════════════════════════════════════════════
        # TAB 1: Build Trade
        # ════════════════════════════════════════════════════════════════
        with tab_build:
            _prefill_give = st.session_state.pop("trade_finder_giving", [])
            _prefill_recv = st.session_state.pop("trade_finder_receiving", [])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("You Give")
                if not user_roster.empty and "name" in user_roster.columns:
                    give_options = sorted(user_roster["name"].dropna().unique().tolist())
                else:
                    give_options = sorted(pool[pool["player_id"].isin(user_roster_ids)]["player_name"].tolist())
                giving_names = st.multiselect(
                    "Select players to trade away",
                    options=give_options,
                    default=[n for n in _prefill_give if n in give_options],
                    key="giving",
                )
            with col2:
                st.subheader("You Receive")
                other_players = pool[~pool["player_id"].isin(user_roster_ids)]
                recv_options = sorted(other_players["player_name"].tolist())
                receiving_names = st.multiselect(
                    "Select players to receive",
                    options=recv_options,
                    default=[n for n in _prefill_recv if n in recv_options],
                    key="receiving",
                )

            # ── Deep Analysis Toggle ─────────────────────────────────
            deep_analysis = st.checkbox(
                "Enable Deep Analysis (Monte Carlo + Game Theory)",
                key="deep_analysis",
                help=(
                    "Activates Phases 2-6: Monte Carlo simulation (10,000 iterations), "
                    "concentration risk, market intelligence, adverse selection, and "
                    "sensitivity analysis. Takes longer but provides deeper insight."
                ),
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

                        trade_progress.progress(10, text="Running marginal elasticity analysis...")
                        trade_progress.progress(20, text="Computing category gaps and punt detection...")
                        trade_progress.progress(30, text="Optimizing lineup assignments...")

                        if deep_analysis:
                            trade_progress.progress(40, text="Running Monte Carlo simulation (10,000 iterations)...")
                            trade_progress.progress(50, text="Analyzing concentration risk...")
                            trade_progress.progress(60, text="Computing market intelligence...")
                            trade_progress.progress(70, text="Running game theory analysis...")

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

                    grade_html = ""
                    if "grade" in result:
                        grade = result["grade"]
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

                    if engine_used == "phase1" and not result.get("category_analysis"):
                        st.warning(
                            "No league standings data available. All categories are weighted equally, "
                            "so the analysis cannot account for your team's strategic position "
                            "(punt detection, marginal elasticity). Sync your Yahoo league for "
                            "a more accurate evaluation."
                        )

                    # Metrics row
                    if engine_used == "phase1":
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Trade Grade", result.get("grade", "N/A"))
                        col2.metric(
                            "Surplus Standings Gained Points",
                            f"{result.get('surplus_sgp', 0):+.3f}",
                            help=METRIC_TOOLTIPS["sgp"],
                        )
                        roster_move = "None"
                        if result.get("drop_candidate"):
                            roster_move = f"Drop {result['drop_candidate']}"
                        elif result.get("fa_pickup"):
                            roster_move = f"Add {result['fa_pickup']}"
                        col3.metric("Roster Move", roster_move, help=METRIC_TOOLTIPS.get("roster_move", ""))
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

                    # Category impact table
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

                    # ── Deep Analysis Results (Phases 2-6) ────────────────────
                    if deep_analysis and engine_used == "phase1":
                        # Monte Carlo Results (Phase 2)
                        mc_mean = result.get("mc_mean")
                        if mc_mean is not None and mc_mean != 0:
                            st.subheader("Monte Carlo Simulation Results")
                            mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
                            mc_c1.metric(
                                "Monte Carlo Mean",
                                f"{mc_mean:+.3f}",
                                help="Average trade surplus across 10,000 simulated seasons.",
                            )
                            mc_c2.metric(
                                "Monte Carlo Standard Deviation",
                                f"{result.get('mc_std', 0):.3f}",
                                help="Spread of outcomes across simulations.",
                            )
                            mc_c3.metric(
                                "Probability Positive",
                                f"{result.get('prob_positive', 0):.1%}",
                                help="Percentage of simulations where the trade helps your team.",
                            )
                            mc_c4.metric(
                                "Sharpe Ratio",
                                f"{result.get('sharpe', 0):.2f}",
                                help="Risk-adjusted return (mean / standard deviation). Higher is better.",
                            )

                            # Risk Assessment
                            st.subheader("Risk Assessment")
                            risk_c1, risk_c2 = st.columns(2)
                            risk_c1.metric(
                                "Value at Risk (5th Percentile)",
                                f"{result.get('var5', 0):+.3f}",
                                help=(
                                    "Worst-case scenario at the 5th percentile. If negative, you could "
                                    "lose this much Standings Gained Points in a bad season."
                                ),
                            )
                            risk_c2.metric(
                                "Conditional Value at Risk (5th Percentile)",
                                f"{result.get('cvar5', 0):+.3f}",
                                help=(
                                    "Average loss in the worst 5% of simulations. A stricter "
                                    "downside measure than Value at Risk."
                                ),
                            )

                        # Concentration Risk (Phase 4)
                        if result.get("concentration_hhi_before") is not None:
                            st.subheader("Roster Concentration Risk")
                            conc_c1, conc_c2, conc_c3 = st.columns(3)
                            conc_c1.metric(
                                "Concentration Before (Herfindahl-Hirschman Index)",
                                f"{result.get('concentration_hhi_before', 0):.3f}",
                                help=(
                                    "Team exposure concentration index before the trade. "
                                    "Lower values indicate better diversification."
                                ),
                            )
                            conc_c2.metric(
                                "Concentration After (Herfindahl-Hirschman Index)",
                                f"{result.get('concentration_hhi_after', 0):.3f}",
                            )
                            penalty = result.get("concentration_penalty", 0)
                            conc_c3.metric(
                                "Concentration Penalty",
                                f"{penalty:+.3f} Standings Gained Points",
                                help=(
                                    "Standings Gained Points penalty applied for excessive "
                                    "concentration on a single MLB team (above 0.15 threshold)."
                                ),
                            )

                        # Market Intelligence (Phase 5)
                        market_vals = result.get("market_values")
                        if market_vals and isinstance(market_vals, dict):
                            st.subheader("Market Intelligence")
                            mkt_c1, mkt_c2 = st.columns(2)
                            mkt_c1.metric(
                                "Market Clearing Price",
                                f"{market_vals.get('market_price', 0):.2f} Standings Gained Points",
                                help=(
                                    "Nash equilibrium fair price based on the second-highest bidder "
                                    "(Vickrey auction). Paying more than this overpays."
                                ),
                            )
                            mkt_c2.metric(
                                "Demand (Teams Interested)",
                                str(market_vals.get("demand", 0)),
                                help=(
                                    "Number of league teams that value this player above "
                                    "0.5 Standings Gained Points. Higher demand means "
                                    "less negotiating power."
                                ),
                            )

                        # Adverse Selection (Phase 5)
                        adverse = result.get("adverse_selection")
                        if adverse and isinstance(adverse, dict):
                            st.subheader("Adverse Selection Analysis")
                            adv_c1, adv_c2 = st.columns(2)
                            adv_c1.metric(
                                "Discount Factor",
                                f"{adverse.get('discount_factor', 1.0):.2f}",
                                help=(
                                    "Bayesian probability adjustment for hidden flaws in offered "
                                    "players. 1.0 = no discount, 0.75 = maximum 25% haircut."
                                ),
                            )
                            risk_level = adverse.get("risk_level", "Unknown")
                            if risk_level == "low":
                                risk_color = T["ok"]
                            elif risk_level == "moderate":
                                risk_color = T["hot"]
                            else:
                                risk_color = T["danger"]
                            adv_c2.markdown(
                                f'<div class="glass" style="padding:12px;text-align:center;">'
                                f'<span style="color:{T["tx2"]};font-size:14px;">Risk Level</span><br>'
                                f'<span style="color:{risk_color};font-size:24px;font-weight:bold;">'
                                f"{risk_level.upper()}</span></div>",
                                unsafe_allow_html=True,
                            )

                        # Sensitivity Analysis (Phase 5)
                        sensitivity = result.get("sensitivity_report")
                        if sensitivity and isinstance(sensitivity, dict):
                            st.subheader("Sensitivity Analysis")
                            sens_rows = []
                            biggest_driver = sensitivity.get("biggest_driver")
                            biggest_drag = sensitivity.get("biggest_drag")
                            vulnerability = sensitivity.get("vulnerability", "Unknown")

                            if biggest_driver:
                                sens_rows.append(
                                    {
                                        "Metric": "Biggest Driver",
                                        "Category": biggest_driver.get("category", "N/A"),
                                        "Impact": f"{biggest_driver.get('impact', 0):+.3f} Standings Gained Points",
                                    }
                                )
                            if biggest_drag:
                                sens_rows.append(
                                    {
                                        "Metric": "Biggest Drag",
                                        "Category": biggest_drag.get("category", "N/A"),
                                        "Impact": f"{biggest_drag.get('impact', 0):+.3f} Standings Gained Points",
                                    }
                                )
                            if sens_rows:
                                render_styled_table(pd.DataFrame(sens_rows))

                            if vulnerability == "robust":
                                vuln_color = T["ok"]
                            elif vulnerability == "moderate":
                                vuln_color = T["sky"]
                            elif vulnerability == "fragile":
                                vuln_color = T["hot"]
                            else:
                                vuln_color = T["danger"]
                            st.markdown(
                                f'<div class="glass" style="padding:12px;margin-top:8px;">'
                                f'<span style="color:{T["tx2"]};">Trade Vulnerability: </span>'
                                f'<span style="color:{vuln_color};font-weight:bold;">'
                                f"{vulnerability.upper()}</span></div>",
                                unsafe_allow_html=True,
                            )

                            # Counter-offer suggestions
                            counter_offers = sensitivity.get("counter_offers", [])
                            if counter_offers:
                                st.subheader("Counter-Offer Suggestions")
                                counter_rows = []
                                for co in counter_offers:
                                    counter_rows.append(
                                        {
                                            "Swap Out": co.get("swap_out", "N/A"),
                                            "Swap In": co.get("swap_in", "N/A"),
                                            "Improvement": f"{co.get('improvement', 0):+.3f} Standings Gained Points",
                                            "New Grade": co.get("new_grade", "N/A"),
                                        }
                                    )
                                render_styled_table(pd.DataFrame(counter_rows))

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

        # ════════════════════════════════════════════════════════════════
        # TAB 2: Find Trades
        # ════════════════════════════════════════════════════════════════
        with tab_find:
            if not _HAS_TRADE_FINDER:
                st.warning(
                    "Trade finder module is not available. Check that src/trade_finder.py is installed correctly."
                )
            else:
                # Build league_rosters dict: {team_name: [player_ids]}
                league_rosters_dict: dict[str, list[int]] = {}
                if not rosters.empty and "team_name" in rosters.columns and "player_id" in rosters.columns:
                    for _, rrow in rosters.iterrows():
                        tname = rrow["team_name"]
                        pid_val = rrow["player_id"]
                        pname = rrow.get("name", "")
                        resolved_pid = name_to_id.get(pname, pid_val) if pname else pid_val
                        league_rosters_dict.setdefault(tname, []).append(resolved_pid)

                if not league_rosters_dict:
                    st.warning(
                        "League roster data is required for the trade finder. "
                        "Connect your Yahoo league and sync roster data to enable this feature."
                    )
                elif not all_team_totals:
                    st.warning(
                        "League standings data is required for the trade finder. "
                        "Sync your Yahoo league to load standings data."
                    )
                else:
                    # Show complementary teams preview
                    try:
                        comp_teams = find_complementary_teams(user_team_name, all_team_totals, config, top_n=5)
                        if comp_teams:
                            partners_text = ", ".join(f"{t[0]} ({t[1]:.2f})" for t in comp_teams)
                            st.markdown(
                                f'<div class="glass" style="padding:12px;margin-bottom:16px;">'
                                f'<span style="color:{T["tx2"]};font-weight:bold;">Best Trade Partners '
                                f"(by category complementarity):</span> "
                                f"{partners_text}</div>",
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        pass

                    if st.button("Scan for Trades", type="primary", width="stretch", key="scan_trades"):
                        scan_progress = st.progress(0, text="Scanning league for trade opportunities...")
                        try:
                            scan_progress.progress(30, text="Computing team category vectors...")
                            scan_progress.progress(60, text="Evaluating 1-for-1 trade candidates...")

                            opportunities = find_trade_opportunities(
                                user_roster_ids=user_roster_ids,
                                player_pool=pool,
                                config=config,
                                all_team_totals=all_team_totals,
                                user_team_name=user_team_name,
                                league_rosters=league_rosters_dict,
                                weeks_remaining=16,
                            )
                            scan_progress.progress(100, text="Scan complete!")
                            time.sleep(0.3)
                            scan_progress.empty()

                            if not opportunities:
                                st.info("No beneficial trade opportunities found at this time.")
                            else:
                                st.success(f"Found {len(opportunities)} trade opportunities.")

                                # Display results table
                                opp_rows = []
                                for i, opp in enumerate(opportunities):
                                    opp_rows.append(
                                        {
                                            "Rank": i + 1,
                                            "Giving": ", ".join(opp.get("giving_names", [])),
                                            "Receiving": ", ".join(opp.get("receiving_names", [])),
                                            "Opponent Team": opp.get("opponent_team", "Unknown"),
                                            "Your Standings Gained Points Gain": f"{opp.get('user_sgp_gain', 0):+.3f}",
                                            "Acceptance Probability": f"{opp.get('acceptance_probability', 0):.0%}",
                                            "Composite Score": f"{opp.get('composite_score', 0):.2f}",
                                        }
                                    )
                                render_styled_table(pd.DataFrame(opp_rows))

                                # Analyze buttons for each opportunity
                                st.subheader("Analyze a Trade")
                                for i, opp in enumerate(opportunities[:10]):
                                    give_names_list = opp.get("giving_names", [])
                                    recv_names_list = opp.get("receiving_names", [])
                                    label = (
                                        f"Analyze: Give {', '.join(give_names_list)} for {', '.join(recv_names_list)}"
                                    )
                                    if st.button(label, key=f"analyze_opp_{i}"):
                                        st.session_state["trade_finder_giving"] = give_names_list
                                        st.session_state["trade_finder_receiving"] = recv_names_list
                                        st.rerun()

                        except Exception as e:
                            scan_progress.empty()
                            st.error(f"Trade scan failed: {e}")
