"""Trade Finder — Proactive trade recommendations based on team needs.

Scans all 12 league rosters for mutually beneficial 1-for-1 trades using
cosine dissimilarity team pairing and behavioral acceptance modeling.
Three tab views: By Partner, By Need, By Value.
"""

import time

import pandas as pd
import streamlit as st

from src.database import (
    coerce_numeric_df,
    init_db,
    load_player_pool,
)
from src.in_season import _roster_category_totals
from src.trade_finder import (
    find_complementary_teams,
    find_trade_opportunities,
)
from src.ui_shared import (
    T,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_sortable_table,
)
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

# Full display names for stat categories (no abbreviations per CLAUDE.md)
_CAT_DISPLAY = {
    "R": "Runs",
    "HR": "Home Runs",
    "RBI": "Runs Batted In",
    "SB": "Stolen Bases",
    "AVG": "Batting Average",
    "OBP": "On-Base Percentage",
    "W": "Wins",
    "L": "Losses",
    "SV": "Saves",
    "K": "Strikeouts",
    "ERA": "Earned Run Average",
    "WHIP": "Walks + Hits per Inning Pitched",
}

st.set_page_config(
    page_title="Heater | Trade Finder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Helpers ────────────────────────────────────────────────────────────


def _build_trade_df(trades: list[dict]) -> pd.DataFrame:
    """Convert trade dicts to a display DataFrame."""
    rows = []
    for trade in trades:
        giving = ", ".join(trade.get("giving_names", []))
        receiving = ", ".join(trade.get("receiving_names", []))
        row = {
            "You Give": giving,
            "You Receive": receiving,
            "Type": trade.get("trade_type", "1-for-1"),
            "Partner": trade.get("opponent_team", ""),
            "Your Gain": round(trade.get("user_sgp_gain", 0), 2),
            "Their Gain": round(trade.get("opponent_sgp_gain", 0), 2),
            "Give ADP": trade.get("give_adp_round", "N/A") if trade.get("give_adp_round") else "N/A",
            "Recv ADP": trade.get("recv_adp_round", "N/A") if trade.get("recv_adp_round") else "N/A",
            "ADP Fair": f"{trade.get('adp_fairness', 0):.0%}"
            if isinstance(trade.get("adp_fairness"), (int, float))
            else "N/A",
            "ECR Fair": f"{trade.get('ecr_fairness', 0):.0%}"
            if isinstance(trade.get("ecr_fairness"), (int, float))
            else "",
            "Grade": trade.get("grade", ""),
            "Acceptance": trade.get("acceptance_label", ""),
            "Health": trade.get("health_risk", "") or "",
            "Score": round(trade.get("composite_score", 0), 2),
            "FA Alt": trade.get("fa_name", "") if trade.get("fa_alternative") else "",
        }
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _get_user_weak_categories(
    user_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    config: LeagueConfig,
    bottom_n: int = 4,
) -> list[str]:
    """Identify the user's weakest categories (bottom N ranks)."""
    cats = config.all_categories
    rankings: dict[str, int] = {}

    for cat in cats:
        all_vals = sorted(
            [t.get(cat, 0) for t in all_team_totals.values()],
            reverse=(cat not in config.inverse_stats),
        )
        user_val = user_totals.get(cat, 0)
        # Find rank (1 = best)
        rank = 1
        for val in all_vals:
            if cat in config.inverse_stats:
                if val < user_val:
                    rank += 1
            else:
                if val > user_val:
                    rank += 1
        rankings[cat] = rank

    # Sort by rank descending (worst categories first)
    sorted_cats = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, _rank in sorted_cats[:bottom_n]]


def _categorize_trades_by_need(
    trades: list[dict],
    weak_categories: list[str],
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
) -> dict[str, list[dict]]:
    """Group trades by which weak category they improve most."""
    grouped: dict[str, list[dict]] = {cat: [] for cat in weak_categories}

    user_totals = _roster_category_totals(user_roster_ids, player_pool)

    for trade in trades:
        # Compute post-trade totals
        new_ids = [pid for pid in user_roster_ids if pid not in trade["giving_ids"]] + trade["receiving_ids"]
        new_totals = _roster_category_totals(new_ids, player_pool)

        # Find which weak category improves most
        best_cat = None
        best_improvement = -999

        for cat in weak_categories:
            old_val = user_totals.get(cat, 0)
            new_val = new_totals.get(cat, 0)
            if cat in config.inverse_stats:
                improvement = old_val - new_val  # Lower is better
            else:
                improvement = new_val - old_val
            if improvement > best_improvement:
                best_improvement = improvement
                best_cat = cat

        if best_cat and best_improvement > 0:
            trade_copy = dict(trade)
            trade_copy["category_improvement"] = round(best_improvement, 2)
            grouped[best_cat].append(trade_copy)

    # Sort each group by improvement
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x.get("category_improvement", 0), reverse=True)

    return grouped


# ── Main ──────────────────────────────────────────────────────────────


def main():
    page_timer_start()
    inject_custom_css()
    init_db()

    config = LeagueConfig()

    # ── Load data ─────────────────────────────────────────────────────
    pool = load_player_pool()
    if pool.empty:
        st.error("Player pool is empty. Run the app bootstrap first.")
        return

    pool = coerce_numeric_df(pool)
    yds = get_yahoo_data_service()
    rosters_df = yds.get_rosters()

    if rosters_df.empty:
        st.warning("No league rosters loaded. Connect to Yahoo and sync rosters first.")
        return

    # Build {team_name: [player_ids]} dict
    league_rosters: dict[str, list[int]] = {}
    for _, row in rosters_df.iterrows():
        team = row.get("team_name", "")
        pid = row.get("player_id")
        if team and pid is not None:
            league_rosters.setdefault(team, []).append(int(pid))

    if len(league_rosters) < 2:
        st.warning("Need at least 2 teams loaded to find trades.")
        return

    # Identify user team — use is_user_team flag from league_rosters table
    user_teams = rosters_df[rosters_df["is_user_team"] == 1]
    if user_teams.empty:
        # Fallback: try session state, then first team
        user_team_name = st.session_state.get("user_team_name")
        if not user_team_name or user_team_name not in league_rosters:
            # Last resort: pick the first team
            user_team_name = next(iter(league_rosters), None)
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        if isinstance(user_team_name, bytes):
            user_team_name = user_team_name.decode("utf-8", errors="replace")

    if not user_team_name or user_team_name not in league_rosters:
        st.warning("No user team identified. Connect to Yahoo and sync rosters first.")
        return

    user_roster_ids = league_rosters.get(user_team_name, [])
    if not user_roster_ids:
        st.warning(f"No roster found for '{user_team_name}'. Check Yahoo sync.")
        return

    # Compute team category totals from standings or roster stats
    standings_df = yds.get_standings()
    all_team_totals: dict[str, dict[str, float]] = {}

    if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
        # Standings table is normalized (long format): one row per team-category pair.
        # Pivot to wide format: one row per team, columns = categories.
        standings_df["total"] = pd.to_numeric(standings_df["total"], errors="coerce").fillna(0)
        wide = standings_df.pivot_table(
            index="team_name", columns="category", values="total", aggfunc="first"
        ).reset_index()
        for _, row in wide.iterrows():
            team = row.get("team_name", "")
            if team:
                totals = {}
                for cat in config.all_categories:
                    totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                all_team_totals[team] = totals
    elif not standings_df.empty and "team_name" in standings_df.columns:
        # Wide format fallback (legacy schema)
        for _, row in standings_df.iterrows():
            team = row.get("team_name", "")
            if team:
                totals = {}
                for cat in config.all_categories:
                    totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                all_team_totals[team] = totals
    else:
        # Fallback: compute from roster stats
        for team_name, pids in league_rosters.items():
            all_team_totals[team_name] = _roster_category_totals(pids, pool)

    # ── Run trade finder engine ───────────────────────────────────────
    with st.spinner("Scanning league for trade opportunities..."):
        t0 = time.time()
        opportunities = find_trade_opportunities(
            user_roster_ids=user_roster_ids,
            player_pool=pool,
            config=config,
            all_team_totals=all_team_totals,
            user_team_name=user_team_name,
            league_rosters=league_rosters,
            max_results=50,
            top_partners=11,
        )
        scan_time = time.time() - t0

    # ── Banner ────────────────────────────────────────────────────────
    if opportunities:
        best = opportunities[0]
        best_give = ", ".join(best.get("giving_names", []))
        best_recv = ", ".join(best.get("receiving_names", []))
        best_partner = best.get("opponent_team", "?")
        best_gain = best.get("user_sgp_gain", 0)
        banner_text = (
            f"Top opportunity: Send {best_give} to {best_partner} for {best_recv} "
            f"(+{best_gain:.2f} Standings Gained Points)"
        )
        render_page_layout(
            "TRADE FINDER",
            banner_teaser=banner_text,
            banner_icon="trade_analyzer",
        )
    else:
        render_page_layout("TRADE FINDER", banner_teaser="No profitable trades found at this time.")

    # ── Context + Main columns ────────────────────────────────────────
    ctx, main_col = render_context_columns()

    with ctx:
        # Team needs summary — recompute from roster stats if standings have zero values
        user_totals = all_team_totals.get(user_team_name, {})
        if not user_totals or all(v == 0 for v in user_totals.values()):
            user_totals = _roster_category_totals(user_roster_ids, pool)
            all_team_totals[user_team_name] = user_totals
            # Also recompute for other teams if all zeros
            for team_name_fix, pids_fix in league_rosters.items():
                if team_name_fix != user_team_name:
                    existing = all_team_totals.get(team_name_fix, {})
                    if not existing or all(v == 0 for v in existing.values()):
                        all_team_totals[team_name_fix] = _roster_category_totals(pids_fix, pool)
        weak_cats = _get_user_weak_categories(user_totals, all_team_totals, config, bottom_n=4)

        needs_html = "<ul style='margin:0;padding-left:16px;'>"
        for cat in weak_cats:
            display_name = _CAT_DISPLAY.get(cat, cat.upper())
            tx_color = T["tx"]
            needs_html += f"<li style='color:{tx_color};font-size:12px;'>{display_name}</li>"
        needs_html += "</ul>"
        render_context_card("Category Needs", needs_html)

        # Complementary teams
        if all_team_totals and user_team_name:
            partners = find_complementary_teams(user_team_name, all_team_totals, config, top_n=5)
            partners_html = ""
            for i, (team, score) in enumerate(partners, 1):
                partners_html += (
                    f'<div style="font-size:12px;color:{T["tx"]};padding:2px 0;">'
                    f'{i}. {team} <span style="color:{T["tx2"]};">({score:.2f})</span></div>'
                )
            render_context_card("Best Trade Partners", partners_html)

        # Scan stats
        stats_html = (
            f'<div style="font-size:12px;color:{T["tx"]};">'
            f"Trades found: {len(opportunities)}<br>"
            f"Teams scanned: {len(league_rosters) - 1}<br>"
            f"Scan time: {scan_time:.2f}s</div>"
        )
        render_context_card("Scan Summary", stats_html)

        # Schedule urgency card
        try:
            from src.trade_intelligence import compute_schedule_urgency

            urgency_mult = compute_schedule_urgency(weeks_ahead=3, yds=yds)
            if urgency_mult > 1.05:
                urgency_label = "HIGH -- tough opponents ahead"
                urgency_color = T["danger"]
            elif urgency_mult < 0.95:
                urgency_label = "LOW -- easy schedule ahead"
                urgency_color = T["green"]
            else:
                urgency_label = "NORMAL"
                urgency_color = T["tx2"]
            urgency_html = (
                f'<div style="font-size:12px;color:{urgency_color};font-weight:700;">'
                f"{urgency_label}</div>"
                f'<div style="font-size:11px;color:{T["tx2"]};">Urgency multiplier: {urgency_mult:.2f}x</div>'
            )
            render_context_card("Trade Urgency", urgency_html)
        except Exception:
            pass

    with main_col:
        if not opportunities:
            st.info(
                "No profitable trade opportunities found. This can happen when:\n"
                "- League standings data is missing (sync with Yahoo)\n"
                "- Your roster is well-balanced (no clear upgrade paths)\n"
                "- All opponents have weaker players at your need positions"
            )
            page_timer_footer("Trade Finder")
            return

        # ── 5 Tabs ───────────────────────────────────────────────────
        # Group trades by opponent (used by Browse Partners tab)
        trades_by_partner: dict[str, list[dict]] = {}
        for trade in opportunities:
            partner = trade.get("opponent_team", "Unknown")
            trades_by_partner.setdefault(partner, []).append(trade)

        tab_smart, tab_value, tab_target, tab_browse, tab_readiness = st.tabs(
            [
                "Smart Recommendations",
                "By Value",
                "Target a Player",
                "Browse Partners",
                "Trade Readiness",
            ]
        )

        # ── Tab 1: Smart Recommendations ─────────────────────────────
        with tab_smart:
            st.subheader("Smart Recommendations")
            st.caption(
                "Auto-scan all opponents to find trades that boost your weakest "
                "categories for the least cost in strong ones."
            )

            if st.button("Generate Smart Recommendations", type="primary", key="smart_recs_btn"):
                with st.spinner("Scanning all opponents for category-need-efficient trades..."):
                    try:
                        from src.trade_intelligence import recommend_trades_by_need

                        try:
                            from src.validation.dynamic_context import compute_weeks_remaining

                            weeks_rem_s = compute_weeks_remaining()
                        except ImportError:
                            weeks_rem_s = 22
                    except ImportError:
                        st.info("Trade intelligence module not available for smart recommendations.")
                        recommend_trades_by_need = None  # type: ignore[assignment]
                        weeks_rem_s = 22

                    recs: list[dict] = []
                    if recommend_trades_by_need is not None:
                        try:
                            recs = recommend_trades_by_need(
                                user_roster_ids=user_roster_ids,
                                player_pool=pool,
                                config=config,
                                all_team_totals=all_team_totals if all_team_totals else None,
                                user_team_name=user_team_name,
                                league_rosters=rosters_df,
                                weeks_remaining=weeks_rem_s,
                                max_results=20,
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")

                if not recs:
                    st.info("No recommendations found.")
                else:
                    st.success(f"Found {len(recs)} recommendations")
                    for i, rec in enumerate(recs):
                        give_name_s = rec.get("giving_name", "?")
                        recv_name_s = rec.get("receiving_name", "?")
                        partner_s = rec.get("opponent_team", "?")
                        eff_s = rec.get("need_efficiency", 0)
                        accept_s = rec.get("acceptance_probability", 0)
                        grade_s = rec.get("grade_estimate", "?")
                        composite_s = rec.get("composite_score", 0)

                        with st.expander(
                            f"#{i + 1}: Give {give_name_s} -> Get {recv_name_s} ({partner_s}) "
                            f"| Efficiency {eff_s:.1f}x | Accept {accept_s:.0%}",
                            expanded=(i < 3),
                        ):
                            m1_s, m2_s, m3_s, m4_s = st.columns(4)
                            m1_s.metric("Grade", grade_s)
                            m2_s.metric("Efficiency", f"{eff_s:.1f}x")
                            m3_s.metric("Accept Prob", f"{accept_s:.0%}")
                            m4_s.metric("Composite", f"{composite_s:.2f}")

                            # Boosted and costly categories
                            boosted = rec.get("boosted_cats", [])
                            costly = rec.get("costly_cats", [])
                            if boosted:
                                boost_display = ", ".join([_CAT_DISPLAY.get(c, c) for c in boosted])
                                st.markdown(f"**Boosts:** {boost_display}")
                            if costly:
                                cost_display = ", ".join([_CAT_DISPLAY.get(c, c) for c in costly])
                                st.markdown(f"**Costs:** {cost_display}")

                            # ADP + ECR
                            adp_s = rec.get("adp_fairness", 0)
                            ecr_s = rec.get("ecr_fairness", 0)
                            st.caption(f"ADP Fairness: {adp_s:.0%} | ECR Fairness: {ecr_s:.0%}")

        # ── Tab 2: By Value ───────────────────────────────────────────
        with tab_value:
            st.markdown(
                f'<p style="font-size:13px;color:{T["tx2"]};">'
                "All trade opportunities ranked by composite value score.</p>",
                unsafe_allow_html=True,
            )

            df = _build_trade_df(opportunities)
            if not df.empty:
                # Show essential columns with explicit widths to prevent truncation
                essential = ["You Give", "You Receive", "Type", "Partner", "Your Gain", "Grade", "Acceptance", "Score"]
                display_df = df[[c for c in essential if c in df.columns]]
                col_config = {
                    "You Give": st.column_config.TextColumn("You Give", width="medium"),
                    "You Receive": st.column_config.TextColumn("You Receive", width="medium"),
                    "Partner": st.column_config.TextColumn("Partner", width="medium"),
                    "Your Gain": st.column_config.NumberColumn("Your Gain", format="%.2f"),
                    "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                }
                render_sortable_table(display_df, column_config=col_config, height=600)

        # ── Tab 4: Trade Readiness ───────────────────────────────────
        with tab_readiness:
            st.markdown(
                f'<p style="font-size:13px;color:{T["tx2"]};">'
                "Composite player scores (0-100) blending category fit, projection confidence, "
                "injury risk, positional scarcity, and free agent alternatives.</p>",
                unsafe_allow_html=True,
            )

            try:
                from src.trade_intelligence import compute_trade_readiness_batch

                # Collect all opponent player IDs
                opp_ids = []
                for tn, pids in league_rosters.items():
                    if tn != user_team_name:
                        opp_ids.extend(pids)

                if not opp_ids:
                    st.info("No opponent players found. Sync your Yahoo league data.")
                else:
                    # Get FA pool for comparison
                    try:
                        from src.league_manager import get_free_agents as _get_fa_pool

                        _fa_pool = _get_fa_pool(pool)
                    except Exception:
                        _fa_pool = pd.DataFrame()

                    user_totals = all_team_totals.get(user_team_name, {})

                    with st.spinner("Computing Trade Readiness scores..."):
                        readiness_df = compute_trade_readiness_batch(
                            player_ids=opp_ids,
                            user_roster_ids=user_roster_ids,
                            user_totals=user_totals,
                            all_team_totals=all_team_totals,
                            user_team_name=user_team_name,
                            fa_pool=_fa_pool,
                            player_pool=pool,
                            config=config,
                            max_players=100,
                        )

                    if readiness_df.empty:
                        st.info("Could not compute Trade Readiness scores.")
                    else:
                        # Add ADP tier classification
                        def _adp_tier(row):
                            adp = 999.0
                            pid = row.get("player_id")
                            if pid is not None:
                                p_match = pool[pool["player_id"] == pid]
                                if not p_match.empty:
                                    adp = float(p_match.iloc[0].get("adp", 999) or 999)
                            if adp <= 36:
                                return "Elite"
                            elif adp <= 96:
                                return "Core"
                            elif adp <= 180:
                                return "Depth"
                            else:
                                return "Filler"

                        readiness_df["adp_tier"] = readiness_df.apply(_adp_tier, axis=1)

                        # Position filter
                        all_positions = set()
                        for pos_str in readiness_df["positions"].dropna():
                            for p in str(pos_str).split(","):
                                p = p.strip()
                                if p:
                                    all_positions.add(p)
                        pos_filter = st.selectbox(
                            "Filter by position",
                            ["All"] + sorted(all_positions),
                            key="readiness_pos_filter",
                        )
                        if pos_filter != "All":
                            readiness_df = readiness_df[readiness_df["positions"].str.contains(pos_filter, na=False)]

                        # Display columns
                        display_cols = [
                            "name",
                            "positions",
                            "adp_tier",
                            "score",
                            "category_fit",
                            "projection_quality",
                            "health",
                            "scarcity",
                            "fa_advantage",
                            "fa_best",
                        ]
                        display_df = readiness_df[[c for c in display_cols if c in readiness_df.columns]].rename(
                            columns={
                                "name": "Player",
                                "positions": "Pos",
                                "adp_tier": "ADP Tier",
                                "score": "Readiness",
                                "category_fit": "Cat Fit",
                                "projection_quality": "Proj Conf",
                                "health": "Health",
                                "scarcity": "Scarcity",
                                "fa_advantage": "FA Edge",
                                "fa_best": "Best FA",
                            }
                        )
                        render_sortable_table(display_df, height=600)

                        # Scoring explanation
                        st.caption(
                            "Readiness = 40% Category Fit + 25% Projection Confidence "
                            "+ 15% Health + 10% Scarcity + 10% FA Edge. "
                            "Higher = better trade target for your team."
                        )

            except ImportError:
                st.info("Trade intelligence module not available.")

        # ── Tab 5: Target a Player ───────────────────────────────────
        with tab_target:
            st.subheader("Target a Player")
            st.caption(
                "Select any player in the league and get two trade proposals: a lowball offer and a fair value package."
            )

            # Build target options: all non-user rostered players, grouped by team
            target_options: list[str] = []
            for team_t, pids_t in sorted(league_rosters.items()):
                if team_t == user_team_name:
                    continue
                for pid_t in pids_t:
                    p_t = pool[pool["player_id"] == pid_t]
                    if not p_t.empty:
                        pname = str(p_t.iloc[0].get("name", p_t.iloc[0].get("player_name", f"ID {pid_t}")))
                        target_options.append(f"{team_t} | {pname}")

            selected_target = st.selectbox(
                "Select target player (Team | Player)",
                [""] + sorted(target_options),
                key="target_player_select",
            )

            if selected_target:
                # Parse team and player name
                parts = selected_target.split(" | ", 1)
                if len(parts) == 2:
                    target_team, target_name = parts
                    # Find player_id
                    name_col_t = "name" if "name" in pool.columns else "player_name"
                    target_match = pool[pool[name_col_t] == target_name]
                    if target_match.empty:
                        target_match = pool[pool[name_col_t].str.contains(target_name, case=False, na=False)]

                    if not target_match.empty:
                        target_pid = int(target_match.iloc[0]["player_id"])

                        with st.spinner("Generating proposals..."):
                            try:
                                from src.trade_intelligence import generate_targeted_proposals

                                try:
                                    from src.validation.dynamic_context import compute_weeks_remaining

                                    weeks_rem = compute_weeks_remaining()
                                except ImportError:
                                    weeks_rem = 22
                            except ImportError:
                                st.info("Trade intelligence module not available for targeted proposals.")
                                generate_targeted_proposals = None  # type: ignore[assignment]
                                weeks_rem = 22

                            proposals = None
                            if generate_targeted_proposals is not None:
                                try:
                                    proposals = generate_targeted_proposals(
                                        target_player_id=target_pid,
                                        user_roster_ids=user_roster_ids,
                                        player_pool=pool,
                                        config=config,
                                        all_team_totals=all_team_totals if all_team_totals else None,
                                        user_team_name=user_team_name,
                                        opponent_team_name=target_team,
                                        weeks_remaining=weeks_rem,
                                    )
                                except Exception as e:
                                    st.error(f"Error generating proposals: {e}")

                        if proposals:
                            # Target info header
                            target_info = proposals.get("target", {})
                            st.markdown(
                                f"**Target:** {target_info.get('name', target_name)} "
                                f"({target_info.get('positions', '')})"
                            )

                            # Two columns: Lowball | Fair Value
                            col_low, col_fair = st.columns(2)

                            for col_prop, label, key_prop, color in [
                                (col_low, "LOWBALL", "lowball", T["hot"]),
                                (col_fair, "FAIR VALUE", "fair_value", T["ok"]),
                            ]:
                                with col_prop:
                                    proposal = proposals.get(key_prop)
                                    if proposal is None:
                                        st.info(
                                            f"No {label.lower()} proposal found -- no viable package on your roster."
                                        )
                                        continue

                                    # Header
                                    st.markdown(
                                        f'<div style="background:{color}20;border:1px solid {color};'
                                        f'border-radius:8px;padding:12px;margin-bottom:8px;">'
                                        f'<div style="font-family:Bebas Neue,sans-serif;font-size:20px;'
                                        f'color:{color};letter-spacing:2px;">{label}</div>'
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                                    # Give players
                                    give_names = proposal.get("giving_names", [])
                                    st.markdown(f"**You give:** {', '.join(give_names) if give_names else 'N/A'}")

                                    # Metrics row
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Grade", proposal.get("grade", "N/A"))
                                    eff_data = proposal.get("efficiency", {})
                                    eff_ratio = eff_data.get("efficiency_ratio", 0) if isinstance(eff_data, dict) else 0
                                    m2.metric("Efficiency", f"{eff_ratio:.1f}x")
                                    m3.metric(
                                        "Accept",
                                        f"{proposal.get('acceptance_probability', 0):.0%}",
                                    )

                                    # Category impact
                                    cat_impact = proposal.get("category_impact", {})
                                    if cat_impact:
                                        impact_rows = []
                                        for cat_i in config.all_categories:
                                            val_i = cat_impact.get(cat_i, 0)
                                            display_name_i = _CAT_DISPLAY.get(cat_i, cat_i)
                                            if val_i > 0.05:
                                                arrow = "+"
                                            elif val_i < -0.05:
                                                arrow = "-"
                                            else:
                                                arrow = "~"
                                            impact_rows.append(
                                                {
                                                    "Category": display_name_i,
                                                    "Impact": f"{val_i:+.2f}",
                                                    "Direction": arrow,
                                                }
                                            )
                                        st.dataframe(
                                            pd.DataFrame(impact_rows),
                                            hide_index=True,
                                            use_container_width=True,
                                        )

                                    # ADP + ECR
                                    adp_fair_t = proposal.get("adp_fairness", 0)
                                    ecr_fair_t = proposal.get("ecr_fairness", 0)
                                    st.caption(f"ADP Fairness: {adp_fair_t:.0%} | ECR Fairness: {ecr_fair_t:.0%}")
                    else:
                        st.warning("Could not find the selected player in the player pool.")

        # ── Tab 6: Browse Partners ───────────────────────────────────
        with tab_browse:
            st.subheader("Browse Trade Partners")
            st.caption("Explore opponent rosters and compare category strengths to find trade fits.")

            opponent_teams = sorted([t for t in league_rosters if t != user_team_name])
            selected_browse_team = st.selectbox("Select opponent team", opponent_teams, key="browse_team_select")

            if selected_browse_team:
                # Complementarity score
                if all_team_totals:
                    browse_partners = find_complementary_teams(user_team_name, all_team_totals, config, top_n=12)
                    comp_score_browse = dict(browse_partners).get(selected_browse_team, 0.5)
                    st.caption(f"Complementarity score: **{comp_score_browse:.2f}** (higher = better trade fit)")

                # Category comparison table
                user_totals_browse = all_team_totals.get(user_team_name, _roster_category_totals(user_roster_ids, pool))
                opp_totals_browse = all_team_totals.get(
                    selected_browse_team,
                    _roster_category_totals(league_rosters.get(selected_browse_team, []), pool),
                )

                comp_rows: list[dict] = []
                for cat_b in config.all_categories:
                    display_name_b = _CAT_DISPLAY.get(cat_b, cat_b)
                    user_val_b = user_totals_browse.get(cat_b, 0)
                    opp_val_b = opp_totals_browse.get(cat_b, 0)

                    # Compute rank for each team
                    all_vals_b = (
                        [t.get(cat_b, 0) for t in all_team_totals.values()]
                        if all_team_totals
                        else [user_val_b, opp_val_b]
                    )
                    if cat_b in config.inverse_stats:
                        sorted_vals_b = sorted(all_vals_b)
                    else:
                        sorted_vals_b = sorted(all_vals_b, reverse=True)

                    user_rank_b = (
                        sorted_vals_b.index(user_val_b) + 1 if user_val_b in sorted_vals_b else len(sorted_vals_b)
                    )
                    opp_rank_b = (
                        sorted_vals_b.index(opp_val_b) + 1 if opp_val_b in sorted_vals_b else len(sorted_vals_b)
                    )
                    rank_gap_b = opp_rank_b - user_rank_b

                    if rank_gap_b >= 3:
                        opp_label_b = "YOU GAIN"
                    elif rank_gap_b <= -3:
                        opp_label_b = "THEY GAIN"
                    else:
                        opp_label_b = "EVEN"

                    comp_rows.append(
                        {
                            "Category": display_name_b,
                            "Your Rank": user_rank_b,
                            "Their Rank": opp_rank_b,
                            "Rank Gap": rank_gap_b,
                            "Opportunity": opp_label_b,
                        }
                    )

                st.markdown("**Category Comparison**")
                render_sortable_table(pd.DataFrame(comp_rows))

                # Suggested trades with this partner (from trade finder scan)
                partner_trades = trades_by_partner.get(selected_browse_team, [])
                if partner_trades:
                    st.markdown(f"**Suggested Trades ({len(partner_trades)} found)**")
                    df_partner = _build_trade_df(partner_trades[:10])
                    if not df_partner.empty:
                        essential_cols = [
                            "You Give",
                            "You Receive",
                            "Your Gain",
                            "Grade",
                            "Acceptance",
                            "Score",
                        ]
                        display_partner = df_partner[[c for c in essential_cols if c in df_partner.columns]]
                        render_sortable_table(display_partner, height=min(350, 50 + len(display_partner) * 35))
                else:
                    st.caption("No trade opportunities found with this team from the scan.")

                # Opponent roster
                opp_pids_browse = league_rosters.get(selected_browse_team, [])
                opp_pool_browse = pool[pool["player_id"].isin(opp_pids_browse)].copy()
                if not opp_pool_browse.empty:
                    st.markdown("**Opponent Roster**")
                    name_col_b = "name" if "name" in opp_pool_browse.columns else "player_name"
                    display_cols_b = [name_col_b, "positions"]
                    for col_b in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]:
                        if col_b in opp_pool_browse.columns:
                            display_cols_b.append(col_b)
                    display_df_b = opp_pool_browse[[c for c in display_cols_b if c in opp_pool_browse.columns]].copy()
                    display_df_b = display_df_b.rename(
                        columns={"name": "Player", "player_name": "Player", "positions": "Pos"}
                    )
                    render_sortable_table(display_df_b, height=500)

    page_timer_footer("Trade Finder")


if __name__ == "__main__":
    main()
else:
    main()
