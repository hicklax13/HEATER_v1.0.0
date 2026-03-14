"""Trade Analyzer — Evaluate trade proposals with MC simulation + Live SGP."""

import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_player_pool
from src.in_season import analyze_trade
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, T, inject_custom_css
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

st.set_page_config(page_title="Heater | Trade Analyzer", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown('<div class="page-title">TRADE ANALYZER</div>', unsafe_allow_html=True)

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
                        import time

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
        user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

        # Player selection — build name-to-id mapping from both pool AND roster
        # so that roster players missing from player_pool still appear.
        player_names = pool[["player_id", "player_name"]].drop_duplicates()
        name_to_id = dict(zip(player_names["player_name"], player_names["player_id"]))

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

                trade_progress = st.progress(0, text="Computing category impacts...")
                trade_progress.progress(20, text="Computing category impacts...")
                trade_progress.progress(40, text="Running Monte Carlo simulation (200 iterations)...")
                result = analyze_trade(
                    giving_ids=giving_ids,
                    receiving_ids=receiving_ids,
                    user_roster_ids=user_roster_ids,
                    player_pool=pool,
                    config=config,
                )
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

                st.markdown(
                    f'<div class="glass" style="border:2px solid {color};'
                    f"padding:20px;text-align:center;margin:16px 0;"
                    f'animation:slideUp 0.4s ease-out both;">'
                    f"{icon}"
                    f'<span style="font-family:Bebas Neue,sans-serif;font-size:28px;color:{color};'
                    f'letter-spacing:2px;margin-left:12px;">{result["verdict"]}</span>'
                    f'<span style="color:{T["tx2"]};margin-left:12px;font-size:18px;">'
                    f"{result['confidence_pct']:.1f}% confidence</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption(METRIC_TOOLTIPS["trade_verdict"])

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Total Standings Gained Points Change",
                    f"{result['total_sgp_change']:+.3f}",
                    help=METRIC_TOOLTIPS["sgp"],
                )
                col2.metric("Monte Carlo Mean", f"{result['mc_mean']:+.3f}", help=METRIC_TOOLTIPS["mc_mean"])
                col3.metric("Monte Carlo Standard Deviation", f"{result['mc_std']:.3f}", help=METRIC_TOOLTIPS["mc_std"])

                # Category impact table
                st.subheader("Category Impact")
                impact_df = pd.DataFrame(
                    [
                        {"Category": cat, "Standings Gained Points Change": f"{val:+.3f}"}
                        for cat, val in result["category_impact"].items()
                    ]
                )
                st.dataframe(impact_df, width="stretch", hide_index=True)

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
                            icon, label = get_injury_badge(hs)
                            st.markdown(f"{icon} {name} — {label}")
                with trade_col2:
                    st.markdown("**Receiving:**")
                    for name in receiving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            pid = p.iloc[0]["player_id"]
                            hs = health_dict.get(pid, 0.85)
                            icon, label = get_injury_badge(hs)
                            st.markdown(f"{icon} {name} — {label}")

                # P10/P90 risk assessment for traded players
                try:
                    from src.database import get_connection

                    conn = get_connection()
                    try:
                        systems = {}
                        for sys_name in ["steamer", "zips", "depthcharts", "blended"]:
                            df = pd.read_sql_query(
                                "SELECT * FROM projections WHERE system = ?",
                                conn,
                                params=(sys_name,),
                            )
                            if not df.empty:
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
                                st.dataframe(pd.DataFrame(risk_rows), hide_index=True, width="stretch")
                                st.caption(METRIC_TOOLTIPS["p10_p90"])
                except Exception:
                    pass  # Graceful degradation
