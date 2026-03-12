"""My Team — Roster overview and category standings."""

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.ui_shared import METRIC_TOOLTIPS, inject_custom_css, render_theme_toggle

try:
    from src.bayesian import BayesianUpdater

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from src.yahoo_api import YFPY_AVAILABLE, YahooFantasyClient
except ImportError:
    YFPY_AVAILABLE = False

st.set_page_config(page_title="My Team", page_icon="", layout="wide")

init_db()

inject_custom_css()
render_theme_toggle()

st.title("My Team")

# Determine user team
rosters = load_league_rosters()
if rosters.empty:
    st.warning("No league data loaded. Import your league rosters in the main app (Setup Step 3).")
    st.stop()
else:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if user_teams.empty:
        st.warning("No user team identified in roster data.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        st.markdown(f"**Team:** {user_team_name}")

        # Refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Refresh Stats"):
                with st.spinner("Pulling live stats from MLB..."):
                    result = refresh_all_stats(force=True)
                    for source, status in result.items():
                        st.toast(f"{source}: {status}")
                    st.rerun()

        # Yahoo sync button
        if YFPY_AVAILABLE:
            import os

            yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
            yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
            yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
            if yahoo_key and yahoo_secret and yahoo_league_id:
                if st.button("Sync Yahoo"):
                    with st.spinner("Syncing from Yahoo Fantasy..."):
                        try:
                            client = YahooFantasyClient(league_id=yahoo_league_id)
                            if client.authenticate(yahoo_key, yahoo_secret):
                                client.sync_to_db()
                                st.toast("Yahoo sync complete!")
                                st.rerun()
                            else:
                                st.error("Yahoo authentication failed.")
                        except Exception as e:
                            st.error(f"Yahoo sync error: {e}")

        # Load roster
        roster = get_team_roster(user_team_name)
        if roster.empty:
            st.info("No players on your roster yet.")
        else:
            # Compute health scores for badge display
            try:
                from src.database import get_connection

                conn = get_connection()
                try:
                    injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
                finally:
                    conn.close()
            except Exception:
                injury_df = pd.DataFrame()

            if not injury_df.empty and "player_id" in injury_df.columns:
                badges = []
                for _, row in roster.iterrows():
                    pid = row.get("player_id")
                    player_injury = injury_df[injury_df["player_id"] == pid]
                    if not player_injury.empty:
                        gp = player_injury["games_played"].tolist()
                        ga = player_injury["games_available"].tolist()
                        hs = compute_health_score(gp, ga)
                        icon, label = get_injury_badge(hs)
                        badges.append(f"{icon} {label}")
                    else:
                        badges.append(
                            '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                            'vertical-align:middle;margin-right:4px;background:#84cc16;"></span>Low Risk'
                        )
                roster["Health"] = badges

            # Display roster
            st.subheader("Roster")
            display_cols = ["name", "positions", "roster_slot", "Health"]
            available_cols = [c for c in display_cols if c in roster.columns]
            st.dataframe(
                roster[available_cols] if available_cols else roster,
                width="stretch",
                hide_index=True,
            )

            # Category totals
            st.subheader("Category Totals")
            hitters = roster[roster["is_hitter"] == 1]
            pitchers = roster[roster["is_hitter"] == 0]

            hit_stats = {}
            if not hitters.empty:
                for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
                    hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
                ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
                h = hitters["h"].sum() if "h" in hitters.columns else 0
                hit_stats["AVG"] = f"{h / ab:.3f}" if ab > 0 else ".000"

            pitch_stats = {}
            if not pitchers.empty:
                for cat, col in [("W", "w"), ("SV", "sv"), ("K", "k")]:
                    pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
                ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
                er = pitchers["er"].sum() if "er" in pitchers.columns else 0
                bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
                ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
                pitch_stats["ERA"] = f"{er * 9 / ip:.2f}" if ip > 0 else "0.00"
                pitch_stats["WHIP"] = f"{(bb + ha) / ip:.3f}" if ip > 0 else "0.000"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hitting**")
                if hit_stats:
                    st.dataframe(pd.DataFrame([hit_stats]), hide_index=True)
                    st.caption(METRIC_TOOLTIPS["avg"])
            with col2:
                st.markdown("**Pitching**")
                if pitch_stats:
                    st.dataframe(pd.DataFrame([pitch_stats]), hide_index=True)
                    st.caption(METRIC_TOOLTIPS["era"] + " | " + METRIC_TOOLTIPS["whip"])

            # Bayesian-adjusted projections
            if BAYESIAN_AVAILABLE:
                try:
                    from src.database import get_connection

                    conn = get_connection()
                    try:
                        season_stats = pd.read_sql_query("SELECT * FROM season_stats", conn)

                        if not season_stats.empty and season_stats.get("games_played", pd.Series([0])).sum() > 0:
                            preseason = pd.read_sql_query("SELECT * FROM projections WHERE system = 'blended'", conn)
                            updater = BayesianUpdater()
                            updated = updater.batch_update_projections(season_stats, preseason)
                            st.subheader("Bayesian-Adjusted Projections")
                            st.caption(
                                "Stats regressed toward preseason priors using FanGraphs stabilization thresholds"
                            )
                            stat_display = ["player_id", "avg", "hr", "rbi", "sb", "era", "whip", "k"]
                            show_cols = [c for c in stat_display if c in updated.columns]
                            st.dataframe(updated[show_cols], hide_index=True, width="stretch")
                    finally:
                        conn.close()
                except Exception:
                    pass  # Graceful degradation
