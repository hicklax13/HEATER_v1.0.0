"""My Team — Roster overview and category standings."""

import time

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
    # If Yahoo is connected, offer immediate sync instead of a dead-end message
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_now"):
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
        st.warning("No user team identified in roster data.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        st.markdown(f"**Team:** {user_team_name}")

        # Refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Refresh Stats"):
                refresh_progress = st.progress(0, text="Pulling live stats from MLB Stats API...")
                refresh_progress.progress(20, text="Fetching current season statistics...")
                result = refresh_all_stats(force=True)
                refresh_progress.progress(90, text="Processing updated statistics...")
                for source, status in result.items():
                    st.toast(f"{source}: {status}")
                refresh_progress.progress(100, text="Stats refresh complete!")
                time.sleep(0.3)
                refresh_progress.empty()
                st.rerun()

        # Yahoo sync button
        if YFPY_AVAILABLE:
            import os

            yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
            yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
            yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
            if yahoo_key and yahoo_secret and yahoo_league_id:
                if st.button("Sync Yahoo", key="sync_yahoo_roster"):
                    progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                    try:
                        # Reuse authenticated client from session if available
                        client = st.session_state.get("yahoo_client")
                        if client is None:
                            # Fall back: try to authenticate with saved token data
                            token_data = st.session_state.get("yahoo_token_data")
                            client = YahooFantasyClient(league_id=yahoo_league_id)
                            if not client.authenticate(yahoo_key, yahoo_secret, token_data=token_data):
                                progress.empty()
                                st.error("Yahoo authentication failed. Reconnect in Connect League.")
                                client = None
                            else:
                                # Cache the successfully authenticated client
                                st.session_state.yahoo_client = client
                        if client is not None:
                            progress.progress(30, text="Fetching league data...")
                            sync_result = client.sync_to_db()
                            progress.progress(100, text="Sync complete!")
                            standings_count = sync_result.get("standings", 0) if sync_result else 0
                            rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                            if rosters_count > 0:
                                st.success(
                                    f"Synced {rosters_count} roster entries and {standings_count} standing entries."
                                )
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
                        _icon, label = get_injury_badge(hs)
                        # Use text label only — st.dataframe() cannot render HTML
                        badges.append(label)
                    else:
                        badges.append("Low Risk")
                roster["Health"] = badges

            # Display roster
            st.subheader("Roster (2026 Season)")
            display_cols = ["name", "positions", "roster_slot", "Health"]
            available_cols = [c for c in display_cols if c in roster.columns]
            col_config = {
                "name": st.column_config.TextColumn("Player"),
                "positions": st.column_config.TextColumn("Position(s)"),
                "roster_slot": st.column_config.TextColumn("Slot"),
                "Health": st.column_config.TextColumn("Health"),
            }
            st.dataframe(
                roster[available_cols] if available_cols else roster,
                column_config=col_config,
                width="stretch",
                hide_index=True,
            )

            # Category totals
            st.subheader("Category Totals (2026 Projected)")
            st.caption(
                "Projected full-season totals based on preseason projections "
                "(Steamer/ZiPS/Depth Charts blend). Updates to actual stats "
                "once the 2026 MLB season begins."
            )
            hitters = roster[roster["is_hitter"] == 1]
            pitchers = roster[roster["is_hitter"] == 0]

            hit_stats = {}
            if not hitters.empty:
                for cat, col in [("Runs", "r"), ("Home Runs", "hr"), ("Runs Batted In", "rbi"), ("Stolen Bases", "sb")]:
                    hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
                ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
                h = hitters["h"].sum() if "h" in hitters.columns else 0
                hit_stats["Batting Average"] = f"{h / ab:.3f}" if ab > 0 else ".000"

            pitch_stats = {}
            if not pitchers.empty:
                for cat, col in [("Wins", "w"), ("Saves", "sv"), ("Strikeouts", "k")]:
                    pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
                ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
                er = pitchers["er"].sum() if "er" in pitchers.columns else 0
                bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
                ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
                pitch_stats["Earned Run Average"] = f"{er * 9 / ip:.2f}" if ip > 0 else "0.00"
                pitch_stats["Walks + Hits per Inning Pitched"] = f"{(bb + ha) / ip:.3f}" if ip > 0 else "0.000"

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
                            bayes_progress = st.progress(0, text="Loading preseason projections for Bayesian update...")
                            preseason = pd.read_sql_query("SELECT * FROM projections WHERE system = 'blended'", conn)
                            bayes_progress.progress(
                                30, text="Applying Bayesian regression with stabilization thresholds..."
                            )
                            updater = BayesianUpdater()
                            updated = updater.batch_update_projections(season_stats, preseason)
                            bayes_progress.progress(100, text="Bayesian projections complete!")
                            time.sleep(0.3)
                            bayes_progress.empty()
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
