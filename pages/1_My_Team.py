"""My Team — Roster overview and category standings."""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, init_db, load_league_rosters, load_player_pool
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, THEME, inject_custom_css, render_styled_table

try:
    from src.bayesian import BayesianUpdater

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from src.yahoo_api import YFPY_AVAILABLE, YahooFantasyClient
except ImportError:
    YFPY_AVAILABLE = False

try:
    from src.player_news import generate_intel_summary, generate_roster_intel

    PLAYER_NEWS_AVAILABLE = True
except ImportError:
    PLAYER_NEWS_AVAILABLE = False

T = THEME

# -- Source badge colors by provider --
_SOURCE_BADGE_COLORS = {
    "espn": {"bg": "#c41230", "label": "ESPN"},
    "rotowire": {"bg": "#1a73e8", "label": "RotoWire"},
    "mlb": {"bg": "#002d72", "label": "MLB"},
    "yahoo": {"bg": "#6001d2", "label": "Yahoo"},
}

# -- Sentiment indicator helper --
_SENTIMENT_THRESHOLDS = [
    (0.2, T["green"], "Positive"),
    (-0.2, T["warn"], "Neutral"),
    (float("-inf"), T["danger"], "Negative"),
]


def _sentiment_indicator(score: float) -> str:
    """Return an HTML span with colored dot and label for a sentiment score."""
    for threshold, color, label in _SENTIMENT_THRESHOLDS:
        if score >= threshold:
            return (
                f'<span style="display:inline-flex;align-items:center;gap:4px;">'
                f'<span style="width:8px;height:8px;border-radius:50%;'
                f'background:{color};display:inline-block;"></span>'
                f'<span style="font-size:12px;color:{T["tx2"]};">{label}</span></span>'
            )
    # Fallback (should not reach here)
    return ""


def _ownership_arrow(direction: str, delta: float) -> str:
    """Return an HTML snippet for ownership trend arrow and delta."""
    if direction == "up":
        color = T["green"]
        arrow = f'<span style="color:{color};font-weight:700;">&#9650;</span>'
    elif direction == "down":
        color = T["danger"]
        arrow = f'<span style="color:{color};font-weight:700;">&#9660;</span>'
    else:
        color = T["tx2"]
        arrow = f'<span style="color:{color};font-weight:700;">&#8212;</span>'
    delta_str = f"{delta:+.1f}%" if delta != 0.0 else "0.0%"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:3px;">'
        f'{arrow} <span style="font-size:12px;color:{color};">{delta_str}</span></span>'
    )


def _source_badge(source: str) -> str:
    """Return an HTML badge for the news source."""
    info = _SOURCE_BADGE_COLORS.get(source, {"bg": T["tx2"], "label": source.upper()})
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:10px;'
        f"background:{info['bg']};color:#ffffff;font-size:11px;font-weight:700;"
        f'letter-spacing:0.5px;vertical-align:middle;">{info["label"]}</span>'
    )


def _news_type_label(news_type: str) -> str:
    """Return a styled label for the news type."""
    type_map = {
        "injury": {"color": T["danger"], "label": "Injury"},
        "transaction": {"color": T["purple"], "label": "Transaction"},
        "callup": {"color": T["green"], "label": "Call-Up"},
        "lineup": {"color": T["sky"], "label": "Lineup"},
        "general": {"color": T["tx2"], "label": "General"},
    }
    info = type_map.get(news_type, type_map["general"])
    return (
        f'<span style="font-size:11px;font-weight:600;color:{info["color"]};'
        f'text-transform:uppercase;letter-spacing:0.5px;">{info["label"]}</span>'
    )


def _render_news_card(player_name: str, news_item: dict, ownership: dict) -> str:
    """Build a self-contained HTML card for one news item."""
    headline = news_item.get("headline", "No headline")
    detail = news_item.get("detail", "")
    source = news_item.get("source", "")
    news_type = news_item.get("news_type", "general")
    sentiment = news_item.get("sentiment_score", 0.0)
    if sentiment is None:
        sentiment = 0.0
    il_status = news_item.get("il_status", "")

    # Generate analytical summary if the player_news module is available
    if PLAYER_NEWS_AVAILABLE:
        try:
            summary = generate_intel_summary(news_item)
            if summary and summary != detail:
                detail = summary
        except Exception:
            pass  # Fall back to raw detail

    # Build header line: source badge + news type + sentiment
    header_parts = [_source_badge(source), _news_type_label(news_type)]
    if il_status:
        header_parts.append(
            f'<span style="font-size:11px;font-weight:700;color:{T["danger"]};'
            f'padding:1px 6px;border:1px solid {T["danger"]};border-radius:8px;">'
            f"{il_status}</span>"
        )
    header_html = " ".join(header_parts)

    # Ownership trend line
    ownership_html = ""
    if ownership:
        current = ownership.get("current")
        direction = ownership.get("direction", "flat")
        delta = ownership.get("delta_7d", 0.0)
        if current is not None:
            ownership_html = (
                f'<div style="margin-top:6px;font-size:12px;color:{T["tx2"]};">'
                f"Ownership: {current:.1f}% {_ownership_arrow(direction, delta)}"
                f"</div>"
            )

    # Sentiment
    sentiment_html = _sentiment_indicator(sentiment)

    # Truncate detail for display
    display_detail = detail[:200] + "..." if len(detail) > 200 else detail

    return (
        f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
        f"border-radius:10px;padding:14px 16px;margin-bottom:10px;"
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f"{header_html} {sentiment_html}</div>"
        f'<div style="margin-top:8px;font-family:Figtree,sans-serif;">'
        f'<span style="font-size:15px;font-weight:700;color:{T["tx"]};">'
        f"{player_name}</span>"
        f'<span style="font-size:14px;color:{T["tx"]};margin-left:8px;">'
        f"{headline}</span></div>"
        f'<div style="margin-top:4px;font-size:13px;color:{T["tx2"]};">'
        f"{display_detail}</div>"
        f"{ownership_html}"
        f"</div>"
    )


def _render_news_tab(roster: "pd.DataFrame") -> None:
    """Render the News and Alerts tab content."""
    if not PLAYER_NEWS_AVAILABLE:
        st.info("Player news module is not available. Install required dependencies to enable news intelligence.")
        return

    # Get roster player IDs
    if "player_id" not in roster.columns:
        st.info("No player identifiers available for news lookup.")
        return

    roster_ids = roster["player_id"].dropna().astype(int).tolist()
    if not roster_ids:
        st.info("No players on your roster to fetch news for.")
        return

    # Load player pool for mlb_id resolution
    try:
        player_pool = load_player_pool()
    except Exception:
        player_pool = pd.DataFrame()

    # Build player name lookup from roster
    name_lookup = {}
    for _, row in roster.iterrows():
        pid = int(row.get("player_id", 0))
        name_lookup[pid] = row.get("name", f"Player {pid}")

    # Fetch intel with progress indicator
    with st.spinner("Fetching player news from all sources..."):
        try:
            intel = generate_roster_intel(roster_ids, player_pool)
        except Exception:
            intel = {}

    # Collect all news items with player context
    all_news: list[dict] = []
    for pid in roster_ids:
        player_intel = intel.get(pid, {})
        news_items = player_intel.get("news", [])
        ownership = player_intel.get("ownership", {})
        for item in news_items:
            all_news.append(
                {
                    "player_id": pid,
                    "player_name": name_lookup.get(pid, f"Player {pid}"),
                    "item": item,
                    "ownership": ownership,
                    "sentiment": item.get("sentiment_score", 0.0) or 0.0,
                    "news_type": item.get("news_type", "general"),
                    "published_at": item.get("published_at", ""),
                }
            )

    if not all_news:
        st.markdown(
            f'<div style="text-align:center;padding:40px 20px;">'
            f"{PAGE_ICONS.get('check', '')}"
            f'<p style="font-family:Figtree,sans-serif;font-size:16px;'
            f'color:{T["tx2"]};margin-top:12px;">'
            f"No recent news for your roster. Check back later for updates "
            f"on injuries, transactions, call-ups, and lineup changes.</p></div>",
            unsafe_allow_html=True,
        )
        return

    # Sort controls
    sort_options = {
        "Most Recent": "recency",
        "Severity (Injuries First)": "severity",
        "Sentiment Impact": "sentiment",
    }
    sort_label = st.selectbox(
        "Sort news by",
        options=list(sort_options.keys()),
        index=0,
        key="news_sort_order",
    )
    sort_key = sort_options[sort_label]

    # Apply sorting
    if sort_key == "recency":
        all_news.sort(key=lambda x: x["published_at"], reverse=True)
    elif sort_key == "severity":
        severity_order = {"injury": 0, "transaction": 1, "callup": 2, "lineup": 3, "general": 4}
        all_news.sort(key=lambda x: severity_order.get(x["news_type"], 5))
    elif sort_key == "sentiment":
        all_news.sort(key=lambda x: x["sentiment"])

    # Summary count
    injury_count = sum(1 for n in all_news if n["news_type"] == "injury")
    transaction_count = sum(1 for n in all_news if n["news_type"] == "transaction")
    other_count = len(all_news) - injury_count - transaction_count

    summary_parts = []
    if injury_count > 0:
        summary_parts.append(f'<span style="color:{T["danger"]};font-weight:700;">{injury_count} injury</span>')
    if transaction_count > 0:
        summary_parts.append(
            f'<span style="color:{T["purple"]};font-weight:700;">{transaction_count} transaction</span>'
        )
    if other_count > 0:
        summary_parts.append(f"{other_count} other")

    st.markdown(
        f'<div style="font-family:Figtree,sans-serif;font-size:14px;'
        f'color:{T["tx2"]};margin-bottom:12px;">'
        f"{PAGE_ICONS.get('alert', '')} "
        f"{len(all_news)} news item{'s' if len(all_news) != 1 else ''} "
        f"found: {', '.join(summary_parts)}</div>",
        unsafe_allow_html=True,
    )

    # Render news cards
    for entry in all_news:
        card_html = _render_news_card(
            player_name=entry["player_name"],
            news_item=entry["item"],
            ownership=entry["ownership"],
        )
        st.markdown(card_html, unsafe_allow_html=True)


st.set_page_config(page_title="Heater | My Team", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>MY TEAM</span></div></div>', unsafe_allow_html=True
)

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
        # Team name with styled monogram avatar
        initials = "".join(w[0].upper() for w in user_team_name.split()[:2]) if user_team_name else "T"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0 8px 2px;">'
            f'<div style="width:36px;height:36px;min-width:36px;border-radius:50%;'
            f"background:linear-gradient(135deg,#e65c00,#cc5200);"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:1px;"
            f'color:#ffffff;font-weight:700;box-shadow:0 2px 8px rgba(230,92,0,0.25);">'
            f"{initials}</div>"
            f'<span style="font-family:Figtree,sans-serif;font-size:16px;font-weight:700;'
            f'color:#1d1d1f;">Team: {user_team_name}</span></div>',
            unsafe_allow_html=True,
        )

        # Action buttons — inline row
        btn1, btn2, btn_spacer = st.columns([1, 1, 3])
        with btn1:
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

        # Yahoo sync button — in btn2 column (same row as Refresh)
        if YFPY_AVAILABLE:
            import os

            yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
            yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
            yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
            if yahoo_key and yahoo_secret and yahoo_league_id:
                with btn2:
                    yahoo_clicked = st.button("Sync Yahoo", key="sync_yahoo_roster")
                if yahoo_clicked:
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
                    injury_df = coerce_numeric_df(injury_df)
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

            # -- Tab layout: Roster Overview + News and Alerts --
            tab_roster, tab_news = st.tabs(["Roster Overview", "News and Alerts"])

            with tab_roster:
                # Display roster
                st.subheader("Roster (2026 Season)")
                display_cols = ["name", "positions", "roster_slot", "Health"]
                available_cols = [c for c in display_cols if c in roster.columns]
                display_df = (roster[available_cols] if available_cols else roster).copy()
                rename_map = {
                    "name": "Player",
                    "positions": "Position(s)",
                    "roster_slot": "Slot",
                }
                display_df.rename(
                    columns={k: v for k, v in rename_map.items() if k in display_df.columns},
                    inplace=True,
                )
                render_styled_table(display_df)

                # Category totals
                st.subheader("Category Totals (2026 Projected)")
                st.caption(
                    "Projected full-season totals based on preseason projections "
                    "(Steamer/ZiPS/Depth Charts blend). Updates to actual stats "
                    "once the 2026 MLB season begins."
                )
                hitters = roster[roster["is_hitter"] == 1]
                pitchers = roster[roster["is_hitter"] == 0]

                # Coerce numeric columns (Python 3.13+ SQLite may return bytes)
                num_cols = ["r", "hr", "rbi", "sb", "ab", "h", "w", "sv", "k", "ip", "er", "bb_allowed", "h_allowed"]
                for c in num_cols:
                    if c in roster.columns:
                        roster[c] = pd.to_numeric(roster[c], errors="coerce").fillna(0)
                hitters = roster[roster["is_hitter"] == 1]
                pitchers = roster[roster["is_hitter"] == 0]

                hit_stats = {}
                if not hitters.empty:
                    for cat, col in [
                        ("Runs", "r"),
                        ("Home Runs", "hr"),
                        ("Runs Batted In", "rbi"),
                        ("Stolen Bases", "sb"),
                    ]:
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
                        render_styled_table(pd.DataFrame([hit_stats]))
                        st.caption(METRIC_TOOLTIPS["avg"])
                with col2:
                    st.markdown("**Pitching**")
                    if pitch_stats:
                        render_styled_table(pd.DataFrame([pitch_stats]))
                        st.caption(METRIC_TOOLTIPS["era"] + " | " + METRIC_TOOLTIPS["whip"])

                # Bayesian-adjusted projections
                if BAYESIAN_AVAILABLE:
                    try:
                        from src.database import get_connection

                        conn = get_connection()
                        try:
                            season_stats = pd.read_sql_query("SELECT * FROM season_stats", conn)
                            season_stats = coerce_numeric_df(season_stats)

                            if not season_stats.empty and season_stats.get("games_played", pd.Series([0])).sum() > 0:
                                bayes_progress = st.progress(
                                    0, text="Loading preseason projections for Bayesian update..."
                                )
                                preseason = pd.read_sql_query(
                                    "SELECT * FROM projections WHERE system = 'blended'", conn
                                )
                                preseason = coerce_numeric_df(preseason)
                                bayes_progress.progress(
                                    30,
                                    text="Applying Bayesian regression with stabilization thresholds...",
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
                                bayes_df = updated[show_cols].copy()
                                bayes_rename = {
                                    "player_id": "ID",
                                    "avg": "AVG",
                                    "hr": "HR",
                                    "rbi": "RBI",
                                    "sb": "SB",
                                    "era": "ERA",
                                    "whip": "WHIP",
                                    "k": "K",
                                }
                                bayes_df.rename(
                                    columns={k: v for k, v in bayes_rename.items() if k in bayes_df.columns},
                                    inplace=True,
                                )
                                # Format numeric columns for display
                                for c in ["AVG", "WHIP"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.3f}")
                                for c in ["ERA"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.2f}")
                                for c in ["HR", "RBI", "SB", "K", "ID"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.0f}")
                                render_styled_table(bayes_df)
                        finally:
                            conn.close()
                    except Exception:
                        pass  # Graceful degradation

            # -- News and Alerts tab --
            with tab_news:
                _render_news_tab(roster)
