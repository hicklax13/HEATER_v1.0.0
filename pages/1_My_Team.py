"""My Team — Roster overview and category standings."""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, init_db, load_league_rosters, load_player_pool
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.ui_shared import (
    PAGE_ICONS,
    THEME,
    inject_custom_css,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
)

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
    import html as _ht

    headline = _ht.escape(news_item.get("headline", "No headline"))
    detail = news_item.get("detail", "")
    source = news_item.get("source", "")
    news_type = news_item.get("news_type", "general")
    sentiment = news_item.get("sentiment_score", 0.0)
    if sentiment is None:
        sentiment = 0.0
    il_status = _ht.escape(news_item.get("il_status", ""))
    published_at = news_item.get("published_at", "")
    player_name = _ht.escape(player_name)

    # Generate analytical summary — but only if there's real detail content.
    # Yahoo injury entries have only a body part as headline (e.g., "Hand")
    # with no detail text. Skip the template generator for these to avoid
    # fake-sounding summaries with placeholder zeros.
    has_real_content = bool(detail and detail.strip())
    if PLAYER_NEWS_AVAILABLE and has_real_content:
        try:
            summary = generate_intel_summary(news_item)
            if summary and summary != detail:
                detail = summary
        except Exception:
            pass

    # For Yahoo injury-only entries, build a clean status line instead
    if not has_real_content and il_status and headline:
        detail = f"{player_name} is listed as {il_status} with a {headline.lower()} issue."

    # Format date for display
    date_html = ""
    if published_at:
        try:
            from datetime import datetime

            if isinstance(published_at, str) and len(published_at) >= 10:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                date_html = (
                    f'<span style="font-size:11px;color:{T["tx2"]};'
                    f'font-family:IBM Plex Mono,monospace;">'
                    f"{dt.strftime('%b %d, %Y')}</span>"
                )
        except (ValueError, TypeError):
            pass

    # Build header line: source badge + news type + sentiment
    header_parts = [_source_badge(source), _news_type_label(news_type)]
    if date_html:
        header_parts.append(date_html)
    if il_status:
        header_parts.append(
            f'<span style="font-size:11px;font-weight:700;color:{T["danger"]};'
            f'padding:1px 6px;border:1px solid {T["danger"]};border-radius:8px;">'
            f"{il_status}</span>"
        )
    header_html = " ".join(header_parts)

    # Ownership trend line — only show if there's meaningful data
    ownership_html = ""
    if ownership:
        current = ownership.get("current")
        direction = ownership.get("direction", "flat")
        delta = ownership.get("delta_7d", 0.0)
        if current is not None and current > 0:
            ownership_html = (
                f'<div style="margin-top:6px;font-size:12px;color:{T["tx2"]};">'
                f"Ownership: {current:.1f}% {_ownership_arrow(direction, delta)}"
                f"</div>"
            )

    # Sentiment
    sentiment_html = _sentiment_indicator(sentiment)

    # Truncate detail for display
    display_detail = _ht.escape(detail[:200] + "..." if len(detail) > 200 else detail)

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

    # Deduplicate news by headline + player (case-insensitive)
    seen_keys: set[str] = set()
    unique_news: list[dict] = []
    for entry in all_news:
        key = (entry.get("player_name", "") + "|" + (entry.get("item", {}).get("headline", ""))).strip().lower()
        if key not in seen_keys:
            seen_keys.add(key)
            unique_news.append(entry)
    all_news = unique_news

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
        all_news.sort(key=lambda x: x["published_at"] or "", reverse=True)
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


st.set_page_config(page_title="Heater | My Team", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()

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
        if isinstance(user_team_name, bytes):
            user_team_name = user_team_name.decode("utf-8", errors="replace")

        # Try to get Yahoo client — from session or reconnect from saved token
        yahoo_client = st.session_state.get("yahoo_client")
        if yahoo_client is None and YFPY_AVAILABLE:
            import os

            _ykey = os.environ.get("YAHOO_CLIENT_ID", "")
            _ysecret = os.environ.get("YAHOO_CLIENT_SECRET", "")
            _yleague = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
            if _ykey and _ysecret and _yleague:
                try:
                    import json

                    with open("data/yahoo_token.json") as _tf:
                        _tdata = json.load(_tf)
                    _client = YahooFantasyClient(league_id=_yleague)
                    if _client.authenticate(_ykey, _ysecret, token_data=_tdata):
                        yahoo_client = _client
                        st.session_state.yahoo_client = _client
                except Exception:
                    pass

        # Try to fetch team logo from Yahoo
        team_logo_url = ""
        if yahoo_client:
            try:
                standings_data = yahoo_client._query.get_league_standings()
                for t in getattr(standings_data, "teams", None) or []:
                    t_name = t.name
                    if isinstance(t_name, bytes):
                        t_name = t_name.decode("utf-8", errors="replace")
                    if t_name == user_team_name:
                        logos = getattr(t, "team_logos", None) or []
                        if logos:
                            team_logo_url = getattr(logos[0], "url", "") or ""
                        break
            except Exception:
                pass

        # Team header with Yahoo logo or monogram fallback
        import html as _html
        import re as _re

        if team_logo_url:
            avatar_html = (
                f'<img src="{team_logo_url}" '
                f'style="width:40px;height:40px;min-width:40px;border-radius:50%;'
                f'object-fit:cover;box-shadow:0 2px 8px rgba(0,0,0,0.15);" '
                f'alt="Team logo"/>'
            )
        else:
            # Strip emoji/symbols to get clean initials
            clean_name = _re.sub(r"[^\w\s]", "", user_team_name, flags=_re.UNICODE).strip()
            words = [w for w in clean_name.split() if w and w[0].isalpha()]
            initials = "".join(w[0].upper() for w in words[:2]) if words else "T"
            avatar_html = (
                f'<div style="width:40px;height:40px;min-width:40px;border-radius:50%;'
                f"background:linear-gradient(135deg,#e65c00,#cc5200);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:1px;"
                f'color:#ffffff;font-weight:700;box-shadow:0 2px 8px rgba(230,92,0,0.25);">'
                f"{initials}</div>"
            )

        safe_name = _html.escape(user_team_name)
        st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:10px 4px;overflow:visible !important;min-height:52px;">'
            f"{avatar_html}"
            f'<span style="font-family:Figtree,sans-serif;font-size:16px;font-weight:700;'
            f'color:#1d1d1f;">{safe_name}</span></div>',
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
                        il_stints = (
                            player_injury["il_stints"].tolist()
                            if "il_stints" in player_injury.columns
                            else None
                        )
                        il_days = (
                            player_injury["il_days"].tolist()
                            if "il_days" in player_injury.columns
                            else None
                        )
                        hs = compute_health_score(gp, ga, il_stints, il_days)
                        _icon, label = get_injury_badge(hs)
                        badges.append(label)
                    else:
                        badges.append("Low Risk")
                roster["Health"] = badges

            # -- Compute category totals for context panel --
            # Coerce numeric columns (Python 3.13+ SQLite may return bytes)
            num_cols = [
                "r",
                "hr",
                "rbi",
                "sb",
                "ab",
                "h",
                "bb",
                "hbp",
                "sf",
                "w",
                "l",
                "sv",
                "k",
                "ip",
                "er",
                "bb_allowed",
                "h_allowed",
            ]
            for c in num_cols:
                if c in roster.columns:
                    roster[c] = pd.to_numeric(roster[c], errors="coerce").fillna(0)
            hitters = roster[roster["is_hitter"] == 1]
            pitchers = roster[roster["is_hitter"] == 0]

            hit_stats = {}
            if not hitters.empty:
                for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
                    hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
                ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
                h = hitters["h"].sum() if "h" in hitters.columns else 0
                hit_stats["AVG"] = f"{h / ab:.3f}" if ab > 0 else ".000"
                hit_h = hitters["h"].sum() if "h" in hitters.columns else 0
                hit_bb = hitters["bb"].sum() if "bb" in hitters.columns else 0
                hit_hbp = hitters["hbp"].sum() if "hbp" in hitters.columns else 0
                hit_sf = hitters["sf"].sum() if "sf" in hitters.columns else 0
                hit_ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
                obp_denom = hit_ab + hit_bb + hit_hbp + hit_sf
                hit_stats["OBP"] = f"{(hit_h + hit_bb + hit_hbp) / obp_denom:.3f}" if obp_denom > 0 else ".000"

            pitch_stats = {}
            if not pitchers.empty:
                for cat, col in [("W", "w"), ("L", "l"), ("SV", "sv"), ("K", "k")]:
                    pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
                ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
                er = pitchers["er"].sum() if "er" in pitchers.columns else 0
                bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
                ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
                pitch_stats["ERA"] = f"{er * 9 / ip:.2f}" if ip > 0 else "0.00"
                pitch_stats["WHIP"] = f"{(bb + ha) / ip:.3f}" if ip > 0 else "0.000"

            # -- IL alerts --
            il_players = []
            if "Health" in roster.columns:
                for _, row in roster.iterrows():
                    health = row.get("Health", "")
                    if health in ("IL", "IL-60", "Out", "Day-to-Day"):
                        il_players.append({"name": row.get("name", "Unknown"), "status": health})

            # -- Banner teaser --
            n_hitters = len(hitters)
            n_pitchers = len(pitchers)
            n_total = len(roster)
            banner_teaser = f"Roster: {n_total} players | {n_hitters} hitters, {n_pitchers} pitchers"
            if il_players:
                banner_teaser += f" | {len(il_players)} on IL/DTD"

            render_page_layout("MY TEAM", banner_teaser=banner_teaser, banner_icon="my_team")

            # -- 3-Zone Layout --
            ctx, main = render_context_columns()

            # -- Context Panel (left) --
            with ctx:
                # Category totals card — Hitting
                if hit_stats:
                    hit_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                        f'<span style="color:{T["tx2"]};">{k}</span>'
                        f'<span style="font-weight:600;color:{T["tx"]};">{v}</span></div>'
                        for k, v in hit_stats.items()
                    )
                    render_context_card("Hitting Totals", hit_html)

                # Category totals card — Pitching
                if pitch_stats:
                    pitch_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                        f'<span style="color:{T["tx2"]};">{k}</span>'
                        f'<span style="font-weight:600;color:{T["tx"]};">{v}</span></div>'
                        for k, v in pitch_stats.items()
                    )
                    render_context_card("Pitching Totals", pitch_html)

                # IL alerts card
                if il_players:
                    il_html = "".join(
                        f'<div style="padding:2px 0;font-size:12px;">'
                        f'<span class="health-dot" style="background:{T["danger"]};"></span>'
                        f'<span style="font-weight:600;">{p["name"]}</span> '
                        f'<span style="color:{T["tx2"]};">({p["status"]})</span></div>'
                        for p in il_players
                    )
                    render_context_card("Injured List Alerts", il_html)

            # -- Main Content (right) --
            with main:
                # Season/projection toggle
                stat_view = st.segmented_control(
                    "View stats from",
                    options=["2026 Projected", "2025", "2024", "2023"],
                    default="2026 Projected",
                    key="roster_stat_view",
                )

                base_cols = ["name", "positions", "roster_slot"]
                stat_cols_ordered = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
                rename_map = {
                    "name": "Player",
                    "positions": "Pos",
                    "roster_slot": "Slot",
                    "r": "R",
                    "hr": "HR",
                    "rbi": "RBI",
                    "sb": "SB",
                    "avg": "AVG",
                    "obp": "OBP",
                    "w": "W",
                    "l": "L",
                    "sv": "SV",
                    "k": "K",
                    "era": "ERA",
                    "whip": "WHIP",
                }

                if stat_view in ("2025", "2024", "2023"):
                    # Load historical stats for selected season
                    season_year = int(stat_view)
                    from src.database import get_connection as _gc

                    _conn = _gc()
                    try:
                        hist = pd.read_sql_query(
                            "SELECT * FROM season_stats WHERE season = ?",
                            _conn,
                            params=[season_year],
                        )
                        hist = coerce_numeric_df(hist)
                    finally:
                        _conn.close()

                    display_df = roster[["name", "positions", "roster_slot", "player_id"]].copy()
                    if not hist.empty and "player_id" in hist.columns:
                        hist_stat_cols = [c.lower() for c in stat_cols_ordered if c.lower() in hist.columns]
                        hist_slim = hist[["player_id"] + hist_stat_cols].copy()
                        display_df = display_df.merge(hist_slim, on="player_id", how="left")

                    if "Health" in roster.columns:
                        display_df["Health"] = roster["Health"].values

                    player_ids_list = display_df["player_id"].tolist()
                    display_df = display_df.drop(columns=["player_id"])
                    display_df.rename(
                        columns={k: v for k, v in rename_map.items() if k in display_df.columns},
                        inplace=True,
                    )

                    caption = f"Actual stats from the {season_year} MLB season."
                    if hist.empty:
                        caption = f"No historical data available for {season_year}."

                    st.markdown(
                        f'<div class="sec-head">Roster — {season_year} Season Stats</div>'
                        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
                        f'font-family:Figtree,sans-serif;">{caption}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # 2026 projected stats from roster
                    display_cols = list(base_cols)
                    for sc in stat_cols_ordered:
                        lc = sc.lower()
                        if lc in roster.columns:
                            display_cols.append(lc)
                    if "Health" in roster.columns:
                        display_cols.append("Health")
                    available_cols = [c for c in display_cols if c in roster.columns]
                    display_df = roster[available_cols].copy()
                    player_ids_list = roster["player_id"].tolist() if "player_id" in roster.columns else []
                    display_df.rename(
                        columns={k: v for k, v in rename_map.items() if k in display_df.columns},
                        inplace=True,
                    )

                    st.markdown(
                        f'<div class="sec-head">Roster — 2026 Projected Stats</div>'
                        f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:8px;'
                        f'font-family:Figtree,sans-serif;">'
                        f"Full-season projections from blended system "
                        f"(Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X). "
                        f"Updates to actual stats once the season begins.</div>",
                        unsafe_allow_html=True,
                    )

                # Assign row classes: starters vs bench
                row_cls = {}
                if "Slot" in display_df.columns:
                    for i, slot in enumerate(display_df["Slot"]):
                        if slot and str(slot).upper() in ("BN", "BENCH", "IL", "IL+", "NA"):
                            row_cls[i] = "row-bench"
                        else:
                            row_cls[i] = "row-start"

                health_col_name = "Health" if "Health" in display_df.columns else None
                render_compact_table(
                    display_df,
                    row_classes=row_cls,
                    health_col=health_col_name,
                    max_height=600,
                )

                # Export to Excel
                import io

                excel_buf = io.BytesIO()
                display_df.to_excel(excel_buf, index=False, sheet_name="Roster")
                excel_buf.seek(0)
                view_label = stat_view if stat_view else "2026_Projected"
                st.download_button(
                    "Export to Excel",
                    data=excel_buf,
                    file_name=f"heater_roster_{view_label.replace(' ', '_').lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="export_roster_excel",
                )

                # Player card selector
                if player_ids_list:
                    player_names = display_df["Player"].tolist() if "Player" in display_df.columns else []
                    if player_names:
                        render_player_select(
                            player_names,
                            player_ids_list,
                            key_suffix="myteam",
                        )

                # Bayesian-adjusted projections
                if BAYESIAN_AVAILABLE:
                    try:
                        from src.database import get_connection

                        conn = get_connection()
                        try:
                            # Filter to roster players only, latest season
                            roster_pids = roster["player_id"].tolist() if "player_id" in roster.columns else []
                            if roster_pids:
                                pids_str = ",".join(str(int(p)) for p in roster_pids)
                                season_stats = pd.read_sql_query(
                                    f"SELECT * FROM season_stats WHERE player_id IN ({pids_str}) ORDER BY season DESC",
                                    conn,
                                )
                            else:
                                season_stats = pd.DataFrame()
                            season_stats = coerce_numeric_df(season_stats)

                            # Keep only the latest season per player
                            if not season_stats.empty and "season" in season_stats.columns:
                                season_stats = season_stats.sort_values("season", ascending=False).drop_duplicates(
                                    subset=["player_id"], keep="first"
                                )

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
                                st.markdown(
                                    '<div class="sec-head">Bayesian-Adjusted Projections</div>',
                                    unsafe_allow_html=True,
                                )
                                stat_display = ["player_id", "avg", "hr", "rbi", "sb", "era", "whip", "k"]
                                show_cols = [c for c in stat_display if c in updated.columns]
                                bayes_df = updated[show_cols].copy()
                                # Replace player_id with player name from players table
                                if "player_id" in bayes_df.columns:
                                    players_lookup = pd.read_sql_query("SELECT player_id, name FROM players", conn)
                                    pid_to_name = dict(zip(players_lookup["player_id"], players_lookup["name"]))
                                    bayes_df["player_id"] = bayes_df["player_id"].map(
                                        lambda x: pid_to_name.get(int(x), f"Player {int(x)}") if pd.notna(x) else ""
                                    )
                                bayes_rename = {
                                    "player_id": "Player",
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
                                for c in ["AVG", "WHIP"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.3f}")
                                for c in ["ERA"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.2f}")
                                for c in ["HR", "RBI", "SB", "K", "ID"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.0f}")
                                render_compact_table(bayes_df, max_height=400)
                        finally:
                            conn.close()
                    except Exception:
                        pass  # Graceful degradation

                # -- News section (below table, no tab) --
                st.markdown(
                    '<div class="sec-head" style="margin-top:20px;">News and Alerts</div>',
                    unsafe_allow_html=True,
                )
                _render_news_tab(roster)
