"""My Team — Roster overview and category standings."""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, init_db, load_player_pool
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.ui_shared import (
    PAGE_ICONS,
    THEME,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_data_freshness_card,
    render_player_select,
    sort_roster_for_display,
)
from src.yahoo_data_service import get_yahoo_data_service

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


def _compute_category_totals(df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute hitting and pitching category totals from a DataFrame with is_hitter column.

    Returns (hit_stats, pitch_stats) dicts with display-ready values.
    """
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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    hitters = df[df["is_hitter"] == 1]
    pitchers = df[df["is_hitter"] == 0]

    hit_stats: dict = {}
    if not hitters.empty:
        for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
        ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
        h = hitters["h"].sum() if "h" in hitters.columns else 0
        hit_stats["AVG"] = f"{h / ab:.3f}" if ab > 0 else ".000"
        hit_bb = hitters["bb"].sum() if "bb" in hitters.columns else 0
        hit_hbp = hitters["hbp"].sum() if "hbp" in hitters.columns else 0
        hit_sf = hitters["sf"].sum() if "sf" in hitters.columns else 0
        obp_denom = ab + hit_bb + hit_hbp + hit_sf
        hit_stats["OBP"] = f"{(h + hit_bb + hit_hbp) / obp_denom:.3f}" if obp_denom > 0 else ".000"

    pitch_stats: dict = {}
    if not pitchers.empty:
        for cat, col in [("W", "w"), ("L", "l"), ("SV", "sv"), ("K", "k")]:
            pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
        ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
        er = pitchers["er"].sum() if "er" in pitchers.columns else 0
        bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
        ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
        pitch_stats["ERA"] = f"{er * 9 / ip:.2f}" if ip > 0 else "0.00"
        pitch_stats["WHIP"] = f"{(bb + ha) / ip:.2f}" if ip > 0 else "0.00"

    return hit_stats, pitch_stats


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

    headline = _ht.escape(news_item.get("headline") or "No headline")
    detail = news_item.get("detail") or ""
    source = news_item.get("source") or ""
    news_type = news_item.get("news_type") or "general"
    sentiment = news_item.get("sentiment_score", 0.0)
    if sentiment is None:
        sentiment = 0.0
    il_status = _ht.escape(news_item.get("il_status") or "")
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


@st.cache_data(ttl=300)
def _check_2026_live_stats() -> bool:
    """Return True if the 2026 season has started (season_stats rows with games_played > 0)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        result = conn.execute("SELECT COUNT(*) FROM season_stats WHERE season = 2026 AND games_played > 0").fetchone()
        return bool(result and result[0] > 0)
    except Exception:
        return False
    finally:
        conn.close()


st.set_page_config(page_title="Heater | My Team", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
page_timer_start()

# Determine user team
yds = get_yahoo_data_service()
rosters = yds.get_rosters()
if rosters.empty:
    st.warning(
        "No league data loaded. Connect your Yahoo league in Connect League, "
        "or league data will load automatically on next app launch."
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

        # Yahoo sync button — uses YahooDataService (handles sync internally)
        with btn2:
            if st.button("Sync Yahoo", key="sync_yahoo_roster"):
                with st.spinner("Syncing league data..."):
                    yds.force_refresh_all()
                st.rerun()

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
                            player_injury["il_stints"].tolist() if "il_stints" in player_injury.columns else None
                        )
                        il_days = player_injury["il_days"].tolist() if "il_days" in player_injury.columns else None
                        hs = compute_health_score(gp, ga, il_stints, il_days)
                        _icon, label = get_injury_badge(hs)
                        badges.append(label)
                    else:
                        badges.append("Low Risk")
                roster["Health"] = badges

            # -- Determine stat view for context panel totals --
            # Read from session state so totals react to the segmented control
            # (which renders later in the main column).
            _season_started = _check_2026_live_stats()
            if _season_started:
                _stat_options = ["2026 Live", "2026 Projected", "2025", "2024", "2023"]
                _stat_default = "2026 Live"
            else:
                _stat_options = ["2026 Projected", "2025", "2024", "2023"]
                _stat_default = "2026 Projected"
            stat_view = st.session_state.get("roster_stat_view", _stat_default)
            if stat_view not in _stat_options:
                stat_view = _stat_default

            # -- Compute category totals based on selected stat view --
            # Coerce roster numeric columns (always needed for the table)
            _num_cols = [
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
            for c in _num_cols:
                if c in roster.columns:
                    roster[c] = pd.to_numeric(roster[c], errors="coerce").fillna(0)

            if stat_view == "2026 Projected":
                hit_stats, pitch_stats = _compute_category_totals(roster.copy())
                _totals_label = "2026 Projected"
            else:
                # Load historical/live season stats for the selected year
                _hist_year = 2026 if stat_view == "2026 Live" else int(stat_view)
                _totals_label = "2026 Live" if stat_view == "2026 Live" else str(_hist_year)
                from src.database import get_connection as _gc_totals

                _conn_t = _gc_totals()
                try:
                    _hist_df = pd.read_sql_query(
                        "SELECT * FROM season_stats WHERE season = ?",
                        _conn_t,
                        params=[_hist_year],
                    )
                    _hist_df = coerce_numeric_df(_hist_df)
                finally:
                    _conn_t.close()

                if not _hist_df.empty and "player_id" in _hist_df.columns:
                    # Merge with roster to get is_hitter and filter to roster players
                    _roster_ids = roster[["player_id", "is_hitter"]].copy()
                    _totals_df = _roster_ids.merge(_hist_df, on="player_id", how="left")
                    hit_stats, pitch_stats = _compute_category_totals(_totals_df)
                else:
                    hit_stats, pitch_stats = {}, {}

            hitters = roster[roster["is_hitter"] == 1]
            pitchers = roster[roster["is_hitter"] == 0]

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

            # Build "Last synced" timestamp to embed alongside the page title
            _lr_badge_html = ""
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _dt
                from datetime import timedelta
                from datetime import timezone as _tz

                from src.database import get_connection as _gc_refresh

                _ET = _tz(timedelta(hours=-4))
                _refresh_conn = _gc_refresh()
                try:
                    _last_refresh = pd.read_sql_query("SELECT MAX(last_refresh) AS lr FROM refresh_log", _refresh_conn)
                    _lr_val = _last_refresh["lr"].iloc[0] if not _last_refresh.empty else None
                    if _lr_val:
                        _lr_dt = _dt.fromisoformat(str(_lr_val).replace("Z", "+00:00"))
                        if _lr_dt.tzinfo is None:
                            _lr_dt = _lr_dt.replace(tzinfo=_UTC)
                        _lr_et = _lr_dt.astimezone(_ET)
                        _lr_hour = _lr_et.hour % 12 or 12
                        _lr_ampm = "AM" if _lr_et.hour < 12 else "PM"
                        _lr_str = f"{_lr_et.strftime('%b %d')}, {_lr_hour}:{_lr_et.strftime('%M')} {_lr_ampm} ET"
                        _lr_badge_html = (
                            f'<span style="display:inline-block;vertical-align:middle;'
                            f"margin-left:12px;padding:6px 16px;border-radius:50px;"
                            f"background:{T['card']};border:2px solid {T['green']};"
                            f"font-family:IBM Plex Mono,monospace;font-size:12px;font-weight:600;"
                            f"color:{T['green']};letter-spacing:0.5px;font-style:normal;"
                            f'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                            f"Last synced: {_lr_str}</span>"
                        )
                finally:
                    _refresh_conn.close()
            except Exception:
                pass  # Non-fatal

            # Render page title with inline sync badge
            from src.ui_shared import PAGE_ICONS as _PI

            _my_team_icon = _PI.get("my_team", "")
            st.markdown(
                f'<div style="text-align:center !important;margin-bottom:8px !important;">'
                f'<div class="page-title" style="display:inline-block !important;">'
                f"{_my_team_icon} MY TEAM</div>"
                f"{_lr_badge_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if banner_teaser:
                from src.ui_shared import render_reco_banner

                render_reco_banner(banner_teaser, "", "my_team")
            from src.ui_shared import render_matchup_ticker

            render_matchup_ticker()

            # ── AVIS Alerts ──────────────────────────────────────────────
            try:
                from src.opponent_intel import get_current_opponent

                opp = get_current_opponent(yds=yds)
                # Merge AVIS hardcoded profile when Yahoo lacks tier/strengths/weaknesses
                if opp and (opp.get("tier", 3) == 3 and not opp.get("strengths") and not opp.get("weaknesses")):
                    _avis_banner = get_current_opponent()  # AVIS-only fallback
                    if _avis_banner:
                        opp["tier"] = _avis_banner.get("tier", opp.get("tier", 3))
                        opp["threat"] = _avis_banner.get("threat", opp.get("threat", "Unknown"))
                        opp["strengths"] = _avis_banner.get("strengths", [])
                        opp["weaknesses"] = _avis_banner.get("weaknesses", [])
                if opp:
                    tier_colors = {1: T["danger"], 2: T["warn"], 3: T["sky"], 4: T["green"]}
                    tier_color = tier_colors.get(opp["tier"], T["tx2"])
                    st.markdown(
                        f'<div style="background:{T["card"]};border-left:4px solid {tier_color};'
                        f"padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:13px;"
                        f'font-family:IBM Plex Mono,monospace;">'
                        f"<b>Week {opp['week']}</b> vs <b>{opp['name']}</b> "
                        f"(Tier {opp['tier']} — {opp['threat']} threat) "
                        f"| Strengths: {', '.join(opp.get('strengths', []))} "
                        f"| Weaknesses: {', '.join(opp.get('weaknesses', []))}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass  # Non-fatal

            # AVIS Rule #1: IP Watch
            try:
                from src.ip_tracker import compute_weekly_ip_projection, get_days_remaining_in_week

                pitcher_data = []
                for _, p in roster.iterrows():
                    if p.get("is_hitter") == 0 or any(
                        pos.strip() in ("P", "SP", "RP") for pos in str(p.get("positions", "")).upper().split(",")
                    ):
                        pitcher_data.append({"name": p.get("name", ""), "ip": p.get("ip", 0)})
                if pitcher_data:
                    ip_result = compute_weekly_ip_projection(pitcher_data, get_days_remaining_in_week())
                    ip_color = {"safe": T["green"], "warning": T["warn"], "danger": T["danger"]}
                    st.markdown(
                        f'<div style="background:{T["card"]};border-left:4px solid '
                        f"{ip_color.get(ip_result['status'], T['tx2'])};"
                        f"padding:6px 12px;border-radius:6px;margin-bottom:8px;font-size:12px;"
                        f'font-family:IBM Plex Mono,monospace;">'
                        f"IP Watch: {ip_result['projected_ip']} / {ip_result['ip_needed']:.0f} "
                        f"({ip_result['ip_pace']:.0f}% pace) — {ip_result['message']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass  # Non-fatal

            # AVIS Rule #2: Closer count
            # Check actual SV OR projected SV (early-season closers may have <5 actual saves)
            try:
                closer_count = 0
                proj_sv_map = {}
                try:
                    _conn2 = get_connection()
                    try:
                        roster_ids = roster["player_id"].dropna().astype(int).tolist()
                        if roster_ids:
                            placeholders = ",".join("?" * len(roster_ids))
                            proj_df = pd.read_sql_query(
                                f"SELECT player_id, sv FROM projections WHERE system='blended' AND player_id IN ({placeholders})",
                                _conn2,
                                params=roster_ids,
                            )
                            proj_sv_map = dict(zip(proj_df["player_id"], proj_df["sv"]))
                    finally:
                        _conn2.close()
                except Exception:
                    pass

                for _, p in roster.iterrows():
                    actual_sv = float(p.get("sv", 0) or 0)
                    pid = p.get("player_id")
                    proj_sv = float(proj_sv_map.get(pid, 0) or 0)
                    if actual_sv >= 5 or proj_sv >= 5:
                        closer_count += 1
                if closer_count < 2:
                    st.warning(f"Closer Alert: Only {closer_count} closer(s) rostered. AVIS requires minimum 2.")
            except Exception:
                pass

            # Proactive Alerts (AVIS Section 6)
            try:
                from src.alerts import generate_roster_alerts, render_alerts_html

                news_df = pd.DataFrame()
                try:
                    _conn = get_connection()
                    try:
                        news_df = pd.read_sql_query(
                            "SELECT * FROM player_news ORDER BY fetched_at DESC LIMIT 20", _conn
                        )
                    finally:
                        _conn.close()
                except Exception:
                    pass

                # Fetch recent transactions for league trade monitoring
                _txn_df = pd.DataFrame()
                _user_team = ""
                try:
                    from src.yahoo_data_service import get_yahoo_data_service

                    _yds = get_yahoo_data_service()
                    if _yds and _yds.is_connected():
                        _txn_df = _yds.get_transactions()
                        _user_team = st.session_state.get("user_team_name", "")
                except Exception:
                    pass

                alerts = generate_roster_alerts(
                    roster=roster,
                    player_news=news_df if not news_df.empty else None,
                    max_roster_size=23,
                    transactions=_txn_df if not _txn_df.empty else None,
                    user_team_name=_user_team,
                )
                if alerts:
                    alerts_html = render_alerts_html(alerts, T)
                    if alerts_html:
                        st.markdown(alerts_html, unsafe_allow_html=True)
            except Exception:
                pass  # Non-fatal

            # AVIS Section 5: Weekly Report (always visible, auto-expanded on Mondays)
            try:
                from datetime import UTC, datetime

                from src.weekly_report import generate_monday_report, generate_thursday_checkpoint

                today_dow = datetime.now(UTC).weekday()  # 0=Mon
                is_monday = today_dow == 0

                # Determine current week and opponent
                _report_opp = None
                _report_week = 1
                try:
                    from src.opponent_intel import get_current_opponent, get_week_number

                    _report_opp = get_current_opponent(yds=yds)
                    _report_week = get_week_number()

                    # If live Yahoo profile lacks strengths/weaknesses, merge from AVIS hardcoded profile
                    if _report_opp and (not _report_opp.get("strengths") and not _report_opp.get("weaknesses")):
                        _avis_opp = get_current_opponent()
                        if _avis_opp:
                            if _avis_opp.get("strengths"):
                                _report_opp["strengths"] = _avis_opp["strengths"]
                            if _avis_opp.get("weaknesses"):
                                _report_opp["weaknesses"] = _avis_opp["weaknesses"]
                except Exception:
                    pass

                _expander_label = f"Weekly Report — Week {_report_week}"
                if _report_opp:
                    _expander_label += f" vs {_report_opp.get('name', 'Unknown')}"

                # Force-regeneration button state
                _force_report = st.session_state.get("_force_weekly_report", False)
                if _force_report:
                    st.session_state["_force_weekly_report"] = False

                with st.expander(_expander_label, expanded=is_monday or _force_report):
                    # "Generate Report" button for on-demand regeneration
                    if st.button("Generate Report", key="gen_weekly_report"):
                        st.session_state["_force_weekly_report"] = True
                        st.rerun()

                    _report_rendered = False

                    # --- Monday Matchup Report ---
                    if _report_opp:
                        try:
                            monday = generate_monday_report(roster, None, _report_opp, _report_week)
                            _report_rendered = True

                            # Opponent summary card
                            _opp_tier = monday.get("tier", 3)
                            _tier_colors = {
                                1: T["danger"],
                                2: T["warn"],
                                3: T["sky"],
                                4: T["green"],
                            }
                            _opp_color = _tier_colors.get(_opp_tier, T["tx2"])
                            _opp_html = (
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                f'<span style="color:{T["tx2"]};">Opponent</span>'
                                f'<span style="font-weight:600;color:{T["tx"]};">'
                                f"{monday.get('opponent', 'Unknown')}</span></div>"
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                f'<span style="color:{T["tx2"]};">Tier</span>'
                                f'<span style="font-weight:600;color:{_opp_color};">'
                                f"{_opp_tier}</span></div>"
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                f'<span style="color:{T["tx2"]};">Threat</span>'
                                f'<span style="font-weight:600;color:{T["tx"]};">'
                                f"{monday.get('threat', 'Unknown')}</span></div>"
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                f'<span style="color:{T["tx2"]};">Manager</span>'
                                f'<span style="font-weight:600;color:{T["tx"]};">'
                                f"{monday.get('manager', 'Unknown')}</span></div>"
                            )
                            _opp_strengths = monday.get("opponent_strengths", [])
                            _opp_weaknesses = monday.get("opponent_weaknesses", [])
                            if _opp_strengths:
                                _opp_html += (
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                    f'<span style="color:{T["tx2"]};">Strengths</span>'
                                    f'<span style="font-weight:600;color:{T["danger"]};">'
                                    f"{', '.join(_opp_strengths)}</span></div>"
                                )
                            if _opp_weaknesses:
                                _opp_html += (
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                    f'<span style="color:{T["tx2"]};">Weaknesses</span>'
                                    f'<span style="font-weight:600;color:{T["green"]};">'
                                    f"{', '.join(_opp_weaknesses)}</span></div>"
                                )
                            render_context_card("This Week Opponent", _opp_html)

                            # Category projections card
                            cat_projs = monday.get("category_projections", [])
                            if cat_projs:
                                _cat_html = ""
                                for cp in cat_projs:
                                    _outlook = cp.get("outlook", "TOSS-UP")
                                    if _outlook == "LIKELY WIN":
                                        _out_color = T["green"]
                                    elif _outlook == "LIKELY LOSS":
                                        _out_color = T["danger"]
                                    else:
                                        _out_color = T["warn"]
                                    _cat_html += (
                                        f'<div style="display:flex;justify-content:space-between;'
                                        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                                        f'<span style="color:{T["tx2"]};">{cp["category"]}</span>'
                                        f'<span style="font-weight:600;color:{_out_color};">'
                                        f"{_outlook}</span></div>"
                                    )
                                render_context_card("Category Projections", _cat_html)

                            # Action items card
                            _actions = []
                            for tip in monday.get("exploit_weaknesses", []):
                                _actions.append(tip)
                            for tip in monday.get("protect_floor", []):
                                _actions.append(tip)
                            if _actions:
                                _action_html = ""
                                for i, action in enumerate(_actions, 1):
                                    _action_html += (
                                        f'<div style="padding:3px 0;font-size:12px;'
                                        f'font-family:Figtree,sans-serif;color:{T["tx"]};">'
                                        f'<span style="font-weight:700;color:{T["primary"]};">'
                                        f"{i}.</span> {action}</div>"
                                    )
                                render_context_card("Action Items", _action_html)

                            # Streaming targets card
                            streaming = monday.get("streaming_guidance", [])
                            if streaming:
                                _stream_html = ""
                                for sg in streaming:
                                    _stream_html += (
                                        f'<div style="padding:3px 0;font-size:12px;'
                                        f'font-family:Figtree,sans-serif;color:{T["tx"]};">'
                                        f"{sg}</div>"
                                    )
                                render_context_card("Streaming Targets", _stream_html)

                        except Exception:
                            pass  # Fall through to fallback

                    # --- Thursday Checkpoint (inline if Thursday) ---
                    if today_dow == 3:
                        try:
                            ip_proj = 0.0
                            try:
                                from src.ip_tracker import (
                                    compute_weekly_ip_projection,
                                    get_days_remaining_in_week,
                                )

                                pitcher_data_thu = []
                                for _, p in roster.iterrows():
                                    if p.get("is_hitter") == 0 or any(
                                        pos.strip() in ("P", "SP", "RP")
                                        for pos in str(p.get("positions", "")).upper().split(",")
                                    ):
                                        pitcher_data_thu.append({"name": p.get("name", ""), "ip": p.get("ip", 0)})
                                if pitcher_data_thu:
                                    ip_res = compute_weekly_ip_projection(
                                        pitcher_data_thu, get_days_remaining_in_week()
                                    )
                                    ip_proj = ip_res.get("projected_ip", 0)
                            except Exception:
                                pass

                            checkpoint = generate_thursday_checkpoint(roster, matchup_score=None, ip_projected=ip_proj)
                            _chk_html = (
                                f'<div style="padding:3px 0;font-size:12px;'
                                f'font-family:IBM Plex Mono,monospace;color:{T["tx"]};">'
                                f"{checkpoint['ip_status']}</div>"
                            )
                            if checkpoint["categories_at_risk"]:
                                _chk_html += (
                                    f'<div style="padding:3px 0;font-size:12px;'
                                    f'font-family:Figtree,sans-serif;color:{T["warn"]};">'
                                    f"At-Risk: {', '.join(checkpoint['categories_at_risk'])}</div>"
                                )
                            for rec in checkpoint["recommendations"]:
                                _chk_html += (
                                    f'<div style="padding:3px 0;font-size:12px;'
                                    f'font-family:Figtree,sans-serif;color:{T["tx"]};">'
                                    f"{rec}</div>"
                                )
                            if not checkpoint["recommendations"]:
                                _chk_html += (
                                    f'<div style="padding:3px 0;font-size:12px;'
                                    f'font-family:Figtree,sans-serif;color:{T["green"]};">'
                                    f"No adjustments needed — stay the course.</div>"
                                )
                            render_context_card("Thursday Mid-Week Checkpoint", _chk_html)
                            _report_rendered = True
                        except Exception:
                            pass

                    # --- Fallback report when full generator fails or no opponent data ---
                    if not _report_rendered:
                        _fb_parts = []
                        _fb_parts.append(
                            f'<div style="padding:3px 0;font-size:12px;'
                            f'font-family:IBM Plex Mono,monospace;color:{T["tx"]};">'
                            f'<span style="font-weight:700;">Week {_report_week} Report</span></div>'
                        )
                        # Opponent line
                        _fb_opp_name = _report_opp.get("name", "Unknown") if _report_opp else "Unknown"
                        _fb_parts.append(
                            f'<div style="padding:2px 0;font-size:12px;'
                            f'font-family:IBM Plex Mono,monospace;color:{T["tx2"]};">'
                            f"Opponent: {_fb_opp_name}</div>"
                        )
                        # Roster summary
                        n_il = sum(
                            1
                            for _, r in roster.iterrows()
                            if str(r.get("roster_slot", "")).upper() in ("IL", "IL+", "NA")
                        )
                        _fb_parts.append(
                            f'<div style="padding:2px 0;font-size:12px;'
                            f'font-family:IBM Plex Mono,monospace;color:{T["tx2"]};">'
                            f"Roster: {len(roster)} players, {n_il} on IL</div>"
                        )
                        # Category totals summary
                        if hit_stats:
                            _fb_hit = ", ".join(f"{k}: {v}" for k, v in hit_stats.items())
                            _fb_parts.append(
                                f'<div style="padding:2px 0;font-size:12px;'
                                f'font-family:IBM Plex Mono,monospace;color:{T["tx2"]};">'
                                f"Hitting: {_fb_hit}</div>"
                            )
                        if pitch_stats:
                            _fb_pit = ", ".join(f"{k}: {v}" for k, v in pitch_stats.items())
                            _fb_parts.append(
                                f'<div style="padding:2px 0;font-size:12px;'
                                f'font-family:IBM Plex Mono,monospace;color:{T["tx2"]};">'
                                f"Pitching: {_fb_pit}</div>"
                            )
                        render_context_card("Weekly Summary", "".join(_fb_parts))

            except Exception:
                pass  # Non-fatal

            # Daily Lineup Validation (AVIS Section 5)
            try:
                from src.weekly_report import get_todays_mlb_games, validate_daily_lineup

                todays_teams = get_todays_mlb_games()
                if todays_teams:
                    lineup_issues = validate_daily_lineup(roster, todays_teams)
                    # Store for context panel card (rendered later)
                    st.session_state["_lineup_issues"] = lineup_issues
                    st.session_state["_lineup_teams_playing"] = len(todays_teams)
                else:
                    st.session_state["_lineup_issues"] = []
                    st.session_state["_lineup_teams_playing"] = 0
            except Exception:
                st.session_state["_lineup_issues"] = []
                st.session_state["_lineup_teams_playing"] = 0

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
                    render_context_card(f"Hitting Totals — {_totals_label}", hit_html)

                # Category totals card — Pitching
                if pitch_stats:
                    pitch_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:2px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">'
                        f'<span style="color:{T["tx2"]};">{k}</span>'
                        f'<span style="font-weight:600;color:{T["tx"]};">{v}</span></div>'
                        for k, v in pitch_stats.items()
                    )
                    render_context_card(f"Pitching Totals — {_totals_label}", pitch_html)

                # Category Gaps card — compare team totals to league or H2H benchmarks
                if hit_stats or pitch_stats:
                    # Default H2H benchmarks (12-team, full-season averages)
                    _BENCHMARKS: dict[str, float] = {
                        "R": 800.0,
                        "HR": 250.0,
                        "RBI": 800.0,
                        "SB": 100.0,
                        "AVG": 0.260,
                        "OBP": 0.330,
                        "W": 85.0,
                        "SV": 80.0,
                        "K": 1200.0,
                        "ERA": 3.80,
                        "WHIP": 1.22,
                    }
                    # Inverse categories: lower is better
                    _INVERSE_CATS: set[str] = {"ERA", "WHIP"}

                    # Attempt to derive per-team league averages from league_standings
                    try:
                        _ls = yds.get_standings()
                        if not _ls.empty and "category" in _ls.columns and "total" in _ls.columns:
                            # Compute mean total per category across all teams
                            _ls_avg = _ls.groupby("category")["total"].mean().to_dict()
                            # Normalise category keys to uppercase
                            _ls_avg = {str(k).upper(): v for k, v in _ls_avg.items()}
                            # Override defaults only for categories present in standings
                            for _cat, _val in _ls_avg.items():
                                if _cat in _BENCHMARKS and _val and _val > 0:
                                    _BENCHMARKS[_cat] = float(_val)
                    except Exception:
                        pass  # Fall back to hardcoded benchmarks silently

                    # Merge hit_stats and pitch_stats into one lookup; cast rate stats to float
                    _team_totals: dict[str, float] = {}
                    for _cat, _val in {**hit_stats, **pitch_stats}.items():
                        try:
                            _team_totals[_cat] = float(str(_val))
                        except (ValueError, TypeError):
                            pass

                    # Compute gap and direction for each benchmarked category
                    _gap_rows: list[dict] = []
                    for _cat, _bench in _BENCHMARKS.items():
                        if _cat not in _team_totals:
                            continue
                        _team_val = _team_totals[_cat]
                        _is_inverse = _cat in _INVERSE_CATS
                        # "above" means better: for inverse cats, lower team value is better
                        _above = (_team_val < _bench) if _is_inverse else (_team_val >= _bench)
                        if _is_inverse and _bench != 0:
                            # Percentage gap: negative means below (worse for inverse)
                            _pct = (_bench - _team_val) / _bench * 100.0
                        elif _bench != 0:
                            _pct = (_team_val - _bench) / _bench * 100.0
                        else:
                            _pct = 0.0
                        _gap_rows.append(
                            {
                                "cat": _cat,
                                "team_val": _team_val,
                                "bench": _bench,
                                "above": _above,
                                "pct": _pct,
                            }
                        )

                    if _gap_rows:
                        # Identify 2 weakest categories (lowest pct gap, i.e. most below bench)
                        _sorted_by_gap = sorted(_gap_rows, key=lambda x: x["pct"])
                        _priority_cats = {r["cat"] for r in _sorted_by_gap[:2]}

                        # Build priority targets block
                        _priority_names = [r["cat"] for r in _sorted_by_gap[:2]]
                        _priority_html = (
                            f'<div style="margin-bottom:8px;padding:6px 8px;'
                            f"background:rgba(230,57,70,0.07);border-radius:6px;"
                            f'border-left:3px solid {T["danger"]};">'
                            f'<div style="font-size:10px;font-weight:700;letter-spacing:0.8px;'
                            f'text-transform:uppercase;color:{T["danger"]};margin-bottom:3px;">'
                            f"Priority Targets</div>"
                            f'<div style="font-size:12px;font-weight:600;'
                            f'font-family:IBM Plex Mono,monospace;color:{T["tx"]};">'
                            f"{' · '.join(_priority_names)}</div>"
                            f"</div>"
                        )

                        # Build per-category rows
                        _rows_html = ""
                        for _row in _gap_rows:
                            _cat = _row["cat"]
                            _above = _row["above"]
                            _pct = _row["pct"]
                            _team_val = _row["team_val"]
                            _bench = _row["bench"]
                            _is_priority = _cat in _priority_cats
                            _color = T["green"] if _above else T["danger"]
                            _direction = "+" if _pct >= 0 else ""
                            # Format values sensibly
                            if _cat in ("AVG", "OBP"):
                                _tv_str = f"{_team_val:.3f}"
                                _bv_str = f"{_bench:.3f}"
                            elif _cat in ("ERA", "WHIP"):
                                _tv_str = f"{_team_val:.2f}"
                                _bv_str = f"{_bench:.2f}"
                            else:
                                _tv_str = f"{int(_team_val)}"
                                _bv_str = f"{int(_bench)}"

                            _label_weight = "700" if _is_priority else "400"
                            _rows_html += (
                                f'<div style="display:flex;justify-content:space-between;'
                                f"align-items:center;padding:3px 0;font-size:12px;"
                                f"font-family:IBM Plex Mono,monospace;"
                                f'border-bottom:1px solid {T["border"]};">'
                                f'<span style="font-weight:{_label_weight};color:{T["tx2"]};">'
                                f"{_cat}</span>"
                                f'<span style="display:flex;gap:6px;align-items:center;">'
                                f'<span style="color:{T["tx2"]};font-size:11px;">'
                                f"{_tv_str} / {_bv_str}</span>"
                                f'<span style="font-weight:700;color:{_color};min-width:40px;'
                                f'text-align:right;">'
                                f"{_direction}{_pct:.1f}%</span>"
                                f"</span></div>"
                            )

                        _gaps_html = _priority_html + _rows_html
                        render_context_card("Category Gaps", _gaps_html)

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

                # Lineup Validation card
                _lineup_issues = st.session_state.get("_lineup_issues", [])
                _teams_playing = st.session_state.get("_lineup_teams_playing", 0)
                _warnings = [li for li in _lineup_issues if li["severity"] == "warning"]
                _benched = [li for li in _lineup_issues if li["severity"] == "info"]
                _issue_count = len(_warnings)

                if _teams_playing > 0:
                    # Count badge
                    if _issue_count == 0:
                        _badge_color = T["green"]
                        _badge_text = "Lineup clean — no issues"
                    else:
                        _badge_color = T["danger"]
                        _badge_text = f"{_issue_count} lineup issue{'s' if _issue_count != 1 else ''} found"
                    _lv_html = (
                        f'<div style="display:inline-block;padding:2px 10px;border-radius:10px;'
                        f"background:{_badge_color};color:#ffffff;font-size:11px;font-weight:700;"
                        f'letter-spacing:0.5px;margin-bottom:8px;">{_badge_text}</div>'
                    )

                    # Off-day starter warnings
                    for _w in _warnings:
                        _repl_html = ""
                        if _w.get("replacements"):
                            _repl_names = ", ".join(_w["replacements"])
                            _repl_html = (
                                f'<div style="margin-top:2px;font-size:11px;color:{T["green"]};">'
                                f"Swap in: {_repl_names}</div>"
                            )
                        elif _w["severity"] == "warning":
                            _repl_html = (
                                f'<div style="margin-top:2px;font-size:11px;color:{T["tx2"]};">'
                                f"No eligible bench replacement playing today</div>"
                            )
                        _lv_html += (
                            f'<div style="padding:4px 0;border-bottom:1px solid {T["border"]};">'
                            f'<div style="font-size:12px;">'
                            f'<span class="health-dot" style="background:{T["warn"]};"></span>'
                            f'<span style="font-weight:600;">{_w["player"]}</span></div>'
                            f'<div style="font-size:11px;color:{T["tx2"]};">'
                            f"{_w['issue']}</div>"
                            f"{_repl_html}</div>"
                        )

                    # Benched players with games (info-level)
                    if _benched:
                        _lv_html += (
                            f'<div style="margin-top:6px;font-size:10px;font-weight:700;'
                            f"letter-spacing:0.8px;text-transform:uppercase;"
                            f'color:{T["sky"]};margin-bottom:3px;">Bench — Playing Today</div>'
                        )
                        for _b in _benched:
                            _lv_html += (
                                f'<div style="padding:2px 0;font-size:12px;">'
                                f'<span class="health-dot" style="background:{T["sky"]};"></span>'
                                f'<span style="font-weight:600;">{_b["player"]}</span> '
                                f'<span style="color:{T["tx2"]};font-size:11px;">'
                                f"available to start</span></div>"
                            )

                    render_context_card(f"Lineup Validation ({_teams_playing} teams playing)", _lv_html)

                # Live matchup score card
                matchup = yds.get_matchup()
                if matchup and isinstance(matchup, dict):
                    _mw = matchup.get("week", "?")
                    _mopp = matchup.get("opp_name", "Unknown")
                    # Yahoo returns pre-computed wins/losses/ties + per-category list
                    _wins = int(matchup.get("wins", 0))
                    _losses = int(matchup.get("losses", 0))
                    _ties = int(matchup.get("ties", 0))
                    _cats = matchup.get("categories", [])

                    _match_rows = ""
                    for _cdict in _cats:
                        _cat = _cdict.get("cat", "")
                        _yv_str = str(_cdict.get("you", "-"))
                        _ov_str = str(_cdict.get("opp", "-"))
                        _result = _cdict.get("result", "-")

                        # Color based on result
                        if _result == "WIN":
                            _color = T["green"]
                        elif _result == "LOSS":
                            _color = T["danger"]
                        else:
                            _color = T["tx2"]

                        # Format numbers
                        try:
                            _yv = float(_yv_str) if _yv_str not in ("-", "") else None
                            _ov = float(_ov_str) if _ov_str not in ("-", "") else None
                        except (ValueError, TypeError):
                            _yv = _ov = None

                        if _yv is not None and _ov is not None:
                            if _cat in ("AVG", "OBP"):
                                _fmt = ".3f"
                            elif _cat in ("ERA", "WHIP"):
                                _fmt = ".2f"
                            else:
                                _fmt = ".0f"
                            _yv_display = f"{_yv:{_fmt}}"
                            _ov_display = f"{_ov:{_fmt}}"
                        else:
                            _yv_display = _yv_str
                            _ov_display = _ov_str

                        _match_rows += (
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:1px 0;font-size:11px!important;font-family:IBM Plex Mono,monospace!important">'
                            f'<span style="color:{_color}!important;font-weight:600!important">'
                            f"{_yv_display}</span>"
                            f'<span style="color:{T["tx2"]}!important">{_cat}</span>'
                            f'<span style="color:{T["tx2"]}!important">{_ov_display}</span></div>'
                        )

                    _header = (
                        f'<div style="text-align:center;font-size:10px!important;'
                        f'color:{T["tx2"]}!important;margin-bottom:4px!important">'
                        f"Week {_mw} vs {_mopp}</div>"
                        f'<div style="text-align:center;font-size:16px!important;'
                        f'font-weight:700!important;margin-bottom:6px!important">'
                        f'<span style="color:{T["green"]}!important">{_wins}</span>'
                        f' - <span style="color:{T["danger"]}!important">{_losses}</span>'
                        f' - <span style="color:{T["tx2"]}!important">{_ties}</span></div>'
                    )
                    render_context_card("Live Matchup", _header + _match_rows)

                # Data freshness card
                render_data_freshness_card()

            # -- Main Content (right) --
            with main:
                # Season/projection toggle (options computed earlier for context panel)
                stat_view = st.segmented_control(
                    "View stats from",
                    options=_stat_options,
                    default=_stat_default,
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

                if stat_view in ("2026 Live", "2025", "2024", "2023"):
                    # Load actual stats for the selected season (including live 2026)
                    season_year = 2026 if stat_view == "2026 Live" else int(stat_view)
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

                    _roster_cols = ["name", "positions", "roster_slot", "player_id"]
                    if "mlb_id" in roster.columns:
                        _roster_cols.append("mlb_id")
                    display_df = roster[_roster_cols].copy()
                    if not hist.empty and "player_id" in hist.columns:
                        hist_stat_cols = [c.lower() for c in stat_cols_ordered if c.lower() in hist.columns]
                        hist_slim = hist[["player_id"] + hist_stat_cols].copy()
                        display_df = display_df.merge(hist_slim, on="player_id", how="left")

                    if "Health" in roster.columns:
                        display_df["Health"] = roster["Health"].values

                    display_df.rename(
                        columns={k: v for k, v in rename_map.items() if k in display_df.columns},
                        inplace=True,
                    )

                    if stat_view == "2026 Live":
                        caption = "Live 2026 season stats. Updates hourly from MLB Stats API."
                        if hist.empty:
                            caption = "No 2026 season stats available yet."
                        section_label = "Roster — 2026 Live Stats"
                    else:
                        caption = f"Actual stats from the {season_year} MLB season."
                        if hist.empty:
                            caption = f"No historical data available for {season_year}."
                        section_label = f"Roster — {season_year} Season Stats"

                    st.markdown(
                        f'<div class="sec-head">{section_label}</div>'
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
                    # Include mlb_id for headshot rendering (auto-hidden by table)
                    if "mlb_id" in roster.columns:
                        display_cols.append("mlb_id")
                    # Include player_id temporarily for sort-stable extraction
                    if "player_id" in roster.columns and "player_id" not in display_cols:
                        display_cols.append("player_id")
                    available_cols = [c for c in display_cols if c in roster.columns]
                    display_df = roster[available_cols].copy()
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

                # Sort roster into Yahoo slot order for display
                display_df = sort_roster_for_display(display_df)

                # Extract player IDs after sorting so order matches display
                player_ids_list = display_df["player_id"].tolist() if "player_id" in display_df.columns else []
                if "player_id" in display_df.columns:
                    display_df = display_df.drop(columns=["player_id"])

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
                                    0, text="Loading preseason projections for Marcel stabilization..."
                                )
                                preseason = pd.read_sql_query(
                                    "SELECT * FROM projections WHERE system = 'blended'", conn
                                )
                                preseason = coerce_numeric_df(preseason)
                                bayes_progress.progress(
                                    30,
                                    text="Applying Marcel regression with stabilization thresholds...",
                                )
                                updater = BayesianUpdater()
                                updated = updater.batch_update_projections(season_stats, preseason)
                                bayes_progress.progress(100, text="Marcel-adjusted projections complete!")
                                time.sleep(0.3)
                                bayes_progress.empty()
                                st.markdown(
                                    '<div class="sec-head">Marcel-Adjusted Projections</div>',
                                    unsafe_allow_html=True,
                                )
                                stat_display = ["player_id", "avg", "hr", "rbi", "sb", "era", "whip", "k"]
                                show_cols = [c for c in stat_display if c in updated.columns]
                                bayes_df = updated[show_cols].copy()
                                # Replace player_id with player name + add mlb_id for headshots
                                if "player_id" in bayes_df.columns:
                                    players_lookup = pd.read_sql_query(
                                        "SELECT player_id, name, mlb_id FROM players", conn
                                    )
                                    pid_to_name = dict(zip(players_lookup["player_id"], players_lookup["name"]))
                                    pid_to_mlb = dict(zip(players_lookup["player_id"], players_lookup["mlb_id"]))
                                    bayes_df["mlb_id"] = bayes_df["player_id"].map(
                                        lambda x: pid_to_mlb.get(int(x)) if pd.notna(x) else None
                                    )
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
                                for c in ["AVG"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.3f}")
                                for c in ["ERA", "WHIP"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.2f}")
                                for c in ["HR", "RBI", "SB", "K", "ID"]:
                                    if c in bayes_df.columns:
                                        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.2f}")
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

page_timer_footer("My Team")
