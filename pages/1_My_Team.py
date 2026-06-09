"""My Team — Roster overview and category standings."""

import time

import pandas as pd
import streamlit as st

from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name, viewer_can_write
from src.database import coerce_numeric_df, init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.optimizer.h2h_engine import default_weekly_sigmas
from src.ui_shared import (
    PAGE_ICONS,
    THEME,
    build_heatbar_html,
    build_panel_html,
    build_roster_table_html,
    build_stat_readout_html,
    format_stat,
    inject_custom_css,
    no_league_data_message,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_data_freshness_card,
    render_page_header,
    render_player_select,
    show_player_card_dialog,
    sort_roster_for_display,
)
from src.usage import log_page_view
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

_LC = LeagueConfig()

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
        hit_stats["AVG"] = format_stat(h / ab, "AVG") if ab > 0 else ".000"
        hit_bb = hitters["bb"].sum() if "bb" in hitters.columns else 0
        hit_hbp = hitters["hbp"].sum() if "hbp" in hitters.columns else 0
        hit_sf = hitters["sf"].sum() if "sf" in hitters.columns else 0
        obp_denom = ab + hit_bb + hit_hbp + hit_sf
        hit_stats["OBP"] = format_stat((h + hit_bb + hit_hbp) / obp_denom, "OBP") if obp_denom > 0 else ".000"

    pitch_stats: dict = {}
    if not pitchers.empty:
        for cat, col in [("W", "w"), ("L", "l"), ("SV", "sv"), ("K", "k")]:
            pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
        ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
        er = pitchers["er"].sum() if "er" in pitchers.columns else 0
        bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
        ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
        pitch_stats["ERA"] = format_stat(er * 9 / ip, "ERA") if ip > 0 else "0.00"
        pitch_stats["WHIP"] = format_stat((bb + ha) / ip, "WHIP") if ip > 0 else "0.00"

    return hit_stats, pitch_stats


def _rank_priority_losing_cats(
    gap_rows: list[dict],
    sigmas: dict[str, float],
    top_n: int = 2,
) -> list[dict]:
    """Rank the losing categories by NORMALIZED closeness-to-flip (BR-2b).

    The "Priority Targets" callout should surface the most ACTIONABLE losing
    categories — the ones closest to flipping — not the ones with the biggest
    raw gap. Sorting losing cats by raw ``diff`` mixes counting cats (R behind
    by 6) with rate cats (AVG behind by 0.069) on incompatible scales, so the
    big-count cats almost always sort first and a winnable rate cat never
    surfaces.

    Each losing cat's gap is normalized by that category's weekly standard
    deviation (the canonical ``h2h_engine.default_weekly_sigmas()``, keyed by
    uppercase cat) to a unit-free z-gap ``|diff| / sigma``. The SMALLEST
    |z-gap| losing cats are the closest to flipping = the real priority
    targets. Inverse cats use the gap MAGNITUDE (the ``diff`` is already
    oriented so positive = winning), so ``abs`` is correct for all cats. A
    category with no sigma entry is treated as an infinitely-wide gap (sorted
    last) rather than crashing.

    Args:
        gap_rows: Per-category gap rows (each carrying ``cat``, ``diff``,
            ``above``, ``tied``).
        sigmas: Uppercase-keyed per-team weekly category standard deviations.
        top_n: Number of priority targets to return.

    Returns:
        Up to ``top_n`` losing gap rows, ordered closest-to-flip first.
    """
    losing = [r for r in gap_rows if not r["above"] and not r["tied"]]

    def _z_gap(row: dict) -> float:
        sigma = sigmas.get(row["cat"])
        if not sigma or sigma <= 0:
            return float("inf")  # no scale -> treat as far from flipping
        return abs(float(row["diff"])) / float(sigma)

    return sorted(losing, key=_z_gap)[:top_n]


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
    """Return True if the 2026 season has started.

    Uses the enriched player pool's ytd_gp column instead of direct SQL so all
    consumers see the same staleness window and Statcast/regression-flag enrichments
    flow through (per Unified Services mandate).
    """
    try:
        pool = load_player_pool()
        if pool.empty or "ytd_gp" not in pool.columns:
            return False
        return bool(pd.to_numeric(pool["ytd_gp"], errors="coerce").fillna(0).gt(0).any())
    except Exception:
        return False


if not multi_user_enabled():
    st.set_page_config(page_title="Heater | My Team", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
require_auth()
require_page_enabled("page:1_My_Team")
log_page_view("My Team")
page_timer_start()

# Determine user team
yds = get_yahoo_data_service()
rosters = yds.get_rosters()
if rosters.empty:
    st.warning(no_league_data_message(yds.data_unavailable_reason()))
    st.stop()
else:
    user_team_name = resolve_viewer_team_name(rosters)
    if not user_team_name:
        st.warning("No user team identified in roster data.")
        st.stop()
    else:
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

        # Try to fetch team logo + season record/rank from Yahoo standings.
        # Record (W-L-T) and overall rank are not reliably present in the
        # SQLite long-format standings fallback, so we read them straight off
        # the live yfpy standings object when a client is connected. Read-only
        # members without a client fall back to "—" in the identity strip.
        team_logo_url = ""
        _id_record = ""
        _id_rank = None
        if yahoo_client:
            try:
                standings_data = yahoo_client._query.get_league_standings()
                _teams = getattr(standings_data, "teams", None) or []
                for _entry in _teams:
                    t = getattr(_entry, "team", _entry)
                    t_name = getattr(t, "name", "")
                    if isinstance(t_name, bytes):
                        t_name = t_name.decode("utf-8", errors="replace")
                    if str(t_name) == user_team_name:
                        logos = getattr(t, "team_logos", None) or []
                        if logos:
                            team_logo_url = getattr(logos[0], "url", "") or ""
                        # Season record + rank from team_standings.
                        _ts = getattr(t, "team_standings", None)
                        _rank_v = getattr(t, "rank", None)
                        if _ts is not None and _rank_v is None:
                            _rank_v = getattr(_ts, "rank", None)
                        try:
                            _id_rank = int(_rank_v) if _rank_v else None
                        except (TypeError, ValueError):
                            _id_rank = None
                        if _ts is not None:
                            _outcome = getattr(_ts, "outcome_totals", None)
                            if _outcome is not None:
                                try:
                                    _w = int(float(getattr(_outcome, "wins", 0) or 0))
                                    _l = int(float(getattr(_outcome, "losses", 0) or 0))
                                    _t = int(float(getattr(_outcome, "ties", 0) or 0))
                                    _id_record = f"{_w}–{_l}–{_t}"
                                except (TypeError, ValueError):
                                    _id_record = ""
                        break
            except Exception:
                pass

        # Build the Combustion identity avatar (mockup .idav — a navy rounded
        # square with initials, or the Yahoo logo when available). Rendered
        # later in the identity strip once the roster counts are known.
        import html as _html
        import re as _re

        # Strip emoji/symbols to get clean initials for the monogram fallback.
        _clean_name = _re.sub(r"[^\w\s]", "", user_team_name, flags=_re.UNICODE).strip()
        _id_words = [w for w in _clean_name.split() if w and w[0].isalpha()]
        _id_initials = "".join(w[0].upper() for w in _id_words[:2]) if _id_words else "T"
        if team_logo_url:
            avatar_html = (
                f'<div class="idav" style="width:54px;height:54px;min-width:54px;border-radius:13px;'
                f"overflow:hidden;box-shadow:0 4px 14px rgba(14,34,68,.28);"
                f'display:flex;align-items:center;justify-content:center;">'
                f'<img src="{team_logo_url}" style="width:54px;height:54px;object-fit:cover;" '
                f'alt="Team logo"/></div>'
            )
        else:
            avatar_html = (
                f'<div class="idav" style="width:54px;height:54px;min-width:54px;border-radius:13px;'
                f"background:radial-gradient(circle at 36% 30%,var(--fp-navy),var(--fp-navy2));"
                f"box-shadow:0 4px 14px rgba(14,34,68,.28),inset 0 0 0 1px rgba(255,255,255,.10);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-family:var(--font-display);font-weight:900;color:#eef1f6;"
                f'font-size:18px;letter-spacing:.04em;">{_id_initials}</div>'
            )

        safe_name = _html.escape(user_team_name)

        # ── Combustion page header (mockup .phead) ──
        # Replaces the old navy .page-title pill. The native Refresh/Sync
        # buttons render just below (Streamlit buttons can't live inside the
        # HTML header); a "LIVE" pill carries the freshness affordance.
        _live_pill_html = (
            '<div class="livepill" style="display:flex;align-items:center;gap:7px;'
            "font-family:var(--font-mono);font-size:11px;color:var(--fp-tx-muted);"
            'letter-spacing:.06em;">'
            '<span style="width:7px;height:7px;border-radius:50%;background:var(--fp-primary);'
            'box-shadow:0 0 10px var(--fp-primary);"></span>LIVE</div>'
        )
        render_page_header(
            "My Team",
            eyebrow="SEASON",
            fig="FIG.01 — ROSTER CONTROL",
            actions_html=_live_pill_html,
        )

        # Action buttons — inline row
        btn1, btn2, btn_spacer = st.columns([1, 1, 3])
        with btn1:
            if viewer_can_write() and st.button("Refresh Stats"):
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
            if viewer_can_write() and st.button("Sync Yahoo", key="sync_yahoo_roster"):
                with st.spinner("Syncing league data..."):
                    yds.force_refresh_all()
                st.rerun()

        # Load roster
        roster = get_team_roster(user_team_name)

        # Filter ghost players: traded-away players may linger in the DB with
        # empty/null selected_position or roster_slot.  Also deduplicate by
        # player_id and cap at league roster size (23).
        if not roster.empty:
            if "selected_position" in roster.columns:
                roster = roster[
                    roster["selected_position"].fillna("").astype(str).str.strip().ne("")
                    & roster["selected_position"].astype(str).str.lower().ne("none")
                ]
            if "player_id" in roster.columns:
                roster = roster.drop_duplicates(subset=["player_id"], keep="first")

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
            # (which renders later, in the full-width roster section below).
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
                # Load historical/live season stats for the selected year via the
                # canonical loader. The player pool's ytd_* columns lack the raw
                # counting cols (ab/h/bb/hbp/sf/ip/er/bb_allowed/h_allowed) that
                # _compute_category_totals needs for weighted AVG/OBP/ERA/WHIP, so
                # we route through load_season_stats which centralises the SQL
                # and applies coerce_numeric_df.
                from src.database import load_season_stats

                _hist_year = 2026 if stat_view == "2026 Live" else int(stat_view)
                _totals_label = "2026 Live" if stat_view == "2026 Live" else str(_hist_year)
                _hist_df = load_season_stats(_hist_year)

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

            # ── Identity strip (mockup .identity) ──
            # Navy avatar + team name (Archivo) + manager eyebrow line, with a
            # right-aligned row of stat readouts (Record / Roster / Hitters /
            # Pitchers / Rank). Record + Rank accent orange. Record + Rank come
            # from the live Yahoo standings (read above); when unavailable
            # (read-only member with no client) they show "—".
            _rank_value = f"{_id_rank}" if _id_rank else "—"
            _rank_sub = "/ 12" if _id_rank else None
            _readouts = (
                build_stat_readout_html("Record", _id_record or "—", accent=bool(_id_record))
                + build_stat_readout_html("Roster", n_total)
                + build_stat_readout_html("Hitters", n_hitters)
                + build_stat_readout_html("Pitchers", n_pitchers)
                + build_stat_readout_html("Rank", _rank_value, accent=bool(_id_rank), sub=_rank_sub)
            )
            _meta_line = (
                '<div class="fig" style="font-family:var(--font-mono);font-weight:500;font-size:10px;'
                "letter-spacing:.12em;color:var(--fp-tx-muted);margin-top:4px;"
                '">MANAGER · FOURZYNBURN · 12-TEAM H2H</div>'
            )
            # Per-stat dividers (mockup .stat border-left) — wrap each readout.
            _readout_wrap = (
                '<div class="idmeta" style="display:flex;margin-left:auto;gap:0;">'
                + _readouts.replace(
                    '<div class="stat" style="',
                    '<div class="stat" style="padding:6px 18px;border-left:1px solid var(--fp-divider);',
                )
                + "</div>"
            )
            st.markdown(
                '<div class="identity" style="display:flex;align-items:center;gap:22px;'
                'margin:18px 0 6px;flex-wrap:wrap;">'
                f"{avatar_html}"
                "<div>"
                f'<div class="idname" style="font-family:var(--font-display);font-weight:800;'
                f'font-size:22px;color:var(--fp-tx);letter-spacing:.01em;">{safe_name}</div>'
                f"{_meta_line}"
                "</div>"
                f"{_readout_wrap}"
                "</div>",
                unsafe_allow_html=True,
            )
            if _lr_badge_html:
                st.markdown(
                    f'<div style="margin:0 0 6px;">{_lr_badge_html}</div>',
                    unsafe_allow_html=True,
                )
            if banner_teaser:
                from src.ui_shared import render_reco_banner

                render_reco_banner(banner_teaser, "", "my_team")
            from src.ui_shared import render_matchup_ticker

            render_matchup_ticker()

            # ── WAR ROOM BRIEFING ────────────────────────────────────────
            # Dynamic, actionable intelligence replacing static alerts.
            # Card 1: Matchup Pulse — live W-L-T with category breakdown
            # Card 2: Flippable Categories — closest cats to flip + suggestions
            # Card 3: Today's Actions — schedule-aware roster moves
            # Card 4: Hot/Cold Report — L7 performance streaks
            try:
                from src.war_room import compute_matchup_pulse, get_flippable_categories
                from src.war_room_actions import compute_todays_actions
                from src.war_room_hotcold import compute_hot_cold_report

                # Fetch matchup data. Do NOT gate on is_connected(): read-only
                # MULTI_USER members never hold a live Yahoo client, so gating
                # here blanked the matchup for every member. get_matchup() already
                # falls back to the SQLite cache the scheduler writes through.
                _wr_matchup = None
                _wr_losing_cats: list[str] = []
                try:
                    if yds:
                        _wr_matchup = yds.get_matchup()
                except Exception:
                    pass

                # ── Card 1: Matchup Pulse (mockup .panel + .cats heat grid) ──
                # Per-category win-probability drives the heat bars. Data source:
                # the live Yahoo matchup category totals (you/opp), fed through
                # pivot_advisor.compute_category_flip_probabilities — the same
                # Normal-CDF model the Category Flip card uses. flip_prob is
                # converted to a WIN probability: winning cats → 1 - P(lose lead),
                # losing cats → P(catch up), tied → 0.5. Win cats render orange,
                # losing cats steel (build_heatbar_html(win=...)). Categories are
                # iterated in LeagueConfig order (no hardcoded list).
                pulse = compute_matchup_pulse(_wr_matchup)
                if pulse["available"]:
                    _wr_losing_cats = pulse.get("losing_cats", [])
                    # Build my/opp per-category totals from the matchup.
                    _mp_my: dict[str, float] = {}
                    _mp_opp: dict[str, float] = {}
                    for _ce in (_wr_matchup or {}).get("categories", []):
                        _cc = str(_ce.get("cat", "")).upper()
                        if not _cc:
                            continue
                        try:
                            _mp_my[_cc] = float(_ce.get("you", 0) or 0)
                        except (ValueError, TypeError):
                            _mp_my[_cc] = 0.0
                        try:
                            _mp_opp[_cc] = float(_ce.get("opp", 0) or 0)
                        except (ValueError, TypeError):
                            _mp_opp[_cc] = 0.0

                    # Games remaining in the matchup week (Mon=0 → 7-dow).
                    from datetime import UTC as _MP_UTC
                    from datetime import datetime as _mp_dt

                    _mp_games_left = max(0, 7 - _mp_dt.now(_MP_UTC).weekday())

                    _mp_winprobs: dict[str, tuple[float, bool]] = {}
                    try:
                        from src.optimizer.pivot_advisor import (
                            compute_category_flip_probabilities as _mp_flip,
                        )

                        _mp_probs = _mp_flip(_mp_my, _mp_opp, _mp_games_left, config=_LC)
                        for _cat, _info in _mp_probs.items():
                            _margin = float(_info.get("margin", 0) or 0)
                            _fp = float(_info.get("flip_prob", 0.5) or 0.5)
                            if _margin > 0:
                                _wp = 1.0 - _fp
                                _is_win = True
                            elif _margin < 0:
                                _wp = _fp
                                _is_win = False
                            else:
                                _wp = 0.5
                                _is_win = False
                            _mp_winprobs[str(_cat).upper()] = (_wp, _is_win)
                    except Exception:
                        # Fall back to the win/loss result flags (0/100) if the
                        # probability model is unavailable.
                        for _c in pulse.get("winning_cats", []):
                            _mp_winprobs[str(_c).upper()] = (0.62, True)
                        for _c in pulse.get("losing_cats", []):
                            _mp_winprobs[str(_c).upper()] = (0.30, False)
                        for _c in pulse.get("tied_cats", []):
                            _mp_winprobs[str(_c).upper()] = (0.50, False)

                    # Heat-bar grid in LeagueConfig category order.
                    _cat_cells = ""
                    for _cat in _LC.all_categories:
                        _wp, _is_win = _mp_winprobs.get(_cat.upper(), (0.5, False))
                        _pct = max(0.0, min(100.0, _wp * 100.0))
                        _cn_color = "var(--fp-ember)" if _is_win else "var(--fp-tx-muted)"
                        _pct_color = "var(--fp-ember)" if _is_win else "var(--fp-tx-subtle)"
                        _cell_border = (
                            "border-color:rgba(255,109,0,.5);background:rgba(255,109,0,.07);"
                            if _is_win
                            else "background:var(--fp-surface);"
                        )
                        _cat_cells += (
                            f'<div class="cat" style="position:relative;border:1px solid var(--fp-divider);'
                            f'border-radius:9px;padding:9px 8px 8px;text-align:center;{_cell_border}">'
                            f'<div class="cn" style="font-size:10px;font-weight:700;letter-spacing:.08em;'
                            f'color:{_cn_color};">{_html.escape(_cat)}</div>'
                            f'<div style="margin-top:7px;">{build_heatbar_html(_pct, win=_is_win)}</div>'
                            f'<div class="pct" style="font-family:var(--font-mono);font-size:9px;'
                            f'color:{_pct_color};margin-top:5px;">{int(round(_pct))}%</div>'
                            f"</div>"
                        )
                    _cats_grid = (
                        '<div class="cats" style="display:grid;'
                        f'grid-template-columns:repeat(6,1fr);gap:8px;">{_cat_cells}</div>'
                    )

                    # Opponent + weekly score sub-header.
                    _mp_score_color = (
                        "var(--fp-primary)"
                        if pulse["margin"] > 0
                        else (T["danger"] if pulse["margin"] < 0 else "var(--fp-tx-muted)")
                    )
                    _mp_sub = (
                        '<div class="matchsub" style="display:flex;align-items:baseline;'
                        'justify-content:space-between;margin-bottom:16px;">'
                        f'<div class="opp" style="font-size:12px;color:var(--fp-tx-muted);">vs '
                        f'<b style="color:var(--fp-tx);font-weight:600;">{_html.escape(str(pulse["opponent"]))}</b></div>'
                        f'<div class="rec" style="font-family:var(--font-display);font-weight:900;'
                        f'font-size:28px;color:{_mp_score_color};letter-spacing:.02em;">{pulse["score"]}</div>'
                        "</div>"
                    )
                    st.markdown(
                        build_panel_html(
                            "Matchup Pulse",
                            _mp_sub + _cats_grid,
                            fig_label=f"WK {pulse['week']}",
                        ),
                        unsafe_allow_html=True,
                    )

                # ── Card 2: Flippable Categories ──
                flippables = get_flippable_categories(_wr_matchup)
                if flippables:
                    _flip_rows = ""
                    for fl in flippables:
                        _fl_cat = fl["category"]
                        _fl_dir = fl["direction"]
                        if _fl_dir == "flip_to_win":
                            _fl_icon_color = T["warn"]
                            _fl_label = "FLIP TO WIN"
                        else:
                            _fl_icon_color = T["sky"]
                            _fl_label = "AT RISK"
                        _fl_gap = fl["gap"]
                        # Format gap based on category type.
                        # Only ERA/WHIP get 2-decimal float here — L (also inverse) is a
                        # counting stat and formats as integer in the else branch.
                        if _fl_cat in ("AVG", "OBP"):
                            _fl_gap_str = f".{int(abs(_fl_gap) * 1000):03d}"
                        elif _fl_cat in ("ERA", "WHIP"):
                            _fl_gap_str = f"{abs(_fl_gap):.2f}"
                        else:
                            _fl_gap_str = str(int(abs(_fl_gap)))
                        _flip_rows += (
                            '<div class="arow" style="display:flex;align-items:center;gap:12px;'
                            'padding:10px 4px;border-bottom:1px solid var(--fp-divider);">'
                            f'<span style="font-family:var(--font-mono);font-size:9.5px;font-weight:600;'
                            f"letter-spacing:.12em;padding:4px 8px;border-radius:5px;white-space:nowrap;"
                            f"background:{_fl_icon_color}1f;color:{_fl_icon_color};"
                            f'border:1px solid {_fl_icon_color}55;">'
                            f"{_fl_label}</span>"
                            f'<span style="font-family:var(--font-mono);font-weight:600;font-size:12px;'
                            f'letter-spacing:.04em;color:var(--fp-tx);min-width:40px;">{_html.escape(str(_fl_cat))}</span>'
                            f'<span style="font-family:var(--font-mono);font-size:10px;'
                            f'color:var(--fp-tx-subtle);white-space:nowrap;">GAP {_fl_gap_str}</span>'
                            f'<span style="font-size:12.5px;color:var(--fp-tx-muted);'
                            f'margin-left:auto;text-align:right;">{_html.escape(str(fl["suggestion"]))}</span>'
                            f"</div>"
                        )
                    st.markdown(
                        build_panel_html(
                            "Flippable Categories",
                            _flip_rows,
                            fig_label=f"{len(flippables)} FLAGGED",
                        ),
                        unsafe_allow_html=True,
                    )

                # ── Card 3: Today's Actions (mockup .panel + .arow rows) ──
                # Each row: PRI·N orange chip + player name + team logo/abbr +
                # the reason (detail), right-aligned. Data from
                # war_room_actions.compute_todays_actions (priority/player/team/
                # detail). The reason text strips the leading "<name> " so it
                # reads as a clean why-clause beside the bold name.
                try:
                    from src.ui_shared import team_logo_url as _action_logo

                    _wr_actions = compute_todays_actions(
                        roster=roster,
                        matchup=_wr_matchup,
                        losing_cats=_wr_losing_cats,
                    )
                    if _wr_actions:
                        _action_rows = ""
                        for _ai, _act in enumerate(_wr_actions, 1):
                            _pri = _act.get("priority", _ai)
                            _aname = _html.escape(str(_act.get("player", "")))
                            _ateam = _html.escape(str(_act.get("team", "")).strip().upper())
                            _detail = str(_act.get("detail", ""))
                            # Trim the leading "<player> " so the why reads cleanly.
                            _why = _detail
                            _plain = str(_act.get("player", ""))
                            if _plain and _why.startswith(_plain):
                                _why = _why[len(_plain) :].lstrip(" -—:")
                            _why = _html.escape(_why) if _why else _html.escape(_detail)
                            _logo_html = (
                                f'<img class="tlogo" src="{_action_logo(_ateam)}" '
                                f'style="width:13px;height:13px;vertical-align:-3px;margin-right:5px;" '
                                f'loading="lazy"/>'
                                if _ateam
                                else ""
                            )
                            _action_rows += (
                                '<div class="arow" style="display:flex;align-items:center;gap:14px;'
                                'padding:12px 4px;border-bottom:1px solid var(--fp-divider);">'
                                '<span class="pri" style="font-family:var(--font-mono);font-size:9.5px;'
                                "font-weight:600;letter-spacing:.12em;padding:4px 8px;border-radius:5px;"
                                "background:rgba(255,109,0,.12);color:var(--fp-ember);"
                                f'border:1px solid rgba(255,109,0,.32);white-space:nowrap;">PRI·{_pri}</span>'
                                f'<span class="name" style="font-weight:600;font-size:13.5px;color:var(--fp-tx);'
                                f'min-width:128px;">{_aname}</span>'
                                f'<span class="team" style="font-family:var(--font-mono);font-size:10px;'
                                f'color:var(--fp-tx-subtle);white-space:nowrap;">{_logo_html}{_ateam}</span>'
                                f'<span class="why" style="font-size:12.5px;color:var(--fp-tx-muted);'
                                f'margin-left:auto;text-align:right;">{_why}</span>'
                                "</div>"
                            )
                        st.markdown(
                            build_panel_html(
                                "Today's Actions",
                                _action_rows,
                                fig_label=f"{len(_wr_actions)} FLAGGED",
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass  # Actions are non-fatal

                # ── Card 4: Hot/Cold Report ──
                try:
                    from src.ui_shared import team_logo_url as _streak_logo

                    _wr_hotcold = compute_hot_cold_report(roster, max_entries=4)
                    if _wr_hotcold:
                        # Mockup .streaks grid of .streak tiles. Each tile: HOT/COLD
                        # tag, player name, team logo, the L7 line (headline) and the
                        # season-delta line (detail). The mockup's decorative
                        # per-game sparkline is omitted — compute_hot_cold_report's
                        # entry dict carries no per-game series, and synthesizing one
                        # would be fabricated data (flagged in the C3 report).
                        _hc_tiles = ""
                        for _hc in _wr_hotcold:
                            _is_hot = _hc.get("status") == "hot"
                            _tone = "hot" if _is_hot else "cold"
                            _accent_grad = (
                                "linear-gradient(180deg,var(--fp-flame),var(--fp-ember))"
                                if _is_hot
                                else "linear-gradient(180deg,#8aa6c0,var(--fp-cold))"
                            )
                            _tag_style = (
                                "background:rgba(255,109,0,.14);color:var(--fp-ember);"
                                if _is_hot
                                else "background:rgba(95,125,156,.16);color:var(--fp-cold);"
                            )
                            _hc_name = _html.escape(str(_hc.get("player", "")))
                            _hc_team = _html.escape(str(_hc.get("team", "")).strip().upper())
                            _hc_head = _html.escape(str(_hc.get("headline", "")))
                            _hc_detail = _html.escape(str(_hc.get("detail", "")))
                            _logo_html = (
                                f'<img class="tlogo" src="{_streak_logo(_hc_team)}" '
                                f'style="width:13px;height:13px;vertical-align:-3px;margin-right:5px;" '
                                f'loading="lazy"/>'
                                if _hc_team
                                else ""
                            )
                            _hc_tiles += (
                                f'<div class="streak {_tone}" style="position:relative;'
                                "border:1px solid var(--fp-divider);border-radius:11px;padding:14px 15px;"
                                'background:var(--fp-surface);overflow:hidden;">'
                                f"<span style=\"content:'';position:absolute;left:0;top:0;bottom:0;"
                                f'width:3px;background:{_accent_grad};"></span>'
                                '<div class="top" style="display:flex;align-items:center;gap:10px;margin-bottom:9px;">'
                                f'<span class="tag" style="font-family:var(--font-mono);font-size:9px;'
                                f"font-weight:600;letter-spacing:.14em;padding:3px 7px;border-radius:4px;"
                                f'{_tag_style}">{"HOT" if _is_hot else "COLD"}</span>'
                                "<div>"
                                f'<div class="pn" style="font-weight:700;font-size:14px;color:var(--fp-tx);">{_hc_name}</div>'
                                f'<div class="pt" style="font-family:var(--font-mono);font-size:10px;'
                                f'color:var(--fp-tx-subtle);">{_logo_html}{_hc_team}</div>'
                                "</div></div>"
                                f'<div class="line" style="font-family:var(--font-mono);font-size:12px;'
                                f'color:var(--fp-tx);letter-spacing:.02em;">{_hc_head}</div>'
                                f'<div class="delta" style="font-size:11px;color:var(--fp-tx-muted);'
                                f'margin-top:4px;">{_hc_detail}</div>'
                                "</div>"
                            )
                        _streaks_grid = (
                            '<div class="streaks" style="display:grid;'
                            f'grid-template-columns:1fr 1fr;gap:12px;">{_hc_tiles}</div>'
                        )
                        st.markdown(
                            build_panel_html(
                                "Player Streaks",
                                _streaks_grid,
                                fig_label="L7 GP",
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass  # Hot/cold is non-fatal

                # ── Card 5: Category Flip Analysis ──
                try:
                    from src.optimizer.pivot_advisor import (
                        compute_category_flip_probabilities,
                        get_pivot_summary,
                    )

                    if _wr_matchup and _wr_matchup.get("categories"):
                        # Extract my/opp totals from matchup categories
                        _flip_my: dict[str, float] = {}
                        _flip_opp: dict[str, float] = {}
                        for _fe in _wr_matchup["categories"]:
                            _fc = _fe.get("cat", "")
                            if _fc:
                                try:
                                    _flip_my[_fc] = float(_fe.get("you", 0) or 0)
                                except (ValueError, TypeError):
                                    _flip_my[_fc] = 0.0
                                try:
                                    _flip_opp[_fc] = float(_fe.get("opp", 0) or 0)
                                except (ValueError, TypeError):
                                    _flip_opp[_fc] = 0.0

                        # Approximate games remaining: 7 - day_of_week (Mon=0)
                        from datetime import UTC
                        from datetime import datetime as _dt_flip

                        _flip_dow = _dt_flip.now(UTC).weekday()  # 0=Mon
                        _flip_games_remaining = max(0, 7 - _flip_dow)

                        _flip_probs = compute_category_flip_probabilities(_flip_my, _flip_opp, _flip_games_remaining)
                        _flip_summary = get_pivot_summary(_flip_my, _flip_opp, _flip_games_remaining)

                        # Build category rows grouped: CONTESTED first, WON, LOST
                        _flip_rows = ""

                        # Shared row builder: mono uppercase chip + mono stat code +
                        # Inter body status text. Keeps mono for figures/labels only.
                        def _flip_row(_chip_label: str, _chip_color: str, _cat: str, _status: str) -> str:
                            return (
                                '<div class="arow" style="display:flex;align-items:center;gap:12px;'
                                'padding:9px 4px;border-bottom:1px solid var(--fp-divider);">'
                                f'<span style="font-family:var(--font-mono);font-size:9px;font-weight:600;'
                                f"letter-spacing:.12em;padding:4px 7px;border-radius:5px;"
                                f"white-space:nowrap;min-width:74px;text-align:center;"
                                f"background:{_chip_color}1f;color:{_chip_color};"
                                f'border:1px solid {_chip_color}55;">{_chip_label}</span>'
                                f'<span style="font-family:var(--font-mono);font-weight:600;font-size:12px;'
                                f'letter-spacing:.04em;color:var(--fp-tx);min-width:36px;">{_html.escape(str(_cat))}</span>'
                                f'<span style="font-size:12px;color:var(--fp-tx-muted);'
                                f'margin-left:auto;text-align:right;">{_status}</span>'
                                "</div>"
                            )

                        # Contested categories (orange) with flip probability
                        for _fc in _flip_summary.get("contested", []):
                            _fi = _flip_probs.get(_fc, {})
                            _fp = _fi.get("flip_prob", 0.5)
                            _fp_pct = int(_fp * 100)
                            _flip_rows += _flip_row(
                                "CONTESTED",
                                T["hot"],
                                _fc,
                                f'<b style="color:var(--fp-primary);font-weight:600;">{_fp_pct}%</b> flip',
                            )
                        # Won categories (green)
                        for _fc in _flip_summary.get("won", []):
                            _flip_rows += _flip_row("WON", T["green"], _fc, "Protect")
                        # Lost categories (ember — functional negative)
                        for _fc in _flip_summary.get("lost", []):
                            _flip_rows += _flip_row("LOST", T["danger"], _fc, "Concede")

                        # Top 1-2 recommended actions (Inter body, not mono)
                        _flip_actions_html = ""
                        _top_actions = _flip_summary.get("recommended_actions", [])[:2]
                        if _top_actions:
                            _flip_actions_html = (
                                '<div style="margin-top:10px;padding-top:9px;border-top:1px solid var(--fp-divider);">'
                            )
                            for _fa in _top_actions:
                                _flip_actions_html += (
                                    '<div style="font-size:12px;color:var(--fp-tx-muted);'
                                    f'padding:2px 0;">{_html.escape(str(_fa))}</div>'
                                )
                            _flip_actions_html += "</div>"

                        st.markdown(
                            build_panel_html(
                                "Category Flip Analysis",
                                f"{_flip_rows}{_flip_actions_html}",
                                fig_label=f"{_flip_games_remaining}D LEFT",
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass  # Flip analysis is non-fatal

            except ImportError:
                pass  # War room modules not available — skip gracefully

            # ── Card 6: Regression Alerts ──
            try:
                from src.alerts import generate_regression_alerts

                _reg_alerts = generate_regression_alerts(roster, min_pa=50)
                if _reg_alerts:
                    _reg_rows = ""
                    for _ra in _reg_alerts[:5]:
                        _ra_type = _ra["alert_type"]
                        if _ra_type == "SELL_HIGH":
                            _ra_color = T["danger"]
                            _ra_label = "SELL HIGH"
                        else:
                            _ra_color = T["green"]
                            _ra_label = "BUY LOW"
                        _ra_div = _ra.get("divergence_sd", 0)
                        _reg_rows += (
                            f'<div style="display:flex;align-items:flex-start;gap:8px;'
                            f'padding:4px 0;border-bottom:1px solid {T["border"]};">'
                            f'<span style="background:{_ra_color};color:#fff;padding:1px 6px;'
                            f"border-radius:4px;font-size:10px;font-weight:700;"
                            f'white-space:nowrap;">{_ra_label}</span>'
                            f'<div style="flex:1;">'
                            f'<div style="font-size:12px;font-weight:600;color:{T["tx"]};">'
                            f"{_ra['player_name']}"
                            f'<span style="color:{T["tx2"]};font-weight:400;font-size:10px;'
                            f'margin-left:6px;">{_ra_div:.1f} SD</span></div>'
                            f'<div style="font-size:10px;color:{T["tx2"]};">'
                            f"xwOBA {_ra['expected']:.3f} vs actual wOBA {_ra['actual']:.3f}</div>"
                            f"</div></div>"
                        )
                    st.markdown(
                        f'<div style="background:{T["card"]};border-left:4px solid {T["warn"]};'
                        f"border-radius:8px;padding:8px 12px;margin-bottom:8px;"
                        f'font-family:IBM Plex Mono,monospace;">'
                        f'<div style="font-size:11px;color:{T["tx2"]};margin-bottom:4px;'
                        f'font-weight:600;">Regression Alerts</div>'
                        f"{_reg_rows}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass  # Regression alerts are non-fatal

            # Weekly Report (always visible, auto-expanded on Mondays)
            try:
                from datetime import UTC, datetime

                from src.weekly_report import generate_monday_report, generate_thursday_checkpoint

                today_dow = datetime.now(UTC).weekday()  # 0=Mon
                is_monday = today_dow == 0

                # Determine current week and opponent
                _report_opp = None
                _report_week = 1
                try:
                    # V4: Use MatchupContextService for opponent intel (unified)
                    # 2026-05-20: function is `get_matchup_context()` (returns
                    # the service singleton), not `get_matchup_context_service`
                    # which was renamed in an earlier sweep. The stale import
                    # was silently dying in the bare except below and
                    # `_report_week` was stuck at its initialized value of 1,
                    # rendering "Weekly Report — Week 1" even in Week 9.
                    from src.matchup_context import get_matchup_context

                    _mcs = get_matchup_context()
                    _report_opp = _mcs.get_opponent_context()
                    from src.opponent_intel import get_week_number

                    _report_week = get_week_number()

                    # If live Yahoo profile lacks strengths/weaknesses, merge from fallback hardcoded profile
                    if _report_opp and (not _report_opp.get("strengths") and not _report_opp.get("weaknesses")):
                        from src.opponent_intel import get_current_opponent

                        _fallback_opp = get_current_opponent()
                        if _fallback_opp:
                            if _fallback_opp.get("strengths"):
                                _report_opp["strengths"] = _fallback_opp["strengths"]
                            if _fallback_opp.get("weaknesses"):
                                _report_opp["weaknesses"] = _fallback_opp["weaknesses"]
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

            # Daily Lineup Validation
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

            # -- Context band --
            # Phase C3: the context totals render in a full-width container
            # rather than a narrow left rail. The Active Roster now spans the
            # full page width below this band (mockup places it full-width at
            # the bottom), so a 2-column split would have stranded an empty
            # right column. ``roster_stat_view`` is unaffected (the totals are
            # computed from session state, not the layout).
            ctx = st.container()

            # -- Context Panel --
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

                # Category Gaps card — compare user vs opponent in weekly H2H matchup
                if hit_stats or pitch_stats:
                    # Inverse stats (lower = winning): {"L", "ERA", "WHIP"} from LeagueConfig.
                    # Hardcoding {"ERA", "WHIP"} here would silently treat MORE losses as winning.
                    _INVERSE_CATS: set[str] = set(_LC.inverse_stats)

                    # Try to get live H2H matchup data from Yahoo
                    _matchup_for_gaps = yds.get_matchup()
                    _opp_totals: dict[str, float] = {}
                    _matchup_label = ""
                    if _matchup_for_gaps and isinstance(_matchup_for_gaps, dict):
                        _matchup_label = (
                            f"Week {_matchup_for_gaps.get('week', '?')} "
                            f"vs {_matchup_for_gaps.get('opp_name', 'Opponent')}"
                        )
                        for _cdict in _matchup_for_gaps.get("categories", []):
                            _cat = str(_cdict.get("cat", "")).upper()
                            _ov_str = str(_cdict.get("opp", "-"))
                            try:
                                if _ov_str not in ("-", ""):
                                    _opp_totals[_cat] = float(_ov_str)
                            except (ValueError, TypeError):
                                pass

                    if _opp_totals:
                        # Build user totals lookup
                        _team_totals: dict[str, float] = {}
                        for _cat, _val in {**hit_stats, **pitch_stats}.items():
                            try:
                                _team_totals[_cat.upper()] = float(str(_val))
                            except (ValueError, TypeError):
                                pass

                        # Also get user values from matchup for consistency
                        _user_matchup_totals: dict[str, float] = {}
                        for _cdict in _matchup_for_gaps.get("categories", []):
                            _cat = str(_cdict.get("cat", "")).upper()
                            _yv_str = str(_cdict.get("you", "-"))
                            try:
                                if _yv_str not in ("-", ""):
                                    _user_matchup_totals[_cat] = float(_yv_str)
                            except (ValueError, TypeError):
                                pass

                        # Use matchup user values if available, else fall back to computed totals
                        _user_vals = _user_matchup_totals if _user_matchup_totals else _team_totals

                        # Compute gap vs opponent for each category
                        _gap_rows: list[dict] = []
                        for _cat, _opp_val in _opp_totals.items():
                            if _cat not in _user_vals:
                                continue
                            _user_val = _user_vals[_cat]
                            _is_inverse = _cat in _INVERSE_CATS
                            # "above" means winning: for inverse cats, lower user value is better
                            _above = (_user_val < _opp_val) if _is_inverse else (_user_val > _opp_val)
                            _tied = abs(_user_val - _opp_val) < 1e-6
                            # Diff: positive = winning for user
                            if _is_inverse:
                                _diff = _opp_val - _user_val  # positive when user ERA < opp ERA
                            else:
                                _diff = _user_val - _opp_val
                            _gap_rows.append(
                                {
                                    "cat": _cat,
                                    "user_val": _user_val,
                                    "opp_val": _opp_val,
                                    "above": _above,
                                    "tied": _tied,
                                    "diff": _diff,
                                    "is_inverse": _is_inverse,
                                }
                            )

                        if _gap_rows:
                            # Priority targets = losing cats CLOSEST TO FLIPPING,
                            # ranked by a normalized z-gap (|diff| / weekly sigma)
                            # so counting cats (R) and rate cats (AVG) are
                            # cross-comparable. Raw-diff sort buried winnable
                            # rate cats behind big-count cats (BR-2b).
                            _sigmas = default_weekly_sigmas()
                            _priority_rows = _rank_priority_losing_cats(_gap_rows, _sigmas, top_n=2)
                            _priority_cats = {r["cat"] for r in _priority_rows}
                            _priority_names = [r["cat"] for r in _priority_rows]

                            _priority_html = ""
                            if _priority_names:
                                _priority_html = (
                                    f'<div style="margin-bottom:8px;padding:6px 8px;'
                                    f"background:rgba(255,109,0,0.07);border-radius:6px;"
                                    f'border-left:3px solid {T["danger"]};">'
                                    f'<div style="font-size:10px;font-weight:700;letter-spacing:0.8px;'
                                    f'color:{T["danger"]};margin-bottom:3px;">'
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
                                _tied = _row["tied"]
                                _diff = _row["diff"]
                                _user_val = _row["user_val"]
                                _opp_val = _row["opp_val"]
                                _is_priority = _cat in _priority_cats
                                if _tied:
                                    _color = T["tx2"]
                                elif _above:
                                    _color = T["green"]
                                else:
                                    _color = T["danger"]
                                # Format values. Only ERA/WHIP get 2-decimal float — L
                                # (also inverse per LeagueConfig.inverse_stats) is a counting
                                # stat and formats as integer in the else branch.
                                if _cat in ("AVG", "OBP"):
                                    _uv_str = format_stat(_user_val, _cat)
                                    _ov_str = format_stat(_opp_val, _cat)
                                    _diff_str = f"{_diff:+.3f}"
                                elif _cat in ("ERA", "WHIP"):
                                    _uv_str = format_stat(_user_val, _cat)
                                    _ov_str = format_stat(_opp_val, _cat)
                                    _diff_str = f"{_diff:+.2f}"
                                else:
                                    _uv_str = f"{int(_user_val)}"
                                    _ov_str = f"{int(_opp_val)}"
                                    _diff_str = f"{_diff:+.0f}"

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
                                    f"{_uv_str} / {_ov_str}</span>"
                                    f'<span style="font-weight:700;color:{_color};min-width:40px;'
                                    f'text-align:right;">'
                                    f"{_diff_str}</span>"
                                    f"</span></div>"
                                )

                            _title = f"Category Gaps — {_matchup_label}" if _matchup_label else "Category Gaps"
                            _gaps_html = _priority_html + _rows_html
                            render_context_card(_title, _gaps_html)
                    else:
                        # No matchup data available
                        render_context_card(
                            "Category Gaps",
                            f'<div style="font-size:12px;color:{T["tx2"]};">'
                            f"No weekly matchup data available. Connect Yahoo to see "
                            f"head-to-head category gaps vs your current opponent.</div>",
                        )

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
                            f"letter-spacing:0.8px;"
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
                            # Only ERA/WHIP get 2-decimal float — L (also inverse) is
                            # a counting stat and falls to .0f in the else branch.
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

                # Statcast Signals card
                try:
                    _sc_pool = load_player_pool()
                    _sc_cols = [
                        "player_id",
                        "xwoba",
                        "barrel_pct",
                        "hard_hit_pct",
                        "stuff_plus",
                        "regression_flag",
                        "is_hitter",
                    ]
                    _sc_available = [c for c in _sc_cols if c in _sc_pool.columns]
                    if "player_id" in _sc_available and len(_sc_available) > 1:
                        _sc_data = _sc_pool[_sc_available].copy()
                        for _nc in ["xwoba", "barrel_pct", "hard_hit_pct", "stuff_plus"]:
                            if _nc in _sc_data.columns:
                                _sc_data[_nc] = pd.to_numeric(_sc_data[_nc], errors="coerce").fillna(0)
                        _sc_roster = roster[["player_id", "name"]].merge(_sc_data, on="player_id", how="left")
                        _sc_rows: list[str] = []

                        # Elite barrel rate hitters
                        if "barrel_pct" in _sc_roster.columns:
                            _elite_barrel = _sc_roster[
                                (_sc_roster.get("is_hitter", 0) == 1) & (_sc_roster["barrel_pct"] > 10)
                            ]
                            for _, _sbr in _elite_barrel.iterrows():
                                _xw_str = ""
                                if "xwoba" in _sc_roster.columns and _sbr.get("xwoba", 0) > 0:
                                    _xw_str = f", xwOBA .{int(_sbr['xwoba'] * 1000):03d}"
                                _sc_rows.append(
                                    f'<div style="padding:2px 0;font-size:12px;">'
                                    f'<span class="health-dot" style="background:{T["green"]};"></span>'
                                    f'<span style="font-weight:600;">{_sbr["name"]}</span> '
                                    f'<span style="color:{T["tx2"]};">barrel {_sbr["barrel_pct"]:.1f}%'
                                    f"{_xw_str}</span></div>"
                                )

                        # Elite xwOBA hitters (not already shown via barrel)
                        _barrel_names = set()
                        if "barrel_pct" in _sc_roster.columns:
                            _barrel_names = set(
                                _sc_roster[(_sc_roster.get("is_hitter", 0) == 1) & (_sc_roster["barrel_pct"] > 10)][
                                    "name"
                                ]
                            )
                        if "xwoba" in _sc_roster.columns:
                            _elite_xw = _sc_roster[
                                (_sc_roster.get("is_hitter", 0) == 1)
                                & (_sc_roster["xwoba"] > 0.350)
                                & (~_sc_roster["name"].isin(_barrel_names))
                            ]
                            for _, _sxr in _elite_xw.iterrows():
                                _sc_rows.append(
                                    f'<div style="padding:2px 0;font-size:12px;">'
                                    f'<span class="health-dot" style="background:{T["green"]};"></span>'
                                    f'<span style="font-weight:600;">{_sxr["name"]}</span> '
                                    f'<span style="color:{T["tx2"]};">xwOBA '
                                    f".{int(_sxr['xwoba'] * 1000):03d}</span></div>"
                                )

                        # Elite Stuff+ pitchers
                        if "stuff_plus" in _sc_roster.columns:
                            _elite_stuff = _sc_roster[
                                (_sc_roster.get("is_hitter", 0) == 0) & (_sc_roster["stuff_plus"] > 110)
                            ]
                            for _, _ssr in _elite_stuff.iterrows():
                                _sc_rows.append(
                                    f'<div style="padding:2px 0;font-size:12px;">'
                                    f'<span class="health-dot" style="background:{T["sky"]};"></span>'
                                    f'<span style="font-weight:600;">{_ssr["name"]}</span> '
                                    f'<span style="color:{T["tx2"]};">Stuff+ '
                                    f"{_ssr['stuff_plus']:.0f}</span></div>"
                                )

                        # BUY_LOW regression candidates
                        if "regression_flag" in _sc_roster.columns:
                            _buy_low = _sc_roster[_sc_roster["regression_flag"] == "BUY_LOW"]
                            for _, _blr in _buy_low.iterrows():
                                _xw_tag = ""
                                if "xwoba" in _sc_roster.columns and _blr.get("xwoba", 0) > 0:
                                    _xw_tag = f" (xwOBA .{int(_blr['xwoba'] * 1000):03d})"
                                _sc_rows.append(
                                    f'<div style="padding:2px 0;font-size:12px;">'
                                    f'<span class="health-dot" style="background:{T["warn"]};"></span>'
                                    f'<span style="font-weight:600;">{_blr["name"]}</span> '
                                    f'<span style="color:{T["warn"]};">BUY LOW'
                                    f"{_xw_tag}</span></div>"
                                )

                        if _sc_rows:
                            render_context_card("Statcast Signals", "".join(_sc_rows))
                except Exception:
                    pass  # Graceful degradation — skip card on any error

                # Data freshness card
                render_data_freshness_card()

            # ══ ACTIVE ROSTER — full-width below the context band ══════════
            # Moved out of the old right column (Phase C3) so the roster panel
            # spans the full page width, matching docs/design/mockup-myteam-v3.png.

            # ── Clickability: a roster player cell links to ?player=<id>.
            # We open the @st.dialog directly from the param and guard re-open
            # with a session sentinel: mutating st.query_params would change the
            # URL and reset session_state (which silently broke the earlier
            # stash/rerun approach), so we leave the param in place and instead
            # remember the last id we opened. Same value on a rerun (e.g. the
            # built-in ✕ dismiss) → do NOT re-open, so close works. The reliable
            # opener is the "Open player dossier" selectbox in the panel header.
            _qp_player = st.query_params.get("player")
            if _qp_player is not None and str(_qp_player) != str(st.session_state.get("_dossier_last_shown")):
                st.session_state["_dossier_last_shown"] = str(_qp_player)
                try:
                    show_player_card_dialog(int(_qp_player))
                except (TypeError, ValueError):
                    pass

            # ── Stat-source control (drives the context-panel totals via the
            # ``roster_stat_view`` session key). Kept; the table adds its own
            # timeframe + hitter/pitcher toggles below.
            stat_view = st.segmented_control(
                "Stat source",
                options=_stat_options,
                default=_stat_default,
                key="roster_stat_view",
            )

            # ── Combustion roster toolbar: timeframe + Hitters/Pitchers + opener ──
            _tf_col, _hp_col, _open_col = st.columns([3, 2, 3])
            with _tf_col:
                timeframe = st.segmented_control(
                    "Timeframe",
                    options=["Season", "L30", "L14", "L7", "Today"],
                    default="Season",
                    key="roster_timeframe",
                )
            with _hp_col:
                side = st.segmented_control(
                    "Side",
                    options=["Hitters", "Pitchers"],
                    default="Hitters",
                    key="roster_side",
                )
            if not timeframe:
                timeframe = "Season"
            if not side:
                side = "Hitters"

            rename_map = {
                "name": "Player",
                "positions": "Pos",
                "roster_slot": "Slot",
            }

            # Base identity columns always carried into the display frame.
            _base_keep = [
                c
                for c in ("player_id", "name", "positions", "roster_slot", "mlb_id", "team", "status", "is_hitter")
                if c in roster.columns
            ]
            display_df = roster[_base_keep].copy()

            # Stat columns we may render (hitter + pitcher union).
            _stat_keys = ["ab", "r", "h", "hr", "rbi", "sb", "avg", "obp", "ip", "w", "l", "sv", "k", "era", "whip"]

            _today_unavailable = False
            if timeframe == "Season":
                # 2026 live season stats (canonical loader; no raw SQL here).
                from src.database import load_season_stats

                _hist = load_season_stats(2026)
                if not _hist.empty and "player_id" in _hist.columns:
                    _hist_cols = [c for c in _stat_keys if c in _hist.columns]
                    display_df = display_df.merge(
                        _hist[["player_id"] + _hist_cols],
                        on="player_id",
                        how="left",
                    )
                else:
                    # No 2026 actuals yet → fall back to roster's blended values.
                    for k in _stat_keys:
                        if k in roster.columns and k not in display_df.columns:
                            display_df[k] = roster[k].values
                _caption = "Full 2026 season totals. Updates hourly from MLB Stats API."
            elif timeframe in ("L30", "L14", "L7"):
                # Rolling window from per-game logs.
                from src.player_databank import compute_rolling_stats

                _days = {"L30": 30, "L14": 14, "L7": 7}[timeframe]
                _pids = display_df["player_id"].tolist() if "player_id" in display_df.columns else []
                _roll = compute_rolling_stats(_pids, days=_days, season=2026)
                if not _roll.empty:
                    # compute_rolling_stats returns summed counting stats +
                    # *_calc rate columns — fold the rate calcs into avg/obp/
                    # era/whip so the renderer's column keys resolve.
                    for _raw, _calc in (
                        ("avg", "avg_calc"),
                        ("obp", "obp_calc"),
                        ("era", "era_calc"),
                        ("whip", "whip_calc"),
                    ):
                        if _calc in _roll.columns:
                            _roll[_raw] = _roll[_calc]
                    _roll_cols = [c for c in _stat_keys if c in _roll.columns]
                    display_df = display_df.merge(
                        _roll[["player_id"] + _roll_cols],
                        on="player_id",
                        how="left",
                    )
                _caption = f"Last {_days} days from per-game logs (weighted rate stats)."
            else:  # "Today"
                # Today's single-game line, if game logs carry it.
                from src.player_databank import compute_rolling_stats

                _pids = display_df["player_id"].tolist() if "player_id" in display_df.columns else []
                _today = compute_rolling_stats(_pids, days=1, season=2026)
                if not _today.empty:
                    for _raw, _calc in (
                        ("avg", "avg_calc"),
                        ("obp", "obp_calc"),
                        ("era", "era_calc"),
                        ("whip", "whip_calc"),
                    ):
                        if _calc in _today.columns:
                            _today[_raw] = _today[_calc]
                    _today_cols = [c for c in _stat_keys if c in _today.columns]
                    display_df = display_df.merge(
                        _today[["player_id"] + _today_cols],
                        on="player_id",
                        how="left",
                    )
                else:
                    _today_unavailable = True
                _caption = "Today's game line. Cells show — when no game has been logged yet."

            # Sort into Yahoo slot order, then split hitters vs pitchers.
            _sortable = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            _sortable = sort_roster_for_display(_sortable)
            # Map renamed cols back to lowercase keys the renderer expects.
            _sortable = _sortable.rename(columns={"Player": "name", "Pos": "positions", "Slot": "roster_slot"})

            def _is_pitcher_row(r) -> bool:
                _ih = r.get("is_hitter")
                if _ih is not None and not (isinstance(_ih, float) and pd.isna(_ih)):
                    try:
                        return not bool(int(_ih))
                    except (ValueError, TypeError):
                        pass
                _pos = str(r.get("positions", "")).upper()
                _tokens = {t.strip() for t in _pos.replace("/", ",").split(",") if t.strip()}
                # Pitcher iff every eligible slot is a pitching slot.
                return bool(_tokens) and _tokens.issubset({"SP", "RP", "P"})

            _mask_pitch = _sortable.apply(_is_pitcher_row, axis=1) if not _sortable.empty else pd.Series([], dtype=bool)
            _is_hitter_view = side == "Hitters"
            _view_df = _sortable[~_mask_pitch] if _is_hitter_view else _sortable[_mask_pitch]
            _view_df = _view_df.reset_index(drop=True)

            _h_count = int((~_mask_pitch).sum()) if not _sortable.empty else 0
            _p_count = int(_mask_pitch.sum()) if not _sortable.empty else 0

            # Player IDs aligned to the displayed (filtered, sorted) rows.
            player_ids_list = _view_df["player_id"].tolist() if "player_id" in _view_df.columns else []

            # ── Reliable dossier opener (selectbox) in the toolbar's right cell.
            # The C1 row <a href="?player="> links are the visual affordance; this
            # relabeled selectbox is the dependable opener (mockup top-right).
            with _open_col:
                if player_ids_list:
                    _player_names = _view_df["name"].tolist() if "name" in _view_df.columns else []
                    if _player_names:
                        render_player_select(
                            _player_names,
                            player_ids_list,
                            key_suffix="myteam",
                            label="Open player dossier",
                        )

            # ── Render inside the full-width instrument panel ──
            _tf_label = {
                "Season": "FULL SEASON",
                "L30": "LAST 30 DAYS",
                "L14": "LAST 14 DAYS",
                "L7": "LAST 7 DAYS",
                "Today": "TODAY",
            }[timeframe]
            _table_html = build_roster_table_html(
                _view_df,
                is_hitter=_is_hitter_view,
                player_ids=player_ids_list,
            )
            _footer_arrow = (
                '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" '
                'stroke="var(--fp-primary)" stroke-width="2.5" style="vertical-align:-1px;margin-right:6px;">'
                '<path d="M9 6l6 6-6 6"/></svg>'
            )
            _footer = (
                '<div class="rfoot" style="font-family:var(--font-mono);font-size:10px;'
                'color:var(--fp-tx-subtle);text-align:center;margin-top:14px;letter-spacing:.08em;">'
                f"{_footer_arrow}CLICK ANY PLAYER FOR GAME LOG · OUTCOMES · UPCOMING PROJECTIONS</div>"
            )
            _cap_html = (
                f'<div style="font-family:var(--font-mono);font-size:10.5px;color:var(--fp-tx-subtle);'
                f'letter-spacing:.04em;margin:-4px 0 6px;">{_html.escape(_caption)}</div>'
            )
            st.markdown(
                build_panel_html(
                    "Active Roster",
                    _cap_html + _table_html + _footer,
                    fig_label=f"{_h_count} HITTERS · {_p_count} PITCHERS · {_tf_label}",
                ),
                unsafe_allow_html=True,
            )

            if _today_unavailable:
                st.caption("No game logged today yet — showing dashes.")

            # ── Export to Excel (current view) ──
            import io

            _export_df = _view_df.drop(
                columns=[c for c in ("is_hitter",) if c in _view_df.columns],
                errors="ignore",
            )
            excel_buf = io.BytesIO()
            _export_df.to_excel(excel_buf, index=False, sheet_name="Roster")
            excel_buf.seek(0)
            _view_label = f"{side}_{timeframe}"
            st.download_button(
                "Export to Excel",
                data=excel_buf,
                file_name=f"heater_roster_{_view_label.replace(' ', '_').lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_roster_excel",
            )

            # ══ Bayesian projections + News — full-width below the roster ══
            # Bayesian-adjusted projections
            if BAYESIAN_AVAILABLE:
                try:
                    from src.database import get_connection, load_season_stats

                    conn = get_connection()
                    try:
                        # Marcel needs the most recent season per rostered player.
                        # The player pool's ytd_* columns only cover 2026, so we
                        # fan out load_season_stats across the most recent few
                        # seasons and dedupe to the latest non-empty row per player.
                        # Routing through the canonical loader keeps SQL out of the
                        # page and applies coerce_numeric_df consistently.
                        roster_pids = roster["player_id"].tolist() if "player_id" in roster.columns else []
                        if roster_pids:
                            _roster_pid_set = {int(p) for p in roster_pids}
                            _frames: list[pd.DataFrame] = []
                            for _yr in (2026, 2025, 2024, 2023):
                                _yr_df = load_season_stats(_yr)
                                if _yr_df.empty or "player_id" not in _yr_df.columns:
                                    continue
                                _yr_df = _yr_df[_yr_df["player_id"].isin(_roster_pid_set)]
                                if not _yr_df.empty:
                                    _frames.append(_yr_df)
                            season_stats = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
                        else:
                            season_stats = pd.DataFrame()

                        # Keep only the latest season per player
                        if not season_stats.empty and "season" in season_stats.columns:
                            season_stats = season_stats.sort_values("season", ascending=False).drop_duplicates(
                                subset=["player_id"], keep="first"
                            )

                        if not season_stats.empty and season_stats.get("games_played", pd.Series([0])).sum() > 0:
                            bayes_progress = st.progress(
                                0, text="Loading preseason projections for Marcel stabilization..."
                            )
                            preseason = pd.read_sql_query("SELECT * FROM projections WHERE system = 'blended'", conn)
                            preseason = coerce_numeric_df(preseason)
                            # Filter preseason to roster players only (Bug fix: was showing ~9K players)
                            if roster_pids and "player_id" in preseason.columns:
                                preseason = preseason[preseason["player_id"].isin(roster_pids)]
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
                                players_lookup = pd.read_sql_query("SELECT player_id, name, mlb_id FROM players", conn)
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
                                    bayes_df[c] = bayes_df[c].map(lambda x, _c=c: format_stat(x, _c))
                            for c in ["ERA", "WHIP"]:
                                if c in bayes_df.columns:
                                    bayes_df[c] = bayes_df[c].map(lambda x, _c=c: format_stat(x, _c))
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
render_feedback_widget("My Team")
