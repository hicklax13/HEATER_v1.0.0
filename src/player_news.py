"""Multi-source player news intelligence module.

Aggregates news from ESPN, RotoWire RSS, MLB Stats API enhanced status,
Yahoo injury fields, and existing MLB transaction feed. Classifies news
by type (injury/transaction/callup/lineup/general), computes ownership
trends, generates template-based analytical summaries, and persists to
the player_news DB table for deduplication.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import pandas as pd

from src.database import get_connection
from src.il_manager import classify_il_type
from src.news_sentiment import compute_news_sentiment

logger = logging.getLogger(__name__)

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    feedparser = None  # type: ignore[assignment]
    FEEDPARSER_AVAILABLE = False


# -- Constants -----------------------------------------------------------

_RATE_LIMIT_SECONDS = 0.5
_MAX_SOURCE_FAILURES = 3
_STUB_SOURCES_WARNED = False


# -- News type classification --------------------------------------------

_INJURY_KEYWORDS = [
    "il",
    "injured list",
    "disabled list",
    "injury",
    "surgery",
    "rehab",
    "day-to-day",
]

_TRANSACTION_KEYWORDS = [
    "traded",
    "trade",
    "waiver",
    "claimed",
    "released",
    "designated",
    "dfa",
]

_CALLUP_KEYWORDS = [
    "called up",
    "recalled",
    "promoted",
    "optioned",
]

_LINEUP_KEYWORDS = [
    "lineup",
    "batting order",
    "leadoff",
    "batting",
    "starting",
]


def _classify_news_type(text: str) -> str:
    """Classify news text into injury/transaction/callup/lineup/general."""
    lower = text.lower()
    # Check in priority order: injury > transaction > callup > lineup
    for kw in _INJURY_KEYWORDS:
        if kw in lower:
            return "injury"
    for kw in _TRANSACTION_KEYWORDS:
        if kw in lower:
            return "transaction"
    for kw in _CALLUP_KEYWORDS:
        if kw in lower:
            return "callup"
    for kw in _LINEUP_KEYWORDS:
        if kw in lower:
            return "lineup"
    return "general"


def _classify_il_status(text: str) -> str | None:
    """Extract IL type from description text. Returns IL10/IL15/IL60/DTD or None."""
    if not text:
        return None
    # Only delegate to classify_il_type if text contains IL-related keywords;
    # otherwise classify_il_type returns None for any text (e.g. trades, callups)
    upper = text.upper()
    il_indicators = ("IL", "DL", "INJURED", "DTD", "DAY-TO-DAY", "10-DAY", "15-DAY", "60-DAY")
    if not any(kw in upper for kw in il_indicators):
        return None
    return classify_il_type(text)


# -- ESPN news parsing ---------------------------------------------------


def _parse_espn_news(data: dict) -> list[dict]:
    """Parse ESPN news JSON, extract articles with athleteId."""
    items: list[dict] = []
    for article in data.get("articles", []):
        # Find athlete IDs in categories
        athlete_ids = []
        for cat in article.get("categories", []):
            aid = cat.get("athleteId")
            if aid is not None:
                athlete_ids.append(aid)
        if not athlete_ids:
            continue
        headline = article.get("headline", "")
        description = article.get("description", "")
        published = article.get("published", "")
        full_text = f"{headline} {description}"
        news_type = _classify_news_type(full_text)
        il_status = _classify_il_status(full_text) if news_type == "injury" else None
        sentiment = compute_news_sentiment([headline, description])
        for aid in athlete_ids:
            items.append(
                {
                    "source": "espn",
                    "headline": headline,
                    "detail": description,
                    "published_at": published,
                    "espn_athlete_id": aid,
                    "news_type": news_type,
                    "il_status": il_status,
                    "sentiment_score": sentiment,
                }
            )
    return items


def fetch_espn_news() -> list[dict]:
    """ESPN player news — not yet implemented (requires ESPN API key)."""
    global _STUB_SOURCES_WARNED  # noqa: PLW0603
    if not _STUB_SOURCES_WARNED:
        logger.info("ESPN news and RotoWire RSS sources are not yet implemented. Using MLB Stats API and Yahoo only.")
        _STUB_SOURCES_WARNED = True
    logger.debug("ESPN news fetch not implemented")
    return []


# -- RotoWire RSS parsing ------------------------------------------------


def _parse_rotowire_entries(entries: list) -> list[dict]:
    """Parse feedparser entries from RotoWire RSS."""
    items: list[dict] = []
    for entry in entries:
        title = entry.get("title", "")
        description = entry.get("description", entry.get("summary", ""))
        published = entry.get("published", "")
        full_text = f"{title} {description}"
        news_type = _classify_news_type(full_text)
        il_status = _classify_il_status(full_text) if news_type == "injury" else None
        sentiment = compute_news_sentiment([title, description])
        items.append(
            {
                "source": "rotowire",
                "headline": title,
                "detail": description,
                "published_at": published,
                "news_type": news_type,
                "il_status": il_status,
                "sentiment_score": sentiment,
            }
        )
    return items


def fetch_rotowire_rss() -> list[dict]:
    """RotoWire RSS feed — not yet implemented (requires RotoWire subscription)."""
    if not FEEDPARSER_AVAILABLE:
        logger.debug("feedparser not installed, skipping RotoWire RSS")
        return []
    logger.debug("RotoWire RSS fetch not implemented")
    return []


# -- MLB enhanced status parsing -----------------------------------------


def _parse_mlb_enhanced_status(data: dict, mlb_id: int) -> list[dict]:
    """Parse MLB API hydrated person response for IL status and transactions."""
    items: list[dict] = []
    people = data.get("people", [])
    if not people:
        return items
    person = people[0]
    player_name = person.get("fullName", "")
    team_name = person.get("currentTeam", {}).get("name", "")

    # Check roster entries for IL status
    il_status = None
    for entry in person.get("rosterEntries", []):
        status = entry.get("status", {})
        code = status.get("code", "")
        desc = status.get("description", "")
        if code or desc:
            il_status = _classify_il_status(f"{code} {desc}")
            if il_status and il_status != "DTD":
                break

    # Process transactions
    for txn in person.get("transactions", []):
        desc = txn.get("description", "")
        date = txn.get("date", "")
        news_type = _classify_news_type(desc)
        txn_il = _classify_il_status(desc) if news_type == "injury" else None
        sentiment = compute_news_sentiment([desc])
        items.append(
            {
                "source": "mlb",
                "headline": f"{player_name}: {desc[:80]}" if len(desc) > 80 else f"{player_name}: {desc}",
                "detail": desc,
                "published_at": date,
                "mlb_id": mlb_id,
                "news_type": news_type,
                "il_status": txn_il or il_status,
                "sentiment_score": sentiment,
            }
        )

    # If no transactions but IL status detected, create a status item
    if not items and il_status:
        items.append(
            {
                "source": "mlb",
                "headline": f"{player_name} on {il_status}",
                "detail": f"Current status: {il_status} for {team_name}",
                "published_at": datetime.now(UTC).isoformat(),
                "mlb_id": mlb_id,
                "news_type": "injury",
                "il_status": il_status,
                "sentiment_score": -0.3,
            }
        )
    return items


def fetch_mlb_enhanced_status(player_ids: list[int] | None = None) -> list[dict]:
    """Batch fetch MLB API enhanced status for multiple players.

    Wraps fetch_player_enhanced_status from live_stats.py.
    """
    if not player_ids:
        return []
    items: list[dict] = []
    try:
        from src.live_stats import fetch_player_enhanced_status
    except ImportError:
        logger.warning("live_stats not available for enhanced status")
        return []

    failure_count = 0
    for mlb_id in player_ids:
        if failure_count >= _MAX_SOURCE_FAILURES:
            logger.warning("Too many MLB API failures, stopping batch")
            break
        try:
            data = fetch_player_enhanced_status(mlb_id)
            if data:
                parsed = _parse_mlb_enhanced_status({"people": [data]}, mlb_id)
                items.extend(parsed)
            time.sleep(_RATE_LIMIT_SECONDS)
        except Exception:
            failure_count += 1
            logger.debug("Enhanced status fetch failed for mlb_id=%s", mlb_id, exc_info=True)
    return items


# -- Yahoo injury extraction ---------------------------------------------


def _extract_yahoo_injury_news(player_data: dict) -> list[dict]:
    """Extract injury info from Yahoo player dict."""
    items: list[dict] = []
    injury_note = player_data.get("injury_note", "")
    status_full = player_data.get("status_full", "")
    if not injury_note and not status_full:
        return items

    combined = f"{status_full} {injury_note}".strip()
    il_status = _classify_il_status(combined)
    player_id = player_data.get("player_id", 0)
    name = player_data.get("name", "Unknown")
    sentiment = compute_news_sentiment([combined])

    items.append(
        {
            "source": "yahoo",
            "headline": f"{name}: {combined[:80]}",
            "detail": combined,
            "published_at": datetime.now(UTC).isoformat(),
            "player_id": player_id,
            "news_type": "injury",
            "il_status": il_status,
            "sentiment_score": sentiment,
            "injury_body_part": _extract_body_part(injury_note),
        }
    )
    return items


def _extract_body_part(injury_note: str) -> str | None:
    """Try to extract body part from injury description."""
    if not injury_note:
        return None
    lower = injury_note.lower()
    body_parts = [
        "knee",
        "hamstring",
        "oblique",
        "shoulder",
        "elbow",
        "back",
        "wrist",
        "ankle",
        "hip",
        "groin",
        "calf",
        "finger",
        "thumb",
        "forearm",
        "quad",
        "rib",
        "foot",
        "hand",
        "neck",
        "ucl",
    ]
    for part in body_parts:
        if part in lower:
            return part
    return None


# -- ESPN athlete ID cross-reference -------------------------------------


def _lookup_player_by_espn_id(espn_id: int) -> int | None:
    """Query player_id_map table for espn_id -> player_id."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT player_id FROM player_id_map WHERE espn_id = ?",
            (espn_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception:
        logger.debug("ESPN ID lookup failed for %s", espn_id, exc_info=True)
        return None
    finally:
        conn.close()


def _resolve_espn_athlete_id(espn_id: int) -> int | None:
    """Look up player_id from player_id_map table by ESPN athlete ID."""
    return _lookup_player_by_espn_id(espn_id)


# -- DB persistence ------------------------------------------------------


def _store_news_items(items: list[dict]) -> int:
    """Store news items to player_news table. Returns count of new rows inserted.

    Uses INSERT OR IGNORE for dedup (UNIQUE on player_id+source+headline+published_at).
    """
    if not items:
        return 0
    conn = get_connection()
    try:
        cursor = conn.cursor()
        inserted = 0
        now = datetime.now(UTC).isoformat()
        for item in items:
            pid = item.get("player_id")
            if not pid or pid == 0:
                logger.debug("Skipping news item without valid player_id: %s", item.get("headline", ""))
                continue
            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO player_news
                       (player_id, source, headline, detail, news_type,
                        injury_body_part, il_status, sentiment_score,
                        published_at, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(pid),
                        item.get("source", ""),
                        item.get("headline", ""),
                        item.get("detail", ""),
                        item.get("news_type", "general"),
                        item.get("injury_body_part"),
                        item.get("il_status"),
                        item.get("sentiment_score"),
                        item.get("published_at", ""),
                        now,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception:
                logger.debug("Failed to insert news item: %s", item.get("headline", ""), exc_info=True)
        conn.commit()
        return inserted
    finally:
        conn.close()


def _query_player_news(player_id: int, limit: int = 20) -> list[dict]:
    """Query recent news for a player from DB."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, player_id, source, headline, detail, news_type,
                      injury_body_part, il_status, sentiment_score,
                      published_at, fetched_at
               FROM player_news
               WHERE player_id = ?
               ORDER BY published_at DESC
               LIMIT ?""",
            (player_id, limit),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
        return rows
    except Exception:
        logger.debug("Failed to query news for player_id=%s", player_id, exc_info=True)
        return []
    finally:
        conn.close()


# -- Ownership trends ----------------------------------------------------


def _load_ownership_history(player_id: int, lookback_days: int = 30) -> list[dict]:
    """Load ownership history from ownership_trends table."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT date, percent_owned
               FROM ownership_trends
               WHERE player_id = ?
               ORDER BY date DESC
               LIMIT ?""",
            (player_id, lookback_days),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
        return rows
    except Exception:
        logger.debug("Failed to load ownership for player_id=%s", player_id, exc_info=True)
        return []
    finally:
        conn.close()


def compute_ownership_trend(
    player_id: int,
    lookback_days: int = 30,
) -> dict:
    """Compute ownership trend from ownership_trends table.

    Returns dict with current, delta_7d, direction, or empty dict if no data.
    """
    history = _load_ownership_history(player_id, lookback_days)
    if not history:
        return {}

    # History is sorted newest first
    current = history[0]["percent_owned"]
    if current is None:
        return {}

    result: dict = {"current": current}

    # Find entry closest to 7 days ago (7th newest, or last available)
    if len(history) >= 2:
        seven_day_idx = min(6, len(history) - 1)
        older = history[seven_day_idx]["percent_owned"]
        if older is not None:
            delta = current - older
            result["delta_7d"] = delta
            if delta > 0:
                result["direction"] = "up"
            elif delta < 0:
                result["direction"] = "down"
            else:
                result["direction"] = "flat"
        else:
            result["delta_7d"] = 0.0
            result["direction"] = "flat"
    else:
        result["delta_7d"] = 0.0
        result["direction"] = "flat"

    return result


# -- Template summary generation -----------------------------------------


def generate_intel_summary(intel: dict) -> str:
    """Generate template-based summary by news_type.

    Templates:
        injury: IL status, body part, duration, SGP loss, replacement, ownership.
        transaction: New team, park factor, SGP impact.
        callup: MiLB level, slash line, projected SGP, roster context.
        lineup: Batting slot, projected PA/week, PA change, SGP impact.
        general: Headline or default message.
    """
    news_type = intel.get("news_type", "general")

    if news_type == "injury":
        il_status = intel.get("il_status", "Unknown")
        body_part = intel.get("injury_body_part", "unknown")
        duration = intel.get("duration_weeks", 0.0)
        lost_sgp = intel.get("lost_sgp", 0.0)
        replacement = intel.get("replacement_name", "None")
        replacement_sgp = intel.get("replacement_sgp", 0.0)
        ownership_dir = intel.get("ownership_trend", "flat")
        ownership_delta = intel.get("ownership_delta", 0.0)
        return (
            f"Placed on {il_status} with {body_part} issue. "
            f"Expected duration: {duration} weeks. "
            f"Projected SGP loss: {lost_sgp}. "
            f"Best replacement: {replacement} ({replacement_sgp} SGP). "
            f"Ownership trending {ownership_dir} ({ownership_delta}%)."
        )

    elif news_type == "transaction":
        new_team = intel.get("new_team", "Unknown")
        pf = intel.get("park_factor", 1.0)
        context = intel.get("park_context", "")
        sgp_delta = intel.get("sgp_delta", 0.0)
        return f"Traded to {new_team}. Park factor: {pf} ({context}). Projected SGP impact: {sgp_delta}."

    elif news_type == "callup":
        level = intel.get("milb_level", "MiLB")
        avg = intel.get("milb_avg", 0.0)
        obp = intel.get("milb_obp", 0.0)
        slg = intel.get("milb_slg", 0.0)
        proj_sgp = intel.get("projected_sgp", 0.0)
        roster_ctx = intel.get("roster_context", "")
        return (
            f"Called up from {level}. "
            f"MiLB slash line: {avg}/{obp}/{slg}. "
            f"Projected SGP: {proj_sgp}. "
            f"{roster_ctx}".rstrip()
        )

    elif news_type == "lineup":
        slot = intel.get("batting_slot", 0)
        pa_week = intel.get("projected_pa_week", 0.0)
        pa_delta = intel.get("pa_delta", 0.0)
        sgp_delta = intel.get("sgp_delta", 0.0)
        return (
            f"New batting order position ({slot}). "
            f"Projected PA/week: {pa_week}. "
            f"PA change: {pa_delta}. "
            f"SGP impact: {sgp_delta}."
        )

    else:
        headline = intel.get("headline", "")
        if headline:
            return headline
        return "No additional context available."


# -- Aggregation ---------------------------------------------------------


def aggregate_news(
    player_id: int,
    mlb_id: int | None = None,
) -> list[dict]:
    """Combine all news sources for one player.

    Tries each source independently with try/except for graceful degradation.
    """
    all_items: list[dict] = []

    # Source 1: ESPN
    try:
        espn_items = fetch_espn_news()
        # Filter to this player's ESPN items (would need player_id resolution)
        for item in espn_items:
            espn_aid = item.get("espn_athlete_id")
            if espn_aid:
                resolved = _resolve_espn_athlete_id(espn_aid)
                if resolved == player_id:
                    item["player_id"] = player_id
                    all_items.append(item)
    except Exception:
        logger.debug("ESPN news fetch failed for player_id=%s", player_id, exc_info=True)

    # Source 2: RotoWire RSS
    try:
        roto_items = fetch_rotowire_rss()
        # RotoWire items need name matching -- skip for now
        all_items.extend(roto_items)
    except Exception:
        logger.debug("RotoWire RSS failed for player_id=%s", player_id, exc_info=True)

    # Source 3: MLB enhanced status
    if mlb_id:
        try:
            mlb_items = fetch_mlb_enhanced_status([mlb_id])
            for item in mlb_items:
                item["player_id"] = player_id
            all_items.extend(mlb_items)
        except Exception:
            logger.debug("MLB enhanced status failed for player_id=%s", player_id, exc_info=True)

    # Source 4: Cached news from DB
    try:
        cached = _query_player_news(player_id)
        # Avoid duplicates -- cached items are already deduplicated by DB constraint
        existing_headlines = {item.get("headline", "") for item in all_items}
        for row in cached:
            if row.get("headline", "") not in existing_headlines:
                all_items.append(row)
    except Exception:
        logger.debug("DB news query failed for player_id=%s", player_id, exc_info=True)

    # Store any new items
    if all_items:
        storable = [item for item in all_items if "player_id" in item]
        if storable:
            try:
                _store_news_items(storable)
            except Exception:
                logger.debug("Failed to store news items", exc_info=True)

    return all_items


# -- Batch roster intel --------------------------------------------------


def generate_roster_intel(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: object | None = None,
) -> dict[int, dict]:
    """Batch generate news intel for a roster.

    Returns: {player_id: {"news": list[dict], "ownership": dict}}
    """
    result: dict[int, dict] = {}

    # Build player_id -> mlb_id lookup from pool
    mlb_id_lookup: dict[int, int | None] = {}
    if "mlb_id" in player_pool.columns:
        for _, row in player_pool.iterrows():
            pid = int(row.get("player_id", 0))
            mid = row.get("mlb_id")
            if pd.notna(mid):
                mlb_id_lookup[pid] = int(mid)
            else:
                mlb_id_lookup[pid] = None

    for pid in roster_ids:
        mlb_id = mlb_id_lookup.get(pid)
        news = aggregate_news(pid, mlb_id=mlb_id)
        ownership = compute_ownership_trend(pid)
        result[pid] = {
            "news": news,
            "ownership": ownership,
        }

    return result


# -- Top-level refresh orchestrator ----------------------------------------


def refresh_all_news(
    yahoo_client: object | None = None,
    force: bool = False,
) -> int:
    """Batch refresh news from all sources and store to DB.

    Returns total number of new items stored.
    """
    total_stored = 0

    # 1. ESPN news
    try:
        espn_items = fetch_espn_news()
        if espn_items:
            total_stored += _store_news_items(espn_items)
    except Exception:
        logger.warning("ESPN news refresh failed", exc_info=True)

    # 2. RotoWire RSS
    try:
        roto_items = fetch_rotowire_rss()
        if roto_items:
            total_stored += _store_news_items(roto_items)
    except Exception:
        logger.warning("RotoWire news refresh failed", exc_info=True)

    # 3. MLB Stats API enhanced status (requires player mlb_ids)
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            rows = conn.execute("SELECT mlb_id FROM players WHERE mlb_id IS NOT NULL AND mlb_id > 0").fetchall()
            mlb_ids = [r[0] for r in rows]
        finally:
            conn.close()
        if mlb_ids:
            mlb_items = fetch_mlb_enhanced_status(mlb_ids[:50])  # cap to avoid rate limits
            if mlb_items:
                total_stored += _store_news_items(mlb_items)
    except Exception:
        logger.warning("MLB enhanced status refresh failed", exc_info=True)

    # 4. Yahoo injury/ownership data (if client available)
    if yahoo_client is not None:
        try:
            # Extract injury news from Yahoo roster data
            from src.yahoo_api import YahooFantasyClient

            if isinstance(yahoo_client, YahooFantasyClient):
                all_rosters = yahoo_client.get_all_rosters()
                if isinstance(all_rosters, pd.DataFrame) and not all_rosters.empty:
                    for _, roster_row in all_rosters.iterrows():
                        player_data = {
                            "player_id": roster_row.get("player_id", 0),
                            "name": roster_row.get("player_name", "Unknown"),
                            "injury_note": roster_row.get("injury_note", ""),
                            "status_full": roster_row.get("status_full", ""),
                        }
                        yahoo_items = _extract_yahoo_injury_news(player_data)
                        if yahoo_items:
                            total_stored += _store_news_items(yahoo_items)
        except Exception:
            logger.warning("Yahoo news extraction failed", exc_info=True)

    logger.info("News refresh complete: %d new items stored", total_stored)
    return total_stored
