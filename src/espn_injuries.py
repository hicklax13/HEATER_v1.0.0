"""ESPN Injuries — real-time MLB injury data from ESPN's public API.

Fetches injury reports from ESPN's undocumented but free API endpoint.
No authentication required. Stores results in the player_news table
for integration with existing news intelligence pipeline.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"
REQUEST_TIMEOUT = 15


def fetch_espn_injuries() -> list[dict]:
    """Fetch current MLB injury data from ESPN API.

    Returns:
        List of dicts with keys: player_name, team, status, injury_type,
        detail, return_date (if available).
    """
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("ESPN injuries fetch failed", exc_info=True)
        return []

    injuries: list[dict] = []
    for team_entry in data.get("items", data.get("injuries", [])):
        # ESPN returns either items[] or a nested structure
        team_name = ""
        team_abbr = ""

        # Handle different ESPN response formats
        team_data = team_entry.get("team", {})
        if isinstance(team_data, dict):
            team_name = team_data.get("displayName", "")
            team_abbr = team_data.get("abbreviation", "")

        for injury in team_entry.get("injuries", []):
            athlete = injury.get("athlete", {})
            player_name = athlete.get("displayName", "")
            if not player_name:
                continue

            status_obj = injury.get("status", "")
            if isinstance(status_obj, dict):
                status = status_obj.get("type", {}).get("abbreviation", "")
                detail = status_obj.get("detail", "")
            else:
                status = str(status_obj)
                detail = ""

            injury_type = injury.get("type", {})
            if isinstance(injury_type, dict):
                injury_desc = injury_type.get("description", "")
            else:
                injury_desc = str(injury_type) if injury_type else ""

            injuries.append(
                {
                    "player_name": player_name,
                    "team": team_abbr,
                    "team_full": team_name,
                    "status": status,
                    "detail": detail,
                    "injury_type": injury_desc,
                    "return_date": injury.get("returnDate", ""),
                }
            )

    logger.info("Fetched %d ESPN injury reports", len(injuries))
    return injuries


def save_espn_injuries_to_db(injuries: list[dict]) -> int:
    """Store ESPN injury data in the player_news table.

    Uses source='espn_injuries' to distinguish from other news sources.
    Deduplicates via the existing UNIQUE constraint on player_news.
    """
    if not injuries:
        return 0

    from src.database import get_connection
    from src.live_stats import match_player_id

    conn = get_connection()
    count = 0
    try:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()

        for inj in injuries:
            player_name = inj.get("player_name", "")
            if not player_name:
                continue

            # Try to resolve to local player_id
            player_id = match_player_id(player_name, inj.get("team", ""))
            if player_id is None:
                continue

            status = inj.get("status", "")
            detail = inj.get("detail", "")
            injury_type_val = inj.get("injury_type", "")
            headline = f"{status}: {injury_type_val}" if injury_type_val else status

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO player_news
                       (player_id, source, headline, detail, news_type,
                        il_status, injury_body_part, published_at, fetched_at)
                       VALUES (?, 'espn_injuries', ?, ?, 'injury', ?, ?, ?, ?)""",
                    (
                        player_id,
                        headline,
                        detail,
                        status,
                        injury_type_val,
                        now,
                        now,
                    ),
                )
                count += 1
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()

    logger.info("Stored %d ESPN injury entries", count)
    return count


def update_player_injury_flags(injuries: list[dict]) -> int:
    """Write injury status back to the players table.

    Sets ``is_injured=1`` and ``injury_note`` for matched players.
    Does NOT clear ``is_injured`` for unmatched players — absence from
    the ESPN list does not mean a player is healthy.

    Returns the number of players updated.
    """
    if not injuries:
        return 0

    from src.database import get_connection
    from src.live_stats import match_player_id

    conn = get_connection()
    count = 0
    try:
        for inj in injuries:
            player_name = inj.get("player_name", "")
            if not player_name:
                continue

            player_id = match_player_id(player_name, inj.get("team", ""))
            if player_id is None:
                continue

            status = inj.get("status", "")
            injury_type = inj.get("injury_type", "")
            note = f"{status}: {injury_type}" if injury_type else status

            try:
                conn.execute(
                    "UPDATE players SET is_injured = 1, injury_note = ? WHERE player_id = ?",
                    (note, player_id),
                )
                count += 1
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()

    logger.info("Updated is_injured flag for %d players", count)
    return count


def refresh_espn_injuries() -> str:
    """Orchestrator: fetch + store ESPN injuries.

    Returns:
        Status message string.
    """
    try:
        injuries = fetch_espn_injuries()
        if not injuries:
            return "No injuries fetched"
        count = save_espn_injuries_to_db(injuries)
        return f"Stored {count} ESPN injury reports"
    except Exception as exc:
        logger.warning("ESPN injury refresh failed: %s", exc)
        return f"Error: {exc}"
