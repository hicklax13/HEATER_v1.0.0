"""Fetch recent MLB transactions from the MLB Stats API.

Provides a real news feed that integrates with the existing
news_sentiment module. Transactions include trades, IL placements,
call-ups, signings, and other roster moves.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import statsapi

    STATSAPI_AVAILABLE = True
except ImportError:
    statsapi = None
    STATSAPI_AVAILABLE = False


# ── Fuzzy matching ───────────────────────────────────────────────────


def _fuzzy_match_name(
    name: str,
    candidates: dict[str, int],
    threshold: float = 0.75,
) -> int | None:
    """Case-insensitive fuzzy match of *name* against candidate names.

    Args:
        name: Player name from the transaction feed.
        candidates: Mapping of lowercase player name -> player_id.
        threshold: Minimum SequenceMatcher ratio to accept.

    Returns:
        Matched player_id, or None if no match above threshold.
    """
    name_lower = name.lower().strip()

    # Pass 1: exact case-insensitive match
    if name_lower in candidates:
        return candidates[name_lower]

    # Pass 2: best fuzzy match above threshold
    best_ratio = 0.0
    best_id: int | None = None
    for cand_name, pid in candidates.items():
        ratio = SequenceMatcher(None, name_lower, cand_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = pid

    if best_ratio >= threshold:
        return best_id
    return None


# ── Public API ───────────────────────────────────────────────────────


def fetch_recent_transactions(days_back: int = 7) -> list[dict]:
    """Query MLB Stats API for recent transactions.

    Args:
        days_back: Number of days to look back. Default 7.

    Returns:
        List of dicts, each with keys:
            player_name (str), description (str),
            date (str), transaction_type (str).
        Returns empty list on any failure.
    """
    if not STATSAPI_AVAILABLE:
        logger.warning("statsapi not installed — skipping transaction fetch")
        return []

    end_date = datetime.now(UTC).strftime("%m/%d/%Y")
    start_date = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%m/%d/%Y")

    try:
        data = statsapi.get(
            "transactions",
            {"startDate": start_date, "endDate": end_date},
        )
    except Exception:
        logger.warning("Failed to fetch transactions from MLB Stats API", exc_info=True)
        return []

    raw_transactions = data.get("transactions", [])
    results: list[dict] = []

    for txn in raw_transactions:
        person = txn.get("person")
        if not person:
            continue

        player_name = person.get("fullName", "")
        if not player_name:
            continue

        description = txn.get("description", "")
        date = txn.get("date", "")
        txn_type = txn.get("typeDesc", txn.get("type", "Unknown"))

        results.append(
            {
                "player_name": player_name,
                "description": description,
                "date": date,
                "transaction_type": str(txn_type),
            }
        )

    logger.info("Fetched %d transactions from last %d days", len(results), days_back)
    return results


def aggregate_player_news(
    transactions: list[dict],
    player_name_to_id: dict[str, int],
) -> dict[int, list[str]]:
    """Map transactions to player_ids using fuzzy name matching.

    Args:
        transactions: List of transaction dicts from fetch_recent_transactions().
        player_name_to_id: Mapping of player name (any case) -> player_id.

    Returns:
        Dict mapping player_id to list of description strings.
    """
    if not transactions or not player_name_to_id:
        return {}

    # Build a lowercase lookup for case-insensitive matching
    lower_lookup: dict[str, int] = {name.lower().strip(): pid for name, pid in player_name_to_id.items()}

    result: dict[int, list[str]] = {}

    for txn in transactions:
        player_name = txn.get("player_name", "")
        description = txn.get("description", "")
        if not player_name or not description:
            continue

        pid = _fuzzy_match_name(player_name, lower_lookup)
        if pid is not None:
            result.setdefault(pid, []).append(description)

    return result


def fetch_player_news(
    player_pool: pd.DataFrame,
    days_back: int = 7,
) -> dict[int, list[str]]:
    """Convenience: fetch transactions and aggregate using player pool names.

    Args:
        player_pool: DataFrame with 'player_id' and 'name' columns
            (as returned by load_player_pool()).
        days_back: Number of days to look back.

    Returns:
        Dict mapping player_id to list of news description strings.
    """
    transactions = fetch_recent_transactions(days_back=days_back)
    if not transactions:
        return {}

    # Build name -> player_id mapping from the pool
    name_col = "name" if "name" in player_pool.columns else "player_name"
    if name_col not in player_pool.columns or "player_id" not in player_pool.columns:
        logger.warning("player_pool missing required columns (%s, player_id)", name_col)
        return {}

    name_to_id: dict[str, int] = {}
    for _, row in player_pool.iterrows():
        pname = row[name_col]
        pid = row["player_id"]
        if pd.notna(pname) and pd.notna(pid):
            name_to_id[str(pname)] = int(pid)

    return aggregate_player_news(transactions, name_to_id)
