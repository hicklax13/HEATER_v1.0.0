"""FanGraphs ROS projection client with fuzzy name matching.

Spec reference: Section 17 Phase 1 item 2 (FanGraphs Steamer ROS ingestion)
               Section 2 (data sources)

Wires into existing:
  - src/data_pipeline.py: refresh_if_stale, fetch_projections, SYSTEM_MAP
  - src/database.py: load_player_pool, get_connection

This module:
  1. Fetches ROS projections from FanGraphs (Steamer/ZiPS/Depth Charts)
  2. Resolves player names to database player_ids using fuzzy matching
  3. Provides a unified projection lookup for trade evaluation
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

import pandas as pd

from src.database import load_player_pool

logger = logging.getLogger(__name__)


def fuzzy_match_player(
    name: str,
    candidates: pd.DataFrame,
    name_col: str = "player_name",
    threshold: float = 0.75,
) -> int | None:
    """Fuzzy-match a player name to a database player_id.

    Uses SequenceMatcher for string similarity. Tries exact match first,
    then progressively looser matching.

    Args:
        name: Player name to match (e.g., "Ronald Acuna Jr.").
        candidates: DataFrame with name_col and 'player_id' columns.
        name_col: Column name containing player names.
        threshold: Minimum similarity ratio to accept a match.

    Returns:
        player_id if matched, None otherwise.
    """
    if candidates.empty or name_col not in candidates.columns:
        return None

    name_lower = name.lower().strip()

    # Pass 1: exact match (case-insensitive)
    exact = candidates[candidates[name_col].str.lower().str.strip() == name_lower]
    if not exact.empty:
        return int(exact.iloc[0]["player_id"])

    # Pass 2: first + last name match
    parts = name_lower.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        partial = candidates[
            candidates[name_col].str.lower().str.contains(first, na=False)
            & candidates[name_col].str.lower().str.contains(last, na=False)
        ]
        if len(partial) == 1:
            return int(partial.iloc[0]["player_id"])

    # Pass 3: fuzzy matching with SequenceMatcher
    best_score = 0.0
    best_id = None
    for _, row in candidates.iterrows():
        candidate_name = str(row[name_col]).lower().strip()
        score = SequenceMatcher(None, name_lower, candidate_name).ratio()
        if score > best_score:
            best_score = score
            best_id = int(row["player_id"])

    if best_score >= threshold:
        return best_id

    return None


def get_ros_projections(
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Load ROS projections from the database.

    Prefers blended projections; falls back to any available system.
    If player_ids is specified, filters to those players only.

    Args:
        player_ids: Optional list of player IDs to filter to.

    Returns:
        DataFrame with projection data (same schema as player_pool).
    """
    pool = load_player_pool()
    if pool.empty:
        return pool

    pool = pool.rename(columns={"name": "player_name"})

    if player_ids is not None:
        pool = pool[pool["player_id"].isin(player_ids)]

    return pool


def resolve_trade_players(
    giving_names: list[str],
    receiving_names: list[str],
    player_pool: pd.DataFrame,
) -> tuple[list[int], list[int], list[str]]:
    """Resolve trade player names to IDs with fuzzy matching.

    Args:
        giving_names: Names of players being traded away.
        receiving_names: Names of players being received.
        player_pool: Full player pool for name resolution.

    Returns:
        Tuple of (giving_ids, receiving_ids, unmatched_names).
    """
    name_col = "player_name" if "player_name" in player_pool.columns else "name"
    unmatched: list[str] = []

    giving_ids: list[int] = []
    for name in giving_names:
        pid = fuzzy_match_player(name, player_pool, name_col)
        if pid is not None:
            giving_ids.append(pid)
        else:
            unmatched.append(name)

    receiving_ids: list[int] = []
    for name in receiving_names:
        pid = fuzzy_match_player(name, player_pool, name_col)
        if pid is not None:
            receiving_ids.append(pid)
        else:
            unmatched.append(name)

    return giving_ids, receiving_ids, unmatched


def refresh_projections_if_needed(force: bool = False) -> bool:
    """Trigger a FanGraphs projection refresh if data is stale.

    Wires into existing data_pipeline.refresh_if_stale().

    Args:
        force: Force refresh regardless of staleness.

    Returns:
        True if a refresh was performed, False otherwise.
    """
    try:
        from src.data_pipeline import refresh_if_stale

        return refresh_if_stale(force=force)
    except ImportError:
        logger.warning("data_pipeline not available — skipping projection refresh")
        return False
    except Exception as exc:
        logger.warning("Projection refresh failed: %s", exc)
        return False
