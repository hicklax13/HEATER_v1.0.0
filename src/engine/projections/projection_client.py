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
