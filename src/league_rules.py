"""League rule enforcement: transaction limits, undroppable players."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_weekly_transaction_count(
    transactions_df: pd.DataFrame,
    matchup_start: pd.Timestamp,
    matchup_end: pd.Timestamp,
) -> int:
    """Count adds + trades in the current matchup week."""
    if transactions_df.empty:
        return 0
    mask = (
        (transactions_df["timestamp"] >= matchup_start)
        & (transactions_df["timestamp"] < matchup_end)
        & (transactions_df["type"].isin(["add", "trade"]))
    )
    return int(mask.sum())


def is_at_transaction_limit(current_count: int, limit: int = 10) -> bool:
    """Return True if the weekly transaction limit has been reached."""
    return current_count >= limit


def get_transactions_remaining(current_count: int, limit: int = 10) -> int:
    """Return how many transactions remain this week."""
    return max(0, limit - current_count)
