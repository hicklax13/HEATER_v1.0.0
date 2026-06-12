"""League rule enforcement: transaction limits, undroppable players, weeks_remaining."""

from __future__ import annotations

import logging
from datetime import date as _date

import pandas as pd

logger = logging.getLogger(__name__)

# FourzynBurn: 10 adds + trades combined per matchup week (FCFS waivers).
# Canonical home of the weekly transaction budget — every consumer
# (streaming, FA engine, UI budget strips) must derive from this value.
WEEKLY_TRANSACTION_LIMIT: int = 10


def weeks_remaining(as_of: _date | None = None, season: int = 2026) -> int:
    """Canonical accessor for weeks remaining in the regular season.

    Section 5 consolidation (2026-05-19). Thin wrapper around
    src.validation.dynamic_context.compute_weeks_remaining that pins
    season + sources total_weeks from LeagueConfig.season_weeks (26 for
    FourzynBurn). Use this everywhere instead of inline
    ``(season_end - now).days // 7`` formulas or the alternate
    ``playoff_sim.estimate_weeks_remaining``.

    Args:
        as_of: Date to compute against (default: today, UTC).
        season: MLB season year (default: 2026).

    Returns:
        Int weeks remaining in the regular season, in [1, 26].
    """
    from src.validation.dynamic_context import compute_weeks_remaining
    from src.valuation import LeagueConfig

    return compute_weeks_remaining(
        as_of=as_of,
        season=season,
        total_weeks=LeagueConfig().season_weeks,
    )


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


def is_at_transaction_limit(current_count: int, limit: int = WEEKLY_TRANSACTION_LIMIT) -> bool:
    """Return True if the weekly transaction limit has been reached."""
    return current_count >= limit


def get_transactions_remaining(current_count: int, limit: int = WEEKLY_TRANSACTION_LIMIT) -> int:
    """Return how many transactions remain this week."""
    return max(0, limit - current_count)
