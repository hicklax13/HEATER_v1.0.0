"""Tests for weekly transaction counting."""

import pandas as pd
import pytest


def test_get_weekly_transaction_count():
    from src.league_rules import get_weekly_transaction_count

    txns = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-04-14 10:00",
                    "2026-04-15 12:00",
                    "2026-04-16 08:00",
                    "2026-04-20 09:00",
                ]
            ),
            "type": ["add", "add", "trade", "add"],
        }
    )
    count = get_weekly_transaction_count(
        txns,
        matchup_start=pd.Timestamp("2026-04-14"),
        matchup_end=pd.Timestamp("2026-04-20"),
    )
    assert count == 3


def test_get_weekly_transaction_count_empty():
    from src.league_rules import get_weekly_transaction_count

    txns = pd.DataFrame(columns=["timestamp", "type"])
    count = get_weekly_transaction_count(
        txns,
        matchup_start=pd.Timestamp("2026-04-14"),
        matchup_end=pd.Timestamp("2026-04-20"),
    )
    assert count == 0


def test_get_weekly_transaction_count_excludes_drops():
    from src.league_rules import get_weekly_transaction_count

    txns = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 10:00", "2026-04-15 11:00"]),
            "type": ["add", "drop"],
        }
    )
    count = get_weekly_transaction_count(
        txns,
        matchup_start=pd.Timestamp("2026-04-14"),
        matchup_end=pd.Timestamp("2026-04-20"),
    )
    assert count == 1


def test_is_at_transaction_limit():
    from src.league_rules import is_at_transaction_limit

    assert is_at_transaction_limit(10, limit=10) is True
    assert is_at_transaction_limit(9, limit=10) is False
    assert is_at_transaction_limit(0, limit=10) is False
    assert is_at_transaction_limit(11, limit=10) is True


def test_get_transactions_remaining():
    from src.league_rules import get_transactions_remaining

    assert get_transactions_remaining(7, limit=10) == 3
    assert get_transactions_remaining(10, limit=10) == 0
    assert get_transactions_remaining(12, limit=10) == 0
