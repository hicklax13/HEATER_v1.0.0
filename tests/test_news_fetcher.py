"""Tests for src/news_fetcher.py — MLB transaction fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.news_fetcher import (
    _fuzzy_match_name,
    aggregate_player_news,
    fetch_player_news,
    fetch_recent_transactions,
)

# ── fetch_recent_transactions ────────────────────────────────────────


class TestFetchRecentTransactions:
    """Tests for the MLB Stats API transaction fetcher."""

    def _mock_response(self, json_data, status_code=200):
        """Create a mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("src.news_fetcher._req.get")
    @patch("src.news_fetcher.STATSAPI_AVAILABLE", True)
    def test_basic_fetch(self, mock_get):
        """Returns parsed transaction dicts from API response."""
        mock_get.return_value = self._mock_response(
            {
                "transactions": [
                    {
                        "person": {"fullName": "Mike Trout"},
                        "description": "Mike Trout placed on 10-day IL with knee strain.",
                        "date": "2026-03-15",
                        "typeDesc": "Injured List",
                    },
                    {
                        "person": {"fullName": "Shohei Ohtani"},
                        "description": "Shohei Ohtani activated from IL.",
                        "date": "2026-03-16",
                        "typeDesc": "Activation",
                    },
                ]
            }
        )

        result = fetch_recent_transactions(days_back=7)

        assert len(result) == 2
        assert result[0]["player_name"] == "Mike Trout"
        assert "knee strain" in result[0]["description"]
        assert result[0]["transaction_type"] == "Injured List"
        assert result[1]["player_name"] == "Shohei Ohtani"

    @patch("src.news_fetcher._req.get")
    @patch("src.news_fetcher.STATSAPI_AVAILABLE", True)
    def test_skips_transactions_without_person(self, mock_get):
        """Transactions missing the 'person' key are skipped."""
        mock_get.return_value = self._mock_response(
            {
                "transactions": [
                    {
                        "description": "Some team-level transaction",
                        "date": "2026-03-15",
                    },
                    {
                        "person": {"fullName": "Aaron Judge"},
                        "description": "Aaron Judge signed extension.",
                        "date": "2026-03-15",
                        "typeDesc": "Signing",
                    },
                ]
            }
        )

        result = fetch_recent_transactions(days_back=3)

        assert len(result) == 1
        assert result[0]["player_name"] == "Aaron Judge"

    @patch("src.news_fetcher._req.get")
    @patch("src.news_fetcher.STATSAPI_AVAILABLE", True)
    def test_api_failure_returns_empty(self, mock_get):
        """API exceptions return empty list gracefully."""
        mock_get.side_effect = Exception("Connection timeout")

        result = fetch_recent_transactions(days_back=7)

        assert result == []

    @patch("src.news_fetcher.STATSAPI_AVAILABLE", False)
    def test_statsapi_not_installed(self):
        """Returns empty list when statsapi is not installed."""
        result = fetch_recent_transactions(days_back=7)
        assert result == []

    @patch("src.news_fetcher._req.get")
    @patch("src.news_fetcher.STATSAPI_AVAILABLE", True)
    def test_empty_transactions_key(self, mock_get):
        """Empty transactions array returns empty list."""
        mock_get.return_value = self._mock_response({"transactions": []})

        result = fetch_recent_transactions(days_back=7)

        assert result == []

    @patch("src.news_fetcher._req.get")
    @patch("src.news_fetcher.STATSAPI_AVAILABLE", True)
    def test_missing_transactions_key(self, mock_get):
        """Response without 'transactions' key returns empty list."""
        mock_get.return_value = self._mock_response({})

        result = fetch_recent_transactions(days_back=7)

        assert result == []


# ── _fuzzy_match_name ────────────────────────────────────────────────


class TestFuzzyMatchName:
    """Tests for internal fuzzy name matching."""

    def test_exact_match_case_insensitive(self):
        candidates = {"mike trout": 1, "shohei ohtani": 2}
        assert _fuzzy_match_name("Mike Trout", candidates) == 1

    def test_fuzzy_match_above_threshold(self):
        candidates = {"ronald acuna jr.": 10, "mookie betts": 20}
        # "Ronald Acuna" should fuzzy-match "ronald acuna jr."
        result = _fuzzy_match_name("Ronald Acuna", candidates, threshold=0.75)
        assert result == 10

    def test_no_match_below_threshold(self):
        candidates = {"mike trout": 1}
        result = _fuzzy_match_name("Completely Different", candidates, threshold=0.75)
        assert result is None

    def test_empty_candidates(self):
        assert _fuzzy_match_name("Mike Trout", {}) is None


# ── aggregate_player_news ────────────────────────────────────────────


class TestAggregatePlayerNews:
    """Tests for mapping transactions to player IDs."""

    def test_basic_aggregation(self):
        transactions = [
            {
                "player_name": "Mike Trout",
                "description": "Placed on IL with knee strain.",
                "date": "2026-03-15",
                "transaction_type": "IL",
            },
            {
                "player_name": "Mike Trout",
                "description": "Had MRI on knee.",
                "date": "2026-03-16",
                "transaction_type": "Status",
            },
            {
                "player_name": "Aaron Judge",
                "description": "Activated from IL.",
                "date": "2026-03-16",
                "transaction_type": "Activation",
            },
        ]
        name_to_id = {"Mike Trout": 1, "Aaron Judge": 2}

        result = aggregate_player_news(transactions, name_to_id)

        assert len(result[1]) == 2  # Two Trout transactions
        assert len(result[2]) == 1  # One Judge transaction
        assert "knee strain" in result[1][0]

    def test_case_insensitive_matching(self):
        transactions = [
            {
                "player_name": "MIKE TROUT",
                "description": "Placed on IL.",
            },
        ]
        name_to_id = {"mike trout": 1}

        result = aggregate_player_news(transactions, name_to_id)

        assert 1 in result
        assert len(result[1]) == 1

    def test_empty_inputs(self):
        assert aggregate_player_news([], {"Mike Trout": 1}) == {}
        assert aggregate_player_news([{"player_name": "X", "description": "Y"}], {}) == {}

    def test_unmatched_players_excluded(self):
        transactions = [
            {
                "player_name": "Unknown Player",
                "description": "Some transaction.",
            },
        ]
        name_to_id = {"Mike Trout": 1}

        result = aggregate_player_news(transactions, name_to_id)

        assert result == {}


# ── fetch_player_news ────────────────────────────────────────────────


class TestFetchPlayerNews:
    """Tests for the convenience wrapper."""

    @patch("src.news_fetcher.fetch_recent_transactions")
    def test_basic_integration(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "player_name": "Mike Trout",
                "description": "Placed on IL with knee strain.",
                "date": "2026-03-15",
                "transaction_type": "IL",
            },
        ]

        pool = pd.DataFrame(
            {
                "player_id": [1, 2],
                "name": ["Mike Trout", "Aaron Judge"],
            }
        )

        result = fetch_player_news(pool, days_back=7)

        assert 1 in result
        assert "knee strain" in result[1][0]

    @patch("src.news_fetcher.fetch_recent_transactions")
    def test_no_transactions_returns_empty(self, mock_fetch):
        mock_fetch.return_value = []

        pool = pd.DataFrame({"player_id": [1], "name": ["Mike Trout"]})

        result = fetch_player_news(pool, days_back=7)

        assert result == {}

    @patch("src.news_fetcher.fetch_recent_transactions")
    def test_missing_columns_returns_empty(self, mock_fetch):
        mock_fetch.return_value = [{"player_name": "Mike Trout", "description": "IL placement."}]

        # DataFrame missing required columns
        pool = pd.DataFrame({"foo": [1], "bar": ["x"]})

        result = fetch_player_news(pool, days_back=7)

        assert result == {}

    @patch("src.news_fetcher.fetch_recent_transactions")
    def test_player_name_column_fallback(self, mock_fetch):
        """Uses 'player_name' column when 'name' is absent."""
        mock_fetch.return_value = [
            {
                "player_name": "Aaron Judge",
                "description": "Activated from IL.",
            },
        ]

        pool = pd.DataFrame(
            {
                "player_id": [2],
                "player_name": ["Aaron Judge"],
            }
        )

        result = fetch_player_news(pool, days_back=7)

        assert 2 in result
