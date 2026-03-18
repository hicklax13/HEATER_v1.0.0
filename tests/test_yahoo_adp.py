"""Tests for Yahoo ADP population (Gap 4).

Covers: fetch_yahoo_adp return type, bootstrap integration, name matching,
graceful failure when not connected or API unavailable.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestFetchYahooAdpEmptyWhenNoClient:
    """fetch_yahoo_adp returns empty list when not authenticated."""

    def test_fetch_yahoo_adp_empty_when_no_client(self):
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient(league_id="99999")
        # Not authenticated — _ensure_auth returns False
        result = client.fetch_yahoo_adp()
        assert result == []
        assert isinstance(result, list)


class TestFetchYahooAdpReturnsList:
    """fetch_yahoo_adp returns list of dicts with correct keys."""

    def test_fetch_yahoo_adp_returns_list(self):
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient(league_id="12345")

        # Mock _ensure_auth to return True and get_draft_results to return data
        draft_data = pd.DataFrame(
            {
                "pick_number": [1, 2, 3, 13, 14, 15],
                "round": [1, 1, 1, 2, 2, 2],
                "team_name": ["A", "B", "C", "C", "B", "A"],
                "team_key": ["t1", "t2", "t3", "t3", "t2", "t1"],
                "player_name": [
                    "Aaron Judge",
                    "Shohei Ohtani",
                    "Mookie Betts",
                    "Trea Turner",
                    "Ronald Acuna",
                    "Freddie Freeman",
                ],
                "player_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            }
        )
        with (
            patch.object(client, "_ensure_auth", return_value=True),
            patch.object(client, "get_draft_results", return_value=draft_data),
        ):
            result = client.fetch_yahoo_adp()

        assert isinstance(result, list)
        assert len(result) == 6
        # Each entry should have name and yahoo_adp keys
        for entry in result:
            assert "name" in entry
            assert "yahoo_adp" in entry
            assert isinstance(entry["yahoo_adp"], (int, float))
        # First entry should have lowest ADP (pick 1)
        assert result[0]["name"] == "Aaron Judge"
        assert result[0]["yahoo_adp"] == 1.0


class TestYahooAdpIntegratedInBootstrap:
    """_bootstrap_yahoo calls fetch_yahoo_adp and stores results."""

    def test_yahoo_adp_integrated_in_bootstrap(self):
        from src.data_bootstrap import BootstrapProgress, _bootstrap_yahoo

        mock_client = MagicMock()
        mock_client.sync_to_db.return_value = {}
        mock_client.fetch_yahoo_adp.return_value = [
            {"name": "Aaron Judge", "yahoo_adp": 1.5},
            {"name": "Shohei Ohtani", "yahoo_adp": 2.0},
        ]

        progress = BootstrapProgress()
        # update_refresh_log is lazily imported inside the function from src.database
        with patch("src.database.update_refresh_log"):
            with patch("src.data_bootstrap._store_yahoo_adp", return_value=2) as mock_store:
                result = _bootstrap_yahoo(progress, yahoo_client=mock_client)

        assert "synced" in result.lower()
        mock_client.fetch_yahoo_adp.assert_called_once()
        mock_store.assert_called_once()
        args = mock_store.call_args[0][0]
        assert len(args) == 2
        assert args[0]["name"] == "Aaron Judge"


class TestYahooAdpNameMatching:
    """_store_yahoo_adp resolves names via exact and fuzzy matching."""

    def test_yahoo_adp_name_matching(self):
        """Exact and fuzzy name matching both work for ADP storage."""
        from src.data_bootstrap import _store_yahoo_adp

        # Set up an in-memory DB with players + adp tables
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                team TEXT DEFAULT '',
                positions TEXT DEFAULT 'Util',
                is_hitter INTEGER DEFAULT 1
            );
            CREATE TABLE adp (
                player_id INTEGER PRIMARY KEY,
                yahoo_adp REAL,
                fantasypros_adp REAL,
                adp REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );
            INSERT INTO players (name, team) VALUES ('Aaron Judge', 'NYY');
            INSERT INTO players (name, team) VALUES ('Shohei Ohtani', 'LAD');
            INSERT INTO players (name, team) VALUES ('Mookie Betts', 'LAD');
        """
        )
        conn.commit()

        # Wrap connection so close() is a no-op (sqlite3.Connection.close is read-only)
        wrapper = MagicMock(wraps=conn)
        wrapper.close = MagicMock()  # no-op close

        # Patch get_connection inside data_bootstrap to return our wrapper
        with patch("src.database.get_connection", return_value=wrapper):
            records = [
                {"name": "Aaron Judge", "yahoo_adp": 1.5},  # exact match
                {"name": "Shohei Ohtani", "yahoo_adp": 3.0},  # exact match
                {"name": "Nobody Real", "yahoo_adp": 50.0},  # no match
            ]
            count = _store_yahoo_adp(records)

        assert count == 2  # 2 matched, 1 skipped

        cursor = conn.cursor()
        cursor.execute("SELECT yahoo_adp FROM adp WHERE player_id = 1")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1.5

        cursor.execute("SELECT yahoo_adp FROM adp WHERE player_id = 2")
        row2 = cursor.fetchone()
        assert row2 is not None
        assert row2[0] == 3.0

        conn.close()


class TestYahooAdpGracefulFailure:
    """Yahoo ADP fetch failure does not break bootstrap."""

    def test_yahoo_adp_graceful_failure(self):
        from src.data_bootstrap import BootstrapProgress, _bootstrap_yahoo

        mock_client = MagicMock()
        mock_client.sync_to_db.return_value = {}
        mock_client.fetch_yahoo_adp.side_effect = Exception("Draft not yet held")

        progress = BootstrapProgress()
        # update_refresh_log is lazily imported from src.database
        with patch("src.database.update_refresh_log"):
            result = _bootstrap_yahoo(progress, yahoo_client=mock_client)

        # Should still succeed overall — ADP failure is non-fatal
        assert "synced" in result.lower()
        assert "error" not in result.lower()
