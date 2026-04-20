"""Tests for injury status writeback to players table."""

from unittest.mock import MagicMock, patch

import pytest


def test_update_player_injury_flags_returns_zero_for_empty():
    from src.espn_injuries import update_player_injury_flags

    assert update_player_injury_flags([]) == 0


def test_update_player_injury_flags_skips_no_name():
    from src.espn_injuries import update_player_injury_flags

    # get_connection and match_player_id are imported lazily inside the function;
    # patch them at their source modules.
    with patch("src.database.get_connection") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.commit = MagicMock()
        mock_conn.return_value.close = MagicMock()
        result = update_player_injury_flags([{"player_name": "", "team": "NYY"}])
    assert result == 0


def test_update_player_injury_flags_skips_unmatched():
    from src.espn_injuries import update_player_injury_flags

    mock_conn_obj = MagicMock()
    mock_conn_obj.execute = MagicMock()
    mock_conn_obj.commit = MagicMock()
    mock_conn_obj.close = MagicMock()

    with patch("src.live_stats.match_player_id", return_value=None):
        with patch("src.database.get_connection", return_value=mock_conn_obj):
            result = update_player_injury_flags(
                [{"player_name": "Unknown Player", "team": "NYY", "status": "IL15", "injury_type": "Elbow"}]
            )
    assert result == 0
