"""Test BUG-013 fix: two-way player injury history uses max games across stat groups."""

from unittest.mock import patch

import pandas as pd
import pytest


def test_two_way_player_uses_max_games_played():
    """Ohtani-like two-way player: hitting=159 games, pitching=10 games.
    The recorded games_played should be 159 (not 10)."""
    mock_response = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "group": {"displayName": "hitting"},
                "splits": [{"season": "2024", "stat": {"gamesPlayed": 159}}],
            },
            {
                "type": {"displayName": "yearByYear"},
                "group": {"displayName": "pitching"},
                "splits": [{"season": "2024", "stat": {"gamesPlayed": 10}}],
            },
        ]
    }
    with patch("src.injury_model.statsapi.player_stat_data", return_value=mock_response):
        from src.injury_model import load_injury_history_from_api

        df = load_injury_history_from_api([660271])
    row_2024 = df[df["season"] == 2024]
    assert not row_2024.empty
    games_played = int(row_2024.iloc[0]["games_played"])
    assert games_played == 159, (
        f"BUG-013: two-way player games_played overwritten by pitching group; "
        f"expected max(159, 10) = 159, got {games_played}"
    )


def test_single_group_player_unchanged():
    """A single-position player should still get the right games_played."""
    mock_response = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "group": {"displayName": "hitting"},
                "splits": [{"season": "2024", "stat": {"gamesPlayed": 145}}],
            },
        ]
    }
    with patch("src.injury_model.statsapi.player_stat_data", return_value=mock_response):
        from src.injury_model import load_injury_history_from_api

        df = load_injury_history_from_api([12345])
    row_2024 = df[df["season"] == 2024]
    assert int(row_2024.iloc[0]["games_played"]) == 145
