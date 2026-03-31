"""Tests for enhanced live_stats functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_stats import fetch_mlb_transactions, fetch_probable_pitchers


def test_probable_pitchers_returns_dataframe():
    mock_schedule = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 12345,
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "NYY"},
                                "probablePitcher": {
                                    "fullName": "Gerrit Cole",
                                    "id": 543037,
                                },
                            },
                            "home": {
                                "team": {"abbreviation": "BOS"},
                                "probablePitcher": {
                                    "fullName": "Brayan Bello",
                                    "id": 678394,
                                },
                            },
                        },
                    }
                ],
            }
        ],
    }
    with patch("src.live_stats.statsapi.get", return_value=mock_schedule):
        df = fetch_probable_pitchers("2026-03-30")
        assert not df.empty
        assert "away_pitcher_name" in df.columns
        assert df.iloc[0]["away_pitcher_name"] == "Gerrit Cole"
        assert df.iloc[0]["home_pitcher_name"] == "Brayan Bello"
        assert df.iloc[0]["away_team"] == "NYY"
        assert df.iloc[0]["home_team"] == "BOS"
        assert df.iloc[0]["game_pk"] == 12345


def test_probable_pitchers_handles_error():
    with patch("src.live_stats.statsapi.get", side_effect=Exception("API error")):
        df = fetch_probable_pitchers("2026-03-30")
        assert df.empty


def test_probable_pitchers_defaults_to_today():
    mock_schedule = {"dates": []}
    with patch("src.live_stats.statsapi.get", return_value=mock_schedule) as mock_get:
        df = fetch_probable_pitchers()
        assert df.empty
        # Verify the API was called (date defaulted to today)
        mock_get.assert_called_once()


def test_probable_pitchers_missing_pitcher():
    """Games without a probable pitcher should still produce a row."""
    mock_schedule = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 99999,
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "LAD"},
                            },
                            "home": {
                                "team": {"abbreviation": "SF"},
                                "probablePitcher": {
                                    "fullName": "Logan Webb",
                                    "id": 657277,
                                },
                            },
                        },
                    }
                ],
            }
        ],
    }
    with patch("src.live_stats.statsapi.get", return_value=mock_schedule):
        df = fetch_probable_pitchers("2026-04-01")
        assert len(df) == 1
        assert df.iloc[0]["away_pitcher_name"] == ""
        assert df.iloc[0]["away_pitcher_id"] is None
        assert df.iloc[0]["home_pitcher_name"] == "Logan Webb"


def test_mlb_transactions_returns_dataframe():
    mock_txns = {
        "transactions": [
            {
                "id": 1,
                "date": "2026-03-30",
                "typeDesc": "Recalled From Minors",
                "person": {"fullName": "Test Player", "id": 999},
                "fromTeam": {"abbreviation": "AAA"},
                "toTeam": {"abbreviation": "NYY"},
                "description": "Recalled from AAA",
            }
        ],
    }
    with patch("src.live_stats.statsapi.get", return_value=mock_txns):
        df = fetch_mlb_transactions("2026-03-25", "2026-03-30")
        assert not df.empty
        assert df.iloc[0]["player_name"] == "Test Player"
        assert df.iloc[0]["type_desc"] == "Recalled From Minors"
        assert df.iloc[0]["from_team"] == "AAA"
        assert df.iloc[0]["to_team"] == "NYY"
        assert df.iloc[0]["description"] == "Recalled from AAA"


def test_mlb_transactions_handles_error():
    with patch("src.live_stats.statsapi.get", side_effect=Exception("API error")):
        df = fetch_mlb_transactions("2026-03-25", "2026-03-30")
        assert df.empty


def test_mlb_transactions_defaults_dates():
    mock_txns = {"transactions": []}
    with patch("src.live_stats.statsapi.get", return_value=mock_txns) as mock_get:
        df = fetch_mlb_transactions()
        assert df.empty
        # Verify API was called with auto-generated dates
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        params = call_args[1] if call_args[1] else call_args[0][1]
        assert "startDate" in params
        assert "endDate" in params


def test_mlb_transactions_multiple_results():
    mock_txns = {
        "transactions": [
            {
                "id": 1,
                "date": "2026-03-28",
                "typeDesc": "Placed on IL",
                "person": {"fullName": "Player A", "id": 100},
                "fromTeam": {"abbreviation": "BOS"},
                "toTeam": {},
                "description": "Placed on 15-day IL",
            },
            {
                "id": 2,
                "date": "2026-03-29",
                "typeDesc": "Recalled From Minors",
                "person": {"fullName": "Player B", "id": 200},
                "fromTeam": {},
                "toTeam": {"abbreviation": "BOS"},
                "description": "Recalled from AAA",
            },
        ],
    }
    with patch("src.live_stats.statsapi.get", return_value=mock_txns):
        df = fetch_mlb_transactions("2026-03-25", "2026-03-30")
        assert len(df) == 2
        assert df.iloc[0]["player_name"] == "Player A"
        assert df.iloc[1]["player_name"] == "Player B"


def test_probable_pitchers_statsapi_none():
    """When statsapi is not installed, return empty DataFrame."""
    with patch("src.live_stats.statsapi", None):
        df = fetch_probable_pitchers("2026-03-30")
        assert df.empty


def test_mlb_transactions_statsapi_none():
    """When statsapi is not installed, return empty DataFrame."""
    with patch("src.live_stats.statsapi", None):
        df = fetch_mlb_transactions("2026-03-25", "2026-03-30")
        assert df.empty
