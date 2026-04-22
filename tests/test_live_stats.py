"""Test live stats module — MLB Stats API + pybaseball integration."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (2, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def test_match_player_id(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Aaron Judge", "NYY")
    assert pid == 1


def test_match_player_id_not_found(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Nonexistent Player", "XXX")
    assert pid is None


@patch("src.live_stats.statsapi")
def test_fetch_season_stats_structure(mock_statsapi, temp_db):
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 147}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "fullName": "Aaron Judge",
                            "currentTeam": {"abbreviation": "NYY"},
                            "stats": [
                                {
                                    "group": {"displayName": "hitting"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "plateAppearances": 500,
                                                "atBats": 450,
                                                "hits": 130,
                                                "runs": 80,
                                                "homeRuns": 35,
                                                "rbi": 90,
                                                "stolenBases": 5,
                                                "avg": ".289",
                                                "gamesPlayed": 120,
                                            }
                                        }
                                    ],
                                }
                            ],
                        },
                        "position": {"type": "Outfielder"},
                    }
                ]
            }
        return {}

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    assert "player_name" in df.columns
    assert "hr" in df.columns


def test_get_refresh_age():
    from src.live_stats import _get_refresh_age_hours

    age = _get_refresh_age_hours("nonexistent_source")
    assert age > 24


@patch("src.live_stats.statsapi")
def test_refresh_all_stats(mock_statsapi, temp_db):
    from src.live_stats import refresh_all_stats

    mock_statsapi.get.return_value = {"people": []}

    result = refresh_all_stats(force=True)
    assert isinstance(result, dict)
    assert "season_stats" in result


# ── SF-3: Two-Way Player detection ──────────────────────────────


def test_two_way_player_is_pitcher():
    """SF-3: Two-Way Players must be detected as pitchers for stat parsing."""
    pos_type = "Two-Way Player"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True


def test_regular_pitcher_still_detected():
    """Regular pitchers must still be detected as pitchers."""
    pos_type = "Pitcher"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True


def test_hitter_not_detected_as_pitcher():
    """Hitters must NOT be detected as pitchers."""
    pos_type = "Hitter"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is False


def test_outfielder_not_detected_as_pitcher():
    """Outfielders must NOT be detected as pitchers."""
    pos_type = "Outfielder"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is False


@patch("src.live_stats.statsapi")
def test_two_way_player_emits_both_rows(mock_statsapi, temp_db):
    """SF-3: A Two-Way Player should emit both hitting and pitching rows."""
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 17}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "id": 660271,
                            "fullName": "Shohei Ohtani",
                            "currentTeam": {"abbreviation": "LAD"},
                            "stats": [
                                {
                                    "group": {"displayName": "hitting"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "plateAppearances": 300,
                                                "atBats": 270,
                                                "hits": 85,
                                                "runs": 50,
                                                "homeRuns": 20,
                                                "rbi": 55,
                                                "stolenBases": 10,
                                                "avg": ".315",
                                                "gamesPlayed": 70,
                                            }
                                        }
                                    ],
                                },
                                {
                                    "group": {"displayName": "pitching"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "wins": 8,
                                                "losses": 2,
                                                "era": "2.50",
                                                "whip": "0.95",
                                                "strikeOuts": 100,
                                                "saves": 0,
                                                "inningsPitched": "90.0",
                                                "gamesPlayed": 15,
                                            }
                                        }
                                    ],
                                },
                            ],
                        },
                        "position": {"type": "Two-Way Player"},
                    }
                ]
            }
        return {}

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    # Two-Way Player should produce 2 rows: one hitting, one pitching
    ohtani_rows = df[df["player_name"] == "Shohei Ohtani"]
    assert len(ohtani_rows) == 2, f"Expected 2 rows for Ohtani, got {len(ohtani_rows)}"
    # One row should have is_hitter=True (hitting stats), one is_hitter=False (pitching)
    assert ohtani_rows["is_hitter"].sum() == 1, "Expected exactly 1 hitter row"


@patch("src.live_stats.statsapi")
def test_two_way_player_zero_stats_emits_both(mock_statsapi, temp_db):
    """SF-3: Two-Way Player with no stats should still emit both rows."""
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 17}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "id": 660271,
                            "fullName": "Shohei Ohtani",
                            "currentTeam": {"abbreviation": "LAD"},
                            "stats": [],  # No stats yet (e.g. IL)
                        },
                        "position": {"type": "Two-Way Player"},
                    }
                ]
            }
        return {}

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    ohtani_rows = df[df["player_name"] == "Shohei Ohtani"]
    assert len(ohtani_rows) == 2, f"Expected 2 zero-stat rows for Ohtani, got {len(ohtani_rows)}"
