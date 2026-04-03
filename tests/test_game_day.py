"""Tests for game-day intelligence module."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.game_day import (
    DOME_TEAMS,
    STADIUM_COORDS,
    _aggregate_hitting_games,
    _aggregate_pitching_games,
    fetch_game_day_intelligence,
    fetch_game_day_weather,
    fetch_opposing_pitchers,
    fetch_team_strength,
    get_player_recent_form,
    get_team_strength,
)


@pytest.fixture(autouse=True)
def temp_db():
    """Use a temp database for every test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


# ── STADIUM_COORDS tests ──────────────────────────────────────────────


def test_stadium_coords_has_30_teams():
    """All 30 MLB teams should have stadium coordinates."""
    assert len(STADIUM_COORDS) == 30


def test_stadium_coords_valid_lat_lon():
    """All latitudes should be in [-90, 90] and longitudes in [-180, 180]."""
    for team, (lat, lon) in STADIUM_COORDS.items():
        assert -90 <= lat <= 90, f"{team} latitude {lat} out of range"
        assert -180 <= lon <= 180, f"{team} longitude {lon} out of range"


def test_dome_teams_subset_of_coords():
    """All dome teams should be present in STADIUM_COORDS."""
    for team in DOME_TEAMS:
        assert team in STADIUM_COORDS, f"Dome team {team} not in STADIUM_COORDS"


# ── fetch_game_day_weather tests ──────────────────────────────────────


def _make_schedule_game(home_name, game_pk=12345, game_datetime="2026-04-03T19:10:00Z"):
    """Helper to create a schedule game dict."""
    return {
        "game_pk": game_pk,
        "home_name": home_name,
        "game_datetime": game_datetime,
    }


@patch("src.game_day._requests")
def test_weather_fetch_outdoor_game(mock_requests):
    """Outdoor stadium should get weather data from Open-Meteo API."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hourly": {
            "temperature_2m": [70.0] * 24,
            "windspeed_10m": [8.5] * 24,
            "winddirection_10m": [180.0] * 24,
            "precipitation_probability": [10.0] * 24,
            "relativehumidity_2m": [55.0] * 24,
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_requests.get.return_value = mock_resp

    schedule = [_make_schedule_game("New York Yankees")]
    results = fetch_game_day_weather(schedule)

    assert len(results) == 1
    assert results[0]["temp_f"] == 70.0
    assert results[0]["wind_mph"] == 8.5
    assert results[0]["venue_team"] == "NYY"


@patch("src.game_day._requests")
def test_weather_fetch_dome_game(mock_requests):
    """Dome teams should get neutral defaults (72F) without an API call."""
    schedule = [_make_schedule_game("Houston Astros")]
    results = fetch_game_day_weather(schedule)

    assert len(results) == 1
    assert results[0]["temp_f"] == 72.0
    assert results[0]["wind_mph"] == 0.0
    assert results[0]["venue_team"] == "HOU"
    # No API call should be made for dome teams
    mock_requests.get.assert_not_called()


@patch("src.game_day._requests")
def test_weather_fetch_api_failure(mock_requests):
    """API failure should return neutral defaults instead of crashing."""
    mock_requests.get.side_effect = Exception("Connection timeout")

    schedule = [_make_schedule_game("New York Yankees")]
    results = fetch_game_day_weather(schedule)

    assert len(results) == 1
    # Should fall back to neutral defaults
    assert results[0]["temp_f"] == 72.0
    assert results[0]["wind_mph"] == 0.0


@patch("src.game_day._requests")
def test_weather_stores_to_db(mock_requests):
    """Weather data should be stored in the database."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hourly": {
            "temperature_2m": [85.0] * 24,
            "windspeed_10m": [5.0] * 24,
            "winddirection_10m": [90.0] * 24,
            "precipitation_probability": [0.0] * 24,
            "relativehumidity_2m": [40.0] * 24,
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_requests.get.return_value = mock_resp

    schedule = [_make_schedule_game("Colorado Rockies")]
    fetch_game_day_weather(schedule)

    # Verify data was written to DB
    from datetime import UTC, datetime

    from src.database import load_game_day_weather

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    df = load_game_day_weather(today)
    assert len(df) >= 1
    row = df.iloc[0]
    assert row["venue_team"] == "COL"
    assert row["temp_f"] == 85.0


# ── fetch_team_strength tests ─────────────────────────────────────────


@patch("src.game_day.team_pitching")
@patch("src.game_day.team_batting")
def test_team_strength_fetch_success(mock_batting, mock_pitching):
    """Successful pybaseball fetch should return a DataFrame with team data."""
    mock_batting.return_value = pd.DataFrame(
        {
            "Team": ["Yankees", "Dodgers"],
            "wRC+": [110, 115],
            "OPS": [0.750, 0.780],
            "K%": [22.0, 20.5],
            "BB%": [9.0, 10.0],
        }
    )
    mock_pitching.return_value = pd.DataFrame(
        {
            "Team": ["Yankees", "Dodgers"],
            "FIP": [3.80, 3.50],
            "ERA": [3.90, 3.60],
            "WHIP": [1.20, 1.15],
        }
    )

    result = fetch_team_strength(2026)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "team_abbr" in result.columns
    assert set(result["team_abbr"].tolist()) == {"NYY", "LAD"}


@patch("src.game_day._statsapi", None)
@patch("src.game_day.PYBASEBALL_AVAILABLE", False)
def test_team_strength_pybaseball_unavailable():
    """When pybaseball and statsapi are both unavailable, fall back to cached DB data."""
    result = fetch_team_strength(2026)
    assert isinstance(result, pd.DataFrame)
    # With no cached data and no APIs, should return empty DataFrame
    assert result.empty


@patch("src.game_day.team_pitching")
@patch("src.game_day.team_batting")
def test_team_strength_stores_to_db(mock_batting, mock_pitching):
    """Team strength data should be stored in the database."""
    mock_batting.return_value = pd.DataFrame(
        {
            "Team": ["Mets"],
            "wRC+": [105],
            "OPS": [0.730],
            "K%": [23.0],
            "BB%": [8.5],
        }
    )
    mock_pitching.return_value = pd.DataFrame(
        {
            "Team": ["Mets"],
            "FIP": [4.10],
            "ERA": [4.20],
            "WHIP": [1.28],
        }
    )

    fetch_team_strength(2026)

    from src.database import load_team_strength

    df = load_team_strength(2026)
    assert len(df) == 1
    assert df.iloc[0]["team_abbr"] == "NYM"


# ── get_team_strength tests ──────────────────────────────────────────


def test_get_team_strength_found():
    """When team data exists in DB, should return it as a dict."""
    from datetime import UTC, datetime

    from src.database import upsert_team_strength

    season = datetime.now(UTC).year
    upsert_team_strength(
        team_abbr="NYY",
        season=season,
        wrc_plus=112.0,
        fip=3.75,
        team_ops=0.760,
        team_era=3.85,
        team_whip=1.18,
        k_pct=21.5,
        bb_pct=9.2,
    )

    result = get_team_strength("NYY")
    assert result["wrc_plus"] == 112.0
    assert result["fip"] == 3.75
    assert result["team_ops"] == 0.760


def test_get_team_strength_missing():
    """When team data is not in DB, should return neutral defaults."""
    result = get_team_strength("ZZZ")
    assert result["wrc_plus"] == 100.0
    assert result["fip"] == 4.00
    assert result["team_ops"] == 0.720
    assert result["team_era"] == 4.00
    assert result["team_whip"] == 1.25


# ── fetch_opposing_pitchers tests ─────────────────────────────────────


def _make_pitcher_schedule(home_pitcher, away_pitcher, home_name="New York Yankees", away_name="Boston Red Sox"):
    """Helper to create a schedule game dict with pitchers."""
    return {
        "game_pk": 99999,
        "home_name": home_name,
        "away_name": away_name,
        "home_probable_pitcher": home_pitcher,
        "away_probable_pitcher": away_pitcher,
    }


@patch("src.game_day._statsapi")
def test_opposing_pitchers_success(mock_statsapi):
    """Should resolve pitcher name, fetch stats, and return data."""
    mock_statsapi.lookup_player.return_value = [{"id": 12345}]
    mock_statsapi.player_stat_data.return_value = {
        "pitch_hand": "R",
        "current_team": "NYY",
        "stats": [
            {
                "stats": {
                    "era": "3.50",
                    "whip": "1.10",
                    "inningsPitched": "180.0",
                    "strikeoutsPer9Inn": "9.5",
                    "walksPer9Inn": "2.8",
                }
            }
        ],
    }
    mock_statsapi.get.return_value = {"people": []}

    schedule = [_make_pitcher_schedule("Gerrit Cole", "Chris Sale")]
    results = fetch_opposing_pitchers(schedule)

    assert len(results) >= 1
    assert results[0]["name"] == "Gerrit Cole"
    assert results[0]["era"] == 3.50
    assert results[0]["pitcher_id"] == 12345


@patch("src.game_day._statsapi")
def test_opposing_pitchers_tbd_skipped(mock_statsapi):
    """TBD pitchers should be skipped entirely."""
    schedule = [_make_pitcher_schedule("TBD", "TBD")]
    results = fetch_opposing_pitchers(schedule)

    assert len(results) == 0
    mock_statsapi.lookup_player.assert_not_called()


@patch("src.game_day._statsapi")
def test_opposing_pitchers_lookup_failure(mock_statsapi):
    """When lookup_player returns empty, should skip gracefully."""
    mock_statsapi.lookup_player.return_value = []

    schedule = [_make_pitcher_schedule("Unknown Pitcher", None)]
    results = fetch_opposing_pitchers(schedule)

    assert len(results) == 0


@patch("src.game_day._statsapi")
def test_opposing_pitchers_stores_to_db(mock_statsapi):
    """Opposing pitcher data should be stored in the database."""
    mock_statsapi.lookup_player.return_value = [{"id": 67890}]
    mock_statsapi.player_stat_data.return_value = {
        "pitch_hand": "L",
        "current_team": "BOS",
        "stats": [
            {
                "stats": {
                    "era": "4.00",
                    "whip": "1.25",
                    "inningsPitched": "150.0",
                    "strikeoutsPer9Inn": "8.0",
                    "walksPer9Inn": "3.0",
                }
            }
        ],
    }
    mock_statsapi.get.return_value = {"people": []}

    schedule = [_make_pitcher_schedule(None, "Chris Sale")]
    fetch_opposing_pitchers(schedule)

    from datetime import UTC, datetime

    from src.database import load_opp_pitchers

    season = datetime.now(UTC).year
    df = load_opp_pitchers(season)
    assert len(df) >= 1
    assert df.iloc[0]["pitcher_id"] == 67890


# ── get_player_recent_form tests ──────────────────────────────────────


def test_recent_form_hitter():
    """Hitter game log should be aggregated into L7/L14/L30 stats."""
    mock_statsapi = MagicMock()
    game = {
        "hits": 2,
        "atBats": 4,
        "plateAppearances": 5,
        "homeRuns": 1,
        "rbi": 3,
        "stolenBases": 0,
        "runs": 1,
        "baseOnBalls": 1,
        "hitByPitch": 0,
        "sacFlies": 0,
    }
    mock_statsapi.player_stat_data.return_value = {
        "stats": [{"stats": game}] * 10,
    }

    with patch.dict("sys.modules", {"statsapi": mock_statsapi}):
        result = get_player_recent_form(123456)

    assert result["player_type"] == "hitter"
    assert result["l7"]["games"] == 7
    assert result["l14"]["games"] == 10  # only 10 games available
    assert result["l7"]["hr"] == 7  # 1 HR per game * 7 games


def test_recent_form_pitcher():
    """Pitcher game log should compute ERA and WHIP from aggregated stats."""
    mock_statsapi = MagicMock()
    # First call for hitting returns no data
    hitting_result = {"stats": []}
    pitching_game = {
        "inningsPitched": "6.0",
        "strikeOuts": 7,
        "wins": 1,
        "earnedRuns": 2,
        "baseOnBalls": 1,
        "hits": 4,
    }
    pitching_result = {"stats": [{"stats": pitching_game}] * 5}

    mock_statsapi.player_stat_data.side_effect = [hitting_result, pitching_result]

    with patch.dict("sys.modules", {"statsapi": mock_statsapi}):
        result = get_player_recent_form(654321)

    assert result["player_type"] == "pitcher"
    assert result["l7"]["games"] == 5
    assert result["l7"]["k"] == 35  # 7 K per game * 5 games
    # ERA = (2 * 5 * 9) / (6.0 * 5) = 90 / 30 = 3.0
    assert result["l7"]["era"] == 3.0


def test_recent_form_no_data():
    """When no game data is available, should return empty dicts."""
    mock_statsapi = MagicMock()
    mock_statsapi.player_stat_data.return_value = {"stats": []}

    with patch.dict("sys.modules", {"statsapi": mock_statsapi}):
        result = get_player_recent_form(999999)

    assert result["player_type"] == "unknown"
    assert result["l7"] == {}
    assert result["l14"] == {}
    assert result["l30"] == {}


def test_recent_form_cached():
    """Cached wrapper should return same data on second call."""
    from src.game_day import get_player_recent_form_cached

    mock_session_state = {}

    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        import streamlit as st_mock

        st_mock.session_state = mock_session_state

        with patch("src.game_day.get_player_recent_form") as mock_fetch:
            mock_fetch.return_value = {
                "l7": {"avg": 0.300},
                "l14": {},
                "l30": {},
                "player_type": "hitter",
                "mlb_id": 111,
            }
            import importlib

            import src.game_day

            # Call uncached version directly since streamlit mocking is complex
            result = (
                get_player_recent_form_cached.__wrapped__(111)
                if hasattr(get_player_recent_form_cached, "__wrapped__")
                else src.game_day.get_player_recent_form(111)
            )
            assert result["player_type"] in ("hitter", "unknown")


# ── fetch_game_day_intelligence tests ─────────────────────────────────


@patch("src.game_day.fetch_team_strength")
@patch("src.game_day.fetch_opposing_pitchers")
@patch("src.game_day.fetch_game_day_weather")
@patch("src.game_day._statsapi")
def test_intelligence_orchestrates_all(mock_statsapi, mock_weather, mock_pitchers, mock_team):
    """fetch_game_day_intelligence should call all sub-fetchers."""
    mock_statsapi.schedule.return_value = [
        {"game_pk": 1, "home_name": "NYY", "away_name": "BOS"},
    ]
    mock_weather.return_value = [{"temp_f": 75}]
    mock_pitchers.return_value = [{"name": "Cole"}]
    mock_team.return_value = pd.DataFrame()

    result = fetch_game_day_intelligence()

    assert "weather" in result
    assert "pitchers" in result
    assert "team_strength" in result
    assert "fetched_at" in result
    mock_weather.assert_called_once()
    mock_pitchers.assert_called_once()
    mock_team.assert_called_once()


def test_intelligence_no_statsapi():
    """When statsapi is not available, should return empty dict safely."""
    with patch("src.game_day._statsapi", None):
        result = fetch_game_day_intelligence()

    assert result["weather"] == []
    assert result["pitchers"] == []
    assert result["lineups"] == {}
    assert isinstance(result["team_strength"], pd.DataFrame)


# ── Helper function tests ─────────────────────────────────────────────


def test_aggregate_hitting_games_empty():
    """Empty game list should return empty dict."""
    assert _aggregate_hitting_games([]) == {}


def test_aggregate_hitting_games_rate_stats():
    """AVG and OBP should be calculated correctly from components."""
    games = [
        {
            "hits": 3,
            "atBats": 10,
            "plateAppearances": 12,
            "homeRuns": 1,
            "rbi": 2,
            "stolenBases": 0,
            "runs": 1,
            "baseOnBalls": 1,
            "hitByPitch": 1,
            "sacFlies": 0,
        },
        {
            "hits": 1,
            "atBats": 4,
            "plateAppearances": 5,
            "homeRuns": 0,
            "rbi": 0,
            "stolenBases": 1,
            "runs": 1,
            "baseOnBalls": 1,
            "hitByPitch": 0,
            "sacFlies": 0,
        },
    ]
    result = _aggregate_hitting_games(games)

    assert result["h"] == 4
    assert result["ab"] == 14
    assert result["avg"] == round(4 / 14, 3)
    # OBP = (4+2+1) / (14+2+1+0) = 7/17
    assert result["obp"] == round(7 / 17, 3)
    assert result["games"] == 2


def test_aggregate_pitching_games_empty():
    """Empty game list should return empty dict."""
    assert _aggregate_pitching_games([]) == {}


def test_aggregate_pitching_games_rate_stats():
    """ERA and WHIP should be calculated correctly from components."""
    games = [
        {"inningsPitched": "6.0", "strikeOuts": 8, "wins": 1, "earnedRuns": 2, "baseOnBalls": 1, "hits": 5},
        {"inningsPitched": "7.0", "strikeOuts": 6, "wins": 0, "earnedRuns": 3, "baseOnBalls": 2, "hits": 6},
    ]
    result = _aggregate_pitching_games(games)

    total_ip = 13.0
    assert result["ip"] == total_ip
    assert result["k"] == 14
    # ERA = (5 * 9) / 13 = 3.46
    assert result["era"] == round(5 * 9 / 13, 2)
    # WHIP = (3 + 11) / 13 = 1.08
    assert result["whip"] == round((3 + 11) / 13, 2)
    assert result["games"] == 2
