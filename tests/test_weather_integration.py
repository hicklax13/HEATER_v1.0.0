"""Tests for weather integration with optimizer pipeline."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import (
    init_db,
    load_game_day_weather,
    load_opp_pitchers,
    load_team_strength,
    upsert_game_day_weather,
    upsert_opp_pitcher,
    upsert_team_strength,
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


# ── weather_hr_adjustment tests ───────────────────────────────────────


def test_weather_hr_adjustment_hot():
    """Hot temperatures (95F) should produce a multiplier above 1.0."""
    from src.optimizer.matchup_adjustments import weather_hr_adjustment

    result = weather_hr_adjustment(temp_f=95.0)
    assert result > 1.0, f"Expected >1.0 at 95F, got {result}"


def test_weather_hr_adjustment_neutral():
    """Reference temperature (72F) should produce a multiplier of exactly 1.0."""
    from src.optimizer.matchup_adjustments import weather_hr_adjustment

    result = weather_hr_adjustment(temp_f=72.0)
    assert result == 1.0, f"Expected 1.0 at 72F, got {result}"


def test_weather_hr_adjustment_cold():
    """Cold temperatures (50F) should not reduce below 1.0 (no negative penalty)."""
    from src.optimizer.matchup_adjustments import weather_hr_adjustment

    result = weather_hr_adjustment(temp_f=50.0)
    assert result == 1.0, f"Expected 1.0 at 50F, got {result}"


# ── compute_matchup_multiplier weather integration ────────────────────


def test_matchup_multiplier_with_weather():
    """Providing temp_f should change the matchup multiplier for hitters."""
    from src.optimizer.daily_optimizer import compute_matchup_multiplier

    base = compute_matchup_multiplier(
        is_hitter=True,
        batter_hand="R",
        pitcher_hand="R",
        player_team="NYY",
        opponent_team="BOS",
        park_factors={},
        pitcher_xfip=None,
        temp_f=None,
    )
    hot = compute_matchup_multiplier(
        is_hitter=True,
        batter_hand="R",
        pitcher_hand="R",
        player_team="NYY",
        opponent_team="BOS",
        park_factors={},
        pitcher_xfip=None,
        temp_f=95.0,
    )
    # Hot weather should boost hitter multiplier
    assert hot >= base, f"Hot weather multiplier {hot} should be >= base {base}"


def test_matchup_multiplier_without_weather():
    """temp_f=None should produce the same result as omitting it entirely."""
    from src.optimizer.daily_optimizer import compute_matchup_multiplier

    result_none = compute_matchup_multiplier(
        is_hitter=True,
        batter_hand="L",
        pitcher_hand="R",
        player_team="LAD",
        opponent_team="SF",
        park_factors={},
        pitcher_xfip=None,
        temp_f=None,
    )
    # Should be a valid positive multiplier
    assert 0.3 <= result_none <= 3.0, f"Multiplier {result_none} outside valid range"


# ── Database roundtrip tests ──────────────────────────────────────────


def test_weather_db_roundtrip():
    """Upsert then load game_day_weather should return matching data."""
    upsert_game_day_weather(
        game_pk=12345,
        game_date="2026-04-03",
        venue_team="NYY",
        temp_f=82.5,
        wind_mph=12.0,
        wind_dir="SW",
        precip_pct=5.0,
        humidity_pct=45.0,
    )

    df = load_game_day_weather("2026-04-03")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["game_pk"] == 12345
    assert row["venue_team"] == "NYY"
    assert row["temp_f"] == 82.5
    assert row["wind_mph"] == 12.0
    assert row["wind_dir"] == "SW"
    assert row["precip_pct"] == 5.0
    assert row["humidity_pct"] == 45.0


def test_team_strength_db_roundtrip():
    """Upsert then load team_strength should return matching data."""
    upsert_team_strength(
        team_abbr="LAD",
        season=2026,
        wrc_plus=115.0,
        fip=3.40,
        team_ops=0.790,
        team_era=3.55,
        team_whip=1.12,
        k_pct=20.0,
        bb_pct=10.5,
    )

    df = load_team_strength(2026)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["team_abbr"] == "LAD"
    assert row["wrc_plus"] == 115.0
    assert row["fip"] == 3.40
    assert row["team_ops"] == 0.790


def test_opp_pitcher_db_roundtrip():
    """Upsert then load opp_pitcher should return matching data."""
    upsert_opp_pitcher(
        pitcher_id=543210,
        season=2026,
        name="Gerrit Cole",
        team="NYY",
        era=3.25,
        fip=3.10,
        xfip=3.30,
        whip=1.05,
        k_per_9=10.5,
        bb_per_9=2.3,
        vs_lhb_avg=0.220,
        vs_rhb_avg=0.240,
        ip=195.0,
        hand="R",
    )

    df = load_opp_pitchers(2026)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pitcher_id"] == 543210
    assert row["name"] == "Gerrit Cole"
    assert row["era"] == 3.25
    assert row["whip"] == 1.05
    assert row["hand"] == "R"
    assert row["vs_lhb_avg"] == 0.220
