# tests/test_schedule_grid.py
"""Tests for 7-day schedule grid."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pandas as pd

from src.schedule_grid import TIER_COLORS, build_schedule_grid, render_schedule_html


def _make_roster(n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "player_id": i + 1,
                "name": f"Player {i + 1}",
                "team": "NYY" if i % 2 == 0 else "BOS",
                "positions": "SS" if i % 2 == 0 else "OF",
                "is_hitter": True,
            }
        )
    return pd.DataFrame(rows)


def test_grid_structure():
    roster = _make_roster()
    grid = build_schedule_grid(roster)
    assert "dates" in grid
    assert "day_labels" in grid
    assert "players" in grid
    assert "games_per_day" in grid
    assert len(grid["dates"]) == 7
    assert len(grid["day_labels"]) == 7
    assert len(grid["games_per_day"]) == 7


def test_grid_player_count():
    roster = _make_roster(3)
    grid = build_schedule_grid(roster)
    assert len(grid["players"]) == 3


def test_grid_off_days_no_schedule():
    """Without schedule data, all days should be off days."""
    roster = _make_roster(1)
    grid = build_schedule_grid(roster)
    for day in grid["players"][0]["days"]:
        assert day["has_game"] is False
        assert day["opponent"] is None


def test_grid_with_games():
    roster = _make_roster(1)
    # Monday game for NYY
    start = datetime(2026, 3, 16, tzinfo=UTC)  # a Monday
    schedule = [
        {"date": "2026-03-16", "home_team": "NYY", "away_team": "BOS"},
    ]
    grid = build_schedule_grid(roster, weekly_schedule=schedule, start_date=start)
    player = grid["players"][0]
    # First day (Monday) should have a game
    assert player["days"][0]["has_game"] is True
    assert player["days"][0]["opponent"] == "BOS"
    assert player["days"][0]["is_home"] is True


def test_games_per_day_count():
    roster = _make_roster(2)
    start = datetime(2026, 3, 16, tzinfo=UTC)
    schedule = [
        {"date": "2026-03-16", "home_team": "NYY", "away_team": "BOS"},
    ]
    grid = build_schedule_grid(roster, weekly_schedule=schedule, start_date=start)
    # Both NYY and BOS play on Monday
    assert grid["games_per_day"][0] == 2


def test_tier_colors_complete():
    for tier in ["smash", "favorable", "neutral", "unfavorable", "avoid"]:
        assert tier in TIER_COLORS


def test_grid_with_matchup_ratings():
    roster = _make_roster(1)
    ratings = pd.DataFrame([{"player_id": 1, "matchup_tier": "smash"}])
    grid = build_schedule_grid(roster, matchup_ratings=ratings)
    # Even without games, the tier should be stored
    assert grid["players"][0]["days"][0]["tier"] is None  # No game = no tier


def test_render_html_empty():
    html = render_schedule_html({})
    assert "No schedule" in html


def test_render_html_basic():
    roster = _make_roster(1)
    start = datetime(2026, 3, 16, tzinfo=UTC)
    schedule = [
        {"date": "2026-03-16", "home_team": "NYY", "away_team": "BOS"},
    ]
    grid = build_schedule_grid(roster, weekly_schedule=schedule, start_date=start)
    html = render_schedule_html(grid)
    assert "<table" in html
    assert "Player 1" in html
    assert "BOS" in html


def test_empty_roster():
    grid = build_schedule_grid(pd.DataFrame())
    assert grid["players"] == []
    assert all(c == 0 for c in grid["games_per_day"])
