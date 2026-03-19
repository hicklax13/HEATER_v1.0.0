"""Tests for multi-league registry CRUD operations."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import src.database as database_module
from src.database import init_db
from src.league_registry import (
    LeagueInfo,
    delete_league,
    get_active_league_id,
    get_league,
    list_leagues,
    register_league,
    set_active_league,
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path):
    """Redirect DB_PATH to a temp directory for every test."""
    db_path = tmp_path / "test_leagues.db"
    with patch.object(database_module, "DB_PATH", db_path):
        init_db()
        yield


# ── register_league ────────────────────────────────────────────────


def test_register_returns_uuid_string():
    league_id = register_league(league_name="Test League")
    # Should be a valid UUID4
    parsed = uuid.UUID(league_id, version=4)
    assert str(parsed) == league_id


def test_first_league_becomes_active():
    league_id = register_league(league_name="First")
    info = get_league(league_id)
    assert info is not None
    assert info.is_active is True


def test_second_league_is_not_active():
    register_league(league_name="First")
    second_id = register_league(league_name="Second")
    info = get_league(second_id)
    assert info is not None
    assert info.is_active is False


def test_register_with_yahoo_league_id():
    league_id = register_league(
        platform="yahoo",
        league_name="Yahoo League",
        yahoo_league_id="469.l.12345",
    )
    info = get_league(league_id)
    assert info is not None
    assert info.platform == "yahoo"
    assert info.yahoo_league_id == "469.l.12345"


def test_register_with_custom_platform_and_scoring():
    league_id = register_league(
        platform="espn",
        league_name="ESPN Points",
        num_teams=10,
        scoring_format="points",
    )
    info = get_league(league_id)
    assert info is not None
    assert info.platform == "espn"
    assert info.num_teams == 10
    assert info.scoring_format == "points"


# ── get_league ─────────────────────────────────────────────────────


def test_get_league_returns_correct_info():
    league_id = register_league(league_name="My League", num_teams=14)
    info = get_league(league_id)
    assert isinstance(info, LeagueInfo)
    assert info.league_id == league_id
    assert info.league_name == "My League"
    assert info.num_teams == 14
    assert info.scoring_format == "h2h_categories"
    assert info.created_at  # Non-empty timestamp


def test_get_league_returns_none_for_missing():
    result = get_league("nonexistent-id")
    assert result is None


# ── list_leagues ───────────────────────────────────────────────────


def test_list_leagues_returns_all_registered():
    register_league(league_name="League A")
    register_league(league_name="League B")
    register_league(league_name="League C")
    leagues = list_leagues()
    assert len(leagues) == 3
    names = [lg.league_name for lg in leagues]
    assert "League A" in names
    assert "League B" in names
    assert "League C" in names


def test_list_leagues_empty_returns_empty_list():
    leagues = list_leagues()
    assert leagues == []


# ── set_active_league ──────────────────────────────────────────────


def test_set_active_league_switches_active():
    first_id = register_league(league_name="First")
    second_id = register_league(league_name="Second")

    # First is active by default
    assert get_league(first_id).is_active is True
    assert get_league(second_id).is_active is False

    # Switch to second
    result = set_active_league(second_id)
    assert result is True
    assert get_league(first_id).is_active is False
    assert get_league(second_id).is_active is True


def test_set_active_league_returns_false_for_missing():
    result = set_active_league("nonexistent-id")
    assert result is False


# ── get_active_league_id ───────────────────────────────────────────


def test_get_active_league_id_returns_default_when_empty():
    active = get_active_league_id()
    assert active == "default"


def test_get_active_league_id_returns_active_league():
    league_id = register_league(league_name="Active League")
    active = get_active_league_id()
    assert active == league_id


# ── delete_league ──────────────────────────────────────────────────


def test_delete_league_removes_league():
    league_id = register_league(league_name="To Delete")
    assert get_league(league_id) is not None

    result = delete_league(league_id)
    assert result is True
    assert get_league(league_id) is None


def test_delete_league_returns_false_for_missing():
    result = delete_league("nonexistent-id")
    assert result is False
