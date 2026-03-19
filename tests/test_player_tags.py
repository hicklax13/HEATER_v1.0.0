# tests/test_player_tags.py
"""Tests for player tag management."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import src.database as db_mod
from src.database import init_db


@pytest.fixture
def _temp_db():
    """Create a temporary database for testing."""
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
        pass  # Windows may hold the file lock


def test_add_tag_success(_temp_db):
    from src.player_tags import add_tag

    assert add_tag(1, "Sleeper", "breakout candidate") is True


def test_add_duplicate_tag(_temp_db):
    from src.player_tags import add_tag

    add_tag(1, "Sleeper")
    assert add_tag(1, "Sleeper") is False


def test_add_invalid_tag(_temp_db):
    from src.player_tags import add_tag

    assert add_tag(1, "InvalidTag") is False


def test_remove_tag(_temp_db):
    from src.player_tags import add_tag, remove_tag

    add_tag(1, "Target")
    assert remove_tag(1, "Target") is True


def test_remove_nonexistent(_temp_db):
    from src.player_tags import remove_tag

    assert remove_tag(999, "Bust") is False


def test_get_tags_empty(_temp_db):
    from src.player_tags import get_tags

    assert get_tags(999) == []


def test_get_tags_multiple(_temp_db):
    from src.player_tags import add_tag, get_tags

    add_tag(1, "Sleeper")
    add_tag(1, "Target")
    tags = get_tags(1)
    assert len(tags) == 2
    assert {t["tag"] for t in tags} == {"Sleeper", "Target"}


def test_get_all_tagged_no_filter(_temp_db):
    from src.player_tags import add_tag, get_all_tagged_players

    add_tag(1, "Sleeper")
    add_tag(2, "Avoid")
    df = get_all_tagged_players()
    assert len(df) == 2


def test_get_all_tagged_with_filter(_temp_db):
    from src.player_tags import add_tag, get_all_tagged_players

    add_tag(1, "Sleeper")
    add_tag(2, "Avoid")
    df = get_all_tagged_players("Sleeper")
    assert len(df) == 1


def test_valid_tags_complete():
    from src.player_tags import TAG_COLORS, VALID_TAGS

    for tag in VALID_TAGS:
        assert tag in TAG_COLORS


def test_render_tag_badges_html():
    from src.player_tags import render_tag_badges_html

    html = render_tag_badges_html([{"tag": "Sleeper"}, {"tag": "Avoid"}])
    assert "Sleeper" in html
    assert "Avoid" in html


def test_render_empty_badges():
    from src.player_tags import render_tag_badges_html

    assert render_tag_badges_html([]) == ""
