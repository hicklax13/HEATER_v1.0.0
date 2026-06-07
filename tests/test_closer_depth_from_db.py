"""DB-C2 / DB-E1: Closer Monitor wires REAL depth_chart_role data from the DB.

The page previously read ``st.session_state['closer_depth_data']`` — a key nothing
ever wrote — and always fell back to a "top season-SV pitcher per team = closer"
heuristic. The real bullpen-role data (``players.depth_chart_role`` ∈
{starter, bullpen, closer, setup, committee}, persisted by
``_persist_depth_chart_roles``) was never surfaced to the closer grid.

``build_depth_data_from_db`` produces the canonical ``build_closer_grid`` input
shape ``{team: {"closer", "setup", "closer_confidence"}}`` straight from the DB.
"""

from __future__ import annotations

import pytest

from src.closer_monitor import build_closer_grid, build_depth_data_from_db
from src.database import get_connection, init_db, upsert_player_bulk


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Fresh temp DB per test."""
    db_path = tmp_path / "test_draft_tool.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    init_db()
    return db_path


def _set_role(name: str, role: str) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE players SET depth_chart_role = ? WHERE name = ?", (role, name))
        conn.commit()
    finally:
        conn.close()


def _set_saves(name: str, sv: int, season: int = 2026) -> None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT player_id FROM players WHERE name = ?", (name,)).fetchone()
        pid = row[0]
        conn.execute(
            "INSERT INTO season_stats (player_id, season, sv) VALUES (?, ?, ?)",
            (pid, season, sv),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_depth_data_from_db_returns_canonical_shape():
    """Helper returns {team: {closer, setup, closer_confidence}} from depth_chart_role."""
    upsert_player_bulk(
        [
            {"name": "Edwin Diaz", "team": "NYM", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("Edwin Diaz", "closer")
    _set_saves("Edwin Diaz", 20)

    depth = build_depth_data_from_db()
    assert "NYM" in depth
    entry = depth["NYM"]
    assert entry["closer"] == "Edwin Diaz"
    assert isinstance(entry["setup"], list)
    assert 0.0 <= entry["closer_confidence"] <= 1.0


def test_build_depth_data_uses_role_not_top_sv():
    """The closer is the depth_chart_role='closer' player, NOT the highest-SV pitcher.

    This is the core DB-C2 fix: a low-SV designated closer must beat a high-SV
    non-closer arm on the same team.
    """
    upsert_player_bulk(
        [
            {"name": "Designated Closer", "team": "AAA", "positions": "RP", "is_hitter": False},
            {"name": "High SV Mopup", "team": "AAA", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("Designated Closer", "closer")
    _set_role("High SV Mopup", "bullpen")
    _set_saves("Designated Closer", 5)
    _set_saves("High SV Mopup", 30)  # more saves, but not the designated closer

    depth = build_depth_data_from_db()
    assert depth["AAA"]["closer"] == "Designated Closer"


def test_build_depth_data_setup_list_reflects_setup_roles():
    """setup-role players land in the setup list (not as the closer)."""
    upsert_player_bulk(
        [
            {"name": "The Closer", "team": "BBB", "positions": "RP", "is_hitter": False},
            {"name": "Setup One", "team": "BBB", "positions": "RP", "is_hitter": False},
            {"name": "Setup Two", "team": "BBB", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("The Closer", "closer")
    _set_role("Setup One", "setup")
    _set_role("Setup Two", "setup")

    depth = build_depth_data_from_db()
    entry = depth["BBB"]
    assert entry["closer"] == "The Closer"
    assert set(entry["setup"]) == {"Setup One", "Setup Two"}


def test_multiple_closers_demote_lower_sv_to_setup():
    """When a team has 2 closer-role rows (statsapi top-2-by-saves heuristic),
    the higher-saves arm is THE closer and the other is demoted to setup."""
    upsert_player_bulk(
        [
            {"name": "Primary", "team": "CCC", "positions": "RP", "is_hitter": False},
            {"name": "Secondary", "team": "CCC", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("Primary", "closer")
    _set_role("Secondary", "closer")
    _set_saves("Primary", 18)
    _set_saves("Secondary", 4)

    depth = build_depth_data_from_db()
    entry = depth["CCC"]
    assert entry["closer"] == "Primary"
    assert "Secondary" in entry["setup"]


def test_committee_role_flagged():
    """A committee-role team surfaces a Committee marker with lower confidence."""
    upsert_player_bulk(
        [
            {"name": "Comm One", "team": "DDD", "positions": "RP", "is_hitter": False},
            {"name": "Comm Two", "team": "DDD", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("Comm One", "committee")
    _set_role("Comm Two", "committee")

    depth = build_depth_data_from_db()
    entry = depth["DDD"]
    assert entry.get("committee") is True
    grid = build_closer_grid(depth)
    # committee → low/uncertain job security
    assert grid[0]["job_security"] < 0.6


def test_no_role_data_returns_empty():
    """No depth_chart_role rows → empty dict so the page can fall back to heuristic."""
    upsert_player_bulk(
        [
            {"name": "Some Reliever", "team": "EEE", "positions": "RP", "is_hitter": False},
        ]
    )
    # No role set → nothing to surface
    depth = build_depth_data_from_db()
    assert depth == {}


def test_grid_uses_real_closer_when_role_present():
    """End-to-end: build_closer_grid on DB-derived depth data names the role closer."""
    upsert_player_bulk(
        [
            {"name": "Role Closer", "team": "FFF", "positions": "RP", "is_hitter": False},
            {"name": "Big Saves Guy", "team": "FFF", "positions": "RP", "is_hitter": False},
        ]
    )
    _set_role("Role Closer", "closer")
    _set_role("Big Saves Guy", "bullpen")
    _set_saves("Role Closer", 8)
    _set_saves("Big Saves Guy", 40)

    depth = build_depth_data_from_db()
    grid = build_closer_grid(depth)
    closers = {item["team"]: item["closer_name"] for item in grid}
    assert closers["FFF"] == "Role Closer"
