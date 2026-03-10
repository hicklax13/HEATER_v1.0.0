"""Test league manager: CSV import, roster management, free agents."""

import csv
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.league_manager import (
    import_league_rosters_csv,
    import_standings_csv,
    get_free_agents,
    get_team_roster,
    add_player_to_roster,
    remove_player_from_roster,
)


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (2, 'Shohei Ohtani', 'LAD', 'DH', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (3, 'Trea Turner', 'PHI', 'SS', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (4, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (5, 'Free Agent Guy', 'FA', 'OF', 1)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture
def roster_csv(tmp_path):
    csv_path = tmp_path / "rosters.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["team_name", "player_name", "position", "roster_slot"])
        writer.writerow(["Team Hickey", "Aaron Judge", "OF", "OF"])
        writer.writerow(["Team Hickey", "Shohei Ohtani", "DH", "Util"])
        writer.writerow(["Team 2", "Trea Turner", "SS", "SS"])
        writer.writerow(["Team 2", "Gerrit Cole", "SP", "SP"])
    return str(csv_path)


@pytest.fixture
def standings_csv(tmp_path):
    csv_path = tmp_path / "standings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "team_name", "R", "HR", "RBI", "SB", "AVG",
            "W", "SV", "K", "ERA", "WHIP",
        ])
        writer.writerow([
            "Team Hickey", "450", "120", "430", "65", ".272",
            "45", "30", "650", "3.50", "1.15",
        ])
        writer.writerow([
            "Team 2", "420", "110", "410", "70", ".265",
            "50", "35", "700", "3.70", "1.20",
        ])
    return str(csv_path)


def test_import_rosters_csv(roster_csv, temp_db):
    count = import_league_rosters_csv(roster_csv, user_team_name="Team Hickey")
    assert count == 4
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 2


def test_import_standings_csv(standings_csv, temp_db):
    count = import_standings_csv(standings_csv)
    assert count == 2


def test_free_agents(roster_csv, temp_db):
    import_league_rosters_csv(roster_csv, user_team_name="Team Hickey")
    all_players = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "name": [
            "Aaron Judge", "Shohei Ohtani", "Trea Turner",
            "Gerrit Cole", "Free Agent Guy",
        ],
    })
    fa = get_free_agents(all_players)
    assert len(fa) == 1
    assert fa.iloc[0]["name"] == "Free Agent Guy"


def test_add_remove_player(temp_db):
    add_player_to_roster("Team Hickey", 0, 3, "SS", is_user_team=True)
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 1

    remove_player_from_roster("Team Hickey", 3)
    roster = get_team_roster("Team Hickey")
    assert len(roster) == 0
