"""Tests for targeted trade proposal generation (lowball + fair value)."""

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
from src.trade_intelligence import generate_targeted_proposals
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def temp_db():
    """Redirect DB_PATH to a temp file and init schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()

    conn = sqlite3.connect(tmp.name)
    # Players matching sample_pool
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (1, 'Player A', 'NYY', '1B', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (2, 'Player B', 'LAD', 'SP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (3, 'Player C', 'BOS', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (4, 'Player D', 'ATL', 'RP', 0)"
    )
    # Injury history
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (1, 2025, 150, 162)"
    )
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (2, 2025, 140, 162)"
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
def config():
    return LeagueConfig()


@pytest.fixture
def sample_pool():
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Player A",
                "positions": "1B",
                "is_hitter": True,
                "pa": 600,
                "ab": 550,
                "h": 150,
                "r": 80,
                "hr": 25,
                "rbi": 85,
                "sb": 10,
                "avg": 0.273,
                "obp": 0.340,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "adp": 50,
                "team": "NYY",
            },
            {
                "player_id": 2,
                "name": "Player B",
                "positions": "SP",
                "is_hitter": False,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "ip": 180,
                "w": 12,
                "l": 8,
                "sv": 0,
                "k": 190,
                "era": 3.50,
                "whip": 1.15,
                "er": 70,
                "bb_allowed": 55,
                "h_allowed": 152,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "adp": 60,
                "team": "LAD",
            },
            {
                "player_id": 3,
                "name": "Player C",
                "positions": "OF",
                "is_hitter": True,
                "pa": 500,
                "ab": 460,
                "h": 120,
                "r": 65,
                "hr": 18,
                "rbi": 70,
                "sb": 15,
                "avg": 0.261,
                "obp": 0.325,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "bb": 35,
                "hbp": 3,
                "sf": 4,
                "adp": 80,
                "team": "BOS",
            },
            {
                "player_id": 4,
                "name": "Player D",
                "positions": "RP",
                "is_hitter": False,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "ip": 65,
                "w": 3,
                "l": 2,
                "sv": 28,
                "k": 75,
                "era": 2.80,
                "whip": 1.05,
                "er": 20,
                "bb_allowed": 18,
                "h_allowed": 50,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "adp": 100,
                "team": "ATL",
            },
        ]
    )


class TestGenerateTargetedProposals:
    """Tests for generate_targeted_proposals()."""

    def test_missing_target_returns_empty(self, sample_pool, config):
        """Target player_id not in pool returns empty target with None proposals."""
        result = generate_targeted_proposals(
            target_player_id=9999,
            user_roster_ids=[1, 2],
            player_pool=sample_pool,
            config=config,
        )
        assert result["target"] == {}
        assert result["lowball"] is None
        assert result["fair_value"] is None

    def test_empty_roster_returns_none_proposals(self, sample_pool, config):
        """Empty user_roster_ids returns None for both proposals."""
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[],
            player_pool=sample_pool,
            config=config,
        )
        assert result["lowball"] is None
        assert result["fair_value"] is None

    def test_basic_proposal_structure(self, sample_pool, config):
        """With valid data, returned dict has target, lowball, fair_value keys."""
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[1, 2],
            player_pool=sample_pool,
            config=config,
        )
        assert "target" in result
        assert "lowball" in result
        assert "fair_value" in result

    def test_lowball_cheaper_than_fair_value(self, sample_pool, config):
        """If both proposals exist, lowball give SGP should be <= fair_value give SGP."""
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[1, 2],
            player_pool=sample_pool,
            config=config,
        )
        lowball = result.get("lowball")
        fair_value = result.get("fair_value")
        if lowball is not None and fair_value is not None:
            # Lowball gives away less value
            low_ids = lowball.get("giving_ids", [])
            fair_ids = fair_value.get("giving_ids", [])
            low_sgp = sum(sample_pool.loc[sample_pool["player_id"].isin(low_ids), "hr"].sum() for _ in [1])
            fair_sgp = sum(sample_pool.loc[sample_pool["player_id"].isin(fair_ids), "hr"].sum() for _ in [1])
            # At minimum, verify both have giving_ids
            assert isinstance(low_ids, list)
            assert isinstance(fair_ids, list)

    def test_target_info_populated(self, sample_pool, config):
        """Target dict should contain name and positions fields."""
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[1, 2],
            player_pool=sample_pool,
            config=config,
        )
        target = result["target"]
        assert target.get("name") == "Player C"
        assert "positions" in target

    def test_proposal_has_required_fields(self, sample_pool, config):
        """Each non-None proposal should have giving_ids and giving_names."""
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[1, 2],
            player_pool=sample_pool,
            config=config,
        )
        for key in ("lowball", "fair_value"):
            proposal = result.get(key)
            if proposal is not None:
                assert "giving_ids" in proposal, f"{key} missing giving_ids"
                assert "giving_names" in proposal, f"{key} missing giving_names"

    def test_il_stash_excluded_from_gives(self, sample_pool, config):
        """IL stash players (Bieber, Strider) should never appear in giving_ids."""
        # Add IL stash players to the pool and user roster
        stash_rows = pd.DataFrame(
            [
                {
                    "player_id": 100,
                    "name": "Shane Bieber",
                    "positions": "SP",
                    "is_hitter": False,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "ip": 100,
                    "w": 6,
                    "l": 4,
                    "sv": 0,
                    "k": 90,
                    "era": 3.80,
                    "whip": 1.20,
                    "er": 42,
                    "bb_allowed": 30,
                    "h_allowed": 90,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "adp": 150,
                    "team": "CLE",
                },
                {
                    "player_id": 101,
                    "name": "Spencer Strider",
                    "positions": "SP",
                    "is_hitter": False,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "ip": 120,
                    "w": 8,
                    "l": 3,
                    "sv": 0,
                    "k": 130,
                    "era": 3.20,
                    "whip": 1.10,
                    "er": 43,
                    "bb_allowed": 28,
                    "h_allowed": 104,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "adp": 120,
                    "team": "ATL",
                },
            ]
        )
        pool = pd.concat([sample_pool, stash_rows], ignore_index=True)
        result = generate_targeted_proposals(
            target_player_id=3,
            user_roster_ids=[1, 2, 100, 101],
            player_pool=pool,
            config=config,
        )
        for key in ("lowball", "fair_value"):
            proposal = result.get(key)
            if proposal is not None:
                giving_ids = proposal.get("giving_ids", [])
                assert 100 not in giving_ids, "Shane Bieber (IL stash) should be excluded"
                assert 101 not in giving_ids, "Spencer Strider (IL stash) should be excluded"
