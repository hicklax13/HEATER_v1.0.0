"""Tests for trade_intelligence module — health, category weights, FA gating, scarcity, readiness."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.trade_intelligence import (
    FA_GATE_THRESHOLD,
    SCARCE_POSITIONS,
    STATUS_MULTIPLIERS,
    SV_SCARCITY_MULT,
    _quick_player_sgp,
    apply_scarcity_flags,
    compute_fa_comparisons,
    compute_trade_readiness,
    compute_trade_readiness_batch,
    get_category_weights,
    get_health_adjusted_pool,
)
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
    # Players
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (2, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (3, 'Raisel Iglesias', 'ATL', 'RP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (4, 'Free Agent Hitter', 'FA', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (5, 'IL Pitcher', 'BOS', 'SP', 0)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (6, 'Minors Guy', 'NYM', 'SS', 1)"
    )
    # Injury history
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (1, 2025, 150, 162)"
    )
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (5, 2025, 60, 162)"
    )
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (5, 2024, 80, 162)"
    )
    # League rosters with status
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team, status) "
        "VALUES ('Team A', 0, 1, 'OF', 1, 'active')"
    )
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team, status) "
        "VALUES ('Team B', 1, 5, 'SP', 0, 'IL15')"
    )
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team, status) "
        "VALUES ('Team B', 1, 6, 'SS', 0, 'NA')"
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
def sample_pool():
    """Create a sample player pool DataFrame."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "positions": "OF",
                "is_hitter": 1,
                "team": "NYY",
                "r": 60,
                "hr": 30,
                "rbi": 70,
                "sb": 5,
                "avg": 0.280,
                "obp": 0.380,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "pa": 300,
                "ab": 270,
                "h": 76,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
            {
                "player_id": 2,
                "name": "Gerrit Cole",
                "positions": "SP",
                "is_hitter": 0,
                "team": "NYY",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": 10,
                "l": 5,
                "sv": 0,
                "k": 150,
                "era": 3.20,
                "whip": 1.10,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "ip": 120,
                "er": 43,
                "bb_allowed": 30,
                "h_allowed": 100,
            },
            {
                "player_id": 3,
                "name": "Raisel Iglesias",
                "positions": "RP",
                "is_hitter": 0,
                "team": "ATL",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": 3,
                "l": 2,
                "sv": 25,
                "k": 50,
                "era": 2.80,
                "whip": 1.05,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "ip": 45,
                "er": 14,
                "bb_allowed": 12,
                "h_allowed": 35,
            },
            {
                "player_id": 4,
                "name": "Free Agent Hitter",
                "positions": "OF",
                "is_hitter": 1,
                "team": "FA",
                "r": 40,
                "hr": 15,
                "rbi": 45,
                "sb": 8,
                "avg": 0.250,
                "obp": 0.320,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "pa": 250,
                "ab": 230,
                "h": 58,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
            {
                "player_id": 5,
                "name": "IL Pitcher",
                "positions": "SP",
                "is_hitter": 0,
                "team": "BOS",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": 8,
                "l": 6,
                "sv": 0,
                "k": 100,
                "era": 3.80,
                "whip": 1.25,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "ip": 100,
                "er": 42,
                "bb_allowed": 30,
                "h_allowed": 95,
            },
            {
                "player_id": 6,
                "name": "Minors Guy",
                "positions": "SS",
                "is_hitter": 1,
                "team": "NYM",
                "r": 10,
                "hr": 3,
                "rbi": 10,
                "sb": 5,
                "avg": 0.220,
                "obp": 0.280,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "pa": 50,
                "ab": 45,
                "h": 10,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
        ]
    )


@pytest.fixture
def config():
    return LeagueConfig()


# ---------------------------------------------------------------------------
# Health adjustment tests
# ---------------------------------------------------------------------------


class TestHealthAdjustedPool:
    def test_na_players_excluded(self, sample_pool, config):
        pool = get_health_adjusted_pool(sample_pool, config)
        # Player 6 (Minors Guy) has status=NA in league_rosters
        assert 6 not in pool["player_id"].values

    def test_il15_reduces_counting_stats(self, sample_pool, config):
        pool = get_health_adjusted_pool(sample_pool, config)
        il_pitcher = pool[pool["player_id"] == 5]
        if not il_pitcher.empty:
            # IL15 multiplier is 0.84, combined with health score
            original_k = 100
            adjusted_k = float(il_pitcher.iloc[0]["k"])
            assert adjusted_k < original_k

    def test_active_players_minimally_affected(self, sample_pool, config):
        pool = get_health_adjusted_pool(sample_pool, config)
        judge = pool[pool["player_id"] == 1]
        if not judge.empty:
            # Active player with good health should retain most production
            assert float(judge.iloc[0]["hr"]) >= 25  # At least 83% of 30

    def test_health_score_column_added(self, sample_pool, config):
        pool = get_health_adjusted_pool(sample_pool, config)
        assert "health_score" in pool.columns

    def test_status_column_added(self, sample_pool, config):
        pool = get_health_adjusted_pool(sample_pool, config)
        assert "status" in pool.columns

    def test_empty_pool_returns_empty(self, config):
        pool = get_health_adjusted_pool(pd.DataFrame(), config)
        assert pool.empty


# ---------------------------------------------------------------------------
# Category weights tests
# ---------------------------------------------------------------------------


class TestCategoryWeights:
    def test_returns_weights_for_all_categories(self, config):
        all_totals = {
            "Team A": {
                "R": 100,
                "HR": 30,
                "RBI": 100,
                "SB": 20,
                "AVG": 0.260,
                "OBP": 0.330,
                "W": 10,
                "L": 8,
                "SV": 5,
                "K": 150,
                "ERA": 3.80,
                "WHIP": 1.25,
            },
            "Team B": {
                "R": 120,
                "HR": 40,
                "RBI": 110,
                "SB": 30,
                "AVG": 0.270,
                "OBP": 0.340,
                "W": 12,
                "L": 6,
                "SV": 15,
                "K": 180,
                "ERA": 3.50,
                "WHIP": 1.18,
            },
        }
        weights = get_category_weights("Team A", all_totals, config)
        assert len(weights) == len(config.all_categories)
        assert all(isinstance(v, float) for v in weights.values())

    def test_returns_equal_weights_without_standings(self, config):
        weights = get_category_weights("Team A", {}, config)
        assert all(v == 1.0 for v in weights.values())


# ---------------------------------------------------------------------------
# FA comparison tests
# ---------------------------------------------------------------------------


class TestFAComparisons:
    def test_fa_gate_flags_comparable_fa(self, sample_pool, config):
        # Player 1 (Judge) is an OF — FA player 4 is also an OF
        results = compute_fa_comparisons(
            opponent_player_ids=[1],
            user_roster_ids=[2, 3],
            fa_pool=sample_pool[sample_pool["player_id"] == 4],
            player_pool=sample_pool,
            config=config,
        )
        assert 1 in results
        # FA at OF position exists — check if comparison was made
        assert "has_alternative" in results[1]

    def test_empty_fa_pool_returns_empty(self, sample_pool, config):
        results = compute_fa_comparisons([1], [2], pd.DataFrame(), sample_pool, config)
        assert results == {}


# ---------------------------------------------------------------------------
# Scarcity premium tests
# ---------------------------------------------------------------------------


class TestScarcityFlags:
    def test_closer_gets_scarcity_mult(self, sample_pool):
        pool = apply_scarcity_flags(sample_pool)
        closer = pool[pool["player_id"] == 3]
        assert not closer.empty
        assert bool(closer.iloc[0]["is_closer"]) is True
        assert closer.iloc[0]["scarcity_mult"] == SV_SCARCITY_MULT

    def test_non_closer_no_scarcity(self, sample_pool):
        pool = apply_scarcity_flags(sample_pool)
        cole = pool[pool["player_id"] == 2]
        assert not cole.empty
        assert bool(cole.iloc[0]["is_closer"]) is False

    def test_scarce_position_gets_premium(self, sample_pool):
        pool = apply_scarcity_flags(sample_pool)
        minors = pool[pool["player_id"] == 6]  # SS position
        if not minors.empty:
            assert minors.iloc[0]["scarcity_mult"] == 1.15


# ---------------------------------------------------------------------------
# Trade Readiness tests
# ---------------------------------------------------------------------------


class TestTradeReadiness:
    def test_score_bounded_0_100(self, sample_pool, config):
        result = compute_trade_readiness(
            player_id=1,
            user_roster_ids=[2, 3],
            user_totals={
                "R": 100,
                "HR": 30,
                "RBI": 100,
                "SB": 20,
                "AVG": 0.260,
                "OBP": 0.330,
                "W": 10,
                "L": 8,
                "SV": 5,
                "K": 150,
                "ERA": 3.80,
                "WHIP": 1.25,
            },
            all_team_totals={
                "Team A": {
                    "R": 100,
                    "HR": 30,
                    "RBI": 100,
                    "SB": 20,
                    "AVG": 0.260,
                    "OBP": 0.330,
                    "W": 10,
                    "L": 8,
                    "SV": 5,
                    "K": 150,
                    "ERA": 3.80,
                    "WHIP": 1.25,
                },
            },
            user_team_name="Team A",
            fa_pool=pd.DataFrame(),
            player_pool=sample_pool,
            config=config,
        )
        assert 0 <= result["score"] <= 100

    def test_all_sub_components_present(self, sample_pool, config):
        result = compute_trade_readiness(1, [2], {}, {}, "Team A", pd.DataFrame(), sample_pool, config)
        for key in ["score", "category_fit", "projection_quality", "health", "scarcity", "fa_advantage"]:
            assert key in result

    def test_missing_player_returns_zero(self, sample_pool, config):
        result = compute_trade_readiness(999, [1], {}, {}, "Team A", pd.DataFrame(), sample_pool, config)
        assert result["score"] == 0


class TestTradeReadinessBatch:
    def test_returns_dataframe(self, sample_pool, config):
        df = compute_trade_readiness_batch(
            player_ids=[1, 2, 3],
            user_roster_ids=[4],
            user_totals={},
            all_team_totals={},
            user_team_name="Team A",
            fa_pool=pd.DataFrame(),
            player_pool=sample_pool,
            config=config,
        )
        assert isinstance(df, pd.DataFrame)
        assert "score" in df.columns
        assert len(df) <= 3

    def test_sorted_by_score_descending(self, sample_pool, config):
        df = compute_trade_readiness_batch([1, 2, 3], [4], {}, {}, "Team A", pd.DataFrame(), sample_pool, config)
        if len(df) >= 2:
            scores = df["score"].tolist()
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Quick SGP tests
# ---------------------------------------------------------------------------


class TestQuickPlayerSGP:
    def test_hitter_has_positive_sgp(self, sample_pool, config):
        judge = sample_pool[sample_pool["player_id"] == 1].iloc[0]
        sgp = _quick_player_sgp(judge, config)
        assert sgp > 0

    def test_pitcher_sgp_is_finite(self, sample_pool, config):
        cole = sample_pool[sample_pool["player_id"] == 2].iloc[0]
        sgp = _quick_player_sgp(cole, config)
        # Pitcher SGP can be negative (ERA/WHIP inverse stats dominate counting)
        assert np.isfinite(sgp)


# ---------------------------------------------------------------------------
# scan_1_for_1 backward compatibility
# ---------------------------------------------------------------------------


class TestScan1For1BackwardCompat:
    def test_works_without_new_params(self, sample_pool, config):
        from src.trade_finder import scan_1_for_1

        # Should work without category_weights, fa_comparisons, roster_statuses
        results = scan_1_for_1(
            user_roster_ids=[1],
            opponent_roster_ids=[4],
            player_pool=sample_pool,
            config=config,
        )
        assert isinstance(results, list)
