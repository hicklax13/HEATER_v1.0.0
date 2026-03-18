"""Tests for Trade Finder (src/trade_finder.py)."""

import numpy as np
import pandas as pd
import pytest

from src.trade_finder import (
    LOSS_AVERSION,
    MIN_SGP_GAIN,
    acceptance_label,
    compute_team_vectors,
    cosine_dissimilarity,
    estimate_acceptance_probability,
    find_complementary_teams,
    find_trade_opportunities,
    scan_1_for_1,
)
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pool(n=30):
    """Create a player pool with hitters and pitchers."""
    players = []
    for i in range(n):
        is_hitter = i < 20
        if is_hitter:
            pos_options = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
            pos = pos_options[i % len(pos_options)]
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Hitter_{i + 1}",
                    "team": f"TM{i % 5}",
                    "positions": pos,
                    "is_hitter": 1,
                    "pa": 500 + i * 10,
                    "ab": 450 + i * 10,
                    "h": 110 + i * 3,
                    "r": 55 + i * 2,
                    "hr": 12 + i * 2,
                    "rbi": 50 + i * 3,
                    "sb": 3 + i,
                    "avg": round(0.250 + i * 0.003, 3),
                    "obp": round(0.320 + i * 0.003, 3),
                    "bb": 35 + i,
                    "hbp": 3,
                    "sf": 4,
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
                    "adp": 10 + i * 8,
                    "is_injured": 0,
                }
            )
        else:
            pos = "SP" if i % 2 == 0 else "RP"
            ip = 150 if pos == "SP" else 60
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Pitcher_{i + 1}",
                    "team": f"TM{i % 5}",
                    "positions": pos,
                    "is_hitter": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "ip": ip,
                    "w": 8,
                    "l": 5,
                    "sv": 15 if pos == "RP" else 0,
                    "k": 90 + (i - 20) * 10,
                    "era": round(3.80 - (i - 20) * 0.05, 2),
                    "whip": round(1.20 - (i - 20) * 0.01, 2),
                    "er": 50,
                    "bb_allowed": 35,
                    "h_allowed": 110,
                    "adp": 30 + i * 5,
                    "is_injured": 0,
                }
            )
    return pd.DataFrame(players)


@pytest.fixture
def pool():
    return _make_pool()


@pytest.fixture
def config():
    return LeagueConfig()


def _make_team_totals(config):
    """Create mock all_team_totals for 4 teams."""
    return {
        "Team_A": {
            "R": 400,
            "HR": 100,
            "RBI": 380,
            "SB": 80,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 50,
            "L": 40,
            "SV": 30,
            "K": 600,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        "Team_B": {
            "R": 350,
            "HR": 120,
            "RBI": 360,
            "SB": 30,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 55,
            "L": 35,
            "SV": 20,
            "K": 650,
            "ERA": 3.60,
            "WHIP": 1.15,
        },
        "Team_C": {
            "R": 380,
            "HR": 80,
            "RBI": 350,
            "SB": 100,
            "AVG": 0.280,
            "OBP": 0.350,
            "W": 45,
            "L": 45,
            "SV": 35,
            "K": 550,
            "ERA": 4.00,
            "WHIP": 1.25,
        },
        "Team_D": {
            "R": 360,
            "HR": 110,
            "RBI": 370,
            "SB": 50,
            "AVG": 0.265,
            "OBP": 0.335,
            "W": 48,
            "L": 42,
            "SV": 25,
            "K": 580,
            "ERA": 3.90,
            "WHIP": 1.22,
        },
    }


# ── Cosine Dissimilarity ─────────────────────────────────────────────


class TestCosineDissimlarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_dissimilarity(v, v) < 0.001

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_dissimilarity(v1, v2) - 2.0) < 0.001

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_dissimilarity(v1, v2) - 1.0) < 0.001

    def test_zero_vector(self):
        v1 = np.zeros(5)
        v2 = np.ones(5)
        assert cosine_dissimilarity(v1, v2) == 1.0

    def test_range(self):
        v1 = np.random.randn(12)
        v2 = np.random.randn(12)
        d = cosine_dissimilarity(v1, v2)
        assert 0.0 <= d <= 2.0


# ── Team Vectors ──────────────────────────────────────────────────────


class TestTeamVectors:
    def test_returns_dict(self, config):
        totals = _make_team_totals(config)
        vectors = compute_team_vectors(totals, config)
        assert isinstance(vectors, dict)
        assert len(vectors) == 4

    def test_vector_dimensions(self, config):
        totals = _make_team_totals(config)
        vectors = compute_team_vectors(totals, config)
        for v in vectors.values():
            assert len(v) == len(config.all_categories)

    def test_z_scored(self, config):
        """Vectors should be approximately z-scored (mean ~0, std ~1)."""
        totals = _make_team_totals(config)
        vectors = compute_team_vectors(totals, config)
        all_vals = np.array(list(vectors.values()))
        # Each category column should have mean ~0
        col_means = all_vals.mean(axis=0)
        for m in col_means:
            assert abs(m) < 0.5


# ── Complementary Teams ──────────────────────────────────────────────


class TestComplementaryTeams:
    def test_returns_list(self, config):
        totals = _make_team_totals(config)
        result = find_complementary_teams("Team_A", totals, config, top_n=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_excludes_self(self, config):
        totals = _make_team_totals(config)
        result = find_complementary_teams("Team_A", totals, config, top_n=5)
        team_names = [t[0] for t in result]
        assert "Team_A" not in team_names

    def test_sorted_by_dissimilarity(self, config):
        totals = _make_team_totals(config)
        result = find_complementary_teams("Team_A", totals, config, top_n=3)
        scores = [t[1] for t in result]
        assert scores == sorted(scores, reverse=True)

    def test_missing_team(self, config):
        totals = _make_team_totals(config)
        result = find_complementary_teams("NonExistent", totals, config)
        assert result == []


# ── Acceptance Probability ────────────────────────────────────────────


class TestAcceptanceProbability:
    def test_fair_trade_high_acceptance(self):
        prob = estimate_acceptance_probability(1.0, 1.0, 0.8)
        assert prob > 0.3

    def test_unfair_trade_low_acceptance(self):
        prob = estimate_acceptance_probability(3.0, -1.0, 0.2)
        assert prob < 0.3

    def test_in_range(self):
        prob = estimate_acceptance_probability(1.5, 0.5, 0.5)
        assert 0.01 <= prob <= 0.99

    def test_loss_aversion_constant(self):
        assert LOSS_AVERSION == 1.5

    def test_labels(self):
        assert acceptance_label(0.7) == "High"
        assert acceptance_label(0.4) == "Medium"
        assert acceptance_label(0.1) == "Low"


# ── 1-for-1 Scanner ──────────────────────────────────────────────────


class TestScan1for1:
    def test_returns_list(self, pool, config):
        user_ids = list(range(1, 11))
        opp_ids = list(range(11, 21))
        result = scan_1_for_1(user_ids, opp_ids, pool, config)
        assert isinstance(result, list)

    def test_trades_have_required_keys(self, pool, config):
        user_ids = list(range(1, 11))
        opp_ids = list(range(11, 21))
        result = scan_1_for_1(user_ids, opp_ids, pool, config)
        if result:
            trade = result[0]
            assert "giving_ids" in trade
            assert "receiving_ids" in trade
            assert "user_sgp_gain" in trade
            assert "acceptance_probability" in trade
            assert "composite_score" in trade

    def test_user_sgp_positive(self, pool, config):
        user_ids = list(range(1, 11))
        opp_ids = list(range(11, 21))
        result = scan_1_for_1(user_ids, opp_ids, pool, config)
        for trade in result:
            assert trade["user_sgp_gain"] >= MIN_SGP_GAIN

    def test_sorted_by_composite(self, pool, config):
        user_ids = list(range(1, 11))
        opp_ids = list(range(11, 21))
        result = scan_1_for_1(user_ids, opp_ids, pool, config)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["composite_score"] >= result[i + 1]["composite_score"]

    def test_empty_rosters(self, pool, config):
        assert scan_1_for_1([], [], pool, config) == []


# ── Main Trade Finder ─────────────────────────────────────────────────


class TestFindTradeOpportunities:
    def test_returns_list(self, pool, config):
        totals = _make_team_totals(config)
        rosters = {
            "Team_A": list(range(1, 11)),
            "Team_B": list(range(11, 21)),
        }
        result = find_trade_opportunities(
            list(range(1, 11)),
            pool,
            config,
            all_team_totals=totals,
            user_team_name="Team_A",
            league_rosters=rosters,
        )
        assert isinstance(result, list)

    def test_max_results_respected(self, pool, config):
        totals = _make_team_totals(config)
        rosters = {
            "Team_A": list(range(1, 11)),
            "Team_B": list(range(11, 21)),
        }
        result = find_trade_opportunities(
            list(range(1, 11)),
            pool,
            config,
            all_team_totals=totals,
            user_team_name="Team_A",
            league_rosters=rosters,
            max_results=3,
        )
        assert len(result) <= 3

    def test_empty_inputs(self, pool, config):
        assert find_trade_opportunities([], pool, config) == []
        assert find_trade_opportunities([1], pd.DataFrame(), config) == []

    def test_no_rosters_returns_empty(self, pool, config):
        result = find_trade_opportunities(
            [1, 2, 3],
            pool,
            config,
            all_team_totals=None,
            league_rosters=None,
        )
        assert result == []

    def test_trades_have_grade(self, pool, config):
        totals = _make_team_totals(config)
        rosters = {
            "Team_A": list(range(1, 11)),
            "Team_B": list(range(11, 21)),
        }
        result = find_trade_opportunities(
            list(range(1, 11)),
            pool,
            config,
            all_team_totals=totals,
            user_team_name="Team_A",
            league_rosters=rosters,
        )
        for trade in result:
            assert "grade" in trade

    def test_no_duplicate_trades(self, pool, config):
        totals = _make_team_totals(config)
        rosters = {
            "Team_A": list(range(1, 11)),
            "Team_B": list(range(11, 21)),
        }
        result = find_trade_opportunities(
            list(range(1, 11)),
            pool,
            config,
            all_team_totals=totals,
            user_team_name="Team_A",
            league_rosters=rosters,
        )
        seen = set()
        for trade in result:
            key = (tuple(trade["giving_ids"]), tuple(trade["receiving_ids"]))
            assert key not in seen
            seen.add(key)
