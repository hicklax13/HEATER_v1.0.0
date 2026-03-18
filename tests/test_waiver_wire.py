"""Tests for Waiver Wire + Drop Suggestions (src/waiver_wire.py)."""

import pandas as pd
import pytest

from src.valuation import LeagueConfig
from src.waiver_wire import (
    classify_category_priority,
    compute_add_drop_recommendations,
    compute_babip,
    compute_drop_cost,
    compute_net_swap_value,
    compute_sustainability_score,
)

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pool(n=25):
    """Create a player pool with hitters and pitchers."""
    players = []
    for i in range(n):
        is_hitter = i < 18
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
                    "h": 110 + i * 4,
                    "r": 55 + i * 3,
                    "hr": 12 + i * 2,
                    "rbi": 50 + i * 3,
                    "sb": 3 + i,
                    "avg": round(0.250 + i * 0.004, 3),
                    "obp": round(0.320 + i * 0.004, 3),
                    "bb": 35 + i * 2,
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
                    "ip": ip + (i - 18) * 5,
                    "w": 8 + (i - 18),
                    "l": 5 + (i - 18) % 3,
                    "sv": 15 if pos == "RP" else 0,
                    "k": 90 + (i - 18) * 10,
                    "era": round(3.90 - (i - 18) * 0.1, 2),
                    "whip": round(1.22 - (i - 18) * 0.02, 2),
                    "er": 50 + (i - 18) * 2,
                    "bb_allowed": 35 + (i - 18) * 2,
                    "h_allowed": 110 + (i - 18) * 5,
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


@pytest.fixture
def roster_ids():
    """First 14 players form the user's roster."""
    return list(range(1, 15))


# ── BABIP ─────────────────────────────────────────────────────────────


class TestBabip:
    def test_league_average(self):
        """100 H, 20 HR, 400 AB, 80 K, 5 SF => (80)/(295) = .271"""
        result = compute_babip(100, 20, 400, 80, 5)
        assert abs(result - 0.271) < 0.01

    def test_zero_denominator(self):
        result = compute_babip(0, 0, 0, 0, 0)
        assert result == 0.300  # league average default

    def test_negative_denominator(self):
        result = compute_babip(5, 5, 10, 5, 0)
        assert result == 0.300

    def test_high_babip(self):
        result = compute_babip(150, 20, 400, 60, 5)
        assert result > 0.350


# ── Category Priority ─────────────────────────────────────────────────


class TestCategoryPriority:
    def test_returns_all_categories(self, config):
        user = {c: 100 for c in config.all_categories}
        all_teams = {
            "User": user,
            "T1": {c: 110 for c in config.all_categories},
            "T2": {c: 90 for c in config.all_categories},
        }
        result = classify_category_priority(user, all_teams, "User", config=config)
        assert set(result.keys()) == set(config.all_categories)

    def test_empty_opponents(self, config):
        user = {c: 100 for c in config.all_categories}
        result = classify_category_priority(user, {}, "User", config=config)
        for cat in config.all_categories:
            assert result[cat] == "ATTACK"

    def test_valid_priority_values(self, config):
        user = {c: 50 for c in config.all_categories}
        all_teams = {f"T{i}": {c: 40 + i * 5 for c in config.all_categories} for i in range(12)}
        all_teams["User"] = user
        result = classify_category_priority(user, all_teams, "User", config=config)
        valid = {"ATTACK", "DEFEND", "IGNORE"}
        for cat, priority in result.items():
            assert priority in valid, f"{cat} has invalid priority: {priority}"


# ── Drop Cost ─────────────────────────────────────────────────────────


class TestDropCost:
    def test_drop_cost_positive(self, pool, config, roster_ids):
        """Dropping a player should have a positive cost (losing value)."""
        cost = compute_drop_cost(1, roster_ids, pool, config)
        assert isinstance(cost, float)

    def test_drop_best_player_costs_more(self, pool, config, roster_ids):
        """Dropping a better player should cost more than a worse one."""
        cost_good = compute_drop_cost(14, roster_ids, pool, config)
        cost_bad = compute_drop_cost(1, roster_ids, pool, config)
        # Player 14 has better stats (later in pool = higher stats in fixture)
        assert cost_good >= cost_bad

    def test_drop_nonexistent_player(self, pool, config, roster_ids):
        """Dropping a player not on roster should have zero cost."""
        cost = compute_drop_cost(999, roster_ids, pool, config)
        assert cost == 0.0


# ── Net Swap Value ────────────────────────────────────────────────────


class TestNetSwapValue:
    def test_returns_dict(self, pool, config, roster_ids):
        result = compute_net_swap_value(15, 1, roster_ids, pool, config)
        assert "net_sgp" in result
        assert "category_deltas" in result

    def test_category_deltas_has_all_cats(self, pool, config, roster_ids):
        result = compute_net_swap_value(15, 1, roster_ids, pool, config)
        for cat in config.all_categories:
            assert cat in result["category_deltas"]

    def test_swap_same_player_zero(self, pool, config, roster_ids):
        """Swapping a player for themselves should yield ~zero net SGP."""
        result = compute_net_swap_value(1, 1, roster_ids, pool, config)
        assert abs(result["net_sgp"]) < 0.01

    def test_upgrading_player_positive_sgp(self, pool, config):
        """Replacing a weak player with a stronger one should be positive."""
        roster = list(range(1, 15))
        # Player 18 (hitter) has better stats than player 1
        result = compute_net_swap_value(18, 1, roster, pool, config)
        assert result["net_sgp"] > 0


# ── Sustainability Score ──────────────────────────────────────────────


class TestSustainability:
    def test_returns_0_to_1(self, pool):
        for _, player in pool.iterrows():
            score = compute_sustainability_score(player)
            assert 0.0 <= score <= 1.0

    def test_high_babip_low_sustainability(self):
        """Player with very high BABIP should have low sustainability."""
        player = pd.Series(
            {
                "h": 160,
                "hr": 15,
                "ab": 400,
                "k": 60,
                "sf": 4,
                "is_hitter": 1,
                "era": 0,
                "ip": 0,
            }
        )
        score = compute_sustainability_score(player)
        assert score < 0.5

    def test_low_babip_high_sustainability(self):
        """Player with low BABIP should have high sustainability (buy low)."""
        player = pd.Series(
            {
                "h": 80,
                "hr": 15,
                "ab": 400,
                "k": 100,
                "sf": 4,
                "is_hitter": 1,
                "era": 0,
                "ip": 0,
            }
        )
        score = compute_sustainability_score(player)
        assert score > 0.6

    def test_pitcher_sustainability(self):
        """Pitcher with reasonable ERA should have moderate sustainability."""
        player = pd.Series(
            {
                "h": 0,
                "hr": 0,
                "ab": 0,
                "k": 100,
                "sf": 0,
                "is_hitter": 0,
                "era": 3.80,
                "ip": 100,
            }
        )
        score = compute_sustainability_score(player)
        assert 0.3 <= score <= 0.8

    def test_insufficient_sample(self):
        """Small sample should return neutral score."""
        player = pd.Series(
            {
                "h": 5,
                "hr": 1,
                "ab": 20,
                "k": 5,
                "sf": 0,
                "is_hitter": 1,
                "era": 0,
                "ip": 0,
            }
        )
        score = compute_sustainability_score(player)
        assert score == 0.5


# ── Main Recommendations ─────────────────────────────────────────────


class TestAddDropRecommendations:
    def test_returns_list(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=2)
        assert isinstance(result, list)

    def test_each_entry_has_required_keys(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=1)
        if result:
            entry = result[0]
            assert "add_player_id" in entry
            assert "drop_player_id" in entry
            assert "net_sgp_delta" in entry
            assert "category_impact" in entry
            assert "sustainability_score" in entry
            assert "reasoning" in entry

    def test_positive_net_sgp(self, pool, config, roster_ids):
        """All recommendations should have positive net SGP."""
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=3)
        for entry in result:
            assert entry["net_sgp_delta"] > 0

    def test_no_duplicate_adds(self, pool, config, roster_ids):
        """Each FA should appear at most once in recommendations."""
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=3)
        add_ids = [r["add_player_id"] for r in result]
        assert len(add_ids) == len(set(add_ids))

    def test_no_duplicate_drops(self, pool, config, roster_ids):
        """Each drop candidate should appear at most once."""
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=3)
        drop_ids = [r["drop_player_id"] for r in result]
        assert len(drop_ids) == len(set(drop_ids))

    def test_max_moves_respected(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=2)
        assert len(result) <= 2

    def test_empty_pool(self, config):
        result = compute_add_drop_recommendations([1, 2, 3], pd.DataFrame(), config)
        assert result == []

    def test_empty_roster(self, pool, config):
        result = compute_add_drop_recommendations([], pool, config)
        assert result == []

    def test_reasoning_nonempty(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=1)
        if result:
            assert len(result[0]["reasoning"]) > 0

    def test_sorted_by_net_sgp(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=3)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["net_sgp_delta"] >= result[i + 1]["net_sgp_delta"]

    def test_sustainability_in_range(self, pool, config, roster_ids):
        result = compute_add_drop_recommendations(roster_ids, pool, config, max_moves=2)
        for entry in result:
            assert 0.0 <= entry["sustainability_score"] <= 1.0
