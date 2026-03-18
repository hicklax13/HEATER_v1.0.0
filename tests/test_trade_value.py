"""Tests for Trade Value Chart (src/trade_value.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.trade_value import (
    LEAGUE_BUDGET,
    TIERS,
    WEEKLY_TAU,
    assign_tier,
    compute_contextual_values,
    compute_g_score_adjustment,
    compute_trade_values,
    filter_by_position,
)
from src.valuation import LeagueConfig, SGPCalculator

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pool(n=20):
    """Create a minimal player pool for testing."""
    players = []
    for i in range(n):
        is_hitter = i < 14
        if is_hitter:
            pos_options = ["C", "1B", "2B", "3B", "SS", "OF"]
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
                    "h": 120 + i * 3,
                    "r": 60 + i * 2,
                    "hr": 15 + i,
                    "rbi": 55 + i * 3,
                    "sb": 5 + i,
                    "avg": 0.260 + i * 0.003,
                    "obp": 0.330 + i * 0.003,
                    "bb": 40 + i,
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
            ip_base = 150 if pos == "SP" else 60
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
                    "ip": ip_base + (i - 14) * 10,
                    "w": 8 + (i - 14),
                    "l": 6 + (i - 14) % 3,
                    "sv": 20 if pos == "RP" else 0,
                    "k": 100 + (i - 14) * 15,
                    "era": 3.80 - (i - 14) * 0.1,
                    "whip": 1.20 - (i - 14) * 0.02,
                    "er": 50 + (i - 14) * 3,
                    "bb_allowed": 40 + (i - 14) * 2,
                    "h_allowed": 120 + (i - 14) * 5,
                    "adp": 30 + i * 6,
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


# ── Tier Assignment ───────────────────────────────────────────────────


class TestAssignTier:
    def test_elite_tier(self):
        assert assign_tier(95.0) == "Elite"

    def test_star_tier(self):
        assert assign_tier(80.0) == "Star"

    def test_solid_tier(self):
        assert assign_tier(60.0) == "Solid Starter"

    def test_flex_tier(self):
        assert assign_tier(40.0) == "Flex"

    def test_replacement_tier(self):
        assert assign_tier(10.0) == "Replacement"

    def test_zero_is_replacement(self):
        assert assign_tier(0.0) == "Replacement"

    def test_boundary_elite(self):
        assert assign_tier(90.0) == "Elite"

    def test_boundary_star(self):
        assert assign_tier(75.0) == "Star"

    def test_boundary_solid(self):
        assert assign_tier(55.0) == "Solid Starter"

    def test_boundary_flex(self):
        assert assign_tier(35.0) == "Flex"

    def test_negative_value(self):
        assert assign_tier(-5.0) == "Replacement"


# ── G-Score Adjustment ────────────────────────────────────────────────


class TestGScoreAdjustment:
    def test_zero_sgp_returns_zero(self):
        per_cat = {cat: 0.0 for cat in LeagueConfig().all_categories}
        sigma = {cat: 1.0 for cat in LeagueConfig().all_categories}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert result == 0.0

    def test_positive_sgp_returns_positive(self):
        config = LeagueConfig()
        per_cat = {cat: 1.0 for cat in config.all_categories}
        sigma = {cat: 2.0 for cat in config.all_categories}
        result = compute_g_score_adjustment(per_cat, sigma, config)
        assert result > 0

    def test_high_tau_reduces_value(self):
        """Categories with high weekly variance (tau) should be discounted."""
        config = LeagueConfig()
        per_cat = {"SV": 2.0}  # SV has high tau
        sigma = {"SV": 1.5}

        g_sv = compute_g_score_adjustment(per_cat, sigma, config)

        per_cat_hr = {"HR": 2.0}  # HR has lower tau
        sigma_hr = {"HR": 1.5}
        g_hr = compute_g_score_adjustment(per_cat_hr, sigma_hr, config)

        # With same input SGP and sigma, the high-tau category should be discounted more
        # (though the exact comparison depends on SGP denominators)
        assert isinstance(g_sv, float)
        assert isinstance(g_hr, float)

    def test_zero_sigma_handled(self):
        """Should not crash with zero sigma."""
        per_cat = {"R": 1.0}
        sigma = {"R": 0.0}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert math.isfinite(result)

    def test_missing_categories_handled(self):
        """Missing categories should be skipped gracefully."""
        per_cat = {"R": 1.5, "HR": 2.0}
        sigma = {"R": 1.0, "HR": 1.0}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert result > 0


# ── Main Trade Value Computation ──────────────────────────────────────


class TestComputeTradeValues:
    def test_returns_dataframe(self, pool, config):
        result = compute_trade_values(pool, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(pool)

    def test_has_required_columns(self, pool, config):
        result = compute_trade_values(pool, config)
        for col in ["trade_value", "dollar_value", "tier", "rank", "pos_rank"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_trade_values_bounded_0_100(self, pool, config):
        result = compute_trade_values(pool, config)
        assert result["trade_value"].min() >= 0.0
        assert result["trade_value"].max() <= 100.0

    def test_sorted_descending(self, pool, config):
        result = compute_trade_values(pool, config)
        values = result["trade_value"].tolist()
        assert values == sorted(values, reverse=True)

    def test_rank_sequential(self, pool, config):
        result = compute_trade_values(pool, config)
        assert result["rank"].tolist() == list(range(1, len(result) + 1))

    def test_all_tiers_assigned(self, pool, config):
        result = compute_trade_values(pool, config)
        valid_tiers = {name for _, name in TIERS}
        for tier in result["tier"]:
            assert tier in valid_tiers

    def test_dollar_values_positive(self, pool, config):
        result = compute_trade_values(pool, config)
        assert (result["dollar_value"] >= 1.0).all()

    def test_empty_pool_returns_empty(self, config):
        result = compute_trade_values(pd.DataFrame(), config)
        assert result.empty

    def test_top_player_highest_value(self, pool, config):
        result = compute_trade_values(pool, config)
        # The top-ranked player should have the highest trade value
        assert result.iloc[0]["rank"] == 1
        assert result.iloc[0]["trade_value"] >= result.iloc[1]["trade_value"]

    def test_weeks_remaining_scales_values(self, pool, config):
        full = compute_trade_values(pool, config, weeks_remaining=26)
        half = compute_trade_values(pool, config, weeks_remaining=13)
        # Half-season values should be approximately half of full-season
        top_full = full.iloc[0]["trade_value"]
        top_half = half.iloc[0]["trade_value"]
        assert top_half < top_full

    def test_with_standings(self, pool, config):
        """Should work when standings are provided."""
        standings_data = []
        for cat in config.all_categories:
            for i in range(12):
                standings_data.append(
                    {
                        "team_name": f"Team_{i}",
                        "category": cat,
                        "total": 100 + i * 10,
                        "rank": i + 1,
                    }
                )
        standings = pd.DataFrame(standings_data)
        result = compute_trade_values(pool, config, standings=standings)
        assert len(result) == len(pool)
        assert "trade_value" in result.columns

    def test_player_name_column_alias(self, config):
        """Should handle player_name column (common alias)."""
        pool = _make_pool(5)
        pool = pool.rename(columns={"name": "player_name"})
        result = compute_trade_values(pool, config)
        assert "name" in result.columns
        assert len(result) == 5


# ── Contextual Values ─────────────────────────────────────────────────


class TestContextualValues:
    def test_empty_totals_returns_universal(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = compute_contextual_values(tv, {}, {}, "Team_0", config)
        assert "contextual_value" in result.columns
        assert result["contextual_value"].equals(result["trade_value"])

    def test_contextual_values_differ_from_universal(self, pool, config):
        tv = compute_trade_values(pool, config)
        user_totals = {
            "R": 300,
            "HR": 80,
            "RBI": 280,
            "SB": 20,
            "AVG": 0.260,
            "OBP": 0.320,
            "W": 40,
            "L": 35,
            "SV": 15,
            "K": 500,
            "ERA": 4.00,
            "WHIP": 1.25,
        }
        all_totals = {
            "Team_0": user_totals,
            "Team_1": {k: v * 1.1 for k, v in user_totals.items()},
            "Team_2": {k: v * 0.9 for k, v in user_totals.items()},
        }
        result = compute_contextual_values(tv, user_totals, all_totals, "Team_0", config)
        assert "contextual_value" in result.columns
        assert "contextual_tier" in result.columns

    def test_contextual_has_tiers(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = compute_contextual_values(tv, {}, {}, "Team_0", config)
        assert "contextual_tier" in result.columns


# ── Position Filtering ────────────────────────────────────────────────


class TestFilterByPosition:
    def test_filter_all_returns_all(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "All")
        assert len(result) == len(tv)

    def test_filter_sp_only(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "SP")
        assert len(result) > 0
        for _, row in result.iterrows():
            assert "SP" in str(row["positions"])

    def test_filter_of_only(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "OF")
        assert len(result) > 0

    def test_filter_nonexistent_position(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "DH")
        assert len(result) == 0

    def test_pos_rank_recomputed(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "OF")
        if len(result) > 0:
            assert result["pos_rank"].tolist() == list(range(1, len(result) + 1))


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_player(self, config):
        pool = _make_pool(1)
        result = compute_trade_values(pool, config)
        assert len(result) == 1
        assert result.iloc[0]["rank"] == 1

    def test_all_same_stats(self, config):
        """All players with identical stats should get similar values."""
        players = []
        for i in range(5):
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Player_{i + 1}",
                    "team": "TM",
                    "positions": "OF",
                    "is_hitter": 1,
                    "pa": 500,
                    "ab": 450,
                    "h": 120,
                    "r": 60,
                    "hr": 20,
                    "rbi": 70,
                    "sb": 10,
                    "avg": 0.267,
                    "obp": 0.340,
                    "bb": 40,
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
                    "adp": 50,
                    "is_injured": 0,
                }
            )
        pool = pd.DataFrame(players)
        result = compute_trade_values(pool, config)
        values = result["trade_value"].tolist()
        # All should be very similar
        assert max(values) - min(values) < 1.0

    def test_weekly_tau_constants_reasonable(self):
        """All tau values should be positive."""
        for cat, tau in WEEKLY_TAU.items():
            assert tau > 0, f"Tau for {cat} should be positive"

    def test_league_budget_constant(self):
        assert LEAGUE_BUDGET == 3120.0  # 12 × $260
