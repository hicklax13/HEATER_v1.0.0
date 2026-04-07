"""Tests for category need scoring and trade need-efficiency."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import compute_category_need_scores, score_trade_by_need_efficiency
from src.valuation import LeagueConfig


@pytest.fixture
def config():
    return LeagueConfig()


# ---------------------------------------------------------------------------
# compute_category_need_scores
# ---------------------------------------------------------------------------


class TestComputeCategoryNeedScores:
    """Tests for compute_category_need_scores()."""

    def test_empty_gap_analysis_returns_defaults(self, config):
        """None/empty input returns all categories = 0.5."""
        result = compute_category_need_scores(None, config)
        assert len(result) == len(config.all_categories)
        for cat in config.all_categories:
            assert result[cat] == 0.5

        result_empty = compute_category_need_scores({}, config)
        for cat in config.all_categories:
            assert result_empty[cat] == 0.5

    def test_punt_category_gets_zero(self, config):
        """A punted category (rank 11, is_punt=True) should get 0.0."""
        gap = {"SB": {"rank": 11, "is_punt": True}}
        result = compute_category_need_scores(gap, config)
        assert result["SB"] == 0.0

    def test_critical_weakness_rank_8(self, config):
        """Rank 8 (not punt) should be scored as critical weakness = 1.0."""
        gap = {"AVG": {"rank": 8, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["AVG"] == 1.0

    def test_critical_weakness_rank_9(self, config):
        """Rank 9 (not punt) should also be critical weakness = 1.0."""
        gap = {"ERA": {"rank": 9, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["ERA"] == 1.0

    def test_competitive_gap_rank_5_to_7(self, config):
        """Rank 6 (not punt) should be competitive gap = 0.6."""
        gap = {"HR": {"rank": 6, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["HR"] == 0.6

    def test_slight_edge_rank_3_to_4(self, config):
        """Rank 3 (not punt) should be slight edge = 0.3."""
        gap = {"RBI": {"rank": 3, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["RBI"] == 0.3

    def test_dominant_rank_1_to_2(self, config):
        """Rank 1 (not punt) should be dominant = 0.1."""
        gap = {"K": {"rank": 1, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["K"] == 0.1

    def test_rank_10_not_punt_gets_zero(self, config):
        """Rank 10 without punt flag still gets 0.0 (very weak, untradeable)."""
        gap = {"W": {"rank": 10, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert result["W"] == 0.0

    def test_missing_category_defaults(self, config):
        """Categories not in gap_analysis default to 0.5."""
        # Only provide 6 of 12 categories
        gap = {
            "R": {"rank": 2, "is_punt": False},
            "HR": {"rank": 4, "is_punt": False},
            "RBI": {"rank": 6, "is_punt": False},
            "SB": {"rank": 8, "is_punt": False},
            "AVG": {"rank": 10, "is_punt": False},
            "OBP": {"rank": 12, "is_punt": True},
        }
        result = compute_category_need_scores(gap, config)
        # Pitching categories should all be 0.5 (missing from gap)
        for cat in config.pitching_categories:
            assert result[cat] == 0.5

    def test_all_12_categories_returned(self, config):
        """Result should have exactly 12 keys matching all_categories."""
        gap = {"R": {"rank": 5, "is_punt": False}}
        result = compute_category_need_scores(gap, config)
        assert len(result) == 12
        assert set(result.keys()) == set(config.all_categories)


# ---------------------------------------------------------------------------
# score_trade_by_need_efficiency
# ---------------------------------------------------------------------------


class TestScoreTradeByNeedEfficiency:
    """Tests for score_trade_by_need_efficiency()."""

    def test_positive_impact_high_need(self):
        """Gaining +1.0 SGP in a category with need=1.0 yields need_weighted_gain=1.0."""
        impact = {"AVG": 1.0}
        needs = {"AVG": 1.0}
        result = score_trade_by_need_efficiency(impact, needs)
        assert result["need_weighted_gain"] == 1.0

    def test_positive_impact_low_need(self):
        """Gaining +1.0 SGP in a category with need=0.1 yields need_weighted_gain=0.1."""
        impact = {"K": 1.0}
        needs = {"K": 0.1}
        result = score_trade_by_need_efficiency(impact, needs)
        assert result["need_weighted_gain"] == 0.1

    def test_negative_impact_low_need(self):
        """Losing -1.0 SGP in a category with need=0.1 is affordable (low cost)."""
        impact = {"K": -1.0}
        needs = {"K": 0.1}
        result = score_trade_by_need_efficiency(impact, needs)
        # affordability = 1.0 - 0.1 = 0.9; cost = 1.0 * (1.0 - 0.9) = 0.1
        assert result["affordability_weighted_cost"] == 0.1

    def test_negative_impact_high_need(self):
        """Losing -1.0 SGP in a category with need=1.0 is very costly."""
        impact = {"AVG": -1.0}
        needs = {"AVG": 1.0}
        result = score_trade_by_need_efficiency(impact, needs)
        # affordability = 1.0 - 1.0 = 0.0; cost = 1.0 * (1.0 - 0.0) = 1.0
        assert result["affordability_weighted_cost"] == 1.0

    def test_efficiency_ratio_gt_1_is_good(self):
        """Big gains in weak cats, small losses in strong cats yields ratio > 1."""
        impact = {"AVG": 2.0, "K": -0.5}
        needs = {"AVG": 1.0, "K": 0.1}
        result = score_trade_by_need_efficiency(impact, needs)
        # gain: 2.0 * 1.0 = 2.0
        # cost: 0.5 * (1.0 - 0.9) = 0.05
        assert result["efficiency_ratio"] > 1.0

    def test_efficiency_ratio_lt_1_is_bad(self):
        """Small gains in strong cats, big losses in weak cats yields ratio < 1."""
        impact = {"K": 0.5, "AVG": -2.0}
        needs = {"K": 0.1, "AVG": 1.0}
        result = score_trade_by_need_efficiency(impact, needs)
        # gain: 0.5 * 0.1 = 0.05
        # cost: 2.0 * (1.0 - 0.0) = 2.0
        assert result["efficiency_ratio"] < 1.0

    def test_boosted_and_costly_cats_populated(self):
        """Verify boosted_cats and costly_cats lists are populated correctly."""
        impact = {"HR": 1.5, "SB": -0.8, "W": 0.3}
        needs = {"HR": 0.6, "SB": 0.3, "W": 0.9}
        result = score_trade_by_need_efficiency(impact, needs)
        assert "HR" in result["boosted_cats"]
        assert "W" in result["boosted_cats"]
        assert "SB" in result["costly_cats"]
        assert len(result["boosted_cats"]) == 2
        assert len(result["costly_cats"]) == 1

    def test_zero_impact_returns_safe_defaults(self):
        """All zero impacts should return efficiency_ratio = 0.0 (0.0 / 0.01)."""
        impact = {"HR": 0.0, "SB": 0.0, "K": 0.0}
        needs = {"HR": 0.5, "SB": 0.5, "K": 0.5}
        result = score_trade_by_need_efficiency(impact, needs)
        assert result["efficiency_ratio"] == 0.0
        assert result["need_weighted_gain"] == 0.0
        assert result["affordability_weighted_cost"] == 0.0
        assert result["boosted_cats"] == []
        assert result["costly_cats"] == []
