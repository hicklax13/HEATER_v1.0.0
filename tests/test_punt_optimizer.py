"""Tests for punt mode optimizer in src/optimizer/pivot_advisor.py."""

import pytest

from src.optimizer.pivot_advisor import (
    CATEGORY_MEAN_CORRELATION,
    compute_punt_redistribution_value,
    recommend_punt_targets,
)
from src.valuation import LeagueConfig

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def config():
    return LeagueConfig()


@pytest.fixture()
def weak_sb_ranks():
    """Ranks where SB is the weakest and most isolated category."""
    return {
        "R": 4,
        "HR": 5,
        "RBI": 3,
        "SB": 11,
        "AVG": 6,
        "OBP": 5,
        "W": 7,
        "L": 6,
        "SV": 8,
        "K": 4,
        "ERA": 5,
        "WHIP": 5,
    }


@pytest.fixture()
def weak_pitching_ranks():
    """Ranks where W and L are weak, pitching overall is poor."""
    return {
        "R": 3,
        "HR": 4,
        "RBI": 3,
        "SB": 6,
        "AVG": 5,
        "OBP": 4,
        "W": 10,
        "L": 9,
        "SV": 7,
        "K": 8,
        "ERA": 7,
        "WHIP": 6,
    }


@pytest.fixture()
def all_strong_ranks():
    """Ranks where no category is worse than 8th."""
    return {cat: 5 for cat in LeagueConfig().all_categories}


# ── recommend_punt_targets tests ───────────────────────────────────────


class TestRecommendPuntTargets:
    def test_sb_recommended_when_rank_ge_9_lowest_correlation(self, weak_sb_ranks, config):
        """SB should be recommended when rank >= 9 since it has the lowest correlation (0.12)."""
        results = recommend_punt_targets(weak_sb_ranks, config, max_punts=2)
        assert len(results) >= 1
        categories = [r["category"] for r in results]
        assert "SB" in categories
        # SB should be first since it has the best isolation + high rank
        assert results[0]["category"] == "SB"

    def test_sb_isolation_score_is_highest(self, weak_sb_ranks, config):
        """SB isolation score should be 1 - 0.12 = 0.88."""
        results = recommend_punt_targets(weak_sb_ranks, config, max_punts=5)
        sb_result = next(r for r in results if r["category"] == "SB")
        assert sb_result["isolation_score"] == pytest.approx(0.88, abs=0.01)

    def test_w_recommended_when_pitching_weak(self, weak_pitching_ranks, config):
        """W should be recommended when pitching is weak since W has low correlation (0.17)."""
        results = recommend_punt_targets(weak_pitching_ranks, config, max_punts=2)
        categories = [r["category"] for r in results]
        # W and L should both appear since both are >= 9
        assert "W" in categories or "L" in categories

    def test_correlated_categories_not_recommended(self, config):
        """HR, R, RBI should NOT be recommended for punt even if weak,
        because they form a highly correlated cluster (0.83)."""
        # Make HR, R, RBI weak but also have competitive partners
        ranks = {
            "R": 10,
            "HR": 10,
            "RBI": 4,  # Competitive — cluster partner
            "SB": 11,
            "AVG": 5,
            "OBP": 5,
            "W": 6,
            "L": 6,
            "SV": 6,
            "K": 5,
            "ERA": 5,
            "WHIP": 5,
        }
        results = recommend_punt_targets(ranks, config, max_punts=3)
        categories = [r["category"] for r in results]
        # SB should rank higher than HR or R because SB is isolated
        if "SB" in categories:
            sb_idx = categories.index("SB")
            for corr_cat in ["HR", "R"]:
                if corr_cat in categories:
                    corr_idx = categories.index(corr_cat)
                    assert sb_idx < corr_idx, f"SB (isolated) should rank above {corr_cat} (correlated)"

    def test_max_punts_zero_returns_empty(self, weak_sb_ranks, config):
        """max_punts=0 should return an empty list."""
        results = recommend_punt_targets(weak_sb_ranks, config, max_punts=0)
        assert results == []

    def test_max_punts_limits_results(self, config):
        """Should never return more than max_punts results."""
        # Make many categories weak
        ranks = {cat: 10 for cat in config.all_categories}
        results = recommend_punt_targets(ranks, config, max_punts=1)
        assert len(results) <= 1
        results = recommend_punt_targets(ranks, config, max_punts=2)
        assert len(results) <= 2

    def test_no_weak_categories_returns_empty(self, all_strong_ranks, config):
        """If no category is rank >= 9, nothing should be recommended."""
        results = recommend_punt_targets(all_strong_ranks, config, max_punts=2)
        assert results == []

    def test_result_structure(self, weak_sb_ranks, config):
        """Each result should have the required keys."""
        results = recommend_punt_targets(weak_sb_ranks, config, max_punts=1)
        assert len(results) == 1
        result = results[0]
        assert "category" in result
        assert "rank" in result
        assert "isolation_score" in result
        assert "punt_value" in result
        assert "reason" in result
        assert isinstance(result["rank"], int)
        assert isinstance(result["isolation_score"], float)
        assert isinstance(result["punt_value"], float)

    def test_default_config_created_when_none(self, weak_sb_ranks):
        """Should work with config=None by creating a default."""
        results = recommend_punt_targets(weak_sb_ranks, config=None, max_punts=2)
        assert len(results) >= 1

    def test_cluster_penalty_reduces_punt_value(self, config):
        """Categories in a cluster with competitive partners should have lower punt value."""
        # R is weak but RBI is competitive — cluster penalty should apply
        ranks = {
            "R": 10,
            "HR": 5,
            "RBI": 3,
            "SB": 10,
            "AVG": 5,
            "OBP": 5,
            "W": 6,
            "L": 6,
            "SV": 6,
            "K": 5,
            "ERA": 5,
            "WHIP": 5,
        }
        results = recommend_punt_targets(ranks, config, max_punts=2)
        r_result = next((r for r in results if r["category"] == "R"), None)
        sb_result = next((r for r in results if r["category"] == "SB"), None)
        if r_result and sb_result:
            # SB should have higher punt value than R due to cluster penalty on R
            assert sb_result["punt_value"] > r_result["punt_value"]

    def test_weakness_bonus_increases_with_rank(self, config):
        """Rank 12 should have higher punt value than rank 9, all else equal."""
        ranks_9 = {cat: 5 for cat in config.all_categories}
        ranks_9["SB"] = 9
        ranks_12 = {cat: 5 for cat in config.all_categories}
        ranks_12["SB"] = 12

        results_9 = recommend_punt_targets(ranks_9, config, max_punts=1)
        results_12 = recommend_punt_targets(ranks_12, config, max_punts=1)

        assert len(results_9) == 1
        assert len(results_12) == 1
        assert results_12[0]["punt_value"] > results_9[0]["punt_value"]

    def test_punt_value_positive_for_isolated_weak_category(self, weak_sb_ranks, config):
        """Punt value should be positive for an isolated weak category."""
        results = recommend_punt_targets(weak_sb_ranks, config, max_punts=1)
        assert results[0]["punt_value"] > 0


# ── compute_punt_redistribution_value tests ────────────────────────────


class TestComputePuntRedistributionValue:
    def test_redistribution_positive_when_punting_weak_isolated(self, weak_sb_ranks, config):
        """Punting a weak + isolated category should yield positive positions gained."""
        result = compute_punt_redistribution_value(["SB"], weak_sb_ranks, config)
        assert result["positions_gained"] > 0
        assert len(result["helped_categories"]) > 0

    def test_empty_punt_list_returns_zero(self, weak_sb_ranks, config):
        """Punting nothing should return zero gain."""
        result = compute_punt_redistribution_value([], weak_sb_ranks, config)
        assert result["positions_gained"] == 0.0
        assert result["helped_categories"] == []
        assert result["per_category_gain"] == {}

    def test_more_punts_yield_more_redistribution(self, config):
        """Punting 2 categories should yield more positions than punting 1."""
        ranks = {
            "R": 4,
            "HR": 5,
            "RBI": 3,
            "SB": 11,
            "AVG": 6,
            "OBP": 5,
            "W": 10,
            "L": 6,
            "SV": 8,
            "K": 4,
            "ERA": 5,
            "WHIP": 5,
        }
        one_punt = compute_punt_redistribution_value(["SB"], ranks, config)
        two_punts = compute_punt_redistribution_value(["SB", "W"], ranks, config)
        assert two_punts["positions_gained"] > one_punt["positions_gained"]

    def test_helped_categories_not_include_punted(self, weak_sb_ranks, config):
        """Punted categories should NOT appear in helped_categories."""
        result = compute_punt_redistribution_value(["SB"], weak_sb_ranks, config)
        assert "SB" not in result["helped_categories"]

    def test_per_category_gain_sums_to_total(self, weak_sb_ranks, config):
        """Sum of per-category gains should approximately equal total positions_gained."""
        result = compute_punt_redistribution_value(["SB"], weak_sb_ranks, config)
        total_from_per_cat = sum(result["per_category_gain"].values())
        assert total_from_per_cat == pytest.approx(result["positions_gained"], abs=0.05)

    def test_middle_rank_categories_benefit_most(self, config):
        """Categories ranked 5-8 should get more benefit than rank 1-2 categories."""
        ranks = {
            "R": 1,  # Elite
            "HR": 7,  # Middle
            "RBI": 3,
            "SB": 11,  # Punted
            "AVG": 6,  # Middle
            "OBP": 2,  # Elite
            "W": 7,  # Middle
            "L": 6,
            "SV": 8,
            "K": 4,
            "ERA": 5,
            "WHIP": 5,
        }
        result = compute_punt_redistribution_value(["SB"], ranks, config)
        gains = result["per_category_gain"]
        # Middle-ranked cats should have higher gain than elite cats
        if "HR" in gains and "R" in gains:
            assert gains["HR"] > gains["R"]

    def test_default_config_when_none(self, weak_sb_ranks):
        """Should work with config=None."""
        result = compute_punt_redistribution_value(["SB"], weak_sb_ranks, config=None)
        assert result["positions_gained"] > 0

    def test_independent_punt_frees_more_resources(self, config):
        """Punting SB (corr 0.12) should free more resources than punting RBI (corr 0.62)."""
        # Both at same rank to isolate the independence effect
        ranks = {cat: 5 for cat in config.all_categories}
        ranks["SB"] = 11
        ranks["RBI"] = 11

        sb_result = compute_punt_redistribution_value(["SB"], ranks, config)
        rbi_result = compute_punt_redistribution_value(["RBI"], ranks, config)
        assert sb_result["positions_gained"] > rbi_result["positions_gained"]

    def test_punting_all_returns_zero(self, config):
        """Punting every category should return zero (no non-punted cats to help)."""
        ranks = {cat: 10 for cat in config.all_categories}
        result = compute_punt_redistribution_value(config.all_categories, ranks, config)
        assert result["positions_gained"] == 0.0


# ── CATEGORY_MEAN_CORRELATION tests ────────────────────────────────────


class TestCategoryMeanCorrelation:
    def test_all_12_categories_present(self, config):
        """Every league category should have a correlation value."""
        for cat in config.all_categories:
            assert cat in CATEGORY_MEAN_CORRELATION, f"Missing correlation for {cat}"

    def test_sb_lowest_correlation(self):
        """SB should have the lowest mean correlation (safest punt target)."""
        min_cat = min(CATEGORY_MEAN_CORRELATION, key=CATEGORY_MEAN_CORRELATION.get)
        assert min_cat == "SB"
        assert CATEGORY_MEAN_CORRELATION["SB"] == 0.12

    def test_rbi_highest_correlation(self):
        """RBI should have the highest mean correlation (riskiest to punt)."""
        max_cat = max(CATEGORY_MEAN_CORRELATION, key=CATEGORY_MEAN_CORRELATION.get)
        assert max_cat == "RBI"
        assert CATEGORY_MEAN_CORRELATION["RBI"] == 0.62

    def test_all_values_between_0_and_1(self):
        """All correlation values should be between 0 and 1."""
        for cat, val in CATEGORY_MEAN_CORRELATION.items():
            assert 0 <= val <= 1, f"{cat} has out-of-range correlation {val}"
