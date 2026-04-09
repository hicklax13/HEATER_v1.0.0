"""Tests for src/weekly_h2h_strategy.py — weekly H2H matchup strategy engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.valuation import LeagueConfig
from src.weekly_h2h_strategy import (
    CLOSEABLE_FRACTION,
    PUNT_COUNTING_FRACTION,
    PUNT_RATE_STDS,
    RATE_CLOSEABLE_STDS,
    RATE_STAT_STD,
    WEEKLY_COUNTING_RATES,
    _closeable_priority,
    _is_closeable,
    _is_puntable,
    _lerp,
    _parse_categories,
    compute_desperation_level,
    compute_weekly_matchup_state,
    generate_weekly_action_plan,
    get_adjusted_trade_params,
    get_category_targets,
    get_dynamic_weeks_remaining,
    get_matchup_aware_fa_criteria,
    get_optimal_alpha,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return LeagueConfig()


@pytest.fixture
def sample_matchup():
    """A realistic Yahoo matchup dict with mixed winning/losing categories."""
    return {
        "week": 3,
        "wins": 5,
        "losses": 4,
        "ties": 3,
        "user_name": "Team Hickey",
        "categories": [
            {"cat": "R", "you": "35", "opp": "30", "result": "WIN"},
            {"cat": "HR", "you": "8", "opp": "12", "result": "LOSS"},
            {"cat": "RBI", "you": "30", "opp": "28", "result": "WIN"},
            {"cat": "SB", "you": "3", "opp": "6", "result": "LOSS"},
            {"cat": "AVG", "you": "0.270", "opp": "0.260", "result": "WIN"},
            {"cat": "OBP", "you": "0.330", "opp": "0.340", "result": "LOSS"},
            {"cat": "W", "you": "2", "opp": "4", "result": "LOSS"},
            {"cat": "L", "you": "3", "opp": "2", "result": "LOSS"},
            {"cat": "SV", "you": "3", "opp": "1", "result": "WIN"},
            {"cat": "K", "you": "45", "opp": "50", "result": "LOSS"},
            {"cat": "ERA", "you": "3.50", "opp": "3.20", "result": "LOSS"},
            {"cat": "WHIP", "you": "1.15", "opp": "1.25", "result": "WIN"},
        ],
    }


@pytest.fixture
def sample_standings():
    """Minimal standings DataFrame."""
    return pd.DataFrame(
        {
            "team_name": ["Team Hickey", "Other Team"],
            "rank": [4, 8],
            "category": ["R", "R"],
            "total": [100, 80],
        }
    )


def _make_matchup_state(
    categories=None,
    winnable_cats=None,
    protect_cats=None,
    punt_cats=None,
    score=(5, 4, 3),
    week=3,
):
    """Helper to build a matchup_state dict for testing downstream functions."""
    if categories is None:
        categories = []
    return {
        "week": week,
        "score": score,
        "record": (10, 8, 0),
        "standings_rank": 4,
        "categories": categories,
        "winnable_cats": winnable_cats or [],
        "protect_cats": protect_cats or [],
        "punt_cats": punt_cats or [],
        "desperation_level": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests: _lerp
# ---------------------------------------------------------------------------


class TestLerp:
    def test_lerp_at_zero(self):
        assert _lerp(10.0, 20.0, 0.0) == 10.0

    def test_lerp_at_one(self):
        assert _lerp(10.0, 20.0, 1.0) == 20.0

    def test_lerp_midpoint(self):
        assert _lerp(0.0, 100.0, 0.5) == 50.0


# ---------------------------------------------------------------------------
# Tests: _is_closeable / _is_puntable
# ---------------------------------------------------------------------------


class TestCloseableAndPuntable:
    def test_closeable_already_winning(self):
        """Gap <= 0 means already winning or tied -- always closeable."""
        assert _is_closeable("HR", -5.0, is_rate=False) is True
        assert _is_closeable("HR", 0.0, is_rate=False) is True

    def test_closeable_counting_small_gap(self):
        """Small gap relative to weekly rate is closeable."""
        weekly = WEEKLY_COUNTING_RATES["HR"]
        small_gap = weekly * CLOSEABLE_FRACTION * 0.5  # well within threshold
        assert _is_closeable("HR", small_gap, is_rate=False) is True

    def test_not_closeable_counting_large_gap(self):
        """Large gap is NOT closeable."""
        weekly = WEEKLY_COUNTING_RATES["HR"]
        large_gap = weekly * CLOSEABLE_FRACTION * 2.0
        assert _is_closeable("HR", large_gap, is_rate=False) is False

    def test_closeable_rate_small_gap(self):
        std = RATE_STAT_STD["ERA"]
        small_gap = std * RATE_CLOSEABLE_STDS * 0.5
        assert _is_closeable("ERA", small_gap, is_rate=True) is True

    def test_not_closeable_rate_large_gap(self):
        std = RATE_STAT_STD["ERA"]
        large_gap = std * RATE_CLOSEABLE_STDS * 3.0
        assert _is_closeable("ERA", large_gap, is_rate=True) is False

    def test_puntable_never_if_winning(self):
        assert _is_puntable("HR", -1.0, is_rate=False) is False
        assert _is_puntable("HR", 0.0, is_rate=False) is False

    def test_puntable_counting_huge_gap(self):
        weekly = WEEKLY_COUNTING_RATES["HR"]
        huge_gap = weekly * PUNT_COUNTING_FRACTION * 1.5
        assert _is_puntable("HR", huge_gap, is_rate=False) is True

    def test_not_puntable_counting_small_gap(self):
        weekly = WEEKLY_COUNTING_RATES["HR"]
        small_gap = weekly * PUNT_COUNTING_FRACTION * 0.5
        assert _is_puntable("HR", small_gap, is_rate=False) is False

    def test_puntable_rate_huge_gap(self):
        std = RATE_STAT_STD["AVG"]
        huge_gap = std * PUNT_RATE_STDS * 2.0
        assert _is_puntable("AVG", huge_gap, is_rate=True) is True


# ---------------------------------------------------------------------------
# Tests: _closeable_priority
# ---------------------------------------------------------------------------


class TestCloseablePriority:
    def test_tied_returns_half(self):
        assert _closeable_priority("HR", 0.0, is_rate=False) == 0.5

    def test_negative_gap_returns_half(self):
        assert _closeable_priority("HR", -5.0, is_rate=False) == 0.5

    def test_small_counting_gap_high_priority(self):
        """A very small gap should yield higher priority than a large gap."""
        p = _closeable_priority("HR", 0.5, is_rate=False)
        assert p > 0.5

    def test_large_counting_gap_low_priority(self):
        """A large gap should yield low priority."""
        weekly = WEEKLY_COUNTING_RATES["HR"]
        p = _closeable_priority("HR", weekly * 0.5, is_rate=False)
        assert p < 0.5

    def test_rate_stat_priority_bounded(self):
        p = _closeable_priority("ERA", 0.5, is_rate=True)
        assert 0.1 <= p <= 0.95


# ---------------------------------------------------------------------------
# Tests: compute_desperation_level
# ---------------------------------------------------------------------------


class TestDesperationLevel:
    def test_comfortable_team(self):
        """Good record, winning week, high rank -> low desperation."""
        d = compute_desperation_level(
            record=(50, 10, 0),
            current_score=(8, 2, 2),
            standings_rank=1,
        )
        assert d < 0.3

    def test_dire_team(self):
        """Bad record, losing week, low rank -> high desperation."""
        d = compute_desperation_level(
            record=(10, 50, 0),
            current_score=(2, 8, 2),
            standings_rank=12,
        )
        assert d > 0.7

    def test_neutral_no_games(self):
        """No decisions yet -> mid-range desperation."""
        d = compute_desperation_level(
            record=(0, 0, 0),
            current_score=(0, 0, 0),
            standings_rank=6,
        )
        assert 0.3 < d < 0.7

    def test_clamped_to_unit_interval(self):
        """Output is always in [0, 1]."""
        d = compute_desperation_level(
            record=(0, 100, 0),
            current_score=(0, 12, 0),
            standings_rank=12,
        )
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Tests: get_dynamic_weeks_remaining
# ---------------------------------------------------------------------------


class TestDynamicWeeksRemaining:
    def test_explicit_week(self):
        assert get_dynamic_weeks_remaining(current_week=10, total_weeks=24) == 14

    def test_explicit_week_past_end(self):
        assert get_dynamic_weeks_remaining(current_week=30, total_weeks=24) == 0

    def test_explicit_week_zero(self):
        assert get_dynamic_weeks_remaining(current_week=0, total_weeks=24) == 24

    def test_auto_detect_returns_nonnegative(self):
        """Without explicit week, result should be >= 0."""
        result = get_dynamic_weeks_remaining()
        assert result >= 0


# ---------------------------------------------------------------------------
# Tests: get_optimal_alpha
# ---------------------------------------------------------------------------


class TestOptimalAlpha:
    def test_bounds(self):
        """Alpha should always be in [0.4, 0.95]."""
        for desp in [0.0, 0.5, 1.0]:
            for wr in [0, 12, 24]:
                a = get_optimal_alpha(desp, wr)
                assert 0.4 <= a <= 0.95, f"alpha={a} for desp={desp}, wr={wr}"

    def test_more_desperate_higher_alpha(self):
        a_low = get_optimal_alpha(0.1, 12)
        a_high = get_optimal_alpha(0.9, 12)
        assert a_high > a_low

    def test_late_season_higher_alpha(self):
        a_early = get_optimal_alpha(0.5, 20)
        a_late = get_optimal_alpha(0.5, 2)
        assert a_late > a_early


# ---------------------------------------------------------------------------
# Tests: get_adjusted_trade_params
# ---------------------------------------------------------------------------


class TestAdjustedTradeParams:
    def test_returns_all_keys(self):
        params = get_adjusted_trade_params(0.5)
        assert "elite_return_floor" in params
        assert "max_weight_ratio" in params
        assert "loss_aversion" in params
        assert "max_opp_loss" in params

    def test_comfortable_vs_desperate(self):
        calm = get_adjusted_trade_params(0.0)
        dire = get_adjusted_trade_params(1.0)
        # When desperate: lower return floor, higher weight ratio, lower loss aversion
        assert dire["elite_return_floor"] < calm["elite_return_floor"]
        assert dire["max_weight_ratio"] > calm["max_weight_ratio"]
        assert dire["loss_aversion"] < calm["loss_aversion"]

    def test_clamped_input(self):
        """Values outside [0, 1] should be clamped."""
        p1 = get_adjusted_trade_params(-5.0)
        p2 = get_adjusted_trade_params(0.0)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Tests: _parse_categories
# ---------------------------------------------------------------------------


class TestParseCategories:
    def test_no_matchup_returns_defaults(self, config):
        result = _parse_categories(None, config)
        assert len(result) == len(config.all_categories)
        for cat_info in result:
            assert cat_info["gap"] == 0.0
            assert cat_info["status"] == "tied"

    def test_parses_win_correctly(self, config):
        matchup = {
            "categories": [
                {"cat": "HR", "you": "10", "opp": "5", "result": "WIN"},
            ]
        }
        result = _parse_categories(matchup, config)
        assert len(result) == 1
        hr = result[0]
        assert hr["name"] == "HR"
        assert hr["user_val"] == 10.0
        assert hr["opp_val"] == 5.0
        assert hr["status"] == "winning"
        # HR is higher-is-better: gap = opp - user = -5 (negative = ahead)
        assert hr["gap"] == -5.0

    def test_inverse_stat_gap_direction(self, config):
        """For inverse stats (ERA), gap = user - opp (positive = user is worse)."""
        matchup = {
            "categories": [
                {"cat": "ERA", "you": "4.00", "opp": "3.00", "result": "LOSS"},
            ]
        }
        result = _parse_categories(matchup, config)
        era = result[0]
        assert era["is_inverse"] is True
        assert era["gap"] == 1.0  # user ERA higher = worse

    def test_dash_values_treated_as_zero(self, config):
        matchup = {
            "categories": [
                {"cat": "AVG", "you": "-", "opp": "-", "result": "TIE"},
            ]
        }
        result = _parse_categories(matchup, config)
        assert result[0]["user_val"] == 0.0
        assert result[0]["opp_val"] == 0.0

    def test_empty_categories_list(self, config):
        matchup = {"categories": []}
        result = _parse_categories(matchup, config)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: get_category_targets
# ---------------------------------------------------------------------------


class TestGetCategoryTargets:
    def test_returns_all_buckets(self):
        state = _make_matchup_state()
        targets = get_category_targets(state)
        assert "must_win" in targets
        assert "protect" in targets
        assert "concede" in targets
        assert "swing" in targets

    def test_protect_cats_go_to_protect(self):
        cats = [
            {"name": "R", "gap": -5.0, "status": "winning", "is_rate": False, "is_inverse": False},
        ]
        state = _make_matchup_state(categories=cats, protect_cats=["R"])
        targets = get_category_targets(state)
        assert len(targets["protect"]) == 1
        assert targets["protect"][0]["name"] == "R"

    def test_punt_cats_go_to_concede(self):
        cats = [
            {"name": "SB", "gap": 15.0, "status": "losing", "is_rate": False, "is_inverse": False},
        ]
        state = _make_matchup_state(categories=cats, punt_cats=["SB"])
        targets = get_category_targets(state)
        assert len(targets["concede"]) == 1
        assert targets["concede"][0]["name"] == "SB"

    def test_winnable_losing_cats_go_to_must_win(self):
        cats = [
            {"name": "HR", "gap": 2.0, "status": "losing", "is_rate": False, "is_inverse": False},
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["HR"])
        targets = get_category_targets(state)
        assert len(targets["must_win"]) == 1
        assert targets["must_win"][0]["name"] == "HR"

    def test_tied_cats_go_to_swing(self):
        cats = [
            {"name": "K", "gap": 0.0, "status": "tied", "is_rate": False, "is_inverse": False},
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["K"])
        targets = get_category_targets(state)
        assert len(targets["swing"]) == 1

    def test_slim_lead_gets_high_protect_priority(self):
        """A slim counting-stat lead should get priority 0.8."""
        weekly = WEEKLY_COUNTING_RATES["R"]
        slim_gap = weekly * 0.05  # very slim
        cats = [
            {"name": "R", "gap": -slim_gap, "status": "winning", "is_rate": False, "is_inverse": False},
        ]
        state = _make_matchup_state(categories=cats, protect_cats=["R"])
        targets = get_category_targets(state)
        assert targets["protect"][0]["priority_score"] == 0.8

    def test_rate_stat_protect_message(self):
        cats = [
            {"name": "ERA", "gap": -0.3, "status": "winning", "is_rate": True, "is_inverse": True},
        ]
        state = _make_matchup_state(categories=cats, protect_cats=["ERA"])
        targets = get_category_targets(state)
        assert "risky pitcher" in targets["protect"][0]["suggested_action"]


# ---------------------------------------------------------------------------
# Tests: get_matchup_aware_fa_criteria
# ---------------------------------------------------------------------------


class TestMatchupAwareFACriteria:
    def test_empty_winnable_returns_empty(self):
        state = _make_matchup_state(winnable_cats=[])
        result = get_matchup_aware_fa_criteria(state)
        assert result == []

    def test_counting_stat_criteria(self):
        cats = [
            {
                "name": "HR",
                "gap": 4.0,
                "status": "losing",
                "is_rate": False,
                "is_inverse": False,
                "user_val": 8,
                "opp_val": 12,
            },
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["HR"])
        result = get_matchup_aware_fa_criteria(state, days_remaining=5)
        assert len(result) == 1
        assert result[0]["category"] == "HR"
        assert "need 4 more" in result[0]["reason"]

    def test_rate_stat_inverse_criteria(self):
        cats = [
            {
                "name": "ERA",
                "gap": 0.3,
                "status": "losing",
                "is_rate": True,
                "is_inverse": True,
                "user_val": 3.50,
                "opp_val": 3.20,
            },
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["ERA"])
        result = get_matchup_aware_fa_criteria(state)
        assert len(result) == 1
        assert "low-ERA" in result[0]["reason"]

    def test_rate_stat_forward_criteria(self):
        cats = [
            {
                "name": "AVG",
                "gap": 0.010,
                "status": "losing",
                "is_rate": True,
                "is_inverse": False,
                "user_val": 0.260,
                "opp_val": 0.270,
            },
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["AVG"])
        result = get_matchup_aware_fa_criteria(state)
        assert len(result) == 1
        assert "high-AVG" in result[0]["reason"]

    def test_sorted_by_priority_descending(self):
        cats = [
            {
                "name": "HR",
                "gap": 1.0,
                "status": "losing",
                "is_rate": False,
                "is_inverse": False,
                "user_val": 11,
                "opp_val": 12,
            },
            {
                "name": "SB",
                "gap": 8.0,
                "status": "losing",
                "is_rate": False,
                "is_inverse": False,
                "user_val": 2,
                "opp_val": 10,
            },
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["HR", "SB"])
        result = get_matchup_aware_fa_criteria(state)
        assert len(result) == 2
        # Smaller gap (HR) should have higher priority
        assert result[0]["priority"] >= result[1]["priority"]


# ---------------------------------------------------------------------------
# Tests: compute_weekly_matchup_state (integration-like, mocked external)
# ---------------------------------------------------------------------------


class TestComputeWeeklyMatchupState:
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    @patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=None)
    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((0, 0, 0), 6))
    def test_no_yds_returns_defaults(self, _mock_rec, _mock_matchup, _mock_standings):
        result = compute_weekly_matchup_state(yds=None)
        assert result["week"] == 0
        assert result["score"] == (0, 0, 0)
        assert isinstance(result["categories"], list)
        assert len(result["categories"]) == 12  # 6 hitting + 6 pitching
        assert 0.0 <= result["desperation_level"] <= 1.0

    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((10, 8, 0), 4))
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    def test_with_matchup_data(self, _mock_standings, _mock_rec, sample_matchup):
        with patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=sample_matchup):
            result = compute_weekly_matchup_state(yds=MagicMock())

        assert result["week"] == 3
        assert result["score"] == (5, 4, 3)
        assert len(result["categories"]) == 12
        # R is winning -> should be in protect
        assert "R" in result["protect_cats"]

    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((0, 0, 0), 6))
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    @patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=None)
    def test_all_categories_classified(self, _mock_m, _mock_s, _mock_r):
        """Every category ends up in exactly one of winnable/protect/punt."""
        result = compute_weekly_matchup_state(yds=None)
        classified = set(result["winnable_cats"] + result["protect_cats"] + result["punt_cats"])
        all_cat_names = {c["name"] for c in result["categories"]}
        assert classified == all_cat_names


# ---------------------------------------------------------------------------
# Tests: generate_weekly_action_plan (full pipeline, fully mocked)
# ---------------------------------------------------------------------------


class TestGenerateWeeklyActionPlan:
    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((10, 8, 0), 5))
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    @patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=None)
    def test_full_plan_structure(self, _mock_m, _mock_s, _mock_r):
        plan = generate_weekly_action_plan(yds=None)
        assert "matchup_state" in plan
        assert "desperation_level" in plan
        assert "alpha" in plan
        assert "weeks_remaining" in plan
        assert "category_targets" in plan
        assert "trade_params" in plan
        assert "fa_criteria" in plan
        assert "lineup_priorities" in plan
        assert "summary" in plan
        # Alpha should be bounded
        assert 0.4 <= plan["alpha"] <= 0.95
        # Summary should be a list of strings
        assert isinstance(plan["summary"], list)
        assert all(isinstance(s, str) for s in plan["summary"])

    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((10, 8, 0), 5))
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    @patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=None)
    def test_lineup_priorities_capped_at_six(self, _mock_m, _mock_s, _mock_r):
        plan = generate_weekly_action_plan(yds=None)
        assert len(plan["lineup_priorities"]) <= 6

    @patch("src.weekly_h2h_strategy._extract_record_and_rank", return_value=((10, 8, 0), 5))
    @patch("src.weekly_h2h_strategy._safe_get_standings", return_value=None)
    @patch("src.weekly_h2h_strategy._safe_get_matchup", return_value=None)
    def test_summary_capped_at_five(self, _mock_m, _mock_s, _mock_r):
        plan = generate_weekly_action_plan(yds=None)
        assert len(plan["summary"]) <= 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_desperation_zero_decisions(self):
        """Zero season + zero weekly -> neutral."""
        d = compute_desperation_level((0, 0, 0), (0, 0, 0), 6)
        assert 0.4 <= d <= 0.6

    def test_alpha_at_season_boundary(self):
        """Week 24 of 24 (0 weeks remaining) shouldn't crash."""
        a = get_optimal_alpha(0.5, weeks_remaining=0, total_weeks=24)
        assert 0.4 <= a <= 0.95

    def test_weeks_remaining_negative_week(self):
        """Negative current_week => max weeks remaining (clamped at total)."""
        # current_week = -1 -> total - (-1) = 25, but that's fine, it's max(0, 25) = 25
        # The function just does total_weeks - current_week
        result = get_dynamic_weeks_remaining(current_week=-1, total_weeks=24)
        assert result == 25  # no clamping to total, just max(0, ...)

    def test_fa_criteria_zero_days_remaining(self):
        """0 days remaining should not crash (division guarded)."""
        cats = [
            {
                "name": "HR",
                "gap": 4.0,
                "status": "losing",
                "is_rate": False,
                "is_inverse": False,
                "user_val": 8,
                "opp_val": 12,
            },
        ]
        state = _make_matchup_state(categories=cats, winnable_cats=["HR"])
        # Should not raise ZeroDivisionError
        result = get_matchup_aware_fa_criteria(state, days_remaining=0)
        assert isinstance(result, list)

    def test_category_targets_empty_state(self):
        """Empty categories -> all buckets empty."""
        state = _make_matchup_state()
        targets = get_category_targets(state)
        assert targets["must_win"] == []
        assert targets["protect"] == []
        assert targets["concede"] == []
        assert targets["swing"] == []

    def test_trade_params_midpoint(self):
        """Desperation 0.5 should interpolate to midpoints."""
        params = get_adjusted_trade_params(0.5)
        assert abs(params["elite_return_floor"] - 0.625) < 0.01
        assert abs(params["max_weight_ratio"] - 2.0) < 0.01
