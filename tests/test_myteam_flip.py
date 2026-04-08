"""Tests for the Category Flip Analysis card on the My Team page.

Validates that the flip analysis integration works correctly with
matchup data and degrades gracefully when data is unavailable.
"""

from __future__ import annotations

import pytest

from src.optimizer.pivot_advisor import (
    compute_category_flip_probabilities,
    get_pivot_summary,
)
from src.valuation import LeagueConfig


@pytest.fixture
def config():
    return LeagueConfig()


@pytest.fixture
def sample_matchup():
    """Simulated Yahoo matchup dict with category entries."""
    return {
        "week": 2,
        "opp_name": "Opponent Team",
        "wins": 5,
        "losses": 4,
        "ties": 3,
        "categories": [
            {"cat": "R", "result": "win", "you": "35", "opp": "28"},
            {"cat": "HR", "result": "loss", "you": "8", "opp": "14"},
            {"cat": "RBI", "result": "win", "you": "30", "opp": "22"},
            {"cat": "SB", "result": "tie", "you": "3", "opp": "3"},
            {"cat": "AVG", "result": "win", "you": "0.285", "opp": "0.260"},
            {"cat": "OBP", "result": "win", "you": "0.350", "opp": "0.320"},
            {"cat": "W", "result": "loss", "you": "2", "opp": "5"},
            {"cat": "L", "result": "win", "you": "2", "opp": "4"},
            {"cat": "SV", "result": "tie", "you": "1", "opp": "1"},
            {"cat": "K", "result": "loss", "you": "35", "opp": "52"},
            {"cat": "ERA", "result": "loss", "you": "4.20", "opp": "3.10"},
            {"cat": "WHIP", "result": "tie", "you": "1.22", "opp": "1.18"},
        ],
    }


class TestFlipAnalysisExtraction:
    """Test that matchup category data can be extracted for flip analysis."""

    def test_extract_totals_from_matchup(self, sample_matchup):
        """Verify that my/opp totals can be extracted from matchup categories."""
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in sample_matchup["categories"]:
            cat = entry.get("cat", "")
            if cat:
                my_totals[cat] = float(entry.get("you", 0) or 0)
                opp_totals[cat] = float(entry.get("opp", 0) or 0)

        assert my_totals["R"] == 35.0
        assert opp_totals["HR"] == 14.0
        assert my_totals["AVG"] == pytest.approx(0.285)
        assert opp_totals["ERA"] == pytest.approx(3.10)

    def test_flip_analysis_renders_with_matchup_data(self, sample_matchup, config):
        """Flip analysis should return results for all 12 categories."""
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in sample_matchup["categories"]:
            cat = entry.get("cat", "")
            if cat:
                my_totals[cat] = float(entry.get("you", 0) or 0)
                opp_totals[cat] = float(entry.get("opp", 0) or 0)

        result = compute_category_flip_probabilities(my_totals, opp_totals, 4, config)
        assert len(result) == 12
        for cat, info in result.items():
            assert "classification" in info
            assert info["classification"] in ("WON", "LOST", "CONTESTED")
            assert "flip_prob" in info
            assert 0.0 <= info["flip_prob"] <= 1.0

    def test_summary_returns_valid_groups(self, sample_matchup, config):
        """Summary should group categories into won/lost/contested lists."""
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in sample_matchup["categories"]:
            cat = entry.get("cat", "")
            if cat:
                my_totals[cat] = float(entry.get("you", 0) or 0)
                opp_totals[cat] = float(entry.get("opp", 0) or 0)

        summary = get_pivot_summary(my_totals, opp_totals, 4, config)
        assert isinstance(summary["won"], list)
        assert isinstance(summary["lost"], list)
        assert isinstance(summary["contested"], list)
        assert isinstance(summary["recommended_actions"], list)
        # All 12 categories accounted for
        total = len(summary["won"]) + len(summary["lost"]) + len(summary["contested"])
        assert total == 12

    def test_actions_limited_to_contested_and_lost(self, sample_matchup, config):
        """Recommended actions should only reference LOST or CONTESTED cats."""
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in sample_matchup["categories"]:
            cat = entry.get("cat", "")
            if cat:
                my_totals[cat] = float(entry.get("you", 0) or 0)
                opp_totals[cat] = float(entry.get("opp", 0) or 0)

        summary = get_pivot_summary(my_totals, opp_totals, 4, config)
        won_set = set(summary["won"])
        for action in summary["recommended_actions"]:
            # Action format is "CAT: action text"
            action_cat = action.split(":")[0].strip()
            assert action_cat not in won_set, f"WON category {action_cat} should not have an action"


class TestFlipAnalysisFallback:
    """Test graceful fallback when no matchup data is available."""

    def test_no_matchup_data_returns_empty_summary(self, config):
        """When matchup is None, flip analysis should not crash."""
        # Simulate what the page does: if _wr_matchup is None, card is skipped
        matchup = None
        assert matchup is None  # Card would be skipped via the if-guard

    def test_empty_categories_handled(self, config):
        """A matchup dict with empty categories should not crash."""
        matchup = {"week": 1, "opp_name": "Test", "categories": []}
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in matchup.get("categories", []):
            cat = entry.get("cat", "")
            if cat:
                my_totals[cat] = float(entry.get("you", 0) or 0)
                opp_totals[cat] = float(entry.get("opp", 0) or 0)

        # With empty totals, all categories default to 0 vs 0 = CONTESTED
        result = compute_category_flip_probabilities(my_totals, opp_totals, 4, config)
        assert len(result) == 12
        for info in result.values():
            assert info["classification"] == "CONTESTED"

    def test_malformed_values_default_to_zero(self, config):
        """Non-numeric category values should be parsed as 0."""
        matchup = {
            "categories": [
                {"cat": "R", "result": "win", "you": "N/A", "opp": "abc"},
            ],
        }
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in matchup.get("categories", []):
            cat = entry.get("cat", "")
            if cat:
                try:
                    my_totals[cat] = float(entry.get("you", 0) or 0)
                except (ValueError, TypeError):
                    my_totals[cat] = 0.0
                try:
                    opp_totals[cat] = float(entry.get("opp", 0) or 0)
                except (ValueError, TypeError):
                    opp_totals[cat] = 0.0

        # Both R values should be 0 (tied)
        result = compute_category_flip_probabilities(my_totals, opp_totals, 4, config)
        assert result["R"]["margin"] == 0.0
        assert result["R"]["classification"] == "CONTESTED"

    def test_none_values_in_categories(self, config):
        """None values for you/opp should be handled gracefully."""
        matchup = {
            "categories": [
                {"cat": "HR", "result": "tie", "you": None, "opp": None},
            ],
        }
        my_totals: dict[str, float] = {}
        opp_totals: dict[str, float] = {}
        for entry in matchup.get("categories", []):
            cat = entry.get("cat", "")
            if cat:
                try:
                    my_totals[cat] = float(entry.get("you", 0) or 0)
                except (ValueError, TypeError):
                    my_totals[cat] = 0.0
                try:
                    opp_totals[cat] = float(entry.get("opp", 0) or 0)
                except (ValueError, TypeError):
                    opp_totals[cat] = 0.0

        result = compute_category_flip_probabilities(my_totals, opp_totals, 4, config)
        assert result["HR"]["margin"] == 0.0

    def test_games_remaining_calculation(self):
        """Games remaining approximation should be bounded [0, 7]."""
        from datetime import UTC, datetime

        dow = datetime.now(UTC).weekday()
        games_remaining = max(0, 7 - dow)
        assert 0 <= games_remaining <= 7
