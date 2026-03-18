"""Tests for DraftRecommendationEngine._enrich_output().

Covers:
  - composite_value normalized to [0, 100]
  - Single candidate gets composite_value = 50
  - position_rank contains all eligible positions
  - Multi-position players ranked in each position
  - overall_rank is sequential starting at 1
  - HIGH confidence for low CV
  - MEDIUM confidence for moderate CV
  - LOW confidence for high CV or zero mean
  - Empty DataFrame handled gracefully
  - NaN / missing mc columns produce LOW confidence
  - Position rank with single-position players
  - Composite value boundary values (min=0, max=100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.draft_engine import DraftRecommendationEngine
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def config():
    """Standard 12-team H2H league config."""
    return LeagueConfig()


@pytest.fixture
def engine(config):
    """Standard-mode engine."""
    return DraftRecommendationEngine(config, mode="standard")


def _make_candidates(rows: list[dict]) -> pd.DataFrame:
    """Build a candidates DataFrame mimicking evaluate_candidates output."""
    defaults = {
        "player_id": 0,
        "name": "Unknown",
        "positions": "Util",
        "combined_score": 0.0,
        "mc_mean_sgp": 1.0,
        "mc_std_sgp": 0.1,
        "pick_score": 0.0,
    }
    filled = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        filled.append(row)
    df = pd.DataFrame(filled)
    return df.sort_values("combined_score", ascending=False).reset_index(drop=True)


# ── Tests: overall_rank ──────────────────────────────────────────


class TestOverallRank:
    def test_sequential_ranking(self, engine):
        """overall_rank should be 1, 2, 3... for sorted candidates."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "A", "combined_score": 10.0},
                {"player_id": 2, "name": "B", "combined_score": 8.0},
                {"player_id": 3, "name": "C", "combined_score": 5.0},
            ]
        )
        result = engine._enrich_output(df)
        assert list(result["overall_rank"]) == [1, 2, 3]

    def test_single_candidate_rank_is_one(self, engine):
        """A single candidate should have overall_rank = 1."""
        df = _make_candidates([{"player_id": 1, "name": "Solo", "combined_score": 7.0}])
        result = engine._enrich_output(df)
        assert result.iloc[0]["overall_rank"] == 1


# ── Tests: composite_value ───────────────────────────────────────


class TestCompositeValue:
    def test_normalized_to_0_100(self, engine):
        """composite_value should range from 0 to 100."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "Top", "combined_score": 20.0},
                {"player_id": 2, "name": "Mid", "combined_score": 10.0},
                {"player_id": 3, "name": "Bot", "combined_score": 5.0},
            ]
        )
        result = engine._enrich_output(df)
        values = result["composite_value"].tolist()
        assert values[0] == pytest.approx(100.0)
        assert values[-1] == pytest.approx(0.0)
        # Middle value: (10 - 5) / (20 - 5) * 100 = 33.33
        assert values[1] == pytest.approx(33.333, abs=0.01)

    def test_single_candidate_gets_50(self, engine):
        """When only one candidate, composite_value should be 50.0."""
        df = _make_candidates([{"player_id": 1, "name": "Solo", "combined_score": 12.0}])
        result = engine._enrich_output(df)
        assert result.iloc[0]["composite_value"] == pytest.approx(50.0)

    def test_all_equal_scores_get_50(self, engine):
        """When all candidates have the same score, all get 50.0."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "A", "combined_score": 7.0},
                {"player_id": 2, "name": "B", "combined_score": 7.0},
                {"player_id": 3, "name": "C", "combined_score": 7.0},
            ]
        )
        result = engine._enrich_output(df)
        for val in result["composite_value"]:
            assert val == pytest.approx(50.0)

    def test_boundary_values(self, engine):
        """Max score gets 100, min score gets 0."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "Best", "combined_score": 100.0},
                {"player_id": 2, "name": "Worst", "combined_score": 0.0},
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["composite_value"] == pytest.approx(100.0)
        assert result.iloc[1]["composite_value"] == pytest.approx(0.0)


# ── Tests: position_rank ─────────────────────────────────────────


class TestPositionRank:
    def test_single_position_players(self, engine):
        """Single-position players get a single position rank."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "A", "positions": "SS", "combined_score": 10.0},
                {"player_id": 2, "name": "B", "positions": "SS", "combined_score": 8.0},
                {"player_id": 3, "name": "C", "positions": "1B", "combined_score": 6.0},
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["position_rank"] == "SS:1"
        assert result.iloc[1]["position_rank"] == "SS:2"
        assert result.iloc[2]["position_rank"] == "1B:1"

    def test_multi_position_ranked_in_each(self, engine):
        """Multi-position players appear in each position's ranking."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "A", "positions": "SS,2B", "combined_score": 10.0},
                {"player_id": 2, "name": "B", "positions": "2B", "combined_score": 8.0},
                {"player_id": 3, "name": "C", "positions": "SS", "combined_score": 6.0},
            ]
        )
        result = engine._enrich_output(df)
        # Player A is SS:1 and 2B:1
        pr_a = result.iloc[0]["position_rank"]
        assert "SS:1" in pr_a
        assert "2B:1" in pr_a
        # Player B is 2B:2 (second 2B after A)
        assert result.iloc[1]["position_rank"] == "2B:2"
        # Player C is SS:2 (second SS after A)
        assert result.iloc[2]["position_rank"] == "SS:2"

    def test_position_rank_contains_all_positions(self, engine):
        """A player with 3 positions should have all 3 in position_rank."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "Utility",
                    "positions": "1B,3B,OF",
                    "combined_score": 10.0,
                },
            ]
        )
        result = engine._enrich_output(df)
        pr = result.iloc[0]["position_rank"]
        assert "1B:1" in pr
        assert "3B:1" in pr
        assert "OF:1" in pr

    def test_empty_positions_string(self, engine):
        """Empty or missing positions produce empty position_rank."""
        df = _make_candidates(
            [
                {"player_id": 1, "name": "A", "positions": "", "combined_score": 10.0},
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["position_rank"] == ""


# ── Tests: confidence_level ──────────────────────────────────────


class TestConfidenceLevel:
    def test_high_confidence_low_cv(self, engine):
        """CV < 0.15 -> HIGH confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "Steady",
                    "combined_score": 10.0,
                    "mc_mean_sgp": 10.0,
                    "mc_std_sgp": 1.0,  # CV = 0.10
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "HIGH"

    def test_medium_confidence_moderate_cv(self, engine):
        """0.15 <= CV < 0.35 -> MEDIUM confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "Moderate",
                    "combined_score": 10.0,
                    "mc_mean_sgp": 10.0,
                    "mc_std_sgp": 2.5,  # CV = 0.25
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "MEDIUM"

    def test_low_confidence_high_cv(self, engine):
        """CV >= 0.35 -> LOW confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "Volatile",
                    "combined_score": 10.0,
                    "mc_mean_sgp": 5.0,
                    "mc_std_sgp": 3.0,  # CV = 0.60
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "LOW"

    def test_low_confidence_zero_mean(self, engine):
        """mc_mean_sgp near zero -> LOW confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "Zero",
                    "combined_score": 10.0,
                    "mc_mean_sgp": 0.0,
                    "mc_std_sgp": 1.0,
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "LOW"

    def test_low_confidence_nan_mean(self, engine):
        """NaN mc_mean_sgp -> LOW confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "NaN",
                    "combined_score": 10.0,
                    "mc_mean_sgp": float("nan"),
                    "mc_std_sgp": 1.0,
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "LOW"

    def test_low_confidence_missing_mc_columns(self, engine):
        """Missing mc columns entirely -> LOW confidence."""
        df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "NoMC",
                    "positions": "SS",
                    "combined_score": 10.0,
                    "pick_score": 5.0,
                }
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "LOW"


# ── Tests: empty DataFrame ───────────────────────────────────────


class TestEmptyDataFrame:
    def test_empty_input_returns_empty_with_columns(self, engine):
        """Empty DataFrame returns empty with all 4 new columns."""
        df = pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "positions",
                "combined_score",
                "mc_mean_sgp",
                "mc_std_sgp",
                "pick_score",
            ]
        )
        result = engine._enrich_output(df)
        assert len(result) == 0
        assert "overall_rank" in result.columns
        assert "composite_value" in result.columns
        assert "position_rank" in result.columns
        assert "confidence_level" in result.columns


# ── Tests: integration with multiple features ────────────────────


class TestEnrichIntegration:
    def test_all_columns_present(self, engine):
        """All four enrichment columns should be present after enrichment."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "A",
                    "positions": "SS,2B",
                    "combined_score": 15.0,
                    "mc_mean_sgp": 12.0,
                    "mc_std_sgp": 1.0,
                },
                {
                    "player_id": 2,
                    "name": "B",
                    "positions": "OF",
                    "combined_score": 10.0,
                    "mc_mean_sgp": 8.0,
                    "mc_std_sgp": 3.5,
                },
            ]
        )
        result = engine._enrich_output(df)
        assert "overall_rank" in result.columns
        assert "composite_value" in result.columns
        assert "position_rank" in result.columns
        assert "confidence_level" in result.columns

        # Verify types
        assert result["overall_rank"].dtype in (np.int64, int)
        assert result["composite_value"].dtype == np.float64
        assert result["confidence_level"].iloc[0] in ("HIGH", "MEDIUM", "LOW")

    def test_negative_mean_high_confidence(self, engine):
        """Negative mc_mean_sgp with low CV should be HIGH confidence."""
        df = _make_candidates(
            [
                {
                    "player_id": 1,
                    "name": "NegMean",
                    "combined_score": 5.0,
                    "mc_mean_sgp": -10.0,
                    "mc_std_sgp": 1.0,  # CV = 1.0/10.0 = 0.10
                },
            ]
        )
        result = engine._enrich_output(df)
        assert result.iloc[0]["confidence_level"] == "HIGH"
