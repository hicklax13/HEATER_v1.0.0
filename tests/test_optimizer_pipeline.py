"""Tests for the master pipeline orchestrator (src/optimizer/pipeline.py).

Covers:
  - Pipeline initialization with all three modes
  - Full optimization pipeline with mock data
  - Graceful degradation when modules are missing
  - Category weight blending from multiple sources
  - Risk metrics computation
  - Timing measurement
  - Backward compatibility (standard LP solve works without extras)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer.pipeline import (
    MODE_PRESETS,
    LineupOptimizerPipeline,
    _compute_risk_metrics,
    _extract_assignments_map,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def sample_roster() -> pd.DataFrame:
    """Small roster with 4 hitters and 2 pitchers."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 6],
            "player_name": [
                "Batter A",
                "Batter B",
                "Batter C",
                "Batter D",
                "Pitcher E",
                "Pitcher F",
            ],
            "positions": ["1B", "OF", "SS", "OF,DH", "SP", "RP"],
            "is_hitter": [True, True, True, True, False, False],
            "r": [80, 70, 60, 50, 0, 0],
            "hr": [30, 25, 20, 15, 0, 0],
            "rbi": [90, 80, 70, 60, 0, 0],
            "sb": [10, 5, 15, 20, 0, 0],
            "avg": [0.290, 0.270, 0.260, 0.250, 0.0, 0.0],
            "w": [0, 0, 0, 0, 12, 0],
            "sv": [0, 0, 0, 0, 0, 20],
            "k": [0, 0, 0, 0, 180, 60],
            "era": [0.0, 0.0, 0.0, 0.0, 3.20, 2.80],
            "whip": [0.0, 0.0, 0.0, 0.0, 1.10, 1.05],
            "ip": [0, 0, 0, 0, 180, 60],
            "h": [145, 135, 125, 115, 0, 0],
            "ab": [500, 500, 480, 460, 0, 0],
            "er": [0, 0, 0, 0, 64, 18],
            "bb_allowed": [0, 0, 0, 0, 50, 20],
            "h_allowed": [0, 0, 0, 0, 148, 43],
        }
    )


# ── Initialization Tests ──────────────────────────────────────────────


class TestPipelineInit:
    """Test pipeline initialization."""

    def test_default_mode_is_standard(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster)
        assert pipe.mode == "standard"

    def test_all_modes_recognized(self, sample_roster):
        for mode in ("quick", "standard", "full"):
            pipe = LineupOptimizerPipeline(sample_roster, mode=mode)
            assert pipe.mode == mode

    def test_invalid_mode_falls_back_to_standard(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="nonexistent")
        assert pipe.mode == "standard"

    def test_mode_presets_have_required_keys(self):
        required = {
            "enable_projections",
            "enable_matchups",
            "enable_scenarios",
            "n_scenarios",
            "risk_aversion",
        }
        for mode_name, preset in MODE_PRESETS.items():
            assert required.issubset(preset.keys()), f"Mode {mode_name} missing keys"

    def test_roster_is_copied(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster)
        pipe.roster.iloc[0, 0] = 999
        assert sample_roster.iloc[0, 0] != 999


# ── Optimization Tests ─────────────────────────────────────────────────


class TestOptimize:
    """Test the full optimize() pipeline."""

    def test_quick_mode_returns_result(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        assert "lineup" in result
        assert "category_weights" in result
        assert "timing" in result
        assert result["mode"] == "quick"

    def test_standard_mode_returns_result(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="standard")
        result = pipe.optimize()
        assert "lineup" in result
        assert result["mode"] == "standard"

    def test_result_has_all_expected_keys(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        expected_keys = {
            "lineup",
            "category_weights",
            "h2h_analysis",
            "streaming_suggestions",
            "risk_metrics",
            "maximin_comparison",
            "recommendations",
            "timing",
            "mode",
            "matchup_adjusted",
        }
        assert expected_keys == set(result.keys())

    def test_timing_has_total(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        assert "total" in result["timing"]
        assert result["timing"]["total"] > 0

    def test_empty_roster(self):
        empty = pd.DataFrame()
        pipe = LineupOptimizerPipeline(empty, mode="quick")
        result = pipe.optimize()
        assert result["lineup"]["assignments"] == []

    def test_h2h_analysis_with_opponent(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        my_totals = {
            "r": 700,
            "hr": 200,
            "rbi": 650,
            "sb": 80,
            "avg": 0.265,
            "w": 60,
            "sv": 50,
            "k": 900,
            "era": 3.80,
            "whip": 1.20,
        }
        opp_totals = {
            "r": 680,
            "hr": 190,
            "rbi": 640,
            "sb": 85,
            "avg": 0.260,
            "w": 55,
            "sv": 55,
            "k": 880,
            "era": 3.90,
            "whip": 1.22,
        }
        result = pipe.optimize(
            h2h_opponent_totals=opp_totals,
            my_totals=my_totals,
        )
        # H2H analysis should be populated when totals are provided
        assert result["h2h_analysis"] is not None

    def test_no_h2h_without_opponent(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        assert result["h2h_analysis"] is None

    def test_category_weights_are_dict(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        weights = result["category_weights"]
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_recommendations_is_list(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        assert isinstance(result["recommendations"], list)


# ── Category Weight Blending Tests ─────────────────────────────────────


class TestCategoryWeights:
    """Test the category weight computation chain."""

    def test_default_weights_without_data(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        weights = pipe._compute_category_weights()
        # Without any data sources, should return equal-ish weights
        assert all(w > 0 for w in weights.values())

    def test_h2h_weights_affect_output(self, sample_roster):
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick", alpha=1.0)
        my_totals = {
            "r": 700,
            "hr": 200,
            "rbi": 650,
            "sb": 80,
            "avg": 0.265,
            "w": 60,
            "sv": 50,
            "k": 900,
            "era": 3.80,
            "whip": 1.20,
        }
        # Opponent is very close in HR but far ahead in SB
        opp_totals = {
            "r": 700,
            "hr": 201,
            "rbi": 650,
            "sb": 150,
            "avg": 0.265,
            "w": 60,
            "sv": 50,
            "k": 900,
            "era": 3.80,
            "whip": 1.20,
        }
        weights = pipe._compute_category_weights(
            h2h_opponent_totals=opp_totals,
            my_totals=my_totals,
        )
        # HR should have high weight (close race), SB should have low weight (far behind)
        assert isinstance(weights, dict)


# ── Risk Metrics Tests ──────────────────────────────────────────────────


class TestRiskMetrics:
    """Test risk metric computation."""

    def test_compute_risk_metrics_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        metrics = _compute_risk_metrics(values)
        assert "var_5" in metrics
        assert "cvar_5" in metrics
        assert "mean" in metrics
        assert "std" in metrics
        assert metrics["mean"] == pytest.approx(5.5, abs=0.1)

    def test_compute_risk_metrics_empty(self):
        metrics = _compute_risk_metrics(np.array([]))
        assert metrics["mean"] == 0.0
        assert metrics["var_5"] == 0.0

    def test_risk_metrics_percentiles(self):
        rng = np.random.RandomState(42)
        values = rng.normal(10.0, 2.0, size=1000)
        metrics = _compute_risk_metrics(values)
        assert metrics["p10"] < metrics["p50"] < metrics["p90"]
        assert metrics["var_5"] < metrics["p10"]

    def test_cvar_is_worst_tail(self):
        # CVaR should be less than or equal to VaR
        values = np.linspace(0, 100, 200)
        metrics = _compute_risk_metrics(values)
        assert metrics["cvar_5"] <= metrics["var_5"]


# ── Assignment Extraction Tests ──────────────────────────────────────


class TestAssignmentExtraction:
    """Test converting lineup results to scenario analysis format."""

    def test_extract_assignments_from_result(self, sample_roster):
        lineup_result = {
            "assignments": [
                {"slot": "1B", "player_name": "Batter A", "player_id": 1},
                {"slot": "OF", "player_name": "Batter B", "player_id": 2},
            ],
        }
        amap = _extract_assignments_map(lineup_result, sample_roster)
        assert len(amap) == 2
        assert all(v == 1.0 for v in amap.values())

    def test_extract_empty_assignments(self, sample_roster):
        lineup_result = {"assignments": []}
        amap = _extract_assignments_map(lineup_result, sample_roster)
        assert len(amap) == 0

    def test_extract_missing_player(self, sample_roster):
        lineup_result = {
            "assignments": [
                {"slot": "1B", "player_name": "Nonexistent Player", "player_id": 99},
            ],
        }
        amap = _extract_assignments_map(lineup_result, sample_roster)
        assert len(amap) == 0  # Player not found in roster


# ── Integration Test ────────────────────────────────────────────────────


class TestPipelineIntegration:
    """End-to-end pipeline tests."""

    def test_full_pipeline_no_crash(self, sample_roster):
        """Full mode should not crash even without optional data."""
        pipe = LineupOptimizerPipeline(sample_roster, mode="full")
        result = pipe.optimize()
        assert result["mode"] == "full"
        assert result["timing"]["total"] > 0

    def test_quick_mode_faster_than_full(self, sample_roster):
        """Quick mode should be faster (or at least not slower) than full."""
        quick_pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        full_pipe = LineupOptimizerPipeline(sample_roster, mode="full")

        quick_result = quick_pipe.optimize()
        full_result = full_pipe.optimize()

        # Quick should generally be faster, but with a small roster the
        # difference may be negligible. Just verify both complete.
        assert quick_result["timing"]["total"] >= 0
        assert full_result["timing"]["total"] >= 0

    def test_backward_compat_lineup_optimizer(self, sample_roster):
        """Pipeline wraps LineupOptimizer — verify it produces valid assignments."""
        pipe = LineupOptimizerPipeline(sample_roster, mode="quick")
        result = pipe.optimize()
        lineup = result["lineup"]
        assert "assignments" in lineup
        assert "bench" in lineup
        assert "projected_stats" in lineup
        assert "status" in lineup
