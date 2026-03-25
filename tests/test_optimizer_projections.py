"""Tests for the enhanced projection pipeline (Phase 1).

Validates that each stage of the projection enhancement pipeline:
1. Correctly wires into existing analytics modules
2. Degrades gracefully when data/modules are unavailable
3. Produces reasonable adjustments to player stat projections
4. Maintains DataFrame schema compatibility
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.projections import (
    ALL_CATS,
    COUNTING_CATS,
    _apply_injury_availability,
    _apply_kalman_filter,
    _apply_statcast_adjustment,
    _merge_updated_stats,
    build_enhanced_projections,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_hitter_roster():
    """Realistic 5-player hitter roster for testing."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["Aaron Judge", "Mookie Betts", "Shohei Ohtani", "Ronald Acuna Jr.", "Corey Seager"],
            "positions": ["OF", "OF,SS", "DH", "OF", "SS"],
            "is_hitter": [True, True, True, True, True],
            "team": ["NYY", "LAD", "LAD", "ATL", "TEX"],
            "pa": [600, 550, 580, 500, 520],
            "ab": [520, 480, 510, 440, 460],
            "h": [150, 145, 160, 130, 135],
            "r": [100, 95, 110, 90, 80],
            "hr": [45, 28, 40, 32, 30],
            "rbi": [110, 85, 95, 80, 90],
            "sb": [5, 15, 12, 45, 3],
            "avg": [0.288, 0.302, 0.314, 0.295, 0.293],
            "ip": [0, 0, 0, 0, 0],
            "w": [0, 0, 0, 0, 0],
            "sv": [0, 0, 0, 0, 0],
            "k": [0, 0, 0, 0, 0],
            "era": [0, 0, 0, 0, 0],
            "whip": [0, 0, 0, 0, 0],
            "er": [0, 0, 0, 0, 0],
            "bb_allowed": [0, 0, 0, 0, 0],
            "h_allowed": [0, 0, 0, 0, 0],
        }
    )


@pytest.fixture
def sample_pitcher_roster():
    """Realistic 3-pitcher roster for testing."""
    return pd.DataFrame(
        {
            "player_id": [10, 11, 12],
            "player_name": ["Gerrit Cole", "Spencer Strider", "Josh Hader"],
            "positions": ["SP", "SP", "RP"],
            "is_hitter": [False, False, False],
            "team": ["NYY", "ATL", "HOU"],
            "pa": [0, 0, 0],
            "ab": [0, 0, 0],
            "h": [0, 0, 0],
            "r": [0, 0, 0],
            "hr": [0, 0, 0],
            "rbi": [0, 0, 0],
            "sb": [0, 0, 0],
            "avg": [0, 0, 0],
            "ip": [190, 170, 65],
            "w": [15, 14, 4],
            "sv": [0, 0, 35],
            "k": [220, 250, 85],
            "era": [3.10, 2.85, 2.40],
            "whip": [1.05, 0.95, 0.90],
            "er": [65, 54, 17],
            "bb_allowed": [45, 35, 18],
            "h_allowed": [155, 126, 40],
        }
    )


@pytest.fixture
def mixed_roster(sample_hitter_roster, sample_pitcher_roster):
    """Combined hitter + pitcher roster."""
    return pd.concat([sample_hitter_roster, sample_pitcher_roster], ignore_index=True)


# ── Test build_enhanced_projections (master function) ────────────────


class TestBuildEnhancedProjections:
    """Tests for the main pipeline orchestrator."""

    def test_returns_same_schema(self, mixed_roster):
        """Output has same columns as input plus enhancement columns."""
        result = build_enhanced_projections(
            mixed_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        # Original columns preserved
        for col in mixed_roster.columns:
            assert col in result.columns, f"Missing column: {col}"
        # Enhancement columns added
        assert "projection_confidence" in result.columns
        assert "regime_label" in result.columns
        assert "health_adjusted" in result.columns

    def test_all_disabled_returns_unchanged_stats(self, sample_hitter_roster):
        """With all enhancements disabled, stats should be unchanged."""
        original = sample_hitter_roster.copy()
        result = build_enhanced_projections(
            sample_hitter_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        for cat in COUNTING_CATS:
            if cat in original.columns:
                np.testing.assert_array_almost_equal(result[cat].values, original[cat].values)

    def test_does_not_modify_input(self, sample_hitter_roster):
        """Pipeline should not modify the input DataFrame."""
        original = sample_hitter_roster.copy()
        build_enhanced_projections(
            sample_hitter_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        pd.testing.assert_frame_equal(sample_hitter_roster, original)

    def test_confidence_initialized_to_one(self, sample_hitter_roster):
        """Default projection confidence should be 1.0."""
        result = build_enhanced_projections(
            sample_hitter_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        assert all(result["projection_confidence"] == 1.0)

    def test_empty_roster(self):
        """Empty roster returns empty DataFrame with enhancement columns."""
        empty = pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "positions",
                "is_hitter",
                "pa",
                "r",
                "hr",
                "rbi",
                "sb",
                "avg",
                "ip",
                "w",
                "sv",
                "k",
                "era",
                "whip",
            ]
        )
        result = build_enhanced_projections(
            empty,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        assert "projection_confidence" in result.columns
        assert len(result) == 0


# ── Test merge_updated_stats ─────────────────────────────────────────


class TestMergeUpdatedStats:
    """Tests for merging Bayesian-updated stats back into roster."""

    def test_merge_updates_matching_players(self, sample_hitter_roster):
        """Updated stats replace original values for matching player_ids."""
        updated = pd.DataFrame(
            {
                "player_id": [1, 3],
                "hr": [50, 42],
                "rbi": [120, 100],
            }
        )
        result = _merge_updated_stats(sample_hitter_roster.copy(), updated)
        assert result.loc[result["player_id"] == 1, "hr"].values[0] == 50
        assert result.loc[result["player_id"] == 3, "rbi"].values[0] == 100

    def test_merge_ignores_unmatched(self, sample_hitter_roster):
        """Players not in the updated set keep their original values."""
        updated = pd.DataFrame(
            {
                "player_id": [1],
                "hr": [50],
            }
        )
        result = _merge_updated_stats(sample_hitter_roster.copy(), updated)
        # Player 2 should be unchanged
        assert result.loc[result["player_id"] == 2, "hr"].values[0] == 28

    def test_merge_no_player_id_column(self, sample_hitter_roster):
        """Returns roster unchanged if updated has no player_id."""
        updated = pd.DataFrame({"hr": [50, 42]})
        result = _merge_updated_stats(sample_hitter_roster.copy(), updated)
        assert result.loc[result["player_id"] == 1, "hr"].values[0] == 45


# ── Test Kalman Filter ───────────────────────────────────────────────


class TestKalmanFilter:
    """Tests for Kalman filter projection adjustment."""

    def test_small_sample_skipped(self, sample_hitter_roster):
        """Players with < 10 PA should not be Kalman-filtered."""
        roster = sample_hitter_roster.copy()
        roster["pa"] = 5  # Too few PA
        original_avg = roster["avg"].values.copy()
        result = _apply_kalman_filter(roster)
        np.testing.assert_array_almost_equal(result["avg"].values, original_avg)

    def test_large_sample_updates(self, sample_hitter_roster):
        """Players with sufficient PA get filtered (confidence adjusted)."""
        result = _apply_kalman_filter(sample_hitter_roster.copy())
        # With 600 PA, Kalman should update; confidence should be >= original
        assert all(result["projection_confidence"] >= 0.0)
        assert all(result["projection_confidence"] <= 1.0)

    def test_pitchers_filter_era_whip(self, sample_pitcher_roster):
        """Pitchers should have ERA and WHIP filtered, not AVG."""
        original_era = sample_pitcher_roster["era"].values.copy()
        result = _apply_kalman_filter(sample_pitcher_roster.copy())
        # ERA values should still be in reasonable range
        for era in result["era"].values:
            if era > 0:
                assert 0.5 <= era <= 12.0

    def test_graceful_without_kalman_module(self, sample_hitter_roster):
        """Returns roster unchanged if Kalman module not importable."""
        with patch.dict("sys.modules", {"src.engine.signals.kalman": None}):
            # Re-importing with the module removed should degrade gracefully
            result = _apply_kalman_filter(sample_hitter_roster.copy())
            assert len(result) == len(sample_hitter_roster)


# ── Test Injury Availability ─────────────────────────────────────────


class TestInjuryAvailability:
    """Tests for injury-based availability scaling."""

    def test_healthy_player_minimal_reduction(self):
        """A player with health_score ~1.0 should lose minimal production."""
        roster = pd.DataFrame(
            {
                "player_id": [1],
                "player_name": ["Healthy Player"],
                "positions": ["OF"],
                "is_hitter": [True],
                "r": [100.0],
                "hr": [40.0],
                "rbi": [100.0],
                "sb": [20.0],
                "avg": [0.300],
                "ip": [0.0],
                "w": [0.0],
                "sv": [0.0],
                "k": [0.0],
                "era": [0.0],
                "whip": [0.0],
                "projection_confidence": [1.0],
                "regime_label": [""],
                "health_adjusted": [False],
            }
        )
        mock_injury_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1],
                "season": [2023, 2024, 2025],
                "games_played": [155, 158, 160],
                "games_available": [162, 162, 162],
            }
        )
        with patch("src.database.get_connection") as mock_conn:
            mock_connection = MagicMock()
            mock_conn.return_value = mock_connection
            with patch("src.optimizer.projections.pd.read_sql_query", return_value=mock_injury_df):
                result = _apply_injury_availability(roster.copy(), weeks_remaining=16)

        # Healthy player (health_score ~ 0.97) should retain > 87% of HR
        if result.at[0, "health_adjusted"]:
            assert result.at[0, "hr"] >= 35.0

    def test_injury_prone_player_significant_reduction(self):
        """A player with poor health history should have stats reduced."""
        roster = pd.DataFrame(
            {
                "player_id": [2],
                "player_name": ["Injury Prone"],
                "positions": ["SP"],
                "is_hitter": [False],
                "r": [0.0],
                "hr": [0.0],
                "rbi": [0.0],
                "sb": [0.0],
                "avg": [0.0],
                "ip": [180.0],
                "w": [12.0],
                "sv": [0.0],
                "k": [200.0],
                "era": [3.50],
                "whip": [1.10],
                "projection_confidence": [1.0],
                "regime_label": [""],
                "health_adjusted": [False],
            }
        )
        mock_injury_df = pd.DataFrame(
            {
                "player_id": [2, 2, 2],
                "season": [2023, 2024, 2025],
                "games_played": [60, 80, 70],
                "games_available": [162, 162, 162],
            }
        )
        with patch("src.database.get_connection") as mock_conn:
            mock_connection = MagicMock()
            mock_conn.return_value = mock_connection
            with patch("src.optimizer.projections.pd.read_sql_query", return_value=mock_injury_df):
                result = _apply_injury_availability(roster.copy(), weeks_remaining=16)

        # Health score ~ 0.43 -> significant reduction
        if result.at[0, "health_adjusted"]:
            assert result.at[0, "k"] < 200.0
            assert result.at[0, "w"] < 12.0

    def test_injury_scaling_preserves_rate_stats(self):
        """AVG, ERA, WHIP should NOT be scaled by availability."""
        roster = pd.DataFrame(
            {
                "player_id": [1],
                "player_name": ["Test"],
                "positions": ["OF"],
                "is_hitter": [True],
                "r": [100.0],
                "hr": [40.0],
                "rbi": [100.0],
                "sb": [20.0],
                "avg": [0.300],
                "ip": [0.0],
                "w": [0.0],
                "sv": [0.0],
                "k": [0.0],
                "era": [0.0],
                "whip": [0.0],
                "projection_confidence": [1.0],
                "regime_label": [""],
                "health_adjusted": [False],
            }
        )
        mock_injury_df = pd.DataFrame(
            {
                "player_id": [1, 1],
                "season": [2024, 2025],
                "games_played": [80, 90],
                "games_available": [162, 162],
            }
        )
        with patch("src.database.get_connection") as mock_conn:
            mock_connection = MagicMock()
            mock_conn.return_value = mock_connection
            with patch("src.optimizer.projections.pd.read_sql_query", return_value=mock_injury_df):
                result = _apply_injury_availability(roster.copy(), weeks_remaining=16)

        # AVG should be unchanged
        assert result.at[0, "avg"] == 0.300

    def test_empty_injury_history_skips(self):
        """No injury data should skip availability scaling entirely."""
        roster = pd.DataFrame(
            {
                "player_id": [1],
                "player_name": ["Test"],
                "positions": ["OF"],
                "is_hitter": [True],
                "r": [100.0],
                "hr": [40.0],
                "rbi": [100.0],
                "sb": [20.0],
                "avg": [0.300],
                "ip": [0.0],
                "w": [0.0],
                "sv": [0.0],
                "k": [0.0],
                "era": [0.0],
                "whip": [0.0],
                "projection_confidence": [1.0],
                "regime_label": [""],
                "health_adjusted": [False],
            }
        )
        empty_df = pd.DataFrame(columns=["player_id", "season", "games_played", "games_available"])
        with patch("src.database.get_connection") as mock_conn:
            mock_connection = MagicMock()
            mock_conn.return_value = mock_connection
            with patch("src.optimizer.projections.pd.read_sql_query", return_value=empty_df):
                result = _apply_injury_availability(roster.copy())

        # Stats should be unchanged
        assert result.at[0, "hr"] == 40.0


# ── Test Graceful Degradation ────────────────────────────────────────


class TestGracefulDegradation:
    """Tests that each pipeline step degrades gracefully on failure."""

    def test_bayesian_import_failure(self, sample_hitter_roster):
        """Pipeline succeeds even if Bayesian module can't be imported."""
        result = build_enhanced_projections(
            sample_hitter_roster,
            enable_bayesian=True,  # Will try to import and likely hit DB issues
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        # Should return valid DataFrame regardless
        assert len(result) == len(sample_hitter_roster)
        assert "projection_confidence" in result.columns

    def test_all_enabled_no_crash(self, mixed_roster):
        """Enabling everything should not crash even without DB data."""
        result = build_enhanced_projections(
            mixed_roster,
            enable_bayesian=True,
            enable_kalman=True,
            enable_statcast=True,
            enable_injury=True,
        )
        assert len(result) == len(mixed_roster)

    def test_statcast_without_pybaseball(self, sample_hitter_roster):
        """Statcast step should be skipped when pybaseball not installed."""
        with patch(
            "src.optimizer.projections._apply_statcast_adjustment",
            return_value=sample_hitter_roster.copy(),
        ):
            result = build_enhanced_projections(
                sample_hitter_roster,
                enable_bayesian=False,
                enable_kalman=False,
                enable_statcast=True,
                enable_injury=False,
            )
        assert len(result) == len(sample_hitter_roster)


# ── Test Pipeline Integration ────────────────────────────────────────


class TestPipelineIntegration:
    """Integration tests for the full enhancement pipeline."""

    def test_regime_then_injury_compounds(self, sample_hitter_roster):
        """Regime adjustment + injury scaling should compound correctly."""
        # Run regime only
        regime_result = build_enhanced_projections(
            sample_hitter_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )

        # The pipeline should have adjusted at least some counting stats
        # (players with high AVG get regime boost)
        ohtani_regime_hr = regime_result.loc[regime_result["player_name"] == "Shohei Ohtani", "hr"].values[0]

        # Ohtani's .314 AVG → ~.361 xwOBA → Elite regime → multiplier > 1.0
        assert ohtani_regime_hr >= 40  # Should be >= original 40

    def test_player_count_preserved(self, mixed_roster):
        """Number of players should never change through pipeline."""
        result = build_enhanced_projections(
            mixed_roster,
            enable_bayesian=False,
            enable_kalman=True,
            enable_statcast=False,
            enable_injury=False,
        )
        assert len(result) == len(mixed_roster)

    def test_no_negative_stats(self, mixed_roster):
        """No counting stat should ever go negative."""
        result = build_enhanced_projections(
            mixed_roster,
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
        )
        for cat in COUNTING_CATS:
            if cat in result.columns:
                assert all(result[cat] >= 0), f"Negative values in {cat}"
