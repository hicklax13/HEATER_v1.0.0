"""Tests for playing time prediction model."""

from __future__ import annotations

import pandas as pd
import pytest

from src.playing_time_model import (
    BENCH_PLAYING_TIME_FACTOR,
    PLATOON_REDUCTION,
    _compute_blend_weights,
    predict_remaining_ip,
    predict_remaining_pa,
    predict_remaining_pa_batch,
)

# ---------------------------------------------------------------------------
# Blend weight tests
# ---------------------------------------------------------------------------


class TestBlendWeights:
    """Tests for _compute_blend_weights."""

    def test_zero_pa_fully_projection(self):
        proj_w, obs_w = _compute_blend_weights(0.0)
        assert proj_w == pytest.approx(1.0)
        assert obs_w == pytest.approx(0.0)

    def test_early_season_weights(self):
        proj_w, obs_w = _compute_blend_weights(30.0)
        # Should be between (1.0, 0.0) and (0.80, 0.20)
        assert 0.80 <= proj_w <= 1.0
        assert 0.0 <= obs_w <= 0.20

    def test_at_early_threshold(self):
        proj_w, obs_w = _compute_blend_weights(50.0)
        assert proj_w == pytest.approx(0.80)
        assert obs_w == pytest.approx(0.20)

    def test_mid_season_weights(self):
        proj_w, obs_w = _compute_blend_weights(125.0)
        # Midpoint between early and mid
        assert 0.50 <= proj_w <= 0.80
        assert 0.20 <= obs_w <= 0.50

    def test_at_mid_threshold(self):
        proj_w, obs_w = _compute_blend_weights(200.0)
        assert proj_w == pytest.approx(0.50)
        assert obs_w == pytest.approx(0.50)

    def test_late_season_weights(self):
        proj_w, obs_w = _compute_blend_weights(400.0)
        assert proj_w == pytest.approx(0.20)
        assert obs_w == pytest.approx(0.80)

    def test_weights_always_sum_to_one(self):
        for pa in [0, 10, 50, 100, 200, 300, 400, 500, 600]:
            proj_w, obs_w = _compute_blend_weights(float(pa))
            assert proj_w + obs_w == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# predict_remaining_pa tests
# ---------------------------------------------------------------------------


class TestPredictRemainingPA:
    """Tests for predict_remaining_pa."""

    def test_starter_high_recent_pa(self):
        """Starter with high observed PA rate gets high prediction."""
        result = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=4.5,
            games_remaining=100,
            health_score=1.0,
            depth_chart_starter=True,
            is_platoon=False,
            observed_pa=200.0,
        )
        # With 100 games left at ~4.5 PA/game, should be ~400+
        assert result > 350.0
        assert result < 600.0

    def test_bench_player_reduced_pa(self):
        """Bench player gets BENCH_PLAYING_TIME_FACTOR reduction."""
        starter = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=3.5,
            games_remaining=100,
            health_score=1.0,
            depth_chart_starter=True,
            observed_pa=100.0,
        )
        bench = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=3.5,
            games_remaining=100,
            health_score=1.0,
            depth_chart_starter=False,
            observed_pa=100.0,
        )
        assert bench == pytest.approx(starter * BENCH_PLAYING_TIME_FACTOR, rel=0.01)

    def test_injured_player_half_pa(self):
        """Player with health_score=0.5 gets roughly half the PA."""
        healthy = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=4.0,
            games_remaining=80,
            health_score=1.0,
            observed_pa=150.0,
        )
        injured = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=4.0,
            games_remaining=80,
            health_score=0.5,
            observed_pa=150.0,
        )
        assert injured == pytest.approx(healthy * 0.5, rel=0.01)

    def test_platoon_player_reduced(self):
        """Platoon player gets 65% of full-time PA."""
        full_time = predict_remaining_pa(
            projected_pa=550.0,
            recent_pa_per_game=4.0,
            games_remaining=90,
            health_score=1.0,
            is_platoon=False,
            observed_pa=100.0,
        )
        platoon = predict_remaining_pa(
            projected_pa=550.0,
            recent_pa_per_game=4.0,
            games_remaining=90,
            health_score=1.0,
            is_platoon=True,
            observed_pa=100.0,
        )
        expected_factor = 1.0 - PLATOON_REDUCTION
        assert platoon == pytest.approx(full_time * expected_factor, rel=0.01)

    def test_early_season_blends_toward_projection(self):
        """Early season (< 50 PA): projection dominates.

        Player projected for 600 PA but only getting 2.0 PA/game observed.
        Early season should still show ~projection-level PA.
        """
        result = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=2.0,
            games_remaining=140,
            health_score=1.0,
            observed_pa=20.0,  # very early
        )
        # Projection-only would give ~600 * 140/162 = ~519
        # Observed-only would give ~2.0 * 140 = 280
        # Early blend should be much closer to projection
        assert result > 400.0

    def test_late_season_blends_toward_observed(self):
        """Late season (> 200 PA): observed rate dominates.

        Player projected for 600 PA but only getting 2.5 PA/game.
        Late season should revise DOWN significantly.
        """
        result = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=2.5,
            games_remaining=50,
            health_score=1.0,
            observed_pa=400.0,  # late season
        )
        # Observed-only: 2.5 * 50 = 125
        # Should be close to observed since late season
        assert result < 200.0

    def test_zero_games_remaining(self):
        """Zero games remaining returns 0."""
        result = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=4.0,
            games_remaining=0,
        )
        assert result == 0.0

    def test_zero_recent_pa_rate(self):
        """Zero recent PA rate with no observed data uses projection."""
        result = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=0.0,
            games_remaining=100,
            health_score=0.85,
            observed_pa=0.0,
        )
        # Should use projection only, prorated
        assert result > 0.0

    def test_negative_games_remaining(self):
        """Negative games remaining returns 0."""
        result = predict_remaining_pa(
            projected_pa=600.0,
            recent_pa_per_game=4.0,
            games_remaining=-5,
        )
        assert result == 0.0

    def test_health_clamped_to_zero_one(self):
        """Health score > 1 or < 0 gets clamped."""
        result_over = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=4.0,
            games_remaining=80,
            health_score=1.5,
            observed_pa=100.0,
        )
        result_normal = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=4.0,
            games_remaining=80,
            health_score=1.0,
            observed_pa=100.0,
        )
        assert result_over == pytest.approx(result_normal, rel=0.01)

        result_neg = predict_remaining_pa(
            projected_pa=500.0,
            recent_pa_per_game=4.0,
            games_remaining=80,
            health_score=-0.5,
        )
        assert result_neg == 0.0

    def test_result_always_non_negative(self):
        """Result is always >= 0."""
        result = predict_remaining_pa(
            projected_pa=0.0,
            recent_pa_per_game=0.0,
            games_remaining=100,
            health_score=0.0,
        )
        assert result >= 0.0


# ---------------------------------------------------------------------------
# predict_remaining_ip tests
# ---------------------------------------------------------------------------


class TestPredictRemainingIP:
    """Tests for predict_remaining_ip."""

    def test_starter_basic(self):
        """Starter with projected 180 IP and 6 IP/start."""
        result = predict_remaining_ip(
            projected_ip=180.0,
            recent_ip_per_start=6.0,
            starts_remaining=20,
            health_score=1.0,
            is_starter=True,
            observed_ip=50.0,
        )
        # ~5.6 IP/start blended, * 20 starts ~ 100-120
        assert result > 80.0
        assert result < 150.0

    def test_reliever_basic(self):
        """Reliever with projected 65 IP."""
        result = predict_remaining_ip(
            projected_ip=65.0,
            recent_ip_per_start=1.0,  # IP per appearance
            starts_remaining=40,  # remaining appearances
            health_score=1.0,
            is_starter=False,
            observed_ip=20.0,
        )
        assert result > 20.0
        assert result < 60.0

    def test_injured_pitcher(self):
        """Injured pitcher gets reduced IP."""
        healthy = predict_remaining_ip(
            projected_ip=180.0,
            recent_ip_per_start=6.0,
            starts_remaining=20,
            health_score=1.0,
            is_starter=True,
            observed_ip=60.0,
        )
        injured = predict_remaining_ip(
            projected_ip=180.0,
            recent_ip_per_start=6.0,
            starts_remaining=20,
            health_score=0.5,
            is_starter=True,
            observed_ip=60.0,
        )
        assert injured == pytest.approx(healthy * 0.5, rel=0.01)

    def test_zero_starts_remaining(self):
        """Zero starts remaining returns 0."""
        result = predict_remaining_ip(
            projected_ip=180.0,
            recent_ip_per_start=6.0,
            starts_remaining=0,
        )
        assert result == 0.0

    def test_no_observed_ip_uses_projection(self):
        """With no observed data, uses projected rate."""
        result = predict_remaining_ip(
            projected_ip=180.0,
            recent_ip_per_start=0.0,
            starts_remaining=20,
            health_score=0.85,
            is_starter=True,
            observed_ip=0.0,
        )
        # Should use projected IP/start (~5.625 per start)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Batch function tests
# ---------------------------------------------------------------------------


class TestPredictBatch:
    """Tests for predict_remaining_pa_batch."""

    def _make_pool(self) -> pd.DataFrame:
        """Create a small test player pool."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "name": ["Hitter A", "Bench B", "Starter C", "Reliever D"],
                "is_hitter": [True, True, False, False],
                "positions": ["SS", "DH", "SP", "RP"],
                "pa": [600, 200, 0, 0],
                "ip": [0, 0, 180, 65],
                "health_score": [0.90, 0.85, 0.80, 1.0],
                "ytd_pa": [100, 30, 0, 0],
                "ytd_ip": [0, 0, 40, 15],
                "games_played": [25, 20, 0, 0],
            }
        )

    def test_adds_columns(self):
        """Batch adds predicted_remaining_pa and predicted_remaining_ip."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        assert "predicted_remaining_pa" in result.columns
        assert "predicted_remaining_ip" in result.columns

    def test_returns_copy(self):
        """Batch returns a copy, does not modify original."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        assert "predicted_remaining_pa" not in pool.columns
        assert result is not pool

    def test_hitter_gets_pa_not_ip(self):
        """Hitters get PA predictions, not IP."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        hitter_row = result[result["name"] == "Hitter A"].iloc[0]
        assert hitter_row["predicted_remaining_pa"] > 0
        assert hitter_row["predicted_remaining_ip"] == 0.0

    def test_pitcher_gets_ip_not_pa(self):
        """Pitchers get IP predictions, not PA."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        pitcher_row = result[result["name"] == "Starter C"].iloc[0]
        assert pitcher_row["predicted_remaining_ip"] > 0
        assert pitcher_row["predicted_remaining_pa"] == 0.0

    def test_bench_hitter_gets_less_pa(self):
        """Bench hitter (low projected PA) gets reduced PA."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        starter_pa = result[result["name"] == "Hitter A"].iloc[0]["predicted_remaining_pa"]
        bench_pa = result[result["name"] == "Bench B"].iloc[0]["predicted_remaining_pa"]
        # Bench player should get significantly less
        assert bench_pa < starter_pa

    def test_zero_weeks_remaining(self):
        """Zero weeks remaining gives all zeros."""
        pool = self._make_pool()
        result = predict_remaining_pa_batch(pool, weeks_remaining=0)
        assert (result["predicted_remaining_pa"] == 0.0).all()
        assert (result["predicted_remaining_ip"] == 0.0).all()

    def test_missing_health_score_defaults(self):
        """Missing health_score defaults to 0.85."""
        pool = self._make_pool()
        pool = pool.drop(columns=["health_score"])
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        # Should not raise and should produce valid results
        assert (result["predicted_remaining_pa"] >= 0).all()
        assert (result["predicted_remaining_ip"] >= 0).all()

    def test_missing_ytd_columns_defaults(self):
        """Missing ytd_pa/ytd_ip/games_played defaults to 0."""
        pool = self._make_pool()
        pool = pool.drop(columns=["ytd_pa", "ytd_ip", "games_played"])
        result = predict_remaining_pa_batch(pool, weeks_remaining=16)
        # Should not raise
        assert len(result) == 4
        assert (result["predicted_remaining_pa"] >= 0).all()
