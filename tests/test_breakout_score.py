"""Tests for Statcast breakout score computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leaders import compute_breakout_score, compute_breakout_scores_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hitter_pool(n: int = 50) -> pd.DataFrame:
    """Create a synthetic hitter pool with Statcast columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"Hitter_{i}" for i in range(1, n + 1)],
            "is_hitter": 1,
            "hr": rng.integers(0, 40, n),
            "rbi": rng.integers(0, 120, n),
            "sb": rng.integers(0, 30, n),
            "avg": rng.uniform(0.200, 0.330, n).round(3),
            "obp": rng.uniform(0.280, 0.420, n).round(3),
            "barrel_pct": rng.uniform(2.0, 20.0, n).round(1),
            "xwoba": rng.uniform(0.270, 0.420, n).round(3),
            "hard_hit_pct": rng.uniform(25.0, 55.0, n).round(1),
            "k_pct": rng.uniform(10.0, 35.0, n).round(1),
        }
    )


def _make_pitcher_pool(n: int = 50) -> pd.DataFrame:
    """Create a synthetic pitcher pool with Stuff+ columns."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "player_id": range(100, 100 + n),
            "name": [f"Pitcher_{i}" for i in range(1, n + 1)],
            "is_hitter": 0,
            "k": rng.integers(30, 250, n),
            "era": rng.uniform(2.50, 6.00, n).round(2),
            "whip": rng.uniform(0.90, 1.60, n).round(2),
            "w": rng.integers(0, 18, n),
            "sv": rng.integers(0, 30, n),
            "ip": rng.uniform(20.0, 200.0, n).round(1),
            "stuff_plus": rng.uniform(70, 140, n).round(0),
            "pitching_plus": rng.uniform(70, 140, n).round(0),
            "k_pct": rng.uniform(15.0, 40.0, n).round(1),
            "bb_pct": rng.uniform(3.0, 15.0, n).round(1),
            "swstr_pct": rng.uniform(6.0, 18.0, n).round(1),
            "siera": rng.uniform(2.30, 5.50, n).round(2),
        }
    )


# ---------------------------------------------------------------------------
# Tests — Hitter with elite Statcast
# ---------------------------------------------------------------------------


class TestHitterBreakoutScore:
    def test_elite_statcast_scores_above_70(self):
        pool = _make_hitter_pool(50)
        elite = {
            "player_id": 999,
            "name": "Elite Hitter",
            "is_hitter": 1,
            "hr": 35,
            "rbi": 100,
            "sb": 20,
            "avg": 0.310,
            "obp": 0.400,
            "barrel_pct": 22.0,  # well above pool max ~20
            "xwoba": 0.430,  # above pool max ~0.420
            "hard_hit_pct": 58.0,  # above pool max ~55
            "k_pct": 12.0,  # very low
        }
        score = compute_breakout_score(elite, pool=pool)
        assert score > 70, f"Elite hitter should score >70, got {score}"

    def test_average_statcast_scores_30_to_60(self):
        pool = _make_hitter_pool(50)
        # Median-ish values
        avg_player = {
            "player_id": 998,
            "name": "Average Hitter",
            "is_hitter": 1,
            "hr": 15,
            "rbi": 55,
            "sb": 10,
            "avg": 0.260,
            "obp": 0.340,
            "barrel_pct": 10.0,
            "xwoba": 0.340,
            "hard_hit_pct": 38.0,
            "k_pct": 22.0,
        }
        score = compute_breakout_score(avg_player, pool=pool)
        assert 20 <= score <= 70, f"Average hitter should score 20-70, got {score}"


# ---------------------------------------------------------------------------
# Tests — Pitcher with high Stuff+
# ---------------------------------------------------------------------------


class TestPitcherBreakoutScore:
    def test_elite_stuff_plus_scores_above_70(self):
        pool = _make_pitcher_pool(50)
        elite = {
            "player_id": 997,
            "name": "Elite Pitcher",
            "is_hitter": 0,
            "k": 220,
            "era": 5.20,  # bad ERA but great process
            "whip": 1.30,
            "w": 10,
            "sv": 0,
            "ip": 180.0,
            "stuff_plus": 150.0,  # way above pool max ~140
            "pitching_plus": 145.0,
            "k_pct": 38.0,
            "bb_pct": 5.0,
            "swstr_pct": 19.0,  # above pool max ~18
            "siera": 2.80,  # good SIERA vs bad ERA = breakout signal
        }
        score = compute_breakout_score(elite, pool=pool)
        assert score > 70, f"Elite pitcher should score >70, got {score}"


# ---------------------------------------------------------------------------
# Tests — Missing Statcast data (fallback)
# ---------------------------------------------------------------------------


class TestFallbackScore:
    def test_missing_statcast_produces_nonzero_score(self):
        """When Statcast columns are absent, fallback z-score method runs."""
        pool = pd.DataFrame(
            {
                "player_id": range(1, 31),
                "name": [f"P{i}" for i in range(1, 31)],
                "is_hitter": 1,
                "hr": list(range(0, 30)),
                "rbi": list(range(10, 40)),
                "sb": list(range(0, 30)),
                "avg": [0.200 + i * 0.004 for i in range(30)],
                "obp": [0.280 + i * 0.004 for i in range(30)],
            }
        )
        player = {
            "player_id": 99,
            "name": "No Statcast",
            "is_hitter": 1,
            "hr": 20,
            "rbi": 30,
            "sb": 15,
            "avg": 0.270,
            "obp": 0.340,
        }
        score = compute_breakout_score(player, pool=pool)
        assert score > 0, f"Fallback score should be >0, got {score}"
        assert score != 50.0, "Fallback should not just return the neutral 50"

    def test_pitcher_fallback_without_stuff_plus(self):
        pool = pd.DataFrame(
            {
                "player_id": range(100, 130),
                "name": [f"SP{i}" for i in range(30)],
                "is_hitter": 0,
                "k": list(range(50, 80)),
                "era": [3.0 + i * 0.1 for i in range(30)],
                "whip": [1.0 + i * 0.02 for i in range(30)],
                "w": list(range(5, 35)),
                "sv": [0] * 30,
            }
        )
        player = {
            "player_id": 999,
            "name": "Old Pitcher",
            "is_hitter": 0,
            "k": 70,
            "era": 3.50,
            "whip": 1.15,
            "w": 12,
            "sv": 0,
        }
        score = compute_breakout_score(player, pool=pool)
        assert score > 0, f"Pitcher fallback should be >0, got {score}"

    def test_no_pool_returns_50(self):
        player = {"player_id": 1, "name": "Test", "is_hitter": 1, "hr": 10}
        assert compute_breakout_score(player, pool=None) == 50.0
        assert compute_breakout_score(player, pool=pd.DataFrame()) == 50.0


# ---------------------------------------------------------------------------
# Tests — Batch function
# ---------------------------------------------------------------------------


class TestBatchBreakout:
    def test_adds_breakout_score_column(self):
        pool = _make_hitter_pool(30)
        result = compute_breakout_scores_batch(pool)
        assert "breakout_score" in result.columns
        assert len(result) == 30

    def test_does_not_modify_original(self):
        pool = _make_hitter_pool(20)
        original_cols = set(pool.columns)
        compute_breakout_scores_batch(pool)
        assert set(pool.columns) == original_cols, "Original pool should not be modified"

    def test_empty_pool_returns_empty_with_column(self):
        pool = pd.DataFrame()
        result = compute_breakout_scores_batch(pool)
        assert "breakout_score" in result.columns
        assert result.empty

    def test_mixed_pool_hitters_and_pitchers(self):
        hitters = _make_hitter_pool(20)
        pitchers = _make_pitcher_pool(20)
        pool = pd.concat([hitters, pitchers], ignore_index=True)
        result = compute_breakout_scores_batch(pool)
        assert "breakout_score" in result.columns
        assert len(result) == 40


# ---------------------------------------------------------------------------
# Tests — Clamping
# ---------------------------------------------------------------------------


class TestClamping:
    def test_score_clamped_to_0_100(self):
        pool = _make_hitter_pool(50)
        result = compute_breakout_scores_batch(pool)
        assert result["breakout_score"].min() >= 0.0
        assert result["breakout_score"].max() <= 100.0

    def test_pitcher_score_clamped(self):
        pool = _make_pitcher_pool(50)
        result = compute_breakout_scores_batch(pool)
        assert result["breakout_score"].min() >= 0.0
        assert result["breakout_score"].max() <= 100.0
