"""P5c: ROS projection should be scaled down for FAs with low YTD playing
time, so IL stash phantoms can't ride an inflated preseason projection."""

import pandas as pd
import pytest

from src.optimizer.fa_recommender import _scale_ros_by_playing_time
from src.optimizer.shared_data_layer import OptimizerDataContext


def _ctx_at_day(days_elapsed=70):
    ctx = OptimizerDataContext()
    ctx.weeks_remaining = max(0, (26 * 7 - days_elapsed) / 7)
    return ctx


def test_zero_gp_player_ros_heavily_discounted():
    """A 0-GP hitter at Day 70 should have ROS scaled to ~0.2x (floor)."""
    ctx = _ctx_at_day(70)
    fa = pd.Series(
        {
            "is_hitter": 1,
            "ytd_gp": 0,
            "r": 50,
            "hr": 16,
            "rbi": 40,
            "sb": 0,
            "ab": 300,
            "h": 75,
            "bb": 30,
            "hbp": 2,
            "sf": 2,
        }
    )
    scaled = _scale_ros_by_playing_time(fa, ctx)
    assert scaled["hr"] == pytest.approx(16 * 0.2, abs=0.5), (
        f"0-GP hitter should have HR scaled to ~3.2 (16x0.2). Got {scaled['hr']:.2f}"
    )
    assert scaled["r"] == pytest.approx(50 * 0.2, abs=1.5)


def test_healthy_player_ros_unchanged():
    """A healthy hitter (60% expected GP+) should have ROS unchanged."""
    ctx = _ctx_at_day(70)
    fa = pd.Series(
        {
            "is_hitter": 1,
            "ytd_gp": 50,  # 50/60 expected = 0.83 ratio
            "r": 50,
            "hr": 16,
            "rbi": 40,
            "sb": 0,
            "ab": 300,
            "h": 75,
            "bb": 30,
            "hbp": 2,
            "sf": 2,
        }
    )
    scaled = _scale_ros_by_playing_time(fa, ctx)
    assert scaled["hr"] == 16
    assert scaled["r"] == 50


def test_partial_gp_partial_discount():
    """30% expected -> ROS scaled to ~0.5x (midpoint of [0.2, 1.0] discount)."""
    ctx = _ctx_at_day(70)  # ~60 expected GP
    fa = pd.Series(
        {
            "is_hitter": 1,
            "ytd_gp": 18,  # 18/60 = 0.30 ratio
            "r": 50,
            "hr": 16,
            "rbi": 0,
            "sb": 0,
            "ab": 0,
            "h": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
        }
    )
    scaled = _scale_ros_by_playing_time(fa, ctx)
    # 0.30 ratio -> scale ~= 0.30 (linear in [0.2, 1.0] above the 0.6 threshold,
    # or 0.30 directly under the simpler max(0.2, ratio) formulation)
    assert 0.20 <= (scaled["hr"] / 16) <= 0.50
