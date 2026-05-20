"""FA-engine P2 PR4 (2026-05-20): _blend_fa_row blends ROS projection
with YTD pace using canonical published weights.

Background:
  The engine previously evaluated FAs using only the ROS projection
  columns (r/hr/rbi/etc.), ignoring the YTD performance loaded into
  ytd_r/ytd_hr/etc. Result: a player on a hot 2-month streak was
  scored at his preseason projection — ignoring real in-season
  evidence. Industry methodology (Smart Fantasy Baseball, The
  Athletic, FanGraphs) is to blend ROS + YTD + L14, with ROS as the
  dominant anchor.

  PR4 implements ROS + YTD blending. L14 is deferred to a follow-up.

These tests pin:
  * Insufficient YTD games → pure ROS (no blend)
  * With enough YTD, columns are blended at the renormalized 0.778
    ROS / 0.222 YTD ratio (since L14 weight 0.10 is not yet active)
  * Returns a new Series — original fa_data not mutated
  * Sample-size threshold _BLEND_YTD_MIN_GAMES is configurable
  * Counting stats blend; rate stats (avg/obp/era/whip) are NOT
    directly modified — they regenerate from numerator/denominator
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer.fa_recommender import (
    _BLEND_WEIGHT_ROS,
    _BLEND_WEIGHT_YTD,
    _BLEND_YTD_MIN_GAMES,
    _BLENDABLE_COUNTING_COLS,
    _blend_fa_row,
)


def _row(ytd_gp: float = 80, **overrides) -> pd.Series:
    """Build a player row dict with ROS projection + YTD totals."""
    base = {
        "player_id": 100,
        "name": "Test Player",
        "is_hitter": 1,
        "status": "active",
        # ROS projection columns (what the engine reads today)
        "r": 50,
        "hr": 15,
        "rbi": 60,
        "sb": 8,
        "ab": 250,
        "h": 70,
        "bb": 30,
        "hbp": 3,
        "sf": 3,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0.0,
        "er": 0.0,
        "bb_allowed": 0,
        "h_allowed": 0,
        # YTD totals
        "ytd_gp": ytd_gp,
        "ytd_r": 80,
        "ytd_hr": 24,
        "ytd_rbi": 96,
        "ytd_sb": 12,
        "ytd_ab": 400,
        "ytd_h": 110,
        "ytd_bb": 40,
        "ytd_hbp": 5,
        "ytd_sf": 4,
        "ytd_w": 0,
        "ytd_l": 0,
        "ytd_sv": 0,
        "ytd_k": 0,
        "ytd_ip": 0.0,
        "ytd_er": 0.0,
        "ytd_bb_allowed": 0,
        "ytd_h_allowed": 0,
    }
    base.update(overrides)
    return pd.Series(base)


# ── Sample-size gating ───────────────────────────────────────────────


def test_insufficient_ytd_games_returns_pure_ros():
    """With ytd_gp below threshold, no blending — fa_data returned as-is."""
    row = _row(ytd_gp=_BLEND_YTD_MIN_GAMES - 1)
    blended = _blend_fa_row(row)
    # The HR column should equal ROS projection (15), NOT a blended value.
    assert blended["hr"] == 15
    assert blended["r"] == 50


def test_zero_ytd_games_returns_pure_ros():
    """Edge case: ytd_gp=0 → pure ROS (avoids divide-by-zero)."""
    row = _row(ytd_gp=0)
    blended = _blend_fa_row(row)
    assert blended["hr"] == 15
    assert blended["r"] == 50


def test_threshold_minus_one_returns_pure_ros():
    """Just below the gate → no blend."""
    row = _row(ytd_gp=_BLEND_YTD_MIN_GAMES - 1, hr=20, ytd_hr=40)
    blended = _blend_fa_row(row)
    assert blended["hr"] == 20  # original ROS, not blended


# ── Blending math ────────────────────────────────────────────────────


def test_blending_combines_ros_and_ytd_per_game_rates():
    """With enough YTD games, counting stats are blended at the
    renormalized 0.778 ROS / 0.222 YTD ratio (because L14 not yet
    wired, so 0.70/0.20 normalize to 0.778/0.222)."""
    # 80 games YTD: 24 HR / 80 games = 0.30 HR/game
    # ROS projection: 15 HR over (162-80)=82 games = 15/82 ≈ 0.183 HR/game
    # Renormalized blend ratio: 0.70 / (0.70 + 0.20) = 0.778 ROS, 0.222 YTD
    # Blended per-game: 0.778 × 0.183 + 0.222 × 0.30 ≈ 0.142 + 0.0666 = 0.208
    # Projected back to ROS total: 0.208 × 82 ≈ 17.1 HR
    row = _row(ytd_gp=80, hr=15, ytd_hr=24)
    blended = _blend_fa_row(row)

    games_remaining = 162 - 80
    ros_per_game = 15 / games_remaining
    ytd_per_game = 24 / 80
    total = _BLEND_WEIGHT_ROS + _BLEND_WEIGHT_YTD
    ros_w = _BLEND_WEIGHT_ROS / total
    ytd_w = _BLEND_WEIGHT_YTD / total
    expected_per_game = ros_w * ros_per_game + ytd_w * ytd_per_game
    expected_hr = expected_per_game * games_remaining

    assert blended["hr"] == pytest.approx(expected_hr, abs=0.1)


def test_hot_streak_player_gets_uplift():
    """A player on a hot pace (YTD outperforming ROS) should see
    their blended projection lifted above pure ROS."""
    # Hot player: 30 HR in 80 games = 0.375/g pace, projecting 60 HR full season
    # ROS projection: 15 HR in 82 games left — that's a CONSERVATIVE projection
    # Blended should land between the two: above 15, below 30
    row = _row(ytd_gp=80, hr=15, ytd_hr=30)
    blended = _blend_fa_row(row)
    assert 15 < blended["hr"] < 30, "Hot-streak player should have blended HR between ROS and YTD-extrapolated"


def test_cold_streak_player_gets_pulldown():
    """A cold-streak player should see their blended projection
    pulled DOWN below pure ROS."""
    row = _row(ytd_gp=80, hr=15, ytd_hr=4)  # 4 HR in 80g — very cold
    blended = _blend_fa_row(row)
    assert blended["hr"] < 15, "Cold-streak player's HR should be lower than pure ROS projection"


def test_pitcher_stats_blend_correctly():
    """Pitcher counting stats (k, w, ip, etc.) also blend."""
    row = _row(
        ytd_gp=20,  # pitcher GP threshold is lower in reality but tests share threshold
        is_hitter=0,
        k=80,
        ytd_k=120,
        ip=100.0,
        ytd_ip=130.0,
    )
    # ytd_gp=20 is below the threshold (default 30 in this PR) — so no blend
    blended = _blend_fa_row(row)
    assert blended["k"] == 80
    assert blended["ip"] == 100.0


def test_pitcher_stats_blend_when_above_threshold():
    """Pitcher with enough YTD games gets blended."""
    row = _row(
        ytd_gp=40,
        is_hitter=0,
        k=80,
        ytd_k=120,
        ip=100.0,
        ytd_ip=130.0,
    )
    blended = _blend_fa_row(row)
    # K should be blended (likely up since YTD pace 120/40=3.0/g > ROS pace 80/122≈0.66/g)
    assert blended["k"] != 80, "Pitcher K should be blended"


# ── Immutability + edge cases ────────────────────────────────────────


def test_returns_new_series_not_mutating_input():
    """Original fa_data must NOT be mutated by the blend."""
    row = _row(ytd_gp=80, hr=15, ytd_hr=30)
    original_hr = row["hr"]
    _ = _blend_fa_row(row)
    assert row["hr"] == original_hr  # source untouched


def test_missing_ytd_column_skipped_gracefully():
    """If a specific ytd_* column is missing, that stat just isn't blended.
    Other stats still blend normally."""
    base = _row(ytd_gp=80)
    # Drop ytd_hr to simulate missing column
    base_dict = base.to_dict()
    del base_dict["ytd_hr"]
    row = pd.Series(base_dict)
    blended = _blend_fa_row(row)
    # HR should fall back to pure ROS since ytd_hr column is missing
    assert blended["hr"] == 15
    # But other blendable cols should still blend
    assert blended["rbi"] != 60, "Other counting stats should still blend"


def test_rate_stats_not_directly_blended():
    """Rate stats (avg/obp/era/whip) are NOT in _BLENDABLE_COUNTING_COLS.
    They derive from the blended numerator + denominator counting stats
    downstream — direct blending would risk inconsistency with the
    underlying h/ab counts."""
    rate_stats = {"avg", "obp", "era", "whip"}
    blendable = set(_BLENDABLE_COUNTING_COLS)
    assert not (rate_stats & blendable), (
        "Rate stats (avg/obp/era/whip) must NOT be in _BLENDABLE_COUNTING_COLS — "
        "they regenerate from the blended numerator/denominator columns "
        "(h/ab for AVG, er/ip for ERA, etc.)."
    )


def test_blend_weights_sum_to_one_when_l14_added():
    """Canonical published weights: 0.70 ROS + 0.20 YTD + 0.10 L14 = 1.0.
    Even though L14 isn't wired yet, the constants must respect this."""
    from src.optimizer.fa_recommender import _BLEND_WEIGHT_L14

    assert _BLEND_WEIGHT_ROS + _BLEND_WEIGHT_YTD + _BLEND_WEIGHT_L14 == pytest.approx(1.0), (
        "Blend weights ROS + YTD + L14 must sum to 1.0 (canonical published blend)."
    )
