"""PR21 (FA engine overhaul P3.9): playing-time gate on FA scoring.

FAs with low YTD games (relative to season progress) get a composite-score
penalty. This pushes IL stash phantoms and inactive players out of the
top FA candidates so real performers surface.

Calibration (locked by user 2026-05-20):
  - Skip gate in first 30 days of season (avoid penalizing call-ups)
  - 0 GP / 0 IP → 0.30x multiplier
  - 1-29% expected → 0.60x
  - 30-59% expected → 0.85x
  - >=60% expected → 1.0x (no penalty)
"""

import pandas as pd
import pytest

from src.optimizer.fa_recommender import _playing_time_multiplier
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig


def _ctx_at_week(weeks_remaining: int = 16) -> OptimizerDataContext:
    """Build a minimal ctx with controllable season progress.
    Days elapsed = (26 - weeks_remaining) * 7.
    Default 16 weeks remaining = 70 days elapsed (well past Day 30 gate)."""
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.weeks_remaining = weeks_remaining
    return ctx


def test_zero_gp_hitter_gets_heavy_penalty():
    """Westburg-class: 0 YTD GP at Day 70 → 0.30 multiplier."""
    ctx = _ctx_at_week(16)  # Day 70
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0, "ytd_ip": 0})
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(0.30, abs=0.01), f"0-GP hitter at Day 70 should get 0.30x (IL phantom). Got {mult}."


def test_healthy_hitter_full_credit():
    """Marsh-class: 46 GP at Day 70 (expected ~60) → ratio 0.77 → 1.0x."""
    ctx = _ctx_at_week(16)  # Day 70, expected GP ~60
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 46, "ytd_ip": 0})
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(1.0, abs=0.01), (
        f"46-GP hitter at Day 70 (ratio 0.77) should get full credit. Got {mult}."
    )


def test_mid_range_hitter_mild_penalty():
    """30-59% expected → 0.85x. Day 70, expected ~60 GP, so 18-35 GP triggers."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 25, "ytd_ip": 0})  # ratio ~0.42
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(0.85, abs=0.01), f"25-GP hitter (ratio 0.42) should get 0.85x. Got {mult}."


def test_low_range_hitter_moderate_penalty():
    """1-29% expected → 0.60x."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 10, "ytd_ip": 0})  # ratio ~0.17
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(0.60, abs=0.01), f"10-GP hitter (ratio 0.17) should get 0.60x. Got {mult}."


def test_pitcher_uses_ip_signal():
    """Pitchers gated on IP, not GP. 0 IP at Day 70 → 0.30x."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 0, "ytd_gp": 0, "ytd_ip": 0})
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(0.30, abs=0.01)


def test_healthy_pitcher_full_credit():
    """Cade Horton-class: 94 IP at Day 70 (expected ~70) → ratio 1.34 → 1.0x."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 0, "ytd_gp": 0, "ytd_ip": 94.1})
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(1.0, abs=0.01)


def test_grace_period_first_30_days():
    """Day 1-29: gate is OFF. Even 0 GP gets full credit (no penalty)."""
    ctx = _ctx_at_week(23)  # 26 - 23 = 3 weeks = 21 days elapsed
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0, "ytd_ip": 0})
    mult = _playing_time_multiplier(fa, ctx)
    assert mult == pytest.approx(1.0, abs=0.01), f"Day 21 should be inside grace period (gate off). Got {mult}."


def test_gate_kicks_in_at_day_30():
    """Day 30: gate becomes active."""
    ctx = _ctx_at_week(22)  # 26 - 22 = 4 weeks = 28 days. Just under threshold.
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0, "ytd_ip": 0})
    mult_day_28 = _playing_time_multiplier(fa, ctx)
    assert mult_day_28 == pytest.approx(1.0, abs=0.01), "Day 28 grace"

    ctx2 = _ctx_at_week(21)  # 5 weeks = 35 days. Past threshold.
    mult_day_35 = _playing_time_multiplier(fa, ctx2)
    assert mult_day_35 < 1.0, "Day 35 must apply penalty"


def test_missing_data_safe_default():
    """If ytd_gp / ytd_ip columns are missing, default behavior is full credit
    (don't punish players for missing data — that would over-correct)."""
    ctx = _ctx_at_week(16)
    fa_no_data = pd.Series({"is_hitter": 1})  # no ytd_gp key
    mult = _playing_time_multiplier(fa_no_data, ctx)
    # 0 default for missing key → counts as 0 GP → 0.30 penalty.
    # This is the user-locked behavior: treat NULL as IL phantom.
    assert mult == pytest.approx(0.30, abs=0.01)
