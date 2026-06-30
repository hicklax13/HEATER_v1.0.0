"""Tests for the Layer-0 availability survival layer (Phase 1 slice 2)."""

import math

import numpy as np
import pandas as pd
import pytest


def _row(**over):
    base = dict(
        player_id=1,
        name="P",
        is_hitter=1,
        age=28,
        positions="OF",
        health_score=0.90,
        ytd_pa=300,
        ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def test_chronic_availability_monotonic_in_health():
    from src.player_model.availability import chronic_availability

    lo = chronic_availability(health_score=0.60, age=28, is_pitcher=False, position="OF")
    hi = chronic_availability(health_score=0.95, age=28, is_pitcher=False, position="OF")
    assert hi > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_chronic_availability_older_pitcher_lower():
    from src.player_model.availability import chronic_availability

    young = chronic_availability(health_score=0.90, age=25, is_pitcher=True, position="SP")
    old = chronic_availability(health_score=0.90, age=37, is_pitcher=True, position="SP")
    assert old < young  # age/position risk drags an older SP down


def test_chronic_availability_nan_safe():
    from src.player_model.availability import chronic_availability

    v = chronic_availability(health_score=float("nan"), age=None, is_pitcher=False, position=None)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_active_player_full_availability():
    from src.player_model.availability import availability_survival

    s = availability_survival(_row(health_score=1.0, age=27), status=None, weeks_remaining=20)
    assert s.status == "ACTIVE"
    assert s.status_weight == pytest.approx(1.0)
    assert s.il_weeks_out == pytest.approx(0.0)
    assert s.expected_active_fraction == pytest.approx(s.chronic_avail)


def test_il60_player_mostly_out():
    from src.player_model.availability import availability_survival

    healthy = availability_survival(_row(), status=None, weeks_remaining=20)
    il60 = availability_survival(_row(), status="IL60", weeks_remaining=20)
    assert il60.il_weeks_out >= 8.0  # estimate_il_duration("IL60") ~ 10 weeks
    assert il60.expected_active_weeks < healthy.expected_active_weeks
    assert 0.0 <= il60.expected_active_fraction <= 1.0


def test_return_date_overrides_status_curve():
    from src.player_model.availability import availability_survival

    # 14-days-to-return -> _return_date_weight ~ 0.70 near-term weight (more specific than raw IL10).
    s = availability_survival(_row(), status="IL10", expected_return_days=14, weeks_remaining=20)
    assert s.status_weight == pytest.approx(0.70, abs=0.05)


def test_expected_active_weeks_bounded():
    from src.player_model.availability import availability_survival

    s = availability_survival(_row(health_score=0.8), status="IL15", weeks_remaining=12)
    assert 0.0 <= s.expected_active_weeks <= 12.0
    assert 0.0 <= s.weekly_hazard <= 1.0


def test_sample_active_weeks_mean_matches_expectation():
    from src.player_model.availability import availability_survival, sample_active_weeks

    s = availability_survival(_row(health_score=0.9, age=27), status=None, weeks_remaining=20)
    rng = np.random.default_rng(0)
    draws = sample_active_weeks(s, rng, n_samples=5000)
    assert draws.shape == (5000,)
    assert draws.min() >= 0 and draws.max() <= 20
    assert abs(draws.mean() - s.expected_active_weeks) < 0.5  # Monte-Carlo mean ~ expectation


def test_sample_active_weeks_reproducible():
    from src.player_model.availability import availability_survival, sample_active_weeks

    s = availability_survival(_row(), status="IL15", weeks_remaining=15)
    a = sample_active_weeks(s, np.random.default_rng(7), n_samples=100)
    b = sample_active_weeks(s, np.random.default_rng(7), n_samples=100)
    assert np.array_equal(a, b)


def test_sample_active_weeks_zero_when_out_all_season():
    from src.player_model.availability import availability_survival, sample_active_weeks

    s = availability_survival(_row(), status="IL60", weeks_remaining=4)  # IL ~10wk > 4wk left
    draws = sample_active_weeks(s, np.random.default_rng(1), n_samples=50)
    assert draws.max() == 0  # sidelined the whole remaining window


def test_never_raises_on_sparse_row():
    from src.player_model.availability import availability_survival, sample_active_weeks

    sparse = pd.Series({"player_id": 9})  # no is_hitter/health/age/positions
    s = availability_survival(sparse, status=None, weeks_remaining=None)
    assert math.isfinite(s.chronic_avail) and 0.0 <= s.chronic_avail <= 1.0
    assert math.isfinite(s.expected_active_weeks)
    draws = sample_active_weeks(s, np.random.default_rng(0), n_samples=10)
    assert draws.shape == (10,)


def test_garbage_status_treated_active():
    from src.player_model.availability import availability_survival

    s = availability_survival(_row(), status="not-a-real-status", weeks_remaining=20)
    assert s.status == "ACTIVE"  # unrecognized -> classify_il_type None -> active
    assert s.il_weeks_out == pytest.approx(0.0)
