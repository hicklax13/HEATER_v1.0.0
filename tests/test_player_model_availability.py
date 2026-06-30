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
