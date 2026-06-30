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
