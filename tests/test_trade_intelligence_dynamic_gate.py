"""Tests for dynamic FA gate threshold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import compute_dynamic_fa_threshold


def test_early_season_threshold_is_higher():
    threshold = compute_dynamic_fa_threshold(avg_pa=20)
    assert threshold >= 0.80


def test_mid_season_threshold_is_standard():
    threshold = compute_dynamic_fa_threshold(avg_pa=250)
    assert 0.65 <= threshold <= 0.75


def test_late_season_threshold_is_lower():
    threshold = compute_dynamic_fa_threshold(avg_pa=500)
    assert threshold <= 0.65


def test_zero_pa_uses_max_threshold():
    threshold = compute_dynamic_fa_threshold(avg_pa=0)
    assert threshold >= 0.85
