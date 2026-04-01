"""Tests for schedule-aware trade urgency."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import compute_schedule_urgency


def test_urgency_returns_float():
    result = compute_schedule_urgency(weeks_ahead=3)
    assert isinstance(result, float)


def test_urgency_in_range():
    result = compute_schedule_urgency(weeks_ahead=3)
    assert 0.85 <= result <= 1.25


def test_urgency_with_zero_weeks():
    result = compute_schedule_urgency(weeks_ahead=0)
    assert result == 1.0
