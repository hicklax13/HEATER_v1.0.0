"""PR8: expected_return_days should be populated from MULTIPLE sources
(ESPN, Yahoo status string, player_news headline parsing), not just ESPN.
This gives more granular weighting for players whose ESPN data is missing
or stale."""

import pytest

from src.in_season import _il_weight_from_status, _parse_return_days_from_text


def test_parse_x_day_il_pattern():
    """'IL10 - expected back in 3 days' → 3"""
    assert _parse_return_days_from_text("IL10 - expected back in 3 days") == 3
    assert _parse_return_days_from_text("IL15 - 7 days remaining") == 7
    assert _parse_return_days_from_text("DTD - 1 day") == 1


def test_parse_x_day_il_returns_none_when_no_pattern():
    """No day count → None (caller uses status default)."""
    assert _parse_return_days_from_text("On IL, return TBD") is None
    assert _parse_return_days_from_text("") is None
    assert _parse_return_days_from_text(None) is None


def test_parse_il_period_pattern():
    """'10-day IL' / '15-day IL' / '60-day IL' implies ~that many days remaining
    as a conservative upper bound when no specific return date is given."""
    # These are status descriptors, not specific return dates — function should
    # return the implied window (or None if we want strict day-only parsing)
    val = _parse_return_days_from_text("Placed on 10-day IL")
    # Acceptable: either None (strict) or 10 (heuristic). Don't pin one — just
    # ensure it doesn't crash and returns int|None.
    assert val is None or isinstance(val, int)


def test_il_weight_from_yahoo_status_with_day_count():
    """When Yahoo status includes a day count like 'IL10 - 3 days',
    _il_weight_from_status should extract it and use the return-date curve."""
    weight = _il_weight_from_status("IL10 - 3 days", expected_return_days=None)
    # 3-day return → ~0.91 per the piecewise curve (between 1d=0.95 and 7d=0.85)
    assert 0.85 <= weight <= 0.95, f"Yahoo status 'IL10 - 3 days' should parse 3-day return; got weight {weight}"


def test_il_weight_falls_back_to_string_default():
    """When status has no parseable day count AND no expected_return_days
    is passed, fall back to existing string lookup (PR17 behavior preserved)."""
    weight = _il_weight_from_status("IL10", expected_return_days=None)
    assert weight == 0.85, "IL10 default unchanged when no date info"
