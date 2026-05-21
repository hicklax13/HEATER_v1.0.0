"""PR17 (FA engine overhaul P3.5): _il_weight_from_status must use the
return-date curve when ESPN/Yahoo provides expected_return_days, instead
of treating all 'Suspended'/'NA'/'Restricted' as season-ending."""

import pytest

from src.in_season import _il_weight_from_status


def test_short_suspension_with_return_date_returns_high_weight():
    """Framber Valdez scenario: status='Suspended', returns in 1 day.
    OLD behavior: weight=0.0 (any Suspended is season-ending).
    NEW behavior: weight ~= 0.95 (1-day return is essentially active)."""
    weight = _il_weight_from_status("Suspended", expected_return_days=1.0)
    assert 0.90 <= weight <= 1.0, (
        f"1-day suspension should weight near 1.0; got {weight:.3f}. "
        f"The engine must not treat short suspensions as free drops."
    )


def test_indefinite_suspension_without_return_date_returns_zero():
    """No return date -> fall through to existing string lookup.
    Suspended/NA/Restricted without ETA -> 0.0 (season-ending behavior preserved)."""
    weight = _il_weight_from_status("Suspended", expected_return_days=None)
    assert weight == 0.0


def test_return_date_curve_anchors():
    """Verify the locked anchor points."""
    # 0-day return = back today = 1.0
    assert _il_weight_from_status("OUT", expected_return_days=0) == pytest.approx(1.0, abs=0.01)
    # 1-day = 0.95
    assert _il_weight_from_status("OUT", expected_return_days=1) == pytest.approx(0.95, abs=0.01)
    # 7-day = 0.85
    assert _il_weight_from_status("OUT", expected_return_days=7) == pytest.approx(0.85, abs=0.01)
    # 14-day = 0.70 (anchors to IL10 default)
    assert _il_weight_from_status("OUT", expected_return_days=14) == pytest.approx(0.70, abs=0.01)
    # 21-day = 0.55
    assert _il_weight_from_status("OUT", expected_return_days=21) == pytest.approx(0.55, abs=0.01)


def test_negative_days_returns_full_weight():
    """ESPN says return date was yesterday -> player is back -> weight=1.0."""
    weight = _il_weight_from_status("DTD", expected_return_days=-1.0)
    assert weight == 1.0


def test_above_60_days_falls_through_to_status():
    """Beyond 60-day ceiling, return-date curve returns None, caller falls
    through to existing status string lookup. 'Suspended' with 90-day return
    is effectively season-ending -> 0.0."""
    weight = _il_weight_from_status("Suspended", expected_return_days=90.0)
    assert weight == 0.0


def test_active_status_ignores_return_date():
    """Active players: return-date is moot -- weight=1.0 always."""
    assert _il_weight_from_status("Active", expected_return_days=None) == 1.0
    assert _il_weight_from_status("Active", expected_return_days=10.0) == 1.0


def test_il10_string_unchanged_no_return_date():
    """When no return-date data exists, IL10/IL15/IL60 string lookups
    return their established defaults -- full backward compat."""
    assert _il_weight_from_status("IL10") == 0.85
    assert _il_weight_from_status("IL15") == 0.70
    assert _il_weight_from_status("IL60") == 0.20


def test_il10_with_short_return_date_upgrades():
    """An IL10 player with a 1-day return date should weight HIGHER
    than the IL10 default (0.85), since we have specific info."""
    weight = _il_weight_from_status("IL10", expected_return_days=1.0)
    assert weight > 0.85, f"IL10 with 1-day return should outweigh default IL10=0.85; got {weight}."


def test_invalid_return_days_falls_through():
    """Non-numeric expected_return_days -> fall through, don't crash."""
    # str that doesn't parse to float
    weight = _il_weight_from_status("Suspended", expected_return_days="N/A")
    assert weight == 0.0  # falls through to string lookup -> 0.0
