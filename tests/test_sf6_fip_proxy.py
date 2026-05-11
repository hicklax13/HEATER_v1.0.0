"""SF-6 Option B: FIP/xFIP proxy for K-boost when Stuff+ is unavailable.

CLAUDE.md SF-6: FanGraphs blocks the Stuff+ scrape with HTTP 403, leaving
``stuff_plus`` NULL for every pitcher. Wave 4-J Option A made the helper a
provable no-op (returns 1.0). Option B (this module's tests) extends the
helper so a valid FIP / xFIP yields a derived multiplier in [0.85, 1.15].

xFIP wins over FIP when both are present (it normalises HR luck).
Real Stuff+ values, when available, win over the FIP proxy (the proxy is
a much weaker signal).
"""

from __future__ import annotations

import math

import pytest

from src.optimizer.daily_optimizer import _stuff_plus_k_multiplier

# ---------------------------------------------------------------------------
# Tier 1: real Stuff+ wins over FIP
# ---------------------------------------------------------------------------


def test_stuff_plus_wins_over_fip_when_both_present():
    """Real Stuff+ value (when valid) takes precedence over the FIP proxy."""
    # stuff_plus 110 (>110 ramp -> 1.05x); fip 4.5 would give proxy ~0.95x.
    # Helper must return 1.05, not 0.95.
    assert _stuff_plus_k_multiplier(stuff_plus=110.5, fip=4.5) == pytest.approx(1.05)


def test_stuff_plus_125_wins_over_xfip():
    """Top-tier Stuff+ (>120) -> 1.10x even with neutral xfip."""
    assert _stuff_plus_k_multiplier(stuff_plus=125, fip=None, xfip=4.0) == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# Tier 2: FIP proxy when Stuff+ is missing
# ---------------------------------------------------------------------------


def test_fip_proxy_active_when_stuff_plus_missing():
    """Stuff+ missing + good FIP (3.0) -> proxy returns >1.0."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=3.0)
    assert result > 1.0
    # 1.0 + (4.0 - 3.0) / 10.0 = 1.10
    assert result == pytest.approx(1.10)


def test_fip_proxy_below_one_when_fip_high():
    """High FIP (5.0) -> proxy returns <1.0."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=5.0)
    assert result < 1.0
    # 1.0 + (4.0 - 5.0) / 10.0 = 0.90
    assert result == pytest.approx(0.90)


def test_fip_proxy_neutral_at_league_average():
    """League-average FIP (4.0) -> proxy returns ~1.0."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=4.0)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tier 2 (alt): xFIP preferred over FIP when both present
# ---------------------------------------------------------------------------


def test_xfip_used_when_only_xfip_present():
    """Stuff+ missing, FIP missing, xFIP=3.0 -> uses xFIP proxy."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=None, xfip=3.0)
    assert result == pytest.approx(1.10)


def test_xfip_preferred_over_fip_when_both_present():
    """When Stuff+ is missing but both FIP and xFIP are present, xFIP wins.

    xFIP normalises home-run luck, so it's a cleaner stuff proxy.
    Pick xfip=3.0 (proxy 1.10) and fip=5.0 (proxy 0.90) -- if xfip is used
    the result is 1.10; if fip leaks through it would be 0.90.
    """
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=5.0, xfip=3.0)
    assert result == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# Tier 3: nothing usable -> neutral 1.0
# ---------------------------------------------------------------------------


def test_neutral_when_all_inputs_missing():
    """Stuff+, FIP, xFIP all missing -> 1.0 (neutral)."""
    assert _stuff_plus_k_multiplier(stuff_plus=None, fip=None, xfip=None) == 1.0


def test_neutral_when_all_inputs_zero():
    """Sentinel zeros for every input -> still neutral 1.0."""
    assert _stuff_plus_k_multiplier(stuff_plus=0, fip=0, xfip=0) == 1.0


def test_neutral_when_fip_nan():
    """NaN FIP -> falls through to neutral 1.0."""
    assert _stuff_plus_k_multiplier(stuff_plus=None, fip=float("nan")) == 1.0


def test_neutral_when_inputs_non_numeric():
    """Non-numeric inputs -> 1.0 (helper must not crash)."""
    assert _stuff_plus_k_multiplier(stuff_plus="N/A", fip="bad") == 1.0


# ---------------------------------------------------------------------------
# Bounds: proxy must be clipped to [0.85, 1.15]
# ---------------------------------------------------------------------------


def test_fip_proxy_clipped_to_upper_bound():
    """Absurdly low FIP (1.0) would give proxy 1.30 -- must clip to 1.15."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=1.0)
    assert result == pytest.approx(1.15)
    assert result <= 1.15


def test_fip_proxy_clipped_to_lower_bound():
    """Disastrous FIP (8.0) would give proxy 0.60 -- must clip to 0.85."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=8.0)
    assert result == pytest.approx(0.85)
    assert result >= 0.85


def test_xfip_proxy_also_clipped():
    """Lower-bound clip applies to xFIP path too."""
    result = _stuff_plus_k_multiplier(stuff_plus=None, fip=None, xfip=10.0)
    assert result == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Wave 4-J Option A regression: original ramp + invalid-input contract
# ---------------------------------------------------------------------------


def test_wave_4j_ramp_120_returns_110():
    """Wave 4-J: stuff_plus > 120 -> 1.10x."""
    assert _stuff_plus_k_multiplier(stuff_plus=125) == pytest.approx(1.10)


def test_wave_4j_ramp_110_returns_105():
    """Wave 4-J: stuff_plus in (110, 120] -> 1.05x."""
    assert _stuff_plus_k_multiplier(stuff_plus=115) == pytest.approx(1.05)


def test_wave_4j_ramp_below_110_returns_100():
    """Wave 4-J: stuff_plus <= 110 -> 1.00x."""
    assert _stuff_plus_k_multiplier(stuff_plus=105) == pytest.approx(1.00)


def test_wave_4j_neutral_for_negative_stuff():
    """Negative Stuff+ is bad data -> 1.0 (and FIP fallback if available)."""
    # Negative + no FIP -> 1.0
    assert _stuff_plus_k_multiplier(stuff_plus=-5) == 1.0
    # Negative Stuff+ + valid FIP -> proxy kicks in
    assert _stuff_plus_k_multiplier(stuff_plus=-5, fip=3.0) == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# Math sanity: result is always within reasonable global bounds
# ---------------------------------------------------------------------------


def test_result_always_finite():
    """Helper must never return NaN/inf for any input combination."""
    test_cases = [
        (None, None, None),
        (0, 0, 0),
        (100, 4.0, 4.0),
        (None, 3.5, 3.5),
        ("bad", "input", "values"),
        (float("nan"), float("nan"), float("nan")),
        (-1, -1, -1),
        (200, 0.5, 0.5),
    ]
    for stuff, fip, xfip in test_cases:
        result = _stuff_plus_k_multiplier(stuff, fip=fip, xfip=xfip)
        assert math.isfinite(result), f"Got non-finite for stuff={stuff} fip={fip} xfip={xfip}"
        assert 0.85 <= result <= 1.15, f"Out of bounds: {result} for stuff={stuff} fip={fip} xfip={xfip}"
