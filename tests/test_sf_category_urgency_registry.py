"""Regression test for Bonus 3: category_urgency reads sigmoid_k from the registry.

Pre-fix, category_urgency.py captured COUNTING_STAT_K and RATE_STAT_K as
module-level literals.  Calibration via scripts/calibrate_sigmoid.py would
rewrite the registry values, but the urgency module never read from the
registry, so calibration was effectively dead code for daily optimizer urgency.

This test verifies that mutating the registry value DOES change the sigmoid
output on the next call to compute_category_urgency().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer import category_urgency
from src.optimizer.category_urgency import compute_category_urgency
from src.optimizer.constants_registry import CONSTANTS_REGISTRY, ConstantEntry
from src.valuation import LeagueConfig


@pytest.fixture
def restore_registry():
    """Save & restore the sigmoid_k entries so test mutations are scoped."""
    saved_counting = CONSTANTS_REGISTRY["sigmoid_k_counting"]
    saved_rate = CONSTANTS_REGISTRY["sigmoid_k_rate"]
    yield
    CONSTANTS_REGISTRY["sigmoid_k_counting"] = saved_counting
    CONSTANTS_REGISTRY["sigmoid_k_rate"] = saved_rate


def _replace_constant(name: str, new_value: float) -> None:
    """Replace a ConstantEntry in the registry with a new value (kept frozen)."""
    old = CONSTANTS_REGISTRY[name]
    CONSTANTS_REGISTRY[name] = ConstantEntry(
        value=new_value,
        lower_bound=old.lower_bound,
        upper_bound=old.upper_bound,
        citation=old.citation,
        module=old.module,
        sensitivity=old.sensitivity,
        description=old.description,
    )


def test_counting_k_read_from_registry(restore_registry):
    """Mutating sigmoid_k_counting should change urgency for a counting-stat gap."""
    config = LeagueConfig()
    # Losing R category by a fixed gap
    my = {"R": 40, "HR": 10}
    opp = {"R": 60, "HR": 10}

    baseline = compute_category_urgency(my, opp, config)
    baseline_r = baseline["R"]

    # Crank counting k WAY up — sigmoid output for the same gap should
    # be sharper (closer to 1.0 for "losing" side).
    _replace_constant("sigmoid_k_counting", 5.0)
    sharpened = compute_category_urgency(my, opp, config)
    sharpened_r = sharpened["R"]

    assert sharpened_r > baseline_r, (
        f"Increasing counting k from 2.0 to 5.0 should increase urgency for a losing gap, "
        f"but {sharpened_r:.4f} <= {baseline_r:.4f}.  Likely the urgency module is "
        f"reading a cached module-level constant instead of the registry."
    )


def test_rate_k_read_from_registry(restore_registry):
    """Mutating sigmoid_k_rate should change urgency for a rate-stat gap."""
    config = LeagueConfig()
    # Losing ERA category (inverse stat — higher ERA means losing)
    my = {"ERA": 4.50}
    opp = {"ERA": 3.50}

    baseline = compute_category_urgency(my, opp, config)
    baseline_era = baseline["ERA"]

    # Crank rate k up — sigmoid should sharpen.
    _replace_constant("sigmoid_k_rate", 5.0)
    sharpened = compute_category_urgency(my, opp, config)
    sharpened_era = sharpened["ERA"]

    assert sharpened_era > baseline_era, (
        f"Increasing rate k from 3.0 to 5.0 should increase urgency for a losing ERA gap, "
        f"but {sharpened_era:.4f} <= {baseline_era:.4f}.  Likely the urgency module is "
        f"reading a cached module-level constant instead of the registry."
    )


def test_module_level_aliases_still_exposed():
    """The legacy COUNTING_STAT_K / RATE_STAT_K module-level aliases should still exist."""
    assert hasattr(category_urgency, "COUNTING_STAT_K")
    assert hasattr(category_urgency, "RATE_STAT_K")
    assert category_urgency.COUNTING_STAT_K == CONSTANTS_REGISTRY["sigmoid_k_counting"].value
    assert category_urgency.RATE_STAT_K == CONSTANTS_REGISTRY["sigmoid_k_rate"].value


def test_get_counting_k_helper_reads_registry(restore_registry):
    """The _get_counting_k() helper must read the current registry value."""
    _replace_constant("sigmoid_k_counting", 4.2)
    assert category_urgency._get_counting_k() == pytest.approx(4.2)


def test_get_rate_k_helper_reads_registry(restore_registry):
    """The _get_rate_k() helper must read the current registry value."""
    _replace_constant("sigmoid_k_rate", 1.8)
    assert category_urgency._get_rate_k() == pytest.approx(1.8)
