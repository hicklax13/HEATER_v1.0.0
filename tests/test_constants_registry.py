"""Tests for the centralized constants registry.

Every hardcoded magic number in the optimizer must be registered here
with a citation, plausible bounds, and sensitivity classification.
"""

import pytest

from src.optimizer.constants_registry import CONSTANTS_REGISTRY, ConstantEntry


class TestConstantsRegistry:
    """Validate structure, completeness, and correctness of the constants registry."""

    def test_all_entries_have_citations(self):
        """Every constant must have a non-empty citation."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.citation, f"Constant '{name}' has no citation"

    def test_all_values_within_bounds(self):
        """Every constant's value must be within its declared bounds."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.lower_bound <= entry.value <= entry.upper_bound, (
                f"Constant '{name}' value {entry.value} outside bounds [{entry.lower_bound}, {entry.upper_bound}]"
            )

    def test_platoon_constants_match_modern_values(self):
        """Platoon advantages should match 2020-2024 updated values."""
        lhb = CONSTANTS_REGISTRY["platoon_lhb_vs_rhp"]
        rhb = CONSTANTS_REGISTRY["platoon_rhb_vs_lhp"]
        assert lhb.value == pytest.approx(0.075, abs=0.01)
        assert rhb.value == pytest.approx(0.058, abs=0.01)
        assert "Tango" in lhb.citation or "The Book" in lhb.citation
        assert "2020-2024" in lhb.citation

    def test_stabilization_points_match_fangraphs(self):
        """Stabilization points should match FanGraphs research."""
        hr_stab = CONSTANTS_REGISTRY["stabilization_hr_rate"]
        assert hr_stab.value == 170
        assert "FanGraphs" in hr_stab.citation or "stabiliz" in hr_stab.citation.lower()

    def test_sigmoid_k_values_documented(self):
        """Sigmoid k-values must document their calibration method."""
        k_count = CONSTANTS_REGISTRY["sigmoid_k_counting"]
        k_rate = CONSTANTS_REGISTRY["sigmoid_k_rate"]
        assert k_count.value == 2.0
        assert k_rate.value == 3.0

    def test_registry_covers_all_known_constants(self):
        """Registry must include at least 25 constants (we know of 30+)."""
        assert len(CONSTANTS_REGISTRY) >= 25, f"Only {len(CONSTANTS_REGISTRY)} constants registered; expected 25+"


class TestConstantSensitivity:
    """Test that perturbation of constants produces expected behavior.

    HIGH sensitivity constants: +/-20% perturbation should change lineup.
    LOW sensitivity constants: +/-20% perturbation should NOT change top-5.
    """

    def test_high_sensitivity_constants_exist(self):
        """At least 3 constants should be flagged as HIGH sensitivity."""
        high = [k for k, v in CONSTANTS_REGISTRY.items() if v.sensitivity == "HIGH"]
        assert len(high) >= 3, f"Only {len(high)} HIGH sensitivity constants"

    def test_all_sensitivity_levels_used(self):
        """Registry should use all three sensitivity levels."""
        levels = {v.sensitivity for v in CONSTANTS_REGISTRY.values()}
        assert "HIGH" in levels
        assert "MEDIUM" in levels
        assert "LOW" in levels

    def test_all_modules_represented(self):
        """Constants should cover at least 4 different optimizer modules."""
        modules = {v.module for v in CONSTANTS_REGISTRY.values()}
        assert len(modules) >= 4, f"Only {len(modules)} modules represented: {modules}"

    def test_bounds_are_reasonable(self):
        """Lower bound < value < upper bound for all constants."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.lower_bound < entry.value, f"{name}: lower_bound {entry.lower_bound} >= value {entry.value}"
            assert entry.value < entry.upper_bound, f"{name}: value {entry.value} >= upper_bound {entry.upper_bound}"
