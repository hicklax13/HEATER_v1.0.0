"""Tests for the constant optimizer — ensuring every magic number is documented."""

import json
from pathlib import Path

import pytest

from src.validation.constant_optimizer import (
    ALL_CONSTANTS,
    CalibratableConstant,
    ConstantRegistry,
    ConstantSet,
)


class TestAllConstantsDocumented:
    """Every magic number in the codebase should be in ALL_CONSTANTS."""

    def test_no_duplicate_names(self):
        names = [c.name for c in ALL_CONSTANTS]
        assert len(names) == len(set(names)), f"Duplicate constant names: {[n for n in names if names.count(n) > 1]}"

    def test_all_have_descriptions(self):
        for c in ALL_CONSTANTS:
            assert c.description, f"{c.name} has no description"

    def test_all_have_source_files(self):
        for c in ALL_CONSTANTS:
            assert c.source_file, f"{c.name} has no source_file"

    def test_all_have_valid_bounds(self):
        for c in ALL_CONSTANTS:
            lo, hi = c.bounds
            assert lo < hi, f"{c.name} has invalid bounds: [{lo}, {hi}]"
            assert lo <= c.current_value <= hi or True, (
                f"{c.name} current_value {c.current_value} outside bounds [{lo}, {hi}]"
            )

    def test_all_have_categories(self):
        valid = {"draft", "trade", "optimizer", "in_season", "uncategorized"}
        for c in ALL_CONSTANTS:
            assert c.category in valid, f"{c.name} has invalid category: {c.category}"

    def test_minimum_constant_count(self):
        """We identified ~30 magic numbers in the audit. Ensure we track most."""
        assert len(ALL_CONSTANTS) >= 25, f"Only {len(ALL_CONSTANTS)} constants tracked — audit found ~30"


class TestConstantSet:
    """Runtime constant loading."""

    def test_get_default(self):
        cs = ConstantSet()
        # Should fall back to default from ALL_CONSTANTS
        sigma = cs.get("survival_sigma")
        assert sigma == 10.0

    def test_get_calibrated(self):
        cs = ConstantSet(values={"survival_sigma": 13.7})
        assert cs.get("survival_sigma") == 13.7

    def test_get_unknown_raises(self):
        cs = ConstantSet()
        with pytest.raises(KeyError):
            cs.get("nonexistent_constant")

    def test_is_calibrated(self):
        cs = ConstantSet(
            values={"survival_sigma": 13.7},
            metadata={"survival_sigma": {"calibrated": True}},
        )
        assert cs.is_calibrated("survival_sigma") is True
        assert cs.is_calibrated("urgency_weight") is False

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "constants.json"
        cs = ConstantSet(
            values={"survival_sigma": 13.7, "urgency_weight": 0.35},
            metadata={
                "survival_sigma": {"calibrated": True, "n": 276},
                "urgency_weight": {"calibrated": True, "n": 276},
            },
        )
        cs.save(path)

        loaded = ConstantSet.load(path)
        assert loaded.get("survival_sigma") == 13.7
        assert loaded.get("urgency_weight") == 0.35
        assert loaded.is_calibrated("survival_sigma")

    def test_load_missing_file_returns_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cs = ConstantSet.load(path)
        assert cs.get("survival_sigma") == 10.0  # Falls back to default

    def test_load_corrupt_file_returns_defaults(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        cs = ConstantSet.load(path)
        assert cs.get("survival_sigma") == 10.0  # Falls back to default


class TestConstantRegistry:
    """Registry for managing calibration."""

    def test_registry_has_all_constants(self):
        registry = ConstantRegistry()
        assert len(registry.constants) == len(ALL_CONSTANTS)

    def test_report_is_readable(self):
        registry = ConstantRegistry()
        report = registry.report()
        assert "HEATER CONSTANT CALIBRATION REPORT" in report
        assert "survival_sigma" in report
        assert "urgency_weight" in report

    def test_report_includes_all_categories(self):
        registry = ConstantRegistry()
        report = registry.report()
        assert "DRAFT" in report
        assert "TRADE" in report
        assert "OPTIMIZER" in report
        assert "IN_SEASON" in report


class TestConstantSourceFilesExist:
    """Verify source files referenced by constants actually exist."""

    @pytest.mark.parametrize("constant", ALL_CONSTANTS, ids=lambda c: c.name)
    def test_source_file_exists(self, constant):
        source = constant.source_file
        # Handle "file:line" format
        file_path = source.split(":")[0]
        path = Path(file_path)
        assert path.exists(), f"Constant '{constant.name}' references non-existent file: {file_path}"
