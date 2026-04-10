"""Tests for the optimizer sensitivity analysis framework.

Verifies:
  - compute_lineup_diff correctly counts symmetric differences
  - compute_weight_diff finds the largest absolute delta
  - _classify_sensitivity maps thresholds correctly
  - _get_patchable_constants filters properly
  - run_sensitivity_analysis returns SensitivityResult objects end-to-end
  - summarize_results produces a well-formed DataFrame
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer.sensitivity_analysis import (
    CONSTANT_PATCH_TARGETS,
    SensitivityResult,
    _classify_sensitivity,
    _compute_total_value,
    _get_patchable_constants,
    compute_lineup_diff,
    compute_weight_diff,
    run_sensitivity_analysis,
    summarize_results,
)

# ── Roster fixture (reused from test_optimizer_integration.py) ──────

_HITTERS = [
    {
        "player_id": 101,
        "player_name": "C_Starter",
        "positions": "C",
        "is_hitter": 1,
        "team": "NYY",
        "r": 55,
        "hr": 18,
        "rbi": 60,
        "sb": 2,
        "avg": 0.245,
        "obp": 0.320,
        "h": 110,
        "ab": 449,
        "bb": 40,
        "hbp": 3,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 102,
        "player_name": "1B_Starter",
        "positions": "1B",
        "is_hitter": 1,
        "team": "BOS",
        "r": 85,
        "hr": 32,
        "rbi": 95,
        "sb": 3,
        "avg": 0.275,
        "obp": 0.360,
        "h": 150,
        "ab": 545,
        "bb": 60,
        "hbp": 5,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 103,
        "player_name": "2B_Starter",
        "positions": "2B",
        "is_hitter": 1,
        "team": "HOU",
        "r": 75,
        "hr": 15,
        "rbi": 55,
        "sb": 20,
        "avg": 0.270,
        "obp": 0.340,
        "h": 140,
        "ab": 519,
        "bb": 45,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 104,
        "player_name": "3B_Starter",
        "positions": "3B",
        "is_hitter": 1,
        "team": "LAD",
        "r": 90,
        "hr": 28,
        "rbi": 85,
        "sb": 8,
        "avg": 0.280,
        "obp": 0.370,
        "h": 155,
        "ab": 554,
        "bb": 65,
        "hbp": 4,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 105,
        "player_name": "SS_Starter",
        "positions": "SS",
        "is_hitter": 1,
        "team": "ATL",
        "r": 80,
        "hr": 22,
        "rbi": 70,
        "sb": 15,
        "avg": 0.265,
        "obp": 0.335,
        "h": 138,
        "ab": 521,
        "bb": 48,
        "hbp": 3,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 106,
        "player_name": "OF1_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "NYY",
        "r": 95,
        "hr": 35,
        "rbi": 100,
        "sb": 12,
        "avg": 0.290,
        "obp": 0.380,
        "h": 162,
        "ab": 559,
        "bb": 70,
        "hbp": 5,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 107,
        "player_name": "OF2_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "SEA",
        "r": 70,
        "hr": 20,
        "rbi": 65,
        "sb": 25,
        "avg": 0.260,
        "obp": 0.330,
        "h": 135,
        "ab": 519,
        "bb": 42,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 108,
        "player_name": "OF3_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "CHC",
        "r": 78,
        "hr": 24,
        "rbi": 72,
        "sb": 8,
        "avg": 0.272,
        "obp": 0.345,
        "h": 142,
        "ab": 522,
        "bb": 50,
        "hbp": 4,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 109,
        "player_name": "Util1",
        "positions": "1B,OF",
        "is_hitter": 1,
        "team": "PHI",
        "r": 65,
        "hr": 20,
        "rbi": 68,
        "sb": 5,
        "avg": 0.255,
        "obp": 0.325,
        "h": 125,
        "ab": 490,
        "bb": 38,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 110,
        "player_name": "Util2",
        "positions": "3B,SS",
        "is_hitter": 1,
        "team": "SDP",
        "r": 60,
        "hr": 14,
        "rbi": 50,
        "sb": 18,
        "avg": 0.262,
        "obp": 0.330,
        "h": 130,
        "ab": 496,
        "bb": 40,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
]

_PITCHERS = [
    {
        "player_id": 201,
        "player_name": "SP1_Ace",
        "positions": "SP",
        "is_hitter": 0,
        "team": "LAD",
        "w": 15,
        "l": 6,
        "sv": 0,
        "k": 220,
        "era": 2.90,
        "whip": 1.05,
        "ip": 200.0,
        "er": 64,
        "bb_allowed": 45,
        "h_allowed": 165,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 202,
        "player_name": "SP2_Mid",
        "positions": "SP",
        "is_hitter": 0,
        "team": "NYY",
        "w": 12,
        "l": 8,
        "sv": 0,
        "k": 180,
        "era": 3.60,
        "whip": 1.18,
        "ip": 180.0,
        "er": 72,
        "bb_allowed": 50,
        "h_allowed": 162,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 203,
        "player_name": "SP3_Back",
        "positions": "SP",
        "is_hitter": 0,
        "team": "BOS",
        "w": 9,
        "l": 10,
        "sv": 0,
        "k": 150,
        "era": 4.20,
        "whip": 1.30,
        "ip": 165.0,
        "er": 77,
        "bb_allowed": 55,
        "h_allowed": 160,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 204,
        "player_name": "RP1_Closer",
        "positions": "RP",
        "is_hitter": 0,
        "team": "NYY",
        "w": 4,
        "l": 3,
        "sv": 30,
        "k": 70,
        "era": 2.80,
        "whip": 1.00,
        "ip": 65.0,
        "er": 20,
        "bb_allowed": 18,
        "h_allowed": 47,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 205,
        "player_name": "RP2_Setup",
        "positions": "RP",
        "is_hitter": 0,
        "team": "ATL",
        "w": 5,
        "l": 2,
        "sv": 5,
        "k": 75,
        "era": 3.10,
        "whip": 1.08,
        "ip": 70.0,
        "er": 24,
        "bb_allowed": 20,
        "h_allowed": 56,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 206,
        "player_name": "P_Flex1",
        "positions": "SP,RP",
        "is_hitter": 0,
        "team": "HOU",
        "w": 7,
        "l": 5,
        "sv": 2,
        "k": 110,
        "era": 3.80,
        "whip": 1.22,
        "ip": 120.0,
        "er": 51,
        "bb_allowed": 35,
        "h_allowed": 112,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 207,
        "player_name": "P_Flex2",
        "positions": "SP,RP",
        "is_hitter": 0,
        "team": "SDP",
        "w": 6,
        "l": 4,
        "sv": 3,
        "k": 100,
        "era": 3.90,
        "whip": 1.25,
        "ip": 110.0,
        "er": 48,
        "bb_allowed": 32,
        "h_allowed": 106,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 208,
        "player_name": "P_Flex3",
        "positions": "RP",
        "is_hitter": 0,
        "team": "CHC",
        "w": 3,
        "l": 2,
        "sv": 8,
        "k": 60,
        "era": 3.40,
        "whip": 1.12,
        "ip": 55.0,
        "er": 21,
        "bb_allowed": 15,
        "h_allowed": 47,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
]

_BENCH = [
    {
        "player_id": 301,
        "player_name": "BN_Hitter1",
        "positions": "OF,1B",
        "is_hitter": 1,
        "team": "MIA",
        "r": 45,
        "hr": 12,
        "rbi": 40,
        "sb": 6,
        "avg": 0.242,
        "obp": 0.310,
        "h": 105,
        "ab": 434,
        "bb": 30,
        "hbp": 2,
        "sf": 2,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 302,
        "player_name": "BN_Hitter2",
        "positions": "2B,SS",
        "is_hitter": 1,
        "team": "CIN",
        "r": 50,
        "hr": 10,
        "rbi": 35,
        "sb": 12,
        "avg": 0.248,
        "obp": 0.315,
        "h": 108,
        "ab": 435,
        "bb": 32,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 303,
        "player_name": "BN_Pitcher",
        "positions": "SP",
        "is_hitter": 0,
        "team": "TBR",
        "w": 5,
        "l": 7,
        "sv": 0,
        "k": 90,
        "era": 4.50,
        "whip": 1.35,
        "ip": 100.0,
        "er": 50,
        "bb_allowed": 40,
        "h_allowed": 95,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 304,
        "player_name": "IL_Player",
        "positions": "OF",
        "is_hitter": 1,
        "team": "LAD",
        "r": 70,
        "hr": 22,
        "rbi": 65,
        "sb": 10,
        "avg": 0.268,
        "obp": 0.340,
        "h": 130,
        "ab": 485,
        "bb": 45,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "IL10",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 305,
        "player_name": "BN_Catcher",
        "positions": "C,1B",
        "is_hitter": 1,
        "team": "STL",
        "r": 40,
        "hr": 8,
        "rbi": 32,
        "sb": 1,
        "avg": 0.235,
        "obp": 0.300,
        "h": 90,
        "ab": 383,
        "bb": 25,
        "hbp": 2,
        "sf": 2,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
]


@pytest.fixture()
def full_roster():
    """23-player roster matching Yahoo H2H format."""
    return pd.DataFrame(_HITTERS + _PITCHERS + _BENCH)


# ── Unit tests: pure functions ──────────────────────────────────────


class TestComputeLineupDiff:
    """Verify symmetric-difference counting of lineup assignments."""

    def test_identical_lineups_zero_diff(self):
        """Identical lineups should have 0 difference."""
        a = [{"player_id": 1}, {"player_id": 2}]
        assert compute_lineup_diff(a, a) == 0

    def test_one_player_swapped(self):
        """One player swapped should count as 2 (one removed, one added)."""
        a = [{"player_id": 1}, {"player_id": 2}]
        b = [{"player_id": 1}, {"player_id": 3}]
        assert compute_lineup_diff(a, b) == 2

    def test_completely_different(self):
        """Entirely different rosters count all players."""
        a = [{"player_id": 1}, {"player_id": 2}]
        b = [{"player_id": 3}, {"player_id": 4}]
        assert compute_lineup_diff(a, b) == 4

    def test_empty_lineups(self):
        """Two empty lineups have 0 difference."""
        assert compute_lineup_diff([], []) == 0

    def test_one_empty(self):
        """One empty lineup counts all players from the other."""
        a = [{"player_id": 1}, {"player_id": 2}, {"player_id": 3}]
        assert compute_lineup_diff(a, []) == 3

    def test_superset(self):
        """Extra players in the perturbed lineup get counted."""
        a = [{"player_id": 1}]
        b = [{"player_id": 1}, {"player_id": 2}]
        assert compute_lineup_diff(a, b) == 1


class TestComputeWeightDiff:
    """Verify max absolute weight delta computation."""

    def test_identical_weights(self):
        """Same weights -> 0 diff."""
        w = {"R": 1.0, "HR": 1.5}
        assert compute_weight_diff(w, w) == pytest.approx(0.0)

    def test_one_category_differs(self):
        """Single category change returns that delta."""
        a = {"R": 1.0, "HR": 1.5}
        b = {"R": 1.2, "HR": 1.5}
        assert compute_weight_diff(a, b) == pytest.approx(0.2)

    def test_multiple_diffs_returns_max(self):
        """With multiple changes, return the largest."""
        a = {"R": 1.0, "HR": 1.5, "SB": 2.0}
        b = {"R": 1.1, "HR": 2.0, "SB": 2.3}
        # diffs: 0.1, 0.5, 0.3 -> max is 0.5
        assert compute_weight_diff(a, b) == pytest.approx(0.5)

    def test_extra_keys_ignored(self):
        """Keys only in one dict do not affect the result."""
        a = {"R": 1.0}
        b = {"R": 1.0, "HR": 5.0}
        assert compute_weight_diff(a, b) == pytest.approx(0.0)

    def test_empty_weights(self):
        """Empty dicts produce 0."""
        assert compute_weight_diff({}, {}) == pytest.approx(0.0)


class TestClassifySensitivity:
    """Verify threshold-based sensitivity classification."""

    def test_high_by_lineup_changes(self):
        assert _classify_sensitivity(3, 0.0) == "HIGH"
        assert _classify_sensitivity(5, 1.0) == "HIGH"

    def test_high_by_value_change(self):
        assert _classify_sensitivity(0, 5.0) == "HIGH"
        assert _classify_sensitivity(0, -6.0) == "HIGH"

    def test_medium_by_lineup_changes(self):
        assert _classify_sensitivity(1, 0.0) == "MEDIUM"
        assert _classify_sensitivity(2, 1.5) == "MEDIUM"

    def test_medium_by_value_change(self):
        assert _classify_sensitivity(0, 2.0) == "MEDIUM"
        assert _classify_sensitivity(0, -3.5) == "MEDIUM"

    def test_low(self):
        assert _classify_sensitivity(0, 0.0) == "LOW"
        assert _classify_sensitivity(0, 1.9) == "LOW"


class TestComputeTotalValue:
    """Verify weighted value computation."""

    def test_basic_sum(self):
        stats = {"R": 10.0, "HR": 5.0}
        weights = {"R": 2.0, "HR": 3.0}
        assert _compute_total_value(stats, weights) == pytest.approx(35.0)

    def test_missing_stat_ignored(self):
        stats = {"R": 10.0}
        weights = {"R": 2.0, "HR": 3.0}
        assert _compute_total_value(stats, weights) == pytest.approx(20.0)

    def test_empty_returns_zero(self):
        assert _compute_total_value({}, {}) == pytest.approx(0.0)


class TestGetPatchableConstants:
    """Verify filtering of patchable constant names."""

    def test_returns_sorted_list(self):
        result = _get_patchable_constants()
        assert result == sorted(result)
        assert len(result) > 0

    def test_all_results_in_both_dicts(self):
        """Every returned name must exist in registry AND patch targets."""
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        for name in _get_patchable_constants():
            assert name in CONSTANTS_REGISTRY
            assert name in CONSTANT_PATCH_TARGETS

    def test_filter_by_name(self):
        result = _get_patchable_constants(["platoon_lhb_vs_rhp", "nonexistent"])
        assert result == ["platoon_lhb_vs_rhp"]

    def test_filter_all_nonexistent(self):
        result = _get_patchable_constants(["nonexistent_a", "nonexistent_b"])
        assert result == []


# ── Integration: full pipeline run ──────────────────────────────────


class TestRunSensitivityAnalysis:
    """End-to-end tests for the sensitivity analysis pipeline."""

    def test_returns_results_for_single_constant(self, full_roster):
        """Perturbing one constant should produce 2 results (+/-20%)."""
        results = run_sensitivity_analysis(
            full_roster,
            constants_to_test=["platoon_lhb_vs_rhp"],
            mode="quick",
        )
        assert len(results) == 2
        assert all(isinstance(r, SensitivityResult) for r in results)

    def test_result_fields_populated(self, full_roster):
        """Every field in SensitivityResult should be populated."""
        results = run_sensitivity_analysis(
            full_roster,
            constants_to_test=["sigmoid_k_counting"],
            perturbation_pcts=[0.20],
            mode="quick",
        )
        assert len(results) == 1
        r = results[0]
        assert r.constant_name == "sigmoid_k_counting"
        assert r.original_value == pytest.approx(2.0)
        assert r.perturbed_value == pytest.approx(2.4)
        assert r.perturbation_pct == pytest.approx(0.20)
        assert isinstance(r.lineup_change_count, int)
        assert r.lineup_change_count >= 0
        assert isinstance(r.weight_change_max, float)
        assert r.declared_sensitivity in ("HIGH", "MEDIUM", "LOW")
        assert r.actual_sensitivity in ("HIGH", "MEDIUM", "LOW")

    def test_custom_perturbation_pcts(self, full_roster):
        """Custom perturbation list should be honoured."""
        pcts = [-0.10, 0.10, 0.30]
        results = run_sensitivity_analysis(
            full_roster,
            constants_to_test=["recent_form_blend"],
            perturbation_pcts=pcts,
            mode="quick",
        )
        assert len(results) == 3
        got_pcts = [r.perturbation_pct for r in results]
        assert got_pcts == pytest.approx(pcts)

    def test_empty_constants_list(self, full_roster):
        """No matching constants -> empty results."""
        results = run_sensitivity_analysis(
            full_roster,
            constants_to_test=["does_not_exist"],
            mode="quick",
        )
        assert results == []

    def test_multiple_constants(self, full_roster):
        """Two constants x 2 perturbations = 4 results."""
        results = run_sensitivity_analysis(
            full_roster,
            constants_to_test=["platoon_lhb_vs_rhp", "platoon_rhb_vs_lhp"],
            mode="quick",
        )
        assert len(results) == 4
        names = {r.constant_name for r in results}
        assert names == {"platoon_lhb_vs_rhp", "platoon_rhb_vs_lhp"}


class TestSummarizeResults:
    """Verify DataFrame summary output."""

    def test_empty_results(self):
        df = summarize_results([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_columns_present(self):
        """Summary DataFrame must have the expected columns."""
        result = SensitivityResult(
            constant_name="test",
            original_value=1.0,
            perturbed_value=1.2,
            perturbation_pct=0.20,
            lineup_change_count=0,
            weight_change_max=0.0,
            value_change_pct=0.0,
            declared_sensitivity="LOW",
            actual_sensitivity="LOW",
        )
        df = summarize_results([result])
        expected_cols = {
            "constant",
            "original",
            "perturbed",
            "pct",
            "lineup_changes",
            "max_weight_delta",
            "value_change_pct",
            "declared",
            "actual",
            "mismatch",
        }
        assert set(df.columns) == expected_cols

    def test_mismatch_flagged(self):
        """Mismatch column should be True when declared != actual."""
        result = SensitivityResult(
            constant_name="test",
            original_value=1.0,
            perturbed_value=1.2,
            perturbation_pct=0.20,
            lineup_change_count=5,
            weight_change_max=0.5,
            value_change_pct=10.0,
            declared_sensitivity="LOW",
            actual_sensitivity="HIGH",
        )
        df = summarize_results([result])
        assert bool(df.iloc[0]["mismatch"]) is True

    def test_no_mismatch(self):
        """Mismatch column should be False when declared == actual."""
        result = SensitivityResult(
            constant_name="test",
            original_value=1.0,
            perturbed_value=1.2,
            perturbation_pct=0.20,
            lineup_change_count=0,
            weight_change_max=0.0,
            value_change_pct=0.0,
            declared_sensitivity="LOW",
            actual_sensitivity="LOW",
        )
        df = summarize_results([result])
        assert bool(df.iloc[0]["mismatch"]) is False
