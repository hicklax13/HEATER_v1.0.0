"""Tests for advanced LP formulations in src/optimizer/advanced_lp.py.

Covers maximin, epsilon-constraint, and stochastic MIP lineup
optimizations with a small mock roster.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.optimizer.advanced_lp import (
    PULP_AVAILABLE,
    epsilon_constraint_lineup,
    maximin_lineup,
    stochastic_mip,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def roster() -> pd.DataFrame:
    """Small 6-player roster: 4 hitters + 2 pitchers."""
    return pd.DataFrame(
        {
            "name": [
                "Batter A",
                "Batter B",
                "Batter C",
                "Batter D",
                "Pitcher E",
                "Pitcher F",
            ],
            "is_hitter": [True, True, True, True, False, False],
            "r": [80, 70, 60, 50, 0, 0],
            "hr": [30, 25, 20, 15, 0, 0],
            "rbi": [90, 80, 70, 60, 0, 0],
            "sb": [10, 5, 15, 20, 0, 0],
            "avg": [0.290, 0.270, 0.260, 0.250, 0, 0],
            "obp": [0.360, 0.340, 0.330, 0.320, 0, 0],
            "w": [0, 0, 0, 0, 12, 8],
            "l": [0, 0, 0, 0, 8, 5],
            "sv": [0, 0, 0, 0, 0, 20],
            "k": [0, 0, 0, 0, 180, 60],
            "era": [0, 0, 0, 0, 3.20, 2.80],
            "whip": [0, 0, 0, 0, 1.10, 1.05],
        }
    )


@pytest.fixture()
def scale_factors() -> dict[str, float]:
    return {
        "r": 20,
        "hr": 7,
        "rbi": 20,
        "sb": 5,
        "avg": 0.005,
        "obp": 0.005,
        "w": 3,
        "l": 3,
        "sv": 5,
        "k": 25,
        "era": 0.30,
        "whip": 0.03,
    }


@pytest.fixture()
def category_weights() -> dict[str, float]:
    return {cat: 1.0 for cat in ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]}


# ── Maximin tests ────────────────────────────────────────────────────


@pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
class TestMaximinLineup:
    def test_optimal_status(self, roster, scale_factors):
        result = maximin_lineup(roster, scale_factors)
        assert result["status"] == "Optimal"

    def test_z_value_finite(self, roster, scale_factors):
        """z_value should be finite and match the worst category total."""
        result = maximin_lineup(roster, scale_factors)
        assert np.isfinite(result["z_value"])
        # With only hitter-focused cats in active_categories,
        # z should be positive (no inverse stat drag)
        active = ["r", "hr", "rbi", "sb"]
        result2 = maximin_lineup(roster, scale_factors, active_categories=active)
        assert result2["z_value"] > 0

    def test_all_assignments_binary(self, roster, scale_factors):
        result = maximin_lineup(roster, scale_factors)
        for name, val in result["assignments"].items():
            assert val in (0, 1), f"{name} assigned {val}, expected 0 or 1"

    def test_correct_starter_counts(self, roster, scale_factors):
        result = maximin_lineup(roster, scale_factors)
        hitter_names = set(roster.loc[roster["is_hitter"], "name"])
        pitcher_names = set(roster.loc[~roster["is_hitter"], "name"])

        started_hitters = sum(result["assignments"].get(n, 0) for n in hitter_names)
        started_pitchers = sum(result["assignments"].get(n, 0) for n in pitcher_names)
        assert started_hitters == 4  # min(4 hitters, 13 slots)
        assert started_pitchers == 2  # min(2 pitchers, 10 slots)

    def test_worst_category_is_maximized(self, roster, scale_factors):
        """The maximin objective targets the worst category, not the sum."""
        result = maximin_lineup(roster, scale_factors)
        assert result["status"] == "Optimal"

        started = [i for i, name in enumerate(roster["name"]) if result["assignments"].get(name, 0) == 1]

        # Compute each category's scaled total
        cat_totals = {}
        cats = ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]
        for cat in cats:
            vals = pd.to_numeric(roster[cat], errors="coerce").fillna(0).values
            raw = sum(float(vals[i]) for i in started)
            sf = scale_factors[cat]
            if cat in ("era", "whip"):
                cat_totals[cat] = -raw / sf  # inverse
            else:
                cat_totals[cat] = raw / sf

        worst = min(cat_totals.values())
        # z_value should equal (approximately) the worst category
        assert abs(result["z_value"] - worst) < 0.01

    def test_active_categories_subset(self, roster, scale_factors):
        """Excluding categories via active_categories works."""
        active = ["r", "hr", "rbi"]
        result = maximin_lineup(roster, scale_factors, active_categories=active)
        assert result["status"] == "Optimal"
        assert result["z_value"] > 0

    def test_punted_weight_zero_excluded(self, roster, scale_factors):
        """Categories with weight=0 do not constrain z."""
        weights = {cat: 1.0 for cat in scale_factors}
        weights["sb"] = 0.0  # punt SB
        result_punted = maximin_lineup(roster, scale_factors, category_weights=weights)
        result_full = maximin_lineup(roster, scale_factors)
        # Punting SB should raise z (one fewer constraint)
        assert result_punted["z_value"] >= result_full["z_value"] - 0.01

    def test_empty_roster(self, scale_factors):
        empty = pd.DataFrame(columns=["name", "is_hitter", "r", "hr"])
        result = maximin_lineup(empty, scale_factors)
        assert result["status"] == "empty_roster"
        assert result["assignments"] == {}


# ── Epsilon-constraint tests ─────────────────────────────────────────


@pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
class TestEpsilonConstraintLineup:
    def test_optimal_status(self, roster, scale_factors):
        eps = {"r": 2.0, "avg": 30.0}
        result = epsilon_constraint_lineup(roster, scale_factors, "hr", eps)
        assert result["status"] == "Optimal"

    def test_primary_maximized(self, roster, scale_factors):
        """Primary category value should be maximized."""
        eps = {}  # no constraints — pure maximize HR
        result = epsilon_constraint_lineup(roster, scale_factors, "hr", eps)
        assert result["status"] == "Optimal"
        # All 4 hitters should start, maximizing total HR
        started_hitters = sum(
            1 for name in ["Batter A", "Batter B", "Batter C", "Batter D"] if result["assignments"].get(name, 0) == 1
        )
        assert started_hitters == 4

    def test_constraints_satisfied(self, roster, scale_factors):
        """Epsilon bounds should be met in the solution."""
        # Set a feasible lower bound on r (scaled)
        eps = {"r": 2.0}  # sum(r) / 20 >= 2.0 => need 40 runs
        result = epsilon_constraint_lineup(roster, scale_factors, "hr", eps)
        assert result["status"] == "Optimal"
        assert result["constraint_satisfaction"]["r"] >= 2.0 - 0.01

    def test_inverse_constraint_direction(self, roster, scale_factors):
        """ERA/WHIP epsilon bounds use <= (keep below threshold)."""
        # Both pitchers must start (2 pitchers, 2 pitcher slots).
        # sum(era) = 3.20 + 2.80 = 6.0; scaled = 6.0 / 0.30 = 20.0.
        # Set bound above that so it's feasible.
        eps = {"era": 25.0}  # sum(era)/0.30 <= 25.0
        result = epsilon_constraint_lineup(roster, scale_factors, "k", eps)
        assert result["status"] == "Optimal"
        assert result["constraint_satisfaction"]["era"] <= 25.0 + 0.01

    def test_infeasible_bounds(self, roster, scale_factors):
        """Impossible epsilon bounds should yield Infeasible."""
        eps = {"r": 999.0}  # impossible: need 999*20 = 19980 runs
        result = epsilon_constraint_lineup(roster, scale_factors, "hr", eps)
        assert result["status"] == "Infeasible"

    def test_all_assignments_binary(self, roster, scale_factors):
        result = epsilon_constraint_lineup(roster, scale_factors, "hr", {})
        for name, val in result["assignments"].items():
            assert val in (0, 1), f"{name} assigned {val}"

    def test_empty_roster(self, scale_factors):
        empty = pd.DataFrame(columns=["name", "is_hitter", "hr"])
        result = epsilon_constraint_lineup(empty, scale_factors, "hr", {})
        assert result["status"] == "empty_roster"
        assert result["assignments"] == {}


# ── Stochastic MIP tests ────────────────────────────────────────────


@pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
class TestStochasticMIP:
    @pytest.fixture()
    def scenarios(self, roster):
        """Generate small scenario array (50 scenarios, 6 players, 12 cats)."""
        rng = np.random.RandomState(42)
        n_scen, n_players, n_cats = 50, len(roster), 12
        base = np.zeros((n_players, n_cats), dtype=float)
        cats = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
        for j, cat in enumerate(cats):
            base[:, j] = pd.to_numeric(roster[cat], errors="coerce").fillna(0).values
        # Add noise
        out = np.zeros((n_scen, n_players, n_cats), dtype=float)
        for s in range(n_scen):
            noise = rng.normal(0, 0.1, size=(n_players, n_cats))
            out[s] = base * (1 + noise)
        return out

    def test_optimal_status(self, roster, scenarios, category_weights, scale_factors):
        result = stochastic_mip(roster, scenarios, category_weights, scale_factors)
        assert result["status"] == "Optimal"

    def test_objective_components(self, roster, scenarios, category_weights, scale_factors):
        result = stochastic_mip(roster, scenarios, category_weights, scale_factors)
        assert result["status"] == "Optimal"
        assert "mean_component" in result
        assert "cvar_component" in result
        assert "eta_value" in result

    def test_all_assignments_binary(self, roster, scenarios, category_weights, scale_factors):
        result = stochastic_mip(roster, scenarios, category_weights, scale_factors)
        for name, val in result["assignments"].items():
            assert val in (0, 1), f"{name} assigned {val}"

    def test_correct_starter_counts(self, roster, scenarios, category_weights, scale_factors):
        result = stochastic_mip(roster, scenarios, category_weights, scale_factors)
        hitter_names = set(roster.loc[roster["is_hitter"], "name"])
        pitcher_names = set(roster.loc[~roster["is_hitter"], "name"])

        started_h = sum(result["assignments"].get(n, 0) for n in hitter_names)
        started_p = sum(result["assignments"].get(n, 0) for n in pitcher_names)
        assert started_h == 4
        assert started_p == 2

    def test_cvar_vs_mean_only(self, roster, scenarios, category_weights, scale_factors):
        """With cvar_weight=0, result is pure mean optimization."""
        result_mean = stochastic_mip(
            roster,
            scenarios,
            category_weights,
            scale_factors,
            mean_weight=1.0,
            cvar_weight=0.0,
        )
        result_cvar = stochastic_mip(
            roster,
            scenarios,
            category_weights,
            scale_factors,
            mean_weight=0.0,
            cvar_weight=1.0,
        )
        # Both should be optimal but may produce different lineups
        assert result_mean["status"] == "Optimal"
        assert result_cvar["status"] == "Optimal"

    def test_high_variance_player_scenario(self, roster, scale_factors, category_weights):
        """With high-variance scenarios, CVaR weighting may differ from mean."""
        rng = np.random.RandomState(99)
        n_scen, n_players, n_cats = 100, len(roster), 12
        base = np.zeros((n_players, n_cats), dtype=float)
        cats = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
        for j, cat in enumerate(cats):
            base[:, j] = pd.to_numeric(roster[cat], errors="coerce").fillna(0).values

        out = np.zeros((n_scen, n_players, n_cats), dtype=float)
        for s in range(n_scen):
            noise = rng.normal(0, 0.3, size=(n_players, n_cats))
            out[s] = base * (1 + noise)

        result = stochastic_mip(
            roster,
            out,
            category_weights,
            scale_factors,
            mean_weight=0.5,
            cvar_weight=0.5,
        )
        assert result["status"] == "Optimal"
        # Objective should reflect both components
        assert result["objective_value"] != 0.0

    def test_empty_roster(self, category_weights, scale_factors):
        empty = pd.DataFrame(columns=["name", "is_hitter", "r"])
        scenarios = np.zeros((10, 0, 12))
        result = stochastic_mip(empty, scenarios, category_weights, scale_factors)
        assert result["status"] == "empty_roster"
        assert result["assignments"] == {}


# ── PuLP unavailable tests ───────────────────────────────────────────


class TestPuLPUnavailable:
    def test_maximin_no_pulp(self, roster, scale_factors):
        with patch("src.optimizer.advanced_lp.PULP_AVAILABLE", False):
            result = maximin_lineup(roster, scale_factors)
        assert result["status"] == "PuLP not available"

    def test_epsilon_no_pulp(self, roster, scale_factors):
        with patch("src.optimizer.advanced_lp.PULP_AVAILABLE", False):
            result = epsilon_constraint_lineup(roster, scale_factors, "hr", {})
        assert result["status"] == "PuLP not available"

    def test_stochastic_no_pulp(self, roster, scale_factors, category_weights):
        scenarios = np.zeros((10, 6, 12))
        with patch("src.optimizer.advanced_lp.PULP_AVAILABLE", False):
            result = stochastic_mip(roster, scenarios, category_weights, scale_factors)
        assert result["status"] == "PuLP not available"
