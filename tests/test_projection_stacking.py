"""Tests for src/projection_stacking.py — ridge regression stacking weights."""

import numpy as np
import pandas as pd
import pytest

from src.projection_stacking import (
    MIN_PLAYERS_FOR_REGRESSION,
    compute_all_stat_weights,
    compute_stacking_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(player_ids: list[int], stat_name: str, values: list[float]) -> pd.DataFrame:
    """Build a minimal DataFrame with player_id + one stat column."""
    return pd.DataFrame({"player_id": player_ids, stat_name: values})


def _make_systems_and_actuals(
    n_players: int = 20,
    n_systems: int = 3,
    seed: int = 42,
    stat: str = "hr",
):
    """Generate synthetic projection systems and actuals for testing."""
    rng = np.random.RandomState(seed)
    pids = list(range(1, n_players + 1))

    # True talent + noise for actuals
    true_talent = rng.uniform(5, 40, size=n_players)
    actuals = _make_df(pids, stat, (true_talent + rng.normal(0, 3, n_players)).tolist())

    systems = {}
    for i in range(n_systems):
        noise_scale = 2.0 + i * 2.0  # system 0 is best, system 2 is worst
        proj = true_talent + rng.normal(0, noise_scale, n_players)
        systems[f"system_{i}"] = _make_df(pids, stat, proj.tolist())

    return systems, actuals


# ---------------------------------------------------------------------------
# Tests: compute_stacking_weights
# ---------------------------------------------------------------------------


class TestComputeStackingWeights:
    """Core stacking weight tests."""

    def test_weights_sum_to_one(self):
        systems, actuals = _make_systems_and_actuals()
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_non_negative(self):
        systems, actuals = _make_systems_and_actuals()
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        for w in weights.values():
            assert w >= 0.0

    def test_better_system_gets_higher_weight(self):
        """System 0 has less noise, so it should receive the highest weight."""
        systems, actuals = _make_systems_and_actuals(n_players=50, seed=99)
        weights = compute_stacking_weights(systems, actuals, stat="hr", alpha=0.5)
        # system_0 should have the largest weight
        assert weights["system_0"] == max(weights.values())

    def test_single_system(self):
        """With one system, it should get weight 1.0."""
        systems, actuals = _make_systems_and_actuals(n_systems=1)
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert len(weights) == 1
        assert abs(list(weights.values())[0] - 1.0) < 1e-9

    def test_fallback_uniform_too_few_players(self):
        """Below MIN_PLAYERS_FOR_REGRESSION, weights should be uniform."""
        n = MIN_PLAYERS_FOR_REGRESSION - 1
        systems, actuals = _make_systems_and_actuals(n_players=n)
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        expected = 1.0 / len(systems)
        for w in weights.values():
            assert abs(w - expected) < 1e-9

    def test_exactly_min_players(self):
        """Exactly MIN_PLAYERS_FOR_REGRESSION should run regression (not fallback)."""
        systems, actuals = _make_systems_and_actuals(n_players=MIN_PLAYERS_FOR_REGRESSION)
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_empty_systems(self):
        actuals = _make_df([1, 2, 3], "hr", [10, 20, 30])
        weights = compute_stacking_weights({}, actuals, stat="hr")
        assert weights == {}

    def test_missing_stat_in_actuals(self):
        """If actuals doesn't have the stat column, return uniform."""
        systems, _ = _make_systems_and_actuals()
        bad_actuals = pd.DataFrame({"player_id": [1, 2, 3], "rbi": [10, 20, 30]})
        weights = compute_stacking_weights(systems, bad_actuals, stat="hr")
        expected = 1.0 / len(systems)
        for w in weights.values():
            assert abs(w - expected) < 1e-9

    def test_missing_stat_in_one_system(self):
        """A system missing the stat column should be skipped but not crash."""
        systems, actuals = _make_systems_and_actuals(n_players=20, n_systems=3)
        # Remove the stat column from one system
        systems["system_1"] = systems["system_1"].drop(columns=["hr"])
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert "system_1" not in weights
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_zero_variance_target(self):
        """All-constant actuals should return uniform weights."""
        pids = list(range(1, 21))
        actuals = _make_df(pids, "hr", [15.0] * 20)
        rng = np.random.RandomState(0)
        systems = {
            "a": _make_df(pids, "hr", rng.uniform(10, 20, 20).tolist()),
            "b": _make_df(pids, "hr", rng.uniform(10, 20, 20).tolist()),
        }
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        for w in weights.values():
            assert abs(w - 0.5) < 1e-9

    def test_zero_variance_system(self):
        """A system with constant predictions should be dropped."""
        pids = list(range(1, 21))
        rng = np.random.RandomState(7)
        actuals = _make_df(pids, "hr", rng.uniform(5, 40, 20).tolist())
        systems = {
            "good": _make_df(pids, "hr", (actuals["hr"] + rng.normal(0, 2, 20)).tolist()),
            "constant": _make_df(pids, "hr", [20.0] * 20),
        }
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        # constant system should be excluded
        assert "constant" not in weights
        assert abs(weights["good"] - 1.0) < 1e-9

    def test_partial_player_overlap(self):
        """Only common player_ids should be used."""
        pids_a = list(range(1, 25))  # 24 players
        pids_b = list(range(10, 34))  # 24 players, overlap = 15
        rng = np.random.RandomState(3)

        common = sorted(set(pids_a) & set(pids_b))
        assert len(common) >= MIN_PLAYERS_FOR_REGRESSION

        actuals = _make_df(pids_a + list(range(34, 40)), "hr", rng.uniform(5, 30, 30).tolist())
        systems = {
            "a": _make_df(pids_a, "hr", rng.uniform(5, 30, len(pids_a)).tolist()),
            "b": _make_df(pids_b, "hr", rng.uniform(5, 30, len(pids_b)).tolist()),
        }
        weights = compute_stacking_weights(systems, actuals, stat="hr")
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_works_with_rate_stat(self):
        """Ridge should work fine with fractional rate stats like avg."""
        pids = list(range(1, 30))
        rng = np.random.RandomState(12)
        actuals = _make_df(pids, "avg", rng.uniform(0.200, 0.330, len(pids)).tolist())
        systems = {
            "s1": _make_df(pids, "avg", (actuals["avg"] + rng.normal(0, 0.01, len(pids))).tolist()),
            "s2": _make_df(pids, "avg", (actuals["avg"] + rng.normal(0, 0.03, len(pids))).tolist()),
        }
        weights = compute_stacking_weights(systems, actuals, stat="avg")
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # s1 (less noise) should get higher weight
        assert weights["s1"] > weights["s2"]

    def test_high_alpha_approaches_uniform(self):
        """Very large alpha should push weights toward uniform."""
        systems, actuals = _make_systems_and_actuals(n_players=30)
        weights = compute_stacking_weights(systems, actuals, stat="hr", alpha=1e6)
        expected = 1.0 / len(weights)
        for w in weights.values():
            assert abs(w - expected) < 0.05  # within 5% of uniform


# ---------------------------------------------------------------------------
# Tests: compute_all_stat_weights
# ---------------------------------------------------------------------------


class TestComputeAllStatWeights:
    """Tests for the batch weight computation."""

    def test_returns_all_requested_stats(self):
        systems, actuals = _make_systems_and_actuals()
        # Add a second stat to all DataFrames
        for name in systems:
            systems[name]["rbi"] = systems[name]["hr"] * 2
        actuals["rbi"] = actuals["hr"] * 2

        result = compute_all_stat_weights(systems, actuals, stats=["hr", "rbi"])
        assert "hr" in result
        assert "rbi" in result
        for stat_weights in result.values():
            assert abs(sum(stat_weights.values()) - 1.0) < 1e-9

    def test_default_stats_list(self):
        """With no stats arg, should use the default 12 fantasy categories."""
        pids = list(range(1, 21))
        rng = np.random.RandomState(5)
        # Only provide hr — other stats will trigger uniform fallback
        actuals = pd.DataFrame(
            {
                "player_id": pids,
                "hr": rng.uniform(5, 35, 20),
            }
        )
        systems = {
            "a": actuals.copy(),
            "b": actuals.copy(),
        }
        result = compute_all_stat_weights(systems, actuals)
        assert len(result) == 12  # default list has 12 stats

    def test_custom_alpha(self):
        systems, actuals = _make_systems_and_actuals()
        w1 = compute_all_stat_weights(systems, actuals, stats=["hr"], alpha=0.1)
        w2 = compute_all_stat_weights(systems, actuals, stats=["hr"], alpha=100.0)
        # High alpha should yield more uniform weights
        spread_low = max(w1["hr"].values()) - min(w1["hr"].values())
        spread_high = max(w2["hr"].values()) - min(w2["hr"].values())
        assert spread_high < spread_low
