"""Structural-invariant guard: backtest + calibration framework.

Per Enhanced Trade Engine report Section G — validates that the engine's
predicted probabilities are well-calibrated. Provides:
  • brier_score() — standard Brier scoring
  • calibration_curve() — decile-binned predicted vs actual
  • stability_check() — MC sim variance across seeds
  • self_consistency_check() — playoff_prob == mean(per-sim outcomes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.output.backtest_calibration import (
    brier_score,
    calibration_curve,
    self_consistency_check,
    stability_check,
)


def _make_hitter(pid: int, name: str, hr: int = 20) -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 1,
        "positions": "OF",
        "status": "active",
        "r": 70,
        "hr": hr,
        "rbi": 75,
        "sb": 8,
        "h": 130,
        "ab": 500,
        "bb": 55,
        "hbp": 5,
        "sf": 5,
        "pa": 565,
        "avg": 0.260,
        "obp": 0.330,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "ytd_ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "era": 0,
        "whip": 0,
    }


def _build_league():
    rows, rosters, current_wins = [], {}, {}
    pid = 1
    for team_idx in range(12):
        name = f"T{team_idx}"
        rosters[name] = []
        current_wins[name] = 4 + team_idx // 3
        for _ in range(3):
            rows.append(_make_hitter(pid, f"P{pid}", hr=15 + team_idx))
            rosters[name].append(pid)
            pid += 1
    pool = pd.DataFrame(rows)
    user_sched = {w: f"T{(w % 11) + 1}" for w in range(9, 25)}
    return pool, rosters, current_wins, user_sched


# ── Brier score ──────────────────────────────────────────────────────


def test_brier_score_perfect_prediction_zero() -> None:
    """Predicting 1.0 for every actual-1 and 0.0 for every actual-0 → 0."""
    preds = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(preds, outcomes) == 0.0


def test_brier_score_naive_50_50_quarter() -> None:
    """Always predicting 0.5 on uniform outcomes → 0.25."""
    preds = np.full(100, 0.5)
    outcomes = np.array([0.0, 1.0] * 50)
    assert abs(brier_score(preds, outcomes) - 0.25) < 1e-9


def test_brier_score_worst_case_one() -> None:
    """Always wrong (predict 1.0 for actual-0) → 1.0."""
    preds = np.array([1.0, 1.0, 1.0])
    outcomes = np.array([0.0, 0.0, 0.0])
    assert brier_score(preds, outcomes) == 1.0


def test_brier_score_empty_returns_nan() -> None:
    """Edge case: empty arrays return NaN (not crash)."""
    assert np.isnan(brier_score(np.array([]), np.array([])))


# ── Calibration curve ────────────────────────────────────────────────


def test_calibration_curve_returns_required_keys() -> None:
    """Output dict must have bin_predicted_mean, bin_actual_freq, bin_count."""
    preds = np.random.RandomState(42).uniform(0, 1, 100)
    outcomes = np.random.RandomState(42).binomial(1, preds, 100)
    curve = calibration_curve(preds, outcomes, n_bins=5)
    for key in ("bin_predicted_mean", "bin_actual_freq", "bin_count", "n_bins"):
        assert key in curve, f"Missing {key} in calibration_curve output"
    assert curve["n_bins"] == 5
    assert len(curve["bin_predicted_mean"]) == 5


def test_calibration_curve_bin_counts_sum_to_total() -> None:
    """Sum of per-bin counts should equal total predictions."""
    n = 100
    preds = np.random.RandomState(0).uniform(0, 1, n)
    outcomes = np.random.RandomState(0).binomial(1, preds, n)
    curve = calibration_curve(preds, outcomes, n_bins=10)
    total = sum(curve["bin_count"])
    assert total == n


# ── Stability check ─────────────────────────────────────────────────


def test_stability_check_returns_expected_structure() -> None:
    """Stability check output has all required keys."""
    pool, rosters, wins, sched = _build_league()
    result = stability_check(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        n_trials=3,
        n_sims_per_trial=500,
    )
    for key in (
        "playoff_prob_estimates",
        "playoff_mean",
        "playoff_std",
        "champ_mean",
        "champ_std",
        "theoretical_playoff_std",
        "empirical_to_theoretical_ratio",
        "stability_quality",
        "n_trials",
        "n_sims_per_trial",
    ):
        assert key in result, f"Missing {key} in stability_check output"
    assert result["n_trials"] == 3
    assert result["n_sims_per_trial"] == 500


def test_stability_check_quality_classification() -> None:
    """Quality must be one of the documented categories."""
    pool, rosters, wins, sched = _build_league()
    result = stability_check(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        n_trials=3,
        n_sims_per_trial=500,
    )
    assert result["stability_quality"] in ("excellent", "good", "marginal", "poor")


def test_stability_check_estimates_within_bounds() -> None:
    """All estimates must be in [0, 1]."""
    pool, rosters, wins, sched = _build_league()
    result = stability_check(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        n_trials=3,
        n_sims_per_trial=500,
    )
    for est in result["playoff_prob_estimates"]:
        assert 0.0 <= est <= 1.0


# ── Self-consistency check ──────────────────────────────────────────


def test_self_consistency_check_returns_consistent_true() -> None:
    """playoff_prob == mean(playoff_outcomes) by construction."""
    pool, rosters, wins, sched = _build_league()
    result = self_consistency_check(
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        n_sims=500,
    )
    assert result["consistent"] is True
    assert abs(result["playoff_prob_reported"] - result["playoff_outcomes_mean"]) < 1e-6
