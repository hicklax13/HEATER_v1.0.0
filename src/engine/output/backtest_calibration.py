"""Backtest + calibration framework for the playoff probability engine.

Per Enhanced Trade Engine report Section G — validates that the engine's
predicted probabilities are well-calibrated (predicted P matches actual
frequency over many trials).

This module provides two layers:

1. SELF-CALIBRATION (always available, no external data needed):
   - Stability check: run the playoff sim K times with different seeds,
     measure variance of the playoff_prob estimate. Verify MC noise
     is bounded and shrinks as ~1/√N.
   - Self-consistency: predicted P(playoffs) for user team equals the
     fraction of inner-sim trials where user actually made top-4.

2. BRIER SCORE FRAMEWORK (skeleton for historical data):
   - Given (predicted_prob, actual_outcome) pairs from historical
     rollouts, compute Brier score: mean((p - y)²) for y ∈ {0, 1}.
   - Calibration curve: bin predictions by decile, plot actual frequency.
   - Skeleton in place; external data ingestion is future work.

The Brier score interpretation:
  0.0  = perfect calibration (predicted p == actual y every time)
  0.25 = trivial baseline (predicting 0.5 always)
  > 0.25 = worse than naive 50/50 guess

For 12-team H2H Cats league, well-calibrated playoff predictions
should yield Brier scores in 0.15-0.20 range mid-season.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.engine.output.playoff_sim import simulate_playoff_outcomes
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score: mean((p - y)²) for y ∈ {0, 1}, p ∈ [0, 1].

    Args:
        predictions: Array of predicted probabilities.
        outcomes: Array of binary outcomes (0/1).

    Returns:
        Brier score in [0, 1]. Lower = better calibration.
        0.0 = perfect, 0.25 = always-50/50 baseline.
    """
    if len(predictions) == 0 or len(predictions) != len(outcomes):
        return float("nan")
    diff = np.asarray(predictions, dtype=float) - np.asarray(outcomes, dtype=float)
    return float(np.mean(diff * diff))


def calibration_curve(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calibration curve via decile binning.

    For each bin (prediction quantile), compute the mean predicted
    probability and the actual frequency. Perfectly calibrated → these
    are equal at every bin (45° line).

    Args:
        predictions: Predicted probabilities.
        outcomes: Binary outcomes.
        n_bins: Number of quantile bins (default 10 = deciles).

    Returns:
        Dict with bin_predicted_mean, bin_actual_freq, bin_count.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(predictions) == 0:
        return {"bin_predicted_mean": [], "bin_actual_freq": [], "bin_count": []}

    # Bin by predicted-probability quantile
    quantile_edges = np.quantile(predictions, np.linspace(0.0, 1.0, n_bins + 1))
    bin_indices = np.digitize(predictions, quantile_edges[1:-1])

    pred_means: list[float] = []
    actual_freqs: list[float] = []
    counts: list[int] = []
    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        if n == 0:
            pred_means.append(float("nan"))
            actual_freqs.append(float("nan"))
            counts.append(0)
            continue
        pred_means.append(float(predictions[mask].mean()))
        actual_freqs.append(float(outcomes[mask].mean()))
        counts.append(n)

    return {
        "bin_predicted_mean": pred_means,
        "bin_actual_freq": actual_freqs,
        "bin_count": counts,
        "n_bins": n_bins,
    }


# ─────────────────────────────────────────────────────────────────────
# Section G error metrics + historical-data ingestion
# ─────────────────────────────────────────────────────────────────────

# Report Section G targets. A backtest is "passing" on a metric when the
# value sits on the better side of the target.
_SECTION_G_TARGETS: dict[str, dict[str, Any]] = {
    "rank_mae": {"target": 1.5, "direction": "lower"},
    "cat_win_rmse": {"target": 8.0, "direction": "lower"},
    "brier": {"target": 0.20, "direction": "lower"},
    "trade_spearman": {"target": 0.40, "direction": "higher"},
    "hr_mae": {"target": 5.0, "direction": "lower"},
    "avg_mae": {"target": 0.020, "direction": "lower"},
}


def rank_mae(predicted_ranks: np.ndarray, actual_ranks: np.ndarray) -> float:
    """Mean absolute error on final standings rank (report Section G #1).

    Target < 1.5 ranks.
    """
    p = np.asarray(predicted_ranks, dtype=float)
    a = np.asarray(actual_ranks, dtype=float)
    if len(p) == 0 or len(p) != len(a):
        return float("nan")
    return float(np.mean(np.abs(p - a)))


def cat_win_rmse(predicted_cat_wins: np.ndarray, actual_cat_wins: np.ndarray) -> float:
    """RMSE on category wins out of 312 possible (report Section G #2).

    Target < 8 cat-wins.
    """
    p = np.asarray(predicted_cat_wins, dtype=float)
    a = np.asarray(actual_cat_wins, dtype=float)
    if len(p) == 0 or len(p) != len(a):
        return float("nan")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def trade_prediction_spearman(
    predicted_delta: np.ndarray,
    realized_rank_change: np.ndarray,
) -> float:
    """Spearman ρ between predicted ΔΠ and realized rank change (report G #4).

    Target ρ > 0.4 (high-noise domain). Returns NaN when there is too little
    data or no variance to rank.
    """
    from scipy.stats import spearmanr

    p = np.asarray(predicted_delta, dtype=float)
    r = np.asarray(realized_rank_change, dtype=float)
    if len(p) < 3 or len(p) != len(r):
        return float("nan")
    if np.ptp(p) == 0 or np.ptp(r) == 0:
        return float("nan")
    rho, _ = spearmanr(p, r)
    return float(rho)


def projection_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Out-of-sample projection MAE for a single stat (report Section G #5).

    Targets: HR MAE < 5, AVG MAE < 0.020 (ATC historical level).
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if len(p) == 0 or len(p) != len(a):
        return float("nan")
    return float(np.mean(np.abs(p - a)))


def run_historical_backtest(records: pd.DataFrame) -> dict[str, Any]:
    """Score engine predictions against realized historical outcomes (Section G).

    This is the ingestion + scoring path the report calls for. It is
    data-agnostic: callers supply a DataFrame of historical observations and
    the function computes whichever Section G metrics the available columns
    support, then grades each against the report's targets.

    No historical H2H-Cats data ships with the repo (it requires
    week-by-week prior-season Yahoo/NFBC logs that aren't scraped), so this
    function operates on whatever the caller provides — making it testable
    today and immediately usable once historical data lands.

    Recognised optional columns:
      - predicted_rank, actual_rank            → rank_mae
      - predicted_cat_wins, actual_cat_wins    → cat_win_rmse
      - predicted_playoff_prob, made_playoffs  → brier + calibration curve
      - predicted_trade_delta, realized_rank_change → trade_spearman
      - predicted_hr, actual_hr                → hr_mae
      - predicted_avg, actual_avg              → avg_mae

    Returns a dict with ``metrics`` (computed values), ``targets`` (the
    Section G goals + pass/fail per metric), and ``n_records``.
    """
    metrics: dict[str, float] = {}
    cols = set(records.columns)

    if {"predicted_rank", "actual_rank"} <= cols:
        metrics["rank_mae"] = rank_mae(records["predicted_rank"], records["actual_rank"])
    if {"predicted_cat_wins", "actual_cat_wins"} <= cols:
        metrics["cat_win_rmse"] = cat_win_rmse(records["predicted_cat_wins"], records["actual_cat_wins"])
    if {"predicted_playoff_prob", "made_playoffs"} <= cols:
        metrics["brier"] = brier_score(records["predicted_playoff_prob"], records["made_playoffs"])
    if {"predicted_trade_delta", "realized_rank_change"} <= cols:
        metrics["trade_spearman"] = trade_prediction_spearman(
            records["predicted_trade_delta"], records["realized_rank_change"]
        )
    if {"predicted_hr", "actual_hr"} <= cols:
        metrics["hr_mae"] = projection_mae(records["predicted_hr"], records["actual_hr"])
    if {"predicted_avg", "actual_avg"} <= cols:
        metrics["avg_mae"] = projection_mae(records["predicted_avg"], records["actual_avg"])

    # Grade each computed metric against the Section G target.
    target_report: dict[str, dict[str, Any]] = {}
    for name, value in metrics.items():
        spec = _SECTION_G_TARGETS.get(name)
        if spec is None or value != value:  # NaN check
            target_report[name] = {"value": value, "target": spec, "passing": None}
            continue
        if spec["direction"] == "lower":
            passing = value <= spec["target"]
        else:
            passing = value >= spec["target"]
        target_report[name] = {
            "value": round(float(value), 4),
            "target": spec["target"],
            "direction": spec["direction"],
            "passing": bool(passing),
        }

    return {
        "metrics": {k: (round(float(v), 4) if v == v else v) for k, v in metrics.items()},
        "targets": target_report,
        "n_records": int(len(records)),
    }


def stability_check(
    user_roster_ids: list[int],
    user_team_name: str,
    all_team_rosters: dict[str, list[int]],
    user_schedule: dict[int, str],
    current_wins: dict[str, int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    n_trials: int = 10,
    n_sims_per_trial: int = 5_000,
    seed_offset: int = 1000,
    full_league_schedule: dict[int, list[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Stability check: run simulate_playoff_outcomes K times with different seeds.

    A well-calibrated MC sim should produce playoff_prob estimates that
    cluster tightly around the true (infinite-sim) value, with variance
    shrinking as 1/√N. This check verifies:
      • Std-dev of playoff_prob estimates is bounded
      • Std-dev × √N_per_trial scales correctly (variance ~1/N)

    Args:
        user_roster_ids, user_team_name, ..., variance_cv: Same as simulate_playoff_outcomes.
        n_trials: Number of independent MC trials (each with a different seed).
        n_sims_per_trial: Sims per trial.
        seed_offset: Base seed for trials (each trial uses seed_offset + trial_idx).
        full_league_schedule: Optional full schedule for non-Binomial sim.

    Returns:
        Dict with:
          - playoff_prob_estimates: per-trial estimates
          - champ_prob_estimates: per-trial estimates
          - playoff_mean, playoff_std: aggregate
          - champ_mean, champ_std: aggregate
          - expected_std_at_n: theoretical sqrt(p(1-p)/n) for playoff at n_sims_per_trial
          - stability_quality: 'excellent' | 'good' | 'marginal' | 'poor'
    """
    playoff_estimates: list[float] = []
    champ_estimates: list[float] = []
    for trial in range(n_trials):
        result = simulate_playoff_outcomes(
            user_roster_ids=user_roster_ids,
            user_team_name=user_team_name,
            all_team_rosters=all_team_rosters,
            user_schedule=user_schedule,
            current_wins=current_wins,
            player_pool=player_pool,
            config=config,
            n_sims=n_sims_per_trial,
            seed=seed_offset + trial,
            full_league_schedule=full_league_schedule,
        )
        playoff_estimates.append(result["playoff_prob"])
        champ_estimates.append(result["champ_prob"])

    p_arr = np.array(playoff_estimates)
    c_arr = np.array(champ_estimates)

    playoff_mean = float(p_arr.mean())
    playoff_std = float(p_arr.std(ddof=1)) if n_trials > 1 else 0.0
    champ_mean = float(c_arr.mean())
    champ_std = float(c_arr.std(ddof=1)) if n_trials > 1 else 0.0

    # Theoretical std for Bernoulli sampling at this n: sqrt(p(1-p)/n)
    theoretical_playoff_std = (
        float(np.sqrt(playoff_mean * (1.0 - playoff_mean) / n_sims_per_trial)) if 0 < playoff_mean < 1 else 0.0
    )

    # Stability quality: how close is empirical std to theoretical?
    if theoretical_playoff_std < 1e-9:
        ratio = 0.0
    else:
        ratio = playoff_std / theoretical_playoff_std

    if ratio < 1.5:
        quality = "excellent"  # within 50% of theoretical noise floor
    elif ratio < 2.5:
        quality = "good"
    elif ratio < 5.0:
        quality = "marginal"
    else:
        quality = "poor"

    return {
        "playoff_prob_estimates": playoff_estimates,
        "champ_prob_estimates": champ_estimates,
        "playoff_mean": round(playoff_mean, 4),
        "playoff_std": round(playoff_std, 4),
        "champ_mean": round(champ_mean, 4),
        "champ_std": round(champ_std, 4),
        "theoretical_playoff_std": round(theoretical_playoff_std, 4),
        "empirical_to_theoretical_ratio": round(ratio, 2),
        "stability_quality": quality,
        "n_trials": n_trials,
        "n_sims_per_trial": n_sims_per_trial,
    }


def self_consistency_check(
    user_team_name: str,
    all_team_rosters: dict[str, list[int]],
    user_schedule: dict[int, str],
    current_wins: dict[str, int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    n_sims: int = 20_000,
    seed: int = 42,
    full_league_schedule: dict[int, list[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Self-consistency check: predicted P(playoffs) equals fraction of
    inner-sim trials where user actually made top-4.

    This validates that the aggregation logic correctly converts per-sim
    outcomes to probability. It's tautological by construction in our
    implementation (playoff_prob = user_in_top4.mean()), but provides a
    regression guard if that contract ever breaks.

    Returns:
        Dict with consistent (bool) + the matched values.
    """
    user_roster_ids = all_team_rosters.get(user_team_name, [])
    result = simulate_playoff_outcomes(
        user_roster_ids=user_roster_ids,
        user_team_name=user_team_name,
        all_team_rosters=all_team_rosters,
        user_schedule=user_schedule,
        current_wins=current_wins,
        player_pool=player_pool,
        config=config,
        n_sims=n_sims,
        seed=seed,
        full_league_schedule=full_league_schedule,
    )
    # Self-consistency: playoff_prob == mean(playoff_outcomes)
    playoff_outcomes_mean = float(np.asarray(result["playoff_outcomes"]).mean())
    consistent = abs(result["playoff_prob"] - playoff_outcomes_mean) < 1e-6
    return {
        "consistent": consistent,
        "playoff_prob_reported": result["playoff_prob"],
        "playoff_outcomes_mean": round(playoff_outcomes_mean, 6),
        "champ_prob_reported": result["champ_prob"],
        "champ_outcomes_mean": round(float(np.asarray(result["champ_outcomes"]).mean()), 6),
    }
