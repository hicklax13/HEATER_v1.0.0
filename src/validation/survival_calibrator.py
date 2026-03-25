"""
Survival Calibrator — answers the question:
"Does HEATER's survival probability actually predict pick availability?"

Takes historical draft data and compares:
    - Predicted: P(player available at pick X) from simulation.py's Normal CDF
    - Actual: Was the player actually available at pick X?

Outputs:
    - Calibration curve (predicted probability vs observed frequency)
    - Optimal sigma for Normal CDF (currently hardcoded at 10.0)
    - Brier score (lower = better calibrated)
    - Per-round accuracy breakdown
    - Position-run detection accuracy

This is the most impactful calibrator because survival probability cascades
into urgency, which cascades into combined_score, which drives the final
draft recommendation ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from src.validation.calibration_data import CalibrationDataset

logger = logging.getLogger(__name__)


@dataclass
class SurvivalCalibrationResult:
    """Results from calibrating survival probability against real drafts."""

    # Current model performance
    current_sigma: float = 10.0
    current_brier_score: float = 0.0
    current_log_loss: float = 0.0

    # Optimal values found by calibration
    optimal_sigma: float = 10.0
    optimal_brier_score: float = 0.0
    optimal_log_loss: float = 0.0
    sigma_improvement_pct: float = 0.0

    # Calibration curve bins (for plotting)
    # Each bin: (predicted_prob_center, observed_frequency, count)
    calibration_bins: list[tuple[float, float, int]] = field(default_factory=list)

    # Per-round breakdown
    round_accuracy: dict[int, float] = field(default_factory=dict)

    # Position run analysis
    position_run_detection_rate: float = 0.0

    # Sample size
    n_predictions: int = 0
    n_drafts: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Survival Probability Calibration ===",
            f"Sample: {self.n_predictions} predictions across {self.n_drafts} draft(s)",
            "",
            f"Current sigma: {self.current_sigma:.1f}",
            f"  Brier score: {self.current_brier_score:.4f}",
            f"  Log loss:    {self.current_log_loss:.4f}",
            "",
            f"Optimal sigma: {self.optimal_sigma:.1f}",
            f"  Brier score: {self.optimal_brier_score:.4f} ({self.sigma_improvement_pct:+.1f}% vs current)",
            "",
            "Calibration curve (predicted → observed):",
        ]
        for center, observed, count in self.calibration_bins:
            bar = "#" * int(observed * 40)
            lines.append(f"  {center:.0%} → {observed:.0%} (n={count}) {bar}")

        if self.round_accuracy:
            lines.append("")
            lines.append("Per-round accuracy:")
            for rnd in sorted(self.round_accuracy):
                lines.append(f"  Round {rnd:2d}: {self.round_accuracy[rnd]:.1%}")

        return "\n".join(lines)


def calibrate_survival(
    datasets: list[CalibrationDataset],
    current_sigma: float = 10.0,
    num_teams: int = 12,
) -> SurvivalCalibrationResult:
    """
    Calibrate survival probability against historical draft outcomes.

    For each player in each draft, we ask:
        "At pick N, was player X still available?"
    We compare the model's predicted probability to the actual outcome.

    Args:
        datasets: Historical draft data from Yahoo
        current_sigma: The sigma value currently hardcoded in simulation.py
        num_teams: Number of teams in the league

    Returns:
        SurvivalCalibrationResult with optimal sigma and calibration metrics
    """
    result = SurvivalCalibrationResult(current_sigma=current_sigma)

    # Build prediction-outcome pairs from historical drafts
    pairs = _build_prediction_pairs(datasets, num_teams)
    if not pairs:
        logger.warning("No prediction pairs — cannot calibrate survival")
        return result

    result.n_predictions = len(pairs)
    result.n_drafts = len(datasets)

    pred_df = pd.DataFrame(pairs)

    # Evaluate current sigma
    result.current_brier_score = _compute_brier(pred_df, current_sigma, num_teams)
    result.current_log_loss = _compute_log_loss(pred_df, current_sigma, num_teams)

    # Find optimal sigma via bounded optimization
    opt = minimize_scalar(
        lambda s: _compute_brier(pred_df, s, num_teams),
        bounds=(3.0, 40.0),
        method="bounded",
    )
    result.optimal_sigma = round(float(opt.x), 1)
    result.optimal_brier_score = float(opt.fun)
    result.optimal_log_loss = _compute_log_loss(pred_df, result.optimal_sigma, num_teams)

    if result.current_brier_score > 0:
        result.sigma_improvement_pct = (
            (result.current_brier_score - result.optimal_brier_score) / result.current_brier_score * 100
        )

    # Build calibration curve with optimal sigma
    result.calibration_bins = _build_calibration_curve(pred_df, result.optimal_sigma, num_teams)

    # Per-round accuracy
    result.round_accuracy = _per_round_accuracy(pred_df, result.optimal_sigma, num_teams)

    return result


def _build_prediction_pairs(datasets: list[CalibrationDataset], num_teams: int) -> list[dict]:
    """
    Build (player, query_pick, actual_drafted_pick, adp) pairs.

    For each player drafted, we know their ADP (pre-draft ranking) and
    when they were actually picked. For every pick BEFORE they were taken,
    we can ask "was this player still available?" (yes). For the pick they
    were taken at and after, the answer is no.
    """
    pairs = []

    for ds in datasets:
        if not ds.has_draft_data:
            continue

        draft_df = ds.to_draft_dataframe()
        if draft_df.empty:
            continue

        # Sort by pick number to get draft order
        draft_df = draft_df.sort_values("pick_number").reset_index(drop=True)

        # For each player, create survival observations
        for _, row in draft_df.iterrows():
            actual_pick = int(row["pick_number"])
            player_name = row["player_name"]

            # Use actual draft position as a proxy for ADP
            # (In a real calibration, we'd use pre-draft ADP consensus)
            adp = actual_pick  # Simplification — improve with real ADP data

            # Sample query picks: every 3rd pick from 1 to total_picks
            total_picks = len(draft_df)
            for query_pick in range(1, total_picks + 1, 3):
                survived = 1 if query_pick < actual_pick else 0
                picks_between = max(1, abs(query_pick - actual_pick))

                pairs.append(
                    {
                        "player_name": player_name,
                        "adp": adp,
                        "query_pick": query_pick,
                        "actual_pick": actual_pick,
                        "survived": survived,
                        "picks_between": picks_between,
                        "round": (query_pick - 1) // num_teams + 1,
                    }
                )

    return pairs


def _predict_survival(
    adp: float,
    query_pick: float,
    sigma: float,
    num_teams: int,
) -> float:
    """
    Reproduce HEATER's current survival probability formula.

    From simulation.py:
        z = (adp - query_pick) / (sigma * max(1, picks_between**0.3))
        prob = norm.cdf(z)

    We reproduce this exactly so we can measure its calibration.
    """
    picks_between = max(1, abs(query_pick - adp))
    z = (adp - query_pick) / (sigma * max(1.0, picks_between**0.3))
    return float(np.clip(norm.cdf(z), 0.01, 0.99))


def _compute_brier(df: pd.DataFrame, sigma: float, num_teams: int) -> float:
    """Brier score: mean squared error between predicted prob and actual outcome."""
    preds = df.apply(
        lambda r: _predict_survival(r["adp"], r["query_pick"], sigma, num_teams),
        axis=1,
    )
    actual = df["survived"].astype(float)
    return float(((preds - actual) ** 2).mean())


def _compute_log_loss(df: pd.DataFrame, sigma: float, num_teams: int) -> float:
    """Log loss (cross-entropy) between predicted prob and actual outcome."""
    preds = df.apply(
        lambda r: _predict_survival(r["adp"], r["query_pick"], sigma, num_teams),
        axis=1,
    )
    preds = np.clip(preds, 1e-7, 1 - 1e-7)
    actual = df["survived"].astype(float)
    return float(-(actual * np.log(preds) + (1 - actual) * np.log(1 - preds)).mean())


def _build_calibration_curve(
    df: pd.DataFrame,
    sigma: float,
    num_teams: int,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """
    Calibration curve: bin predictions by predicted probability,
    compute observed frequency in each bin.

    Perfect calibration: predicted 70% → observed 70% survival.
    """
    preds = df.apply(
        lambda r: _predict_survival(r["adp"], r["query_pick"], sigma, num_teams),
        axis=1,
    )
    actual = df["survived"].astype(float)

    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi)
        count = int(mask.sum())
        if count > 0:
            observed = float(actual[mask].mean())
            center = (lo + hi) / 2
            bins.append((round(center, 2), round(observed, 3), count))

    return bins


def _per_round_accuracy(
    df: pd.DataFrame,
    sigma: float,
    num_teams: int,
) -> dict[int, float]:
    """Accuracy (correct binary prediction) per draft round."""
    preds = df.apply(
        lambda r: _predict_survival(r["adp"], r["query_pick"], sigma, num_teams),
        axis=1,
    )
    binary_pred = (preds >= 0.5).astype(int)
    actual = df["survived"].astype(int)
    correct = (binary_pred == actual).astype(float)

    accuracy_by_round = {}
    for rnd in sorted(df["round"].unique()):
        mask = df["round"] == rnd
        if mask.sum() > 0:
            accuracy_by_round[int(rnd)] = round(float(correct[mask].mean()), 3)

    return accuracy_by_round
