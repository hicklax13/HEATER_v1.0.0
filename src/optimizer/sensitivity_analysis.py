"""Sensitivity analysis framework for optimizer constants.

Perturbs each constant +/-20% and measures impact on lineup composition,
category weights, and total projected value. Identifies which constants
are truly HIGH sensitivity vs which are effectively inert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd

from src.optimizer.constants_registry import CONSTANTS_REGISTRY, ConstantEntry
from src.optimizer.pipeline import LineupOptimizerPipeline

logger = logging.getLogger(__name__)


# ── Patch target mapping ────────────────────────────────────────────
# Maps constant registry names to their module-level variable paths
# for use with unittest.mock.patch. Only module-level scalars are
# included; nested dict values (e.g. MODE_PRESETS) are not patchable
# this way.

CONSTANT_PATCH_TARGETS: dict[str, str] = {
    "platoon_lhb_vs_rhp": "src.optimizer.matchup_adjustments._LHB_VS_RHP_ADVANTAGE",
    "platoon_rhb_vs_lhp": "src.optimizer.matchup_adjustments._RHB_VS_LHP_ADVANTAGE",
    "platoon_stab_lhb": "src.optimizer.matchup_adjustments._LHB_STABILIZATION_PA",
    "platoon_stab_rhb": "src.optimizer.matchup_adjustments._RHB_STABILIZATION_PA",
    "sigmoid_k_counting": "src.optimizer.category_urgency.COUNTING_STAT_K",
    "sigmoid_k_rate": "src.optimizer.category_urgency.RATE_STAT_K",
    "two_start_fatigue_factor": "src.optimizer.streaming._TWO_START_FATIGUE_FACTOR",
    "default_ip_per_start": "src.optimizer.streaming._DEFAULT_IP_PER_START",
    "default_team_weekly_ip": "src.optimizer.streaming._DEFAULT_TEAM_WEEKLY_IP",
    "recent_form_blend": "src.optimizer.projections._RECENT_FORM_BLEND",
    "min_recent_games": "src.optimizer.projections._MIN_RECENT_GAMES",
}


@dataclass
class SensitivityResult:
    """Result of perturbing one constant."""

    constant_name: str
    original_value: float
    perturbed_value: float
    perturbation_pct: float
    lineup_change_count: int  # How many players changed vs baseline
    weight_change_max: float  # Max absolute change in any category weight
    value_change_pct: float  # % change in total lineup value
    declared_sensitivity: str  # From registry: HIGH/MEDIUM/LOW
    actual_sensitivity: str  # Computed: HIGH/MEDIUM/LOW based on impact


def compute_lineup_diff(
    baseline_assignments: list[dict],
    perturbed_assignments: list[dict],
) -> int:
    """Count how many player assignments differ between two lineups.

    Uses symmetric difference on player_id sets, so one swap counts as 2
    (one player removed, one player added).
    """
    baseline_ids = {a.get("player_id") for a in baseline_assignments}
    perturbed_ids = {a.get("player_id") for a in perturbed_assignments}
    return len(baseline_ids.symmetric_difference(perturbed_ids))


def compute_weight_diff(
    baseline_weights: dict[str, float],
    perturbed_weights: dict[str, float],
) -> float:
    """Max absolute difference in category weights."""
    max_diff = 0.0
    for cat in baseline_weights:
        if cat in perturbed_weights:
            diff = abs(baseline_weights[cat] - perturbed_weights[cat])
            max_diff = max(max_diff, diff)
    return max_diff


def _compute_total_value(
    projected_stats: dict[str, float],
    category_weights: dict[str, float],
) -> float:
    """Weighted sum of projected stats as a proxy for total lineup value."""
    total = 0.0
    for cat, weight in category_weights.items():
        stat_val = projected_stats.get(cat, 0.0)
        if stat_val and weight:
            total += float(stat_val) * float(weight)
    return total


def _classify_sensitivity(
    lineup_change: int,
    value_change_pct: float,
) -> str:
    """Classify actual sensitivity based on observed impact.

    HIGH:   lineup_change >= 3  OR  value_change >= 5%
    MEDIUM: lineup_change >= 1  OR  value_change >= 2%
    LOW:    everything else
    """
    if lineup_change >= 3 or abs(value_change_pct) >= 5.0:
        return "HIGH"
    if lineup_change >= 1 or abs(value_change_pct) >= 2.0:
        return "MEDIUM"
    return "LOW"


def _get_patchable_constants(
    constants_to_test: list[str] | None = None,
) -> list[str]:
    """Return the list of constant names that can be patched.

    If *constants_to_test* is provided, filters to those that exist in
    both CONSTANTS_REGISTRY and CONSTANT_PATCH_TARGETS.  Otherwise
    returns all patchable constants.
    """
    patchable = set(CONSTANTS_REGISTRY) & set(CONSTANT_PATCH_TARGETS)
    if constants_to_test is not None:
        patchable = patchable & set(constants_to_test)
    return sorted(patchable)


def run_sensitivity_analysis(
    roster: pd.DataFrame,
    constants_to_test: list[str] | None = None,
    perturbation_pcts: list[float] | None = None,
    mode: str = "quick",
) -> list[SensitivityResult]:
    """Run sensitivity analysis on specified constants.

    Args:
        roster: Player roster DataFrame.
        constants_to_test: List of constant names from CONSTANTS_REGISTRY.
            Defaults to all patchable constants.
        perturbation_pcts: List of perturbation percentages to test.
            Defaults to [-0.20, +0.20] (plus/minus 20%).
        mode: Pipeline mode to use ("quick" for speed).

    Returns:
        List of SensitivityResult for each constant x perturbation.
    """
    if perturbation_pcts is None:
        perturbation_pcts = [-0.20, 0.20]

    names = _get_patchable_constants(constants_to_test)
    if not names:
        logger.warning("No patchable constants found for sensitivity analysis")
        return []

    # ── Run baseline ────────────────────────────────────────────
    logger.info("Sensitivity analysis: running baseline (mode=%s)", mode)
    baseline_pipe = LineupOptimizerPipeline(roster, mode=mode)
    baseline_result = baseline_pipe.optimize()

    baseline_lineup = baseline_result.get("lineup", {})
    baseline_assignments = baseline_lineup.get("assignments", [])
    baseline_weights = baseline_result.get("category_weights", {})
    baseline_projected = baseline_lineup.get("projected_stats", {})
    baseline_value = _compute_total_value(baseline_projected, baseline_weights)

    results: list[SensitivityResult] = []

    # ── Perturb each constant ───────────────────────────────────
    for name in names:
        entry: ConstantEntry = CONSTANTS_REGISTRY[name]
        patch_target = CONSTANT_PATCH_TARGETS[name]

        for pct in perturbation_pcts:
            perturbed_value = entry.value * (1.0 + pct)

            logger.info(
                "Perturbing %s: %.4f -> %.4f (%+.0f%%)",
                name,
                entry.value,
                perturbed_value,
                pct * 100,
            )

            try:
                with patch(patch_target, perturbed_value):
                    pipe = LineupOptimizerPipeline(roster, mode=mode)
                    perturbed_result = pipe.optimize()
            except Exception:
                logger.warning("Failed to run with perturbed %s", name, exc_info=True)
                continue

            perturbed_lineup = perturbed_result.get("lineup", {})
            perturbed_assignments = perturbed_lineup.get("assignments", [])
            perturbed_weights = perturbed_result.get("category_weights", {})
            perturbed_projected = perturbed_lineup.get("projected_stats", {})
            perturbed_value_total = _compute_total_value(
                perturbed_projected,
                perturbed_weights,
            )

            lineup_diff = compute_lineup_diff(baseline_assignments, perturbed_assignments)
            weight_diff = compute_weight_diff(baseline_weights, perturbed_weights)

            if baseline_value != 0:
                value_change = ((perturbed_value_total - baseline_value) / abs(baseline_value)) * 100.0
            else:
                value_change = 0.0

            actual = _classify_sensitivity(lineup_diff, value_change)

            results.append(
                SensitivityResult(
                    constant_name=name,
                    original_value=entry.value,
                    perturbed_value=perturbed_value,
                    perturbation_pct=pct,
                    lineup_change_count=lineup_diff,
                    weight_change_max=weight_diff,
                    value_change_pct=value_change,
                    declared_sensitivity=entry.sensitivity,
                    actual_sensitivity=actual,
                )
            )

    return results


def summarize_results(results: list[SensitivityResult]) -> pd.DataFrame:
    """Convert sensitivity results to a DataFrame for easy inspection.

    Adds a ``mismatch`` column that flags constants where the declared
    sensitivity does not match the actual observed sensitivity.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append(
            {
                "constant": r.constant_name,
                "original": r.original_value,
                "perturbed": r.perturbed_value,
                "pct": f"{r.perturbation_pct:+.0%}",
                "lineup_changes": r.lineup_change_count,
                "max_weight_delta": round(r.weight_change_max, 4),
                "value_change_pct": round(r.value_change_pct, 2),
                "declared": r.declared_sensitivity,
                "actual": r.actual_sensitivity,
                "mismatch": r.declared_sensitivity != r.actual_sensitivity,
            }
        )

    return pd.DataFrame(rows)
