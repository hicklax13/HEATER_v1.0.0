"""Bayesian Model Averaging for projection blending.

Spec reference: Section 6 L3A (Bayesian Model Averaging)

Instead of simple weighted averages of projection systems, this module
computes posterior weights for each system based on how well it predicted
the player's actual YTD performance. Systems that predicted more accurately
get higher weight.

The key insight: if a player is hitting .320 and Steamer projected .310
while ZiPS projected .280, Steamer's posterior weight goes UP because its
prediction was closer to reality.

Wires into existing:
  - src/valuation.py: LeagueConfig
  - src/database.py: load_player_pool (for projection data)
  - src/engine/portfolio/valuation.py: compute_player_zscores
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Forecast standard deviations by projection system and stat category.
# These represent how much each system's projections typically deviate
# from actual results. Tighter sigma = more trustworthy system.
# Spec ref: Section 6 L3A SYSTEM_FORECAST_SIGMA
SYSTEM_FORECAST_SIGMA: dict[str, dict[str, float]] = {
    "steamer": {
        "hr": 5.2,
        "rbi": 14.1,
        "r": 12.8,
        "sb": 4.5,
        "avg": 0.022,
        "w": 2.8,
        "k": 22.5,
        "sv": 6.1,
        "era": 0.65,
        "whip": 0.09,
    },
    "zips": {
        "hr": 5.5,
        "rbi": 14.8,
        "r": 13.2,
        "sb": 4.8,
        "avg": 0.024,
        "w": 3.0,
        "k": 23.0,
        "sv": 6.5,
        "era": 0.70,
        "whip": 0.10,
    },
    "depthcharts": {
        "hr": 5.0,
        "rbi": 13.5,
        "r": 12.5,
        "sb": 4.3,
        "avg": 0.021,
        "w": 2.7,
        "k": 21.0,
        "sv": 5.8,
        "era": 0.62,
        "whip": 0.085,
    },
}

# Default sigma for unknown systems/stats
DEFAULT_SIGMA: float = 10.0

STAT_KEYS: list[str] = ["hr", "rbi", "r", "sb", "avg", "w", "k", "sv", "era", "whip"]


def bayesian_model_average(
    ytd_stats: dict[str, float],
    projections: dict[str, dict[str, float]],
    prior_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute posterior-weighted blended projections via BMA.

    Spec ref: Section 6 L3A — Bayesian Model Averaging.

    For each projection system, compute the likelihood of observing the
    player's actual YTD stats given that system's projections. Systems
    whose projections are closer to reality get higher posterior weight.

    P(Model_i | YTD) = P(YTD | Model_i) * P(Model_i) / Z

    Args:
        ytd_stats: Player's year-to-date stats (e.g. {"hr": 15, "avg": 0.285}).
        projections: Dict mapping system_name -> projected stats dict.
            e.g. {"steamer": {"hr": 20, "avg": 0.270}, "zips": {"hr": 18, ...}}
        prior_weights: Prior belief about each system's accuracy.
            Defaults to uniform (equal weight for all systems).

    Returns:
        Tuple of (posterior_weights, blended_projection, blended_variance):
          - posterior_weights: {system: weight} summing to 1.0
          - blended_projection: {stat: weighted_mean_value}
          - blended_variance: {stat: total_variance} (within + between)
    """
    systems = list(projections.keys())
    if not systems:
        return {}, dict(ytd_stats), {s: 0.0 for s in ytd_stats}

    if len(systems) == 1:
        sys_name = systems[0]
        return (
            {sys_name: 1.0},
            dict(projections[sys_name]),
            {s: _get_sigma(sys_name, s) ** 2 for s in projections[sys_name]},
        )

    if prior_weights is None:
        prior_weights = {s: 1.0 / len(systems) for s in systems}

    # Compute log-likelihoods for each system
    log_liks: dict[str, float] = {}
    for sys_name in systems:
        sys_proj = projections[sys_name]
        ll = 0.0
        for stat in ytd_stats:
            if stat in sys_proj:
                sigma = _get_sigma(sys_name, stat)
                ll += norm.logpdf(
                    ytd_stats[stat],
                    loc=sys_proj[stat],
                    scale=sigma,
                )
        log_liks[sys_name] = ll

    # Numerically stable softmax for posterior
    max_ll = max(log_liks.values())
    unnormalized = {s: np.exp(log_liks[s] - max_ll) * prior_weights.get(s, 1.0 / len(systems)) for s in systems}
    z_total = sum(unnormalized.values())
    if z_total < 1e-300:
        # All likelihoods are essentially zero — fall back to uniform
        posterior = {s: 1.0 / len(systems) for s in systems}
    else:
        posterior = {s: unnormalized[s] / z_total for s in systems}

    # Compute blended projection and variance
    # Variance = within-model variance + between-model variance
    all_stats = set()
    for sys_proj in projections.values():
        all_stats.update(sys_proj.keys())

    blended: dict[str, float] = {}
    blended_var: dict[str, float] = {}

    for stat in all_stats:
        # Weighted mean
        blended[stat] = sum(projections[s].get(stat, 0.0) * posterior[s] for s in systems)

        # Within-model variance: E[Var(Y|Model)]
        v_within = sum(posterior[s] * _get_sigma(s, stat) ** 2 for s in systems)

        # Between-model variance: Var(E[Y|Model])
        v_between = sum(posterior[s] * (projections[s].get(stat, 0.0) - blended[stat]) ** 2 for s in systems)

        blended_var[stat] = v_within + v_between

    return posterior, blended, blended_var


def _get_sigma(system: str, stat: str) -> float:
    """Look up forecast sigma, falling back to defaults."""
    sys_sigmas = SYSTEM_FORECAST_SIGMA.get(system, {})
    return sys_sigmas.get(stat, DEFAULT_SIGMA)


def compute_bma_for_player(
    player_ytd: dict[str, float],
    system_projections: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convenience: compute BMA blend and return just the blended projection.

    Args:
        player_ytd: Player's YTD stats.
        system_projections: {system_name: {stat: value}}.

    Returns:
        Blended projection dict {stat: value}.
    """
    _, blended, _ = bayesian_model_average(player_ytd, system_projections)
    return blended


def compute_bma_variance(
    player_ytd: dict[str, float],
    system_projections: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convenience: compute BMA and return the total variance per stat.

    Useful for building KDE marginals — the variance captures both
    model uncertainty (which system is right?) and forecast uncertainty
    (how accurate is the best system?).

    Args:
        player_ytd: Player's YTD stats.
        system_projections: {system_name: {stat: value}}.

    Returns:
        Variance dict {stat: variance}.
    """
    _, _, variance = bayesian_model_average(player_ytd, system_projections)
    return variance
