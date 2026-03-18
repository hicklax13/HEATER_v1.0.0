"""KDE marginal distributions for player stat projections.

Spec reference: Section 17 Phase 2 item 10 (KDE marginals replace Normal)

Fantasy stats are NOT normally distributed:
  - SB is heavily right-skewed (most steal <5, a few steal 40+)
  - AVG is bounded [0, 1]
  - SV is bimodal (closers get ~30, everyone else ~0)
  - ERA has a long right tail (blowup risk)

This module fits kernel density estimates to capture the real shape of
each stat distribution, then provides a ppf (percent point function)
for Monte Carlo sampling via copula.

Wires into existing:
  - src/engine/projections/bayesian_blend.py: BMA variance for bandwidth tuning
  - scipy.stats.gaussian_kde: core KDE engine
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import gaussian_kde, norm

# Minimum number of data points for KDE; below this, fall back to Normal
MIN_KDE_SAMPLES: int = 20

# Stat bounds: prevent unrealistic samples
STAT_BOUNDS: dict[str, tuple[float, float]] = {
    "avg": (0.100, 0.400),
    "obp": (0.200, 0.500),
    "era": (0.50, 12.00),
    "whip": (0.60, 3.00),
    "hr": (0, 70),
    "r": (0, 150),
    "rbi": (0, 160),
    "sb": (0, 80),
    "w": (0, 25),
    "l": (0, 20),
    "k": (0, 350),
    "sv": (0, 55),
}


class PlayerMarginal:
    """A marginal distribution for one stat of one player.

    Wraps either a KDE (when enough historical data exists) or a
    Gaussian fallback (centered on the BMA projection with BMA variance).

    Provides ppf() for inverse CDF sampling (copula integration) and
    pdf() for density evaluation.
    """

    def __init__(
        self,
        stat_name: str,
        projected_value: float,
        variance: float,
        historical_values: np.ndarray | None = None,
    ) -> None:
        """Initialize a marginal distribution.

        Args:
            stat_name: Category name (lowercase, e.g. "hr", "avg").
            projected_value: BMA-blended projection for this stat.
            variance: BMA variance (within + between model).
            historical_values: Optional array of historical season values
                for this player in this stat. If len >= MIN_KDE_SAMPLES,
                a KDE is fitted; otherwise falls back to Normal.
        """
        self.stat_name = stat_name
        self.projected_value = projected_value
        self.variance = max(variance, 1e-10)
        self.std = float(np.sqrt(self.variance))
        self._bounds = STAT_BOUNDS.get(stat_name, (None, None))
        self._kde: gaussian_kde | None = None
        self._kde_min: float = 0.0
        self._kde_max: float = 1.0

        if historical_values is not None and len(historical_values) >= MIN_KDE_SAMPLES:
            # Filter out NaN/inf
            clean = historical_values[np.isfinite(historical_values)]
            if len(clean) >= MIN_KDE_SAMPLES:
                try:
                    self._kde = gaussian_kde(clean)
                    self._kde_min = float(np.min(clean))
                    self._kde_max = float(np.max(clean))
                except np.linalg.LinAlgError:
                    # Singular matrix — fall back to Normal
                    self._kde = None

    @property
    def uses_kde(self) -> bool:
        """Whether this marginal uses KDE (True) or Normal fallback (False)."""
        return self._kde is not None

    def ppf(self, quantile: float | np.ndarray) -> Any:
        """Percent point function (inverse CDF).

        Used by copula sampling: given a uniform [0,1] quantile,
        return the stat value at that quantile.

        Args:
            quantile: Scalar or array of quantiles in [0, 1].

        Returns:
            Stat value(s) at the given quantile(s).
        """
        if self._kde is not None:
            return self._ppf_kde(quantile)
        return self._ppf_normal(quantile)

    def _ppf_normal(self, quantile: float | np.ndarray) -> Any:
        """Normal distribution inverse CDF."""
        result = norm.ppf(quantile, loc=self.projected_value, scale=self.std)
        return self._clip(result)

    def _ppf_kde(self, quantile: float | np.ndarray) -> Any:
        """KDE inverse CDF via numerical inversion.

        KDE doesn't have a closed-form ppf, so we build a numerical
        approximation: evaluate the CDF on a fine grid, then interpolate.
        """
        assert self._kde is not None

        # Build grid spanning the KDE support
        spread = self._kde_max - self._kde_min
        lo = self._kde_min - 0.5 * spread
        hi = self._kde_max + 0.5 * spread

        n_points = 500
        grid = np.linspace(lo, hi, n_points)
        pdf_vals = self._kde.evaluate(grid)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]  # Normalize to [0, 1]

        # Interpolate: given quantile, find grid value
        result = np.interp(quantile, cdf_vals, grid)
        return self._clip(result)

    def _clip(self, value: Any) -> Any:
        """Clip to stat bounds if defined."""
        lo, hi = self._bounds
        if lo is not None:
            value = np.maximum(value, lo)
        if hi is not None:
            value = np.minimum(value, hi)
        return value

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> np.ndarray:
        """Draw n samples from this marginal.

        Args:
            n: Number of samples.
            rng: Optional random state for reproducibility.

        Returns:
            Array of n stat values.
        """
        if rng is None:
            rng = np.random.RandomState()

        u = rng.uniform(0, 1, size=n)
        return np.asarray(self.ppf(u), dtype=float)


def build_player_marginals(
    projected_stats: dict[str, float],
    variances: dict[str, float],
    historical_by_stat: dict[str, np.ndarray] | None = None,
) -> dict[str, PlayerMarginal]:
    """Build a full set of marginal distributions for one player.

    Args:
        projected_stats: BMA-blended projections {stat: value}.
        variances: BMA variances {stat: variance}.
        historical_by_stat: Optional {stat: array_of_historical_values}
            for KDE fitting.

    Returns:
        Dict mapping stat_name -> PlayerMarginal.
    """
    marginals: dict[str, PlayerMarginal] = {}
    for stat, proj_val in projected_stats.items():
        var = variances.get(stat, 1.0)
        hist = None
        if historical_by_stat is not None:
            hist = historical_by_stat.get(stat)

        marginals[stat] = PlayerMarginal(
            stat_name=stat,
            projected_value=proj_val,
            variance=var,
            historical_values=hist,
        )

    return marginals
