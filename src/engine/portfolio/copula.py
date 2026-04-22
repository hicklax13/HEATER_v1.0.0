"""Inter-category correlation modeling via copula.

Spec reference: Section 7 L4A (Vine Copula)

Fantasy stat categories are correlated:
  - HR and RBI are strongly positively correlated (~0.85)
  - SB and HR are weakly negatively correlated (~-0.15)
  - ERA and WHIP are strongly positively correlated (~0.90)
  - W and K are moderately positively correlated (~0.50)

A copula separates the marginal distributions (handled by marginals.py)
from the dependency structure. This lets us sample correlated stat lines
that respect both the individual stat distributions AND their real-world
correlations.

Implementation: Uses a Gaussian copula as the primary approach. This
handles the most important aspect (correlation) without requiring the
heavy `copulas` library. Falls back gracefully to independent sampling
when correlation data is unavailable.

Wires into:
  - src/engine/projections/marginals.py: PlayerMarginal ppf for inverse CDF
  - src/engine/portfolio/category_analysis.py: CATEGORIES, INVERSE_CATEGORIES
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

from src.valuation import LeagueConfig as _LC_Class

logger = logging.getLogger(__name__)

CATEGORIES: list[str] = list(_LC_Class().all_categories)
INVERSE_CATEGORIES: set[str] = {"L", "ERA", "WHIP"}

# Empirical correlation matrix for fantasy stat categories.
# Derived from 5+ years of qualified MLB seasons (min 400 PA hitters, 100 IP pitchers).
# Hitter cats (R, HR, RBI, SB, AVG, OBP) have strong internal correlations;
# pitcher cats (W, L, SV, K, ERA, WHIP) have their own cluster.
# Cross-cluster correlations are near zero (hitter stats don't predict pitcher stats).
#
# Order: R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP
DEFAULT_CORRELATION: np.ndarray = np.array(
    [
        # R     HR    RBI   SB    AVG   OBP   W     L     SV    K     ERA   WHIP
        [1.00, 0.75, 0.80, 0.20, 0.55, 0.45, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # R
        [0.75, 1.00, 0.85, -0.15, 0.30, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # HR
        [0.80, 0.85, 1.00, -0.10, 0.40, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # RBI
        [0.20, -0.15, -0.10, 1.00, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # SB
        [0.55, 0.30, 0.40, 0.15, 1.00, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # AVG
        [0.45, 0.20, 0.35, 0.10, 0.80, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # OBP
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, -0.30, -0.20, 0.50, -0.55, -0.50],  # W
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.30, 1.00, -0.10, -0.20, 0.50, 0.45],  # L
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.20, -0.10, 1.00, -0.10, -0.30, -0.25],  # SV
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, -0.20, -0.10, 1.00, -0.65, -0.60],  # K
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.55, 0.50, -0.30, -0.65, 1.00, 0.90],  # ERA
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.50, 0.45, -0.25, -0.60, 0.90, 1.00],  # WHIP
    ],
    dtype=float,
)


CAT_ORDER = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]


def compute_empirical_correlation(
    team_weekly_totals: dict[str, dict[str, float]],
) -> np.ndarray | None:
    """P3: Compute empirical correlation matrix from league weekly totals.

    Uses actual league category totals to build a 12x12 correlation matrix,
    replacing the hardcoded DEFAULT_CORRELATION. Returns None if insufficient
    data (need at least 8 team-weeks).

    Args:
        team_weekly_totals: {team_name: {cat: total}} for all teams.

    Returns:
        12x12 numpy array or None if insufficient data.
    """
    if not team_weekly_totals or len(team_weekly_totals) < 8:
        return None

    try:
        # Build data matrix: rows=teams, cols=categories
        rows = []
        for _team, totals in team_weekly_totals.items():
            row = [float(totals.get(cat, 0)) for cat in CAT_ORDER]
            if any(v != 0 for v in row):
                rows.append(row)

        if len(rows) < 8:
            return None

        data = np.array(rows, dtype=float)
        # Compute Pearson correlation
        corr = np.corrcoef(data.T)

        # Sanitize: replace NaN with 0 (constant columns)
        corr = np.nan_to_num(corr, nan=0.0)
        # Ensure diagonal is 1.0
        np.fill_diagonal(corr, 1.0)
        # Clamp to [-1, 1]
        corr = np.clip(corr, -1.0, 1.0)

        return corr
    except Exception:
        logger.error("Correlation matrix estimation failed — MC sim loses correlation structure", exc_info=True)
        return None


class GaussianCopula:
    """Gaussian copula for sampling correlated uniform variates.

    A Gaussian copula works by:
    1. Drawing correlated normal variates using the Cholesky decomposition
       of the correlation matrix.
    2. Transforming them to uniform [0,1] via the normal CDF (Phi).
    3. These uniform variates preserve the correlation structure.

    The consumer then applies inverse CDF (ppf) of each player's
    marginal distribution to convert uniform -> realistic stat values.
    """

    def __init__(self, correlation: np.ndarray | None = None) -> None:
        """Initialize with a correlation matrix.

        P3: Prefers empirical correlation when provided; falls back to
        DEFAULT_CORRELATION (12×12 hardcoded from historical MLB data).

        Args:
            correlation: n×n correlation matrix. Must be positive semi-definite.
                Defaults to DEFAULT_CORRELATION (12×12 for all fantasy categories).
        """
        if correlation is None:
            correlation = DEFAULT_CORRELATION.copy()

        self.correlation = correlation
        self.n_dims = correlation.shape[0]

        # Ensure positive definiteness via nearest PD approximation
        self.correlation = _nearest_pd(self.correlation)

        # Cholesky decomposition for efficient sampling
        self._cholesky = np.linalg.cholesky(self.correlation)

    def sample(
        self,
        n: int = 1,
        rng: np.random.RandomState | None = None,
    ) -> np.ndarray:
        """Draw n sets of correlated uniform variates.

        Args:
            n: Number of samples to draw.
            rng: Random state for reproducibility.

        Returns:
            Array of shape (n, n_dims) with values in [0, 1].
            Each row is one correlated sample. Columns correspond
            to the category order in the correlation matrix.
        """
        if rng is None:
            rng = np.random.RandomState()

        # Draw independent standard normals
        z = rng.standard_normal((n, self.n_dims))

        # Correlate them via Cholesky
        correlated = z @ self._cholesky.T

        # Transform to uniform via normal CDF
        u = norm.cdf(correlated)

        # Clip to avoid exactly 0 or 1 (which blow up ppf)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        return u


def sample_correlated_stats(
    copula: GaussianCopula,
    player_marginals: dict[str, object],
    n: int = 1,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Draw n correlated stat lines using copula + player marginals.

    Spec ref: Section 7 L4A — sample_correlated_stats.

    Pipeline:
    1. Draw correlated uniform variates from the copula
    2. For each category, apply the player's marginal ppf (inverse CDF)
       to convert uniform -> stat value

    For inverse stats (ERA, WHIP), we flip the quantile (1-u) because
    the copula correlation matrix has negative entries for these
    (lower ERA = better, but we want the copula to treat "better" as
    higher quantiles for all stats).

    Args:
        copula: Fitted GaussianCopula instance.
        player_marginals: Dict mapping stat_name -> PlayerMarginal.
            Must have a .ppf(quantile) method.
        n: Number of stat lines to sample.
        rng: Random state for reproducibility.

    Returns:
        Array of shape (n, len(CATEGORIES)) with sampled stat values.
    """
    u = copula.sample(n, rng)
    stats = np.zeros((n, len(CATEGORIES)))

    for i, cat in enumerate(CATEGORIES):
        cat_lower = cat.lower()
        if cat_lower not in player_marginals:
            continue

        marginal = player_marginals[cat_lower]
        if cat in INVERSE_CATEGORIES:
            # Flip quantile: low u -> good (low) ERA
            stats[:, i] = marginal.ppf(1 - u[:, i])
        else:
            stats[:, i] = marginal.ppf(u[:, i])

    return stats


def fit_copula_from_data(
    player_seasons: np.ndarray,
) -> GaussianCopula:
    """Fit a Gaussian copula from historical player-season data.

    Args:
        player_seasons: Array of shape (n_seasons, 12) with columns in
            CATEGORIES order. For ERA/WHIP, pass raw values (they'll be
            inverted internally for correlation estimation).

    Returns:
        Fitted GaussianCopula.
    """
    data = player_seasons.copy()

    # Invert L/ERA/WHIP so "better" = higher for correlation estimation
    for inv_cat in INVERSE_CATEGORIES:
        idx = CATEGORIES.index(inv_cat)
        data[:, idx] = -data[:, idx]

    # Compute Spearman rank correlation (more robust than Pearson for non-normal data)
    from scipy.stats import spearmanr

    corr_matrix, _ = spearmanr(data)

    # Ensure it's a proper correlation matrix
    if corr_matrix.ndim == 0:
        # Single column — shouldn't happen but guard
        corr_matrix = np.array([[1.0]])

    return GaussianCopula(corr_matrix)


def _nearest_pd(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest positive-definite matrix.

    Uses the Higham (2002) algorithm: project onto the cone of
    positive semi-definite matrices via eigenvalue clipping.

    Args:
        matrix: Square symmetric matrix.

    Returns:
        Nearest positive-definite matrix.
    """
    # Symmetrize
    b = (matrix + matrix.T) / 2

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(b)

    # Clip negative eigenvalues to small positive
    eigvals = np.maximum(eigvals, 1e-10)

    # Reconstruct
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure diagonal is exactly 1 (correlation matrix)
    d = np.sqrt(np.diag(result))
    result = result / np.outer(d, d)

    return result
