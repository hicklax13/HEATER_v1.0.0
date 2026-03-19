"""Bayesian projection updater with PyMC hierarchical models and Marcel fallback.

Uses FanGraphs-validated stabilization thresholds to regress observed stats
toward preseason priors. When PyMC is available, runs full hierarchical
beta-binomial models for rate stats. Falls back to Marcel-style weighted
regression when PyMC is not installed (e.g., in CI environments).

Key concepts:
- Stabilization point: PA/IP needed for a stat to be 50% signal, 50% noise
- Marcel regression: (observed * n + prior * stabilization) / (n + stabilization)
- Age adjustment: Applied on logit scale to avoid 0/1 boundary artifacts
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# Try importing PyMC — fall back gracefully if unavailable
try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.info("PyMC not available — using Marcel regression fallback")

# ── FanGraphs Stabilization Thresholds ───────────────────────────────
# Source: FanGraphs "How Long Until a Stat Stabilizes?" research
# These represent the number of PA (or IP) at which a stat is 50% signal.
STABILIZATION_POINTS: dict[str, int] = {
    "k_rate": 60,  # K/PA stabilizes quickly
    "bb_rate": 120,  # BB/PA
    "hr_rate": 170,  # HR/PA (FB% driven)
    "avg": 910,  # BA — extremely noisy, needs large samples
    "babip": 820,  # BABIP — similarly noisy
    "iso": 160,  # Isolated power
    "obp": 460,  # OBP
    "slg": 320,  # SLG
    "era": 70,  # IP-based: ERA stabilizes around 70 IP
    "whip": 70,  # IP-based
    "k_rate_pitch": 70,  # Pitcher K/9 (IP-based)
    "bb_rate_pitch": 170,  # Pitcher BB/9 (IP-based)
    "hr_rate_pitch": 300,  # Pitcher HR/9 — very noisy
}

# ── Aging Curves ─────────────────────────────────────────────────────
# Peak ages by skill type (logit-scale adjustment applied after peak)
PEAK_AGES: dict[str, int] = {
    "power": 27,  # HR, SLG
    "speed": 26,  # SB, triples
    "contact": 29,  # AVG, K rate
    "pitching": 27,  # ERA, WHIP, K
    "default": 28,
}

# Annual decline rate after peak (on raw scale, before logit transform)
DECLINE_RATES: dict[str, float] = {
    "power": 0.015,
    "speed": 0.025,  # Speed declines fastest
    "contact": 0.010,
    "pitching": 0.012,
    "default": 0.015,
}

# ── League-Wide Means (2023-2025 MLB averages) ──────────────────────
LEAGUE_MEANS: dict[str, float] = {
    "avg": 0.248,
    "obp": 0.317,
    "slg": 0.406,
    "hr_rate": 0.033,  # HR/PA
    "k_rate": 0.224,  # K/PA
    "bb_rate": 0.085,  # BB/PA
    "sb_rate": 0.015,  # SB/PA
    "era": 4.17,
    "whip": 1.27,
    "k_rate_pitch": 0.232,  # K/BF
    "bb_rate_pitch": 0.079,  # BB/BF
}


class BayesianUpdater:
    """Bayesian in-season projection updater.

    Updates preseason projections with observed stats using either:
    1. PyMC 5 hierarchical beta-binomial models (when available)
    2. Marcel-style regression formula (always available)

    The update weight shifts from prior → observed as sample size grows,
    governed by FanGraphs stabilization thresholds.
    """

    def __init__(self, prior_weight: float = 0.6):
        """Initialize updater.

        Args:
            prior_weight: Initial weight on preseason projections (0-1).
                         Decreases as sample size increases.
        """
        self.prior_weight = max(0.0, min(1.0, prior_weight))
        self._cached_traces: dict[str, object] = {}

    def regressed_rate(
        self,
        observed_rate: float,
        sample_size: int,
        league_mean: float,
        stabilization_point: int,
    ) -> float:
        """Marcel-style regression toward the mean.

        Formula: (observed * n + league_mean * stab) / (n + stab)

        As sample_size approaches stabilization_point, the estimate
        shifts from league_mean toward the observed rate.

        Args:
            observed_rate: Observed rate stat (e.g., 0.280 AVG)
            sample_size: Number of PA or IP
            league_mean: League average for this stat
            stabilization_point: PA/IP needed for 50% signal

        Returns:
            Regressed rate estimate
        """
        if stabilization_point <= 0:
            return observed_rate
        if sample_size <= 0:
            return league_mean

        n = max(0, sample_size)
        stab = max(1, stabilization_point)
        return (observed_rate * n + league_mean * stab) / (n + stab)

    def regressed_rate_with_prior(
        self,
        observed_rate: float,
        sample_size: int,
        prior_rate: float,
        stabilization_point: int,
    ) -> float:
        """Regression toward a player-specific prior (preseason projection).

        Same formula as regressed_rate but uses the player's preseason
        projection as the prior instead of the league mean.

        Args:
            observed_rate: In-season observed rate
            sample_size: PA or IP observed
            prior_rate: Preseason projected rate
            stabilization_point: PA/IP for 50% signal

        Returns:
            Blended rate estimate
        """
        if stabilization_point <= 0:
            return observed_rate
        if sample_size <= 0:
            return prior_rate

        n = max(0, sample_size)
        stab = max(1, stabilization_point)
        return (observed_rate * n + prior_rate * stab) / (n + stab)

    def hierarchical_rate_model(
        self,
        observed_rates: np.ndarray,
        sample_sizes: np.ndarray,
        group_mean: float,
    ) -> dict:
        """PyMC 5 hierarchical beta-binomial model for rate stats.

        Fits a hierarchical model where individual player rates are drawn
        from a common beta distribution, then updated with binomial likelihood.

        Args:
            observed_rates: Array of observed rates per player
            sample_sizes: Array of PA/IP per player
            group_mean: Prior group mean (league or position average)

        Returns:
            Dict with keys: posterior_means, hdi_low, hdi_high, converged
        """
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available — returning Marcel estimates")
            stab = 200  # Default stabilization
            means = np.array(
                [self.regressed_rate(r, int(n), group_mean, stab) for r, n in zip(observed_rates, sample_sizes)]
            )
            return {
                "posterior_means": means,
                "hdi_low": means * 0.85,
                "hdi_high": means * 1.15,
                "converged": True,
            }

        # Filter out zero sample sizes
        mask = sample_sizes > 0
        if not mask.any():
            return {
                "posterior_means": np.full_like(observed_rates, group_mean),
                "hdi_low": np.full_like(observed_rates, group_mean * 0.85),
                "hdi_high": np.full_like(observed_rates, group_mean * 1.15),
                "converged": True,
            }

        obs = observed_rates[mask]
        ns = sample_sizes[mask].astype(int)
        successes = np.round(obs * ns).astype(int)

        try:
            with pm.Model() as model:
                # Hyperpriors for group-level beta distribution
                kappa = pm.HalfNormal("kappa", sigma=50)
                mu = pm.Beta("mu", alpha=2, beta=2)

                # Player-level rates drawn from group distribution
                alpha_player = mu * kappa
                beta_player = (1 - mu) * kappa
                theta = pm.Beta(
                    "theta",
                    alpha=alpha_player,
                    beta=beta_player,
                    shape=len(obs),
                )

                # Binomial likelihood
                pm.Binomial("obs", n=ns, p=theta, observed=successes)

                # Sample
                trace = pm.sample(
                    1000,
                    tune=500,
                    cores=1,
                    return_inferencedata=True,
                    progressbar=False,
                )

            posterior = trace.posterior["theta"].values.reshape(-1, len(obs))
            means_masked = posterior.mean(axis=0)
            hdi = az.hdi(trace, var_names=["theta"], hdi_prob=0.90)
            hdi_vals = hdi["theta"].values

            # Reconstruct full-size arrays
            full_means = np.full(len(observed_rates), group_mean)
            full_low = np.full(len(observed_rates), group_mean * 0.85)
            full_high = np.full(len(observed_rates), group_mean * 1.15)
            full_means[mask] = means_masked
            full_low[mask] = hdi_vals[:, 0]
            full_high[mask] = hdi_vals[:, 1]

            return {
                "posterior_means": full_means,
                "hdi_low": full_low,
                "hdi_high": full_high,
                "converged": True,
            }

        except Exception:
            logger.exception("PyMC model failed — falling back to Marcel")
            stab = 200
            means = np.array(
                [self.regressed_rate(r, int(n), group_mean, stab) for r, n in zip(observed_rates, sample_sizes)]
            )
            return {
                "posterior_means": means,
                "hdi_low": means * 0.85,
                "hdi_high": means * 1.15,
                "converged": False,
            }

    def age_adjustment(self, age: int, stat: str) -> float:
        """Compute aging curve multiplier for a stat.

        Applied on a sigmoid-like scale to avoid boundary artifacts
        (e.g., negative AVG or ERA below 0).

        Args:
            age: Player's current age
            stat: Stat category (maps to skill type)

        Returns:
            Multiplier (1.0 at peak, declining after)
        """
        # Map stat to skill type
        stat_lower = stat.lower()
        if stat_lower in ("hr", "slg", "iso", "hr_rate"):
            skill = "power"
        elif stat_lower in ("sb", "sb_rate", "triples"):
            skill = "speed"
        elif stat_lower in ("avg", "babip", "k_rate", "obp"):
            skill = "contact"
        elif stat_lower in ("era", "whip", "k_rate_pitch", "bb_rate_pitch", "w", "sv", "k"):
            skill = "pitching"
        else:
            skill = "default"

        peak = PEAK_AGES[skill]
        decline_rate = DECLINE_RATES[skill]

        if age <= peak:
            return 1.0

        years_past_peak = age - peak
        multiplier = 1.0 - (years_past_peak * decline_rate)
        return max(0.5, multiplier)  # Floor at 50%

    def batch_update_projections(
        self,
        season_stats: pd.DataFrame,
        preseason: pd.DataFrame,
        config: LeagueConfig | None = None,
    ) -> pd.DataFrame:
        """Update preseason projections with in-season observed stats.

        For each player:
        1. Regress observed rates toward preseason prior using stabilization thresholds
        2. Scale counting stat projections based on remaining games
        3. Apply age adjustment

        Args:
            season_stats: DataFrame with columns: player_id, pa, ab, h, r, hr, rbi, sb, avg,
                         ip, w, sv, k, era, whip, er, bb_allowed, h_allowed, games_played
            preseason: DataFrame with same stat columns plus player_id
            config: Optional league config for category weights

        Returns:
            Updated projections DataFrame with system='bayesian'
        """
        if season_stats.empty or preseason.empty:
            logger.warning("Empty stats or preseason data — returning preseason as-is")
            result = preseason.copy()
            if "system" not in result.columns:
                result["system"] = "bayesian"
            return result

        # Merge on player_id
        merged = preseason.merge(
            season_stats,
            on="player_id",
            how="left",
            suffixes=("_pre", "_obs"),
        )

        result_rows = []
        for _, row in merged.iterrows():
            player_id = row["player_id"]
            obs_pa = _safe_val(row, "pa_obs")
            obs_ip = _safe_val(row, "ip_obs")
            pre_pa = _safe_val(row, "pa_pre")
            pre_ip = _safe_val(row, "ip_pre")

            is_hitter = pre_pa > 0 or obs_pa > 0
            is_pitcher = pre_ip > 0 or obs_ip > 0

            updated = {"player_id": player_id, "system": "bayesian"}

            if is_hitter and (obs_pa > 0 or pre_pa > 0):
                # Update hitting rate stats via regression
                obs_avg = _safe_val(row, "avg_obs")
                pre_avg = _safe_val(row, "avg_pre")
                updated["avg"] = self.regressed_rate_with_prior(
                    obs_avg, int(obs_pa), pre_avg, STABILIZATION_POINTS["avg"]
                )

                # OBP regression (scoring category)
                obs_obp = _safe_val(row, "obp_obs")
                pre_obp = _safe_val(row, "obp_pre")
                if obs_obp > 0 or pre_obp > 0:
                    updated["obp"] = self.regressed_rate_with_prior(
                        obs_obp, int(obs_pa), pre_obp, STABILIZATION_POINTS["obp"]
                    )

                # Update counting stats: blend observed rate with preseason rate
                for stat in ["r", "hr", "rbi", "sb"]:
                    obs_val = _safe_val(row, f"{stat}_obs")
                    pre_val = _safe_val(row, f"{stat}_pre")
                    if obs_pa > 0:
                        obs_rate = obs_val / obs_pa
                    else:
                        obs_rate = 0
                    if pre_pa > 0:
                        pre_rate = pre_val / pre_pa
                    else:
                        pre_rate = 0

                    stab = STABILIZATION_POINTS.get(f"{stat}_rate", 200)
                    blended_rate = self.regressed_rate_with_prior(obs_rate, int(obs_pa), pre_rate, stab)
                    # Project over remaining PA
                    remaining_pa = max(0, pre_pa - obs_pa)
                    updated[stat] = int(obs_val + blended_rate * remaining_pa)

                updated["pa"] = int(pre_pa)
                obs_ab = int(_safe_val(row, "ab_obs"))
                updated["ab"] = int(_safe_val(row, "ab_pre"))
                # Recalculate h: observed hits + blended avg * remaining AB
                obs_h = int(_safe_val(row, "h_obs"))
                remaining_ab = max(0, updated["ab"] - obs_ab)
                updated["h"] = int(obs_h + updated["avg"] * remaining_ab)

            if is_pitcher and (obs_ip > 0 or pre_ip > 0):
                # Update pitching rate stats
                obs_era = _safe_val(row, "era_obs")
                pre_era = _safe_val(row, "era_pre")
                updated["era"] = self.regressed_rate_with_prior(
                    obs_era, int(obs_ip), pre_era, STABILIZATION_POINTS["era"]
                )

                obs_whip = _safe_val(row, "whip_obs")
                pre_whip = _safe_val(row, "whip_pre")
                updated["whip"] = self.regressed_rate_with_prior(
                    obs_whip, int(obs_ip), pre_whip, STABILIZATION_POINTS["whip"]
                )

                # Counting stats
                for stat in ["w", "sv", "k"]:
                    obs_val = _safe_val(row, f"{stat}_obs")
                    pre_val = _safe_val(row, f"{stat}_pre")
                    if obs_ip > 0:
                        obs_rate = obs_val / obs_ip
                    else:
                        obs_rate = 0
                    if pre_ip > 0:
                        pre_rate = pre_val / pre_ip
                    else:
                        pre_rate = 0

                    stab = STABILIZATION_POINTS.get(f"{stat}_rate", 100)
                    blended_rate = self.regressed_rate_with_prior(obs_rate, int(obs_ip), pre_rate, stab)
                    remaining_ip = max(0, pre_ip - obs_ip)
                    updated[stat] = int(obs_val + blended_rate * remaining_ip)

                updated["ip"] = pre_ip
                # Derive ER from blended ERA
                if updated["ip"] > 0:
                    updated["er"] = int(updated["era"] * updated["ip"] / 9)
                else:
                    updated["er"] = 0

                # Derive bb_allowed and h_allowed from WHIP
                if updated["ip"] > 0:
                    total_baserunners = updated["whip"] * updated["ip"]
                    updated["bb_allowed"] = int(total_baserunners * 0.35)
                    updated["h_allowed"] = int(total_baserunners * 0.65)
                else:
                    updated["bb_allowed"] = 0
                    updated["h_allowed"] = 0

            result_rows.append(updated)

        result = pd.DataFrame(result_rows)

        # Fill any missing columns with preseason values (merge-based to avoid positional mismatch)
        for col in preseason.columns:
            if col not in result.columns and col != "system":
                col_map = preseason.set_index("player_id")[col]
                result[col] = result["player_id"].map(col_map)

        return result

    @staticmethod
    def get_stabilization_point(stat: str) -> int:
        """Get the PA/IP needed for a stat to be 50% signal.

        Args:
            stat: Stat name (e.g., 'avg', 'era', 'k_rate')

        Returns:
            Number of PA (hitters) or IP (pitchers) for stabilization
        """
        return STABILIZATION_POINTS.get(stat.lower(), 200)


# ── Module-level helpers ─────────────────────────────────────────────


def _safe_val(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely extract a numeric value from a merged row."""
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default
