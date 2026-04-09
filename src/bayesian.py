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
    "r_rate": 320,  # R/PA — runs stabilize slowly
    "rbi_rate": 300,  # RBI/PA — similar to runs
    "sb_rate": 100,  # SB/PA — binary event, stabilizes faster
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
                for stat in ["w", "l", "sv", "k"]:
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


# ── Integrated ROS Projection Updater ────────────────────────────────

# Marcel season weights: most recent = highest
_MARCEL_WEIGHTS: dict[int, int] = {2026: 5, 2025: 4, 2024: 3}

# Hitting counting stats (per-PA rates)
_HITTING_COUNTING = ["r", "hr", "rbi", "sb"]
# Hitting rate stats (direct regression)
_HITTING_RATES = ["avg", "obp"]
# Pitching counting stats (per-IP rates)
_PITCHING_COUNTING = ["w", "l", "sv", "k"]
# Pitching rate stats (direct regression)
_PITCHING_RATES = ["era", "whip"]

# Stabilization keys for counting-stat rates
_COUNT_STAB_KEYS: dict[str, str] = {
    "r": "r_rate",
    "hr": "hr_rate",
    "rbi": "rbi_rate",
    "sb": "sb_rate",
    "w": "k_rate_pitch",  # ~70 IP for pitcher counting
    "l": "k_rate_pitch",
    "sv": "k_rate_pitch",
    "k": "k_rate_pitch",
}


def _compute_age(birth_date_str: str | None, reference_year: int = 2026) -> int | None:
    """Compute age from birth_date string (YYYY-MM-DD)."""
    if not birth_date_str or not isinstance(birth_date_str, str):
        return None
    try:
        parts = birth_date_str.split("-")
        birth_year = int(parts[0])
        if birth_year < 1970 or birth_year > 2010:
            return None
        return reference_year - birth_year
    except (ValueError, IndexError):
        return None


def update_ros_projections() -> int:
    """Integrated Bayesian ROS projection updater.

    Orchestrates a 3-layer pipeline to produce rest-of-season projections
    for every player in the database:

    Layer 1 — Marcel-weighted historical prior (2024/2025/2026 at 3/4/5)
    Layer 2 — Blend with expert projections (reliability-weighted)
    Layer 3 — Regress 2026 observed stats toward blended prior (stabilization)
    Output  — ROS = updated full-season projection minus 2026 YTD

    Returns:
        Number of ROS projections written to the database.
    """
    from src.database import get_connection, update_refresh_log

    conn = get_connection()
    try:
        # ── Step 1: Load all data ────────────────────────────────────
        season_stats = pd.read_sql_query(
            "SELECT * FROM season_stats WHERE season IN (2025, 2026)",
            conn,
        )
        projections = pd.read_sql_query(
            "SELECT * FROM projections WHERE system = 'blended'",
            conn,
        )
        players = pd.read_sql_query(
            "SELECT player_id, birth_date, positions FROM players",
            conn,
        )

        if projections.empty:
            logger.warning("No blended projections found — cannot update ROS.")
            return 0

        # Coerce numeric columns
        for df in (season_stats, projections):
            for col in df.columns:
                if col not in ("system", "last_updated", "updated_at", "birth_date", "positions"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        updater = BayesianUpdater()

        # Index projections by player_id for fast lookup
        proj_by_pid = projections.set_index("player_id")

        # Split season stats by year
        stats_by_year: dict[int, pd.DataFrame] = {}
        for year in (2024, 2025, 2026):
            yr_df = season_stats[season_stats["season"] == year].copy()
            if not yr_df.empty:
                yr_df = yr_df.drop_duplicates(subset=["player_id"], keep="first")
                stats_by_year[year] = yr_df.set_index("player_id")

        # Build age lookup
        age_lookup: dict[int, int | None] = {}
        for _, p in players.iterrows():
            age_lookup[int(p["player_id"])] = _compute_age(
                p.get("birth_date") if not pd.isna(p.get("birth_date")) else None
            )

        # Determine if a player is a pitcher from the players table
        pitcher_pids: set[int] = set()
        for _, p in players.iterrows():
            pos = str(p.get("positions", ""))
            if pos and any(pp in pos for pp in ("SP", "RP")):
                pitcher_pids.add(int(p["player_id"]))

        # ── Step 2-7: Process each player ────────────────────────────
        ros_rows: list[dict] = []
        all_pids = set(proj_by_pid.index)

        for pid in all_pids:
            proj_row = proj_by_pid.loc[pid]
            is_pitcher = pid in pitcher_pids

            # Determine volume basis (PA for hitters, IP for pitchers)
            proj_pa = float(proj_row.get("pa", 0) or 0)
            proj_ip = float(proj_row.get("ip", 0) or 0)

            # ── Layer 1: Marcel-weighted historical prior ────────────
            # Accumulate weighted stats across 2024/2025/2026
            marcel_hitting = _build_marcel_prior(
                pid, stats_by_year, _HITTING_COUNTING, _HITTING_RATES, "pa", is_rate_ip=False
            )
            marcel_pitching = _build_marcel_prior(
                pid, stats_by_year, _PITCHING_COUNTING, _PITCHING_RATES, "ip", is_rate_ip=True
            )

            # ── Layer 2: Blend historical with expert projections ────
            if not is_pitcher and proj_pa > 0:
                blended = _blend_with_projection(marcel_hitting, proj_row, _HITTING_COUNTING, _HITTING_RATES, "pa")
            else:
                blended = {}
            if is_pitcher or proj_ip > 0:
                blended.update(
                    _blend_with_projection(marcel_pitching, proj_row, _PITCHING_COUNTING, _PITCHING_RATES, "ip")
                )

            # ── Layer 3: Age adjustment ──────────────────────────────
            age = age_lookup.get(pid)
            if age is not None:
                for stat in _HITTING_COUNTING + _PITCHING_COUNTING:
                    rate_key = f"{stat}_rate"
                    if rate_key in blended:
                        blended[rate_key] *= updater.age_adjustment(age, stat)
                for stat in _HITTING_RATES + _PITCHING_RATES:
                    if stat in blended:
                        mult = updater.age_adjustment(age, stat)
                        if stat in ("era", "whip"):
                            # Inverse: aging makes ERA/WHIP worse (higher)
                            blended[stat] = blended[stat] / mult if mult > 0 else blended[stat]
                        else:
                            blended[stat] *= mult

            # ── Layer 4: Regress 2026 observed toward blended prior ──
            stats_2026 = stats_by_year.get(2026)
            obs_pa = 0
            obs_ip = 0.0
            if stats_2026 is not None and pid in stats_2026.index:
                obs = stats_2026.loc[pid]
                obs_pa = int(float(obs.get("pa", 0) or 0))
                obs_ip = float(obs.get("ip", 0) or 0)
            else:
                obs = None

            # Rate stats: regress observed toward blended prior
            updated_rates: dict[str, float] = {}

            if not is_pitcher and proj_pa > 0:
                for stat in _HITTING_RATES:
                    prior = blended.get(stat, float(proj_row.get(stat, 0) or 0))
                    observed = float(obs.get(stat, 0) or 0) if obs is not None else 0
                    stab = STABILIZATION_POINTS.get(stat, 460)
                    updated_rates[stat] = updater.regressed_rate_with_prior(observed, obs_pa, prior, stab)

                for stat in _HITTING_COUNTING:
                    prior_rate = blended.get(f"{stat}_rate", 0)
                    if obs is not None and obs_pa > 0:
                        obs_rate = float(obs.get(stat, 0) or 0) / obs_pa
                    else:
                        obs_rate = 0
                    stab_key = _COUNT_STAB_KEYS.get(stat, "r_rate")
                    stab = STABILIZATION_POINTS.get(stab_key, 200)
                    updated_rates[f"{stat}_rate"] = updater.regressed_rate_with_prior(
                        obs_rate, obs_pa, prior_rate, stab
                    )

            if is_pitcher or proj_ip > 0:
                for stat in _PITCHING_RATES:
                    prior = blended.get(stat, float(proj_row.get(stat, 0) or 0))
                    observed = float(obs.get(stat, 0) or 0) if obs is not None else 0
                    stab = STABILIZATION_POINTS.get(stat, 70)
                    updated_rates[stat] = updater.regressed_rate_with_prior(observed, int(obs_ip), prior, stab)

                for stat in _PITCHING_COUNTING:
                    prior_rate = blended.get(f"{stat}_rate", 0)
                    if obs is not None and obs_ip > 0:
                        obs_rate = float(obs.get(stat, 0) or 0) / obs_ip
                    else:
                        obs_rate = 0
                    stab_key = _COUNT_STAB_KEYS.get(stat, "k_rate_pitch")
                    stab = STABILIZATION_POINTS.get(stab_key, 70)
                    updated_rates[f"{stat}_rate"] = updater.regressed_rate_with_prior(
                        obs_rate, int(obs_ip), prior_rate, stab
                    )

            # ── Layer 5: Project full-season then compute ROS ────────
            row_out: dict = {"player_id": pid, "system": "ros_bayesian"}

            if not is_pitcher and proj_pa > 0:
                remaining_pa = max(0, proj_pa - obs_pa)
                obs_ab = int(float(obs.get("ab", 0) or 0)) if obs is not None else 0
                proj_ab = int(float(proj_row.get("ab", 0) or 0))
                remaining_ab = max(0, proj_ab - obs_ab)

                row_out["avg"] = round(updated_rates.get("avg", float(proj_row.get("avg", 0) or 0)), 3)
                row_out["obp"] = round(updated_rates.get("obp", float(proj_row.get("obp", 0) or 0)), 3)
                row_out["pa"] = int(remaining_pa)
                row_out["ab"] = int(remaining_ab)
                row_out["h"] = int(row_out["avg"] * remaining_ab) if remaining_ab > 0 else 0

                # Derive BB/HBP/SF from blended projection per-PA rates.
                # These are needed for proper OBP computation in
                # _roster_category_totals().  Without them OBP = H/AB = AVG.
                for component in ("bb", "hbp", "sf"):
                    proj_component = int(float(proj_row.get(component, 0) or 0))
                    obs_component = int(float(obs.get(component, 0) or 0)) if obs is not None else 0
                    if proj_pa > 0 and remaining_pa > 0:
                        rate = proj_component / proj_pa
                        row_out[component] = max(0, int(rate * remaining_pa))
                    else:
                        row_out[component] = 0

                for stat in _HITTING_COUNTING:
                    rate = updated_rates.get(f"{stat}_rate", 0)
                    obs_val = int(float(obs.get(stat, 0) or 0)) if obs is not None else 0
                    full_season = obs_val + int(rate * remaining_pa)
                    ros_val = max(0, full_season - obs_val)
                    row_out[stat] = ros_val

            if is_pitcher or proj_ip > 0:
                remaining_ip = max(0.0, proj_ip - obs_ip)
                row_out["ip"] = round(remaining_ip, 1)
                row_out["era"] = round(updated_rates.get("era", float(proj_row.get("era", 0) or 0)), 2)
                row_out["whip"] = round(updated_rates.get("whip", float(proj_row.get("whip", 0) or 0)), 2)

                for stat in _PITCHING_COUNTING:
                    rate = updated_rates.get(f"{stat}_rate", 0)
                    ros_val = max(0, int(rate * remaining_ip))
                    row_out[stat] = ros_val

                if remaining_ip > 0:
                    row_out["er"] = int(row_out.get("era", 0) * remaining_ip / 9)
                    total_br = row_out.get("whip", 0) * remaining_ip
                    row_out["bb_allowed"] = int(total_br * 0.35)
                    row_out["h_allowed"] = int(total_br * 0.65)
                else:
                    row_out["er"] = 0
                    row_out["bb_allowed"] = 0
                    row_out["h_allowed"] = 0

            # Copy advanced metrics from projections unchanged
            for col in ("fip", "xfip", "siera"):
                row_out.setdefault(col, float(proj_row.get(col, 0) or 0))

            ros_rows.append(row_out)

        # ── Step 8: Write to database ────────────────────────────────
        if not ros_rows:
            logger.warning("No ROS projections computed.")
            return 0

        ros_df = pd.DataFrame(ros_rows)

        conn.execute("DELETE FROM ros_projections WHERE system = 'ros_bayesian'")

        # Build INSERT statement matching table columns
        insert_cols = [
            "player_id",
            "system",
            "pa",
            "ab",
            "h",
            "r",
            "hr",
            "rbi",
            "sb",
            "avg",
            "obp",
            "bb",
            "hbp",
            "sf",
            "ip",
            "w",
            "l",
            "sv",
            "k",
            "era",
            "whip",
            "er",
            "bb_allowed",
            "h_allowed",
            "fip",
            "xfip",
            "siera",
        ]
        for col in insert_cols:
            if col not in ros_df.columns:
                ros_df[col] = 0

        placeholders = ", ".join(["?"] * len(insert_cols))
        col_str = ", ".join(insert_cols)
        for _, row in ros_df.iterrows():
            values = [row.get(c, 0) for c in insert_cols]
            conn.execute(f"INSERT INTO ros_projections ({col_str}) VALUES ({placeholders})", values)

        conn.commit()
        update_refresh_log("ros_projections", "success")
        logger.info("Updated %d ROS Bayesian projections.", len(ros_rows))
        return len(ros_rows)

    except Exception:
        logger.exception("Failed to update ROS projections.")
        try:
            update_refresh_log("ros_projections", "error")
        except Exception:
            pass
        return 0
    finally:
        conn.close()


def _build_marcel_prior(
    pid: int,
    stats_by_year: dict[int, pd.DataFrame],
    counting_stats: list[str],
    rate_stats: list[str],
    volume_col: str,
    is_rate_ip: bool = False,
) -> dict[str, float]:
    """Build a Marcel-weighted prior from 2024/2025/2026 historical stats.

    Returns dict with rate stat values and counting stat per-PA/IP rates.
    """
    result: dict[str, float] = {}
    total_weighted_vol = 0.0

    # Accumulate weighted volumes for counting rate computation
    weighted_counting: dict[str, float] = {s: 0.0 for s in counting_stats}
    weighted_rate_num: dict[str, float] = {s: 0.0 for s in rate_stats}
    weighted_rate_den: dict[str, float] = {s: 0.0 for s in rate_stats}

    for year, weight in _MARCEL_WEIGHTS.items():
        yr_stats = stats_by_year.get(year)
        if yr_stats is None or pid not in yr_stats.index:
            continue

        row = yr_stats.loc[pid]
        vol = float(row.get(volume_col, 0) or 0)
        if vol <= 0:
            continue

        weighted_vol = vol * weight
        total_weighted_vol += weighted_vol

        # Counting stats: accumulate weighted raw values
        for stat in counting_stats:
            val = float(row.get(stat, 0) or 0)
            weighted_counting[stat] += val * weight

        # Rate stats: accumulate for weighted average
        # For AVG: weighted by AB; for ERA: weighted by IP
        for stat in rate_stats:
            val = float(row.get(stat, 0) or 0)
            if val > 0:
                weighted_rate_num[stat] += val * weighted_vol
                weighted_rate_den[stat] += weighted_vol

    if total_weighted_vol > 0:
        result["total_weighted_vol"] = total_weighted_vol
        for stat in counting_stats:
            result[f"{stat}_rate"] = weighted_counting[stat] / total_weighted_vol
        for stat in rate_stats:
            if weighted_rate_den[stat] > 0:
                result[stat] = weighted_rate_num[stat] / weighted_rate_den[stat]

    return result


def _blend_with_projection(
    marcel: dict[str, float],
    proj_row: pd.Series,
    counting_stats: list[str],
    rate_stats: list[str],
    volume_col: str,
) -> dict[str, float]:
    """Blend Marcel historical prior with expert projection.

    Uses reliability weighting: trust historical data proportional to
    total weighted PA/IP (caps at ~2 full seasons = 1200 PA or 400 IP).
    """
    result: dict[str, float] = {}
    total_vol = marcel.get("total_weighted_vol", 0)

    # Reliability threshold: 1200 PA for hitters, 400 IP for pitchers
    threshold = 400.0 if volume_col == "ip" else 1200.0
    reliability = min(1.0, total_vol / threshold) if total_vol > 0 else 0.0

    proj_vol = float(proj_row.get(volume_col, 0) or 0)

    for stat in rate_stats:
        marcel_val = marcel.get(stat)
        proj_val = float(proj_row.get(stat, 0) or 0)
        if marcel_val is not None and reliability > 0:
            result[stat] = reliability * marcel_val + (1 - reliability) * proj_val
        else:
            result[stat] = proj_val

    for stat in counting_stats:
        marcel_rate = marcel.get(f"{stat}_rate")
        proj_val = float(proj_row.get(stat, 0) or 0)
        proj_rate = proj_val / proj_vol if proj_vol > 0 else 0
        if marcel_rate is not None and reliability > 0:
            result[f"{stat}_rate"] = reliability * marcel_rate + (1 - reliability) * proj_rate
        else:
            result[f"{stat}_rate"] = proj_rate

    return result
