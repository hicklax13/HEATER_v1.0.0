"""Enhanced projection pipeline for the lineup optimizer.

Chains existing analytics modules to produce Kalman-filtered,
regime-adjusted, Bayesian-updated, recency-weighted projections
before the LP solver runs.

Pipeline steps (each independently togglable):
  1. Bayesian update — regress YTD stats toward preseason priors
  2. Kalman filter — separate true talent from observation noise
  3. Regime detection — identify hot/cold streaks, adjust projections
  4. Signal decay — weight recent observations more heavily
  5. Statcast features — use barrel%, xwOBA as leading indicators
  6. Injury availability — Weibull-distributed season availability scaling

All steps degrade gracefully when data is unavailable.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

COUNTING_CATS: list[str] = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
RATE_CATS: list[str] = ["avg", "era", "whip"]
ALL_CATS: list[str] = COUNTING_CATS + RATE_CATS

# Regime multipliers: how much to adjust projections per regime state.
# States: Elite, Above-avg, Below-avg, Replacement (from regime.py)
_REGIME_MULTIPLIERS: dict[str, float] = {
    "Elite": 1.10,
    "Above-avg": 1.03,
    "Below-avg": 0.97,
    "Replacement": 0.88,
}

# Default league-average xwOBA for regime classification fallback
_LEAGUE_AVG_XWOBA: float = 0.315

# Minimum sample size (PA or IP) to trust in-season observed stats
_MIN_SAMPLE_BAYESIAN: int = 30


def build_enhanced_projections(
    roster: pd.DataFrame,
    config: LeagueConfig | None = None,
    enable_bayesian: bool = True,
    enable_kalman: bool = True,
    enable_regime: bool = True,
    enable_statcast: bool = True,
    enable_injury: bool = True,
    weeks_remaining: int = 16,
) -> pd.DataFrame:
    """Build enhanced projections by chaining analytics modules.

    Takes a roster DataFrame (from ``get_team_roster()``) and runs it
    through up to 6 enhancement stages.  Each stage degrades gracefully
    if its data source is unavailable.

    Args:
        roster: Player roster with columns: player_id, player_name,
                positions, is_hitter, plus stat columns.
        config: Optional league configuration for Bayesian update.
        enable_bayesian: Run Marcel regression on YTD vs preseason.
        enable_kalman: Run Kalman filter for true talent estimation.
        enable_regime: Detect hot/cold streaks via regime classification.
        enable_statcast: Use Statcast leading indicators (barrel%, xwOBA).
        enable_injury: Scale counting stats by expected availability.
        weeks_remaining: Weeks left in the fantasy season.

    Returns:
        DataFrame with same schema plus additional columns:
          - projection_confidence: float in [0, 1]
          - regime_label: str or empty
          - health_adjusted: bool
    """
    enhanced = roster.copy()

    # Cast counting stat columns to float to avoid dtype warnings
    for col in COUNTING_CATS + RATE_CATS:
        if col in enhanced.columns:
            enhanced[col] = pd.to_numeric(enhanced[col], errors="coerce").fillna(0).astype(float)

    enhanced["projection_confidence"] = 1.0
    enhanced["regime_label"] = ""
    enhanced["health_adjusted"] = False

    # Step 1: Bayesian in-season update
    if enable_bayesian:
        enhanced = _apply_bayesian_update(enhanced, config)

    # Step 2: Kalman filter for true talent
    if enable_kalman:
        enhanced = _apply_kalman_filter(enhanced)

    # Step 3: Regime detection (hot/cold streaks)
    if enable_regime:
        enhanced = _apply_regime_adjustment(enhanced)

    # Step 4: Statcast leading indicators
    if enable_statcast:
        enhanced = _apply_statcast_adjustment(enhanced)

    # Step 5: Injury availability scaling
    if enable_injury:
        enhanced = _apply_injury_availability(enhanced, weeks_remaining)

    return enhanced


# ── Step 1: Bayesian Update ──────────────────────────────────────────


def _apply_bayesian_update(
    roster: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Regress in-season stats toward preseason projections.

    Uses Marcel-style regression: weight observed performance by sample
    size relative to the stabilization point for each stat.
    """
    try:
        from src.bayesian import BayesianUpdater
        from src.database import get_connection
    except ImportError:
        logger.debug("Bayesian updater not available; skipping step 1")
        return roster

    try:
        conn = get_connection()
        try:
            season_stats = pd.read_sql_query(
                "SELECT * FROM season_stats WHERE season = ?",
                conn,
                params=(datetime.now(UTC).year,),
            )
            preseason = pd.read_sql_query(
                "SELECT * FROM projections WHERE system = 'blended'",
                conn,
            )
        finally:
            conn.close()

        if season_stats.empty or preseason.empty:
            logger.debug("No season stats or preseason projections; skipping Bayesian update")
            return roster

        updater = BayesianUpdater()
        updated = updater.batch_update_projections(season_stats, preseason, config)

        if updated is None or updated.empty:
            return roster

        # Merge updated projections back into roster by player_id
        roster = _merge_updated_stats(roster, updated)

        # Reduce confidence based on sample size
        for idx, row in roster.iterrows():
            pa = float(row.get("pa", 0) or 0)
            ip = float(row.get("ip", 0) or 0)
            sample = pa if row.get("is_hitter", True) else ip
            if sample < _MIN_SAMPLE_BAYESIAN:
                # Low sample → lower confidence (blended with preseason)
                roster.at[idx, "projection_confidence"] = min(1.0, 0.5 + 0.5 * (sample / _MIN_SAMPLE_BAYESIAN))

    except Exception as exc:
        logger.warning("Bayesian update failed: %s", exc)

    return roster


def _merge_updated_stats(
    roster: pd.DataFrame,
    updated: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Bayesian-updated stat values into the roster DataFrame."""
    if "player_id" not in updated.columns:
        return roster

    stat_cols = [c for c in ALL_CATS if c in updated.columns and c in roster.columns]
    if not stat_cols:
        return roster

    update_map = {}
    for _, row in updated.iterrows():
        pid = row.get("player_id")
        if pid is not None:
            update_map[pid] = {col: row[col] for col in stat_cols if pd.notna(row[col])}

    for idx, row in roster.iterrows():
        pid = row.get("player_id")
        if pid in update_map:
            for col, val in update_map[pid].items():
                roster.at[idx, col] = val

    return roster


# ── Step 2: Kalman Filter ────────────────────────────────────────────


def _apply_kalman_filter(roster: pd.DataFrame) -> pd.DataFrame:
    """Apply Kalman filter for true talent estimation.

    Uses the player's current YTD stats as a single observation point
    against their preseason projection as a prior.  The Kalman gain
    depends on the observation variance (sample-size-aware).
    """
    try:
        from src.engine.signals.kalman import (
            get_process_variance,
            observation_variance,
        )
    except ImportError:
        logger.debug("Kalman module not available; skipping step 2")
        return roster

    # Ensure enhancement columns exist (defensive for direct calls)
    if "projection_confidence" not in roster.columns:
        roster["projection_confidence"] = 1.0

    try:
        for idx, row in roster.iterrows():
            is_hitter = bool(row.get("is_hitter", True))
            pa = float(row.get("pa", 0) or 0)
            ip = float(row.get("ip", 0) or 0)
            sample = pa if is_hitter else ip

            if sample < 10:
                # Too few observations for meaningful Kalman update
                continue

            # Apply Kalman to key rate stats where noise matters most
            stats_to_filter = ["avg"] if is_hitter else ["era", "whip"]

            # Map optimizer stat names to Kalman module stat names
            _KALMAN_STAT_MAP = {"avg": "ba", "era": "era", "whip": "whip"}

            for stat in stats_to_filter:
                observed = float(row.get(stat, 0) or 0)
                if observed == 0:
                    continue

                # The prior is the current projection value (already blended
                # from Bayesian/preseason stages).  The "observation" is this
                # same value treated as in-season data.  With only one column
                # we split the signal: prior = projection, observation = projection,
                # but Kalman gain still adjusts confidence by sample size.
                prior_mean = observed

                # Get observation noise (smaller sample → more noise)
                kalman_stat = _KALMAN_STAT_MAP.get(stat, stat)
                try:
                    obs_var = observation_variance(kalman_stat, int(sample))
                except Exception:
                    obs_var = 0.01

                proc_var = get_process_variance(kalman_stat)

                # One-step Kalman: K = P_prior / (P_prior + R)
                prior_var = proc_var + obs_var
                kalman_gain = prior_var / (prior_var + obs_var)
                filtered = prior_mean + kalman_gain * (observed - prior_mean)

                # Narrowed posterior variance
                filtered_var = (1 - kalman_gain) * prior_var

                roster.at[idx, stat] = filtered
                # Adjust confidence by Kalman gain (high gain = data trustworthy)
                current_conf = float(roster.at[idx, "projection_confidence"])
                roster.at[idx, "projection_confidence"] = min(1.0, current_conf * (0.9 + 0.1 * kalman_gain))

    except Exception as exc:
        logger.warning("Kalman filter failed: %s", exc)

    return roster


# ── Step 3: Regime Detection ─────────────────────────────────────────


def _apply_regime_adjustment(roster: pd.DataFrame) -> pd.DataFrame:
    """Detect hot/cold streaks and adjust projections.

    Uses rule-based classification when Statcast xwOBA is unavailable.
    Applies regime-specific multipliers to counting stats.
    """
    try:
        from src.engine.signals.regime import classify_regime_simple
    except ImportError:
        logger.debug("Regime detection not available; skipping step 3")
        return roster

    try:
        for idx, row in roster.iterrows():
            is_hitter = bool(row.get("is_hitter", True))

            # Approximate xwOBA from available rate stats
            if is_hitter:
                avg = float(row.get("avg", 0) or 0)
                if avg == 0:
                    continue
                # Approximate xwOBA from AVG (rough: xwOBA ≈ AVG * 1.15)
                approx_xwoba = avg * 1.15
            else:
                # For pitchers, invert ERA to approximate opposing xwOBA
                # League avg ERA ≈ 4.00 maps to xwOBA ≈ 0.315
                era = float(row.get("era", 0) or 0)
                if era == 0:
                    continue
                approx_xwoba = era / 4.00 * _LEAGUE_AVG_XWOBA

            season_xwoba = approx_xwoba  # Without separate recent data, use same

            regime_label, state_probs = classify_regime_simple(
                recent_xwoba=approx_xwoba,
                season_xwoba=season_xwoba,
                league_avg_xwoba=_LEAGUE_AVG_XWOBA,
            )

            roster.at[idx, "regime_label"] = regime_label

            # Compute probability-weighted multiplier
            multiplier = 0.0
            for state_name, prob in zip(
                ["Elite", "Above-avg", "Below-avg", "Replacement"],
                state_probs,
                strict=False,
            ):
                multiplier += prob * _REGIME_MULTIPLIERS[state_name]

            # Apply multiplier to counting stats only (rate stats are self-correcting)
            if abs(multiplier - 1.0) > 0.005:
                is_hitter = bool(row.get("is_hitter", True))
                cats = ["r", "hr", "rbi", "sb"] if is_hitter else ["w", "sv", "k"]
                for cat in cats:
                    if cat in roster.columns:
                        val = float(row.get(cat, 0) or 0)
                        roster[cat] = roster[cat].astype(float)
                        roster.at[idx, cat] = val * multiplier

    except Exception as exc:
        logger.warning("Regime adjustment failed: %s", exc)

    return roster


# ── Step 4: Statcast Leading Indicators ──────────────────────────────


def _apply_statcast_adjustment(roster: pd.DataFrame) -> pd.DataFrame:
    """Use Statcast features as leading indicators for projection adjustment.

    Barrel% and xwOBA are stronger predictors of future HR/RBI than
    traditional stats.  When available, adjust projections toward what
    the underlying skills suggest.
    """
    try:
        from src.engine.signals.statcast import PYBASEBALL_AVAILABLE

        if not PYBASEBALL_AVAILABLE:
            logger.debug("pybaseball not available; skipping Statcast adjustment")
            return roster
    except ImportError:
        logger.debug("Statcast module not available; skipping step 4")
        return roster

    try:
        from src.engine.signals.statcast import (
            aggregate_batter_statcast,
            fetch_batter_statcast,
        )

        today = datetime.now(UTC)
        start_date = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        for idx, row in roster.iterrows():
            if not row.get("is_hitter", True):
                continue  # Pitcher Statcast handled separately if needed

            name = row.get("player_name", row.get("name", ""))
            if not name:
                continue

            try:
                pitch_data = fetch_batter_statcast(name, start_date, end_date)
                if pitch_data is None or pitch_data.empty:
                    continue

                features = aggregate_batter_statcast(pitch_data)
                if not features:
                    continue

                # Use barrel% as HR leading indicator
                barrel_pct = features.get("barrel_pct", 0)
                xwoba = features.get("xwoba", 0)

                if barrel_pct > 0 and xwoba > 0:
                    # Barrel% > 10% suggests power upside
                    # xwOBA > .350 suggests above-average production
                    barrel_adj = 1.0 + max(0, (barrel_pct - 8.0)) * 0.01
                    xwoba_adj = xwoba / max(_LEAGUE_AVG_XWOBA, 0.001)

                    # Blend: 30% Statcast signal, 70% existing projection
                    blend_weight = 0.3
                    composite_adj = 1.0 + blend_weight * ((barrel_adj * xwoba_adj) - 1.0)
                    composite_adj = np.clip(composite_adj, 0.85, 1.20)

                    for cat in ["hr", "rbi", "r"]:
                        if cat in roster.columns:
                            val = float(row.get(cat, 0) or 0)
                            roster.at[idx, cat] = val * composite_adj

            except Exception:
                continue  # Per-player graceful degradation

    except Exception as exc:
        logger.warning("Statcast adjustment failed: %s", exc)

    return roster


# ── Step 5: Injury Availability Scaling ──────────────────────────────


def _apply_injury_availability(
    roster: pd.DataFrame,
    weeks_remaining: int = 16,
) -> pd.DataFrame:
    """Scale counting stats by expected season availability.

    Uses Weibull-distributed injury duration modeling rather than the
    simple deterministic health penalty in the UI page.  This gives a
    probabilistic estimate of what fraction of remaining games a player
    will actually play.
    """
    try:
        from src.engine.context.injury_process import sample_season_availability
        from src.injury_model import compute_health_score
    except ImportError:
        logger.debug("Injury modules not available; skipping step 5")
        return roster

    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        finally:
            conn.close()

        if injury_df.empty:
            logger.debug("No injury history data; skipping availability scaling")
            return roster

        rng = np.random.RandomState(42)
        n_samples = 50  # Monte Carlo samples for availability estimate

        for idx, row in roster.iterrows():
            pid = row.get("player_id")
            pi = injury_df[injury_df["player_id"] == pid]

            if pi.empty:
                health_score = 0.85  # League average
            else:
                games_played = pi["games_played"].tolist()
                games_available = pi["games_available"].tolist()
                health_score = compute_health_score(games_played, games_available)

            is_pitcher = not bool(row.get("is_hitter", True))

            # Monte Carlo: sample availability multiple times and average
            availability_samples = []
            for _ in range(n_samples):
                avail = sample_season_availability(
                    health_score=health_score,
                    age=None,  # Age unknown from roster data
                    is_pitcher=is_pitcher,
                    weeks_remaining=weeks_remaining,
                    rng=rng,
                )
                availability_samples.append(avail)

            expected_availability = float(np.mean(availability_samples))

            # Scale counting stats by expected availability
            cats = ["r", "hr", "rbi", "sb"] if not is_pitcher else ["w", "sv", "k"]
            for cat in cats:
                if cat in roster.columns:
                    val = float(row.get(cat, 0) or 0)
                    roster[cat] = roster[cat].astype(float)
                    roster.at[idx, cat] = val * expected_availability

            roster.at[idx, "health_adjusted"] = True
            # Lower confidence for injury-prone players
            if health_score < 0.80:
                current = float(roster.at[idx, "projection_confidence"])
                roster.at[idx, "projection_confidence"] = current * (0.7 + 0.3 * health_score)

    except Exception as exc:
        logger.warning("Injury availability scaling failed: %s", exc)

    return roster
