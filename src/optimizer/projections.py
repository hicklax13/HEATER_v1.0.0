"""Enhanced projection pipeline for the lineup optimizer.

Chains existing analytics modules to produce Kalman-filtered,
Marcel-stabilized, recency-weighted projections before the LP solver runs.

Pipeline steps (each independently togglable):
  1. Marcel stabilization — regress YTD stats toward preseason priors
  2. Kalman filter — separate true talent from observation noise
  3. Signal decay — weight recent observations more heavily
  4. Statcast features — use barrel%, xwOBA as leading indicators

All steps degrade gracefully when data is unavailable.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

COUNTING_CATS: list[str] = ["r", "hr", "rbi", "sb", "w", "l", "sv", "k"]
RATE_CATS: list[str] = ["avg", "obp", "era", "whip"]
ALL_CATS: list[str] = COUNTING_CATS + RATE_CATS

# Minimum sample size (PA or IP) to trust in-season observed stats
_MIN_SAMPLE_BAYESIAN: int = 30


def compute_fatigue_multiplier(current_ip: float, is_elite: bool = False) -> float:
    """Fatigue discount for pitchers above 100 IP.

    -0.3% per IP above 100, capped at 15% discount.
    Elite pitchers (top Stuff+) are exempt -- research shows 16/20 Cy Young
    winners had better second-half ERA.

    Returns: multiplier in [0.85, 1.0]
    """
    if is_elite:
        return 1.0
    if current_ip <= 100:
        return 1.0
    discount = 0.003 * (current_ip - 100)
    return max(0.85, 1.0 - discount)


def build_enhanced_projections(
    roster: pd.DataFrame,
    config: LeagueConfig | None = None,
    enable_bayesian: bool = True,
    enable_kalman: bool = True,
    enable_statcast: bool = True,
    enable_injury: bool = True,
    weeks_remaining: int = 16,
) -> pd.DataFrame:
    """Build enhanced projections by chaining analytics modules.

    Takes a roster DataFrame (from ``get_team_roster()``) and runs it
    through up to 4 enhancement stages.  Each stage degrades gracefully
    if its data source is unavailable.

    Args:
        roster: Player roster with columns: player_id, player_name,
                positions, is_hitter, plus stat columns.
        config: Optional league configuration for Bayesian update.
        enable_bayesian: Run Marcel regression on YTD vs preseason.
        enable_kalman: Run Kalman filter for true talent estimation.
        enable_statcast: Use Statcast leading indicators (barrel%, xwOBA).
        enable_injury: Scale counting stats by expected availability.
        weeks_remaining: Weeks left in the fantasy season.

    Returns:
        DataFrame with same schema plus additional columns:
          - projection_confidence: float in [0, 1]
          - health_adjusted: bool
    """
    enhanced = roster.copy()

    # Cast counting stat columns to float to avoid dtype warnings
    RATE_DEFAULTS = {"avg": 0.250, "obp": 0.320, "era": 4.50, "whip": 1.30}
    for col in ALL_CATS:
        if col in enhanced.columns:
            enhanced[col] = pd.to_numeric(enhanced[col], errors="coerce")
            if col in RATE_DEFAULTS:
                enhanced[col] = enhanced[col].fillna(RATE_DEFAULTS[col]).astype(float)
            else:
                enhanced[col] = enhanced[col].fillna(0).astype(float)

    enhanced["projection_confidence"] = 1.0
    enhanced["regime_label"] = ""
    enhanced["health_adjusted"] = False

    # Save pre-Bayesian projection values for Kalman prior
    # The Kalman filter needs a separate prior (original projection) and
    # observation (Bayesian-updated value) to produce a meaningful update.
    _pre_bayesian_cols = {}
    for col in ["avg", "era", "whip"]:
        if col in enhanced.columns:
            _pre_bayesian_cols[col] = enhanced[col].copy()

    # Step 1: Bayesian in-season update
    if enable_bayesian:
        enhanced = _apply_bayesian_update(enhanced, config)

    # Step 2: Kalman filter for true talent
    if enable_kalman:
        enhanced = _apply_kalman_filter(enhanced, _pre_bayesian_cols)

    # Step 3: Statcast leading indicators
    if enable_statcast:
        enhanced = _apply_statcast_adjustment(enhanced)

    # Step 4: Recent form adjustment (L7/L14/L30)
    enhanced = _apply_recent_form_adjustment(enhanced)

    # Step 5: Injury availability scaling
    if enable_injury:
        enhanced = _apply_injury_availability(enhanced, weeks_remaining)

    # Step 6: K3 consistency premium for H2H
    # Consistent players are more valuable in weekly H2H matchups.
    # Use xwOBA/BABIP delta as volatility proxy. Small multiplier on counting stats.
    _K3_BONUS = 0.05  # 5% bonus for consistent players
    _K3_PENALTY = 0.05  # 5% penalty for volatile players
    _K3_THRESHOLD = 1.0  # volatility score threshold (1.0 = 1 SD)
    if "xwoba_delta" in enhanced.columns or "babip_delta" in enhanced.columns:
        xd = pd.to_numeric(enhanced.get("xwoba_delta", 0), errors="coerce").fillna(0).abs()
        bd = pd.to_numeric(enhanced.get("babip_delta", 0), errors="coerce").fillna(0).abs()
        vol_score = (xd / 0.030 + bd / 0.030) / 2.0  # normalized to SDs
        counting_cols = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
        for col in counting_cols:
            if col in enhanced.columns:
                # Low volatility (<0.5 SD) → bonus. High (>1.5 SD) → penalty.
                mult = 1.0 + _K3_BONUS * (1.0 - vol_score).clip(-1.0, 1.0)
                enhanced[col] = enhanced[col] * mult

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
        from src.database import coerce_numeric_df, get_connection
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
        # Coerce bytes from SQLite (Python 3.13+)
        season_stats = coerce_numeric_df(season_stats)
        preseason = coerce_numeric_df(preseason)

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


def _apply_kalman_filter(
    roster: pd.DataFrame,
    pre_bayesian_cols: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Apply Kalman filter for true talent estimation.

    Uses the pre-Bayesian projection as the prior and the Bayesian-updated
    value as the observation. The Kalman gain depends on observation
    variance (sample-size-aware) and process variance.

    Args:
        roster: Player roster with Bayesian-updated stat values.
        pre_bayesian_cols: Dict mapping stat name to Series of original
            (pre-Bayesian) projection values. When available, provides
            a meaningful prior distinct from the observation.
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

    if pre_bayesian_cols is None:
        pre_bayesian_cols = {}

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

                # Prior = pre-Bayesian projection (original blend).
                # Observation = Bayesian-updated value (post step 1).
                # If no pre-Bayesian data available, skip this stat —
                # running Kalman with prior == observed is a no-op.
                if stat in pre_bayesian_cols:
                    prior_val = pre_bayesian_cols[stat].get(idx, None)
                    if prior_val is not None and not pd.isna(prior_val):
                        prior_mean = float(prior_val)
                    else:
                        continue
                else:
                    # No separate prior available; skip Kalman for this stat
                    continue

                # If prior equals observation (Bayesian update was a no-op for
                # this player/stat), skip — innovation would be zero.
                if abs(prior_mean - observed) < 1e-9:
                    continue

                # Get observation noise (smaller sample → more noise)
                kalman_stat = _KALMAN_STAT_MAP.get(stat, stat)
                try:
                    obs_var = observation_variance(kalman_stat, int(sample))
                except Exception:
                    obs_var = 0.01

                proc_var = get_process_variance(kalman_stat)

                # One-step Kalman: K = P_prior / (P_prior + R)
                prior_var = proc_var
                kalman_gain = prior_var / (prior_var + obs_var)
                filtered = prior_mean + kalman_gain * (observed - prior_mean)

                # Narrowed posterior variance
                filtered_var = (1 - kalman_gain) * prior_var  # noqa: F841

                roster.at[idx, stat] = filtered
                # Adjust confidence by Kalman gain (high gain = data trustworthy)
                current_conf = float(roster.at[idx, "projection_confidence"])
                roster.at[idx, "projection_confidence"] = min(1.0, current_conf * (0.9 + 0.1 * kalman_gain))

    except Exception as exc:
        logger.warning("Kalman filter failed: %s", exc)

    return roster


# ── Step 3: Statcast Leading Indicators ──────────────────────────────


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

        today = datetime.now(UTC).date()
        start_date = today - timedelta(days=30)
        end_date = today

        for idx, row in roster.iterrows():
            if not row.get("is_hitter", True):
                continue  # Pitcher Statcast handled separately if needed

            mlb_id = row.get("mlb_id", None)
            if mlb_id is None or (isinstance(mlb_id, float) and mlb_id != mlb_id):
                continue  # Skip players without MLB ID
            try:
                mlb_id = int(mlb_id)
            except (ValueError, TypeError):
                continue

            try:
                pitch_data = fetch_batter_statcast(mlb_id, start_date, end_date)
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
                    xwoba_adj = xwoba / 0.320

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


# ── Step 4: Recent Form Adjustment ───────────────────────────────────


# Weight given to L14 recent form vs existing projection
_RECENT_FORM_BLEND = 0.20

# Minimum games in the L14 window to trust recent form data
_MIN_RECENT_GAMES = 7


def _apply_recent_form_adjustment(roster: pd.DataFrame) -> pd.DataFrame:
    """Blend last-14-game stats into projections for hot/cold adjustment.

    Uses ``get_player_recent_form_cached()`` (2h session-state cache) to
    fetch L14 game-log aggregates from the MLB Stats API.  Blends into
    the current projection at 20% weight when at least 7 games exist.

    Hitter stats adjusted: avg, obp, hr (rate), rbi (rate), sb (rate), r (rate).
    Pitcher stats adjusted: era, whip, k (rate per IP).

    Counting-stat adjustments use a *rate ratio*: if recent HR rate is
    1.3x the projected rate, counting projection scales by
    ``1 + blend * (1.3 - 1) = 1.06``.
    """
    try:
        from src.game_day import get_player_recent_form_cached
    except ImportError:
        logger.debug("game_day module not available; skipping recent form adjustment")
        return roster

    adjusted_count = 0

    for idx, row in roster.iterrows():
        mlb_id = row.get("mlb_id", None)
        if mlb_id is None or (isinstance(mlb_id, float) and mlb_id != mlb_id):
            continue
        try:
            mlb_id = int(mlb_id)
        except (ValueError, TypeError):
            continue

        try:
            form = get_player_recent_form_cached(mlb_id)
        except Exception:
            continue

        l14 = form.get("l14", {})
        games = l14.get("games", 0)
        if games < _MIN_RECENT_GAMES:
            continue

        is_hitter = bool(row.get("is_hitter", True))
        player_type = form.get("player_type", "unknown")

        if is_hitter and player_type == "hitter":
            _blend_hitter_form(roster, idx, row, l14)
            adjusted_count += 1
        elif not is_hitter and player_type == "pitcher":
            _blend_pitcher_form(roster, idx, row, l14)
            adjusted_count += 1

    if adjusted_count > 0:
        logger.info("Step 4 (Recent Form): adjusted %d players from L14 game logs", adjusted_count)

    return roster


def _blend_hitter_form(
    roster: pd.DataFrame,
    idx: int,
    row: pd.Series,
    l14: dict,
) -> None:
    """Blend L14 hitter stats into projection at ``_RECENT_FORM_BLEND`` weight."""
    # Rate stats: direct blend
    for stat in ("avg", "obp"):
        recent_val = l14.get(stat, 0)
        proj_val = float(row.get(stat, 0) or 0)
        if recent_val > 0 and proj_val > 0:
            blended = proj_val * (1 - _RECENT_FORM_BLEND) + recent_val * _RECENT_FORM_BLEND
            roster.at[idx, stat] = blended

    # Counting stats: use rate ratio from L14 games
    l14_pa = l14.get("pa", 0)
    if l14_pa < 20:
        return  # Not enough PA for rate reliability

    for stat in ("hr", "rbi", "sb", "r"):
        recent_total = l14.get(stat, 0)
        proj_val = float(row.get(stat, 0) or 0)
        if proj_val <= 0:
            continue
        # Project L14 rate to full-season scale using PA
        proj_pa = float(row.get("pa", 0) or 0)
        if proj_pa <= 0:
            continue
        proj_rate = proj_val / proj_pa
        recent_rate = recent_total / l14_pa
        if proj_rate <= 0:
            continue
        ratio = recent_rate / proj_rate
        # Clamp ratio so adjustments stay within ±15%
        clamped_ratio = max(0.70, min(1.30, ratio))
        adj_factor = 1 + _RECENT_FORM_BLEND * (clamped_ratio - 1)
        roster.at[idx, stat] = proj_val * adj_factor


def _blend_pitcher_form(
    roster: pd.DataFrame,
    idx: int,
    row: pd.Series,
    l14: dict,
) -> None:
    """Blend L14 pitcher stats into projection at ``_RECENT_FORM_BLEND`` weight."""
    # Rate stats: direct blend
    for stat in ("era", "whip"):
        recent_val = l14.get(stat, 0)
        proj_val = float(row.get(stat, 0) or 0)
        if recent_val > 0 and proj_val > 0:
            blended = proj_val * (1 - _RECENT_FORM_BLEND) + recent_val * _RECENT_FORM_BLEND
            roster.at[idx, stat] = blended

    # K rate: use per-IP ratio
    l14_ip = l14.get("ip", 0)
    if l14_ip < 5:
        return  # Not enough IP for rate reliability

    proj_ip = float(row.get("ip", 0) or 0)
    proj_k = float(row.get("k", 0) or 0)
    recent_k = l14.get("k", 0)
    if proj_ip > 0 and proj_k > 0 and recent_k > 0:
        proj_rate = proj_k / proj_ip
        recent_rate = recent_k / l14_ip
        ratio = recent_rate / proj_rate
        clamped_ratio = max(0.70, min(1.30, ratio))
        adj_factor = 1 + _RECENT_FORM_BLEND * (clamped_ratio - 1)
        roster.at[idx, "k"] = proj_k * adj_factor


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
        from src.database import coerce_numeric_df, get_connection

        conn = get_connection()
        try:
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        finally:
            conn.close()
        injury_df = coerce_numeric_df(injury_df)

        if injury_df.empty:
            logger.debug("No injury history data; skipping availability scaling")
            return roster

        rng = np.random.RandomState(42)
        n_samples = 50  # Monte Carlo samples for availability estimate

        for idx, row in roster.iterrows():
            pid = row.get("player_id")
            pi = injury_df[injury_df["player_id"] == pid].sort_values("season", ascending=False)

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
            cats = ["r", "hr", "rbi", "sb"] if not is_pitcher else ["w", "l", "sv", "k"]
            for cat in cats:
                if cat in roster.columns:
                    val = float(row.get(cat, 0) or 0)
                    roster.at[idx, cat] = val * expected_availability

            roster.at[idx, "health_adjusted"] = True
            # Lower confidence for injury-prone players
            if health_score < 0.80:
                current = float(roster.at[idx, "projection_confidence"])
                roster.at[idx, "projection_confidence"] = current * (0.7 + 0.3 * health_score)

    except Exception as exc:
        logger.warning("Injury availability scaling failed: %s", exc)

    return roster


# ── V2 Bayesian Blend ──────────────────────────────────────────────────

# V2 stat-specific stabilization points (from FanGraphs/Russell Carleton research)
V2_STABILIZATION_POINTS: dict[str, float] = {
    "k_rate": 60,
    "bb_rate": 120,
    "hr_rate": 170,
    "sb_rate": 200,
    "avg": 910,
    "obp": 460,
    "era": 630,  # BF, not IP
    "whip": 540,
    "k_rate_pitch": 70,
}


def v2_bayesian_blend(
    preseason_rate: float,
    observed_numerator: float,
    observed_denominator: float,
    stabilization_pa: float,
) -> float:
    """Bayesian blend of preseason projection with observed in-season data.

    Uses the stabilization point as the prior strength. At PA=0, returns
    the preseason projection. At PA=stabilization_pa, returns the average
    of preseason and observed. As PA increases beyond stabilization, the
    observed rate dominates.

    Args:
        preseason_rate: Pre-season projected rate (e.g., 0.280 AVG).
        observed_numerator: Observed counting stat (e.g., hits).
        observed_denominator: Observed sample size (e.g., at-bats).
        stabilization_pa: Stabilization point for this stat type.

    Returns:
        Blended rate estimate.
    """
    if stabilization_pa <= 0:
        stabilization_pa = 200  # Safe default

    prior_numerator = preseason_rate * stabilization_pa
    total_denom = stabilization_pa + observed_denominator

    if total_denom <= 0:
        return preseason_rate

    return (prior_numerator + observed_numerator) / total_denom
