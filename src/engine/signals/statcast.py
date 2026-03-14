"""Statcast data harvesting via pybaseball.

Spec reference: Section 3 L0 (Signal Harvesting — 147 dimensions)
               Section 17 Phase 3 item 13

Fetches pitch-level data from Baseball Savant and aggregates into
player-level metrics for batters and pitchers. These metrics capture
underlying skill changes that precede fantasy stat changes:
  - Exit velocity / barrel rate → future HR, RBI, AVG
  - Whiff rate / K rate → future K, ERA
  - Sprint speed → future SB, R

Wires into:
  - pybaseball: statcast_batter, statcast_pitcher, playerid_lookup
  - src/engine/signals/decay.py: apply recency weighting
  - src/engine/signals/kalman.py: feed observations for filtering
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Feature groups from spec Section 3 L0
BATTED_BALL_FEATURES: list[str] = [
    "ev_mean",
    "ev_p90",
    "ev_std",
    "la_mean",
    "la_sweet_spot_pct",
    "barrel_pct",
    "hard_hit_pct",
    "pull_pct",
    "center_pct",
    "oppo_pct",
    "gb_pct",
    "fb_pct",
    "ld_pct",
    "xba",
    "xslg",
    "xwoba",
    "xiso",
    "avg_hr_distance",
]

PLATE_DISCIPLINE_FEATURES: list[str] = [
    "o_swing_pct",
    "z_swing_pct",
    "o_contact_pct",
    "z_contact_pct",
    "swstr_pct",
    "csw_pct",
    "f_strike_pct",
    "whiff_pct",
]

SPEED_FEATURES: list[str] = [
    "sprint_speed",
    "sprint_speed_delta",
]

PITCHING_FEATURES: list[str] = [
    "ff_avg_speed",
    "ff_spin_rate",
    "extension",
    "k_pct",
    "bb_pct",
    "hr_per_fb",
    "gb_pct_pitcher",
    "whiff_pct",
    "csw_pct_pitcher",
]

# Pybaseball availability flag
try:
    from pybaseball import statcast_batter, statcast_pitcher

    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


def fetch_batter_statcast(
    mlb_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    days_back: int = 60,
) -> pd.DataFrame:
    """Fetch Statcast pitch-level data for a batter.

    Args:
        mlb_id: MLB player ID (from Baseball Savant).
        start_date: Start of date range. Defaults to days_back ago.
        end_date: End of date range. Defaults to today.
        days_back: If start_date not given, fetch this many days back.

    Returns:
        DataFrame with pitch-level data, or empty DataFrame if unavailable.
    """
    if not PYBASEBALL_AVAILABLE:
        logger.warning("pybaseball not installed — cannot fetch Statcast data")
        return pd.DataFrame()

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days_back)

    try:
        data = statcast_batter(
            start_dt=start_date.strftime("%Y-%m-%d"),
            end_dt=end_date.strftime("%Y-%m-%d"),
            player_id=mlb_id,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        return data
    except Exception as exc:
        logger.warning("Statcast fetch failed for batter %d: %s", mlb_id, exc)
        return pd.DataFrame()


def fetch_pitcher_statcast(
    mlb_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    days_back: int = 60,
) -> pd.DataFrame:
    """Fetch Statcast pitch-level data for a pitcher.

    Args:
        mlb_id: MLB player ID.
        start_date: Start of date range.
        end_date: End of date range.
        days_back: Default lookback window.

    Returns:
        DataFrame with pitch-level data.
    """
    if not PYBASEBALL_AVAILABLE:
        logger.warning("pybaseball not installed — cannot fetch Statcast data")
        return pd.DataFrame()

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days_back)

    try:
        data = statcast_pitcher(
            start_dt=start_date.strftime("%Y-%m-%d"),
            end_dt=end_date.strftime("%Y-%m-%d"),
            player_id=mlb_id,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        return data
    except Exception as exc:
        logger.warning("Statcast fetch failed for pitcher %d: %s", mlb_id, exc)
        return pd.DataFrame()


def aggregate_batter_statcast(pitch_data: pd.DataFrame) -> dict[str, float]:
    """Aggregate pitch-level data into batter feature vector.

    Spec ref: Section 3 L0, Groups 1-3.

    Converts raw pitch-by-pitch data into the feature set used by
    the signal processing pipeline (decay, Kalman, regime detection).

    Args:
        pitch_data: DataFrame from statcast_batter().

    Returns:
        Dict with feature name → aggregated value.
    """
    if pitch_data.empty:
        return {}

    features: dict[str, float] = {}

    # Filter to batted ball events (BIP)
    bip = pitch_data[pitch_data["type"] == "X"] if "type" in pitch_data.columns else pitch_data

    # Group 1: Batted Ball
    if "launch_speed" in pitch_data.columns and not bip.empty:
        ev = bip["launch_speed"].dropna()
        if len(ev) > 0:
            features["ev_mean"] = float(ev.mean())
            features["ev_p90"] = float(ev.quantile(0.9))
            features["ev_std"] = float(ev.std()) if len(ev) > 1 else 0.0

    if "launch_angle" in pitch_data.columns and not bip.empty:
        la = bip["launch_angle"].dropna()
        if len(la) > 0:
            features["la_mean"] = float(la.mean())
            sweet = la.between(8, 32)
            features["la_sweet_spot_pct"] = float(sweet.mean())

    if "barrel" in pitch_data.columns and not bip.empty:
        barrels = bip["barrel"].dropna()
        if len(barrels) > 0:
            features["barrel_pct"] = float(barrels.mean())

    if "launch_speed" in pitch_data.columns and not bip.empty:
        ev = bip["launch_speed"].dropna()
        if len(ev) > 0:
            features["hard_hit_pct"] = float((ev >= 95).mean())

    # Expected stats (from Statcast)
    for col in ["estimated_ba_using_speedangle", "estimated_woba_using_speedangle"]:
        if col in pitch_data.columns:
            vals = pitch_data[col].dropna()
            if len(vals) > 0:
                key = "xba" if "ba" in col else "xwoba"
                features[key] = float(vals.mean())

    # Spray direction
    if "hc_x" in pitch_data.columns and not bip.empty:
        hc = bip["hc_x"].dropna()
        if len(hc) > 0:
            # hc_x: catcher's view, 125 = center
            features["pull_pct"] = float((hc < 100).mean())
            features["center_pct"] = float(hc.between(100, 150).mean())
            features["oppo_pct"] = float((hc > 150).mean())

    # Batted ball types
    if "bb_type" in pitch_data.columns and not bip.empty:
        bb = bip["bb_type"].dropna()
        if len(bb) > 0:
            features["gb_pct"] = float((bb == "ground_ball").mean())
            features["fb_pct"] = float((bb == "fly_ball").mean())
            features["ld_pct"] = float((bb == "line_drive").mean())

    # Group 2: Plate Discipline
    if "description" in pitch_data.columns:
        desc = pitch_data["description"]
        total = len(desc)
        if total > 0:
            swinging_strikes = desc.str.contains("swinging_strike", na=False)
            called_strikes = desc.str.contains("called_strike", na=False)
            features["swstr_pct"] = float(swinging_strikes.mean())
            features["csw_pct"] = float((swinging_strikes | called_strikes).mean())

    if "zone" in pitch_data.columns and "description" in pitch_data.columns:
        zone = pitch_data["zone"]
        desc = pitch_data["description"]
        in_zone = zone.between(1, 9)
        out_zone = ~in_zone & zone.notna()

        swings = desc.str.contains("swing|foul|hit_into", na=False, case=False)

        if out_zone.sum() > 0:
            features["o_swing_pct"] = float((swings & out_zone).sum() / out_zone.sum())
        if in_zone.sum() > 0:
            features["z_swing_pct"] = float((swings & in_zone).sum() / in_zone.sum())

    # Whiff rate
    if "description" in pitch_data.columns:
        desc = pitch_data["description"]
        swings = desc.str.contains("swing|foul|hit_into", na=False, case=False)
        whiffs = desc.str.contains("swinging_strike", na=False)
        if swings.sum() > 0:
            features["whiff_pct"] = float(whiffs.sum() / swings.sum())

    return features


def aggregate_pitcher_statcast(pitch_data: pd.DataFrame) -> dict[str, float]:
    """Aggregate pitch-level data into pitcher feature vector.

    Spec ref: Section 3 L0, Group 4.

    Args:
        pitch_data: DataFrame from statcast_pitcher().

    Returns:
        Dict with feature name → aggregated value.
    """
    if pitch_data.empty:
        return {}

    features: dict[str, float] = {}

    # Fastball metrics
    if "pitch_type" in pitch_data.columns:
        ff = pitch_data[pitch_data["pitch_type"].isin(["FF", "SI"])]
        if "release_speed" in ff.columns and len(ff) > 0:
            features["ff_avg_speed"] = float(ff["release_speed"].dropna().mean())
        if "release_spin_rate" in ff.columns and len(ff) > 0:
            features["ff_spin_rate"] = float(ff["release_spin_rate"].dropna().mean())

    # Release extension
    if "release_extension" in pitch_data.columns:
        ext = pitch_data["release_extension"].dropna()
        if len(ext) > 0:
            features["extension"] = float(ext.mean())

    # Outcome rates
    if "events" in pitch_data.columns:
        events = pitch_data["events"].dropna()
        total_pa = len(events)
        if total_pa > 0:
            strikeouts = events.str.contains("strikeout", na=False)
            walks = events.isin(["walk", "hit_by_pitch"])
            features["k_pct"] = float(strikeouts.sum() / total_pa)
            features["bb_pct"] = float(walks.sum() / total_pa)

    # HR/FB rate
    if "bb_type" in pitch_data.columns and "events" in pitch_data.columns:
        bip = pitch_data[pitch_data["type"] == "X"] if "type" in pitch_data.columns else pitch_data
        fly_balls = bip[bip["bb_type"] == "fly_ball"] if "bb_type" in bip.columns else pd.DataFrame()
        if len(fly_balls) > 0:
            hrs = (
                fly_balls["events"].str.contains("home_run", na=False)
                if "events" in fly_balls.columns
                else pd.Series(dtype=bool)
            )
            features["hr_per_fb"] = float(hrs.mean()) if len(hrs) > 0 else 0.0

    # Ground ball rate
    if "bb_type" in pitch_data.columns:
        bip = pitch_data[pitch_data["type"] == "X"] if "type" in pitch_data.columns else pitch_data
        bb = bip["bb_type"].dropna() if "bb_type" in bip.columns else pd.Series(dtype=str)
        if len(bb) > 0:
            features["gb_pct_pitcher"] = float((bb == "ground_ball").mean())

    # Whiff rate
    if "description" in pitch_data.columns:
        desc = pitch_data["description"]
        swings = desc.str.contains("swing|foul|hit_into", na=False, case=False)
        whiffs = desc.str.contains("swinging_strike", na=False)
        if swings.sum() > 0:
            features["whiff_pct"] = float(whiffs.sum() / swings.sum())

    # CSW rate
    if "description" in pitch_data.columns:
        desc = pitch_data["description"]
        csw = desc.str.contains("swinging_strike|called_strike", na=False)
        features["csw_pct_pitcher"] = float(csw.mean())

    return features


def compute_rolling_features(
    pitch_data: pd.DataFrame,
    window_days: int = 14,
    is_pitcher: bool = False,
) -> list[dict[str, Any]]:
    """Compute rolling-window feature vectors over time.

    Creates a time series of feature snapshots, one per rolling window.
    These feed into the Kalman filter and regime detector.

    Args:
        pitch_data: Full pitch-level DataFrame (must have 'game_date' column).
        window_days: Rolling window size in days.
        is_pitcher: Whether to use pitcher aggregation.

    Returns:
        List of {date, features} dicts sorted chronologically.
    """
    if pitch_data.empty or "game_date" not in pitch_data.columns:
        return []

    pitch_data = pitch_data.copy()
    pitch_data["game_date"] = pd.to_datetime(pitch_data["game_date"])
    pitch_data = pitch_data.sort_values("game_date")

    min_date = pitch_data["game_date"].min()
    max_date = pitch_data["game_date"].max()

    results: list[dict[str, Any]] = []
    current = min_date + timedelta(days=window_days)

    agg_fn = aggregate_pitcher_statcast if is_pitcher else aggregate_batter_statcast

    while current <= max_date:
        window_start = current - timedelta(days=window_days)
        window_data = pitch_data[(pitch_data["game_date"] >= window_start) & (pitch_data["game_date"] <= current)]

        if len(window_data) >= 20:  # Minimum pitches for meaningful aggregation
            features = agg_fn(window_data)
            if features:
                results.append(
                    {
                        "date": current.date() if hasattr(current, "date") else current,
                        "features": features,
                        "n_pitches": len(window_data),
                    }
                )

        current += timedelta(days=7)  # Advance by 1 week

    return results
