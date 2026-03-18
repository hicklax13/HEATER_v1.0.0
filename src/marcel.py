"""Marcel projection system — 3-year weighted average with regression and age adjustment.

Implements Tom Tango's Marcel the Monkey projection method:
1. Weighted average of last 3 seasons: 5/4/3 (most recent gets highest weight)
2. Regression toward league mean: reliability = PA / (PA + 1200)
3. Age adjustment: peak at 27 (hitters) / 26 (pitchers), decline after peak

Marcel is intentionally simple and serves as a robust baseline that any
sophisticated system should beat. It is used here as a fallback when
FanGraphs projections are unavailable and as a blending component in
the Bayesian Model Averaging pipeline.

Reference: https://www.tangotiger.com/marcel/
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Year Weights (Tom Tango's Marcel) ─────────────────────────────────
# Most recent season = 5, prior = 4, two years ago = 3
YEAR_WEIGHTS: tuple[int, ...] = (5, 4, 3)

# ── Regression Constants ──────────────────────────────────────────────
# PA (or IP for pitchers) at which the projection is 50% player, 50% league
REGRESSION_PA: int = 1200

# ── Age Curve Parameters ──────────────────────────────────────────────
HITTER_PEAK_AGE: int = 27
PITCHER_PEAK_AGE: int = 26
HITTER_DECLINE_RATE: float = 0.005  # 0.5% per year
PITCHER_DECLINE_RATE: float = 0.007  # 0.7% per year

# ── Rate Stats ────────────────────────────────────────────────────────
RATE_STATS: set[str] = {"avg", "obp", "era", "whip"}

# ── League Averages ───────────────────────────────────────────────────
# Per 600 PA for hitters, per 200 IP for pitchers
LEAGUE_AVG_HITTING: dict[str, float] = {
    "r": 72.0,
    "hr": 18.0,
    "rbi": 68.0,
    "sb": 10.0,
    "avg": 0.248,
    "obp": 0.315,
}

LEAGUE_AVG_PITCHING: dict[str, float] = {
    "w": 9.0,
    "l": 9.0,
    "sv": 2.0,
    "k": 160.0,
    "era": 4.20,
    "whip": 1.28,
}

# Full mapping (lowercase stat name -> league average)
LEAGUE_AVERAGES: dict[str, float] = {**LEAGUE_AVG_HITTING, **LEAGUE_AVG_PITCHING}

# ── PA/IP keys by player type ────────────────────────────────────────
HITTER_PA_KEY: str = "pa"
PITCHER_PA_KEY: str = "ip"

# Full-season baseline denominators for counting stat scaling
FULL_SEASON_PA: int = 600
FULL_SEASON_IP: int = 200


def marcel_age_adjustment(age: int | float, is_hitter: bool = True) -> float:
    """Compute Marcel age adjustment multiplier.

    Returns a multiplicative factor centered at 1.0 at peak age.
    Before peak: slight positive adjustment (young player improving).
    After peak: decline at the sport-specific rate.

    Parameters
    ----------
    age : int or float
        Player's age for the projection season.
    is_hitter : bool
        True for position players, False for pitchers.

    Returns
    -------
    float
        Multiplicative adjustment factor. 1.0 at peak, <1.0 after peak,
        >1.0 before peak (capped at 1.02 to avoid over-crediting youth).
    """
    peak = HITTER_PEAK_AGE if is_hitter else PITCHER_PEAK_AGE
    rate = HITTER_DECLINE_RATE if is_hitter else PITCHER_DECLINE_RATE
    years_from_peak = age - peak

    if years_from_peak <= 0:
        # Young player — modest improvement, capped at 2%
        improvement = min(abs(years_from_peak) * rate, 0.02)
        return 1.0 + improvement
    else:
        # Post-peak decline — floor at 0.5 to prevent negative multipliers
        return max(0.5, 1.0 - years_from_peak * rate)


def compute_marcel_projection(
    historical_stats: list[float | None],
    stat: str,
    is_rate: bool = False,
    pa_values: list[float | None] | None = None,
    is_hitter: bool = True,
) -> float:
    """Compute a Marcel projection for a single stat.

    Parameters
    ----------
    historical_stats : list[float | None]
        Up to 3 seasons of stat values, ordered most-recent-first.
        None or missing values are excluded from the weighted average.
    stat : str
        Lowercase stat name (e.g., "hr", "avg", "era").
    is_rate : bool
        If True, use PA-weighted averaging instead of simple weighted sum.
        Automatically set True for stats in RATE_STATS.
    pa_values : list[float | None] | None
        Corresponding PA (or IP) values for each season. Required for
        rate stat weighting. For counting stats, used to compute
        regression toward the mean.
    is_hitter : bool
        True for position players, False for pitchers.

    Returns
    -------
    float
        The Marcel projected value for the stat.
    """
    is_rate = is_rate or (stat.lower() in RATE_STATS)
    league_avg = LEAGUE_AVERAGES.get(stat.lower(), 0.0)

    # Filter out None values and pair with weights
    valid: list[tuple[float, int, float]] = []  # (stat_val, weight, pa)
    for i, val in enumerate(historical_stats[:3]):
        if val is None:
            continue
        weight = YEAR_WEIGHTS[i] if i < len(YEAR_WEIGHTS) else 0
        pa = 0.0
        if pa_values and i < len(pa_values) and pa_values[i] is not None:
            pa = float(pa_values[i])
        valid.append((float(val), weight, pa))

    if not valid:
        # No historical data — return league average
        return league_avg

    if is_rate:
        return _weighted_rate_projection(valid, league_avg, is_hitter)
    else:
        return _weighted_counting_projection(valid, league_avg, is_hitter)


def _weighted_rate_projection(
    valid: list[tuple[float, int, float]],
    league_avg: float,
    is_hitter: bool,
) -> float:
    """PA-weighted average for rate stats, regressed toward league mean."""
    total_weighted_pa = 0.0
    total_weighted_stat = 0.0

    for stat_val, year_weight, pa in valid:
        total_weighted_pa += year_weight
        total_weighted_stat += stat_val * year_weight

    if total_weighted_pa == 0:
        return league_avg

    weighted_avg = total_weighted_stat / total_weighted_pa

    # Regression toward league mean
    # Use sum of raw PA across valid seasons for reliability
    raw_pa = sum(pa for _, _, pa in valid)
    reliability = raw_pa / (raw_pa + REGRESSION_PA)

    projected = reliability * weighted_avg + (1.0 - reliability) * league_avg
    return projected


def _weighted_counting_projection(
    valid: list[tuple[float, int, float]],
    league_avg: float,
    is_hitter: bool,
) -> float:
    """Weighted average for counting stats, regressed toward league mean."""
    total_weight = 0
    weighted_sum = 0.0

    for stat_val, year_weight, _ in valid:
        total_weight += year_weight
        weighted_sum += stat_val * year_weight

    if total_weight == 0:
        return league_avg

    weighted_avg = weighted_sum / total_weight

    # Regression: use total PA across valid seasons
    raw_pa = sum(pa for _, _, pa in valid)
    reliability = raw_pa / (raw_pa + REGRESSION_PA)

    projected = reliability * weighted_avg + (1.0 - reliability) * league_avg
    return projected


def project_player_marcel(
    hist: list[dict[str, float | None]],
    age: int | float,
    is_hitter: bool = True,
) -> dict[str, float]:
    """Project a full stat line for a player using Marcel.

    Parameters
    ----------
    hist : list[dict[str, float | None]]
        Up to 3 seasons of stats, ordered most-recent-first.
        Each dict should have lowercase stat keys (e.g., "hr", "pa", "ip").
    age : int or float
        Player's age for the projection season.
    is_hitter : bool
        True for position players, False for pitchers.

    Returns
    -------
    dict[str, float]
        Projected stat line with all relevant categories.
    """
    if is_hitter:
        stat_cats = list(LEAGUE_AVG_HITTING.keys())
        pa_key = HITTER_PA_KEY
    else:
        stat_cats = list(LEAGUE_AVG_PITCHING.keys())
        pa_key = PITCHER_PA_KEY

    # Extract PA/IP values for each season
    pa_values = [season.get(pa_key) if season else None for season in _pad_hist(hist)]

    age_adj = marcel_age_adjustment(age, is_hitter)
    projection: dict[str, float] = {}

    for stat in stat_cats:
        # Extract historical values for this stat
        historical = [season.get(stat) if season else None for season in _pad_hist(hist)]

        is_rate = stat in RATE_STATS
        raw = compute_marcel_projection(
            historical_stats=historical,
            stat=stat,
            is_rate=is_rate,
            pa_values=pa_values,
            is_hitter=is_hitter,
        )

        # Apply age adjustment
        if is_rate:
            # For rate stats, apply as a multiplicative factor toward the mean.
            # For inverse stats (ERA/WHIP), aging makes them worse (higher),
            # so we invert the adjustment direction: decline (age_adj < 1.0)
            # should push ERA UP, not down.
            league_avg = LEAGUE_AVERAGES.get(stat, 0.0)
            delta = raw - league_avg
            if stat in ("era", "whip"):
                # Inverse: age_adj < 1 means decline, so use (2 - age_adj) to flip
                projection[stat] = league_avg + delta * (2.0 - age_adj)
            else:
                projection[stat] = league_avg + delta * age_adj
        else:
            # For counting stats, straightforward multiplication (floor at 0)
            projection[stat] = max(0.0, raw * age_adj)

    # Project PA/IP as well (useful for downstream consumers)
    pa_hist = [season.get(pa_key) if season else None for season in _pad_hist(hist)]
    full_season = FULL_SEASON_PA if is_hitter else FULL_SEASON_IP
    pa_projection = compute_marcel_projection(
        historical_stats=pa_hist,
        stat=pa_key,
        is_rate=False,
        pa_values=pa_values,
        is_hitter=is_hitter,
    )
    # PA/IP also subject to age adjustment (older players lose playing time)
    projection[pa_key] = max(0.0, pa_projection * age_adj)

    return projection


def _pad_hist(hist: list[dict[str, float | None]]) -> list[dict[str, float | None]]:
    """Pad history to exactly 3 entries (None-fill for missing seasons)."""
    padded = list(hist[:3])
    while len(padded) < 3:
        padded.append({})
    return padded


def compute_marcel_reliability(pa_total: float) -> float:
    """Return the Marcel reliability factor for a given PA total.

    Parameters
    ----------
    pa_total : float
        Total plate appearances (or innings pitched) across available seasons.

    Returns
    -------
    float
        Reliability in [0, 1). Higher values mean the projection leans
        more on observed performance rather than league average.
    """
    if pa_total <= 0:
        return 0.0
    return pa_total / (pa_total + REGRESSION_PA)


def project_batch_marcel(
    players: list[dict],
    age_key: str = "age",
    is_hitter_key: str = "is_hitter",
    history_key: str = "history",
) -> list[dict[str, float]]:
    """Project a batch of players using Marcel.

    Parameters
    ----------
    players : list[dict]
        Each dict must contain:
        - ``age_key``: player age (int/float)
        - ``is_hitter_key``: bool
        - ``history_key``: list of up to 3 season stat dicts (most-recent-first)

    Returns
    -------
    list[dict[str, float]]
        One projection dict per player, in the same order.
    """
    results = []
    for player in players:
        age = player.get(age_key, 28)
        is_hitter = player.get(is_hitter_key, True)
        hist = player.get(history_key, [])
        proj = project_player_marcel(hist, age, is_hitter)
        results.append(proj)
    return results
