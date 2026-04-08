# src/power_rankings.py
"""5-factor composite league power rankings with bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd

# B7: Component weights (sum to 1.0) — momentum increased for H2H hot-streak value
WEIGHTS: dict[str, float] = {
    "roster_quality": 0.30,
    "category_balance": 0.25,
    "schedule_strength": 0.15,
    "injury_exposure": 0.10,
    "momentum": 0.20,
}


def compute_roster_quality(team_trade_values: float, max_trade_values: float) -> float:
    """Roster quality as fraction of league-best total trade value [0, 1]."""
    if max_trade_values <= 0:
        return 0.0
    return max(0.0, min(1.0, team_trade_values / max_trade_values))


def compute_category_balance_score(
    team_zscores: dict[str, float],
) -> float:
    """Category balance = 1 / (1 + CV(z_scores)) where CV = std/|mean|.

    Returns [0, 1]. Higher = more balanced.
    """
    if not team_zscores:
        return 0.5
    values = list(team_zscores.values())
    std = float(np.std(values))
    mean = float(np.mean(values))
    if abs(mean) < 1e-9:
        # Near-zero mean: use just std-based score
        return max(0.0, min(1.0, 1.0 / (1.0 + std)))
    cv = std / abs(mean)
    return max(0.0, min(1.0, 1.0 / (1.0 + cv)))


def compute_schedule_strength_index(
    team_name: str,
    roster_qualities: dict[str, float],
    schedule: list[list[tuple[str, str]]] | None = None,
) -> float:
    """Mean opponent roster quality, normalized [0, 1].

    Falls back to league average if no schedule provided.
    """
    if not roster_qualities:
        return 0.5

    if schedule:
        opponents = []
        for week in schedule:
            for a, b in week:
                if a == team_name:
                    opponents.append(roster_qualities.get(b, 0.5))
                elif b == team_name:
                    opponents.append(roster_qualities.get(a, 0.5))
        if opponents:
            return round(float(np.mean(opponents)), 4)

    # Fallback: average of all other teams
    others = [v for k, v in roster_qualities.items() if k != team_name]
    return round(float(np.mean(others)), 4) if others else 0.5


def compute_injury_exposure(
    roster_trade_values: list[float],
    roster_health_scores: list[float],
) -> float:
    """Injury risk weighted by player value [0, 1]. Higher = more exposed.

    Risk = 1 - sum(tv_i * hs_i) / sum(tv_i) for top-10 players.
    """
    if not roster_trade_values or not roster_health_scores:
        return 0.0

    # Use top-10 by trade value
    paired = sorted(
        zip(roster_trade_values, roster_health_scores),
        key=lambda x: x[0],
        reverse=True,
    )[:10]

    tv_sum = sum(tv for tv, _ in paired)
    if tv_sum <= 0:
        return 0.0

    weighted_health = sum(tv * hs for tv, hs in paired)
    return max(0.0, min(1.0, 1.0 - weighted_health / tv_sum))


def compute_momentum(
    recent_win_rate: float | None = None,
    season_win_rate: float | None = None,
) -> float:
    """Momentum = recent/season win rate, clipped [0.5, 2.0]. Default 1.0."""
    if recent_win_rate is None or season_win_rate is None:
        return 1.0
    if season_win_rate <= 0:
        return 1.0
    ratio = recent_win_rate / season_win_rate
    return max(0.5, min(2.0, ratio))


def compute_power_rating(
    roster_quality: float,
    category_balance: float,
    schedule_strength: float | None = None,
    injury_exposure: float | None = None,
    momentum: float | None = None,
) -> float:
    """Composite power rating [0, 100].

    When components are None (data unavailable), re-weights the remaining
    components so the rating is based only on real data — not fabricated
    defaults that silently make up 35% of the score.

    100 * (w_rq*roster + w_cb*balance + w_ss*sos + w_ie*(1-risk) + w_m*momentum_norm)
    """
    # Build dict of (weight_key -> normalized_value) for available components
    components: dict[str, float] = {
        "roster_quality": roster_quality,
        "category_balance": category_balance,
    }
    if schedule_strength is not None:
        components["schedule_strength"] = schedule_strength
    if injury_exposure is not None:
        components["injury_exposure"] = 1.0 - injury_exposure
    if momentum is not None:
        # Normalize momentum from [0.5, 2.0] to [0, 1]
        components["momentum"] = (momentum - 0.5) / 1.5

    # Re-normalize weights for available components
    total_weight = sum(WEIGHTS[k] for k in components)
    if total_weight <= 0:
        return 0.0

    rating = sum(components[k] * WEIGHTS[k] / total_weight for k in components)
    return round(100 * rating, 1)


def compute_power_rankings(
    team_data: list[dict],
) -> pd.DataFrame:
    """Compute power rankings for all teams.

    Args:
        team_data: List of dicts with keys:
            team_name, roster_quality, category_balance,
            schedule_strength, injury_exposure, momentum

    Returns:
        DataFrame sorted by power_rating descending with rank column.
    """
    results = []
    for td in team_data:
        ss = td.get("schedule_strength")
        ie = td.get("injury_exposure")
        mom = td.get("momentum")
        rating = compute_power_rating(
            roster_quality=td.get("roster_quality", 0.5),
            category_balance=td.get("category_balance", 0.5),
            schedule_strength=ss,
            injury_exposure=ie,
            momentum=mom,
        )
        results.append(
            {
                "team_name": td["team_name"],
                "power_rating": rating,
                "roster_quality": td.get("roster_quality", 0.5),
                "category_balance": td.get("category_balance", 0.5),
                "schedule_strength": ss,
                "injury_exposure": ie,
                "momentum": mom,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def bootstrap_confidence_interval(
    team_rating: float,
    component_stds: dict[str, float] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 90% CI for a team's power rating.

    Resamples component values using provided standard deviations.

    Returns:
        (p5, p95) tuple
    """
    rng = np.random.default_rng(seed)

    # Compute combined uncertainty from component stds if provided,
    # otherwise use a default proportional to the rating
    if component_stds:
        stds = component_stds
        # Combine component uncertainties weighted by their WEIGHTS
        combined_var = sum((WEIGHTS.get(k, 0.1) * stds.get(k, 0.05)) ** 2 for k in WEIGHTS)
        rating_std = max(0.5, 100.0 * (combined_var**0.5))
    else:
        rating_std = max(1.0, team_rating * 0.1)

    # Perturb the final rating directly
    ratings = []
    for _ in range(n_bootstrap):
        perturbed = rng.normal(team_rating, rating_std)
        ratings.append(max(0, min(100, perturbed)))

    return round(float(np.percentile(ratings, 5)), 1), round(float(np.percentile(ratings, 95)), 1)
