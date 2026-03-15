"""Pitcher streaming optimisation and two-start pitcher valuation.

Provides marginal SGP calculations for streaming pitchers (one-start
pickups) and for deciding whether a two-start pitcher's second start
helps or hurts a roto team.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Default SGP Denominators ─────────────────────────────────────────

DEFAULT_SGP_DENOMS: dict[str, float] = {
    "r": 20.0,
    "hr": 7.0,
    "rbi": 20.0,
    "sb": 5.0,
    "avg": 0.005,
    "w": 3.0,
    "sv": 5.0,
    "k": 25.0,
    "era": 0.30,
    "whip": 0.03,
}

# Default IP per start for a typical starting pitcher.
_DEFAULT_IP_PER_START = 5.5

# Default team total IP per week used for rate-stat dilution.
_DEFAULT_TEAM_WEEKLY_IP = 55.0


# ── Helpers ──────────────────────────────────────────────────────────


def _get(obj: Any, key: str, default: float = 0.0) -> float:
    """Extract a numeric value from a dict, Series, or object."""
    if isinstance(obj, dict):
        return float(obj.get(key, default))
    # pandas Series / DataFrame row
    try:
        return float(obj[key])
    except (KeyError, TypeError, IndexError):
        return default


def _safe_denoms(sgp_denominators: dict[str, float] | None) -> dict[str, float]:
    """Return SGP denominators, falling back to defaults."""
    if sgp_denominators:
        return sgp_denominators
    return dict(DEFAULT_SGP_DENOMS)


def _safe_weights(category_weights: dict[str, float] | None) -> dict[str, float]:
    """Return category weights, falling back to equal weights."""
    if category_weights:
        return category_weights
    return {cat: 1.0 for cat in DEFAULT_SGP_DENOMS}


# ── Streaming Value ──────────────────────────────────────────────────


def compute_streaming_value(
    pitcher: Any,
    weekly_games: int = 1,
    team_park_factor: float = 1.0,
    category_weights: dict[str, float] | None = None,
    sgp_denominators: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute expected marginal SGP from streaming a pitcher for one start.

    Counting SGP comes from strikeouts and win probability.  Rate damage
    comes from the expected ERA impact weighted by IP per start.

    Args:
        pitcher: Dict-like with keys ``k``, ``w``, ``era``, ``whip``,
            ``ip``.  Per-game averages or full-season totals both work;
            the function normalises to per-start values.
        weekly_games: Number of starts this pitcher has in the week.
        team_park_factor: Park factor for the game venue (>1.0 = hitter
            friendly = worse for pitchers).
        category_weights: Optional per-category SGP weight multipliers.
        sgp_denominators: Optional per-category SGP denominators.

    Returns:
        dict with keys: ``counting_sgp``, ``rate_impact``, ``net_value``,
        ``k_per_start``, ``era_per_start``.
    """
    denoms = _safe_denoms(sgp_denominators)
    weights = _safe_weights(category_weights)

    # Extract per-start stats
    ip = _get(pitcher, "ip", _DEFAULT_IP_PER_START)
    total_k = _get(pitcher, "k", 0.0)
    total_w = _get(pitcher, "w", 0.0)
    era = _get(pitcher, "era", 4.50)
    whip = _get(pitcher, "whip", 1.30)

    # Normalise to per-start if season totals are large
    if ip > 30:
        # Looks like season totals -- compute per-start rates
        games_started = max(ip / _DEFAULT_IP_PER_START, 1.0)
        k_per_start = total_k / games_started
        w_per_start = total_w / games_started
        ip_per_start = _DEFAULT_IP_PER_START
    else:
        k_per_start = total_k
        w_per_start = total_w
        ip_per_start = ip if ip > 0 else _DEFAULT_IP_PER_START

    # Apply park factor to ERA (hitter-friendly parks inflate ERA)
    era_adjusted = era * team_park_factor

    # Counting SGP contribution
    sgp_k = denoms.get("k", 25.0)
    sgp_w = denoms.get("w", 3.0)
    w_k = weights.get("k", 1.0)
    w_w = weights.get("w", 1.0)

    counting_sgp = (k_per_start / sgp_k) * w_k + (w_per_start / sgp_w) * w_w
    counting_sgp *= weekly_games

    # Rate impact: how much this pitcher's ERA/WHIP shifts team totals.
    # Positive rate_impact means the pitcher HELPS (lowers team ERA/WHIP).
    # We model the dilution effect on team totals.
    team_ip = _DEFAULT_TEAM_WEEKLY_IP
    ip_contribution = ip_per_start * weekly_games

    sgp_era = denoms.get("era", 0.30)
    sgp_whip = denoms.get("whip", 0.03)
    w_era = weights.get("era", 1.0)
    w_whip = weights.get("whip", 1.0)

    # ERA impact: negative ERA delta = good (lower team ERA)
    # A league-average ERA is ~4.00; improvement relative to that
    baseline_era = 4.00
    era_delta = (baseline_era - era_adjusted) * ip_contribution / (team_ip + ip_contribution)
    era_sgp = (era_delta / sgp_era) * w_era

    baseline_whip = 1.25
    whip_delta = (baseline_whip - whip) * ip_contribution / (team_ip + ip_contribution)
    whip_sgp = (whip_delta / sgp_whip) * w_whip

    rate_impact = era_sgp + whip_sgp

    net_value = counting_sgp + rate_impact

    return {
        "counting_sgp": round(counting_sgp, 4),
        "rate_impact": round(rate_impact, 4),
        "net_value": round(net_value, 4),
        "k_per_start": round(k_per_start, 2),
        "era_per_start": round(era_adjusted, 2),
    }


# ── Streaming Candidate Ranking ─────────────────────────────────────


def rank_streaming_candidates(
    free_agent_pitchers: list[dict],
    weekly_schedule: dict[str, Any] | None = None,
    park_factors: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
    sgp_denominators: dict[str, float] | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Rank free-agent pitchers by streaming value for the upcoming week.

    Args:
        free_agent_pitchers: List of pitcher dicts with at least
            ``player_name``, ``team``, ``k``, ``w``, ``era``, ``whip``,
            ``ip``.  Optionally ``opponent`` and ``weekly_games``.
        weekly_schedule: Optional schedule info (currently unused;
            reserved for future matchup integration).
        park_factors: Optional dict mapping team codes to park factors.
        category_weights: Optional per-category weight multipliers.
        sgp_denominators: Optional per-category SGP denominators.
        max_results: Maximum number of candidates to return.

    Returns:
        List of dicts sorted by ``net_value`` descending, each with:
        ``player_name``, ``team``, ``opponent``, ``net_value``,
        ``counting_sgp``, ``rate_impact``.
    """
    if not free_agent_pitchers:
        return []

    pf = park_factors or {}
    results = []

    for pitcher in free_agent_pitchers:
        name = pitcher.get("player_name", pitcher.get("name", "Unknown"))
        team = pitcher.get("team", "")
        opponent = pitcher.get("opponent", "")
        weekly_games = int(pitcher.get("weekly_games", 1))

        # Look up park factor for the opponent's park (if available)
        park = pf.get(opponent, pf.get(team, 1.0))

        sv = compute_streaming_value(
            pitcher,
            weekly_games=weekly_games,
            team_park_factor=park,
            category_weights=category_weights,
            sgp_denominators=sgp_denominators,
        )

        results.append(
            {
                "player_name": name,
                "team": team,
                "opponent": opponent,
                "net_value": sv["net_value"],
                "counting_sgp": sv["counting_sgp"],
                "rate_impact": sv["rate_impact"],
            }
        )

    # Sort by net_value descending
    results.sort(key=lambda x: x["net_value"], reverse=True)

    return results[:max_results]


# ── Two-Start Pitcher Valuation ──────────────────────────────────────


def quantify_two_start_value(
    pitcher_stats: Any,
    team_era: float = 4.00,
    team_whip: float = 1.25,
    category_weights: dict[str, float] | None = None,
    sgp_denominators: dict[str, float] | None = None,
) -> dict[str, float | str]:
    """Compute the marginal SGP of a pitcher's SECOND start.

    The second start always adds counting stats (K, W) at the same
    per-game rate -- always positive.

    For rate stats (ERA, WHIP), the second start *dilutes* the team
    total:

    - If pitcher ERA < team ERA: 2nd start **helps** (positive).
    - If pitcher ERA > team ERA: 2nd start **hurts** (negative).

    Formula::

        rate_impact_era = (team_era - pitcher_era) * ip_per_start
                          / (total_team_ip + ip_per_start)

    Args:
        pitcher_stats: Dict-like with ``k``, ``w``, ``era``, ``whip``,
            ``ip`` (per-game averages preferred).
        team_era: The team's current ERA.
        team_whip: The team's current WHIP.
        category_weights: Optional per-category weight multipliers.
        sgp_denominators: Optional per-category SGP denominators.

    Returns:
        dict with ``counting_sgp``, ``rate_impact``, ``net_value``,
        ``recommendation`` ("Start" / "Sit" / "Close call").
    """
    denoms = _safe_denoms(sgp_denominators)
    weights = _safe_weights(category_weights)

    ip = _get(pitcher_stats, "ip", _DEFAULT_IP_PER_START)
    total_k = _get(pitcher_stats, "k", 0.0)
    total_w = _get(pitcher_stats, "w", 0.0)
    era = _get(pitcher_stats, "era", 4.50)
    whip = _get(pitcher_stats, "whip", 1.30)

    # Normalise to per-start
    if ip > 30:
        games_started = max(ip / _DEFAULT_IP_PER_START, 1.0)
        k_per_start = total_k / games_started
        w_per_start = total_w / games_started
        ip_per_start = _DEFAULT_IP_PER_START
    else:
        k_per_start = total_k
        w_per_start = total_w
        ip_per_start = ip if ip > 0 else _DEFAULT_IP_PER_START

    # Counting SGP from the second start (same per-game rate)
    sgp_k = denoms.get("k", 25.0)
    sgp_w = denoms.get("w", 3.0)
    w_k = weights.get("k", 1.0)
    w_w = weights.get("w", 1.0)

    counting_sgp = (k_per_start / sgp_k) * w_k + (w_per_start / sgp_w) * w_w

    # Rate impact of the second start
    team_ip = _DEFAULT_TEAM_WEEKLY_IP

    sgp_era = denoms.get("era", 0.30)
    sgp_whip = denoms.get("whip", 0.03)
    w_era = weights.get("era", 1.0)
    w_whip = weights.get("whip", 1.0)

    era_delta = (team_era - era) * ip_per_start / (team_ip + ip_per_start)
    era_sgp = (era_delta / sgp_era) * w_era

    whip_delta = (team_whip - whip) * ip_per_start / (team_ip + ip_per_start)
    whip_sgp = (whip_delta / sgp_whip) * w_whip

    rate_impact = era_sgp + whip_sgp

    net_value = counting_sgp + rate_impact

    # Recommendation
    threshold = 0.02  # Close call band
    if net_value > threshold:
        recommendation = "Start"
    elif net_value < -threshold:
        recommendation = "Sit"
    else:
        recommendation = "Close call"

    return {
        "counting_sgp": round(counting_sgp, 4),
        "rate_impact": round(rate_impact, 4),
        "net_value": round(net_value, 4),
        "recommendation": recommendation,
    }


# ── Optimal Streaming Schedule ───────────────────────────────────────


def optimal_streaming_schedule(
    candidates: list[dict],
    max_adds: int = 7,
) -> list[dict]:
    """Select the optimal subset of streaming candidates.

    Uses a greedy approach: take the best candidate for each available
    slot within the weekly transaction limit.

    Args:
        candidates: Ranked list of candidate dicts (as returned by
            :func:`rank_streaming_candidates`).  Must include at least
            ``net_value``.
        max_adds: Maximum number of add/drop transactions allowed
            per week.

    Returns:
        Ordered list of recommended pickups (up to *max_adds* entries),
        each a copy of the candidate dict.
    """
    if not candidates:
        return []

    # Sort by net_value descending (in case caller didn't pre-sort)
    sorted_cands = sorted(candidates, key=lambda c: c.get("net_value", 0), reverse=True)

    schedule: list[dict] = []
    seen_players: set[str] = set()

    for cand in sorted_cands:
        if len(schedule) >= max_adds:
            break

        name = cand.get("player_name", cand.get("name", ""))
        if name in seen_players:
            continue

        # Only pick up pitchers with positive expected value
        if cand.get("net_value", 0) <= 0:
            break

        seen_players.add(name)
        schedule.append(dict(cand))

    return schedule
