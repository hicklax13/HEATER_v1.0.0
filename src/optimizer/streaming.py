"""Pitcher streaming optimisation and two-start pitcher valuation.

Provides marginal SGP calculations for streaming pitchers (one-start
pickups) and for deciding whether a two-start pitcher's second start
helps or hurts the team's season-long totals.
"""

from __future__ import annotations

import logging
from typing import Any

from src.validation.constant_optimizer import load_constants

logger = logging.getLogger(__name__)

_CONSTANTS = load_constants()

# ── Default SGP Denominators ─────────────────────────────────────────

DEFAULT_SGP_DENOMS: dict[str, float] = {
    "r": 20.0,
    "hr": 7.0,
    "rbi": 20.0,
    "sb": 5.0,
    "avg": 0.005,
    "obp": 0.005,
    "w": 3.0,
    "l": 3.0,
    "sv": 5.0,
    "k": 25.0,
    "era": 0.30,
    "whip": 0.03,
}

# Weekly transaction budget for streaming adds.
WEEKLY_ADDS_BUDGET: int = 7

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
    if ip > 15:
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
    sgp_l = denoms.get("l", 3.0)
    w_k = weights.get("k", 1.0)
    w_w = weights.get("w", 1.0)
    w_l = weights.get("l", 1.0)

    # L (losses) is inverse: adding losses hurts, so subtract L contribution
    total_l = _get(pitcher, "l", 0.0)
    if ip > 15:
        l_per_start = total_l / max(ip / _DEFAULT_IP_PER_START, 1.0)
    else:
        l_per_start = total_l

    counting_sgp = (k_per_start / sgp_k) * w_k + (w_per_start / sgp_w) * w_w - (l_per_start / sgp_l) * w_l
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
    baseline_era = _CONSTANTS.get("streaming_baseline_era")
    era_delta = (baseline_era - era_adjusted) * ip_contribution / (team_ip + ip_contribution)
    era_sgp = (era_delta / sgp_era) * w_era

    baseline_whip = 1.25
    whip_adjusted = whip * team_park_factor
    whip_delta = (baseline_whip - whip_adjusted) * ip_contribution / (team_ip + ip_contribution)
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


def compute_streaming_composite(
    pitcher: Any,
    opp_woba: float = 0.320,
    park_factor: float = 1.0,
    recent_form_era: float | None = None,
    career_whip: float | None = None,
    weekly_games: int = 1,
    category_weights: dict[str, float] | None = None,
    sgp_denominators: dict[str, float] | None = None,
) -> dict[str, float]:
    """K5: Enhanced streaming composite score incorporating opponent quality,
    recent form, and WHIP safety gate.

    Formula: K_proj * (1/opp_wOBA) * park * form_L3 * whip_safety
    Career WHIP >1.40 = avoid (ratio destruction risk).

    Args:
        pitcher: Dict-like with k, w, l, era, whip, ip.
        opp_woba: Opponent team wOBA (lower = tougher, default league avg).
        park_factor: Venue park factor (>1.0 = hitter friendly).
        recent_form_era: Pitcher's L3 start ERA (None = use projected).
        career_whip: Career WHIP for safety check (None = skip).
        weekly_games: Number of starts this week.
        category_weights: Per-category SGP weights.
        sgp_denominators: Per-category SGP denominators.

    Returns:
        dict with composite_score, base_value, opp_mult, form_mult,
        whip_safe, and the underlying net_value from compute_streaming_value.
    """
    # Base streaming value
    base = compute_streaming_value(
        pitcher, weekly_games, park_factor, category_weights, sgp_denominators
    )
    net_value = base["net_value"]

    # Opponent quality multiplier: weaker opponents = better streaming target
    # League avg wOBA ~0.320. Inverse scaling: 0.280 opp → 1.14x, 0.360 → 0.89x
    opp_mult = min(1.3, max(0.7, 0.320 / max(opp_woba, 0.200)))

    # Recent form multiplier: L3 starts ERA relative to season projection
    form_mult = 1.0
    proj_era = _get(pitcher, "era", 4.50)
    if recent_form_era is not None and proj_era > 0:
        # Good recent form (lower ERA than projected) = bonus
        form_ratio = proj_era / max(recent_form_era, 1.0)
        form_mult = min(1.2, max(0.8, form_ratio))

    # WHIP safety gate: career WHIP > 1.40 = high ratio destruction risk
    whip_safe = True
    whip_penalty = 1.0
    if career_whip is not None and career_whip > 1.40:
        whip_safe = False
        whip_penalty = 0.5  # Halve value for WHIP-risky pitchers

    composite = net_value * opp_mult * form_mult * whip_penalty

    return {
        "composite_score": round(composite, 4),
        "base_value": round(net_value, 4),
        "opp_mult": round(opp_mult, 3),
        "form_mult": round(form_mult, 3),
        "whip_safe": whip_safe,
        "k_per_start": base["k_per_start"],
        "era_per_start": base["era_per_start"],
    }


# ── Streaming Candidate Ranking ─────────────────────────────────────


def rank_streaming_candidates(
    free_agent_pitchers: list[dict],
    weekly_schedule: dict[str, Any] | None = None,
    park_factors: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
    sgp_denominators: dict[str, float] | None = None,
    max_results: int = 10,
    adds_used_this_week: int = 0,
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
        adds_used_this_week: Number of add/drop transactions already
            used this week.  When only 1 add remains, candidate values
            are halved to discourage using the last add on a streamer.
            When the budget is exhausted, returns an empty list.

    Returns:
        List of dicts sorted by ``net_value`` descending, each with:
        ``player_name``, ``team``, ``opponent``, ``net_value``,
        ``counting_sgp``, ``rate_impact``.
    """
    if not free_agent_pitchers:
        return []

    # Budget exhausted — no streaming adds available
    if adds_used_this_week >= WEEKLY_ADDS_BUDGET:
        return []

    pf = park_factors or {}
    results = []

    for pitcher in free_agent_pitchers:
        name = pitcher.get("player_name", pitcher.get("name", "Unknown"))
        team = pitcher.get("team", "")
        opponent = pitcher.get("opponent", "")
        weekly_games = int(pitcher.get("weekly_games", 1))
        if weekly_games <= 0:
            continue

        # Look up park factor for the opponent's park (if available)
        park = pf.get(opponent, pf.get(team, 1.0))

        sv = compute_streaming_value(
            pitcher,
            weekly_games=weekly_games,
            team_park_factor=park,
            category_weights=category_weights,
            sgp_denominators=sgp_denominators,
        )

        net_value = sv["net_value"]

        # Penalize when only 1 add remains to discourage using it on a streamer
        if adds_used_this_week >= WEEKLY_ADDS_BUDGET - 1:
            net_value *= 0.5

        results.append(
            {
                "player_name": name,
                "team": team,
                "opponent": opponent,
                "net_value": round(net_value, 4),
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
    sgp_l = denoms.get("l", 3.0)
    w_k = weights.get("k", 1.0)
    w_w = weights.get("w", 1.0)
    w_l = weights.get("l", 1.0)

    # L (losses) from second start — inverse stat, subtract contribution
    total_l = _get(pitcher_stats, "l", 0.0)
    if ip > 30:
        l_per_start = total_l / max(ip / _DEFAULT_IP_PER_START, 1.0)
    else:
        l_per_start = total_l

    counting_sgp = (k_per_start / sgp_k) * w_k + (w_per_start / sgp_w) * w_w - (l_per_start / sgp_l) * w_l

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


# --- Bayesian Stream Scoring ---

_STREAM_SGP_DEFAULTS: dict[str, float] = {
    "k": 28.0,
    "w": 2.5,
    "era": 0.27,
}


def compute_bayesian_stream_score(
    pitcher_era: float,
    pitcher_k9: float,
    pitcher_fip: float,
    opp_k_pct: float = 0.225,
    opp_woba: float = 0.320,
    is_home: bool = False,
    sgp_denominators: dict[str, float] | None = None,
) -> dict:
    """Compute Bayesian streaming score for a single pitcher start.

    Returns dict with stream_score, expected_k, expected_ip, expected_er,
    win_probability, risk_penalty, matchup_grade.
    """
    sgp = sgp_denominators or _STREAM_SGP_DEFAULTS

    # Expected IP: clamped [4.0, 6.5]
    ip_adj = 1.05 if pitcher_fip < 4.0 else 0.95
    expected_ip = min(6.5, max(4.0, 5.5 * ip_adj))

    # Expected K
    expected_k = (pitcher_k9 / 9.0) * expected_ip * (opp_k_pct / 0.225)

    # Expected ER
    expected_er = (pitcher_era / 9.0) * expected_ip * (opp_woba / 0.320)

    # Win probability
    home_adj = 1.04 if is_home else 1.0
    era_factor = min(1.0, (4.50 - pitcher_era) / 4.50)  # Allow negative for ERA > 4.50
    win_prob = min(0.95, max(0.05, 0.5 * (1.0 + era_factor) * home_adj))

    # Risk penalty
    if expected_ip > 0:
        era_implied = (expected_er / expected_ip) * 9.0
        risk_penalty = max(1.0, era_implied / 4.50)
    else:
        risk_penalty = 2.0

    # Stream score (SGP-based)
    sgp_k = sgp.get("k", 28.0)
    sgp_w = sgp.get("w", 2.5)
    sgp_era = sgp.get("era", 0.27)

    # ERA cost: diluted impact on team ERA (ER -> ERA -> SGP conversion)
    baseline_era = _CONSTANTS.get("streaming_baseline_era")
    team_weekly_ip = 55.0
    implied_era = expected_er * 9.0 / max(expected_ip, 1.0)
    era_diluted = (implied_era - baseline_era) * expected_ip / (team_weekly_ip + expected_ip)
    era_cost_sgp = era_diluted / sgp_era * risk_penalty

    score = expected_k / sgp_k + win_prob * 0.5 / sgp_w - era_cost_sgp

    grade = _assign_matchup_grade(score)

    return {
        "stream_score": round(score, 4),
        "expected_k": round(expected_k, 2),
        "expected_ip": round(expected_ip, 2),
        "expected_er": round(expected_er, 2),
        "win_probability": round(win_prob, 4),
        "risk_penalty": round(risk_penalty, 3),
        "matchup_grade": grade,
    }


def _assign_matchup_grade(score: float) -> str:
    """Assign A+/A/B+/B/C grade based on stream score."""
    if score >= 0.35:
        return "A+"
    elif score >= 0.25:
        return "A"
    elif score >= 0.15:
        return "B+"
    elif score >= 0.05:
        return "B"
    else:
        return "C"
