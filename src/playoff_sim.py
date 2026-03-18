"""Playoff Odds Simulator: MC simulation of remaining H2H season."""

from __future__ import annotations

import itertools
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig

try:
    from src.optimizer.h2h_engine import default_category_variances

    _HAS_H2H = True
except ImportError:
    _HAS_H2H = False

logger = logging.getLogger(__name__)

# ── Default weekly standard deviations ──────────────────────────────
# Used when the h2h_engine module is unavailable.  These represent
# typical weekly fluctuation for a single 12-team H2H roster.

_DEFAULT_WEEKLY_SIGMAS: dict[str, float] = {
    "R": 8.5,
    "HR": 3.2,
    "RBI": 8.0,
    "SB": 2.8,
    "AVG": 0.015,
    "OBP": 0.014,
    "W": 1.6,
    "L": 1.5,
    "SV": 2.4,
    "K": 12.0,
    "ERA": 0.75,
    "WHIP": 0.06,
}

# Reasonable clip bounds for sampled rate stats
_RATE_CLIP: dict[str, tuple[float, float]] = {
    "AVG": (0.100, 0.400),
    "OBP": (0.120, 0.500),
    "ERA": (0.50, 12.00),
    "WHIP": (0.50, 2.50),
}

# Number of playoff spots (top half of league)
_PLAYOFF_SPOTS = 6


def _build_weekly_sigmas(config: LeagueConfig | None = None) -> dict[str, float]:
    """Return per-category weekly standard deviations.

    Prefers h2h_engine variances (converting variance -> sigma) when
    available, otherwise falls back to hardcoded defaults.
    """
    cfg = config or LeagueConfig()

    if _HAS_H2H:
        variances = default_category_variances()
        sigmas: dict[str, float] = {}
        for cat in cfg.all_categories:
            key = cfg.STAT_MAP.get(cat, cat.lower())
            var = variances.get(key, 1.0)
            sigmas[cat] = float(np.sqrt(max(var, 0.0)))
        return sigmas

    # Fallback
    return {cat: _DEFAULT_WEEKLY_SIGMAS.get(cat, 1.0) for cat in cfg.all_categories}


# ── Schedule generation ─────────────────────────────────────────────


def generate_schedule(
    team_names: list[str],
    weeks_remaining: int,
) -> list[list[tuple[str, str]]]:
    """Generate a round-robin H2H schedule for the remaining weeks.

    Each week every team plays exactly one opponent.  The round-robin
    cycle repeats as needed to fill the requested number of weeks.

    Args:
        team_names: List of team name strings (length should be even).
        weeks_remaining: How many weeks of matchups to generate.

    Returns:
        A list of weeks, where each week is a list of (team_a, team_b)
        matchup tuples.
    """
    n = len(team_names)
    if n < 2:
        return []

    teams = list(team_names)
    # If odd number of teams, add a BYE placeholder
    if n % 2 == 1:
        teams.append("__BYE__")
        n += 1

    # Standard round-robin: fix team 0, rotate the rest
    # Uses the "circle method" — team 0 stays, others rotate clockwise
    rounds: list[list[tuple[str, str]]] = []
    rotation = list(range(1, n))  # indices of teams 1..n-1
    for _ in range(n - 1):
        week_matchups: list[tuple[str, str]] = []
        # Team 0 plays the team at the "top" of the rotation
        week_matchups.append((teams[0], teams[rotation[0]]))
        # Pair remaining from outside in
        for j in range(1, n // 2):
            a_idx = rotation[j]
            b_idx = rotation[n - 1 - j]
            week_matchups.append((teams[a_idx], teams[b_idx]))
        rounds.append(week_matchups)
        # Rotate: shift right by one
        rotation = [rotation[-1]] + rotation[:-1]

    # Repeat the round-robin cycle to fill weeks_remaining
    schedule: list[list[tuple[str, str]]] = []
    cycle = itertools.cycle(rounds)
    for _, week in zip(range(weeks_remaining), cycle):
        # Filter out BYE matchups
        real = [(a, b) for a, b in week if a != "__BYE__" and b != "__BYE__"]
        schedule.append(real)

    return schedule


# ── Weekly projection ────────────────────────────────────────────────


def project_weekly_totals(
    team_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict[str, float]:
    """Project a team's weekly category totals from ROS projections.

    Sums counting stats across the roster then divides by an estimated
    number of remaining game-weeks (~22 weeks for a full season) to
    derive a per-week rate.  Rate stats are computed directly from
    component columns (H/AB, ER*9/IP, etc.).

    Args:
        team_roster_ids: Player IDs on the team's roster.
        player_pool: Full player pool DataFrame (with stat columns).
        config: League configuration (defaults to standard 12-team).

    Returns:
        Dict mapping each uppercase category name to a weekly projected value.
    """
    cfg = config or LeagueConfig()

    roster = player_pool[player_pool["player_id"].isin(team_roster_ids)]
    if roster.empty:
        return {cat: 0.0 for cat in cfg.all_categories}

    # Approximate remaining game-weeks for per-week rates
    # Full season ~ 26 weeks; default to 22 for partway through
    season_weeks = 22.0

    hitters = roster[roster["is_hitter"] == 1]
    pitchers = roster[roster["is_hitter"] == 0]

    totals: dict[str, float] = {}

    # Counting stats — sum then divide by season_weeks
    counting_map = {
        "R": ("r", hitters),
        "HR": ("hr", hitters),
        "RBI": ("rbi", hitters),
        "SB": ("sb", hitters),
        "W": ("w", pitchers),
        "L": ("l", pitchers),
        "SV": ("sv", pitchers),
        "K": ("k", pitchers),
    }
    for cat, (col, subset) in counting_map.items():
        if col in subset.columns:
            totals[cat] = float(subset[col].sum()) / season_weeks
        else:
            totals[cat] = 0.0

    # Rate stats — compute from components
    total_h = float(hitters["h"].sum()) if "h" in hitters.columns else 0.0
    total_ab = float(hitters["ab"].sum()) if "ab" in hitters.columns else 0.0
    total_bb = float(hitters["bb"].sum()) if "bb" in hitters.columns else 0.0
    total_hbp = float(hitters["hbp"].sum()) if "hbp" in hitters.columns else 0.0
    total_sf = float(hitters["sf"].sum()) if "sf" in hitters.columns else 0.0

    totals["AVG"] = total_h / total_ab if total_ab > 0 else 0.250
    obp_num = total_h + total_bb + total_hbp
    obp_den = total_ab + total_bb + total_hbp + total_sf
    totals["OBP"] = obp_num / obp_den if obp_den > 0 else 0.320

    total_ip = float(pitchers["ip"].sum()) if "ip" in pitchers.columns else 0.0
    total_er = float(pitchers["er"].sum()) if "er" in pitchers.columns else 0.0
    total_bb_p = float(pitchers["bb_allowed"].sum()) if "bb_allowed" in pitchers.columns else 0.0
    total_h_p = float(pitchers["h_allowed"].sum()) if "h_allowed" in pitchers.columns else 0.0

    totals["ERA"] = (total_er * 9.0 / total_ip) if total_ip > 0 else 4.00
    totals["WHIP"] = ((total_bb_p + total_h_p) / total_ip) if total_ip > 0 else 1.25

    return totals


# ── Single matchup simulation ────────────────────────────────────────


def simulate_h2h_matchup(
    team_a_weekly: dict[str, float],
    team_b_weekly: dict[str, float],
    sigmas: dict[str, float],
    rng: np.random.Generator,
    config: LeagueConfig | None = None,
) -> tuple[int, int]:
    """Simulate one H2H category matchup between two teams.

    For each category, sample each team's actual weekly output from
    Normal(projected, sigma).  The team with the better value wins the
    category (lower is better for inverse stats).

    Args:
        team_a_weekly: Team A projected weekly totals per category.
        team_b_weekly: Team B projected weekly totals per category.
        sigmas: Per-category weekly standard deviations.
        rng: NumPy random Generator for reproducibility.
        config: League configuration.

    Returns:
        (team_a_cat_wins, team_b_cat_wins) tuple.
    """
    cfg = config or LeagueConfig()
    inverse = cfg.inverse_stats
    a_wins = 0
    b_wins = 0

    for cat in cfg.all_categories:
        sigma = sigmas.get(cat, 1.0)
        a_val = rng.normal(team_a_weekly.get(cat, 0.0), sigma)
        b_val = rng.normal(team_b_weekly.get(cat, 0.0), sigma)

        # Clip rate stats to reasonable bounds
        clip = _RATE_CLIP.get(cat)
        if clip is not None:
            a_val = np.clip(a_val, clip[0], clip[1])
            b_val = np.clip(b_val, clip[0], clip[1])

        if cat in inverse:
            # Lower is better
            if a_val < b_val:
                a_wins += 1
            elif b_val < a_val:
                b_wins += 1
            # Ties: no one gets the point
        else:
            if a_val > b_val:
                a_wins += 1
            elif b_val > a_val:
                b_wins += 1

    return a_wins, b_wins


# ── Main simulator ──────────────────────────────────────────────────


def simulate_season(
    all_team_totals: dict[str, dict[str, float]],
    league_rosters: dict[str, list[int]],
    player_pool: pd.DataFrame,
    weeks_remaining: int = 16,
    n_sims: int = 500,
    config: LeagueConfig | None = None,
    seed: int = 42,
    on_progress: callable | None = None,
) -> dict[str, dict]:
    """Run Monte Carlo simulation of the remaining H2H season.

    For each simulation, generates a round-robin schedule, simulates
    every weekly matchup by sampling from Normal distributions around
    each team's projected weekly output, then accumulates W-L records
    to determine final standings and playoff probability.

    Args:
        all_team_totals: Dict of {team_name: {category: season_total}}.
            Used as baseline context; actual weekly projections come
            from the player pool.
        league_rosters: Dict of {team_name: [player_id, ...]}.
        player_pool: Full player pool DataFrame with projection columns.
        weeks_remaining: Number of weeks left in the fantasy season.
        n_sims: Number of Monte Carlo simulations to run.
        config: League configuration.
        seed: RNG seed for reproducibility.
        on_progress: Optional callback(fraction: float) for progress updates.

    Returns:
        Dict mapping team_name to:
            avg_wins: float — average wins across sims
            avg_losses: float — average losses across sims
            playoff_prob: float — fraction of sims finishing top 6
            rank_distribution: list[int] — count at each rank (index 0 = rank 1)
    """
    cfg = config or LeagueConfig()
    rng = np.random.default_rng(seed)
    sigmas = _build_weekly_sigmas(cfg)

    team_names = sorted(league_rosters.keys())
    n_teams = len(team_names)
    if n_teams < 2:
        logger.warning("Need at least 2 teams for simulation, got %d", n_teams)
        return {}

    weeks_remaining = max(weeks_remaining, 1)
    playoff_spots = min(_PLAYOFF_SPOTS, n_teams // 2)

    # Pre-compute weekly projections for each team
    weekly_proj: dict[str, dict[str, float]] = {}
    for team in team_names:
        ids = league_rosters[team]
        weekly_proj[team] = project_weekly_totals(ids, player_pool, cfg)

    # Generate one schedule template (repeats across sims)
    schedule = generate_schedule(team_names, weeks_remaining)

    # Accumulators
    total_wins = {t: 0.0 for t in team_names}
    total_losses = {t: 0.0 for t in team_names}
    rank_counts = {t: [0] * n_teams for t in team_names}

    for sim_idx in range(n_sims):
        # Per-sim win/loss accumulators
        sim_wins = {t: 0 for t in team_names}
        sim_losses = {t: 0 for t in team_names}

        for week_matchups in schedule:
            for team_a, team_b in week_matchups:
                a_cats, b_cats = simulate_h2h_matchup(
                    weekly_proj[team_a],
                    weekly_proj[team_b],
                    sigmas,
                    rng,
                    cfg,
                )
                # Determine matchup winner (majority of 12 categories)
                if a_cats > b_cats:
                    sim_wins[team_a] += 1
                    sim_losses[team_b] += 1
                elif b_cats > a_cats:
                    sim_wins[team_b] += 1
                    sim_losses[team_a] += 1
                else:
                    # Tie — both get 0.5 W and 0.5 L
                    sim_wins[team_a] += 0.5
                    sim_losses[team_a] += 0.5
                    sim_wins[team_b] += 0.5
                    sim_losses[team_b] += 0.5

        # Rank teams by wins (desc), ties broken randomly
        sorted_teams = sorted(
            team_names,
            key=lambda t: (sim_wins[t], rng.random()),
            reverse=True,
        )

        for rank_idx, team in enumerate(sorted_teams):
            total_wins[team] += sim_wins[team]
            total_losses[team] += sim_losses[team]
            rank_counts[team][rank_idx] += 1

        if on_progress is not None:
            on_progress((sim_idx + 1) / n_sims)

    # Build results
    results: dict[str, dict] = {}
    for team in team_names:
        avg_w = total_wins[team] / n_sims
        avg_l = total_losses[team] / n_sims
        playoff_count = sum(rank_counts[team][:playoff_spots])
        results[team] = {
            "avg_wins": round(avg_w, 1),
            "avg_losses": round(avg_l, 1),
            "playoff_prob": round(playoff_count / n_sims, 3),
            "rank_distribution": rank_counts[team],
        }

    return results


def estimate_weeks_remaining(season_end_month: int = 9, season_end_day: int = 21) -> int:
    """Estimate weeks remaining in the fantasy season from today's date.

    Args:
        season_end_month: Month the fantasy season ends (default September).
        season_end_day: Day the fantasy season ends (default 21st).

    Returns:
        Weeks remaining, clamped to [1, 26].
    """
    now = datetime.now(UTC)
    end = datetime(now.year, season_end_month, season_end_day, tzinfo=UTC)
    if now >= end:
        return 1
    delta = end - now
    weeks = max(1, delta.days // 7)
    return min(weeks, 26)
