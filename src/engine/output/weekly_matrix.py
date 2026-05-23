"""26-week × 12-category H2H win-probability matrix.

Per report Section B.5: for each remaining matchup week w and category c,
compute

    p_{w,c} = Phi((mu_H - mu_O) / sqrt(sigma_H^2 + sigma_O^2))

where (mu_H, sigma_H) are the user's projected weekly mean and standard
deviation for category c, and (mu_O, sigma_O) are the same for whichever
opponent is scheduled in week w. For inverse cats (L, ERA, WHIP) the
result is flipped (1 - p).

Variance source (per 2026-05-23 design Q3 + design-note in the commit):
the user requested Kalman-filter posterior variance. The existing Kalman
pipeline in src/engine/signals/kalman.py captures TRUE-TALENT uncertainty
("Trout's true AVG is .300 ± .005"), which is the wrong distribution for
PER-WEEK win probability — per-week noise is dominated by sampling
variance ("Trout will hit somewhere between .150 and .450 this week"),
not by talent uncertainty.

This module uses a per-category coefficient-of-variation (CV) approach
calibrated by category-level empirical week-to-week stability:
  - Rate hitting (AVG, OBP): low CV (~4-5%) — stabilize quickly with PA
  - Counting hitting (R, RBI): moderate CV (~15%)
  - Power/speed (HR, SB): higher CV (~20-25%) — Poisson-like
  - Pitching wins/losses: high CV (~30%) — small per-week sample
  - Pitching counting K: moderate CV (~15%)
  - Pitching rates ERA/WHIP: moderate CV (~12-18%) — IP-bounded

Kalman augmentation is a layered enhancement for future work; the CV
approach gives a working weekly matrix today.

Wires into:
  - src/valuation.py: LeagueConfig for cats / inverse_stats / season_weeks
  - src/engine/output/trade_evaluator.py: invoked when enable_weekly_matrix=True

Returns DataFrames indexed by week, columns by category — naturally
renderable as a heatmap by Streamlit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# Per-category coefficient of variation — fraction of mean used as per-week SD.
# Calibrated by category type per the docstring rationale above. These are
# defaults; callers can override per-trade if backtesting suggests different
# empirical values for the league.
_DEFAULT_CV: dict[str, float] = {
    # Hitting rates (stabilize fastest)
    "AVG": 0.05,
    "OBP": 0.04,
    # Counting hitting
    "R": 0.15,
    "RBI": 0.15,
    "HR": 0.20,
    "SB": 0.25,
    # Pitching counting
    "W": 0.30,
    "L": 0.30,
    "SV": 0.35,
    "K": 0.15,
    # Pitching rates
    "ERA": 0.18,
    "WHIP": 0.12,
}

# Floor for sigma to avoid divide-by-zero when projected mean is 0.
# A weekly mean of exactly 0 with CV>0 would give sigma=0, breaking Phi().
# The floor represents irreducible variance — even a 0-projection player
# can pop a HR or steal a base in a given week.
_MIN_SIGMA: float = 0.01


def compute_weekly_matrix(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    schedule: dict[int, str],
    all_team_rosters: dict[str, list[int]],
    config: LeagueConfig | None = None,
    variance_cv: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the 26 × 12 weekly win-probability matrix for one roster.

    Args:
        user_roster_ids: Hickey's player IDs (post-trade if evaluating a trade).
        player_pool: Full enriched player pool DataFrame.
        schedule: Mapping {week_number: opponent_team_name} for each remaining
            matchup week. Caller is responsible for sourcing this from Yahoo
            (typically via yds.get_schedule() or similar).
        all_team_rosters: Mapping {team_name: [player_ids]} for all 12 teams.
            Used to compute opponent per-week means.
        config: LeagueConfig instance (defaults to LeagueConfig()).
        variance_cv: Optional per-category CV override. Defaults to
            _DEFAULT_CV calibrated per the docstring rationale.

    Returns:
        Dict with keys:
          - matrix: pd.DataFrame indexed by week, columns are categories,
                    values are p_{w,c} in [0, 1].
          - schedule: Echo of the schedule used.
          - user_weekly_means: Dict[cat -> float] of user's per-week means.
          - opponent_weekly_means: Dict[team -> Dict[cat -> float]].
          - method: 'cv-based' (future: 'kalman-augmented' once wired).
          - cv_used: The CV dict actually used (for diagnostic transparency).
    """
    if config is None:
        config = LeagueConfig()
    if variance_cv is None:
        variance_cv = dict(_DEFAULT_CV)

    cats = list(config.all_categories)
    inverse = set(config.inverse_stats)
    season_weeks = int(config.season_weeks)

    # User per-week means (one set, used for every week — projections are
    # season-flat; per-week schedule strength comes from opponent variance)
    user_means = _per_week_means(user_roster_ids, player_pool, cats, season_weeks)

    # Opponent per-week means — pre-computed per team
    opp_team_means: dict[str, dict[str, float]] = {}
    for team_name, ids in all_team_rosters.items():
        opp_team_means[team_name] = _per_week_means(ids, player_pool, cats, season_weeks)

    weeks = sorted(schedule.keys())
    matrix = pd.DataFrame(index=weeks, columns=cats, dtype=float)

    for week in weeks:
        opp_name = schedule.get(week)
        if opp_name is None or opp_name not in opp_team_means:
            # Unknown opponent for this week — neutral 0.5 across all cats
            logger.debug(
                "weekly_matrix: no opponent data for week %s (opp=%r) — defaulting to 0.5",
                week,
                opp_name,
            )
            matrix.loc[week, :] = 0.5
            continue
        opp_means = opp_team_means[opp_name]

        for cat in cats:
            p = _category_win_prob(
                mu_h=user_means.get(cat, 0.0),
                mu_o=opp_means.get(cat, 0.0),
                cv=variance_cv.get(cat, 0.15),
                inverse=cat in inverse,
            )
            matrix.loc[week, cat] = p

    return {
        "matrix": matrix,
        "schedule": dict(schedule),
        "user_weekly_means": user_means,
        "opponent_weekly_means": opp_team_means,
        "method": "cv-based",
        "cv_used": dict(variance_cv),
    }


def compute_trade_weekly_delta(
    before_roster_ids: list[int],
    after_roster_ids: list[int],
    player_pool: pd.DataFrame,
    schedule: dict[int, str],
    all_team_rosters: dict[str, list[int]],
    config: LeagueConfig | None = None,
    variance_cv: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the before / after / delta weekly matrices for a trade.

    Used by evaluate_trade() when enable_weekly_matrix=True. Returns three
    matrices: pre-trade win probability per week-cat, post-trade ditto, and
    the element-wise delta (positive = trade improves your win probability
    in that week-cat).

    Args:
        before_roster_ids: Pre-trade roster IDs.
        after_roster_ids: Post-trade roster IDs.
        player_pool: Full enriched pool.
        schedule: {week: opponent_team_name} for remaining weeks.
        all_team_rosters: {team: [player_ids]} for opponent means.
        config: LeagueConfig.
        variance_cv: Optional CV override.

    Returns:
        Dict with keys:
          - before: pre-trade matrix (DataFrame)
          - after: post-trade matrix (DataFrame)
          - delta: after - before (DataFrame)
          - schedule: echo
          - summary: per-week summary including expected_cats_won_delta
          - method: 'cv-based'
    """
    before = compute_weekly_matrix(before_roster_ids, player_pool, schedule, all_team_rosters, config, variance_cv)
    after = compute_weekly_matrix(after_roster_ids, player_pool, schedule, all_team_rosters, config, variance_cv)
    delta = after["matrix"] - before["matrix"]

    # Per-week summary: expected cat-wins (sum of p_c) before/after/delta.
    # In a 12-cat H2H, expected cat-wins ranges 0-12. A delta of +1 means
    # the trade flips one full category in your favor in expectation.
    summary = pd.DataFrame(
        {
            "expected_cat_wins_before": before["matrix"].sum(axis=1),
            "expected_cat_wins_after": after["matrix"].sum(axis=1),
            "delta_cat_wins": delta.sum(axis=1),
            "opponent": [schedule.get(w, "?") for w in before["matrix"].index],
        },
        index=before["matrix"].index,
    )

    return {
        "before": before["matrix"],
        "after": after["matrix"],
        "delta": delta,
        "schedule": before["schedule"],
        "summary": summary,
        "method": before["method"],
        "cv_used": before["cv_used"],
    }


def _category_win_prob(
    mu_h: float,
    mu_o: float,
    cv: float,
    inverse: bool,
) -> float:
    """Per-category single-week win probability via normal approximation.

    p = Phi((mu_h - mu_o) / sqrt(sigma_h^2 + sigma_o^2))

    For inverse cats, the result is 1 - p (lower mean wins).
    """
    sigma_h = max(abs(mu_h) * cv, _MIN_SIGMA)
    sigma_o = max(abs(mu_o) * cv, _MIN_SIGMA)
    combined_sigma = float(np.sqrt(sigma_h**2 + sigma_o**2))

    if combined_sigma < 1e-12:
        # Degenerate: both means and sigmas are 0. Coin-flip.
        p = 0.5
    else:
        z = (mu_h - mu_o) / combined_sigma
        p = float(norm.cdf(z))

    if inverse:
        p = 1.0 - p
    # Clamp to (0, 1) for numerical safety
    return max(0.0, min(1.0, p))


def _per_week_means(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    cats: list[str],
    season_weeks: int,
) -> dict[str, float]:
    """Aggregate per-cat season totals from roster, divide by season weeks.

    For counting stats: sum projection / season_weeks → per-week mean.
    For rate stats: weighted by volume (AB for AVG, PA for OBP, IP for
    ERA/WHIP) so a roster's rate is the volume-correct blend, not a
    naive mean of rates.
    """
    rows = player_pool[player_pool["player_id"].isin(roster_ids)]
    if rows.empty:
        return dict.fromkeys(cats, 0.0)

    def _sum(col: str) -> float:
        if col not in rows.columns:
            return 0.0
        return float(pd.to_numeric(rows[col], errors="coerce").fillna(0).sum())

    sum_h = _sum("h")
    sum_ab = _sum("ab")
    sum_bb = _sum("bb")
    sum_hbp = _sum("hbp")
    sum_sf = _sum("sf")
    sum_ip = _sum("ip")
    sum_er = _sum("er")
    sum_bb_allowed = _sum("bb_allowed")
    sum_h_allowed = _sum("h_allowed")

    means: dict[str, float] = {}
    for cat in cats:
        if cat == "AVG":
            means[cat] = (sum_h / sum_ab) if sum_ab > 0 else 0.0
        elif cat == "OBP":
            denom = sum_ab + sum_bb + sum_hbp + sum_sf
            means[cat] = ((sum_h + sum_bb + sum_hbp) / denom) if denom > 0 else 0.0
        elif cat == "ERA":
            means[cat] = (sum_er * 9.0 / sum_ip) if sum_ip > 0 else 0.0
        elif cat == "WHIP":
            means[cat] = ((sum_bb_allowed + sum_h_allowed) / sum_ip) if sum_ip > 0 else 0.0
        else:
            # Counting stat: per-week mean = season total / season weeks
            col = cat.lower()
            total = _sum(col)
            means[cat] = total / max(season_weeks, 1)

    return means
