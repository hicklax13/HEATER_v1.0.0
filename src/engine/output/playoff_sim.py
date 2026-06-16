"""Playoff + championship probability via 12-team bracket simulation.

Per report Section B.10 + 2026-05-23 design Q2: FourzynBurn / Yahoo
H2H Cats default playoff format —
  * Top 4 of 12 teams advance after the regular-season weeks
  * Round 1: Seed 1 plays Seed 4, Seed 2 plays Seed 3
  * Final (re-seeded): Round-1 winners face off

This module's job is to compute, for a given user roster:

    Pi_playoff = P(user finishes top-4 by regular-season W-L record)
    Pi_champ   = P(user wins the 2-round bracket | user in top-4) * Pi_playoff

And, for trade evaluation, the DELTA between before-trade and after-trade
rosters. Per the report this is the engine's PRIMARY objective — what
the manager actually cares about is "does this trade improve my title
odds?", not "what's the SGP delta?"

Scope (Hickey-centric MVP, 2026-05-23):
  - User's per-week category win probabilities come from weekly_matrix.py
    (computed against user's actual schedule)
  - Other teams' rest-of-season wins are estimated from (current_wins,
    weeks_remaining, average_weekly_win_rate) via a Binomial approximation,
    NOT by simulating each opponent's actual schedule. Layered enhancement
    when full league schedule data is wired.

Wires into:
  - src/engine/output/weekly_matrix.py: per-week per-category probabilities
  - src/engine/output/trade_evaluator.py: invoked when enable_playoff_sim=True
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.engine.output.weekly_matrix import _DEFAULT_CV, _per_week_means
from src.engine.portfolio.copula import DEFAULT_CORRELATION as _COPULA_CORRELATION
from src.engine.portfolio.copula import GaussianCopula
from src.valuation import LeagueConfig

# TE-E2: number of correlated copula draws per week when estimating
# P(win majority of categories). 4000 keeps the per-week estimate stable
# (±0.01) while staying cheap inside the 20K-sim playoff loop (the copula
# draw is vectorised over all weeks at once).
_COPULA_DRAWS: int = 4000

logger = logging.getLogger(__name__)

# Default sim count per 2026-05-23 design Q4.
_DEFAULT_N_SIMS: int = 20_000

# Number of playoff spots (FourzynBurn / CLAUDE.md _PLAYOFF_SPOTS=4).
_PLAYOFF_SPOTS: int = 4

# CARA risk-aversion sensitivity sweep per report Section H.10 + C.7.
# λ=0.15 is the calibrated central value; 0.05 (more risk-seeking, e.g. a
# team that must gamble to make a playoff push) and 0.30 (more risk-averse,
# protecting a strong position) bracket it so the user can see how the
# risk-adjusted utility shifts with their stance.
_LAMBDA_SWEEP: tuple[float, ...] = (0.05, 0.15, 0.30)


def simulate_playoff_outcomes(
    user_roster_ids: list[int],
    user_team_name: str,
    all_team_rosters: dict[str, list[int]],
    user_schedule: dict[int, str],
    current_wins: dict[str, int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int = 42,
    variance_cv: dict[str, float] | None = None,
    full_league_schedule: dict[int, list[tuple[str, str]]] | None = None,
    correlate_categories: bool = True,
) -> dict[str, Any]:
    """Simulate the rest of the regular season + 2-round playoff bracket.

    Args:
        user_roster_ids: User's player IDs (post-trade if evaluating).
        user_team_name: Key in all_team_rosters / current_wins matching user.
        all_team_rosters: {team_name: [player_ids]} for ALL teams (incl. user).
        user_schedule: {week: opponent_team_name} for user's remaining weeks.
        current_wins: {team_name: current W-record} aggregated season wins.
        player_pool: Full enriched player pool.
        config: LeagueConfig.
        n_sims: MC iterations (default 20,000 per user choice).
        seed: Master RNG seed for reproducibility.
        variance_cv: Optional per-category CV for win-prob noise (defaults
            to weekly_matrix._DEFAULT_CV).
        correlate_categories: TE-E2 — when True (default), the user's weekly
            P(win majority) is computed from copula-correlated category
            outcomes (DEFAULT_CORRELATION) instead of the independent
            Poisson-binomial normal approximation. Correlation widens the
            weekly-cat-win variance, de-saturating over-confident playoff/
            champ probabilities. Set False for the fast independent fallback.

    Returns:
        Dict with:
          - playoff_prob: P(user finishes top-4)
          - champ_prob: P(user wins championship)
          - playoff_seed_distribution: P(user finishes seed 1/2/3/4) per seed
          - mean_regular_season_wins: user's avg additional wins this sim batch
          - n_sims: actual sims run
          - method: 'hickey-centric-binomial-opp'
    """
    if config is None:
        config = LeagueConfig()
    if variance_cv is None:
        variance_cv = dict(_DEFAULT_CV)

    rng = np.random.default_rng(seed)

    # Build canonical team order so we can index into numpy arrays
    team_names = sorted(all_team_rosters.keys())
    if user_team_name not in team_names:
        logger.warning(
            "simulate_playoff_outcomes: user_team_name %r not in all_team_rosters %r — "
            "returning zero-probability defaults",
            user_team_name,
            team_names,
        )
        return {
            "playoff_prob": 0.0,
            "champ_prob": 0.0,
            "playoff_seed_distribution": {i: 0.0 for i in range(1, _PLAYOFF_SPOTS + 1)},
            "mean_regular_season_wins": 0.0,
            "current_wins_used": {},
            "current_wins_unmatched": [],
            "n_weeks_remaining": 0,
            "team_proj_additional": {},
            "team_proj_final": {},
            "team_weekly_p": {},
            "n_sims": 0,
            "method": "hickey-centric-binomial-opp",
        }
    user_idx = team_names.index(user_team_name)
    n_teams = len(team_names)

    # ── Compute Hickey's per-week category win probabilities ────────
    # This reuses the weekly_matrix logic: for each remaining matchup week,
    # what's the probability of winning each of the 12 categories vs the
    # scheduled opponent?
    p_user_per_week_per_cat = _user_per_week_per_cat(
        user_roster_ids=user_roster_ids,
        all_team_rosters=all_team_rosters,
        user_schedule=user_schedule,
        player_pool=player_pool,
        config=config,
        variance_cv=variance_cv,
    )  # shape (n_weeks_remaining, n_cats)
    n_weeks_remaining = p_user_per_week_per_cat.shape[0]

    # ── Compute each team's average per-week win probability ────────
    # For Hickey, this comes from the per-week matrix (averaged across weeks).
    # For other teams, we approximate using their roster strength vs league
    # average — they don't have an opponent-specific schedule in this scope.
    team_avg_p_win = _team_average_weekly_win_prob(
        all_team_rosters=all_team_rosters,
        player_pool=player_pool,
        config=config,
        variance_cv=variance_cv,
    )  # dict {team_name: float in [0, 1]}

    # Override user's p with the schedule-aware avg from the matrix.
    # TE-E2: P(win majority) from copula-correlated category outcomes
    # (de-saturates over-confident odds) unless the caller opts out.
    # Same seed across before/after arms → paired-MC discipline holds.
    user_per_week_mean_cats = p_user_per_week_per_cat.sum(axis=1)  # E[cat wins]
    user_per_week_p_win = _prob_majority_cat_wins(
        p_user_per_week_per_cat,
        correlate_categories=correlate_categories,
        seed=seed,
    )
    team_avg_p_win[user_team_name] = float(user_per_week_p_win.mean())

    # Convert team_avg_p_win to numpy array in canonical order
    p_avg_arr = np.array([team_avg_p_win[t] for t in team_names], dtype=float)

    # ── Simulate user's regular-season weekly wins ──────────────────
    # For each sim, for each remaining week, sample Bernoulli(p_user_per_week_p_win[w])
    # Shape: (n_sims, n_weeks_remaining)
    user_weekly_p = user_per_week_p_win  # shape (n_weeks_remaining,)
    user_weekly_outcomes = rng.random((n_sims, n_weeks_remaining)) < user_weekly_p[None, :]
    user_additional_wins = user_weekly_outcomes.sum(axis=1).astype(int)  # (n_sims,)

    # ── Simulate other teams' regular-season wins ──────────────────
    # Phase 3 (2026-05-24): two paths.
    # Path A — full_league_schedule provided: per-team per-week win-prob
    #   matrix using actual matchups. Captures schedule-strength variation
    #   (e.g. Team A faces stronger opponents late in season).
    # Path B — no schedule (fallback): Binomial(N, avg_p) approximation
    #   per team. Assumes all opponents are league-average.
    if full_league_schedule:
        # Path A: build per-week-per-team P(win) matrix using actual matchups.
        # Shape: (n_teams, n_weeks_remaining)
        weeks_to_sim = sorted(user_schedule.keys())
        per_team_p_matrix = _compute_per_week_per_team_p_win(
            full_league_schedule=full_league_schedule,
            all_team_rosters=all_team_rosters,
            player_pool=player_pool,
            config=config,
            variance_cv=variance_cv,
            weeks_to_include=weeks_to_sim,
            team_names=team_names,
        )
        # Override user's row with the schedule-aware probs we already computed.
        # (Otherwise per_team_p_matrix has the same value for user as for others,
        # which would lose the user-side variance from compute_weekly_matrix.)
        per_team_p_matrix[user_idx, :] = user_per_week_p_win
        # Sample Bernoulli per team per week per sim.
        # Shape: (n_sims, n_teams, n_weeks_remaining)
        per_team_samples = rng.random((n_sims, n_teams, n_weeks_remaining)) < per_team_p_matrix[None, :, :]
        additional_wins = per_team_samples.sum(axis=2).astype(int)  # (n_sims, n_teams)
        sim_method = "hickey-centric-full-sched-opp"
        if correlate_categories:
            sim_method += "-corr"
    else:
        # Path B: Binomial approximation (legacy/fallback).
        additional_wins = rng.binomial(n_weeks_remaining, p_avg_arr[None, :], size=(n_sims, n_teams)).astype(int)
        # Override user's column with the schedule-aware draws
        additional_wins[:, user_idx] = user_additional_wins
        sim_method = "hickey-centric-binomial-opp"
        if correlate_categories:
            sim_method += "-corr"

    # ── Final standings ──────────────────────────────────────────────
    current_wins_arr = np.array([current_wins.get(t, 0) for t in team_names], dtype=int)
    final_wins = current_wins_arr[None, :] + additional_wins  # (n_sims, n_teams)

    # Diagnostic (2026-06): record the current-wins value actually USED per team
    # and flag any team whose name was NOT found in the supplied current_wins
    # dict (it silently defaulted to 0 above). current_wins is keyed by the
    # caller's standings team-names while team_names comes from the rosters; a
    # format mismatch (leading emoji/whitespace) drops a real W-L record here,
    # which buries that team's playoff odds. Surfaced so the UI can show whether
    # the user's own record was matched.
    current_wins_used = {t: int(current_wins.get(t, 0)) for t in team_names}
    current_wins_unmatched = [t for t in team_names if t not in current_wins]
    if current_wins_unmatched:
        logger.warning(
            "simulate_playoff_outcomes: %d team(s) not found in current_wins (defaulted to 0 wins): %s",
            len(current_wins_unmatched),
            current_wins_unmatched,
        )

    # Diagnostic (2026-06): per-team PROJECTED rest-of-season picture. Surfaces
    # WHY a team's playoff odds land where they do — the current-wins table alone
    # can't show whether the user is locked out because opponents are projected
    # to gain far more additional wins (e.g. the user is scored with the
    # de-saturated copula path while opponents use the saturated normal-approx in
    # _team_average_weekly_win_prob — an asymmetry that suppresses a strong
    # user's relative standing). All three are means over the sim batch.
    team_proj_additional = {t: round(float(additional_wins[:, i].mean()), 1) for i, t in enumerate(team_names)}
    team_proj_final = {t: round(float(final_wins[:, i].mean()), 1) for i, t in enumerate(team_names)}
    team_weekly_p = {t: round(float(p_avg_arr[i]), 3) for i, t in enumerate(team_names)}

    # ── Determine top-4 per sim and user's playoff seed ─────────────
    # Higher wins = better rank. argsort descending → top 4 = first 4 entries.
    sorted_team_indices = np.argsort(-final_wins, axis=1)  # (n_sims, n_teams)
    top4_indices = sorted_team_indices[:, :_PLAYOFF_SPOTS]  # (n_sims, 4)

    # user_in_top4[sim] = True if user_idx in top4_indices[sim, :]
    user_in_top4 = (top4_indices == user_idx).any(axis=1)
    playoff_prob = float(user_in_top4.mean())

    # User's playoff seed if in top-4 (1-indexed). For sims where not in top-4,
    # seed is set to 0 (sentinel).
    user_seed = np.zeros(n_sims, dtype=int)
    for s_idx in range(_PLAYOFF_SPOTS):
        seed_num = s_idx + 1
        user_seed[top4_indices[:, s_idx] == user_idx] = seed_num

    # ── Bracket simulation: for sims where user in top-4 ────────────
    # Round 1: seed 1 vs seed 4, seed 2 vs seed 3. Re-seeded final.
    champ_wins = _simulate_bracket(
        top4_indices=top4_indices,
        user_idx=user_idx,
        user_in_top4=user_in_top4,
        user_seed=user_seed,
        p_avg_arr=p_avg_arr,
        rng=rng,
    )
    champ_prob = float(champ_wins.mean())

    # Seed distribution (only over sims where user makes top-4)
    seed_dist: dict[int, float] = {}
    for seed_num in range(1, _PLAYOFF_SPOTS + 1):
        seed_dist[seed_num] = float((user_seed == seed_num).mean())

    return {
        "playoff_prob": round(playoff_prob, 4),
        "champ_prob": round(champ_prob, 4),
        "playoff_seed_distribution": {k: round(v, 4) for k, v in seed_dist.items()},
        "mean_regular_season_wins": round(float(user_additional_wins.mean()), 2),
        "current_wins": int(current_wins.get(user_team_name, 0)),
        "current_wins_used": current_wins_used,
        "current_wins_unmatched": current_wins_unmatched,
        # Diagnostic: per-team projected RoS picture (means over the sim batch).
        "n_weeks_remaining": int(n_weeks_remaining),
        "team_proj_additional": team_proj_additional,
        "team_proj_final": team_proj_final,
        "team_weekly_p": team_weekly_p,
        "n_sims": int(n_sims),
        "method": sim_method,
        # CARA Phase (2026-05-24): per-sim Bernoulli outcomes for the
        # downstream CARA mean-CVaR utility computation in
        # simulate_trade_playoff_delta. Each entry is 0 or 1.
        # These arrays paired with the SAME RNG seed across before/after
        # arms enable true per-sim DELTA variance (most deltas = 0).
        "champ_outcomes": champ_wins.astype(int),
        "playoff_outcomes": user_in_top4.astype(int),
    }


def simulate_trade_playoff_delta(
    before_roster_ids: list[int],
    after_roster_ids: list[int],
    user_team_name: str,
    all_team_rosters: dict[str, list[int]],
    user_schedule: dict[int, str],
    current_wins: dict[str, int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    n_sims: int = _DEFAULT_N_SIMS,
    seed: int = 42,
    variance_cv: dict[str, float] | None = None,
    full_league_schedule: dict[int, list[tuple[str, str]]] | None = None,
    correlate_categories: bool = True,
) -> dict[str, Any]:
    """Compute the DELTA playoff + championship probability from a trade.

    Per report Q(a): the primary engine output. Returns probabilities
    BEFORE the trade, AFTER the trade, and the marginal change.

    Uses paired-MC discipline: same seed across before/after arms so the
    randomness cancels and the delta isolates the trade's causal effect.

    Returns:
        Dict with:
          - before: full simulate_playoff_outcomes result for pre-trade roster
          - after: full simulate_playoff_outcomes result for post-trade roster
          - delta_playoff_prob: after.playoff_prob - before.playoff_prob
          - delta_champ_prob: after.champ_prob - before.champ_prob
          - n_sims: sims per arm
          - cara_utility: CARA mean-variance utility per report Section B.9 —
            E[Δchamp] - λ/2 × Var[Δchamp] computed from per-sim deltas.
            Penalizes high-variance trades (uncertain outcomes).
          - cvar20_champ: Conditional VaR at 20% — expected loss in the
            worst 20% of sims. Per report B.9 the "Lose the trade?"
            downside check.
          - var_champ: Var of per-sim champ deltas (input to CARA).
          - lambda: Risk-aversion parameter (LeagueConfig).
    """
    before = simulate_playoff_outcomes(
        user_roster_ids=before_roster_ids,
        user_team_name=user_team_name,
        all_team_rosters={**all_team_rosters, user_team_name: before_roster_ids},
        user_schedule=user_schedule,
        current_wins=current_wins,
        player_pool=player_pool,
        config=config,
        n_sims=n_sims,
        seed=seed,
        variance_cv=variance_cv,
        full_league_schedule=full_league_schedule,
        correlate_categories=correlate_categories,
    )
    after = simulate_playoff_outcomes(
        user_roster_ids=after_roster_ids,
        user_team_name=user_team_name,
        all_team_rosters={**all_team_rosters, user_team_name: after_roster_ids},
        user_schedule=user_schedule,
        current_wins=current_wins,
        player_pool=player_pool,
        config=config,
        n_sims=n_sims,
        seed=seed,  # SAME seed → paired-MC variance reduction
        variance_cv=variance_cv,
        full_league_schedule=full_league_schedule,
        correlate_categories=correlate_categories,
    )

    # ── CARA mean-CVaR utility (report Section B.9 + Q(a)) ────────────
    # Per-sim deltas (paired-MC = same seed across arms → most deltas are 0,
    # only sims where the bracket outcome differs contribute non-zero).
    champ_deltas = after["champ_outcomes"] - before["champ_outcomes"]
    playoff_deltas = after["playoff_outcomes"] - before["playoff_outcomes"]

    # CARA: U = E[delta] - lambda/2 × Var[delta]
    # lambda from LeagueConfig.risk_aversion (default 0.15 per report B.9 calibration).
    cfg = config if config is not None else LeagueConfig()
    lambda_ = float(getattr(cfg, "risk_aversion", 0.15))

    e_champ = float(champ_deltas.mean())
    var_champ = float(champ_deltas.var())
    cara_utility = e_champ - (lambda_ / 2.0) * var_champ

    # λ-sensitivity sweep (report Section H.10 + C.7): recompute the CARA
    # utility at the bracketing risk-aversion levels so the user can see how
    # the recommendation shifts with their risk stance. The central value
    # (0.15) appears both here and as cara_utility above.
    cara_utility_sweep = {lam: round(e_champ - (lam / 2.0) * var_champ, 6) for lam in _LAMBDA_SWEEP}

    # CVaR_20: mean of worst 20% of per-sim deltas. Report B.9 — the
    # "downside reporting" metric that preserves coherence (sub-additive
    # unlike VaR). For paired-MC where most deltas are 0, the worst 20%
    # picks up the sims where the trade hurt championship outcome.
    sorted_deltas = np.sort(champ_deltas)
    cvar20_cutoff = max(1, int(len(sorted_deltas) * 0.20))
    cvar20_champ = float(sorted_deltas[:cvar20_cutoff].mean())

    return {
        "before": before,
        "after": after,
        "delta_playoff_prob": round(after["playoff_prob"] - before["playoff_prob"], 4),
        "delta_champ_prob": round(after["champ_prob"] - before["champ_prob"], 4),
        "n_sims": n_sims,
        # CARA mean-CVaR (report Section B.9 + Q(a)) — risk-adjusted
        # utility on the per-sim championship-probability delta.
        # Recommended primary objective for decision-making per report.
        "cara_utility": round(cara_utility, 6),
        "cara_utility_sweep": cara_utility_sweep,
        "var_champ": round(var_champ, 6),
        "cvar20_champ": round(cvar20_champ, 4),
        "lambda_risk_aversion": lambda_,
        # Also expose playoff-prob mean delta for symmetry (already in
        # delta_playoff_prob, but keep here for ease of access alongside CARA).
        "mean_playoff_delta_per_sim": round(float(playoff_deltas.mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _compute_per_week_per_team_p_win(
    full_league_schedule: dict[int, list[tuple[str, str]]],
    all_team_rosters: dict[str, list[int]],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    variance_cv: dict[str, float],
    weeks_to_include: list[int],
    team_names: list[str],
) -> np.ndarray:
    """Per-team-per-week P(team wins matchup) using full league schedule.

    Per Enhanced Trade Engine report Section B.10 Phase 3 (2026-05-24).
    Replaces the Binomial approximation in playoff sim with schedule-
    aware per-week win-probability for ALL teams (not just user).

    For each (team, week) pair:
      1. Look up team's opponent in full_league_schedule
      2. Compute per-cat per-week win probs using _per_week_means + Phi formula
      3. Aggregate to P(team wins majority of 12 cats) via Poisson-binomial
         normal approximation

    If a team has no scheduled matchup in a given week (bye/playoff weeks
    not in schedule), defaults to 0.5 (neutral) for that cell.

    Returns:
        Array of shape (n_teams, n_weeks_remaining), values in [0, 1].
    """
    cats = list(config.all_categories)
    inverse = set(config.inverse_stats)
    season_weeks = int(config.season_weeks)

    # Local import to keep top-level cycle-free
    from src.engine.output.weekly_matrix import _category_win_prob

    # Pre-compute per-team per-cat means (volume-weighted rates).
    team_means_cache: dict[str, dict[str, float]] = {
        t: _per_week_means(ids, player_pool, cats, season_weeks) for t, ids in all_team_rosters.items()
    }

    n_teams = len(team_names)
    n_weeks = len(weeks_to_include)
    p_matrix = np.full((n_teams, n_weeks), 0.5)  # default neutral

    # Index team_name → team_idx for O(1) lookup
    team_idx = {t: i for i, t in enumerate(team_names)}

    for w_idx, week in enumerate(weeks_to_include):
        matchups = full_league_schedule.get(week, [])
        for team_a, team_b in matchups:
            if team_a not in team_idx or team_b not in team_idx:
                continue  # unknown team (renamed/abandoned)
            means_a = team_means_cache.get(team_a)
            means_b = team_means_cache.get(team_b)
            if means_a is None or means_b is None:
                continue

            # Compute per-cat win probs for team_a vs team_b
            p_per_cat = np.zeros(len(cats))
            for c_idx, cat in enumerate(cats):
                p_per_cat[c_idx] = _category_win_prob(
                    mu_h=means_a.get(cat, 0.0),
                    mu_o=means_b.get(cat, 0.0),
                    cv=variance_cv.get(cat, 0.15),
                    inverse=cat in inverse,
                )

            # P(team_a wins majority of 12 cats) — Poisson-binomial normal approx
            mean_cats = float(p_per_cat.sum())
            var_cats = float((p_per_cat * (1.0 - p_per_cat)).sum())
            sd = float(np.sqrt(max(var_cats, 1e-12)))
            z = (len(cats) / 2.0 - mean_cats) / sd
            p_a_wins = float(1.0 - norm.cdf(z))
            p_a_wins = max(0.0, min(1.0, p_a_wins))

            # Symmetric assignment: team_a wins → team_b loses
            p_matrix[team_idx[team_a], w_idx] = p_a_wins
            p_matrix[team_idx[team_b], w_idx] = 1.0 - p_a_wins

    return p_matrix


def _user_per_week_per_cat(
    user_roster_ids: list[int],
    all_team_rosters: dict[str, list[int]],
    user_schedule: dict[int, str],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    variance_cv: dict[str, float],
) -> np.ndarray:
    """Per-week per-category win probabilities for the user.

    Returns array of shape (n_weeks, n_cats) where each entry is the
    probability that the user wins that category in that week.
    """
    cats = list(config.all_categories)
    inverse = set(config.inverse_stats)
    season_weeks = int(config.season_weeks)

    # User's per-week means (constant across weeks since projections are flat)
    user_means = _per_week_means(user_roster_ids, player_pool, cats, season_weeks)

    # Pre-compute opponent per-week means
    opp_means_cache: dict[str, dict[str, float]] = {}
    for team_name, ids in all_team_rosters.items():
        opp_means_cache[team_name] = _per_week_means(ids, player_pool, cats, season_weeks)

    weeks = sorted(user_schedule.keys())
    n_weeks = len(weeks)
    n_cats = len(cats)
    matrix = np.zeros((n_weeks, n_cats))

    from src.engine.output.weekly_matrix import _category_win_prob

    for w_idx, week in enumerate(weeks):
        opp_name = user_schedule.get(week)
        if opp_name is None or opp_name not in opp_means_cache:
            matrix[w_idx, :] = 0.5
            continue
        opp_means = opp_means_cache[opp_name]
        for c_idx, cat in enumerate(cats):
            matrix[w_idx, c_idx] = _category_win_prob(
                mu_h=user_means.get(cat, 0.0),
                mu_o=opp_means.get(cat, 0.0),
                cv=variance_cv.get(cat, 0.15),
                inverse=cat in inverse,
            )

    return matrix


def _prob_majority_cat_wins(
    p_per_week_per_cat: np.ndarray,
    correlate_categories: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """For each week, compute P(user wins majority of 12 categories).

    Two paths (TE-E2):

    * ``correlate_categories=False`` (default — fast fallback): normal
      approximation to the Poisson-binomial distribution treating the 12
      categories as INDEPENDENT::

          mean = sum(p_c);  var = sum(p_c * (1 - p_c))
          P(X > 6) ≈ 1 - Phi((6 - mean) / sqrt(var))

      Independence understates the win-count variance because HR/RBI/R move
      together and ERA/WHIP/K move together, so it over-states P(majority)
      for a uniformly-favored roster (and under-states it for an underdog).

    * ``correlate_categories=True``: copula-correlated draws using the SAME
      ``DEFAULT_CORRELATION`` matrix LO-E3 + the Matchup Planner use. Each
      category's marginal ``p_c`` is mapped to a latent threshold; correlated
      uniforms are drawn via the Gaussian copula; a category is "won" when
      ``u_c <= p_c``; the per-draw win count gives the majority distribution.
      Positive intra-cluster correlation widens the win-count variance, so the
      estimate is pulled toward 0.5 (less over-confident) — matching the
      Matchup Planner's joint-distribution estimate.

    Args:
        p_per_week_per_cat: (n_weeks, n_cats) per-category win probs.
        correlate_categories: Use the copula-correlated path when True.
        seed: RNG seed for the copula draws (deterministic).

    Returns:
        (n_weeks,) array of P(win majority) in [0, 1].
    """
    n_weeks, n_cats = p_per_week_per_cat.shape

    if not correlate_categories or n_cats != _COPULA_CORRELATION.shape[0]:
        # Independent Poisson-binomial normal approximation (fast fallback).
        # Also the path for any non-canonical category count, where the
        # copula sub-matrix order can't be assumed.
        means = p_per_week_per_cat.sum(axis=1)  # (n_weeks,)
        variances = (p_per_week_per_cat * (1 - p_per_week_per_cat)).sum(axis=1)
        threshold = n_cats / 2.0
        sd = np.sqrt(np.maximum(variances, 1e-12))
        z = (threshold - means) / sd
        p_majority = 1.0 - norm.cdf(z)
        return np.clip(p_majority, 0.0, 1.0)

    # Copula-correlated path. The playoff sim passes all 12 cats in canonical
    # order (== copula CAT_ORDER), so the full DEFAULT_CORRELATION applies
    # directly without sub-matrix reindexing.
    rng = np.random.RandomState(seed)
    copula = GaussianCopula(_COPULA_CORRELATION)
    half = n_cats / 2.0
    p_majority = np.zeros(n_weeks)
    for w in range(n_weeks):
        probs = p_per_week_per_cat[w]  # (n_cats,)
        u = copula.sample(_COPULA_DRAWS, rng)  # (_COPULA_DRAWS, n_cats)
        wins = (u <= probs).sum(axis=1).astype(float)
        # Tie at exactly n/2 scored as half a win (Yahoo splits even ties).
        won = float((wins > half).sum())
        tied = float((wins == half).sum())
        p_majority[w] = (won + 0.5 * tied) / float(_COPULA_DRAWS)
    return np.clip(p_majority, 0.0, 1.0)


def _team_average_weekly_win_prob(
    all_team_rosters: dict[str, list[int]],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    variance_cv: dict[str, float],
) -> dict[str, float]:
    """Estimate each team's average per-week win probability.

    Computes the team's average per-cat strength vs the league average and
    converts to a single weekly win probability using the same Phi(z)
    formulation as the per-week matrix.

    This is the approximation that lets us model 'opponent strength' in the
    season sim without simulating every team's full schedule.
    """
    cats = list(config.all_categories)
    inverse = set(config.inverse_stats)
    season_weeks = int(config.season_weeks)

    # Per-team per-week means
    team_means: dict[str, dict[str, float]] = {
        t: _per_week_means(ids, player_pool, cats, season_weeks) for t, ids in all_team_rosters.items()
    }

    # League average per cat (excluding the team being scored against itself
    # would be ideal, but using the all-team average is a fine first cut).
    league_avg: dict[str, float] = {}
    for cat in cats:
        values = [team_means[t].get(cat, 0.0) for t in team_means]
        league_avg[cat] = float(np.mean(values)) if values else 0.0

    from src.engine.output.weekly_matrix import _category_win_prob

    result: dict[str, float] = {}
    for team_name, means in team_means.items():
        # Per-cat win prob vs league average
        p_per_cat = np.array(
            [
                _category_win_prob(
                    mu_h=means.get(cat, 0.0),
                    mu_o=league_avg.get(cat, 0.0),
                    cv=variance_cv.get(cat, 0.15),
                    inverse=cat in inverse,
                )
                for cat in cats
            ]
        )
        # P(majority of 12 cats) via normal approx
        mean_cats = p_per_cat.sum()
        var_cats = (p_per_cat * (1 - p_per_cat)).sum()
        sd = float(np.sqrt(max(var_cats, 1e-12)))
        z = (len(cats) / 2.0 - mean_cats) / sd
        p_win = float(1.0 - norm.cdf(z))
        result[team_name] = max(0.0, min(1.0, p_win))

    return result


def _simulate_bracket(
    top4_indices: np.ndarray,  # (n_sims, 4)
    user_idx: int,
    user_in_top4: np.ndarray,  # (n_sims,) bool
    user_seed: np.ndarray,  # (n_sims,) int 0-4 (0 = not in top 4)
    p_avg_arr: np.ndarray,  # (n_teams,)
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the 2-round bracket for each sim where user is in top-4.

    Returns (n_sims,) bool: True if user wins championship in that sim.
    """
    n_sims = len(user_in_top4)
    champ = np.zeros(n_sims, dtype=bool)

    sim_indices = np.where(user_in_top4)[0]
    if len(sim_indices) == 0:
        return champ

    # Pre-generate two random draws per top-4 sim (one for round 1, one for
    # the other matchup; the final uses a third draw)
    rand_draws = rng.random((len(sim_indices), 3))

    for ix, sim_i in enumerate(sim_indices):
        seed = int(user_seed[sim_i])
        top4 = top4_indices[sim_i]  # (4,) team indices in seed order 1..4

        # Identify user's Round 1 opponent (1v4, 2v3) and the OTHER matchup
        if seed == 1:
            r1_opp = top4[3]
            other_a, other_b = top4[1], top4[2]
        elif seed == 4:
            r1_opp = top4[0]
            other_a, other_b = top4[1], top4[2]
        elif seed == 2:
            r1_opp = top4[2]
            other_a, other_b = top4[0], top4[3]
        else:  # seed == 3
            r1_opp = top4[1]
            other_a, other_b = top4[0], top4[3]

        # User vs round-1 opp
        p_user_wins_r1 = _matchup_prob(p_avg_arr[user_idx], p_avg_arr[r1_opp])
        if rand_draws[ix, 0] >= p_user_wins_r1:
            continue  # user loses round 1, no champ

        # Other matchup
        p_a_wins = _matchup_prob(p_avg_arr[other_a], p_avg_arr[other_b])
        if rand_draws[ix, 1] < p_a_wins:
            final_opp = other_a
        else:
            final_opp = other_b

        # Final: user vs final_opp
        p_user_wins_final = _matchup_prob(p_avg_arr[user_idx], p_avg_arr[final_opp])
        if rand_draws[ix, 2] < p_user_wins_final:
            champ[sim_i] = True

    return champ


def _matchup_prob(p_a: float, p_b: float) -> float:
    """Approximate P(A wins H2H matchup) given each team's avg weekly win prob.

    Uses the Bradley-Terry-style ratio:
      P(A wins) = p_a / (p_a + p_b)
    For p_a == p_b → 0.5 (correct). For p_a >> p_b → ~1.0. For p_a == 0 → 0.

    Bounded to [0.05, 0.95] to reflect that even mismatched H2H matchups
    have some inherent randomness (one good week can swing the result).
    """
    if p_a <= 0 and p_b <= 0:
        return 0.5
    raw = p_a / (p_a + p_b)
    return max(0.05, min(0.95, raw))
