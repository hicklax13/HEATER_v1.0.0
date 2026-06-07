"""Paired Monte Carlo trade evaluation with correlated stat sampling.

Spec reference: Section 13 L10 (Copula Monte Carlo 100K sims)
               Section 17 Phase 2 item 12 (Basic Monte Carlo 10K sims)

This module runs paired simulations: for each random seed, it simulates
the remaining season BOTH with and without the trade. By using identical
seeds, the difference isolates the causal effect of the trade (variance
reduction technique).

Pipeline per simulation:
  1. Draw correlated stat samples for each player via copula + KDE marginals
  2. Apply injury probability (simple health_score-based attenuation)
  3. Compute roster category totals for pre-trade and post-trade rosters
  4. Convert to SGP and compute standings positions
  5. Aggregate across simulations for distributional metrics

Wires into:
  - src/engine/portfolio/copula.py: GaussianCopula, sample_correlated_stats
  - src/engine/projections/marginals.py: PlayerMarginal
  - src/engine/portfolio/category_analysis.py: CATEGORIES, INVERSE_CATEGORIES
  - src/engine/output/trade_evaluator.py: grade_trade
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.engine.context.injury_process import (
    estimate_injury_probability,
    frailty_from_health_score,
    sample_injury_duration,
)
from src.engine.portfolio.category_analysis import CATEGORIES, INVERSE_CATEGORIES
from src.valuation import LeagueConfig, SGPCalculator

# 2026-05-19 L1 fix: hoist rate-stat set from inside the hot MC loop
# (was called 4× per paired iteration × ~10k sims = 40k LeagueConfig()
# instantiations per run). LeagueConfig.rate_stats is a hardcoded property
# returning a fresh literal set each call, so a one-time snapshot is safe.
_RATE_CATS_SNAPSHOT: set[str] = set(LeagueConfig().rate_stats)

logger = logging.getLogger(__name__)

# Default simulation count for Phase 2 (spec says 100K for full prod,
# Phase 2 targets 10K for speed/accuracy balance)
DEFAULT_N_SIMS: int = 10_000

# Season length in days (mirrors injury_process.SEASON_DAYS) for the
# availability-fraction calc below.
_INJURY_SEASON_DAYS: int = 183


def _sample_availability_fraction(
    health_score: float,
    age: int | None,
    is_pitcher: bool,
    weeks_remaining: int,
    rng: np.random.RandomState,
    negate: bool,
) -> float:
    """Sample a player's available-fraction of the remaining season.

    TE-E5: this is the antithetic-aware twin of
    ``injury_process.sample_season_availability``. It draws from the SAME
    Weibull-duration injury model but threads the paired-MC antithetic
    transform (``u → 1 - u``) through BOTH uniform draws (the injury coin
    flip and the duration quantile), so the before/after arms keep their
    perfectly anti-correlated noise.

    A perfectly healthy player (health_score ≥ ~1.0) has a low injury
    probability and almost always returns 1.0 (fully available). A fragile
    player has a fat left tail of availability < 1.0, which the caller uses
    to scale that player's counting-stat volume down.

    Returns a fraction in [0, 1]. Rate stats are unaffected by the caller;
    only counting-stat volume is scaled.
    """
    days_remaining = max(weeks_remaining * 7, 1)

    p_injury = estimate_injury_probability(
        health_score=health_score,
        age=age,
        is_pitcher=is_pitcher,
        horizon_days=days_remaining,
    )

    # Injury coin flip. Antithetic: reflect the uniform so the paired arm's
    # draw is perfectly anti-correlated (matches the copula u → 1-u and the
    # Gaussian z → -z discipline used elsewhere in this module).
    u_coin = rng.uniform()
    if negate:
        u_coin = 1.0 - u_coin
    if u_coin > p_injury:
        return 1.0  # no injury — fully available

    # Sample injury duration via the Weibull model. sample_injury_duration
    # consumes one rng.uniform() internally; we draw that uniform here so we
    # can reflect it for the antithetic arm, then hand it to the duration
    # sampler through a deterministic single-value RandomState stand-in.
    u_dur = rng.uniform()
    if negate:
        u_dur = 1.0 - u_dur
    frailty = frailty_from_health_score(health_score)
    duration = sample_injury_duration(
        injury_type="other",
        frailty=frailty,
        rng=_FixedUniform(u_dur),
    )

    days_missed = min(duration, days_remaining)
    return max(0.0, 1.0 - days_missed / days_remaining)


class _FixedUniform:
    """Minimal RandomState stand-in returning a preset uniform.

    ``sample_injury_duration`` only calls ``rng.uniform(0, 1)``; this lets the
    caller pre-draw (and reflect, for antithetic) the duration quantile so the
    paired-MC discipline holds end-to-end.
    """

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:  # noqa: ARG002
        return self._value


# Minimum sims for meaningful confidence intervals
MIN_SIMS: int = 100


def run_paired_monte_carlo(
    before_roster_stats: dict[str, dict[str, float]],
    after_roster_stats: dict[str, dict[str, float]],
    before_marginals: dict[str, dict[str, object]] | None = None,
    after_marginals: dict[str, dict[str, object]] | None = None,
    copula: Any | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    sgp_denominators: dict[str, float] | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
    weeks_remaining: int = 16,
    enable_injury_mc: bool = False,
) -> dict[str, Any]:
    """Run paired Monte Carlo simulation for trade evaluation.

    Spec ref: Section 13 L10 — paired comparison with identical seeds.

    For each simulation:
    1. Generate noise perturbations (same seed for pre and post)
    2. Apply to roster stat projections
    3. Compute SGP totals
    4. Measure standings position delta

    The paired design ensures that randomness cancels out, isolating
    the trade's causal effect.

    Args:
        before_roster_stats: {player_id: {stat: projected_value}} for pre-trade roster.
        after_roster_stats: {player_id: {stat: projected_value}} for post-trade roster.
        before_marginals: Optional {player_id: {stat: PlayerMarginal}} for pre-trade.
            If None, uses Gaussian noise based on projection values.
        after_marginals: Optional {player_id: {stat: PlayerMarginal}} for post-trade.
        copula: Optional GaussianCopula for correlated sampling.
            If None, samples categories independently.
        all_team_totals: {team_name: {cat: total}} for standings context.
        sgp_denominators: {cat: denominator} for SGP conversion.
        n_sims: Number of paired simulations.
        seed: Master random seed.
        weeks_remaining: Weeks left in the season (scales noise amplitude).
        enable_injury_mc: TE-E5 — when True, each player's COUNTING-stat
            contributions are scaled by a sampled season-availability
            fraction (Weibull injury-duration model from injury_process).
            Fragile/low-health players get a fat downside tail (CVaR5 drops)
            because missed time reduces their volume. Default OFF so Trade
            Finder bulk scans stay fast and the legacy distribution is
            byte-for-byte preserved; turn ON only for single-trade deep
            evaluation (evaluate_trade(..., enable_mc=True)).

    Returns:
        Dict with keys:
          - mc_mean: Mean SGP surplus across sims
          - mc_std: Standard deviation of surplus
          - mc_median: Median surplus
          - p5, p25, p75, p95: Percentile surplus values
          - prob_positive: Probability that trade helps (surplus > 0)
          - var5: Value at Risk at 5th percentile
          - cvar5: Conditional VaR (expected loss in worst 5%)
          - sharpe: Sharpe ratio (mean / std)
          - grade: Letter grade from distribution metrics
          - confidence_interval: (lower, upper) 95% CI for mean surplus
          - surplus_distribution: Full array of surplus values (for plotting)
    """
    n_sims = max(n_sims, MIN_SIMS)

    # Default SGP denominators if not provided
    if sgp_denominators is None:
        sgp_denominators = _default_sgp_denoms()

    # Hoist SGPCalculator instantiation out of the hot loop.
    # _simulate_roster_sgp is called 4x per paired iteration; with n_sims=10_000
    # that's 40_000 instantiations otherwise.
    sgp_calc = SGPCalculator(LeagueConfig(), denominators=sgp_denominators)

    rng_master = np.random.RandomState(seed)

    # TE-C3: the MC evaluates a roster-vs-self SGP delta, so opponent
    # standings context never enters the surplus math. The former
    # `_compute_other_teams_sgp(all_team_totals, ...)` call here discarded
    # its return value — dead work (a full pass over every team's totals per
    # MC invocation). Removed. `all_team_totals` is retained in the signature
    # for API compatibility; the helper remains defined for potential reuse.

    # C8: True antithetic variate MC — run n_sims/2 paired simulations.
    # Each pair uses the SAME random seed but the antithetic arm negates the
    # underlying uniform quantiles (u → 1-u, equivalently z → -z for normals).
    # This guarantees perfect negative correlation between the two arms'
    # noise, which reduces the variance of their average by ~20-40% vs.
    # independent draws.  Produces 2*half_sims results total.
    #
    # D4B-002: pair count is integer-floor of n_sims/2, so odd n_sims
    # would leave a trailing zero in surpluses (biasing all aggregates
    # toward 0). Round n_sims DOWN to 2*half_sims so every slot in the
    # surpluses array corresponds to an actual simulation.
    half_sims = max(1, n_sims // 2)
    n_sims = 2 * half_sims
    surpluses = np.zeros(n_sims)
    before_sgp_sims = np.zeros(n_sims)
    after_sgp_sims = np.zeros(n_sims)

    for sim_idx in range(half_sims):
        sim_seed = rng_master.randint(0, 2**31)

        # Original simulation (same seed for before/after — paired-MC discipline)
        before_total_sgp = _simulate_roster_sgp(
            roster_stats=before_roster_stats,
            marginals=before_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
            sgp_calc=sgp_calc,
            negate_noise=False,
            enable_injury_mc=enable_injury_mc,
        )
        after_total_sgp = _simulate_roster_sgp(
            roster_stats=after_roster_stats,
            marginals=after_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
            sgp_calc=sgp_calc,
            negate_noise=False,
            enable_injury_mc=enable_injury_mc,
        )

        surpluses[sim_idx * 2] = after_total_sgp - before_total_sgp
        before_sgp_sims[sim_idx * 2] = before_total_sgp
        after_sgp_sims[sim_idx * 2] = after_total_sgp

        # True antithetic arm: SAME seed, but negate every noise draw.
        # For Gaussian noise the negation is exact (z → -z).
        # For copula uniforms, u → 1 - u is applied inside _simulate_roster_sgp.
        # Preserves paired-MC discipline (before/after still share noise).
        before_anti = _simulate_roster_sgp(
            roster_stats=before_roster_stats,
            marginals=before_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
            sgp_calc=sgp_calc,
            negate_noise=True,
            enable_injury_mc=enable_injury_mc,
        )
        after_anti = _simulate_roster_sgp(
            roster_stats=after_roster_stats,
            marginals=after_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
            sgp_calc=sgp_calc,
            negate_noise=True,
            enable_injury_mc=enable_injury_mc,
        )

        surpluses[sim_idx * 2 + 1] = after_anti - before_anti
        before_sgp_sims[sim_idx * 2 + 1] = before_anti
        after_sgp_sims[sim_idx * 2 + 1] = after_anti

    # Compute distributional metrics
    mc_mean = float(np.mean(surpluses))
    mc_std = float(np.std(surpluses))
    mc_median = float(np.median(surpluses))

    p5 = float(np.percentile(surpluses, 5))
    p25 = float(np.percentile(surpluses, 25))
    p75 = float(np.percentile(surpluses, 75))
    p95 = float(np.percentile(surpluses, 95))

    prob_positive = float(np.mean(surpluses > 0))

    # Value at Risk and Conditional VaR
    var5 = p5
    worst_5pct = surpluses[surpluses <= np.percentile(surpluses, 5)]
    cvar5 = float(np.mean(worst_5pct)) if len(worst_5pct) > 0 else var5

    # Sharpe ratio: mean surplus / volatility of surplus
    sharpe = mc_mean / max(mc_std, 0.001)

    # 95% confidence interval for the mean
    se = mc_std / np.sqrt(n_sims)
    ci_lower = mc_mean - 1.96 * se
    ci_upper = mc_mean + 1.96 * se

    # Grade using the composite score from spec L11
    # Spec: s = (ce_delta*100)*0.4 + sharpe*0.3 + kelly*0.3
    # Simplified for Phase 2: use mc_mean as the primary signal
    grade = _grade_from_distribution(mc_mean, sharpe, prob_positive)

    # Verdict
    verdict = "ACCEPT" if mc_mean > 0 else "DECLINE"
    confidence_pct = min(100.0, max(0.0, prob_positive * 100))

    return {
        "mc_mean": round(mc_mean, 3),
        "mc_std": round(mc_std, 3),
        "mc_median": round(mc_median, 3),
        "p5": round(p5, 3),
        "p25": round(p25, 3),
        "p75": round(p75, 3),
        "p95": round(p95, 3),
        "prob_positive": round(prob_positive, 3),
        "var5": round(var5, 3),
        "cvar5": round(cvar5, 3),
        "sharpe": round(sharpe, 3),
        "grade": grade,
        "verdict": verdict,
        "confidence_pct": round(confidence_pct, 1),
        "confidence_interval": (round(ci_lower, 3), round(ci_upper, 3)),
        "surplus_distribution": surpluses,
        "n_sims": n_sims,
    }


def _simulate_roster_sgp(
    roster_stats: dict[str, dict[str, float]],
    marginals: dict[str, dict[str, object]] | None,
    copula: Any | None,
    sgp_denominators: dict[str, float],
    seed: int,
    weeks_remaining: int,
    sgp_calc: SGPCalculator | None = None,
    negate_noise: bool = False,
    enable_injury_mc: bool = False,
) -> float:
    """Simulate one season and compute total SGP for a roster.

    For each player, perturb their projected stats using either:
    1. Correlated copula + KDE marginals (if available)
    2. Independent Gaussian noise (fallback)

    Then sum up all category totals and convert to SGP.

    Args:
        negate_noise: If True, applies the antithetic transform — every
            Gaussian noise draw is negated (z → -z) and copula uniforms
            are reflected (u → 1 - u). Combined with paired-MC seed
            discipline this gives true antithetic variates with perfectly
            anti-correlated noise across the two arms.
        enable_injury_mc: TE-E5 — if True, each player's COUNTING-stat
            component contributions are scaled by a sampled availability
            fraction (Weibull injury-duration model). The availability draw
            uses the SAME per-player ``rng`` stream and honors the antithetic
            reflection, so identical before/after rosters cancel exactly and
            the antithetic arm stays perfectly anti-correlated. Rate stats are
            unaffected (only volume is reduced).
    """
    rng = np.random.RandomState(seed)

    # Noise scale: more weeks remaining = more uncertainty
    noise_scale = min(1.0, np.sqrt(weeks_remaining / 26.0))

    # Antithetic sign multiplier — flips every Gaussian noise draw.
    # Note: noise multipliers below are of the form (1 + rng.normal(0, sigma)),
    # so the negation must be applied to the *normal draw*, not the (1 + ...)
    # term, otherwise the multiplier would invert (e.g. 1.05 → 0.95) instead
    # of being negated (1.05 → -1.05). We instead use a small helper.
    noise_sign = -1.0 if negate_noise else 1.0

    # Aggregate roster totals with noise
    roster_totals: dict[str, float] = {cat: 0.0 for cat in CATEGORIES}

    # Track rate-stat components for proper aggregation
    # Rate stats (AVG, OBP, ERA, WHIP) must NOT be summed across players.
    # Instead, accumulate component stats and derive rate stats at the end.
    total_h = 0.0
    total_ab = 0.0
    total_pa = 0.0
    total_bb = 0.0
    total_hbp = 0.0
    total_sf = 0.0
    total_ip = 0.0
    total_er = 0.0
    total_bb_h_allowed = 0.0  # bb_allowed + h_allowed

    # Categories that are rate stats and need special aggregation.
    # 2026-05-19 D6: snapshot from LeagueConfig.
    # 2026-05-19 L1 follow-up: read from module-level snapshot to avoid
    # 40k LeagueConfig instantiations across the paired-MC loop.
    _RATE_CATS = _RATE_CATS_SNAPSHOT

    for player_id, stats in roster_stats.items():
        # Collect component stats for rate-stat aggregation.
        # If components (h, ab, ip, er, etc.) are available, use them directly.
        # Otherwise, derive from rate stats and playing time estimates.
        p_ab = stats.get("ab", 0.0)
        p_h = stats.get("h", 0.0)
        p_pa = stats.get("pa", 0.0)
        p_bb = stats.get("bb", 0.0)
        p_hbp = stats.get("hbp", 0.0)
        p_sf = stats.get("sf", 0.0)
        p_ip = stats.get("ip", 0.0)
        p_er = stats.get("er", 0.0)
        p_bb_allowed = stats.get("bb_allowed", 0.0)
        p_h_allowed = stats.get("h_allowed", 0.0)

        # If component stats are missing but rate stats exist, derive them
        if p_ab == 0.0 and stats.get("avg", 0.0) > 0:
            # Estimate AB from PA or use a default for a typical hitter
            p_ab = p_pa * 0.92 if p_pa > 0 else 500.0
            p_h = stats["avg"] * p_ab
        if p_pa == 0.0 and p_ab > 0:
            p_pa = p_ab / 0.92  # Estimate PA from AB
        # Derive OBP components if OBP exists but BB missing
        if p_bb == 0.0 and stats.get("obp", 0.0) > 0 and p_pa > 0:
            # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
            # Approximate: BB ≈ (OBP * PA - H) / (1 + OBP) when HBP/SF small
            obp_val = stats["obp"]
            p_bb = max(0.0, (obp_val * p_pa - p_h))
        if p_ip == 0.0 and (stats.get("era", 0.0) > 0 or stats.get("whip", 0.0) > 0):
            # Use position-appropriate IP default: relievers ~60 IP, starters ~150 IP
            sv = stats.get("sv", 0.0)
            p_ip = 60.0 if sv > 5 else 150.0
            if stats.get("era", 0.0) > 0:
                p_er = stats["era"] * p_ip / 9.0
            # D4B-003: when WHIP missing, fall back to league-average (1.30)
            # × synthetic IP. Previous code set p_bb_h_allowed = 0.0, which
            # contributed the synthetic IP to total_ip without contributing
            # baserunners — silently *deflating* team WHIP.
            if stats.get("whip", 0.0) > 0:
                p_bb_h_allowed = stats["whip"] * p_ip
            else:
                # 2026-05-17 Section 3 D1: read from CONSTANTS_REGISTRY.
                from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

                p_bb_h_allowed = _CR["league_avg_whip"].value * p_ip
            p_bb_allowed = p_bb_h_allowed * 0.4  # rough split
            p_h_allowed = p_bb_h_allowed * 0.6

        # TE-E5: sample this player's available-fraction of the season and
        # scale COUNTING-stat volume by it. The draw consumes rng BEFORE any
        # stat-noise draws so the per-player rng position is identical for the
        # before/after arms (paired-MC cancels exactly on identical rosters)
        # and the antithetic arm reflects the same uniforms (negate_noise).
        # Rate-stat components (h/ab, er/ip, ...) are ALL scaled by the same
        # factor, leaving the per-player rate unchanged while reducing its
        # volume weight in the roster aggregate.
        avail = 1.0
        if enable_injury_mc:
            avail = _sample_availability_fraction(
                health_score=float(stats.get("health_score", 1.0)),
                age=(int(stats["age"]) if stats.get("age") not in (None, 0, 0.0) else None),
                is_pitcher=bool(stats.get("is_pitcher", 0)),
                weeks_remaining=weeks_remaining,
                rng=rng,
                negate=negate_noise,
            )
            # Scale rate-stat components (numerator AND denominator) so the
            # per-player rate is preserved but its volume weight shrinks.
            p_h *= avail
            p_ab *= avail
            p_pa *= avail
            p_bb *= avail
            p_hbp *= avail
            p_sf *= avail
            p_ip *= avail
            p_er *= avail
            p_bb_allowed *= avail
            p_h_allowed *= avail

        # Generate noise for this player
        if marginals and player_id in marginals and copula is not None:
            # Use copula + marginals for correlated sampling
            player_marg = marginals[player_id]
            u = copula.sample(1, rng)[0]
            # Antithetic in uniform space: u → 1 - u. Combined with the
            # 1e-6 clip inside copula.sample, 1-u remains in (1e-6, 1-1e-6).
            if negate_noise:
                u = 1.0 - u

            for i, cat in enumerate(CATEGORIES):
                cat_lower = cat.lower()
                if cat in _RATE_CATS:
                    # Skip rate stats — they'll be computed from components below
                    continue
                if cat_lower in player_marg and cat_lower in stats:
                    m = player_marg[cat_lower]
                    # Sample from marginal using copula quantile
                    if cat in INVERSE_CATEGORIES:
                        sampled = m.ppf(1 - u[i])
                    else:
                        sampled = m.ppf(u[i])
                    # TE-E5: scale counting-stat volume by availability.
                    roster_totals[cat] += float(sampled) * avail
                elif cat_lower in stats:
                    roster_totals[cat] += stats[cat_lower] * avail

            # Accumulate components with separate noise for numerator and
            # denominator so rate-stat uncertainty is not cancelled out.
            # Hitting: H gets higher noise than AB (AVG = H/AB)
            noise_h = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            noise_ab = 1.0 + noise_sign * rng.normal(0, 0.02 * noise_scale)
            noise_bb = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            total_h += p_h * max(noise_h, 0.5)
            total_ab += p_ab * max(noise_ab, 0.5)
            total_pa += p_pa * max(noise_ab, 0.5)
            total_bb += p_bb * max(noise_bb, 0.5)
            total_hbp += p_hbp * max(noise_h, 0.5)
            total_sf += p_sf * max(noise_ab, 0.5)
            # Pitching: ER/BB/H_allowed get higher noise than IP
            noise_er = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            noise_ip = 1.0 + noise_sign * rng.normal(0, 0.02 * noise_scale)
            noise_bb_h_allowed = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            total_ip += p_ip * max(noise_ip, 0.5)
            total_er += p_er * max(noise_er, 0.5)
            total_bb_h_allowed += (p_bb_allowed + p_h_allowed) * max(noise_bb_h_allowed, 0.5)
        else:
            # Fallback: independent Gaussian noise
            for cat in CATEGORIES:
                cat_lower = cat.lower()
                if cat in _RATE_CATS:
                    # Skip rate stats — they'll be computed from components below
                    continue
                if cat_lower in stats:
                    base_val = stats[cat_lower]
                    # Noise proportional to the stat value (coefficient of variation ~15%)
                    noise_sigma = abs(base_val) * 0.15 * noise_scale
                    noisy_val = base_val + noise_sign * rng.normal(0, max(noise_sigma, 0.01))
                    # TE-E5: scale counting-stat volume by availability.
                    roster_totals[cat] += noisy_val * avail

            # Accumulate components with separate noise for numerator and
            # denominator so rate-stat uncertainty is not cancelled out.
            noise_h = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            noise_ab = 1.0 + noise_sign * rng.normal(0, 0.02 * noise_scale)
            noise_bb = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            total_h += p_h * max(noise_h, 0.5)
            total_ab += p_ab * max(noise_ab, 0.5)
            total_pa += p_pa * max(noise_ab, 0.5)
            total_bb += p_bb * max(noise_bb, 0.5)
            total_hbp += p_hbp * max(noise_h, 0.5)
            total_sf += p_sf * max(noise_ab, 0.5)
            noise_er = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            noise_ip = 1.0 + noise_sign * rng.normal(0, 0.02 * noise_scale)
            noise_bb_h_allowed = 1.0 + noise_sign * rng.normal(0, 0.05 * noise_scale)
            total_ip += p_ip * max(noise_ip, 0.5)
            total_er += p_er * max(noise_er, 0.5)
            total_bb_h_allowed += (p_bb_allowed + p_h_allowed) * max(noise_bb_h_allowed, 0.5)

    # Compute rate stats from accumulated components (not naive sums)
    # AVG = sum(H) / sum(AB)
    roster_totals["AVG"] = total_h / total_ab if total_ab > 0 else 0.0
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_denom = total_ab + total_bb + total_hbp + total_sf
    roster_totals["OBP"] = (total_h + total_bb + total_hbp) / obp_denom if obp_denom > 0 else 0.0
    # ERA = sum(ER) * 9 / sum(IP)
    roster_totals["ERA"] = total_er * 9.0 / total_ip if total_ip > 0 else 0.0
    # WHIP = sum(BB + H_allowed) / sum(IP)
    roster_totals["WHIP"] = total_bb_h_allowed / total_ip if total_ip > 0 else 0.0

    # Convert totals to SGP via SGPCalculator (single source of truth)
    if sgp_calc is None:
        sgp_calc = SGPCalculator(LeagueConfig(), denominators=sgp_denominators)
    return sgp_calc.totals_sgp(roster_totals)


def _compute_other_teams_sgp(
    all_team_totals: dict[str, dict[str, float]],
    sgp_denominators: dict[str, float],
) -> list[float]:
    """Compute total SGP for all other teams (for standings context)."""
    team_sgps = []
    for team_name, totals in all_team_totals.items():
        sgp = 0.0
        for cat in CATEGORIES:
            denom = sgp_denominators.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            val = totals.get(cat, 0.0)
            if cat in INVERSE_CATEGORIES:
                sgp -= val / denom
            else:
                sgp += val / denom
        team_sgps.append(sgp)
    return team_sgps


def _grade_from_distribution(
    mc_mean: float,
    sharpe: float,
    prob_positive: float,
) -> str:
    """Grade a trade from MC distribution metrics.

    Spec ref: Section 14 L11 — composite score for grading.

    Composite = mc_mean * 0.4 + sharpe * 0.3 + kelly_approx * 0.3

    Kelly approximation: prob_positive * 2 - 1 (simplified Kelly criterion
    assuming roughly symmetric gain/loss).
    """
    kelly_approx = max(0, prob_positive * 2 - 1)
    composite = mc_mean * 0.4 + sharpe * 0.3 + kelly_approx * 0.3

    thresholds = [
        (2.0, "A+"),
        (1.5, "A"),
        (1.0, "A-"),
        (0.7, "B+"),
        (0.4, "B"),
        (0.2, "B-"),
        (0.0, "C+"),
        (-0.2, "C"),
        (-0.5, "C-"),
        (-1.0, "D"),
    ]

    for threshold, grade in thresholds:
        if composite > threshold:
            return grade
    return "F"


def _default_sgp_denoms() -> dict[str, float]:
    """Default SGP denominators from LeagueConfig."""
    from src.valuation import LeagueConfig

    return dict(LeagueConfig().sgp_denominators)


def build_roster_stats(
    player_ids: list[int],
    player_pool: dict[str, dict[str, float]] | Any,
) -> dict[str, dict[str, float]]:
    """Build roster_stats dict from player IDs and pool.

    Converts the player pool (DataFrame or dict) into the
    {player_id_str: {stat: value}} format needed by the MC simulator.

    Args:
        player_ids: List of player IDs on the roster.
        player_pool: Either a dict {player_id: {stat: value}} or
            a DataFrame with player_id column and stat columns.

    Returns:
        Dict mapping str(player_id) -> {stat_lower: value}.
    """
    import pandas as pd

    stat_cols = [
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "obp",
        "w",
        "l",
        "k",
        "sv",
        "era",
        "whip",
        "h",
        "ab",
        "pa",
        "ip",
        "er",
        "bb",
        "hbp",
        "sf",
        "bb_allowed",
        "h_allowed",
    ]
    roster_stats: dict[str, dict[str, float]] = {}

    if isinstance(player_pool, pd.DataFrame):
        for pid in player_ids:
            match = player_pool[player_pool["player_id"] == pid]
            if match.empty:
                continue
            row = match.iloc[0]
            stats: dict[str, float] = {}
            for col in stat_cols:
                if col in row.index:
                    val = row[col]
                    stats[col] = float(val) if pd.notna(val) else 0.0
            # TE-E5: carry injury-MC inputs when available. health_score
            # defaults to 1.0 (fully available) so a pool missing the column
            # behaves exactly as the no-injury path. is_pitcher derives from
            # is_hitter (pool convention: is_hitter ∈ {0, 1}).
            if "health_score" in row.index and pd.notna(row["health_score"]):
                stats["health_score"] = float(row["health_score"])
            if "age" in row.index and pd.notna(row["age"]):
                stats["age"] = float(row["age"])
            if "is_hitter" in row.index and pd.notna(row["is_hitter"]):
                stats["is_pitcher"] = 0.0 if int(row["is_hitter"]) == 1 else 1.0
            roster_stats[str(pid)] = stats
    elif isinstance(player_pool, dict):
        for pid in player_ids:
            if pid in player_pool:
                roster_stats[str(pid)] = dict(player_pool[pid])

    return roster_stats
