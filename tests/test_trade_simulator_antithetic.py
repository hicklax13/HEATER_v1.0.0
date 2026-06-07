"""C8: True antithetic variates for paired Monte Carlo trade simulator.

The XOR-seed approach (`sim_seed ^ 0x7FFFFFFF`) gave an *independent* second
draw, not a true antithetic variate. True antithetic variates negate the
underlying uniform quantiles (u → 1-u, equivalently z → -z for normals)
while keeping the seed the same, guaranteeing perfectly anti-correlated
noise across the two arms.

These tests verify both the math identity and a behavioural property
through the trade simulator's public API.
"""

from __future__ import annotations

import numpy as np

from src.engine.monte_carlo.trade_simulator import (
    _simulate_roster_sgp,
    run_paired_monte_carlo,
)

# ── Math identity: -1 * z gives the antithetic sample ─────────────


def test_antithetic_normals_are_negated():
    """Direct property test: -1 * z gives the antithetic sample.

    For a normal distribution generated via `rng.standard_normal`, the
    antithetic transform is exact element-wise negation. Pearson
    correlation between original and antithetic must be -1.0.
    """
    rng = np.random.default_rng(42)
    base = rng.standard_normal(size=1000)
    anti = -base

    corr = np.corrcoef(base, anti)[0, 1]
    # Allow tiny floating-point slack from correlation arithmetic
    assert abs(corr - (-1.0)) < 1e-12, f"Antithetic should have correlation -1.0, got {corr}"

    # And the means should be negatives of each other
    assert abs(np.mean(base) + np.mean(anti)) < 1e-12

    # And the sum of paired averages should be exactly zero
    paired_avg = (base + anti) / 2.0
    assert np.all(np.abs(paired_avg) < 1e-12)


def test_uniform_antithetic_reflection():
    """For uniform variates, u → 1-u gives the antithetic sample.

    This is the form used inside the copula path of _simulate_roster_sgp.
    """
    rng = np.random.default_rng(7)
    u = rng.uniform(0, 1, size=1000)
    anti = 1.0 - u

    # Pairwise correlation must be exactly -1 (within FP slack)
    corr = np.corrcoef(u, anti)[0, 1]
    assert abs(corr - (-1.0)) < 1e-12, f"u/(1-u) should have correlation -1.0, got {corr}"

    # The sum of every pair is exactly 1
    assert np.all(np.abs(u + anti - 1.0) < 1e-12)


# ── Simulator behaviour: negate_noise produces anti-correlated outputs ─


def _toy_roster() -> dict[str, dict[str, float]]:
    """Minimal 2-player roster to exercise both hitter and pitcher noise paths."""
    return {
        "1": {
            "hr": 25,
            "r": 80,
            "rbi": 80,
            "sb": 10,
            "avg": 0.280,
            "obp": 0.350,
            "w": 0,
            "k": 0,
            "sv": 0,
            "era": 0.0,
            "whip": 0.0,
        },
        "2": {
            "hr": 0,
            "r": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.0,
            "obp": 0.0,
            "w": 12,
            "k": 200,
            "sv": 0,
            "era": 3.50,
            "whip": 1.20,
        },
    }


def test_simulate_roster_sgp_accepts_negate_noise_flag():
    """The new `negate_noise` kwarg must be a valid arg on _simulate_roster_sgp."""
    roster = _toy_roster()
    # Should not raise — both flags should be accepted
    sgp_orig = _simulate_roster_sgp(
        roster_stats=roster,
        marginals=None,
        copula=None,
        sgp_denominators={
            "R": 50,
            "HR": 10,
            "RBI": 50,
            "SB": 10,
            "AVG": 0.005,
            "OBP": 0.005,
            "W": 5,
            "L": 5,
            "SV": 5,
            "K": 50,
            "ERA": 0.20,
            "WHIP": 0.05,
        },
        seed=42,
        weeks_remaining=16,
        negate_noise=False,
    )
    sgp_anti = _simulate_roster_sgp(
        roster_stats=roster,
        marginals=None,
        copula=None,
        sgp_denominators={
            "R": 50,
            "HR": 10,
            "RBI": 50,
            "SB": 10,
            "AVG": 0.005,
            "OBP": 0.005,
            "W": 5,
            "L": 5,
            "SV": 5,
            "K": 50,
            "ERA": 0.20,
            "WHIP": 0.05,
        },
        seed=42,
        weeks_remaining=16,
        negate_noise=True,
    )
    assert np.isfinite(sgp_orig)
    assert np.isfinite(sgp_anti)


def test_negate_noise_flips_noise_direction():
    """For a roster with no copula/marginals, the noise sign flips.

    A high-noise stat (HR=25, sigma~3.75) deviates symmetrically around
    the projection. With the same seed, original SGP and antithetic SGP
    should bracket the noise-free SGP in opposite directions.
    """
    roster = {
        "1": {
            "hr": 25,
            "r": 80,
            "rbi": 80,
            "sb": 10,
            "avg": 0.280,
            "obp": 0.0,
            "w": 0,
            "k": 0,
            "sv": 0,
            "era": 0.0,
            "whip": 0.0,
        }
    }
    denoms = {
        "R": 50,
        "HR": 10,
        "RBI": 50,
        "SB": 10,
        "AVG": 0.005,
        "OBP": 0.005,
        "W": 5,
        "L": 5,
        "SV": 5,
        "K": 50,
        "ERA": 0.20,
        "WHIP": 0.05,
    }

    # Run both arms across many seeds and check that paired noise cancels
    deltas_paired = []
    for seed in range(200):
        sgp_orig = _simulate_roster_sgp(
            roster_stats=roster,
            marginals=None,
            copula=None,
            sgp_denominators=denoms,
            seed=seed,
            weeks_remaining=16,
            negate_noise=False,
        )
        sgp_anti = _simulate_roster_sgp(
            roster_stats=roster,
            marginals=None,
            copula=None,
            sgp_denominators=denoms,
            seed=seed,
            weeks_remaining=16,
            negate_noise=True,
        )
        # The paired average should have lower variance than either arm alone
        deltas_paired.append((sgp_orig + sgp_anti) / 2.0)

    # Compare to running 200 independent (different-seed) draws
    deltas_indep = []
    for seed in range(200):
        sgp = _simulate_roster_sgp(
            roster_stats=roster,
            marginals=None,
            copula=None,
            sgp_denominators=denoms,
            seed=seed,
            weeks_remaining=16,
            negate_noise=False,
        )
        deltas_indep.append(sgp)

    # The paired antithetic average must have *lower* sample variance than
    # the raw independent draws (variance reduction is the entire point).
    var_paired = float(np.var(deltas_paired))
    var_indep = float(np.var(deltas_indep))
    # Allow generous slack — strict guarantee is var_paired <= var_indep,
    # but we want a meaningful reduction (>20%) to flag silent regressions.
    assert var_paired < var_indep * 0.80, (
        f"Antithetic paired variance ({var_paired:.4f}) must be "
        f"meaningfully lower than independent variance ({var_indep:.4f})"
    )


def test_paired_mc_identical_rosters_zero_surplus():
    """Paired-MC discipline preserved: identical rosters → exactly 0 surplus.

    The antithetic refactor must not break the canonical paired-MC
    invariant. With before == after, every paired delta should be exactly 0.
    """
    roster = _toy_roster()
    result = run_paired_monte_carlo(
        before_roster_stats=roster,
        after_roster_stats=roster,
        n_sims=200,
    )
    assert abs(result["mc_mean"]) < 1e-9, f"Identical rosters should give 0 surplus, got {result['mc_mean']}"
    assert abs(result["mc_std"]) < 1e-9


def test_paired_mc_better_roster_still_positive():
    """Sanity: a strictly better after-roster still produces positive mean.

    Confirms the antithetic refactor didn't accidentally cancel signal
    along with noise.
    """
    before = {
        "1": {
            "hr": 10,
            "r": 40,
            "rbi": 40,
            "sb": 5,
            "avg": 0.240,
            "obp": 0.300,
            "w": 5,
            "k": 80,
            "sv": 0,
            "era": 5.00,
            "whip": 1.50,
        },
    }
    after = {
        "1": {
            "hr": 35,
            "r": 100,
            "rbi": 110,
            "sb": 15,
            "avg": 0.300,
            "obp": 0.380,
            "w": 15,
            "k": 250,
            "sv": 0,
            "era": 2.80,
            "whip": 1.00,
        },
    }
    result = run_paired_monte_carlo(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=400,
    )
    assert result["mc_mean"] > 0
    assert result["prob_positive"] > 0.5


def test_all_team_totals_does_not_affect_mc_output():
    """TE-C3: passing all_team_totals must not change the MC surplus math.

    The opponent standings context was computed (`_compute_other_teams_sgp`)
    purely for a discarded return value — dead work, since the MC evaluates a
    roster-vs-self SGP delta. Removing that dead call must leave every MC
    aggregate byte-for-byte identical. This guards against re-introducing a
    branch that silently lets opponent totals leak into the surplus.
    """
    before = _toy_roster()
    after = {
        "1": dict(before["1"], hr=35, r=100, rbi=110),
        "2": before["2"],
    }
    all_team_totals = {
        "Rival_A": {"hr": 220, "r": 800, "rbi": 760, "sb": 90, "era": 3.90, "whip": 1.22},
        "Rival_B": {"hr": 180, "r": 720, "rbi": 690, "sb": 140, "era": 4.10, "whip": 1.30},
    }
    common = dict(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=300,
        seed=123,
    )
    without_totals = run_paired_monte_carlo(**common)
    with_totals = run_paired_monte_carlo(**common, all_team_totals=all_team_totals)

    for key in ("mc_mean", "mc_std", "mc_median", "prob_positive"):
        assert without_totals[key] == with_totals[key], (
            f"all_team_totals must not change MC '{key}': {without_totals[key]} != {with_totals[key]}"
        )
