"""TE-E5: wire the injury availability model into the trade MC risk tails.

`src/engine/context/injury_process.py` implements Weibull-duration injury
sampling + `sample_season_availability(...)`, but `run_paired_monte_carlo`
never called it — the MC assumed full availability, so the downside tail
(CVaR5) understated the dominant real risk for fragile/IL players.

These tests verify:
  1. `enable_injury_mc` is a valid kwarg on `run_paired_monte_carlo` (default OFF).
  2. With injury MC ON, a fragile/low-health player widens the DOWNSIDE
     (cvar5 lower / more negative) vs injury MC OFF.
  3. Paired-MC discipline preserved: identical rosters → exact-zero delta
     even with injury MC ON.
  4. Determinism: same seed → byte-for-byte identical result with injury MC ON.
  5. Default OFF is byte-for-byte identical to the legacy behaviour (fast path
     for Trade Finder bulk scans).
"""

from __future__ import annotations

import numpy as np

from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo


def _healthy_roster() -> dict[str, dict[str, float]]:
    """A roster of fully-healthy hitters (health_score 1.0)."""
    return {
        "1": {
            "hr": 25,
            "r": 80,
            "rbi": 80,
            "sb": 10,
            "avg": 0.280,
            "obp": 0.350,
            "h": 150,
            "ab": 535,
            "pa": 600,
            "bb": 55,
            "hbp": 5,
            "sf": 5,
            "w": 0,
            "k": 0,
            "sv": 0,
            "era": 0.0,
            "whip": 0.0,
            "health_score": 1.0,
            "age": 27,
            "is_pitcher": 0,
        },
    }


def _fragile_after_roster() -> dict[str, dict[str, float]]:
    """After-roster swaps in a much more productive but FRAGILE player.

    Same elite stat line, but health_score 0.45 (very injury-prone) so the
    injury MC should carve a fat left tail into the surplus distribution.
    """
    return {
        "1": {
            "hr": 40,
            "r": 110,
            "rbi": 120,
            "sb": 20,
            "avg": 0.310,
            "obp": 0.400,
            "h": 175,
            "ab": 565,
            "pa": 640,
            "bb": 70,
            "hbp": 6,
            "sf": 5,
            "w": 0,
            "k": 0,
            "sv": 0,
            "era": 0.0,
            "whip": 0.0,
            "health_score": 0.45,
            "age": 33,
            "is_pitcher": 0,
        },
    }


def _healthy_after_roster() -> dict[str, dict[str, float]]:
    """Identical elite stat line but FULLY healthy (control)."""
    after = _fragile_after_roster()
    after["1"]["health_score"] = 1.0
    after["1"]["age"] = 27
    return after


# ── 1. Flag exists and defaults OFF ───────────────────────────────────


def test_enable_injury_mc_flag_accepted() -> None:
    """`enable_injury_mc` must be a valid kwarg; default invocation unaffected."""
    before = _healthy_roster()
    after = _healthy_after_roster()
    # Should not raise
    result = run_paired_monte_carlo(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=200,
        seed=7,
        enable_injury_mc=True,
    )
    assert np.isfinite(result["mc_mean"])
    assert np.isfinite(result["cvar5"])


# ── 2. Fragile player widens the downside tail ────────────────────────


def test_injury_mc_widens_downside_for_fragile_player() -> None:
    """A fragile after-roster has a LOWER (more negative) cvar5 with injury MC on.

    Same elite stat line, but health_score 0.45 means the injury MC samples
    availability fractions < 1.0 that scale down the counting stats, dragging
    the worst-case surplus lower than the no-injury baseline.
    """
    before = _healthy_roster()
    after = _fragile_after_roster()

    common = dict(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=4000,
        seed=2024,
    )
    off = run_paired_monte_carlo(**common, enable_injury_mc=False)
    on = run_paired_monte_carlo(**common, enable_injury_mc=True)

    # The downside tail must get worse (lower cvar5) once injury risk is modeled.
    assert on["cvar5"] < off["cvar5"], (
        f"Injury MC must widen the downside for a fragile player: cvar5 on={on['cvar5']} should be < off={off['cvar5']}"
    )
    # And the p5 (5th-percentile surplus) should also drop.
    assert on["p5"] <= off["p5"]


def test_injury_mc_minimal_effect_for_healthy_player() -> None:
    """A fully-healthy after-roster sees little/no downside widening.

    health_score 1.0 → injury probability is low, so the injury-MC cvar5
    should be close to (not dramatically below) the no-injury cvar5. This
    isolates that the widening in the fragile test is driven by health,
    not by a blanket pessimism applied to every player.
    """
    before = _healthy_roster()
    after = _healthy_after_roster()

    common = dict(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=4000,
        seed=2024,
    )
    off = run_paired_monte_carlo(**common, enable_injury_mc=False)
    on = run_paired_monte_carlo(**common, enable_injury_mc=True)

    # Healthy player: the fragile roster should widen the tail MORE than the
    # healthy roster does (relative comparison avoids brittle absolute bounds).
    fragile_after = _fragile_after_roster()
    fragile_on = run_paired_monte_carlo(
        before_roster_stats=before,
        after_roster_stats=fragile_after,
        n_sims=4000,
        seed=2024,
        enable_injury_mc=True,
    )
    fragile_off = run_paired_monte_carlo(
        before_roster_stats=before,
        after_roster_stats=fragile_after,
        n_sims=4000,
        seed=2024,
        enable_injury_mc=False,
    )
    healthy_widening = off["cvar5"] - on["cvar5"]
    fragile_widening = fragile_off["cvar5"] - fragile_on["cvar5"]
    assert fragile_widening > healthy_widening, (
        f"Fragile player tail widening ({fragile_widening:.3f}) must exceed "
        f"healthy player widening ({healthy_widening:.3f})"
    )


# ── 3. Paired-MC discipline preserved ─────────────────────────────────


def test_injury_mc_identical_rosters_zero_surplus() -> None:
    """Paired-MC discipline: identical rosters → exactly 0 surplus even with injury MC.

    Both arms share the same seed, so the SAME availability fractions are
    drawn for before and after — the injury scaling cancels exactly.
    """
    roster = _fragile_after_roster()
    result = run_paired_monte_carlo(
        before_roster_stats=roster,
        after_roster_stats=roster,
        n_sims=400,
        seed=99,
        enable_injury_mc=True,
    )
    assert abs(result["mc_mean"]) < 1e-9, (
        f"Identical rosters with injury MC must give 0 surplus, got {result['mc_mean']}"
    )
    assert abs(result["mc_std"]) < 1e-9


# ── 4. Determinism ────────────────────────────────────────────────────


def test_injury_mc_deterministic() -> None:
    """Same seed → identical result with injury MC on (reproducibility)."""
    before = _healthy_roster()
    after = _fragile_after_roster()
    kwargs = dict(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=600,
        seed=555,
        enable_injury_mc=True,
    )
    r1 = run_paired_monte_carlo(**kwargs)
    r2 = run_paired_monte_carlo(**kwargs)
    for key in ("mc_mean", "mc_std", "cvar5", "p5", "p95"):
        assert r1[key] == r2[key], f"Injury MC must be deterministic for '{key}'"


# ── 5. Default OFF is byte-for-byte legacy behaviour ──────────────────


def test_injury_mc_default_off_matches_legacy() -> None:
    """Default (no flag) must equal explicit enable_injury_mc=False, byte-for-byte.

    Guards the fast path for Trade Finder bulk scans — the new flag must not
    perturb the legacy distribution when off.
    """
    before = _healthy_roster()
    after = _fragile_after_roster()
    common = dict(
        before_roster_stats=before,
        after_roster_stats=after,
        n_sims=800,
        seed=314,
    )
    default = run_paired_monte_carlo(**common)
    explicit_off = run_paired_monte_carlo(**common, enable_injury_mc=False)
    for key in ("mc_mean", "mc_std", "mc_median", "cvar5", "p5", "p95", "prob_positive"):
        assert default[key] == explicit_off[key], (
            f"Default must match explicit OFF for '{key}': {default[key]} != {explicit_off[key]}"
        )
