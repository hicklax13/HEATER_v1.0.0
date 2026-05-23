"""Structural-invariant guard: Phase 1 weighted SGP is the grade AUTHORITY.

Bug A (2026-05-23 validation): the engine previously had two graders that
could disagree by an entire tier. Phase 1 computed a weighted-SGP grade
(e.g. "F" for surplus_sgp=-1.58), and Phase 2 Monte Carlo computed its own
grade from (mc_mean, sharpe, prob_positive). The MC's grade silently
overrode Phase 1's, so a user could see `grade=B` next to `surplus_sgp=-1.58`
in the UI with no indication that the "B" came from a completely different
computation.

After Bug A fix:
  - Phase 1 grade/verdict/confidence_pct are the AUTHORITY.
  - _run_mc_overlay renames its grade/verdict/confidence_pct →
    mc_grade/mc_verdict/mc_confidence_pct before returning, so the merge in
    evaluate_trade cannot clobber Phase 1.
  - MC distribution metrics (mc_mean, mc_std, percentiles, cvar5, sharpe,
    prob_positive) remain attached to the result as risk diagnostics.
  - When the two graders disagree, both are visible (grade + mc_grade) for
    transparency, but the displayed grade is always Phase 1's.

This test guards against regression of that contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _build_minimal_pool() -> pd.DataFrame:
    """Two players with explicit stat lines so Phase 1 grade is deterministic."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Player A",
                "player_name": "Player A",
                "is_hitter": 1,
                "positions": "OF",
                "r": 80,
                "hr": 20,
                "rbi": 80,
                "sb": 10,
                "h": 130,
                "ab": 500,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "pa": 560,
                "avg": 0.260,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
                "adp": 50,
                "health_score": 0.95,
                "is_injured": 0,
            },
            {
                "player_id": 2,
                "name": "Player B",
                "player_name": "Player B",
                "is_hitter": 1,
                "positions": "OF",
                "r": 75,
                "hr": 25,
                "rbi": 85,
                "sb": 5,
                "h": 125,
                "ab": 500,
                "bb": 60,
                "hbp": 5,
                "sf": 5,
                "pa": 570,
                "avg": 0.250,
                "obp": 0.335,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
                "adp": 60,
                "health_score": 0.95,
                "is_injured": 0,
            },
        ]
    )


def test_mc_does_not_clobber_phase1_grade() -> None:
    """When MC's grader returns 'A+' and Phase 1 says 'F', result.grade is 'F'."""
    from src.engine.output.trade_evaluator import evaluate_trade

    pool = _build_minimal_pool()

    # Patch the MC overlay to return an EXTREMELY POSITIVE distribution
    # (grade A+, verdict ACCEPT, confidence_pct 99). If Bug A regressed,
    # this would overwrite Phase 1's grade.
    fake_mc = {
        "mc_mean": 99.99,
        "mc_std": 0.5,
        "mc_median": 99.99,
        "p5": 95.0,
        "p25": 98.0,
        "p75": 100.0,
        "p95": 100.0,
        "prob_positive": 0.999,
        "var5": 95.0,
        "cvar5": 94.0,
        "sharpe": 200.0,
        "confidence_interval": (99.98, 100.0),
        "mc_grade": "A+",  # already renamed (post-Bug-A contract)
        "mc_verdict": "ACCEPT",
        "mc_confidence_pct": 99.9,
        "convergence_quality": "excellent",
        "convergence_ess": 9000.0,
        "convergence_rhat": 1.001,
    }

    with patch(
        "src.engine.output.trade_evaluator._run_mc_overlay",
        return_value=fake_mc,
    ):
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[2],
            user_roster_ids=[1],
            player_pool=pool,
            enable_mc=True,
            n_sims=100,
            enable_context=False,
            enable_game_theory=False,
            apply_ytd_blend=False,
        )

    # Phase 1 produces a small surplus (Player B vs Player A: +5 HR +5 RBI
    # -5 R -5 SB +0.005 OBP -0.010 AVG). Whatever Phase 1 says, MC's "A+"
    # MUST NOT override it.
    p1_grade = result["grade"]
    p1_verdict = result["verdict"]
    p1_conf = result["confidence_pct"]

    assert p1_grade != "A+", (
        f"REGRESSION: MC's grade 'A+' clobbered Phase 1's grade {p1_grade!r}. "
        "Bug A is back. See src/engine/output/trade_evaluator.py:_run_mc_overlay "
        "for the rename contract."
    )
    # MC's grade should be visible as a separate diagnostic
    assert result.get("mc_grade") == "A+", (
        f"MC grade should be exposed as 'mc_grade' for diagnostic comparison, got {result.get('mc_grade')!r}"
    )
    assert result.get("mc_verdict") == "ACCEPT"
    assert result.get("mc_confidence_pct") == 99.9
    # And Phase 1's verdict/confidence are preserved
    assert p1_verdict in ("ACCEPT", "DECLINE")
    assert isinstance(p1_conf, (int, float))


def test_mc_diagnostics_still_attached() -> None:
    """MC distribution metrics still flow through as risk diagnostics."""
    from src.engine.output.trade_evaluator import evaluate_trade

    pool = _build_minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=True,
        n_sims=200,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )

    # All MC distribution metrics should be present
    for key in (
        "mc_mean",
        "mc_std",
        "mc_median",
        "p5",
        "p95",
        "prob_positive",
        "cvar5",
        "sharpe",
        "confidence_interval",
        "convergence_quality",
    ):
        assert key in result, f"Missing MC diagnostic '{key}' in result"

    # MC's own grader output should be exposed under renamed keys
    assert "mc_grade" in result
    assert "mc_verdict" in result
    assert "mc_confidence_pct" in result

    # Phase 1's grade/verdict/confidence_pct should still be at top level
    assert "grade" in result
    assert "verdict" in result
    assert "confidence_pct" in result


def test_no_mc_means_no_mc_keys() -> None:
    """When MC is disabled, the mc_grade/mc_verdict/mc_confidence_pct keys
    should not appear (they're MC-overlay-only)."""
    from src.engine.output.trade_evaluator import evaluate_trade

    pool = _build_minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )

    # Phase 1 grade is always present
    assert "grade" in result
    assert "verdict" in result
    assert "confidence_pct" in result

    # MC overlay keys should be absent (or be the Phase-1-seeded defaults)
    assert "mc_grade" not in result
    assert "mc_verdict" not in result
    assert "mc_confidence_pct" not in result


def test_mcoverlay_result_does_not_have_grade_key() -> None:
    """Direct test: _run_mc_overlay return value MUST NOT contain 'grade',
    'verdict', or 'confidence_pct' — those were renamed at the boundary."""
    from src.engine.output.trade_evaluator import _run_mc_overlay
    from src.valuation import LeagueConfig

    pool = _build_minimal_pool()
    config = LeagueConfig()

    mc_result = _run_mc_overlay(
        before_ids=[1],
        after_ids=[2],
        player_pool=pool,
        config=config,
        all_team_totals={},
        weeks_remaining=20,
        n_sims=200,
    )

    assert "grade" not in mc_result, (
        "_run_mc_overlay must rename 'grade' to 'mc_grade' to prevent "
        "result.update(mc_result) in evaluate_trade from clobbering Phase 1's grade."
    )
    assert "verdict" not in mc_result, "_run_mc_overlay must rename 'verdict' to 'mc_verdict'."
    assert "confidence_pct" not in mc_result, "_run_mc_overlay must rename 'confidence_pct' to 'mc_confidence_pct'."
    assert "mc_grade" in mc_result
    assert "mc_verdict" in mc_result
    assert "mc_confidence_pct" in mc_result
