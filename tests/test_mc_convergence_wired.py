"""BUG-007 fix: _run_mc_overlay returns convergence_quality field.

src/engine/production/convergence.py defines check_convergence(), but no
production code calls it. _run_mc_overlay must invoke check_convergence
on the surplus distribution and attach quality fields to its result so
the trade UI can flag unreliable MC outputs.
"""

from __future__ import annotations

import pandas as pd


def _build_minimal_pool() -> pd.DataFrame:
    """Build a minimal player pool that satisfies _run_mc_overlay's column
    requirements. Two hitters, one swap. Mirrors the schema used by
    TestEvaluateTradeWithMC._make_pool in tests/test_trade_engine_phase2.py.
    """
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Player A",
                "team": "NYY",
                "positions": "1B",
                "is_hitter": 1,
                "is_injured": 0,
                "system": "blended",
                "pa": 600,
                "ab": 550,
                "r": 90,
                "hr": 25,
                "rbi": 80,
                "sb": 10,
                "avg": 0.275,
                "obp": 0.345,
                "h": 151,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 50,
            },
            {
                "player_id": 2,
                "name": "Player B",
                "team": "BOS",
                "positions": "OF",
                "is_hitter": 1,
                "is_injured": 0,
                "system": "blended",
                "pa": 650,
                "ab": 600,
                "r": 100,
                "hr": 30,
                "rbi": 95,
                "sb": 8,
                "avg": 0.290,
                "obp": 0.360,
                "h": 174,
                "bb": 55,
                "hbp": 4,
                "sf": 6,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 30,
            },
        ]
    )


def test_convergence_quality_in_mc_result():
    """When _run_mc_overlay runs, the result should contain a `convergence_quality`
    field with a known categorical value."""
    from src.engine.output.trade_evaluator import _run_mc_overlay
    from src.valuation import LeagueConfig

    pool = _build_minimal_pool()
    result = _run_mc_overlay(
        before_ids=[1],
        after_ids=[2],
        player_pool=pool,
        config=LeagueConfig(),
        all_team_totals={},
        weeks_remaining=16,
        n_sims=200,
    )
    assert isinstance(result, dict), f"_run_mc_overlay returned {type(result)}"
    assert "convergence_quality" in result, (
        f"BUG-007: _run_mc_overlay result missing 'convergence_quality'. Keys present: {sorted(result.keys())}"
    )
    valid_qualities = {"excellent", "good", "marginal", "poor", "not_assessed"}
    assert result["convergence_quality"] in valid_qualities, (
        f"BUG-007: invalid convergence_quality value: {result['convergence_quality']!r}; "
        f"expected one of {valid_qualities}"
    )


def test_convergence_check_function_actually_invoked():
    """Verify `check_convergence` is called from _run_mc_overlay (not just
    a fake field assignment)."""
    from unittest.mock import patch

    import src.engine.production.convergence as conv_mod
    from src.engine.output.trade_evaluator import _run_mc_overlay
    from src.valuation import LeagueConfig

    pool = _build_minimal_pool()
    with patch.object(conv_mod, "check_convergence", wraps=conv_mod.check_convergence) as mock_cc:
        _run_mc_overlay(
            before_ids=[1],
            after_ids=[2],
            player_pool=pool,
            config=LeagueConfig(),
            all_team_totals={},
            weeks_remaining=16,
            n_sims=200,
        )
    assert mock_cc.called, "BUG-007: check_convergence was never invoked from _run_mc_overlay"


def test_convergence_ess_and_rhat_in_result():
    """Quality alone is insufficient; expose ESS and R-hat for diagnostics."""
    from src.engine.output.trade_evaluator import _run_mc_overlay
    from src.valuation import LeagueConfig

    pool = _build_minimal_pool()
    result = _run_mc_overlay(
        before_ids=[1],
        after_ids=[2],
        player_pool=pool,
        config=LeagueConfig(),
        all_team_totals={},
        weeks_remaining=16,
        n_sims=200,
    )
    assert "convergence_ess" in result, "BUG-007: missing convergence_ess"
    assert "convergence_rhat" in result, "BUG-007: missing convergence_rhat"
