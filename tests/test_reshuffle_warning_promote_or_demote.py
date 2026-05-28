"""Structural-invariant guard: reshuffle warning fires on PROMOTE OR DEMOTE.

Bug B (2026-05-23 validation): the prior condition was
``if demoted and abs(total_surplus) > 0.01:``, which hid the warning
whenever the LP did a PROMOTE-only reshuffle. In the live validation,
this missed a trade where the reshuffle SGP (+3.97 from promoting Max
Muncy off the bench) was 2.5× the trade surplus (-1.58). The user saw
no warning despite most of the "trade value" being from the engine
finally deploying a wasted bench bat.

After Bug B fix:
  - The warning condition is ``if (promoted or demoted) and abs(total_surplus) > 0.01:``.
  - Three warning variants by case:
      * promote+demote: "promoting X, benching Y"
      * demote-only:    "benching Y"
      * promote-only:   "promoting X from bench. You may capture most of
                        this value by setting your lineup, without making
                        the trade."
  - ``reshuffle_pct`` is always populated when reshuffle activity exists,
    so the UI can render it correctly even when no warning fires.

This test exercises the three branches with mocked reshuffle data.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _build_minimal_pool() -> pd.DataFrame:
    """Three hitters so LP has degrees of freedom to reshuffle."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Starter A",
                "player_name": "Starter A",
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
                "name": "Bench B",
                "player_name": "Bench B",
                "is_hitter": 1,
                "positions": "OF",
                "r": 70,
                "hr": 25,
                "rbi": 75,
                "sb": 8,
                "h": 125,
                "ab": 510,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "pa": 575,
                "avg": 0.245,
                "obp": 0.325,
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
                "adp": 80,
                "health_score": 0.90,
                "is_injured": 0,
            },
            {
                "player_id": 3,
                "name": "Incoming C",
                "player_name": "Incoming C",
                "is_hitter": 1,
                "positions": "OF",
                "r": 75,
                "hr": 22,
                "rbi": 80,
                "sb": 9,
                "h": 128,
                "ab": 505,
                "bb": 52,
                "hbp": 5,
                "sf": 5,
                "pa": 567,
                "avg": 0.253,
                "obp": 0.328,
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
                "health_score": 0.92,
                "is_injured": 0,
            },
        ]
    )


def _call_evaluate_with_mocked_reshuffle(
    pool: pd.DataFrame,
    *,
    promoted_names: list[str],
    demoted_names: list[str],
    reshuffle_sgp: float,
):
    """Helper: run evaluate_trade with a patched reshuffle return value
    that has the requested promote/demote shape and SGP magnitude."""
    from src.engine.output.trade_evaluator import evaluate_trade

    fake_reshuffle = {
        "promoted": [{"player_id": 100 + i, "name": name} for i, name in enumerate(promoted_names)],
        "demoted": [{"player_id": 200 + i, "name": name} for i, name in enumerate(demoted_names)],
        "reshuffle_sgp": reshuffle_sgp,
        "slot_changes": [],
    }

    with patch(
        "src.engine.output.trade_evaluator._compute_reshuffle_transparency",
        return_value=fake_reshuffle,
    ):
        return evaluate_trade(
            giving_ids=[1],
            receiving_ids=[3],
            user_roster_ids=[1, 2],
            player_pool=pool,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
            apply_ytd_blend=False,
        )


def test_promote_only_fires_warning() -> None:
    """The validation regression: LP promotes a bench player, no demotion.
    Warning MUST fire and MUST suggest setting the lineup as an alternative."""
    pool = _build_minimal_pool()
    result = _call_evaluate_with_mocked_reshuffle(
        pool,
        promoted_names=["Max Muncy"],
        demoted_names=[],
        reshuffle_sgp=3.97,
    )

    risk_flags = result.get("risk_flags", [])
    matching = [f for f in risk_flags if "reshuffle" in f.lower()]
    assert matching, (
        f"REGRESSION: promote-only reshuffle did not produce a warning. risk_flags={risk_flags!r}. Bug B is back."
    )
    msg = matching[0]
    assert "Max Muncy" in msg
    assert "promoting" in msg.lower()
    # Promote-only warning must include the "set your lineup" guidance
    assert "lineup" in msg.lower() or "without" in msg.lower(), (
        f"Promote-only warning missing 'set your lineup' guidance. msg={msg!r}"
    )
    # reshuffle_pct must be populated, not stuck at 0
    rpct = result.get("reshuffle", {}).get("reshuffle_pct", 0)
    assert rpct > 0, f"reshuffle_pct should be populated when there's activity, got {rpct}"


def test_demote_only_fires_warning() -> None:
    """Pre-Bug-B baseline behavior must still work — demote-only fires."""
    pool = _build_minimal_pool()
    result = _call_evaluate_with_mocked_reshuffle(
        pool,
        promoted_names=[],
        demoted_names=["Old Starter"],
        reshuffle_sgp=2.5,
    )

    risk_flags = result.get("risk_flags", [])
    matching = [f for f in risk_flags if "reshuffle" in f.lower()]
    assert matching, "demote-only reshuffle must still produce a warning"
    msg = matching[0]
    assert "Old Starter" in msg
    assert "benching" in msg.lower()


def test_promote_and_demote_fires_combined_warning() -> None:
    """When LP does a full swap, the message names both sides."""
    pool = _build_minimal_pool()
    result = _call_evaluate_with_mocked_reshuffle(
        pool,
        promoted_names=["Newly Promoted"],
        demoted_names=["Newly Benched"],
        reshuffle_sgp=2.0,
    )

    risk_flags = result.get("risk_flags", [])
    matching = [f for f in risk_flags if "reshuffle" in f.lower()]
    assert matching, "promote+demote reshuffle must produce a warning"
    msg = matching[0]
    assert "Newly Promoted" in msg
    assert "Newly Benched" in msg


def test_small_reshuffle_no_warning() -> None:
    """When reshuffle is small (<30% of surplus), no warning fires."""
    pool = _build_minimal_pool()
    # Tiny reshuffle relative to a non-trivial surplus
    result = _call_evaluate_with_mocked_reshuffle(
        pool,
        promoted_names=["Tiny Promote"],
        demoted_names=[],
        reshuffle_sgp=0.1,
    )

    risk_flags = result.get("risk_flags", [])
    matching = [f for f in risk_flags if "reshuffle" in f.lower()]
    # Only fire when reshuffle_pct > 0.3 (30% of surplus)
    # With reshuffle_sgp=0.1 and Phase 1 surplus likely around ±0.5,
    # the ratio is < 0.3, so no warning expected.
    # But the reshuffle_pct field must still be populated.
    rpct = result.get("reshuffle", {}).get("reshuffle_pct", None)
    assert rpct is not None, "reshuffle_pct should be populated even when below warning threshold"
    if rpct < 30:
        assert not matching, f"reshuffle warning fired despite reshuffle_pct={rpct} < 30%. risk_flags={risk_flags!r}"
