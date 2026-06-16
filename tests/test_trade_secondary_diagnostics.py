"""Guards for the trade-engine secondary diagnostics shipped 2026-05-25.

Covers four report items wired into evaluate_trade():
  • Item 1 — G-score (Rosenof) trade delta        (report B.5 / C.2)
  • Item 3 — DRL replacement chain                 (report B.7 / pseudocode)
  • Item 4 — league-wide VORP/PRP secondary check  (report B.8)
  • Item 7 — specialist per-category cap           (report H.2)

These do NOT feed the grade (Phase 1 surplus remains the grade authority);
they are the report's sanity-check / transparency outputs.
"""

from __future__ import annotations

import pandas as pd

from src.analytics_context import ModuleStatus
from src.engine.output.trade_evaluator import (
    _SPECIALIST_CAP_FRACTION,
    _compute_specialist_cap_penalty,
    evaluate_trade,
)
from src.valuation import LeagueConfig, SGPCalculator

CONFIG = LeagueConfig()


def _hitter(pid: int, name: str, hr: int = 20, sb: int = 8, avg: float = 0.26, positions: str = "OF") -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 1,
        "positions": positions,
        "r": 70,
        "hr": hr,
        "rbi": 75,
        "sb": sb,
        "h": 140,
        "ab": 520,
        "bb": 50,
        "hbp": 5,
        "sf": 5,
        "pa": 580,
        "avg": avg,
        "obp": avg + 0.07,
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
        "ytd_pa": 300,
        "ytd_ip": 0,
        "adp": 100 + pid,
        "is_injured": 0,
    }


def _pitcher(pid: int, name: str, ip: int = 180, k: int = 180, positions: str = "SP") -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 0,
        "positions": positions,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "pa": 0,
        "avg": 0,
        "obp": 0,
        "w": 12,
        "l": 8,
        "sv": 0,
        "k": k,
        "ip": ip,
        "er": 70,
        "bb_allowed": 50,
        "h_allowed": 160,
        "era": 3.5,
        "whip": 1.17,
        "ytd_pa": 0,
        "ytd_ip": 90,
        "adp": 100 + pid,
        "is_injured": 0,
    }


def _full_roster_pool() -> tuple[pd.DataFrame, list[int]]:
    """A roster with enough hitters + pitchers for the LP to fill a lineup."""
    rows = [_hitter(pid, f"H{pid}", positions="OF") for pid in range(1, 13)]
    rows += [_pitcher(pid, f"P{pid}") for pid in range(13, 21)]
    # Trade chips outside the roster.
    rows.append(_hitter(50, "StarBat", hr=35, sb=20, avg=0.31))
    rows.append(_hitter(51, "ScrubBat", hr=5, sb=1, avg=0.235))
    pool = pd.DataFrame(rows)
    roster = list(range(1, 21))
    return pool, roster


# ── Item 1: G-score delta ────────────────────────────────────────────


def test_result_exposes_gscore_delta() -> None:
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "delta_g_score" in result
    assert "gscore_detail" in result
    assert isinstance(result["delta_g_score"], (int, float))
    assert "rosenof" in result["gscore_detail"]["method"].lower()
    # Receiving a clearly-better bat than the one given → positive G-delta.
    assert result["delta_g_score"] > 0


# ── Item 4: VORP secondary check ─────────────────────────────────────


def test_result_exposes_vorp_secondary_check() -> None:
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[51],
        receiving_ids=[50],
        user_roster_ids=roster + [51],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "delta_vorp_prp" in result
    assert "vorp_detail" in result
    detail = result["vorp_detail"]
    assert "receiving_total" in detail and "giving_total" in detail
    # Receiving a star, giving a scrub → positive VORP delta.
    assert result["delta_vorp_prp"] > 0


# ── Item 3: DRL replacement chain ────────────────────────────────────


def test_drl_chain_records_fa_pickup_on_two_for_one() -> None:
    """2-for-1: roster shrinks → engine picks up an FA → chain has a pickup."""
    pool, roster = _full_roster_pool()
    # Give two hitters, receive one star → net -1 slot → FA pickup.
    result = evaluate_trade(
        giving_ids=[1, 2],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    chain = result["drl_replacement_chain"]
    assert isinstance(chain, list)
    actions = {c["action"] for c in chain}
    assert "pickup" in actions
    for entry in chain:
        assert {"action", "player_id", "name", "positions", "sgp", "source"} <= set(entry)


def test_drl_chain_empty_on_equal_trade_no_crash() -> None:
    """Equal 1-for-1 trade must not crash (regression: picked_up_ids NameError)
    and produces an empty replacement chain."""
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert result["drl_replacement_chain"] == []


# ── Item 7: specialist cap ───────────────────────────────────────────


def test_specialist_cap_penalizes_extreme_single_category() -> None:
    """A received starter with an absurd single-category SGP gets capped."""
    sgp_calc = SGPCalculator(CONFIG)
    # A speed-only specialist: 120 SB in a week-equivalent projection.
    specialist = _hitter(70, "SpeedDemon", hr=2, sb=120, avg=0.230)
    balanced = _hitter(71, "Balanced", hr=20, sb=10, avg=0.270)
    pool = pd.DataFrame([specialist, balanced])
    weights = {c: 1.0 for c in CONFIG.all_categories}

    pen_spec, detail_spec = _compute_specialist_cap_penalty(
        receiving_ids=[70],
        after_starter_ids=[70],
        player_pool=pool,
        sgp_calc=sgp_calc,
        config=CONFIG,
        category_weights=weights,
    )
    pen_bal, _ = _compute_specialist_cap_penalty(
        receiving_ids=[71],
        after_starter_ids=[71],
        player_pool=pool,
        sgp_calc=sgp_calc,
        config=CONFIG,
        category_weights=weights,
    )
    assert pen_spec > 0
    assert pen_bal == 0.0
    assert "SB" in detail_spec["capped"].get("SpeedDemon", {})
    assert detail_spec["fraction"] == _SPECIALIST_CAP_FRACTION


def test_specialist_cap_skips_benched_received_player() -> None:
    """A received player who does NOT start contributes nothing → no penalty."""
    sgp_calc = SGPCalculator(CONFIG)
    specialist = _hitter(70, "SpeedDemon", hr=2, sb=120, avg=0.230)
    pool = pd.DataFrame([specialist])
    weights = {c: 1.0 for c in CONFIG.all_categories}
    pen, _ = _compute_specialist_cap_penalty(
        receiving_ids=[70],
        after_starter_ids=[],  # not a starter
        player_pool=pool,
        sgp_calc=sgp_calc,
        config=CONFIG,
        category_weights=weights,
    )
    assert pen == 0.0


def test_result_exposes_specialist_cap_keys() -> None:
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "specialist_cap_penalty" in result
    assert "specialist_cap_detail" in result
    assert isinstance(result["specialist_cap_penalty"], (int, float))


# ── Module-status reporting: precondition-not-met vs ran-found-nothing ──
#
# The analytics badge distinguishes a module that *could not apply* to this
# trade (NOT_APPLICABLE — excluded from the quality score) from one whose
# required inputs were genuinely missing (DISABLED — scored 0.0 + "missing
# inputs" warning). Two penalty modules used DISABLED for the benign cases:
#   • markov_fa_discount is GATED — on a non-roster-shrink trade it computes
#     nothing, so it doesn't apply → NOT_APPLICABLE.
#   • specialist_cap ALWAYS runs its full analysis (like replacement/
#     flexibility/ip_floor); finding no excess is a ran-and-found-nothing
#     result → EXECUTED, not DISABLED.


def test_markov_module_not_applicable_on_equal_trade() -> None:
    """Equal 1-for-1 trade → no FA pickup → markov module is NOT_APPLICABLE."""
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    ctx = result["analytics_context"]
    mod = ctx.modules["phase1_markov_fa_discount"]
    assert mod.status == ModuleStatus.NOT_APPLICABLE
    # Reason text preserved for badge transparency.
    assert "FA pickups" in (mod.fallback_reason or "")


def test_specialist_cap_module_executed_when_no_specialist() -> None:
    """No received specialist → penalty 0, but the analysis still RAN.

    The module computed a real result (no excess to cap), so it must report
    EXECUTED — matching its sibling always-run penalty modules — not DISABLED.
    """
    pool, roster = _full_roster_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[50],
        user_roster_ids=roster,
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert result["specialist_cap_penalty"] == 0  # no specialist received
    ctx = result["analytics_context"]
    assert ctx.modules["phase1_specialist_cap"].status == ModuleStatus.EXECUTED
