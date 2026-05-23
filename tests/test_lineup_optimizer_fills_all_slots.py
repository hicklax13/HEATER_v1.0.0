"""Structural-invariant guard: LP fills all eligible slots, not just net-positive ones.

Bug G (2026-05-23 validation): the LP's per-slot `<= 1` constraint let it
leave slots EMPTY whenever filling them would HURT the objective. For a
high-IP mediocre-ERA pitcher in P-flex, the negative ERA contribution
dominates the positive K/W/SV contribution → LP skips them.

That's locally rational (don't make ERA worse) but globally catastrophic:
Yahoo's 20 IP/week floor converts the shortfall to LOSSES across ALL
pitching cats. ANY pitcher in the lineup is better than no pitcher.

Live validation pre-fix: LP filled only 13/18 slots (2B + all 4 P-flex
empty). Hickey had 14 eligible pitchers but the LP picked 4. Weekly IP
came out to 8.11 (pre-Bug-F) or 3.74 (post-Bug-F, ROS-aware), both far
below the 20 floor.

Fix: add a per-slot SLOT_FILL_BONUS=100 to the objective. The bonus is
large enough to dominate realistic negative player contributions
(typically ±5 after normalization) but constant per assignment, so
relative SGP comparison between players for the SAME slot still wins.
Net effect: LP prefers ANY eligible player in a slot over leaving it
empty, but still picks the BEST eligible player when options exist.

Live post-fix: LP fills 17/18 slots, picks 8 pitchers (the actual P-flex
+ SP + RP capacity).
"""

from __future__ import annotations

import pandas as pd

from src.lineup_optimizer import LineupOptimizer
from src.valuation import LeagueConfig


def _make_pitcher(pid: int, name: str, positions: str, ip: float, era: float, k: int, ytd_ip: float = 0) -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 0,
        "positions": positions,
        "status": "active",
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "pa": 0,
        "avg": 0,
        "obp": 0,
        "w": 8,
        "l": 6,
        "sv": 0,
        "k": k,
        "ip": ip,
        "ytd_ip": ytd_ip,
        "er": ip * era / 9,
        "bb_allowed": int(ip * 0.3),
        "h_allowed": int(ip * 0.85),
        "era": era,
        "whip": 1.20,
    }


def _make_hitter(pid: int, name: str, positions: str) -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 1,
        "positions": positions,
        "status": "active",
        "r": 70,
        "hr": 22,
        "rbi": 75,
        "sb": 8,
        "ab": 500,
        "h": 130,
        "bb": 55,
        "hbp": 5,
        "sf": 5,
        "pa": 565,
        "avg": 0.260,
        "obp": 0.330,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "ytd_ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "era": 0,
        "whip": 0,
    }


def test_lp_fills_all_pitcher_slots_when_enough_eligible() -> None:
    """Bug G regression: with 10 eligible pitchers and 8 pitcher slots
    (2 SP + 2 RP + 4 P), the LP must fill ALL 8 — even if some are
    individually net-negative (e.g. high-IP mediocre-ERA P-flex).
    """
    rows = []
    # Full hitter lineup
    rows.append(_make_hitter(1, "C1", "C"))
    rows.append(_make_hitter(2, "FB1", "1B"))
    rows.append(_make_hitter(3, "SB1", "2B"))
    rows.append(_make_hitter(4, "TB1", "3B"))
    rows.append(_make_hitter(5, "SS1", "SS"))
    rows.append(_make_hitter(6, "OF1", "OF"))
    rows.append(_make_hitter(7, "OF2", "OF"))
    rows.append(_make_hitter(8, "OF3", "OF"))
    rows.append(_make_hitter(9, "Util1", "OF"))
    rows.append(_make_hitter(10, "Util2", "1B"))

    # 10 pitchers: 4 SP (good), 4 RP (good), 2 SP-only with BAD era (the LP
    # would skip these pre-fix because their ERA contribution is so negative)
    rows.append(_make_pitcher(11, "GoodSP1", "SP,P", ip=180, era=3.20, k=180))
    rows.append(_make_pitcher(12, "GoodSP2", "SP,P", ip=170, era=3.40, k=160))
    rows.append(_make_pitcher(13, "GoodSP3", "SP,P", ip=150, era=3.60, k=140))
    rows.append(_make_pitcher(14, "GoodSP4", "SP,P", ip=140, era=3.80, k=130))
    rows.append(_make_pitcher(15, "RP1", "RP,P", ip=60, era=2.50, k=80))
    rows.append(_make_pitcher(16, "RP2", "RP,P", ip=65, era=2.80, k=75))
    rows.append(_make_pitcher(17, "RP3", "RP,P", ip=55, era=3.10, k=70))
    rows.append(_make_pitcher(18, "RP4", "RP,P", ip=50, era=3.30, k=65))
    # The "bad ERA, high IP" pitchers that pre-Bug-G the LP would SKIP:
    rows.append(_make_pitcher(19, "MidSP1", "SP,P", ip=120, era=5.50, k=100))
    rows.append(_make_pitcher(20, "MidSP2", "SP,P", ip=110, era=5.80, k=95))

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    result = opt.optimize_lineup()
    assert result["status"] == "Optimal", f"LP must solve; got status={result['status']}"
    assignments = result["assignments"]

    # Count pitcher slot fills
    pitcher_slots = ["SP", "RP", "P"]
    pitcher_assignments = [a for a in assignments if a["slot"] in pitcher_slots]
    n_pitchers_in_lineup = len(pitcher_assignments)

    # Pool has 10 eligible pitchers, 8 pitcher slots → LP must fill all 8
    assert n_pitchers_in_lineup == 8, (
        f"Bug G regression: LP must fill all 8 pitcher slots when 10 eligible "
        f"pitchers exist. Got {n_pitchers_in_lineup}. "
        f"Pitcher assignments: {[(a['slot'], a['player_name']) for a in pitcher_assignments]}"
    )

    # Count by slot type to ensure even P-flex is filled
    p_flex_count = sum(1 for a in pitcher_assignments if a["slot"] == "P")
    assert p_flex_count == 4, (
        f"Bug G regression: all 4 P-flex slots must be filled. Got {p_flex_count}. "
        f"P-flex assignments: {[a for a in pitcher_assignments if a['slot'] == 'P']}"
    )


def test_lp_fills_partial_when_short_on_eligible_pitchers() -> None:
    """When roster has fewer pitchers than slots, fill what's available
    (no LP infeasibility)."""
    rows = []
    # Full hitter lineup
    rows.append(_make_hitter(1, "C1", "C"))
    rows.append(_make_hitter(2, "FB1", "1B"))
    rows.append(_make_hitter(3, "SB1", "2B"))
    rows.append(_make_hitter(4, "TB1", "3B"))
    rows.append(_make_hitter(5, "SS1", "SS"))
    rows.append(_make_hitter(6, "OF1", "OF"))
    rows.append(_make_hitter(7, "OF2", "OF"))
    rows.append(_make_hitter(8, "OF3", "OF"))
    rows.append(_make_hitter(9, "Util1", "OF"))
    rows.append(_make_hitter(10, "Util2", "1B"))
    # Only 3 pitchers (8 slots want)
    rows.append(_make_pitcher(11, "SP1", "SP,P", ip=180, era=3.20, k=180))
    rows.append(_make_pitcher(12, "RP1", "RP,P", ip=60, era=2.80, k=80))
    rows.append(_make_pitcher(13, "P1", "SP,P", ip=160, era=3.50, k=150))

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    result = opt.optimize_lineup()
    assert result["status"] == "Optimal", (
        f"LP must remain feasible with fewer eligible pitchers than slots; got status={result['status']}"
    )
    # Should fill 3 pitcher slots (one for each available pitcher)
    pitcher_assignments = [a for a in result["assignments"] if a["slot"] in ("SP", "RP", "P")]
    assert len(pitcher_assignments) == 3, (
        f"Expected 3 pitcher slots filled (one per available pitcher); "
        f"got {len(pitcher_assignments)}: {pitcher_assignments}"
    )


def test_lp_does_not_force_fill_when_no_eligible_at_position() -> None:
    """When NO eligible player exists for a position (e.g., no 2B-eligible
    player on roster), the slot stays empty. Should not crash or infeasibly."""
    rows = []
    rows.append(_make_hitter(1, "C1", "C"))
    rows.append(_make_hitter(2, "FB1", "1B"))
    # NO 2B-eligible player
    rows.append(_make_hitter(3, "TB1", "3B"))
    rows.append(_make_hitter(4, "SS1", "SS"))
    rows.append(_make_hitter(5, "OF1", "OF"))
    rows.append(_make_hitter(6, "OF2", "OF"))
    rows.append(_make_hitter(7, "OF3", "OF"))
    rows.append(_make_hitter(8, "Util1", "OF"))
    rows.append(_make_hitter(9, "Util2", "1B"))
    rows.append(_make_pitcher(10, "SP1", "SP,P", ip=180, era=3.20, k=180))
    rows.append(_make_pitcher(11, "SP2", "SP,P", ip=170, era=3.40, k=160))
    rows.append(_make_pitcher(12, "RP1", "RP,P", ip=60, era=2.80, k=80))
    rows.append(_make_pitcher(13, "RP2", "RP,P", ip=65, era=2.50, k=75))
    rows.append(_make_pitcher(14, "P1", "SP,P", ip=140, era=3.50, k=130))
    rows.append(_make_pitcher(15, "P2", "SP,P", ip=130, era=3.70, k=120))
    rows.append(_make_pitcher(16, "P3", "SP,P", ip=110, era=4.00, k=100))
    rows.append(_make_pitcher(17, "P4", "SP,P", ip=100, era=4.20, k=95))

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    result = opt.optimize_lineup()
    assert result["status"] == "Optimal", (
        f"LP must remain feasible when a position has no eligible player; got {result['status']}"
    )
    # 2B slot should be empty; other 17 slots should be filled
    assignments = result["assignments"]
    has_2b = any(a["slot"] == "2B" for a in assignments)
    assert not has_2b, "No 2B-eligible player → 2B slot must stay empty"
    # 17 slots filled (18 minus the 2B)
    assert len(assignments) == 17, f"Expected 17 slots filled when 2B is unfillable, got {len(assignments)}"


def test_lp_excludes_il_pitchers_from_fill_requirement() -> None:
    """IL pitchers don't count toward the 'must fill' requirement —
    LP should treat them as ineligible and not pad lineup with them."""
    rows = []
    # Full hitter lineup
    rows.append(_make_hitter(1, "C1", "C"))
    rows.append(_make_hitter(2, "FB1", "1B"))
    rows.append(_make_hitter(3, "SB1", "2B"))
    rows.append(_make_hitter(4, "TB1", "3B"))
    rows.append(_make_hitter(5, "SS1", "SS"))
    rows.append(_make_hitter(6, "OF1", "OF"))
    rows.append(_make_hitter(7, "OF2", "OF"))
    rows.append(_make_hitter(8, "OF3", "OF"))
    rows.append(_make_hitter(9, "Util1", "OF"))
    rows.append(_make_hitter(10, "Util2", "1B"))
    # 3 active pitchers + 5 IL pitchers
    rows.append(_make_pitcher(11, "SP1", "SP,P", ip=180, era=3.20, k=180))
    rows.append(_make_pitcher(12, "RP1", "RP,P", ip=60, era=2.80, k=80))
    rows.append(_make_pitcher(13, "P1", "SP,P", ip=140, era=3.50, k=130))
    for pid in range(14, 19):
        r = _make_pitcher(pid, f"IL_SP{pid}", "SP,P", ip=120, era=4.00, k=100)
        r["status"] = "il15"
        rows.append(r)

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    result = opt.optimize_lineup()
    assert result["status"] == "Optimal"
    pitcher_assignments = [a for a in result["assignments"] if a["slot"] in ("SP", "RP", "P")]
    # Only 3 ACTIVE pitchers eligible — should fill 3 slots, not pad with IL guys
    assert len(pitcher_assignments) == 3, (
        f"IL pitchers must not be force-included; expected 3, got {len(pitcher_assignments)}"
    )
    # None should be IL
    for a in pitcher_assignments:
        assert "IL_" not in a["player_name"], f"IL pitcher in lineup: {a}"
