"""LO-E1: SLOT_FILL_BONUS dominance + ratio-protect posture.

The Bug-G fill bonus (src/lineup_optimizer.py) was a hardcoded constant
(100.0). Because the LP normalizes each category by its within-roster stat
stdev, a high-IP bad-ERA pitcher's normalized contribution can be far more
negative than the assumed "±5" — so a fixed constant's dominance is
roster-dependent. LO-E1 replaces it with a value computed from the actual
normalized player values so the fill bonus ALWAYS dominates by a known
margin.

Separately, in a ratio-protect posture (ERA+WHIP are locked wins) force-
filling a net-negative pure-SP P-flex candidate can FLIP a winnable ratio —
the opposite of the fill bonus's intent. So in that posture the fill bonus
is DROPPED for pure-SP P-flex candidates whose ERA contribution is
net-negative, letting that flex slot stay empty rather than damage the
locked ratio.
"""

from __future__ import annotations

import pandas as pd

from src.lineup_optimizer import LineupOptimizer
from src.valuation import LeagueConfig


def _make_pitcher(pid: int, name: str, positions: str, ip: float, era: float, k: int, w: int = 8, l: int = 6) -> dict:
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
        "w": w,
        "l": l,
        "sv": 0,
        "k": k,
        "ip": ip,
        "ytd_ip": 0,
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


def _full_hitter_lineup() -> list[dict]:
    return [
        _make_hitter(1, "C1", "C"),
        _make_hitter(2, "FB1", "1B"),
        _make_hitter(3, "SB1", "2B"),
        _make_hitter(4, "TB1", "3B"),
        _make_hitter(5, "SS1", "SS"),
        _make_hitter(6, "OF1", "OF"),
        _make_hitter(7, "OF2", "OF"),
        _make_hitter(8, "OF3", "OF"),
        _make_hitter(9, "Util1", "OF"),
        _make_hitter(10, "Util2", "1B"),
    ]


def test_fill_bonus_dominates_when_constant_would_fail() -> None:
    """LO-E1(a): the roster-relative fill bonus forces every slot to fill even
    when the normalized negative player value FAR exceeds the legacy fixed
    bonus (100).

    Discriminator: when every pitcher shares an identical bad ERA/WHIP, the
    within-roster ERA stdev is ~0, so _compute_scale_factors floors it to 1.0.
    The normalized ERA contribution then equals era × IP unnormalized (~-855
    for era 5.0 × IP 150), which dwarfs a fixed 100 bonus — so under the old
    constant the LP would leave these slots EMPTY (the exact Bug-G pathology
    the bonus was meant to prevent). A roster-relative bonus (k × max-abs
    player value) always dominates, so all 8 pitcher slots must fill.
    """
    rows = _full_hitter_lineup()
    # 8 pitchers, all identical bad ERA/WHIP → era*IP stdev≈0 → scale floored
    # to 1.0 → each pitcher's normalized value ≈ -855 (>> fixed 100 bonus).
    for pid in range(11, 19):
        pos = "SP,P" if pid <= 16 else "RP,P"
        rows.append(_make_pitcher(pid, f"BadP{pid}", pos, ip=150, era=5.00, k=80, w=5, l=10))

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    result = opt.optimize_lineup()
    assert result["status"] == "Optimal", f"LP must solve; got {result['status']}"
    pitcher_assignments = [a for a in result["assignments"] if a["slot"] in ("SP", "RP", "P")]
    # 8 eligible pitchers, 8 pitcher slots → all 8 filled despite -855 values.
    assert len(pitcher_assignments) == 8, (
        f"Roster-relative fill bonus must dominate a normalized value (~-855) "
        f"that exceeds the legacy fixed 100 bonus; expected 8 pitcher slots "
        f"filled, got {len(pitcher_assignments)}: "
        f"{[(a['slot'], a['player_name']) for a in pitcher_assignments]}"
    )


def test_ratio_protect_leaves_negative_pure_sp_pflex_empty() -> None:
    """LO-E1(b): in a ratio-protect posture, a net-negative pure-SP P-flex
    candidate is benched (its P-flex slot stays empty) rather than forced in,
    so the locked ERA/WHIP win is not flipped.

    Roster: full hitters + 2 good SP (fill SP slots) + 2 good RP (fill RP
    slots) + 2 BAD pure-SP (era 7.5, high IP → net-negative). The only
    candidates for the 4 P-flex slots are the 2 bad pure-SPs.

      - Default posture: bad SPs forced into P-flex (Bug-G fill behavior).
      - protect_ratios=True: bad SPs benched; P-flex stays empty.
    """
    rows = _full_hitter_lineup()
    rows.append(_make_pitcher(11, "GoodSP1", "SP,P", ip=180, era=3.20, k=190, w=14, l=5))
    rows.append(_make_pitcher(12, "GoodSP2", "SP,P", ip=175, era=3.30, k=180, w=13, l=6))
    rows.append(_make_pitcher(13, "GoodRP1", "RP,P", ip=65, era=2.40, k=85, w=4, l=2))
    rows.append(_make_pitcher(14, "GoodRP2", "RP,P", ip=62, era=2.60, k=80, w=3, l=2))
    # Net-negative pure-SP candidates (SP-eligible, NOT RP): only P-flex fits
    # them once the SP slots are taken.
    rows.append(_make_pitcher(15, "BadSP1", "SP,P", ip=150, era=7.50, k=40, w=2, l=15))
    rows.append(_make_pitcher(16, "BadSP2", "SP,P", ip=145, era=7.80, k=38, w=2, l=16))

    pool = pd.DataFrame(rows)

    # Default posture: the bad SPs get forced into P-flex (Bug-G fill).
    opt_default = LineupOptimizer(pool, config=LeagueConfig())
    res_default = opt_default.optimize_lineup()
    assert res_default["status"] == "Optimal"
    default_pflex = [a for a in res_default["assignments"] if a["slot"] == "P"]
    assert len(default_pflex) == 2, (
        f"Default posture must force-fill P-flex with the 2 available pure-SPs "
        f"(Bug-G behavior preserved); got {len(default_pflex)}: {default_pflex}"
    )

    # Ratio-protect posture: bad pure-SP P-flex left empty.
    opt_protect = LineupOptimizer(pool, config=LeagueConfig())
    res_protect = opt_protect.optimize_lineup(protect_ratios=True)
    assert res_protect["status"] == "Optimal"
    protect_pflex = [a for a in res_protect["assignments"] if a["slot"] == "P"]
    assert len(protect_pflex) == 0, (
        f"Ratio-protect posture must leave the net-negative pure-SP P-flex EMPTY "
        f"rather than flip the locked ratio; got {len(protect_pflex)}: {protect_pflex}"
    )
    # The dedicated SP + RP slots still fill — the 2 GOOD SPs (in line with the
    # staff) keep their bonus and take the SP slots, leaving only the damaging
    # surplus starters benched. RP slots are never affected by protect.
    sp_filled = [a for a in res_protect["assignments"] if a["slot"] == "SP"]
    rp_filled = [a for a in res_protect["assignments"] if a["slot"] == "RP"]
    assert len(sp_filled) == 2, f"SP slots must still fill under protect; got {sp_filled}"
    assert len(rp_filled) == 2, f"RP slots must still fill under protect; got {rp_filled}"


def test_ratio_protect_still_fills_good_pflex() -> None:
    """LO-E1(b) guard: protect_ratios must NOT bench a good (net-positive)
    pure-SP from P-flex — only net-negative ones. A strong extra starter is a
    legitimate K/W source even when protecting ratios.
    """
    rows = _full_hitter_lineup()
    rows.append(_make_pitcher(11, "GoodSP1", "SP,P", ip=180, era=3.20, k=190, w=14, l=5))
    rows.append(_make_pitcher(12, "GoodSP2", "SP,P", ip=175, era=3.30, k=180, w=13, l=6))
    rows.append(_make_pitcher(13, "GoodRP1", "RP,P", ip=65, era=2.40, k=85, w=4, l=2))
    rows.append(_make_pitcher(14, "GoodRP2", "RP,P", ip=62, era=2.60, k=80, w=3, l=2))
    # Extra pure-SPs that are GOOD (net-positive): elite ERA, high K.
    rows.append(_make_pitcher(15, "ExtraSP1", "SP,P", ip=170, era=3.10, k=185, w=13, l=5))
    rows.append(_make_pitcher(16, "ExtraSP2", "SP,P", ip=165, era=3.25, k=175, w=12, l=6))

    pool = pd.DataFrame(rows)
    opt = LineupOptimizer(pool, config=LeagueConfig())
    res = opt.optimize_lineup(protect_ratios=True)
    assert res["status"] == "Optimal"
    pflex = [a for a in res["assignments"] if a["slot"] == "P"]
    assert len(pflex) == 2, (
        f"protect_ratios must still fill P-flex with net-POSITIVE pure-SPs; got {len(pflex)}: {pflex}"
    )
