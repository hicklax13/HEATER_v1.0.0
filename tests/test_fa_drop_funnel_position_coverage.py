"""BR-8b (2026-06-07): the FA recommender returned ZERO add/drop
recommendations for a clearly-suboptimal roster.

Root cause confirmed via synthetic reproduction (scripts/diag_br8b_repro.py):
the position-cap guard in ``_passes_roster_construction_guard`` correctly
nets the paired drop against the add (post-PR16/PR20), so a SAME-position
upgrade passes the guard. The over-collapse is upstream in the FUNNEL:
``_score_drop_candidates`` returned only the ``_MAX_DROP_CANDIDATES`` (5)
*globally cheapest* players. When a strong FA lives at a position where all
rostered players are expensive keepers (e.g. a stud OF while the 5 cheapest
drops are all spare pitchers), no SAME-position drop ever enters the funnel.
Every cross-position pairing then trips the position cap (OF=5>4), and the
engine emits nothing — even though dropping the weakest OF for the stud OF
is a legitimate same-position upgrade the guard WOULD allow.

The fix makes ``_score_drop_candidates`` position-aware: in addition to the
global cheapest-N, it surfaces the single cheapest droppable player eligible
at each FA-candidate position, so a strong FA always has a viable
same-position partner to compete against. The cap still blocks true hoarding
(a 3rd catcher with no paired C drop).

These tests pin:
  * a suboptimal roster with a clear same-position UPGRADE FA produces >= 1
    recommendation (the cap does not starve the funnel)
  * the position-aware funnel is additive to the global cheapest-N
  * the cap boundary is preserved: a 3rd-at-a-2-slot-position add is never
    recommended via a CROSS-position drop (that would push the position to 3,
    over cap) — only a same-position upgrade (post-swap count unchanged) is
    allowed.
"""

from __future__ import annotations

import pandas as pd

from src.optimizer.fa_recommender import (
    _score_drop_candidates,
    recommend_fa_moves,
)
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig

# Full column set so _roster_category_totals (int/float casts) never sees NaN.
_ALL_COLS = {
    "r": 0,
    "hr": 0,
    "rbi": 0,
    "sb": 0,
    "ab": 0,
    "h": 0,
    "bb": 0,
    "hbp": 0,
    "sf": 0,
    "avg": 0.0,
    "obp": 0.0,
    "w": 0,
    "l": 0,
    "sv": 0,
    "k": 0,
    "ip": 0.0,
    "er": 0.0,
    "bb_allowed": 0,
    "h_allowed": 0,
    "era": 0.0,
    "whip": 0.0,
    "ytd_gp": 0,
    "ytd_pa": 0,
    "ytd_ip": 0.0,
    "percent_owned": 50,
}


def _hitter(pid, name, positions, status="active", **stats):
    base = dict(_ALL_COLS)
    base.update(
        {
            "player_id": pid,
            "name": name,
            "positions": positions,
            "is_hitter": 1,
            "status": status,
            "r": 50,
            "hr": 12,
            "rbi": 45,
            "sb": 5,
            "ab": 320,
            "h": 85,
            "bb": 32,
            "hbp": 2,
            "sf": 2,
            "avg": 0.266,
            "obp": 0.340,
            "ytd_gp": 60,
            "ytd_pa": 260,
        }
    )
    base.update(stats)
    return base


def _pitcher(pid, name, positions, status="active", **stats):
    base = dict(_ALL_COLS)
    base.update(
        {
            "player_id": pid,
            "name": name,
            "positions": positions,
            "is_hitter": 0,
            "status": status,
            "w": 6,
            "l": 4,
            "sv": 0,
            "k": 80,
            "ip": 75,
            "er": 30,
            "bb_allowed": 22,
            "h_allowed": 65,
            "era": 3.60,
            "whip": 1.16,
            "ytd_ip": 70,
        }
    )
    base.update(stats)
    return base


def _make_ctx(roster_rows, fa_rows):
    config = LeagueConfig()
    pool = pd.DataFrame(roster_rows + fa_rows)
    ctx = OptimizerDataContext()
    ctx.config = config
    ctx.player_pool = pool
    ctx.roster = pd.DataFrame(roster_rows)
    ctx.free_agents = pd.DataFrame(fa_rows)
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]
    ctx.adds_remaining_this_week = 5
    ctx.weeks_remaining = 16
    ctx.closer_count = 3
    all_cats = config.all_categories
    ctx.category_weights = {c: 1.0 for c in all_cats}
    ctx.my_totals = {c: 1.0 for c in all_cats}
    ctx.opp_totals = {c: 2.0 for c in all_cats}
    return ctx, pool


def _suboptimal_of_roster():
    """A 9th-place-style roster: hitters are all keepers (expensive to drop),
    the 5 globally-cheapest drops are spare pitchers, and the only standout
    FA is a stud OF at a position already at cap (4 OF). The legitimate move
    is to drop the WEAKEST OF for the stud OF (same-position upgrade)."""
    roster_rows = [
        _hitter(1, "C Keeper", "C", r=60, hr=18),
        _hitter(2, "1B Keeper", "1B", r=70, hr=28),
        _hitter(3, "2B Keeper", "2B", r=65, hr=15),
        _hitter(4, "3B Keeper", "3B", r=72, hr=24),
        _hitter(5, "SS Keeper", "SS", r=68, hr=16),
        _hitter(6, "OF A", "OF", r=70, hr=22),
        _hitter(7, "OF B", "OF", r=66, hr=20),
        _hitter(8, "OF C", "OF", r=64, hr=19),
        # The weakest OF is still a productive starter — the legitimate
        # same-position drop target for an even better FA, but FAR more
        # valuable than a spare flex pitcher, so it never enters the global
        # cheapest-5 drop set (which is saturated by the spare pitchers).
        _hitter(9, "OF D", "OF", r=60, hr=17, rbi=58),
        _hitter(10, "Util Keeper", "1B", r=58, hr=21),
        _hitter(11, "Bench 1B", "1B", r=40, hr=8),
        _hitter(12, "Bench 3B", "3B", r=38, hr=7),
        _pitcher(100, "SP1", "SP", w=10, k=130, ip=110, era=2.9),
        _pitcher(101, "SP2", "SP", w=9, k=120, ip=105, era=3.1),
        _pitcher(102, "SP3", "SP", w=8, k=110, ip=100, era=3.3),
        _pitcher(110, "RP1", "RP", sv=15),
        _pitcher(111, "RP2", "RP", sv=12),
        _pitcher(112, "RP3", "RP", sv=8),
        # Spare flex pitchers — near-zero production, so unambiguously the 5
        # globally cheapest drops (saturating the cheapest-5 set).
        *[
            _pitcher(
                120 + i,
                f"Spare P{i}",
                "P",
                w=0,
                l=0,
                sv=0,
                k=2,
                ip=3,
                er=3,
                bb_allowed=3,
                h_allowed=5,
                era=9.0,
                whip=2.5,
            )
            for i in range(5)
        ],
    ]
    fa_rows = [
        _hitter(
            901,
            "Stud FA OF",
            "OF",
            r=85,
            hr=32,
            rbi=95,
            sb=18,
            avg=0.315,
            obp=0.400,
            h=125,
            ab=397,
            percent_owned=20,
        ),
    ]
    return roster_rows, fa_rows


def test_same_position_upgrade_fa_produces_recommendation():
    """BR-8b: the suboptimal roster with a clear same-position OF upgrade must
    yield >= 1 recommendation. Pre-fix this returned 0 because the cheapest-5
    drops were all spare pitchers and the cap blocked every cross-position
    pairing for the stud OF."""
    roster_rows, fa_rows = _suboptimal_of_roster()
    ctx, _pool = _make_ctx(roster_rows, fa_rows)

    moves = recommend_fa_moves(ctx, max_moves=5)
    assert len(moves) >= 1, (
        "Suboptimal roster with a clear same-position OF upgrade produced no "
        "recommendations — the drop funnel starved the stud FA of a viable "
        "same-position drop partner (cap then blocked all cross-position pairs)."
    )
    # The surfaced rec should be the same-position OF upgrade (Stud OF in,
    # an OF out) — confirms the funnel offered the right partner.
    add_names = {m["add_name"] for m in moves}
    assert "Stud FA OF" in add_names


def _of_ids(drops):
    """player_ids among ``drops`` that are eligible at OF."""
    return {
        d["player_id"]
        for d in drops
        if "OF" in {p.strip().upper() for p in str(d["positions"]).replace("/", ",").split(",")}
    }


def test_drop_funnel_surfaces_cheapest_drop_at_each_fa_position():
    """The drop funnel must include a droppable player eligible at each
    FA-candidate capped position, even when no such player is in the global
    cheapest-N. Here all four OF are keepers (none in the cheapest-5), so the
    plain funnel surfaces NO OF drop; the position-aware funnel must surface
    one so the stud OF has a same-position partner."""
    roster_rows, fa_rows = _suboptimal_of_roster()
    ctx, _pool = _make_ctx(roster_rows, fa_rows)

    # Without position hints, NO OF-eligible player is among the 5 globally
    # cheapest drops (those are saturated by the near-zero spare pitchers).
    plain_drops = _score_drop_candidates(ctx, None)
    assert not _of_ids(plain_drops), (
        "Pre-condition: no OF drop should be in the global cheapest-5 "
        "(all OF are keepers; cheaper drops exist among spare pitchers). "
        f"Got OF drops: {_of_ids(plain_drops)}"
    )

    # With OF as an FA position, the funnel must surface the cheapest OF drop
    # so the stud OF has a viable same-position partner.
    pos_drops = _score_drop_candidates(ctx, None, fa_positions={"OF"})
    assert _of_ids(pos_drops), (
        "Position-aware drop funnel must surface a droppable OF when OF is an FA-candidate position."
    )
    # The global cheapest-N are still all present (additive, not replacing).
    plain_ids = {d["player_id"] for d in plain_drops}
    pos_ids = {d["player_id"] for d in pos_drops}
    assert plain_ids <= pos_ids, "Position-aware funnel must be additive to the global cheapest-N."


def test_cap_boundary_preserved_no_cross_position_hoarding():
    """Boundary preserved: the position-aware funnel must NEVER enable a true
    over-cap hoard. A 3rd catcher (post-swap C=3 > cap=2) added via a
    CROSS-position drop must stay blocked. If the engine recommends the
    catcher FA at all, it may only do so via a same-position C-for-C swap
    (drop a catcher, add a better catcher → post-swap C=2 = cap), which is a
    legitimate upgrade, not hoarding.

    The roster's two catchers are weak (so a C upgrade is plausible); the
    cheapest drops are spare pitchers. The invariant under test: any
    recommended swap that ADDS the catcher FA must DROP a catcher — never a
    pitcher or a non-C hitter (which would push C to 3, over cap)."""
    roster_rows = [
        # Two WEAK catchers — a same-position upgrade is plausible, but any
        # cross-position pairing must remain cap-blocked.
        _hitter(1, "Weak C1", "C", r=35, hr=6, rbi=24, avg=0.230, obp=0.285, h=62, ab=270),
        _hitter(2, "Weak C2", "C", r=33, hr=5, rbi=22, avg=0.225, obp=0.280, h=58, ab=258),
        _hitter(3, "1B Keeper", "1B", r=70, hr=28),
        _hitter(4, "2B Keeper", "2B", r=65, hr=15),
        _hitter(5, "3B Keeper", "3B", r=72, hr=24),
        _hitter(6, "SS Keeper", "SS", r=68, hr=16),
        _hitter(7, "OF A", "OF", r=70, hr=22),
        _hitter(8, "OF B", "OF", r=66, hr=20),
        _hitter(9, "OF C", "OF", r=64, hr=19),
        _hitter(10, "Util Keeper", "1B", r=58, hr=21),
        _hitter(11, "Bench 1B", "1B", r=50, hr=12),
        _hitter(12, "Bench 3B", "3B", r=48, hr=11),
        *[_pitcher(100 + i, f"SP{i}", "SP", w=10, k=130, ip=110, era=2.9) for i in range(3)],
        *[_pitcher(110 + i, f"RP{i}", "RP", sv=12) for i in range(3)],
        *[
            _pitcher(
                120 + i,
                f"Spare P{i}",
                "P",
                w=0,
                l=0,
                sv=0,
                k=2,
                ip=3,
                er=3,
                bb_allowed=3,
                h_allowed=5,
                era=9.0,
                whip=2.5,
            )
            for i in range(5)
        ],
    ]
    fa_rows = [
        _hitter(
            901,
            "Stud Catcher FA",
            "C",
            r=78,
            hr=33,
            rbi=92,
            sb=2,
            avg=0.305,
            obp=0.385,
            h=120,
            ab=394,
            percent_owned=15,
        ),
    ]
    ctx, _pool = _make_ctx(roster_rows, fa_rows)

    moves = recommend_fa_moves(ctx, max_moves=5)
    catcher_swaps = [m for m in moves if m["add_id"] == 901]
    for m in catcher_swaps:
        drop_positions = {p.strip().upper() for p in str(m["drop_positions"]).replace("/", ",").split(",")}
        assert "C" in drop_positions, (
            f"Catcher FA recommended via a non-C drop ({m['drop_name']}, "
            f"{m['drop_positions']}) — post-swap C=3 > cap=2 is hoarding and "
            f"must stay blocked. Cap netting must only permit a C-for-C swap."
        )
