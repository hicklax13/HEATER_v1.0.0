"""PR16 (FA engine overhaul P3.5): roster-construction guard.

Blocks FA adds when the user is at position-cap and blocks drops that
would leave the user with fewer than 10 active hitters or 8 active
pitchers (FourzynBurn starting-lineup minimums).
"""

import pandas as pd

from src.optimizer.fa_recommender import (
    _MIN_ACTIVE_HITTERS,
    _MIN_ACTIVE_PITCHERS,
    _POSITION_CAPS,
    _count_active_by_side,
    _count_eligible_at_position,
    _passes_roster_construction_guard,
)
from src.optimizer.shared_data_layer import OptimizerDataContext


def _hitter_row(pid, name, positions, status="active"):
    return {
        "player_id": pid,
        "name": name,
        "positions": positions,
        "is_hitter": 1,
        "status": status,
        "r": 50,
        "hr": 10,
        "rbi": 40,
        "sb": 5,
        "ab": 300,
        "h": 80,
        "bb": 30,
        "hbp": 2,
        "sf": 2,
        "avg": 0.265,
        "obp": 0.340,
    }


def _pitcher_row(pid, name, positions, status="active"):
    return {
        "player_id": pid,
        "name": name,
        "positions": positions,
        "is_hitter": 0,
        "status": status,
        "w": 6,
        "l": 3,
        "sv": 0,
        "k": 80,
        "ip": 70,
        "er": 25,
        "bb_allowed": 20,
        "h_allowed": 55,
    }


def test_position_caps_constants_match_league_config():
    """Sanity: PR16 constants match the FourzynBurn league config."""
    assert _POSITION_CAPS["C"] == 2  # 1 starter + 1 backup
    assert _POSITION_CAPS["OF"] == 4  # 3 starters + 1 backup
    assert _POSITION_CAPS["SP"] == 3  # 2 starters + 1 backup
    assert _MIN_ACTIVE_HITTERS == 10  # starting hitter count
    assert _MIN_ACTIVE_PITCHERS == 8  # starting pitcher count


def test_count_eligible_at_position_includes_il_players():
    """User spec: a 3rd catcher add must be blocked when user has Raleigh
    (IL) + Dingler — IL counts toward position-cap so 2 catchers is at cap."""
    rows = [
        _hitter_row(1, "Raleigh", "C", status="IL15"),
        _hitter_row(2, "Dingler", "C"),
    ]
    pool = pd.DataFrame(rows)
    count = _count_eligible_at_position([1, 2], pool, "C")
    assert count == 2


def test_count_eligible_at_position_multi_position_player():
    """A 2B/SS-eligible player counts toward BOTH 2B and SS."""
    rows = [
        _hitter_row(1, "Versatile", "2B,SS"),
    ]
    pool = pd.DataFrame(rows)
    assert _count_eligible_at_position([1], pool, "2B") == 1
    assert _count_eligible_at_position([1], pool, "SS") == 1
    assert _count_eligible_at_position([1], pool, "OF") == 0


def test_count_active_by_side_excludes_il():
    """IL players do NOT count toward the active-roster floor."""
    rows = [
        _hitter_row(1, "Active H", "OF"),
        _hitter_row(2, "IL H", "OF", status="IL15"),
        _pitcher_row(3, "Active P", "SP"),
        _pitcher_row(4, "IL P", "SP", status="IL10"),
    ]
    pool = pd.DataFrame(rows)
    assert _count_active_by_side([1, 2, 3, 4], pool, is_hitter=True) == 1
    assert _count_active_by_side([1, 2, 3, 4], pool, is_hitter=False) == 1


def test_count_active_by_side_includes_bn_players():
    """BN players ARE active (they're flex healthy reserves)."""
    rows = [
        _hitter_row(1, "Starter", "OF", status="active"),
        _hitter_row(2, "Benched", "OF", status="active"),  # status=active even if benched
    ]
    pool = pd.DataFrame(rows)
    assert _count_active_by_side([1, 2], pool, is_hitter=True) == 2


def test_guard_blocks_third_catcher_when_two_already_rostered():
    """User has Raleigh (IL) + Dingler at C. Engine must block adding a
    3rd C-only FA. Both at-cap positions for the FA → ALL at cap → block."""
    pool_rows = [
        _hitter_row(1, "Raleigh", "C", status="IL15"),
        _hitter_row(2, "Dingler", "C"),
        _hitter_row(3, "Some Bench OF", "OF"),
        # Add 8 more hitters and 8 pitchers so floor checks pass
        *[_hitter_row(10 + i, f"H{i}", "OF") for i in range(8)],
        *[_pitcher_row(100 + i, f"P{i}", "SP") for i in range(8)],
        # Plus the FA candidate (in pool but not rostered)
        _hitter_row(999, "Kirk", "C"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = [1, 2, 3] + list(range(10, 18)) + list(range(100, 108))
    ctx.user_roster_ids = rostered

    fa = pool[pool["player_id"] == 999].iloc[0]
    drop = pool[pool["player_id"] == 3].iloc[0]  # try to drop a deep OF

    passes, reason = _passes_roster_construction_guard(fa, drop, ctx)
    assert not passes, "3rd catcher add must be blocked — position-cap violation"
    assert "C" in reason or "position" in reason.lower()


def test_guard_allows_multi_position_fa_when_one_position_full():
    """User has 2 SS rostered. FA is 2B/SS-eligible. 2B has 0 rostered
    (under cap). Engine must allow — FA could fill 2B."""
    pool_rows = [
        _hitter_row(1, "SS1", "SS"),
        _hitter_row(2, "SS2", "SS"),
        *[_hitter_row(10 + i, f"H{i}", "OF") for i in range(9)],  # 9 more hitters
        *[_pitcher_row(100 + i, f"P{i}", "SP") for i in range(9)],
        _hitter_row(999, "Multi", "2B,SS"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = [1, 2] + list(range(10, 19)) + list(range(100, 109))
    ctx.user_roster_ids = rostered

    fa = pool[pool["player_id"] == 999].iloc[0]
    drop = pool[pool["player_id"] == 10].iloc[0]  # drop an OF

    passes, reason = _passes_roster_construction_guard(fa, drop, ctx)
    assert passes, f"Multi-position FA should pass — 2B is under cap. Reason: {reason}"


def test_guard_blocks_drop_when_below_10_active_hitters():
    """User has exactly 10 active hitters + various pitchers. Dropping any
    hitter leaves 9 — below the 10-hitter starting-lineup floor → block.

    Positions are spread across capped slots so the OF cap (4) doesn't
    fire first when the FA is a pitcher with Util/P-style flex eligibility.
    """
    # Spread 10 hitters across positions to avoid tripping the OF cap.
    hitter_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "1B", "2B"]
    pool_rows = [
        *[_hitter_row(i + 1, f"H{i + 1}", hitter_positions[i]) for i in range(10)],
        # 9 pitchers spread across SP/RP (under each cap of 3, but cap is
        # 3 each — so spread evenly).
        *[_pitcher_row(100 + i, f"P{i}", "SP" if i < 3 else "RP") for i in range(6)],
        *[_pitcher_row(106 + i, f"P{106 + i}", "P") for i in range(3)],  # flex slots
        # FA is a pitcher with no capped position (flex P only) so
        # position-cap check doesn't apply.
        _pitcher_row(999, "FA P", "P"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = list(range(1, 11)) + list(range(100, 109))
    ctx.user_roster_ids = rostered

    fa = pool[pool["player_id"] == 999].iloc[0]
    drop = pool[pool["player_id"] == 1].iloc[0]  # try to drop a hitter (the C)

    passes, reason = _passes_roster_construction_guard(fa, drop, ctx)
    assert not passes, "Drop must be blocked — leaves 9 active hitters, below floor"
    assert "hitter" in reason.lower() or "10" in reason


def test_guard_blocks_drop_when_below_8_active_pitchers():
    """User has exactly 8 active pitchers. Dropping a pitcher leaves 7 → block.

    Positions are spread to avoid tripping the OF cap (4); the FA is a
    Util-only hitter so position-cap check doesn't apply (Util not in
    ``_POSITION_CAPS``).
    """
    # 11 hitters spread across capped + flex positions
    hitter_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF", "Util", "Util"]
    pool_rows = [
        *[_hitter_row(i + 1, f"H{i + 1}", hitter_positions[i]) for i in range(11)],
        # 8 active pitchers: spread SP/RP so neither cap (3 each) fires
        *[_pitcher_row(100 + i, f"P{i}", "SP" if i < 3 else ("RP" if i < 6 else "P")) for i in range(8)],
        # FA is Util-only (no capped position)
        _hitter_row(999, "FA Util", "Util"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = list(range(1, 12)) + list(range(100, 108))
    ctx.user_roster_ids = rostered

    fa = pool[pool["player_id"] == 999].iloc[0]
    drop = pool[pool["player_id"] == 100].iloc[0]  # try to drop a pitcher (SP)

    passes, reason = _passes_roster_construction_guard(fa, drop, ctx)
    assert not passes, "Drop must be blocked — leaves 7 active pitchers, below floor"
    assert "pitcher" in reason.lower() or "8" in reason


def test_guard_allows_within_bounds_swap():
    """Sanity: a swap with plenty of buffer (12 hitters across positions,
    11 pitchers across SP/RP/P) must pass the guard cleanly.

    Positions are spread to avoid tripping any per-position cap.
    """
    hitter_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF", "Util", "1B", "2B"]
    pool_rows = [
        *[_hitter_row(i + 1, f"H{i + 1}", hitter_positions[i]) for i in range(12)],
        # 11 pitchers across SP/RP/P (3 SP + 3 RP + 5 P)
        *[_pitcher_row(100 + i, f"P{i}", "SP") for i in range(3)],
        *[_pitcher_row(103 + i, f"P{103 + i}", "RP") for i in range(3)],
        *[_pitcher_row(106 + i, f"P{106 + i}", "P") for i in range(5)],
        # FA is Util only (no capped position) - sanity that the within-
        # bounds swap passes when nothing is at cap and floors are met.
        _hitter_row(999, "FA Util", "Util"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = list(range(1, 13)) + list(range(100, 111))
    ctx.user_roster_ids = rostered

    fa = pool[pool["player_id"] == 999].iloc[0]
    drop = pool[pool["player_id"] == 10].iloc[0]  # drop the Util hitter

    passes, reason = _passes_roster_construction_guard(fa, drop, ctx)
    assert passes, f"Valid swap should pass. Reason: {reason}"


def test_guard_il_player_counts_for_position_cap_but_not_active_count():
    """Spec rule: IL players count toward the position cap (so '3rd C
    with Raleigh-IL + Dingler' is blocked) but NOT toward the active
    hitter/pitcher count (so an IL-heavy roster might still be at the
    drop floor even with many rostered)."""
    pool_rows = [
        _hitter_row(1, "Raleigh IL", "C", status="IL15"),
        _hitter_row(2, "Dingler", "C"),
        # Only 9 active hitters (1 IL + 9 active = 10 rostered, but only 9 active)
        *[_hitter_row(10 + i, f"H{i}", "OF") for i in range(9)],
        *[_pitcher_row(100 + i, f"P{i}", "SP") for i in range(10)],
        _hitter_row(999, "Kirk", "C"),
    ]
    pool = pd.DataFrame(pool_rows)
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    rostered = [1, 2] + list(range(10, 19)) + list(range(100, 110))
    ctx.user_roster_ids = rostered

    # Position cap: 2 C rostered (IL + active) → block 3rd
    fa = pool[pool["player_id"] == 999].iloc[0]
    drop_of = pool[pool["player_id"] == 10].iloc[0]
    passes, reason = _passes_roster_construction_guard(fa, drop_of, ctx)
    assert not passes
    assert "C" in reason or "position" in reason.lower()
