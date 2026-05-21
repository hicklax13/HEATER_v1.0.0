"""PR20 (FA engine overhaul P3.8): position cap boundary semantics.

PR16's `_passes_roster_construction_guard` used `cnt >= cap` to block —
this is off-by-one. "Cap = starters + 1" means MAX ALLOWED depth, so
having exactly `cap` players at a position is healthy (full slate of
starter + 1 backup). The block should only fire when post-swap count
EXCEEDS the cap (`cnt > cap`).

Live diagnostic 2026-05-20: this off-by-one blocked all same-type
upgrade swaps where the drop and add shared a position (e.g.
Stott ← Muncy when both are 2B-eligible — post-swap 2B count stays
at 2 = cap, which is fine).
"""

import pandas as pd

from src.optimizer.fa_recommender import (
    _POSITION_CAPS,
    _passes_roster_construction_guard,
)
from src.optimizer.shared_data_layer import OptimizerDataContext


def _hitter(pid, name, positions, status="active"):
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


def _pitcher(pid, name, positions, status="active"):
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


def test_same_position_swap_at_cap_is_allowed():
    """The bug: dropping Muncy (3B-eligible) for Stott (2B-only) when
    user has 2 2B-eligible players (one of which is Muncy).

    Post-swap 2B count = 2 (one out, one in, both eligible at 2B).
    2 = cap, NOT above cap → must allow the swap.

    Pre-fix this returned False with reason 'position-cap: 2B=2>=2'."""
    # Build a roster with TWO 2B-eligible players (Muncy + another) plus
    # enough other positions to satisfy the floor checks.
    roster_rows = [
        _hitter(1, "Max Muncy", "2B,3B"),  # drop target
        _hitter(2, "Other 2B", "2B"),
        _hitter(3, "Catcher", "C"),
        _hitter(4, "1B Guy", "1B"),
        _hitter(5, "3B Guy", "3B"),
        _hitter(6, "SS Guy", "SS"),
        _hitter(7, "OF 1", "OF"),
        _hitter(8, "OF 2", "OF"),
        _hitter(9, "OF 3", "OF"),
        _hitter(10, "Util 1", "1B"),
        _hitter(11, "Util 2", "1B"),  # 11th hitter, above 10 floor
        *[_pitcher(100 + i, f"P{i}", "SP") for i in range(9)],  # 9 pitchers, above 8 floor
    ]
    fa_stott = _hitter(999, "Bryson Stott", "2B")  # FA add

    pool = pd.DataFrame(roster_rows + [fa_stott])
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]

    fa_row = pool[pool["player_id"] == 999].iloc[0]
    drop_row = pool[pool["player_id"] == 1].iloc[0]

    passes, reason = _passes_roster_construction_guard(fa_row, drop_row, ctx)
    assert passes, (
        f"Stott ← Muncy swap blocked: '{reason}'. "
        f"Post-swap 2B count stays at 2 (one out, one in, both eligible). "
        f"2 == cap, NOT over cap → must allow."
    )


def test_above_cap_still_blocked():
    """Inverse case: adding a 3rd catcher when user already has 2 must
    STILL block (post-swap count = 3 > cap = 2)."""
    roster_rows = [
        _hitter(1, "Raleigh", "C", status="IL15"),  # IL catcher counts for cap
        _hitter(2, "Dingler", "C"),
        _hitter(3, "Bench OF", "OF"),  # drop target (non-catcher)
        # Fillers for floor
        *[_hitter(10 + i, f"H{i}", "1B") for i in range(8)],
        *[_pitcher(100 + i, f"P{i}", "SP") for i in range(9)],
    ]
    fa_kirk = _hitter(999, "Alejandro Kirk", "C")  # 3rd catcher attempt
    pool = pd.DataFrame(roster_rows + [fa_kirk])
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]

    fa_row = pool[pool["player_id"] == 999].iloc[0]
    drop_row = pool[pool["player_id"] == 3].iloc[0]  # drop OF, not catcher

    passes, reason = _passes_roster_construction_guard(fa_row, drop_row, ctx)
    assert not passes, "3rd catcher add (post-swap C=3 > cap=2) must still block"
    assert "C" in reason or "cap" in reason.lower()


def test_below_cap_allowed():
    """Sanity: adding when post-swap count is below cap also passes."""
    roster_rows = [
        _hitter(1, "Some 1B", "1B"),
        _hitter(2, "drop me", "OF"),  # drop target
        # Fillers for floor
        *[_hitter(10 + i, f"H{i}", "3B") for i in range(9)],
        *[_pitcher(100 + i, f"P{i}", "SP") for i in range(9)],
    ]
    fa_new_1b = _hitter(999, "New 1B", "1B")  # adds 1B-eligible; post-swap 1B = 2 = cap
    pool = pd.DataFrame(roster_rows + [fa_new_1b])
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]

    fa_row = pool[pool["player_id"] == 999].iloc[0]
    drop_row = pool[pool["player_id"] == 2].iloc[0]

    passes, reason = _passes_roster_construction_guard(fa_row, drop_row, ctx)
    assert passes, f"Below-cap swap should pass. Reason: {reason}"


def test_cap_message_uses_strictly_greater_than():
    """Cosmetic: when blocked, the message should say `> cap`, not `>= cap`,
    to reflect that the violation is strictly exceeding."""
    roster_rows = [
        _hitter(1, "Raleigh", "C", status="IL15"),
        _hitter(2, "Dingler", "C"),
        _hitter(3, "Bench OF", "OF"),
        *[_hitter(10 + i, f"H{i}", "1B") for i in range(8)],
        *[_pitcher(100 + i, f"P{i}", "SP") for i in range(9)],
    ]
    fa_kirk = _hitter(999, "Alejandro Kirk", "C")
    pool = pd.DataFrame(roster_rows + [fa_kirk])
    ctx = OptimizerDataContext()
    ctx.player_pool = pool
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]

    fa_row = pool[pool["player_id"] == 999].iloc[0]
    drop_row = pool[pool["player_id"] == 3].iloc[0]

    passes, reason = _passes_roster_construction_guard(fa_row, drop_row, ctx)
    assert not passes
    # New format should use ">" (strict) — old buggy format was ">="
    assert ">=" not in reason, (
        f"Block reason '{reason}' contains '>=' — PR20 fixed the off-by-one. "
        f"The actual violation is strict-greater-than (post-swap exceeds cap)."
    )
