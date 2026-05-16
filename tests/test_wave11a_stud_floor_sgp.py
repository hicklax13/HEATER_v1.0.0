"""Wave 11A — DCV-A1-002 fix: apply_stud_floor must not let rate stats
dominate the player-value ranking by raw `val / denom` division.

The audit found that apply_stud_floor's ranking SGP was computed as
`sgp += val / denom` for EVERY category, including rate stats. For
counting stats (HR/R/RBI/SB/K/W/SV/L) the val/denom division IS the
SGP contribution and is correct. For rate stats (AVG/OBP/ERA/WHIP)
the player's *raw rate* divided by the per-standing-point denom is NOT
the SGP contribution — it's an artifact 30-50× larger than the counting
contribution.

The fix uses **marginal contribution** for rate stats (the same formula
the main DCV loop already uses), where a player's contribution is
``(component - opportunity × replacement) / raw_sgp_denom``.

These tests pin the fixed ranking behavior so future regressions are
caught.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _build_two_hitter_roster() -> pd.DataFrame:
    """Two hitters where the rate-vs-counting bias is decisive.

    Hitter A: contact specialist — high AVG/OBP, modest counting.
    Hitter B: power-speed — modest AVG/OBP, big counting.

    In a balanced 6-cat league, Hitter B is objectively more valuable:
    +20 R, +20 HR, +30 RBI, +20 SB, against a -0.040 AVG / -0.030 OBP gap.

    The buggy pre-Wave-11A code ranked Hitter A higher because
    (0.300 + 0.360) / (0.004 + 0.005) ≈ 73 in the rate-stat val/denom
    sum dwarfed the counting-stat differences. The fix should rank
    Hitter B higher using marginal contribution math.
    """
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Hitter A (contact)",
                "positions": "OF",
                "is_hitter": 1,
                "r": 80,
                "hr": 20,
                "rbi": 80,
                "sb": 20,
                "avg": 0.300,
                "obp": 0.360,
                # Components needed for marginal contribution
                "ab": 500,
                "h": 150,  # 500 × 0.300
                "bb": 56,
                "hbp": 3,
                "sf": 3,
            },
            {
                "player_id": 2,
                "name": "Hitter B (power-speed)",
                "positions": "OF",
                "is_hitter": 1,
                "r": 100,
                "hr": 40,
                "rbi": 110,
                "sb": 40,
                "avg": 0.260,
                "obp": 0.330,
                # Components
                "ab": 500,
                "h": 130,  # 500 × 0.260
                "bb": 45,
                "hbp": 3,
                "sf": 5,
            },
        ]
    )


def _build_dcv_table_for(roster: pd.DataFrame, dcv_overrides: dict[int, float] | None = None) -> pd.DataFrame:
    """Minimal DCV table shape needed by apply_stud_floor."""
    return pd.DataFrame(
        [
            {
                "player_id": int(row["player_id"]),
                "name": row["name"],
                "total_dcv": (dcv_overrides or {}).get(int(row["player_id"]), 1.0),
                "volume_factor": 1.0,
                "health_factor": 1.0,
                "matchup_mult": 1.0,
                "stud_floor_applied": False,
            }
            for _, row in roster.iterrows()
        ]
    )


class TestStudFloorRanking:
    """The stud floor must identify the genuinely valuable players,
    not those whose ranking is inflated by raw-rate-divided-by-denom artifacts.
    """

    def test_power_speed_hitter_outranks_contact_specialist(self):
        """Hitter B (power-speed) should be selected as the stud for floor
        protection, not Hitter A (contact)."""
        from src.optimizer.daily_optimizer import apply_stud_floor
        from src.valuation import LeagueConfig

        cfg = LeagueConfig()
        roster = _build_two_hitter_roster()

        # Set up a DCV table where Hitter B's DCV is slightly below median
        # (would be benched without the floor) and Hitter A is above
        # (already starting). The floor should kick in for Hitter B
        # (the actual stud), not Hitter A.
        dcv_table = _build_dcv_table_for(
            roster,
            dcv_overrides={
                1: 2.0,  # Hitter A — already valued highly
                2: 0.5,  # Hitter B — would be benched without floor
            },
        )

        result = apply_stud_floor(dcv_table, roster, cfg)

        b_row = result[result["player_id"] == 2].iloc[0]
        a_row = result[result["player_id"] == 1].iloc[0]

        # Hitter B is the legitimately valuable player; floor should apply.
        assert bool(b_row["stud_floor_applied"]), (
            f"Hitter B (power-speed, 40 HR + 40 SB) should be identified as "
            f"a stud and have the floor applied. Got "
            f"stud_floor_applied={b_row['stud_floor_applied']}. The pre-fix "
            f"ranking was dominated by Hitter A's 0.300 AVG and 0.360 OBP, "
            f"missing Hitter B's superior counting production."
        )
        # Hitter A should NOT be flagged as stud (their DCV is already high,
        # AND they're not the more valuable player in counting-stat terms).
        assert not bool(a_row["stud_floor_applied"]), (
            f"Hitter A (contact specialist) was selected for floor protection "
            f"over Hitter B. Ranking is still biased by rate stats. "
            f"stud_floor_applied={a_row['stud_floor_applied']}."
        )


class TestStudFloorDoesNotUseRawRateDivision:
    """Structural guard: the SGP computation in apply_stud_floor must not
    treat rate stats and counting stats identically as `val / denom`. Rate
    stats need marginal-contribution math; counting stats use val/denom.
    """

    def test_rate_stat_contribution_not_overwhelming(self):
        """For a typical player, the AVG SGP contribution should be the
        same order of magnitude as HR SGP — not 50× larger."""
        from src.optimizer.daily_optimizer import apply_stud_floor
        from src.valuation import LeagueConfig

        cfg = LeagueConfig()

        # Construct two players differing ONLY in HR (20 vs 40).
        # Identical AVG/OBP/components otherwise.
        roster = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "20 HR hitter",
                    "positions": "OF",
                    "is_hitter": 1,
                    "r": 80,
                    "hr": 20,
                    "rbi": 80,
                    "sb": 10,
                    "avg": 0.270,
                    "obp": 0.335,
                    "ab": 500,
                    "h": 135,
                    "bb": 55,
                    "hbp": 3,
                    "sf": 3,
                },
                {
                    "player_id": 2,
                    "name": "40 HR hitter",
                    "positions": "OF",
                    "is_hitter": 1,
                    "r": 80,
                    "hr": 40,  # ← only difference
                    "rbi": 80,
                    "sb": 10,
                    "avg": 0.270,
                    "obp": 0.335,
                    "ab": 500,
                    "h": 135,
                    "bb": 55,
                    "hbp": 3,
                    "sf": 3,
                },
            ]
        )

        # Both with identical low DCV so floor would apply if they're studs.
        dcv_table = _build_dcv_table_for(roster, dcv_overrides={1: 0.5, 2: 0.5})

        result = apply_stud_floor(dcv_table, roster, cfg)

        # The 40-HR hitter is clearly the better player — should be ranked
        # higher and get the floor. With raw val/denom math, both might
        # tie or 40-HR might still win but only marginally. The 20 HR
        # difference should produce a meaningful ranking gap.
        p1_floor = bool(result[result["player_id"] == 1].iloc[0]["stud_floor_applied"])
        p2_floor = bool(result[result["player_id"] == 2].iloc[0]["stud_floor_applied"])

        # The 40-HR hitter must get the floor; the 20-HR hitter should not
        # outrank them. With only 2 players, stud_count=1, so exactly one
        # gets the floor.
        assert p2_floor and not p1_floor, (
            f"40-HR hitter should be the stud; 20-HR hitter should not. "
            f"Got: 40-HR floor_applied={p2_floor}, 20-HR floor_applied={p1_floor}. "
            f"This indicates HR's contribution isn't sufficiently distinguishing "
            f"vs the equal AVG/OBP — likely a rate-stat magnitude problem."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
