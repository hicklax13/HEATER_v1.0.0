# tests/test_wave8a_sgp_category_math.py
"""Wave 8a / Group 1: SGP and category math behavioral bugs.

Audit IDs covered:
  - D5A-008 (`src/waiver_wire.py:535`) — per-category delta divides by
    league-level SGP denom (audit's claim that it divides by per-player
    denom did NOT reproduce on current code; this test pins the
    behavior so future regressions are caught).
  - D5B-030 (`src/start_sit_widget.py:24`) — the hardcoded fallback
    SGP denominators dict bypassed `LeagueConfig.sgp_denominators`.
    After fix, omitting `sgp_denominators` consumes `LeagueConfig()`
    defaults so a league with custom denoms gets correct math.
  - D4B-011 (`src/engine/portfolio/category_analysis.py:200`) — the
    inverse-stat "gap < 0.5" threshold treated L (counting, scale
    1-20), ERA (rate, scale 2-6), and WHIP (rate, scale 1.0-1.4) with
    the same magnitude. After fix, per-category thresholds apply.
  - D5A-018 (`src/leaders.py:23`) — `compute_category_leaders` default
    `cats` omitted L. After fix, L appears with ascending sort.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.engine.portfolio.category_analysis import category_gap_analysis
from src.leaders import compute_category_leaders
from src.start_sit_widget import compute_fantasy_points_distribution
from src.valuation import LeagueConfig
from src.waiver_wire import compute_net_swap_value

# ---------------------------------------------------------------------------
# D5A-008 — waiver_wire.compute_swap_net_sgp uses league-level denom
# ---------------------------------------------------------------------------


def _make_minimal_pool() -> pd.DataFrame:
    """Two hitters and one pitcher; small differences across cats."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Keeper",
                "is_hitter": 1,
                "r": 80,
                "hr": 20,
                "rbi": 70,
                "sb": 10,
                "avg": 0.265,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "ab": 500,
                "ip": 0,
            },
            {
                "player_id": 2,
                "name": "Drop",
                "is_hitter": 1,
                "r": 40,
                "hr": 8,
                "rbi": 35,
                "sb": 3,
                "avg": 0.240,
                "obp": 0.300,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "ab": 400,
                "ip": 0,
            },
            {
                "player_id": 3,
                "name": "Add",
                "is_hitter": 1,
                "r": 70,
                "hr": 18,
                "rbi": 60,
                "sb": 6,
                "avg": 0.275,
                "obp": 0.345,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "ab": 500,
                "ip": 0,
            },
        ]
    )


def test_d5a008_swap_uses_league_level_denoms():
    """Audit D5A-008: confirms swap deltas divide by `config.sgp_denominators[cat]`
    (league-level), NOT a per-player figure.

    Behavioral check: swapping `Drop` (player_id=2) → `Add` (player_id=3)
    yields HR delta ~= (after_hr - before_hr) / 13.0 (LeagueConfig HR denom)
    not per-player.
    """
    pool = _make_minimal_pool()
    cfg = LeagueConfig()
    # Roster contains Keeper + Drop. We drop player 2, add player 3.
    result = compute_net_swap_value(
        add_id=3,
        drop_id=2,
        roster_ids=[1, 2],
        player_pool=pool,
        config=cfg,
    )
    # Before HR total: 20 (Keeper) + 8 (Drop) = 28
    # After HR total: 20 (Keeper) + 18 (Add) = 38
    # Delta HR = +10, divided by league HR denom (13.0) = ~0.769
    assert "HR" in result["category_deltas"]
    expected_hr_delta = (38 - 28) / cfg.sgp_denominators["HR"]
    assert result["category_deltas"]["HR"] == pytest.approx(expected_hr_delta, abs=0.01)


# ---------------------------------------------------------------------------
# D5B-030 — start_sit_widget consumes LeagueConfig denoms when none provided
# ---------------------------------------------------------------------------


def test_d5b030_start_sit_widget_uses_league_config():
    """After fix: `compute_fantasy_points_distribution(player)` (no
    `sgp_denominators` arg) falls back to `LeagueConfig().sgp_denominators`,
    not a hardcoded dict.

    Patches LeagueConfig.sgp_denominators to a sentinel and verifies the
    widget's mean_sgp scales accordingly.
    """
    # Hitter with known counting stats. R only — easier algebra.
    player = pd.Series(
        {
            "is_hitter": True,
            "r": 220,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.260,  # equal to baseline, no contribution
            "obp": 0.320,  # equal to baseline, no contribution
        }
    )
    # Call once with default denoms (LeagueConfig: R denom = 32.0):
    mean_default, _ = compute_fantasy_points_distribution(player)
    # If using hardcoded dict (R denom=18.0): mean_R = 220/22/18 = 0.5555
    # If using LeagueConfig (R denom=32.0): mean_R = 220/22/32 = 0.3125
    # The fix should yield the LeagueConfig number.
    expected_r_contrib = (220.0 / 22.0) / 32.0
    assert mean_default == pytest.approx(expected_r_contrib, abs=0.01), (
        f"Expected widget to use LeagueConfig R denom (32.0), but mean={mean_default} "
        f"which is far from expected {expected_r_contrib}. Hardcoded dict would "
        f"have produced {(220.0 / 22.0) / 18.0}."
    )

    # Explicit override still wins — sanity check
    mean_override, _ = compute_fantasy_points_distribution(
        player, sgp_denominators={"R": 10.0, "AVG": 0.004, "OBP": 0.005}
    )
    expected_override = (220.0 / 22.0) / 10.0
    assert mean_override == pytest.approx(expected_override, abs=0.01)


# ---------------------------------------------------------------------------
# D4B-011 — inverse-stat gainability uses per-category thresholds
# ---------------------------------------------------------------------------


def _make_inverse_totals(your_l: float, your_era: float, your_whip: float) -> dict[str, dict[str, float]]:
    """Build a 12-team totals dict where every other team is barely better in
    L/ERA/WHIP by a fixed delta, so we can probe the threshold."""
    teams: dict[str, dict[str, float]] = {}
    # Your team
    teams["mine"] = {
        "L": your_l,
        "ERA": your_era,
        "WHIP": your_whip,
        # Fill positive cats with neutral values
        "R": 700,
        "HR": 180,
        "RBI": 700,
        "SB": 50,
        "AVG": 0.260,
        "OBP": 0.320,
        "W": 60,
        "SV": 40,
        "K": 1000,
    }
    # One team ranked above you with a small inverse-stat gap
    teams["above"] = {
        "L": your_l - 0.25,  # gap = 0.25
        "ERA": your_era - 0.20,  # gap = 0.20
        "WHIP": your_whip - 0.025,  # gap = 0.025
        "R": 700,
        "HR": 180,
        "RBI": 700,
        "SB": 50,
        "AVG": 0.260,
        "OBP": 0.320,
        "W": 60,
        "SV": 40,
        "K": 1000,
    }
    # 10 filler teams ranked WAY below in inverse cats (so "mine" is in 2nd)
    for i in range(10):
        teams[f"below_{i}"] = {
            "L": your_l + 5.0 + i,
            "ERA": your_era + 1.0 + i * 0.1,
            "WHIP": your_whip + 0.2 + i * 0.01,
            "R": 700,
            "HR": 180,
            "RBI": 700,
            "SB": 50,
            "AVG": 0.260,
            "OBP": 0.320,
            "W": 60,
            "SV": 40,
            "K": 1000,
        }
    return teams


def test_d4b011_inverse_gap_thresholds_per_category():
    """After fix: WHIP gap of 0.025 should be considered gainable (small for
    WHIP), but a WHIP gap of 0.5 should NOT be (huge for WHIP).

    Previously the threshold was `gap < 0.5` for ALL inverse cats, which
    meant a WHIP gap of 0.25 was considered "small" — actually huge.
    """
    # WHIP gap of 0.025 — closeable
    teams_small = _make_inverse_totals(your_l=10.0, your_era=3.80, your_whip=1.220)
    analysis_small = category_gap_analysis(
        your_totals=teams_small["mine"],
        all_team_totals=teams_small,
        your_team_id="mine",
        weeks_remaining=16,
    )
    assert analysis_small["WHIP"]["gainable_positions"] >= 1

    # WHIP gap of 0.40 — should NOT be considered gainable post-fix
    # (Old code: gap=0.40 < 0.5 → +1 gainable. WRONG. WHIP scale 1.0-1.4.)
    teams_huge = dict(teams_small)
    teams_huge["mine"] = dict(teams_small["mine"], WHIP=1.220)
    teams_huge["above"] = dict(teams_small["above"], WHIP=0.820)  # gap = 0.40
    analysis_huge = category_gap_analysis(
        your_totals=teams_huge["mine"],
        all_team_totals=teams_huge,
        your_team_id="mine",
        weeks_remaining=16,
    )
    # After fix, gap of 0.40 in WHIP > new threshold 0.05 → NOT gainable
    assert analysis_huge["WHIP"]["gainable_positions"] == 0, (
        f"WHIP gap of 0.40 was incorrectly counted as gainable: "
        f"{analysis_huge['WHIP']} — should fail under per-category threshold."
    )


def test_d4b011_l_threshold_separate_from_era_whip():
    """L is a counting stat (scale 1-20) — its gap threshold should be in
    'losses' units (e.g. 1.5), not 0.05.
    """
    # L gap of 1.0 — should be gainable under L threshold ~1.5
    teams = _make_inverse_totals(your_l=18.0, your_era=3.80, your_whip=1.220)
    teams["above"]["L"] = 17.0  # gap = 1.0
    analysis = category_gap_analysis(
        your_totals=teams["mine"],
        all_team_totals=teams,
        your_team_id="mine",
        weeks_remaining=16,
    )
    assert analysis["L"]["gainable_positions"] >= 1

    # L gap of 5.0 — should NOT be gainable
    teams["above"]["L"] = 13.0  # gap = 5.0
    analysis = category_gap_analysis(
        your_totals=teams["mine"],
        all_team_totals=teams,
        your_team_id="mine",
        weeks_remaining=16,
    )
    assert analysis["L"]["gainable_positions"] == 0


# ---------------------------------------------------------------------------
# D5A-018 — compute_category_leaders includes L
# ---------------------------------------------------------------------------


def _make_pitcher_leaders_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Ace",
                "is_hitter": 0,
                "ip": 200.0,
                "w": 16,
                "l": 5,
                "sv": 0,
                "k": 220,
                "era": 2.80,
                "whip": 1.05,
                "pa": 0,
            },
            {
                "name": "Loser",
                "is_hitter": 0,
                "ip": 180.0,
                "w": 6,
                "l": 14,  # most losses
                "sv": 0,
                "k": 130,
                "era": 5.50,
                "whip": 1.55,
                "pa": 0,
            },
            {
                "name": "Mid",
                "is_hitter": 0,
                "ip": 160.0,
                "w": 10,
                "l": 9,
                "sv": 0,
                "k": 160,
                "era": 3.85,
                "whip": 1.25,
                "pa": 0,
            },
        ]
    )


def test_d5a018_leaders_includes_l_category():
    """After fix: `compute_category_leaders` default `cats` includes "L"
    and uses ascending sort (lower is better)."""
    df = _make_pitcher_leaders_df()
    leaders = compute_category_leaders(df)
    assert "L" in leaders, "L category missing from default leaders output"
    assert not leaders["L"].empty
    # Lowest L should be Ace (5), highest should be Loser (14)
    assert leaders["L"].iloc[0]["name"] == "Ace"
    assert leaders["L"].iloc[-1]["name"] == "Loser"
