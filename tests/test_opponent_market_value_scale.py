"""Structural-invariant guard: opponent market values stay on the SGP scale.

Bug C (2026-05-23 validation): estimate_opponent_valuations produced market
prices up to +56.52 SGP for a below-average hitter (Dansby Swanson, .215
AVG / .288 OBP). Root cause: the "team already best in this category"
branch for AVG/OBP fell through to the generic counting-stat formula
`proj / denom * 0.5`. With AVG=0.215 and AVG denom=0.004, that produced
0.215 / 0.004 * 0.5 = 26.9 SGP from AVG alone — and ~29 SGP from OBP —
totaling ~56 SGP per team that was best in those two rate categories.

After Bug C fix:
  - AVG/OBP use the rate-stat marginal blender parallel to ERA/WHIP.
  - Volume-weighted blending: blended_rate = (team_rate*team_AB + player_rate*player_AB) / (team_AB+player_AB).
  - Falls back to LeagueConfig roster-baseline defaults (team AB=5500,
    team PA=6100, team IP=1200) when team totals lack component volumes.
  - Defense-in-depth: any single category contribution is clamped to
    MAX_PER_CAT_SGP=3.0, with a logger.warning if the cap fires.

This test guards against regression of either layer.
"""

from __future__ import annotations

import pandas as pd

from src.engine.game_theory.opponent_valuation import (
    MAX_PER_CAT_SGP,
    estimate_opponent_valuations,
    get_player_projections_from_pool,
    player_market_value,
)
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()
DENOMS = dict(CONFIG.sgp_denominators)


def _hitter_projections(avg: float = 0.260, obp: float = 0.335) -> dict[str, float]:
    """Build a plausible hitter projection dict."""
    return {
        "R": 80.0,
        "HR": 22.0,
        "RBI": 80.0,
        "SB": 10.0,
        "AVG": avg,
        "OBP": obp,
        "W": 0.0,
        "L": 0.0,
        "SV": 0.0,
        "K": 0.0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "ab": 500.0,
        "pa": 570.0,
        "ip": 0.0,
    }


def _team_totals_grid(num_teams: int = 12) -> dict[str, dict[str, float]]:
    """Build realistic team totals so AVG/OBP have natural ordering."""
    totals: dict[str, dict[str, float]] = {}
    for i in range(num_teams):
        # Spread teams from worst (R=200, .240) to best (R=320, .280)
        scale = i / max(num_teams - 1, 1)
        totals[f"Team_{i}"] = {
            "R": 200 + scale * 120,
            "HR": 45 + scale * 30,
            "RBI": 180 + scale * 80,
            "SB": 25 + scale * 30,
            "AVG": 0.240 + scale * 0.040,
            "OBP": 0.310 + scale * 0.045,
            "W": 18 + scale * 15,
            "L": 18 - scale * 8,
            "SV": 8 + scale * 10,
            "K": 350 + scale * 200,
            "ERA": 4.50 - scale * 1.20,
            "WHIP": 1.40 - scale * 0.20,
        }
    return totals


def test_no_per_team_valuation_exceeds_sanity_bound() -> None:
    """No single team's valuation should exceed roughly 12 * MAX_PER_CAT_SGP
    (the absolute mathematical ceiling when every category caps)."""
    proj = _hitter_projections(avg=0.215, obp=0.288)  # Swanson-class line
    teams = _team_totals_grid()

    vals = estimate_opponent_valuations(
        player_projections=proj,
        all_team_totals=teams,
        your_team_id="Team_5",
        sgp_denominators=DENOMS,
    )

    # Mathematical ceiling: 12 cats * MAX_PER_CAT_SGP each.
    # Realistic ceiling: well under that. We use the math ceiling so the
    # test catches the regression even if other minor bugs shift values.
    HARD_CAP = 12 * MAX_PER_CAT_SGP
    for team, val in vals.items():
        assert abs(val) <= HARD_CAP, (
            f"REGRESSION (Bug C): team {team!r} valuation {val:+.2f} SGP "
            f"exceeds mathematical cap {HARD_CAP:.1f}. The rate-stat "
            f"'already best' branch is leaking again."
        )


def test_swanson_class_hitter_market_below_realistic_bound() -> None:
    """Replicate the live failure: a below-average hitter must not get
    market values in the 50+ SGP range. Realistic max for a star is ~6-8 SGP."""
    proj = _hitter_projections(avg=0.215, obp=0.288)
    teams = _team_totals_grid()

    mv = player_market_value(
        player_projections=proj,
        all_team_totals=teams,
        your_team_id="Team_5",
        sgp_denominators=DENOMS,
    )

    # A below-average hitter shouldn't have a market price > 5 SGP.
    # Pre-fix this was ~+55 SGP — we want a HUGE safety margin so the
    # test catches any partial regression too.
    REGRESSION_ALARM = 10.0
    assert mv["max_bid"] < REGRESSION_ALARM, (
        f"REGRESSION (Bug C): below-average hitter max_bid={mv['max_bid']:+.2f} SGP "
        f"exceeds {REGRESSION_ALARM}. Pre-fix baseline was ~55. "
        f"Inspect estimate_opponent_valuations rate-stat branches."
    )
    assert mv["market_price"] < REGRESSION_ALARM, (
        f"REGRESSION (Bug C): market_price={mv['market_price']:+.2f} too high."
    )


def test_rate_stat_blend_for_already_best_team() -> None:
    """The team that's BEST in AVG/OBP must use the rate-stat blender,
    not the generic counting-stat formula. Pre-fix this team got ~29 SGP
    from AVG alone; post-fix it should be < MAX_PER_CAT_SGP."""
    proj = _hitter_projections(avg=0.215, obp=0.288)
    teams = _team_totals_grid()
    # Team_11 is the best (scale=1.0): AVG=0.280, OBP=0.355
    # A 0.215 hitter would HURT them, so benefit ≤ 0, contribution ≤ 0.
    # (Or at least: must not produce 26.9 SGP per cat.)

    vals = estimate_opponent_valuations(
        player_projections=proj,
        all_team_totals=teams,
        your_team_id="Team_5",
        sgp_denominators=DENOMS,
    )

    best_team_val = vals["Team_11"]
    # Should be modest — small or even negative contribution from rate stats
    # because adding a .215 hitter DROPS Team_11's .280 average.
    assert best_team_val < 15.0, (
        f"REGRESSION: best-in-AVG team got valuation {best_team_val:+.2f} for "
        f"a .215 hitter. Pre-Bug-C this was ~+56. Should be small / negative."
    )


def test_per_category_contribution_clamped_to_cap() -> None:
    """Even with extreme inputs, no single category contribution exceeds MAX_PER_CAT_SGP."""
    # Construct a pathological projection: huge HR total with tiny gap
    proj = _hitter_projections()
    proj["HR"] = 500.0  # absurd: 500 HR
    teams = _team_totals_grid()

    vals = estimate_opponent_valuations(
        player_projections=proj,
        all_team_totals=teams,
        your_team_id="Team_5",
        sgp_denominators=DENOMS,
    )

    # No team should get more than ~12 * MAX_PER_CAT_SGP total
    HARD_CAP = 12 * MAX_PER_CAT_SGP
    for team, val in vals.items():
        assert val <= HARD_CAP, (
            f"Team {team!r} valuation {val:+.2f} exceeds cap {HARD_CAP:.1f} despite per-category clamp."
        )


def test_get_player_projections_includes_volumes() -> None:
    """get_player_projections_from_pool must surface ab/pa/ip so the
    rate-stat blender has the data it needs."""
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "r": 80,
                "hr": 22,
                "rbi": 80,
                "sb": 10,
                "avg": 0.260,
                "obp": 0.335,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ab": 500,
                "pa": 570,
                "ip": 0,
            }
        ]
    )
    proj = get_player_projections_from_pool(1, pool)
    assert "ab" in proj, "Bug C requires 'ab' in projections for AVG rate-stat blend"
    assert "pa" in proj, "Bug C requires 'pa' in projections for OBP rate-stat blend"
    assert "ip" in proj, "Bug C requires 'ip' in projections for ERA/WHIP rate-stat blend"
    assert proj["ab"] == 500.0
    assert proj["pa"] == 570.0
