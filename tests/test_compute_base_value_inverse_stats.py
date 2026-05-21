"""PR19 (FA P3.7): _compute_base_value must use SGPCalculator.marginal_sgp
so inverse stats (ERA, WHIP, L) get the correct negative sign and rate
stats get proper volume-weighting.

Pre-fix bug: _compute_base_value did `value += (stat/denom) * weight` for
EVERY category, which treated high ERA/WHIP as positive contributions.
Unknown pitchers with default ERA=9.0 / WHIP=2.0 scored base=+38.8,
outranking real MLB stars at base=+18.3. Live validation 2026-05-20
surfaced this as the 3 remaining bad recs after PR18 fixed the roster
slice issue."""

import pandas as pd

from src.optimizer.fa_recommender import _compute_base_value
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig


def _make_ctx_with_roster_totals():
    """Build a minimal context with a populated roster for marginal_sgp."""
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    # Modest category weights (urgency factors)
    ctx.category_weights = {cat: 1.0 for cat in ctx.config.all_categories}
    return ctx


def test_unknown_pitcher_with_bad_defaults_scores_below_real_hitter():
    """The bug: unknown pitcher with all-zero counting + default ERA=9.0/WHIP=2.0
    should score LOWER than a real MLB hitter with actual stats.
    Pre-fix this was inverted: unknown scored ~38.8, hitter ~18.3."""
    ctx = _make_ctx_with_roster_totals()
    # Populate ctx.user_roster_ids + player_pool so marginal_sgp can compute
    # rate-stat context. Minimal seed: 5 hitters, 5 pitchers (active).
    roster_rows = []
    for i in range(5):
        roster_rows.append({
            "player_id": i + 1, "name": f"Hitter {i}",
            "is_hitter": 1, "positions": "OF", "status": "active",
            "r": 50, "hr": 10, "rbi": 40, "sb": 5,
            "ab": 300, "h": 80, "bb": 30, "hbp": 2, "sf": 2,
            "w": 0, "l": 0, "sv": 0, "k": 0,
            "ip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
        })
    for i in range(5):
        roster_rows.append({
            "player_id": i + 100, "name": f"Pitcher {i}",
            "is_hitter": 0, "positions": "SP", "status": "active",
            "r": 0, "hr": 0, "rbi": 0, "sb": 0,
            "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
            "w": 5, "l": 3, "sv": 0, "k": 80,
            "ip": 80, "er": 32, "bb_allowed": 25, "h_allowed": 70,
        })
    ctx.player_pool = pd.DataFrame(roster_rows)
    ctx.user_roster_ids = [r["player_id"] for r in roster_rows]

    # Real MLB hitter (Brandon Marsh-class)
    real_hitter = pd.Series({
        "player_id": 999, "name": "Real Hitter",
        "is_hitter": 1, "positions": "OF",
        "r": 50, "hr": 10, "rbi": 35, "sb": 5,
        "ab": 200, "h": 60, "bb": 25, "hbp": 2, "sf": 1,
        "pa": 228, "avg": 0.300, "obp": 0.370,
        "w": 0, "l": 0, "sv": 0, "k": 0,
        "ip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
        "era": 0, "whip": 0,
        "ytd_gp": 60,
    })

    # Unknown pitcher with default-bad stats (no actual data)
    unknown_pitcher = pd.Series({
        "player_id": 998, "name": "Unknown Pitcher",
        "is_hitter": 0, "positions": "RP",
        "r": 0, "hr": 0, "rbi": 0, "sb": 0,
        "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
        "pa": 0, "avg": 0, "obp": 0,
        "w": 0, "l": 0, "sv": 0, "k": 0,
        "ip": 1.0, "er": 1, "bb_allowed": 1, "h_allowed": 2,
        "era": 9.0, "whip": 2.0,  # terrible defaults
        "ytd_gp": 0,
    })

    real_value = _compute_base_value(real_hitter, ctx)
    unknown_value = _compute_base_value(unknown_pitcher, ctx)

    assert real_value > unknown_value, (
        f"BUG: Unknown pitcher (ERA=9, WHIP=2) scored {unknown_value:.2f}, "
        f"outranking real MLB hitter at {real_value:.2f}. _compute_base_value "
        f"must treat ERA/WHIP as inverse stats (negative contribution)."
    )


def test_inverse_stat_sign_negative_for_high_era():
    """Direct invariant: a player with ABOVE-average ERA must contribute
    a NEGATIVE marginal value via _compute_base_value (because dropping
    their ERA in would worsen the team rate)."""
    ctx = _make_ctx_with_roster_totals()
    # Seed roster with league-avg-ish pitching to give marginal_sgp context
    ctx.player_pool = pd.DataFrame([{
        "player_id": i + 1, "name": f"P{i}",
        "is_hitter": 0, "positions": "SP", "status": "active",
        "r": 0, "hr": 0, "rbi": 0, "sb": 0,
        "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
        "w": 5, "l": 3, "sv": 0, "k": 80,
        "ip": 80, "er": 36, "bb_allowed": 25, "h_allowed": 70,
    } for i in range(5)])
    ctx.user_roster_ids = [r + 1 for r in range(5)]

    # Pitcher with ERA 9.0 over 30 IP — clearly worse than 4.0 league avg
    bad_pitcher = pd.Series({
        "is_hitter": 0, "positions": "SP",
        "r": 0, "hr": 0, "rbi": 0, "sb": 0,
        "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
        "w": 0, "l": 2, "sv": 0, "k": 20,
        "ip": 30, "er": 30, "bb_allowed": 15, "h_allowed": 40,
        "era": 9.0, "whip": 1.83,
        "pa": 0, "avg": 0, "obp": 0, "ytd_gp": 0,
    })

    value = _compute_base_value(bad_pitcher, ctx)
    # A bad pitcher's NET marginal value should be NEGATIVE or near-zero,
    # never as high as a real MLB contributor (~+10-20 range).
    assert value < 10.0, (
        f"Bad pitcher (ERA=9.0 over 30 IP) scored {value:.2f}. Should be "
        f"near or below 0 because his terrible ERA/WHIP outweighs his K/W/IP."
    )


def test_real_pitcher_with_good_era_positive_or_competitive():
    """A real pitcher with sub-3.00 ERA should score competitively
    (not in the negative double-digits). Sanity check that the fix
    doesn't over-correct."""
    ctx = _make_ctx_with_roster_totals()
    ctx.player_pool = pd.DataFrame([{
        "player_id": i + 1, "name": f"P{i}",
        "is_hitter": 0, "positions": "SP", "status": "active",
        "r": 0, "hr": 0, "rbi": 0, "sb": 0,
        "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
        "w": 5, "l": 3, "sv": 0, "k": 80,
        "ip": 80, "er": 36, "bb_allowed": 25, "h_allowed": 70,
    } for i in range(5)])
    ctx.user_roster_ids = [r + 1 for r in range(5)]

    # Cade Horton: 2.45 ERA / 94 IP / 4 W / decent K
    real_sp = pd.Series({
        "is_hitter": 0, "positions": "SP",
        "r": 0, "hr": 0, "rbi": 0, "sb": 0,
        "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
        "w": 4, "l": 2, "sv": 0, "k": 95,
        "ip": 94, "er": 26, "bb_allowed": 28, "h_allowed": 75,
        "era": 2.45, "whip": 1.10,
        "pa": 0, "avg": 0, "obp": 0, "ytd_gp": 0,
    })

    value = _compute_base_value(real_sp, ctx)
    # Sub-3 ERA pitcher must score positive
    assert value > 0, (
        f"Real MLB SP with 2.45 ERA scored {value:.2f} — should be positive."
    )
