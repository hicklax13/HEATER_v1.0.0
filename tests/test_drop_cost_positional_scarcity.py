"""PR15 (FA engine overhaul P3.5): _compute_drop_cost must apply
positional scarcity symmetrically with _compute_base_value (P1 PR3)."""

import pandas as pd
import pytest

from src.valuation import LeagueConfig
from src.waiver_wire import compute_drop_cost


def _make_pool():
    """Build a minimal pool with one SS and one OF at equivalent raw SGP."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Top SS",
                "positions": "SS",
                "is_hitter": 1,
                "r": 80,
                "hr": 20,
                "rbi": 70,
                "sb": 15,
                "ab": 500,
                "h": 140,
                "bb": 50,
                "hbp": 5,
                "sf": 3,
                "avg": 0.280,
                "obp": 0.360,
                "status": "active",
            },
            {
                "player_id": 2,
                "name": "25th OF",
                "positions": "OF",
                "is_hitter": 1,
                "r": 80,
                "hr": 20,
                "rbi": 70,
                "sb": 15,
                "ab": 500,
                "h": 140,
                "bb": 50,
                "hbp": 5,
                "sf": 3,
                "avg": 0.280,
                "obp": 0.360,
                "status": "active",
            },
        ]
    )


def test_drop_cost_no_replacement_levels_unchanged():
    """Backward-compat: when replacement_levels=None, drop costs are
    identical for SS vs OF with same raw stats (pre-PR15 behavior)."""
    pool = _make_pool()
    cfg = LeagueConfig()
    ss_cost = compute_drop_cost(1, [1, 2], pool, cfg)
    of_cost = compute_drop_cost(2, [1, 2], pool, cfg)
    # Same raw stats and same status differences in adjustments must
    # produce SAME base cost. Multi-pos bonus (3+ pos) doesn't apply here.
    assert abs(ss_cost - of_cost) < 0.5  # negligible difference (adj only)


def test_dropping_top_ss_costs_more_than_dropping_25th_of_with_replacement_levels():
    """The key PR15 invariant: dropping a player at a SCARCE position
    (low replacement level) costs MORE than dropping a player at a DEEP
    position (high replacement level) with equivalent raw SGP.

    This pins symmetry with the add-side scarcity boost from P1 PR3 —
    without it, backup catchers/SS look like free drops because the cost
    side doesn't reward their scarcity.
    """
    pool = _make_pool()
    cfg = LeagueConfig()
    # SS is scarce (low repl), OF is deep (high repl).
    replacement_levels = {"SS": 1.0, "OF": 5.0, "C": 0.5, "2B": 1.5}

    ss_cost = compute_drop_cost(1, [1, 2], pool, cfg, replacement_levels=replacement_levels)
    of_cost = compute_drop_cost(2, [1, 2], pool, cfg, replacement_levels=replacement_levels)

    assert ss_cost > of_cost, (
        f"Top-SS drop ({ss_cost:.2f}) should cost MORE than 25th-OF drop "
        f"({of_cost:.2f}) when replacement levels mark SS as scarce. "
        f"Scarcity must apply symmetrically to drops and adds."
    )


def test_drop_cost_empty_replacement_levels_unchanged():
    """Defensive: an empty dict triggers the 1.0× neutral fallback."""
    pool = _make_pool()
    cfg = LeagueConfig()
    cost_with_empty = compute_drop_cost(1, [1, 2], pool, cfg, replacement_levels={})
    cost_with_none = compute_drop_cost(1, [1, 2], pool, cfg, replacement_levels=None)
    assert abs(cost_with_empty - cost_with_none) < 0.01


def test_compute_positional_scarcity_factor_public_in_valuation():
    """Structural: PR15 moves the scarcity helper to valuation.py so
    both waiver_wire and fa_recommender can import without circular risk."""
    from src.valuation import compute_positional_scarcity_factor

    # Smoke test: SS scarce vs OF deep
    repl = {"SS": 1.0, "OF": 5.0}
    ss_factor = compute_positional_scarcity_factor("SS", repl)
    of_factor = compute_positional_scarcity_factor("OF", repl)
    assert ss_factor > of_factor
    assert 1.0 <= ss_factor <= 1.25  # capped at +25%
    assert 1.0 <= of_factor <= 1.25
