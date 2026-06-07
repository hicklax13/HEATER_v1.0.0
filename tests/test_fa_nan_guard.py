"""Guard: _score_fa_candidates must not raise ValueError when FA rows contain
NaN player_id or NaN is_hitter (happens after pool merge for unmatched FAs).

Root cause: build_optimizer_context merges raw Yahoo FA data with the player
pool.  For FAs whose name doesn't match any pool row the left-join leaves
is_hitter (and occasionally player_id) as NaN.  Plain int(NaN) raises
ValueError; this guard pins the _is_hitter_safe helper that prevents it.
"""

import math

import pandas as pd
import pytest

from src.optimizer.fa_recommender import _score_fa_candidates
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig
from src.waiver_wire import compute_drop_cost


def _make_ctx_nan(fa_overrides):
    """Build minimal ctx with one user player and one FA with given overrides."""
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.category_weights = {cat: 1.0 for cat in ctx.config.all_categories}
    ctx.urgency_weights = {"summary": {"losing": [], "tied": []}}
    ctx.weeks_remaining = 16

    user_player = {
        "player_id": 1,
        "name": "User1",
        "player_name": "User1",
        "positions": "OF",
        "is_hitter": 1,
        "status": "active",
        "r": 50,
        "hr": 10,
        "rbi": 40,
        "sb": 5,
        "ab": 300,
        "h": 80,
        "bb": 30,
        "hbp": 2,
        "sf": 2,
        "pa": 332,
        "avg": 0.267,
        "obp": 0.340,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "ytd_gp": 40,
    }
    fa_base = {
        "player_id": 2,
        "name": "FA_NaN_test",
        "player_name": "FA_NaN_test",
        "positions": "OF",
        "is_hitter": 1,
        "status": "active",
        "r": 30,
        "hr": 5,
        "rbi": 25,
        "sb": 2,
        "ab": 180,
        "h": 45,
        "bb": 18,
        "hbp": 1,
        "sf": 1,
        "pa": 200,
        "avg": 0.250,
        "obp": 0.320,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "ytd_gp": 25,
    }
    fa_base.update(fa_overrides)
    pool = pd.DataFrame([user_player, fa_base])
    ctx.player_pool = pool
    ctx.user_roster_ids = [1]
    ctx.free_agents = pd.DataFrame([fa_base])
    ctx.league_rostered_ids = set()
    ctx.health_scores = {}
    ctx.ownership_trends = {}
    ctx.news_flags = {}
    return ctx


def test_nan_is_hitter_does_not_raise():
    """FA row with NaN is_hitter must not raise ValueError in _score_fa_candidates."""
    ctx = _make_ctx_nan({"is_hitter": float("nan")})
    # Should not raise — NaN is_hitter defaults to True (hitter)
    result = _score_fa_candidates(ctx)
    assert isinstance(result, list)


def test_nan_player_id_skipped():
    """FA row with NaN player_id must be silently skipped (not crash)."""
    ctx = _make_ctx_nan({"player_id": float("nan")})
    result = _score_fa_candidates(ctx)
    # The NaN-id FA is skipped; result is empty (no valid FAs remaining)
    assert isinstance(result, list)
    assert all(not math.isnan(c["player_id"]) for c in result)


def test_valid_fa_still_scores_after_nan_guard():
    """The helper must not break scoring for well-formed FA rows."""
    ctx = _make_ctx_nan({})
    result = _score_fa_candidates(ctx)
    assert result, "Expected at least one scored FA"
    assert result[0]["player_id"] == 2


# ── Drop-side NaN guard (FA-C4) ───────────────────────────────────────
#
# compute_drop_cost did `int(row.get("is_hitter", 0)) == 1` with no NaN
# guard. A NaN is_hitter (pool left-join artefact, same root cause as the
# add-side guard above) raised ValueError and aborted the whole drop loop
# in compute_add_drop_recommendations. These tests pin the fix.


def _drop_pool(is_hitter_val):
    """Two-player pool where the drop candidate (id=1) has the given
    (possibly NaN) is_hitter value."""
    drop_candidate = {
        "player_id": 1,
        "name": "Dropper",
        "player_name": "Dropper",
        "positions": "OF",
        "is_hitter": is_hitter_val,
        "r": 40,
        "hr": 8,
        "rbi": 30,
        "sb": 1,
        "ab": 250,
        "h": 65,
        "bb": 22,
        "hbp": 1,
        "sf": 2,
        "avg": 0.260,
        "obp": 0.330,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    }
    keeper = {**drop_candidate, "player_id": 2, "name": "Keeper", "player_name": "Keeper"}
    return pd.DataFrame([drop_candidate, keeper])


def test_compute_drop_cost_nan_is_hitter_does_not_raise():
    """A NaN is_hitter on the drop candidate must not raise ValueError."""
    pool = _drop_pool(float("nan"))
    # Must not raise — NaN is_hitter defaults to hitter (matches add-side _is_hitter_safe)
    cost = compute_drop_cost(1, [1, 2], pool)
    assert isinstance(cost, float)


def test_compute_drop_cost_valid_is_hitter_still_works():
    """A well-formed is_hitter still computes a finite cost (no regression)."""
    pool = _drop_pool(1)
    cost = compute_drop_cost(1, [1, 2], pool)
    assert isinstance(cost, float)
    assert not math.isnan(cost)
