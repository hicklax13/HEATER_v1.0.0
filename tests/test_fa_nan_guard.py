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
