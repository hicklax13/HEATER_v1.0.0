"""PR10 Part A: regression_flag must influence the composite score in
_score_fa_candidates."""

import pandas as pd
import pytest

from src.optimizer.fa_recommender import _score_fa_candidates
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig


def _make_ctx_with_fa(fa_row_dict, recent_form=None):
    """Build minimal ctx with one FA and one rostered player."""
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.category_weights = {cat: 1.0 for cat in ctx.config.all_categories}
    ctx.urgency_weights = {"summary": {"losing": [], "tied": []}}
    ctx.weeks_remaining = 16

    # Build pool: 1 user player, 1 FA
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
    fa = {
        "player_id": 2,
        "name": "FA1",
        "player_name": "FA1",
        "positions": "OF",
        "is_hitter": 1,
        "status": "active",
        "r": 40,
        "hr": 12,
        "rbi": 35,
        "sb": 3,
        "ab": 250,
        "h": 65,
        "bb": 25,
        "hbp": 2,
        "sf": 1,
        "pa": 278,
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
        "ytd_gp": 35,
    }
    fa.update(fa_row_dict)
    pool = pd.DataFrame([user_player, fa])
    ctx.player_pool = pool
    ctx.user_roster_ids = [1]
    ctx.free_agents = pd.DataFrame([fa])
    ctx.league_rostered_ids = set()
    ctx.health_scores = {}
    ctx.ownership_trends = {}
    ctx.news_flags = {}
    return ctx


def test_buy_low_boosts_composite():
    """A FA with regression_flag=BUY_LOW should score HIGHER than the same
    player with no flag (5% boost)."""
    ctx_buy = _make_ctx_with_fa({"regression_flag": "BUY_LOW"})
    ctx_neutral = _make_ctx_with_fa({"regression_flag": ""})

    candidates_buy = _score_fa_candidates(ctx_buy)
    candidates_neutral = _score_fa_candidates(ctx_neutral)
    assert candidates_buy and candidates_neutral
    buy_score = candidates_buy[0]["composite_score"]
    neutral_score = candidates_neutral[0]["composite_score"]
    assert buy_score > neutral_score, (
        f"BUY_LOW should boost composite. Got buy={buy_score:.2f} vs neutral={neutral_score:.2f}"
    )
    # For positive composites ratio ≈ 1.05; for negative, buy is less-negative so
    # ratio < 1 (approximately 0.95). Test both cases symmetrically.
    if neutral_score >= 0:
        assert 1.03 < (buy_score / neutral_score) < 1.07
    else:
        assert 0.93 < (buy_score / neutral_score) < 0.97


def test_sell_high_discounts_composite():
    """A FA with regression_flag=SELL_HIGH should score LOWER than same
    player with no flag (5% discount)."""
    ctx_sell = _make_ctx_with_fa({"regression_flag": "SELL_HIGH"})
    ctx_neutral = _make_ctx_with_fa({"regression_flag": ""})
    candidates_sell = _score_fa_candidates(ctx_sell)
    candidates_neutral = _score_fa_candidates(ctx_neutral)
    assert candidates_sell and candidates_neutral
    sell_score = candidates_sell[0]["composite_score"]
    neutral_score = candidates_neutral[0]["composite_score"]
    assert sell_score < neutral_score
    # For positive composites ratio ≈ 0.95; for negative, sell is more-negative so
    # ratio > 1 (approximately 1.05). Test both cases symmetrically.
    if neutral_score >= 0:
        assert 0.93 < (sell_score / neutral_score) < 0.97
    else:
        assert 1.03 < (sell_score / neutral_score) < 1.07


def test_no_flag_no_change():
    """Empty/None regression_flag should be neutral (no adjustment)."""
    ctx_empty = _make_ctx_with_fa({"regression_flag": ""})
    ctx_none = _make_ctx_with_fa({"regression_flag": None})
    s_empty = _score_fa_candidates(ctx_empty)[0]["composite_score"]
    s_none = _score_fa_candidates(ctx_none)[0]["composite_score"]
    assert abs(s_empty - s_none) < 0.01
