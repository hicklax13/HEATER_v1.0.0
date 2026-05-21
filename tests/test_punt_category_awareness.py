"""P5f: FAs contributing to punted categories should NOT get composite
credit for those contributions."""

import pandas as pd

from src.optimizer.fa_recommender import _score_fa_candidates
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig


def _make_ctx(punt_cats=None):
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.urgency_weights = {"summary": {"losing": [], "tied": []}, "per_cat": {}}
    if punt_cats:
        ctx.h2h_strategy = {"punt": list(punt_cats)}
        for c in punt_cats:
            ctx.urgency_weights["per_cat"][c] = {"win_prob": 0.05}
    ctx.category_weights = {cat: 1.0 for cat in ctx.config.all_categories}
    ctx.weeks_remaining = 16
    return ctx


def test_punted_category_weight_zeroed_in_fa_scoring():
    """User punting SB: FA whose value is concentrated in SB should score
    LOWER than an equivalent-stats FA in non-punted categories."""
    sb_specialist = {
        "player_id": 1,
        "name": "Speedy",
        "player_name": "Speedy",
        "positions": "OF",
        "is_hitter": 1,
        "status": "active",
        "r": 40,
        "hr": 4,
        "rbi": 30,
        "sb": 25,  # heavy SB
        "ab": 280,
        "h": 70,
        "bb": 25,
        "hbp": 1,
        "sf": 1,
        "pa": 307,
        "avg": 0.250,
        "obp": 0.310,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "ytd_gp": 50,
    }
    other_hitter = dict(sb_specialist)
    other_hitter["player_id"] = 2
    other_hitter["name"] = "Other"
    other_hitter["player_name"] = "Other"
    # Reallocate 20 SB to 8 HR + 12 R (similar total contribution but
    # in non-SB categories)
    other_hitter["sb"] = 5
    other_hitter["hr"] = 12
    other_hitter["r"] = 52

    user_player = dict(sb_specialist)
    user_player["player_id"] = 99
    user_player["name"] = "Existing"
    user_player["player_name"] = "Existing"

    pool = pd.DataFrame([user_player, sb_specialist, other_hitter])

    # Test 1: no punt → both should score similarly
    ctx_nopunt = _make_ctx()
    ctx_nopunt.player_pool = pool
    ctx_nopunt.user_roster_ids = [99]
    ctx_nopunt.free_agents = pd.DataFrame([sb_specialist, other_hitter])
    ctx_nopunt.league_rostered_ids = set()
    ctx_nopunt.health_scores = {}
    ctx_nopunt.ownership_trends = {}
    ctx_nopunt.news_flags = {}

    candidates_no_punt = _score_fa_candidates(ctx_nopunt)
    no_punt_speedy = next(c for c in candidates_no_punt if c["name"] == "Speedy")
    no_punt_other = next(c for c in candidates_no_punt if c["name"] == "Other")

    # Test 2: punting SB → speedy should score MUCH lower than other
    ctx_punt = _make_ctx(punt_cats=["SB"])
    ctx_punt.player_pool = pool
    ctx_punt.user_roster_ids = [99]
    ctx_punt.free_agents = pd.DataFrame([sb_specialist, other_hitter])
    ctx_punt.league_rostered_ids = set()
    ctx_punt.health_scores = {}
    ctx_punt.ownership_trends = {}
    ctx_punt.news_flags = {}

    candidates_punt = _score_fa_candidates(ctx_punt)
    punt_speedy = next(c for c in candidates_punt if c["name"] == "Speedy")
    punt_other = next(c for c in candidates_punt if c["name"] == "Other")

    # The SB specialist should drop FURTHER under punt than the other FA
    speedy_drop = no_punt_speedy["composite_score"] - punt_speedy["composite_score"]
    other_drop = no_punt_other["composite_score"] - punt_other["composite_score"]
    assert speedy_drop > other_drop, (
        f"SB specialist should lose more under punt. Speedy drop {speedy_drop:.2f} vs other drop {other_drop:.2f}"
    )
