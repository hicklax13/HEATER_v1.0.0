"""FA-E3: surface the stacked playing-time discount in the FA "why" expander.

The FA engine applies TWO playing-time discounts to a low-GP free agent:
  * ``_scale_ros_by_playing_time`` floors ROS counting cols at 0.20x
    (inside ``_compute_base_value``), and
  * ``_playing_time_multiplier`` (0.30/0.60/0.85/1.0) on the composite
    (inside ``_score_fa_candidates``).

For a ~0-GP phantom these COMPOUND to ~0.06x, which correctly suppresses the
phantom but is opaque — a high-ROS FA ranks low with no visible reason.  This
guard locks the transparency line (no scoring change): a low-GP FA's reasoning
must include a playing-time-discount line carrying the combined factor; a
full-time FA must not (or must show ~1.0x).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer.fa_recommender import (
    _combined_playing_time_factor,
    _ros_playing_time_scale,
    _score_fa_candidates,
)
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig


def _ctx_at_week(weeks_remaining: int = 16) -> OptimizerDataContext:
    """Minimal ctx with controllable season progress (Day 70 by default)."""
    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.weeks_remaining = weeks_remaining
    return ctx


# ---------------------------------------------------------------------------
# Unit: the combined factor mirrors the two applied discounts
# ---------------------------------------------------------------------------


def test_ros_scale_matches_floor_for_zero_gp():
    """A 0-GP hitter at Day 70 hits the 0.20x ROS floor."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0})
    assert _ros_playing_time_scale(fa, ctx) == pytest.approx(0.20, abs=0.01)


def test_ros_scale_full_for_healthy():
    """A healthy hitter (>=60% expected GP) keeps full ROS (1.0x)."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 50})  # ratio ~0.83
    assert _ros_playing_time_scale(fa, ctx) == pytest.approx(1.0, abs=0.01)


def test_combined_factor_compounds_for_phantom():
    """0-GP phantom: combined = ROS-scale (0.20) x composite-mult (0.30) = 0.06."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0, "ytd_ip": 0})
    assert _combined_playing_time_factor(fa, ctx) == pytest.approx(0.06, abs=0.01)


def test_combined_factor_full_for_healthy():
    """Full-time player: combined factor is ~1.0 (no discount)."""
    ctx = _ctx_at_week(16)
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 50, "ytd_ip": 0})
    assert _combined_playing_time_factor(fa, ctx) == pytest.approx(1.0, abs=0.01)


def test_combined_factor_grace_period_is_one():
    """Inside the 30-day grace window, no playing-time discount applies."""
    ctx = _ctx_at_week(23)  # Day 21
    fa = pd.Series({"is_hitter": 1, "ytd_gp": 0, "ytd_ip": 0})
    assert _combined_playing_time_factor(fa, ctx) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: low-GP candidate carries the discount, full-time one does not
# ---------------------------------------------------------------------------


def _make_ctx_with_fa(fa_overrides: dict) -> OptimizerDataContext:
    """Build a ctx with one rostered hitter + one FA hitter at Day 70."""
    ctx = _ctx_at_week(16)
    ctx.category_weights = {cat: 1.0 for cat in ctx.config.all_categories}
    ctx.urgency_weights = {"summary": {"losing": [], "tied": []}}

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
        "ytd_gp": 60,
    }
    fa = {
        "player_id": 2,
        "name": "FA1",
        "player_name": "FA1",
        "positions": "OF",
        "is_hitter": 1,
        "status": "active",
        "r": 60,
        "hr": 18,
        "rbi": 55,
        "sb": 3,
        "ab": 350,
        "h": 95,
        "bb": 35,
        "hbp": 2,
        "sf": 2,
        "pa": 384,
        "avg": 0.271,
        "obp": 0.345,
        "ytd_gp": 60,
    }
    fa.update(fa_overrides)
    ctx.player_pool = pd.DataFrame([user_player, fa])
    ctx.user_roster_ids = [1]
    ctx.free_agents = pd.DataFrame([fa])
    ctx.league_rostered_ids = set()
    ctx.health_scores = {}
    ctx.ownership_trends = {}
    ctx.news_flags = {}
    return ctx


def test_low_gp_candidate_carries_discount_field():
    """A low-GP FA candidate dict carries a < 1.0 playing_time_discount."""
    ctx = _make_ctx_with_fa({"ytd_gp": 0})
    cands = _score_fa_candidates(ctx)
    assert cands
    disc = cands[0].get("playing_time_discount")
    assert disc is not None and disc < 0.99, f"low-GP FA should carry a sub-1.0 discount, got {disc}"


def test_full_time_candidate_discount_is_one():
    """A full-time FA candidate dict carries playing_time_discount ~1.0."""
    ctx = _make_ctx_with_fa({"ytd_gp": 60})  # ratio 1.0
    cands = _score_fa_candidates(ctx)
    assert cands
    disc = cands[0].get("playing_time_discount")
    assert disc == pytest.approx(1.0, abs=0.01), f"full-time FA should carry ~1.0 discount, got {disc}"


def test_reasoning_includes_discount_line_for_low_gp():
    """_build_reasoning surfaces a playing-time discount line for a low-GP FA."""
    from src.optimizer.fa_recommender import _build_reasoning

    fa = {
        "name": "Phantom",
        "player_id": 2,
        "sustainability": 0.5,
        "ownership_delta_7d": 0,
        "playing_time_discount": 0.06,
    }
    drop = {"name": "Drop1", "player_id": 1}
    swap = {"net_sgp": 1.0, "category_deltas": {"R": 0.5, "HR": 0.5}}
    reasons = _build_reasoning(fa, drop, swap, _ctx_at_week(16))
    has_pt = any("playing-time discount" in r.lower() for r in reasons)
    assert has_pt, f"low-GP FA reasoning should include a playing-time discount line. Got: {reasons}"


def test_reasoning_omits_discount_line_for_full_time():
    """_build_reasoning omits the discount line when the factor is ~1.0."""
    from src.optimizer.fa_recommender import _build_reasoning

    fa = {
        "name": "Starter",
        "player_id": 2,
        "sustainability": 0.5,
        "ownership_delta_7d": 0,
        "playing_time_discount": 1.0,
    }
    drop = {"name": "Drop1", "player_id": 1}
    swap = {"net_sgp": 1.0, "category_deltas": {"R": 0.5, "HR": 0.5}}
    reasons = _build_reasoning(fa, drop, swap, _ctx_at_week(16))
    has_pt = any("playing-time discount" in r.lower() for r in reasons)
    assert not has_pt, f"full-time FA reasoning should NOT include a discount line. Got: {reasons}"
