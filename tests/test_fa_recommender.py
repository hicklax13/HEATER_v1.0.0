"""Tests for src/optimizer/fa_recommender.py — post-optimization FA recommender."""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer.fa_recommender import (
    _CATEGORY_WORSEN_THRESHOLD,
    _CLOSER_SV_THRESHOLD,
    _FLOOR_IP_MIN,
    _FLOOR_PA_MIN,
    _MAX_WORSENED_CATEGORIES,
    _MIN_CLOSERS,
    _compute_base_value,
    _compute_urgency_boost,
    _is_closer,
    _score_drop_candidates,
    _score_fa_candidates,
    recommend_fa_moves,
    recommend_streaming_moves,
)
from src.optimizer.shared_data_layer import OptimizerDataContext
from src.valuation import LeagueConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_player(
    player_id: int,
    name: str = "Player",
    is_hitter: int = 1,
    positions: str = "OF",
    hr: float = 15.0,
    r: float = 60.0,
    rbi: float = 55.0,
    sb: float = 8.0,
    avg: float = 0.265,
    obp: float = 0.330,
    sv: float = 0.0,
    w: float = 0.0,
    l: float = 0.0,
    k: float = 0.0,
    era: float = 0.0,
    whip: float = 0.0,
    ip: float = 0.0,
    pa: float = 500.0,
    ab: float = 450.0,
    h: float = 120.0,
    sf: float = 3.0,
    is_closer: bool = False,
    status: str = "",
    marginal_value: float | None = None,
) -> dict:
    d = {
        "player_id": player_id,
        "name": name,
        "positions": positions,
        "is_hitter": is_hitter,
        "hr": hr,
        "r": r,
        "rbi": rbi,
        "sb": sb,
        "avg": avg,
        "obp": obp,
        "sv": sv,
        "w": w,
        "l": l,
        "k": k,
        "era": era,
        "whip": whip,
        "ip": ip,
        "pa": pa,
        "ab": ab,
        "h": h,
        "sf": sf,
        "is_closer": is_closer,
        "status": status,
    }
    if marginal_value is not None:
        d["marginal_value"] = marginal_value
    return d


def _make_pitcher(
    player_id: int,
    name: str = "Pitcher",
    sv: float = 0.0,
    w: float = 8.0,
    l: float = 5.0,
    k: float = 120.0,
    era: float = 3.80,
    whip: float = 1.20,
    ip: float = 150.0,
    is_closer: bool = False,
    status: str = "",
    marginal_value: float | None = None,
) -> dict:
    return _make_player(
        player_id=player_id,
        name=name,
        is_hitter=0,
        positions="SP",
        hr=0,
        r=0,
        rbi=0,
        sb=0,
        avg=0,
        obp=0,
        sv=sv,
        w=w,
        l=l,
        k=k,
        era=era,
        whip=whip,
        ip=ip,
        pa=0,
        ab=0,
        h=0,
        sf=0,
        is_closer=is_closer,
        status=status,
        marginal_value=marginal_value,
    )


def _build_pool(players: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(players)


def _build_ctx(
    roster_ids: list[int] | None = None,
    pool_players: list[dict] | None = None,
    fa_players: list[dict] | None = None,
    adds_remaining: int = 10,
    closer_count: int = 2,
    il_stash_ids: set[int] | None = None,
    urgency_weights: dict | None = None,
    category_weights: dict[str, float] | None = None,
    health_scores: dict[int, float] | None = None,
    news_flags: dict[int, str] | None = None,
    ownership_trends: dict[int, dict] | None = None,
    category_gaps: dict[str, float] | None = None,
) -> OptimizerDataContext:
    config = LeagueConfig()
    pool = _build_pool(pool_players or [])
    fa = _build_pool(fa_players or [])

    if category_weights is None:
        category_weights = {c: 1.0 for c in config.all_categories}

    ctx = OptimizerDataContext(
        roster=pool[pool["player_id"].isin(roster_ids or [])] if not pool.empty else pd.DataFrame(),
        player_pool=pool,
        free_agents=fa,
        user_roster_ids=roster_ids or [],
        adds_remaining_this_week=adds_remaining,
        closer_count=closer_count,
        il_stash_ids=il_stash_ids or set(),
        urgency_weights=urgency_weights or {},
        category_weights=category_weights,
        health_scores=health_scores or {},
        news_flags=news_flags or {},
        ownership_trends=ownership_trends or {},
        category_gaps=category_gaps or {},
        config=config,
    )
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """Empty roster/pool returns empty list."""

    def test_empty_roster(self):
        ctx = _build_ctx(roster_ids=[], pool_players=[])
        assert recommend_fa_moves(ctx) == []

    def test_empty_pool(self):
        ctx = _build_ctx(roster_ids=[1], pool_players=[], fa_players=[])
        assert recommend_fa_moves(ctx) == []

    def test_empty_free_agents(self):
        hitter = _make_player(1, "Roster Guy")
        ctx = _build_ctx(roster_ids=[1], pool_players=[hitter], fa_players=[])
        assert recommend_fa_moves(ctx) == []


class TestAddsBudget:
    """No adds remaining returns empty list."""

    def test_no_adds_remaining(self):
        hitter = _make_player(1, "Roster Guy")
        fa = _make_player(100, "FA Guy", marginal_value=5.0)
        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=[hitter, fa],
            fa_players=[fa],
            adds_remaining=0,
        )
        result = recommend_fa_moves(ctx)
        assert result == []

    def test_budget_limits_results(self):
        """max_moves and adds_remaining both limit output."""
        roster = [_make_player(i, f"Roster {i}", hr=1, r=10, rbi=10, sb=0, avg=0.200, obp=0.280) for i in range(1, 6)]
        fas = [_make_player(100 + i, f"FA {i}", marginal_value=10.0 + i, hr=30, r=80, rbi=80, sb=15) for i in range(5)]
        all_players = roster + fas

        # adds_remaining=2 should limit to 2
        ctx = _build_ctx(
            roster_ids=[p["player_id"] for p in roster],
            pool_players=all_players,
            fa_players=fas,
            adds_remaining=2,
        )
        result = recommend_fa_moves(ctx, max_moves=5)
        assert len(result) <= 2


class TestILStashProtection:
    """IL stash players are never dropped."""

    def test_il_stash_never_dropped(self):
        stash = _make_player(1, "Stash Guy", hr=0, r=0, rbi=0, sb=0, avg=0, obp=0)
        normal = _make_player(2, "Normal Guy", hr=5, r=30, rbi=25, sb=2, avg=0.240, obp=0.300)
        fa = _make_player(100, "FA Star", marginal_value=10.0, hr=30, r=80, rbi=80, sb=15)
        all_players = [stash, normal, fa]

        ctx = _build_ctx(
            roster_ids=[1, 2],
            pool_players=all_players,
            fa_players=[fa],
            il_stash_ids={1},
        )
        result = recommend_fa_moves(ctx)
        for move in result:
            assert move["drop_id"] != 1, "IL stash player should never be dropped"


class TestCloserMinimum:
    """Closer minimum (2) is enforced."""

    def test_closer_not_dropped_at_minimum(self):
        closer1 = _make_pitcher(1, "Closer 1", sv=20, is_closer=True)
        closer2 = _make_pitcher(2, "Closer 2", sv=18, is_closer=True)
        filler = _make_player(3, "Filler", hr=1, r=5, rbi=5, sb=0, avg=0.200, obp=0.250)
        fa = _make_player(100, "FA Star", marginal_value=10.0, hr=30, r=80, rbi=80, sb=15)
        all_players = [closer1, closer2, filler, fa]

        ctx = _build_ctx(
            roster_ids=[1, 2, 3],
            pool_players=all_players,
            fa_players=[fa],
            closer_count=2,
        )
        result = recommend_fa_moves(ctx)
        for move in result:
            assert move["drop_id"] not in (1, 2), "Closers should not be dropped at minimum count"

    def test_closer_can_be_dropped_above_minimum(self):
        """When closer_count > 2, a closer CAN be a drop candidate."""
        closer = _make_pitcher(1, "Closer", sv=20, is_closer=True)
        filler1 = _make_player(2, "Filler1", hr=1, r=5, rbi=5, sb=0, avg=0.200, obp=0.250)
        filler2 = _make_player(3, "Filler2", hr=2, r=8, rbi=8, sb=0, avg=0.210, obp=0.260)
        fa = _make_player(100, "FA Star", marginal_value=10.0, hr=30, r=80, rbi=80, sb=15)
        all_players = [closer, filler1, filler2, fa]

        ctx = _build_ctx(
            roster_ids=[1, 2, 3],
            pool_players=all_players,
            fa_players=[fa],
            closer_count=3,  # Above minimum
        )
        drops = _score_drop_candidates(ctx)
        drop_ids = {d["player_id"] for d in drops}
        assert 1 in drop_ids, "Closer should be droppable when above minimum"


class TestCategoryWorsening:
    """3+ category worsening rejects the swap."""

    def test_three_plus_worsen_rejected(self):
        """A swap that worsens 3+ categories should be filtered out."""
        # Build a roster player who contributes broadly
        broad = _make_player(1, "Broad", hr=20, r=70, rbi=70, sb=10, avg=0.280, obp=0.350)
        # FA who is good at one thing but terrible at others
        # This test relies on the net swap actually worsening 3+ cats
        narrow = _make_player(100, "Narrow", marginal_value=2.0, hr=40, r=10, rbi=10, sb=0, avg=0.200, obp=0.260)
        all_players = [broad, narrow]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[narrow],
        )
        result = recommend_fa_moves(ctx)
        # The swap should be rejected because dropping "Broad" worsens multiple categories
        # Even if net SGP is positive, the 3-category guard should block it
        # (or net SGP <= 0 also blocks it, which is also correct behavior)
        for move in result:
            worsened = sum(1 for v in move["category_impact"].values() if v < _CATEGORY_WORSEN_THRESHOLD)
            assert worsened < _MAX_WORSENED_CATEGORIES


class TestCrossType:
    """Cross-type only allowed with positional surplus."""

    def test_cross_type_blocked_without_surplus(self):
        """Hitter-for-pitcher swap blocked when no surplus."""
        hitters = [_make_player(i, f"H{i}") for i in range(1, 14)]  # 13 hitters = exactly at capacity
        pitcher = _make_pitcher(14, "SP1", w=8, k=120, era=3.80, whip=1.20, ip=150)
        fa_pitcher = _make_pitcher(100, "FA SP", marginal_value=10.0, w=12, k=180, era=3.00, whip=1.05, ip=180)
        all_players = hitters + [pitcher, fa_pitcher]

        ctx = _build_ctx(
            roster_ids=[p["player_id"] for p in hitters] + [14],
            pool_players=all_players,
            fa_players=[fa_pitcher],
        )
        result = recommend_fa_moves(ctx)
        # Cross-type swap (adding pitcher, dropping hitter) should be blocked
        for move in result:
            # If a hitter was dropped for a pitcher, that's cross-type
            drop_match = ctx.player_pool[ctx.player_pool["player_id"] == move["drop_id"]]
            add_match = ctx.player_pool[ctx.player_pool["player_id"] == move["add_id"]]
            if not drop_match.empty and not add_match.empty:
                drop_is_hitter = bool(int(drop_match.iloc[0].get("is_hitter", 1)))
                add_is_hitter = bool(int(add_match.iloc[0].get("is_hitter", 1)))
                # If they differ, it is cross-type — should only be allowed with surplus
                if drop_is_hitter != add_is_hitter:
                    # This should not happen without surplus
                    pytest.fail("Cross-type swap should be blocked without surplus")


class TestOwnershipBoost:
    """Ownership trend boost is applied correctly."""

    def test_ownership_boost_applied(self):
        roster_guy = _make_player(1, "Roster", hr=5, r=20, rbi=20, sb=1, avg=0.230, obp=0.280)
        fa_trending = _make_player(100, "Trending FA", marginal_value=3.0, hr=15, r=50, rbi=50, sb=5)
        fa_stable = _make_player(101, "Stable FA", marginal_value=3.0, hr=15, r=50, rbi=50, sb=5)
        all_players = [roster_guy, fa_trending, fa_stable]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa_trending, fa_stable],
            ownership_trends={
                100: {"pct_owned": 45.0, "delta_7d": 8.0},  # Trending up
                101: {"pct_owned": 45.0, "delta_7d": 1.0},  # Stable
            },
        )
        fas = _score_fa_candidates(ctx)
        trending = next(f for f in fas if f["player_id"] == 100)
        stable = next(f for f in fas if f["player_id"] == 101)
        assert trending["composite_score"] > stable["composite_score"]


class TestSustainability:
    """Sustainability scoring is applied."""

    def test_sustainability_in_output(self):
        roster = _make_player(1, "Drop Me", hr=2, r=10, rbi=10, sb=0, avg=0.200, obp=0.260)
        fa = _make_player(100, "FA Good", marginal_value=5.0, hr=20, r=60, rbi=60, sb=8, ab=400, h=110)
        all_players = [roster, fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa],
        )
        result = recommend_fa_moves(ctx)
        for move in result:
            assert "sustainability" in move
            assert 0.0 <= move["sustainability"] <= 1.0


class TestHealthFilter:
    """IL/injured FAs are flagged with is_il=True (included as candidates
    so they can be matched with IL drops for stash upgrades), healthy FAs
    are flagged is_il=False. Cross-health swaps are blocked downstream."""

    def test_low_health_flagged_as_il(self):
        roster = _make_player(1, "Roster", hr=5, r=20, rbi=20, sb=1, avg=0.230, obp=0.280)
        fa_healthy = _make_player(100, "Healthy FA", marginal_value=5.0, hr=20, r=60, rbi=60, sb=8)
        fa_hurt = _make_player(101, "Hurt FA", marginal_value=5.0, hr=20, r=60, rbi=60, sb=8)
        all_players = [roster, fa_healthy, fa_hurt]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa_healthy, fa_hurt],
            health_scores={100: 0.90, 101: 0.50},  # 101 below 0.65
        )
        fas = _score_fa_candidates(ctx)
        fas_by_id = {f["player_id"]: f for f in fas}
        assert fas_by_id[100]["is_il"] is False
        assert fas_by_id[101]["is_il"] is True

    def test_il_status_flagged_as_il(self):
        roster = _make_player(1, "Roster")
        fa_il = _make_player(100, "IL Guy", marginal_value=5.0, status="IL")
        all_players = [roster, fa_il]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa_il],
        )
        fas = _score_fa_candidates(ctx)
        assert len(fas) == 1
        assert fas[0]["is_il"] is True


class TestHealthMatchedSwaps:
    """Drops and adds must come from the same roster pool (both active
    or both IL). Cross-pool swaps are rejected — they'd either overfill
    the active roster or leave an unusable IL slot empty."""

    def test_il_drop_with_healthy_add_rejected(self):
        # IL drop must NOT be paired with a healthy add (the bug that
        # showed "Add Bryson Stott / Drop Alejandro Kirk" where Kirk is IL).
        il_drop = _make_player(1, "IL Catcher", hr=0, r=0, rbi=0, sb=0, avg=0, obp=0, status="IL", positions="C,Util")
        healthy_drop = _make_player(2, "Active OF", hr=5, r=15, rbi=12, sb=1, avg=0.220, obp=0.280, positions="OF")
        healthy_fa = _make_player(100, "Healthy FA", marginal_value=8.0, hr=25, r=70, rbi=70, sb=10, positions="OF")
        all_players = [il_drop, healthy_drop, healthy_fa]

        ctx = _build_ctx(
            roster_ids=[1, 2],
            pool_players=all_players,
            fa_players=[healthy_fa],
        )
        result = recommend_fa_moves(ctx, max_moves=5)
        # No move should drop the IL catcher for the healthy FA
        for move in result:
            assert not (move["drop_id"] == 1 and move["add_id"] == 100), (
                f"Invalid cross-health swap suggested: drop IL {move['drop_name']} / add healthy {move['add_name']}"
            )

    def test_il_drop_with_il_add_allowed(self):
        # IL-for-IL swap IS valid — upgrading stash within the IL pool.
        il_drop = _make_player(1, "Weak IL", hr=0, r=0, rbi=0, sb=0, avg=0, obp=0, status="IL", positions="OF")
        il_fa = _make_player(
            100,
            "Better IL",
            marginal_value=8.0,
            hr=30,
            r=90,
            rbi=100,
            sb=5,
            status="IL",
            positions="OF",
        )
        all_players = [il_drop, il_fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[il_fa],
            health_scores={1: 0.3, 100: 0.3},
        )
        result = recommend_fa_moves(ctx, max_moves=5)
        # At least one valid IL-for-IL move should be allowed (no assertion
        # that it's returned — depends on net SGP — but also must not be
        # BLOCKED by the health matcher).
        from src.optimizer.fa_recommender import _evaluate_swaps, _score_drop_candidates, _score_fa_candidates

        drops = _score_drop_candidates(ctx)
        fas = _score_fa_candidates(ctx)
        swaps = _evaluate_swaps(ctx, drops, fas)
        # Both drop and FA are IL — matching should let them pair
        drop_il = next((d for d in drops if d["player_id"] == 1), None)
        fa_il = next((f for f in fas if f["player_id"] == 100), None)
        assert drop_il is not None and drop_il["is_il"] is True
        assert fa_il is not None and fa_il["is_il"] is True
        # If the swap has positive net SGP it should appear; regardless it
        # must not be filtered out by the health mismatch check.
        for s in swaps:
            # If this is our IL pair it's fine; other pairs just shouldn't
            # cross health.
            if s["drop_id"] == 1:
                assert s["add_id"] == 100


class TestMaxMoves:
    """max_moves is respected."""

    def test_max_moves_limits_output(self):
        roster = [_make_player(i, f"R{i}", hr=1, r=5, rbi=5, sb=0, avg=0.190, obp=0.240) for i in range(1, 6)]
        fas = [_make_player(100 + i, f"FA{i}", marginal_value=8.0 + i, hr=25, r=70, rbi=70, sb=10) for i in range(5)]
        all_players = roster + fas

        ctx = _build_ctx(
            roster_ids=[p["player_id"] for p in roster],
            pool_players=all_players,
            fa_players=fas,
        )
        result = recommend_fa_moves(ctx, max_moves=2)
        assert len(result) <= 2


class TestHappyPath:
    """Happy path returns correctly structured results."""

    def test_happy_path_fields(self):
        roster = _make_player(1, "Bad Guy", hr=2, r=10, rbi=10, sb=0, avg=0.200, obp=0.260, positions="OF")
        fa = _make_player(100, "Good FA", marginal_value=8.0, hr=25, r=70, rbi=70, sb=10, positions="1B")
        all_players = [roster, fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa],
        )
        result = recommend_fa_moves(ctx)
        if result:
            move = result[0]
            assert "add_id" in move
            assert "add_name" in move
            assert "add_positions" in move
            assert "drop_id" in move
            assert "drop_name" in move
            assert "drop_positions" in move
            assert "net_sgp_delta" in move
            assert "category_impact" in move
            assert "reasoning" in move
            assert "urgency_categories" in move
            assert "news_warning" in move
            assert "ownership_trend" in move
            assert "sustainability" in move
            assert isinstance(move["reasoning"], list)
            assert isinstance(move["category_impact"], dict)
            assert isinstance(move["urgency_categories"], list)
            assert move["net_sgp_delta"] > 0


class TestNewsWarning:
    """News warning is populated when available."""

    def test_news_warning_populated(self):
        roster = _make_player(1, "Drop", hr=1, r=5, rbi=5, sb=0, avg=0.180, obp=0.230)
        fa = _make_player(100, "FA News", marginal_value=8.0, hr=25, r=70, rbi=70, sb=10)
        all_players = [roster, fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa],
            news_flags={100: "FA News placed on paternity list"},
        )
        result = recommend_fa_moves(ctx)
        if result:
            move = result[0]
            assert move["news_warning"] == "FA News placed on paternity list"

    def test_no_news_is_none(self):
        roster = _make_player(1, "Drop", hr=1, r=5, rbi=5, sb=0, avg=0.180, obp=0.230)
        fa = _make_player(100, "FA No News", marginal_value=8.0, hr=25, r=70, rbi=70, sb=10)
        all_players = [roster, fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa],
        )
        result = recommend_fa_moves(ctx)
        if result:
            assert result[0]["news_warning"] is None


class TestUrgencyCategories:
    """Urgency categories are populated from matchup state."""

    def test_urgency_categories_populated(self):
        roster = _make_player(1, "Drop", hr=1, r=5, rbi=5, sb=0, avg=0.180, obp=0.230)
        fa = _make_player(100, "HR FA", marginal_value=8.0, hr=35, r=80, rbi=80, sb=5)
        all_players = [roster, fa]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa],
            urgency_weights={
                "urgency": {"HR": 0.9, "R": 0.8, "RBI": 0.5, "SB": 0.3, "AVG": 0.4, "OBP": 0.4},
                "summary": {"losing": ["HR", "R"], "tied": ["RBI"], "winning": ["SB", "AVG", "OBP"]},
            },
        )
        result = recommend_fa_moves(ctx)
        if result:
            assert isinstance(result[0]["urgency_categories"], list)


class TestFloorPreference:
    """Floor preference penalty is applied for low PA/IP players."""

    def test_low_pa_hitter_penalized(self):
        roster = _make_player(1, "Roster")
        fa_solid = _make_player(100, "Solid FA", marginal_value=3.0, pa=500)
        fa_small = _make_player(101, "Small Sample FA", marginal_value=3.0, pa=30)
        all_players = [roster, fa_solid, fa_small]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa_solid, fa_small],
        )
        fas = _score_fa_candidates(ctx)
        solid = next((f for f in fas if f["player_id"] == 100), None)
        small = next((f for f in fas if f["player_id"] == 101), None)
        if solid and small:
            assert solid["composite_score"] > small["composite_score"]

    def test_low_ip_pitcher_penalized(self):
        roster = _make_player(1, "Roster")
        fa_solid = _make_pitcher(100, "Solid SP", marginal_value=3.0, ip=150)
        fa_small = _make_pitcher(101, "Small SP", marginal_value=3.0, ip=10)
        all_players = [roster, fa_solid, fa_small]

        ctx = _build_ctx(
            roster_ids=[1],
            pool_players=all_players,
            fa_players=[fa_solid, fa_small],
        )
        fas = _score_fa_candidates(ctx)
        solid = next((f for f in fas if f["player_id"] == 100), None)
        small = next((f for f in fas if f["player_id"] == 101), None)
        if solid and small:
            assert solid["composite_score"] > small["composite_score"]


class TestIsCloser:
    """_is_closer helper works correctly."""

    def test_is_closer_flag(self):
        closer = _make_pitcher(1, "Closer", sv=20, is_closer=True)
        pool = _build_pool([closer])
        ctx = _build_ctx(roster_ids=[1], pool_players=[closer])
        assert _is_closer(1, ctx) is True

    def test_is_closer_by_sv(self):
        pitcher = _make_pitcher(1, "RP", sv=8, is_closer=False)
        ctx = _build_ctx(roster_ids=[1], pool_players=[pitcher])
        assert _is_closer(1, ctx) is True

    def test_not_closer(self):
        pitcher = _make_pitcher(1, "SP", sv=0, is_closer=False)
        ctx = _build_ctx(roster_ids=[1], pool_players=[pitcher])
        assert _is_closer(1, ctx) is False


class TestDeduplication:
    """Each FA and each drop candidate used at most once."""

    def test_no_duplicate_adds_or_drops(self):
        roster = [
            _make_player(i, f"R{i}", hr=1 + i, r=10 + i, rbi=10 + i, sb=0, avg=0.200, obp=0.260) for i in range(1, 6)
        ]
        fas = [
            _make_player(100 + i, f"FA{i}", marginal_value=7.0 + i, hr=20 + i, r=60 + i, rbi=60 + i, sb=8)
            for i in range(5)
        ]
        all_players = roster + fas

        ctx = _build_ctx(
            roster_ids=[p["player_id"] for p in roster],
            pool_players=all_players,
            fa_players=fas,
        )
        result = recommend_fa_moves(ctx, max_moves=5)
        add_ids = [m["add_id"] for m in result]
        drop_ids = [m["drop_id"] for m in result]
        assert len(add_ids) == len(set(add_ids)), "Duplicate add IDs found"
        assert len(drop_ids) == len(set(drop_ids)), "Duplicate drop IDs found"


# ---------------------------------------------------------------------------
# Streaming recommendations (scope="today")
# ---------------------------------------------------------------------------


def _stream_ctx(
    *,
    roster_ids: list[int],
    pool: list[dict],
    fas: list[dict],
    my_totals: dict[str, float] | None = None,
    opp_totals: dict[str, float] | None = None,
    todays_schedule: list[dict] | None = None,
    remaining_games: dict[str, int] | None = None,
    scope: str = "today",
) -> OptimizerDataContext:
    """Build a context configured for streaming tests (scope=today)."""
    ctx = _build_ctx(roster_ids=roster_ids, pool_players=pool, fa_players=fas)
    ctx.scope = scope
    ctx.my_totals = my_totals or {}
    ctx.opp_totals = opp_totals or {}
    ctx.todays_schedule = todays_schedule or []
    ctx.remaining_games_this_week = remaining_games or {}
    return ctx


class TestStreamingScope:
    """Streaming only fires on scope=today."""

    def test_non_today_scope_returns_empty(self):
        ctx = _stream_ctx(roster_ids=[1], pool=[_make_player(1)], fas=[], scope="rest_of_week")
        result = recommend_streaming_moves(ctx)
        assert result["pitchers"] == []
        assert result["batters"] == []

    def test_no_target_cats_returns_empty(self):
        # No close/winnable categories → nothing to stream
        ctx = _stream_ctx(
            roster_ids=[1],
            pool=[_make_player(1)],
            fas=[],
            my_totals={
                "hr": 100,
                "r": 100,
                "rbi": 100,
                "sb": 100,
                "avg": 0.300,
                "obp": 0.400,
                "w": 10,
                "l": 0,
                "sv": 10,
                "k": 100,
                "era": 2.00,
                "whip": 1.00,
            },
            opp_totals={
                "hr": 1,
                "r": 1,
                "rbi": 1,
                "sb": 1,
                "avg": 0.100,
                "obp": 0.100,
                "w": 0,
                "l": 100,
                "sv": 0,
                "k": 1,
                "era": 10.0,
                "whip": 2.0,
            },
        )
        # User dominates every cat → all win probs ~1.0 (which is >=0.38 so
        # technically in target), but there are no FAs to stream. Empty lists
        # for pitchers/batters.
        result = recommend_streaming_moves(ctx)
        assert result["pitchers"] == []
        assert result["batters"] == []


class TestStreamingCandidateFiltering:
    """FA filtering: probable-starter (pitchers), game-today (batters)."""

    def test_pitcher_must_be_probable_starter_today(self):
        # A rostered SP with bad projection as "worst rostered pitcher" drop,
        # and two FA pitchers: one probable today, one NOT on the schedule.
        my_sp_drop = _make_pitcher(1, "Bad SP Roster", w=2, l=8, k=60, era=5.00, whip=1.50, ip=60)
        my_bat = _make_player(2, "Solid Bat", positions="OF", pa=500, hr=25, r=75, rbi=70, sb=5)
        fa_probable = _make_pitcher(100, "Elite Probable", w=15, l=5, k=220, era=2.90, whip=1.00, ip=190)
        fa_not_probable = _make_pitcher(101, "Elite Not Probable", w=15, l=5, k=220, era=2.90, whip=1.00, ip=190)

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[my_sp_drop, my_bat, fa_probable, fa_not_probable],
            fas=[fa_probable, fa_not_probable],
            my_totals={
                "hr": 10,
                "r": 30,
                "rbi": 25,
                "sb": 2,
                "avg": 0.250,
                "obp": 0.310,
                "w": 1,
                "l": 3,
                "sv": 1,
                "k": 25,
                "era": 5.00,
                "whip": 1.45,
            },
            opp_totals={
                "hr": 9,
                "r": 28,
                "rbi": 24,
                "sb": 3,
                "avg": 0.255,
                "obp": 0.315,
                "w": 2,
                "l": 2,
                "sv": 2,
                "k": 30,
                "era": 4.20,
                "whip": 1.30,
            },
            todays_schedule=[{"home_probable_pitcher": {"id": 100}, "away_probable_pitcher": None}],
        )
        result = recommend_streaming_moves(ctx)
        add_ids = {s["add_id"] for s in result["pitchers"]}
        assert 101 not in add_ids  # not probable → excluded
        # 100 may or may not surface depending on thresholds; the key
        # assertion is the non-probable was filtered out.

    def test_batter_must_have_game_today(self):
        my_bat_drop = _make_player(1, "Weak Bat", positions="OF", hr=3, r=15, rbi=10, sb=0, avg=0.210, obp=0.270)
        my_sp = _make_pitcher(2, "SP", w=10, l=5, k=150, era=3.5, whip=1.20, ip=180)
        fa_playing = _make_player(100, "FA Playing", positions="OF", hr=25, r=70, rbi=65, sb=15, avg=0.290, obp=0.360)
        fa_playing["team"] = "COL"
        fa_off_day = _make_player(101, "FA Off Day", positions="OF", hr=25, r=70, rbi=65, sb=15, avg=0.290, obp=0.360)
        fa_off_day["team"] = "DET"

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[my_bat_drop, my_sp, fa_playing, fa_off_day],
            fas=[fa_playing, fa_off_day],
            my_totals={
                "hr": 5,
                "r": 20,
                "rbi": 15,
                "sb": 1,
                "avg": 0.230,
                "obp": 0.290,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 40,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 5,
                "r": 22,
                "rbi": 17,
                "sb": 4,
                "avg": 0.240,
                "obp": 0.300,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 38,
                "era": 3.4,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
        )
        result = recommend_streaming_moves(ctx)
        add_ids = {s["add_id"] for s in result["batters"]}
        assert 101 not in add_ids  # DET has no game today


class TestStreamingThresholds:
    """Net SGP and hurts threshold filters."""

    def test_low_net_sgp_excluded(self):
        # FA marginally better than drop → net SGP < 0.70 → excluded.
        my_bat = _make_player(1, "Rostered", hr=15, r=50, rbi=45, sb=5, avg=0.260, obp=0.325)
        my_sp = _make_pitcher(2, "SP", w=10, l=5, k=150, era=3.5, whip=1.20, ip=180)
        fa_similar = _make_player(100, "Similar FA", hr=16, r=52, rbi=46, sb=5, avg=0.262, obp=0.326)
        fa_similar["team"] = "HOU"

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[my_bat, my_sp, fa_similar],
            fas=[fa_similar],
            my_totals={
                "hr": 15,
                "r": 50,
                "rbi": 45,
                "sb": 5,
                "avg": 0.260,
                "obp": 0.325,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 50,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 14,
                "r": 48,
                "rbi": 44,
                "sb": 6,
                "avg": 0.262,
                "obp": 0.330,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 48,
                "era": 3.6,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "HOU", "away_team": "LAA"}],
        )
        result = recommend_streaming_moves(ctx)
        # Similar player → low net SGP → should not appear
        for s in result["batters"]:
            assert s["add_id"] != 100 or s["net_sgp"] >= 0.70

    def test_max_per_side_cap(self):
        # Build many potentially-valid FA batters, verify cap of 3.
        my_bad = _make_player(1, "Bad Rostered", hr=1, r=5, rbi=3, sb=0, avg=0.180, obp=0.240, pa=50, ab=45, h=9)
        my_sp = _make_pitcher(2, "SP", w=10, l=5, k=150, era=3.5, whip=1.20, ip=180)
        fas = []
        for i in range(8):
            fa = _make_player(100 + i, f"Elite FA {i}", hr=35, r=100, rbi=95, sb=20, avg=0.310, obp=0.380)
            fa["team"] = "COL"
            fas.append(fa)
        pool = [my_bad, my_sp] + fas

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=pool,
            fas=fas,
            my_totals={
                "hr": 10,
                "r": 30,
                "rbi": 25,
                "sb": 2,
                "avg": 0.240,
                "obp": 0.300,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 40,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 9,
                "r": 28,
                "rbi": 24,
                "sb": 3,
                "avg": 0.245,
                "obp": 0.305,
                "w": 2,
                "l": 2,
                "sv": 2,
                "k": 42,
                "era": 3.6,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
        )
        result = recommend_streaming_moves(ctx, max_per_side=3)
        assert len(result["batters"]) <= 3


class TestStreamingSort:
    """Streaming results are sorted best-to-worst by net SGP."""

    def test_sort_descending_by_net_sgp(self):
        my_bad = _make_player(1, "Bad Rostered", hr=1, r=5, rbi=3, sb=0, avg=0.180, obp=0.240, pa=50, ab=45, h=9)
        my_sp = _make_pitcher(2, "SP", w=10, l=5, k=150, era=3.5, whip=1.20, ip=180)

        # Three FAs of decreasing strength.
        fas = []
        for i, hr_val in enumerate([35, 30, 25]):
            fa = _make_player(
                100 + i,
                f"FA{i}",
                hr=hr_val,
                r=100 - i * 10,
                rbi=95 - i * 10,
                sb=20 - i * 2,
                avg=0.310 - i * 0.01,
                obp=0.380 - i * 0.01,
            )
            fa["team"] = "COL"
            fas.append(fa)

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[my_bad, my_sp] + fas,
            fas=fas,
            my_totals={
                "hr": 10,
                "r": 30,
                "rbi": 25,
                "sb": 2,
                "avg": 0.240,
                "obp": 0.300,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 40,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 9,
                "r": 28,
                "rbi": 24,
                "sb": 3,
                "avg": 0.245,
                "obp": 0.305,
                "w": 2,
                "l": 2,
                "sv": 2,
                "k": 42,
                "era": 3.6,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
        )
        result = recommend_streaming_moves(ctx)
        nets = [s["net_sgp"] for s in result["batters"]]
        assert nets == sorted(nets, reverse=True)


class TestStreamingILExclusion:
    """IL FAs are ineligible for streaming (streaming is for healthy adds only)."""

    def test_il_fa_not_streamed(self):
        my_bad = _make_player(1, "Rostered")
        my_sp = _make_pitcher(2, "SP")
        fa_il = _make_player(100, "IL FA", hr=35, r=100, rbi=95, sb=20, avg=0.310, obp=0.380, status="IL15")
        fa_il["team"] = "COL"

        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[my_bad, my_sp, fa_il],
            fas=[fa_il],
            my_totals={
                "hr": 10,
                "r": 30,
                "rbi": 25,
                "sb": 2,
                "avg": 0.240,
                "obp": 0.300,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 40,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 9,
                "r": 28,
                "rbi": 24,
                "sb": 3,
                "avg": 0.245,
                "obp": 0.305,
                "w": 2,
                "l": 2,
                "sv": 2,
                "k": 42,
                "era": 3.6,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
        )
        result = recommend_streaming_moves(ctx)
        assert all(s["add_id"] != 100 for s in result["batters"])
        assert all(s["add_id"] != 100 for s in result["pitchers"])


class TestStreamingHurtsGuard:
    """Hurts guard protects all in-play cats AND any losing/tied cats,
    even if the losing cat's win probability is below the 27.55% threshold."""

    def test_hurts_block_protects_losing_cat_below_winprob(self):
        """A swap that hurts a currently-losing cat by more than 0.10 SGP
        must be blocked even if that cat's win probability is <27.55%."""
        # Roster: low-value pitcher + low-value batter + an anchor pitcher so
        # dropping the worst still leaves IP/rules feasible.
        worst_bat = _make_player(1, "Weak Bat", positions="OF", hr=1, r=5, rbi=3, sb=0, avg=0.180, obp=0.240, pa=50)
        anchor_sp = _make_pitcher(2, "Anchor SP", w=15, l=5, k=200, era=3.0, whip=1.10, ip=200)
        # FA that helps a lot but *specifically* hurts a losing cat.
        # Easiest way: a hitter with negative effective contribution to
        # ERA via the swap math. Use a much weaker bat as the drop so the
        # swap also zeros out any batter contribution. This isn't a
        # "realistic" move — it's a synthetic scenario to exercise the guard.
        fa_aggressive = _make_player(100, "Aggressive FA", hr=40, r=120, rbi=120, sb=25, avg=0.300, obp=0.370)
        fa_aggressive["team"] = "COL"

        # Engineer matchup: losing K by 30 (win_prob well below threshold)
        # so K is NOT in-play but IS losing.
        ctx = _stream_ctx(
            roster_ids=[1, 2],
            pool=[worst_bat, anchor_sp, fa_aggressive],
            fas=[fa_aggressive],
            my_totals={
                "hr": 5,
                "r": 20,
                "rbi": 15,
                "sb": 1,
                "avg": 0.220,
                "obp": 0.280,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 20,
                "era": 3.50,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 5,
                "r": 20,
                "rbi": 15,
                "sb": 1,
                "avg": 0.220,
                "obp": 0.280,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 50,
                "era": 3.50,
                "whip": 1.20,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
            # Mark K as a losing category via urgency_weights.summary.
            # Build a minimal ctx patch: we can't pass this directly via
            # _stream_ctx, so we set it after build.
        )
        # Force urgency_weights.summary.losing to include K (reflects that
        # user is currently losing K per the matchup above).
        ctx.urgency_weights = {"summary": {"losing": ["k"], "tied": [], "winning": []}}
        # Run — the key assertion is just that the function runs and
        # returns a sane structure. If this FA would hurt K by > 0.10
        # SGP, it must be filtered. For a pure hitter-for-hitter swap
        # K delta is ~0 so nothing is hurt; the guard isn't exercised.
        # This test exists as a scaffold; deeper hurts-guard behavior
        # is covered by dropping a pitcher for a batter (which does hurt
        # K). We only assert the structure here.
        result = recommend_streaming_moves(ctx)
        assert "diagnostics" in result
        assert "protected_cats" in result["diagnostics"]
        # K must appear in protected cats because it's in losing_cats,
        # regardless of whether its win prob crosses the 0.2755 threshold.
        assert "k" in set(result["diagnostics"]["protected_cats"])


class TestStreamingCrossSideOneWay:
    """Cross-side drops are allowed ONLY for pitcher streaming (drop a
    much-worse batter instead of the worst pitcher). Batter streaming
    must never drop a pitcher — that silently shrinks the pitching pool."""

    def test_batter_stream_never_drops_pitcher(self):
        # A very low-value pitcher (Montero-like) plus a moderately-
        # low-value batter. A batter streamer must drop the batter,
        # NOT the pitcher, even though the pitcher scores lower.
        worst_pitcher = _make_pitcher(1, "Bench SP", w=2, l=1, k=20, era=5.5, whip=1.50, ip=15)
        worst_batter = _make_player(
            2, "Weak Bat", positions="OF", hr=3, r=15, rbi=10, sb=1, avg=0.210, obp=0.270, pa=200, ab=180, h=38
        )
        other_sp = _make_pitcher(3, "Good SP", w=15, l=5, k=200, era=3.0, whip=1.10, ip=190)
        fa_bat = _make_player(100, "Streaming Bat", positions="OF", hr=30, r=100, rbi=90, sb=20, avg=0.310, obp=0.380)
        fa_bat["team"] = "COL"

        ctx = _stream_ctx(
            roster_ids=[1, 2, 3],
            pool=[worst_pitcher, worst_batter, other_sp, fa_bat],
            fas=[fa_bat],
            my_totals={
                "hr": 10,
                "r": 30,
                "rbi": 25,
                "sb": 2,
                "avg": 0.240,
                "obp": 0.300,
                "w": 3,
                "l": 1,
                "sv": 2,
                "k": 40,
                "era": 3.5,
                "whip": 1.20,
            },
            opp_totals={
                "hr": 9,
                "r": 28,
                "rbi": 24,
                "sb": 3,
                "avg": 0.245,
                "obp": 0.305,
                "w": 2,
                "l": 2,
                "sv": 2,
                "k": 42,
                "era": 3.6,
                "whip": 1.22,
            },
            todays_schedule=[{"home_team": "COL", "away_team": "SFG"}],
        )
        result = recommend_streaming_moves(ctx)
        # For every batter stream, the drop must also be a batter
        pool = ctx.player_pool
        for s in result["batters"]:
            drop_match = pool[pool["player_id"] == s["drop_id"]]
            assert not drop_match.empty
            assert int(drop_match.iloc[0]["is_hitter"]) == 1, (
                f"Batter stream dropped a pitcher: add={s['add_name']}, drop={s['drop_name']}"
            )
