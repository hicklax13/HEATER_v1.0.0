"""Tests for src/optimizer/shared_data_layer.py.

The shared data layer is the single source of truth for all optimizer tabs.
It builds an immutable OptimizerDataContext that every module consumes.
"""

import pandas as pd
import pytest

from src.optimizer.shared_data_layer import (
    OptimizerDataContext,
    get_recent_form_weight,
    scale_projections_for_scope,
)
from src.valuation import LeagueConfig


class TestOptimizerDataContext:
    """Verify OptimizerDataContext construction and field contracts."""

    @pytest.fixture()
    def _minimal_context(self):
        """Build a minimal valid context with required fields."""
        roster = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "Test Hitter",
                    "positions": "1B",
                    "is_hitter": 1,
                    "team": "NYY",
                    "r": 80,
                    "hr": 25,
                    "rbi": 75,
                    "sb": 10,
                    "avg": 0.280,
                    "obp": 0.350,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "ip": 0,
                    "h": 140,
                    "ab": 500,
                    "bb": 50,
                    "hbp": 5,
                    "sf": 4,
                    "status": "active",
                },
                {
                    "player_id": 2,
                    "player_name": "Test Pitcher",
                    "positions": "SP",
                    "is_hitter": 0,
                    "team": "BOS",
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "w": 12,
                    "l": 6,
                    "sv": 0,
                    "k": 180,
                    "era": 3.50,
                    "whip": 1.15,
                    "ip": 180,
                    "h": 0,
                    "ab": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 70,
                    "bb_allowed": 45,
                    "h_allowed": 160,
                    "status": "active",
                },
            ]
        )
        return OptimizerDataContext(
            roster=roster,
            player_pool=roster.copy(),
            free_agents=pd.DataFrame(),
            user_roster_ids=[1, 2],
            live_matchup=None,
            my_totals={
                "R": 400,
                "HR": 120,
                "RBI": 380,
                "SB": 60,
                "AVG": 0.265,
                "OBP": 0.330,
                "W": 45,
                "L": 30,
                "SV": 40,
                "K": 800,
                "ERA": 3.80,
                "WHIP": 1.22,
            },
            opp_totals={
                "R": 420,
                "HR": 130,
                "RBI": 400,
                "SB": 55,
                "AVG": 0.270,
                "OBP": 0.340,
                "W": 50,
                "L": 28,
                "SV": 45,
                "K": 850,
                "ERA": 3.60,
                "WHIP": 1.18,
            },
            opponent_name="Test Opponent",
            win_loss_tie=(5, 6, 1),
            urgency_weights={},
            category_weights={
                cat: 1.0
                for cat in [
                    "R",
                    "HR",
                    "RBI",
                    "SB",
                    "AVG",
                    "OBP",
                    "W",
                    "L",
                    "SV",
                    "K",
                    "ERA",
                    "WHIP",
                ]
            },
            category_gaps={},
            h2h_strategy={},
            todays_schedule=[],
            confirmed_lineups={},
            remaining_games_this_week={},
            two_start_pitchers=[],
            opposing_pitchers={},
            team_strength={},
            park_factors={},
            weather={},
            recent_form={},
            health_scores={1: 0.95, 2: 0.90},
            news_flags={},
            ownership_trends={},
            scope="rest_of_week",
            weeks_remaining=16,
            config=LeagueConfig(),
            adds_remaining_this_week=7,
            closer_count=2,
            il_stash_ids=set(),
        )

    def test_context_has_all_required_fields(self, _minimal_context):
        """Context must expose all fields consumed by optimizer modules."""
        ctx = _minimal_context
        assert isinstance(ctx.roster, pd.DataFrame)
        assert isinstance(ctx.player_pool, pd.DataFrame)
        assert isinstance(ctx.free_agents, pd.DataFrame)
        assert isinstance(ctx.user_roster_ids, list)
        assert isinstance(ctx.my_totals, dict)
        assert isinstance(ctx.opp_totals, dict)
        assert isinstance(ctx.category_weights, dict)
        assert isinstance(ctx.health_scores, dict)
        assert isinstance(ctx.scope, str)
        assert isinstance(ctx.weeks_remaining, int)
        assert isinstance(ctx.config, LeagueConfig)

    def test_context_totals_have_all_12_categories(self, _minimal_context):
        """my_totals and opp_totals must have all 12 scoring categories."""
        lc = LeagueConfig()
        all_cats = lc.hitting_categories + lc.pitching_categories
        for cat in all_cats:
            assert cat in _minimal_context.my_totals, f"Missing {cat} in my_totals"
            assert cat in _minimal_context.opp_totals, f"Missing {cat} in opp_totals"

    def test_category_weights_has_all_12(self, _minimal_context):
        """Category weights dict must cover all 12 categories."""
        lc = LeagueConfig()
        all_cats = lc.hitting_categories + lc.pitching_categories
        for cat in all_cats:
            assert cat in _minimal_context.category_weights, f"Missing weight for {cat}"

    def test_win_loss_tie_is_three_tuple(self, _minimal_context):
        """W-L-T must be a 3-element tuple of ints."""
        wlt = _minimal_context.win_loss_tie
        assert len(wlt) == 3
        assert all(isinstance(v, int) for v in wlt)


class TestScaleProjectionsForScope:
    """Verify projection scaling correctly handles scope differences.

    Today: counting stats / 162 (single game)
    Rest of Week: scale by remaining games + two-start pitcher doubling
    Rest of Season: full projections with schedule strength
    Rate stats: NEVER scaled (AVG, OBP, ERA, WHIP stay the same)
    """

    @pytest.fixture()
    def _roster(self):
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "player_name": "Hitter",
                    "positions": "1B",
                    "is_hitter": 1,
                    "team": "NYY",
                    "r": 80,
                    "hr": 25,
                    "rbi": 75,
                    "sb": 10,
                    "avg": 0.280,
                    "obp": 0.350,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "ip": 0,
                },
            ]
        )

    @pytest.fixture()
    def _today_context(self, _roster):
        return OptimizerDataContext(
            roster=_roster,
            player_pool=_roster.copy(),
            free_agents=pd.DataFrame(),
            user_roster_ids=[1],
            live_matchup=None,
            my_totals={},
            opp_totals={},
            opponent_name="",
            win_loss_tie=(0, 0, 0),
            urgency_weights={},
            category_weights={},
            category_gaps={},
            h2h_strategy={},
            todays_schedule=[],
            confirmed_lineups={},
            remaining_games_this_week={"NYY": 1},
            two_start_pitchers=[],
            opposing_pitchers={},
            team_strength={},
            park_factors={},
            weather={},
            recent_form={},
            health_scores={},
            news_flags={},
            ownership_trends={},
            scope="today",
            weeks_remaining=16,
            config=LeagueConfig(),
            adds_remaining_this_week=7,
            closer_count=2,
            il_stash_ids=set(),
        )

    def test_today_scope_scales_counting_stats_down(self, _today_context, _roster):
        """Today scope: counting stats should be ~1/162 of season projection."""
        scaled = scale_projections_for_scope(_today_context, _roster)
        # HR should be roughly 25/162 ~ 0.154 (single game)
        assert scaled.iloc[0]["hr"] < 1.0
        assert scaled.iloc[0]["hr"] == pytest.approx(25 / 162, abs=0.05)

    def test_rate_stats_unchanged_by_scope(self, _today_context, _roster):
        """Rate stats (AVG, OBP) must never change when scaling scope."""
        scaled = scale_projections_for_scope(_today_context, _roster)
        assert scaled.iloc[0]["avg"] == pytest.approx(0.280, abs=0.001)
        assert scaled.iloc[0]["obp"] == pytest.approx(0.350, abs=0.001)

    def test_returns_copy_not_mutation(self, _today_context, _roster):
        """scale_projections_for_scope must return a COPY, not mutate input."""
        original_hr = _roster.iloc[0]["hr"]
        _ = scale_projections_for_scope(_today_context, _roster)
        assert _roster.iloc[0]["hr"] == original_hr


class TestDataFreshnessInContext:
    """Verify OptimizerDataContext tracks data freshness."""

    def test_context_has_data_timestamps_field(self):
        """OptimizerDataContext must have a data_timestamps field."""
        assert hasattr(OptimizerDataContext, "__dataclass_fields__")
        assert "data_timestamps" in OptimizerDataContext.__dataclass_fields__


class TestRecentFormWeight:
    """Verify scope-specific recent form weights."""

    def test_today_weight(self):
        assert get_recent_form_weight("today") == pytest.approx(0.25, abs=0.01)

    def test_week_weight(self):
        assert get_recent_form_weight("rest_of_week") == pytest.approx(0.30, abs=0.01)

    def test_season_weight(self):
        assert get_recent_form_weight("rest_of_season") == pytest.approx(0.20, abs=0.01)
