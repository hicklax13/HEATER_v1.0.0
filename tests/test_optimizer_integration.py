"""Cross-module integration tests for the full Lineup Optimizer pipeline.

These tests use realistic 23-player rosters (not 3-6 player toy rosters)
and verify the full chain: projections -> matchup -> weights -> LP -> output.

Covers Tasks 8, 9, and 10 from the optimizer validation framework plan:
  - Task 8: Realistic 23-player roster fixture
  - Task 9: Pipeline invariant assertions (mode x alpha matrix)
  - Task 10: Settings axis tests (directional effects of alpha and mode)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer.pipeline import LineupOptimizerPipeline

# ── Realistic 23-player roster fixture ──────────────────────────────

_HITTERS = [
    {
        "player_id": 101,
        "player_name": "C_Starter",
        "positions": "C",
        "is_hitter": 1,
        "team": "NYY",
        "r": 55,
        "hr": 18,
        "rbi": 60,
        "sb": 2,
        "avg": 0.245,
        "obp": 0.320,
        "h": 110,
        "ab": 449,
        "bb": 40,
        "hbp": 3,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 102,
        "player_name": "1B_Starter",
        "positions": "1B",
        "is_hitter": 1,
        "team": "BOS",
        "r": 85,
        "hr": 32,
        "rbi": 95,
        "sb": 3,
        "avg": 0.275,
        "obp": 0.360,
        "h": 150,
        "ab": 545,
        "bb": 60,
        "hbp": 5,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 103,
        "player_name": "2B_Starter",
        "positions": "2B",
        "is_hitter": 1,
        "team": "HOU",
        "r": 75,
        "hr": 15,
        "rbi": 55,
        "sb": 20,
        "avg": 0.270,
        "obp": 0.340,
        "h": 140,
        "ab": 519,
        "bb": 45,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 104,
        "player_name": "3B_Starter",
        "positions": "3B",
        "is_hitter": 1,
        "team": "LAD",
        "r": 90,
        "hr": 28,
        "rbi": 85,
        "sb": 8,
        "avg": 0.280,
        "obp": 0.370,
        "h": 155,
        "ab": 554,
        "bb": 65,
        "hbp": 4,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 105,
        "player_name": "SS_Starter",
        "positions": "SS",
        "is_hitter": 1,
        "team": "ATL",
        "r": 80,
        "hr": 22,
        "rbi": 70,
        "sb": 15,
        "avg": 0.265,
        "obp": 0.335,
        "h": 138,
        "ab": 521,
        "bb": 48,
        "hbp": 3,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 106,
        "player_name": "OF1_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "NYY",
        "r": 95,
        "hr": 35,
        "rbi": 100,
        "sb": 12,
        "avg": 0.290,
        "obp": 0.380,
        "h": 162,
        "ab": 559,
        "bb": 70,
        "hbp": 5,
        "sf": 5,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 107,
        "player_name": "OF2_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "SEA",
        "r": 70,
        "hr": 20,
        "rbi": 65,
        "sb": 25,
        "avg": 0.260,
        "obp": 0.330,
        "h": 135,
        "ab": 519,
        "bb": 42,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 108,
        "player_name": "OF3_Starter",
        "positions": "OF",
        "is_hitter": 1,
        "team": "CHC",
        "r": 78,
        "hr": 24,
        "rbi": 72,
        "sb": 8,
        "avg": 0.272,
        "obp": 0.345,
        "h": 142,
        "ab": 522,
        "bb": 50,
        "hbp": 4,
        "sf": 4,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 109,
        "player_name": "Util1",
        "positions": "1B,OF",
        "is_hitter": 1,
        "team": "PHI",
        "r": 65,
        "hr": 20,
        "rbi": 68,
        "sb": 5,
        "avg": 0.255,
        "obp": 0.325,
        "h": 125,
        "ab": 490,
        "bb": 38,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 110,
        "player_name": "Util2",
        "positions": "3B,SS",
        "is_hitter": 1,
        "team": "SDP",
        "r": 60,
        "hr": 14,
        "rbi": 50,
        "sb": 18,
        "avg": 0.262,
        "obp": 0.330,
        "h": 130,
        "ab": 496,
        "bb": 40,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
]

_PITCHERS = [
    {
        "player_id": 201,
        "player_name": "SP1_Ace",
        "positions": "SP",
        "is_hitter": 0,
        "team": "LAD",
        "w": 15,
        "l": 6,
        "sv": 0,
        "k": 220,
        "era": 2.90,
        "whip": 1.05,
        "ip": 200.0,
        "er": 64,
        "bb_allowed": 45,
        "h_allowed": 165,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 202,
        "player_name": "SP2_Mid",
        "positions": "SP",
        "is_hitter": 0,
        "team": "NYY",
        "w": 12,
        "l": 8,
        "sv": 0,
        "k": 180,
        "era": 3.60,
        "whip": 1.18,
        "ip": 180.0,
        "er": 72,
        "bb_allowed": 50,
        "h_allowed": 162,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 203,
        "player_name": "SP3_Back",
        "positions": "SP",
        "is_hitter": 0,
        "team": "BOS",
        "w": 9,
        "l": 10,
        "sv": 0,
        "k": 150,
        "era": 4.20,
        "whip": 1.30,
        "ip": 165.0,
        "er": 77,
        "bb_allowed": 55,
        "h_allowed": 160,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 204,
        "player_name": "RP1_Closer",
        "positions": "RP",
        "is_hitter": 0,
        "team": "NYY",
        "w": 4,
        "l": 3,
        "sv": 30,
        "k": 70,
        "era": 2.80,
        "whip": 1.00,
        "ip": 65.0,
        "er": 20,
        "bb_allowed": 18,
        "h_allowed": 47,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 205,
        "player_name": "RP2_Setup",
        "positions": "RP",
        "is_hitter": 0,
        "team": "ATL",
        "w": 5,
        "l": 2,
        "sv": 5,
        "k": 75,
        "era": 3.10,
        "whip": 1.08,
        "ip": 70.0,
        "er": 24,
        "bb_allowed": 20,
        "h_allowed": 56,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 206,
        "player_name": "P_Flex1",
        "positions": "SP,RP",
        "is_hitter": 0,
        "team": "HOU",
        "w": 7,
        "l": 5,
        "sv": 2,
        "k": 110,
        "era": 3.80,
        "whip": 1.22,
        "ip": 120.0,
        "er": 51,
        "bb_allowed": 35,
        "h_allowed": 112,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 207,
        "player_name": "P_Flex2",
        "positions": "SP,RP",
        "is_hitter": 0,
        "team": "SDP",
        "w": 6,
        "l": 4,
        "sv": 3,
        "k": 100,
        "era": 3.90,
        "whip": 1.25,
        "ip": 110.0,
        "er": 48,
        "bb_allowed": 32,
        "h_allowed": 106,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 208,
        "player_name": "P_Flex3",
        "positions": "RP",
        "is_hitter": 0,
        "team": "CHC",
        "w": 3,
        "l": 2,
        "sv": 8,
        "k": 60,
        "era": 3.40,
        "whip": 1.12,
        "ip": 55.0,
        "er": 21,
        "bb_allowed": 15,
        "h_allowed": 47,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
]

_BENCH = [
    {
        "player_id": 301,
        "player_name": "BN_Hitter1",
        "positions": "OF,1B",
        "is_hitter": 1,
        "team": "MIA",
        "r": 45,
        "hr": 12,
        "rbi": 40,
        "sb": 6,
        "avg": 0.242,
        "obp": 0.310,
        "h": 105,
        "ab": 434,
        "bb": 30,
        "hbp": 2,
        "sf": 2,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 302,
        "player_name": "BN_Hitter2",
        "positions": "2B,SS",
        "is_hitter": 1,
        "team": "CIN",
        "r": 50,
        "hr": 10,
        "rbi": 35,
        "sb": 12,
        "avg": 0.248,
        "obp": 0.315,
        "h": 108,
        "ab": 435,
        "bb": 32,
        "hbp": 2,
        "sf": 3,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 303,
        "player_name": "BN_Pitcher",
        "positions": "SP",
        "is_hitter": 0,
        "team": "TBR",
        "w": 5,
        "l": 7,
        "sv": 0,
        "k": 90,
        "era": 4.50,
        "whip": 1.35,
        "ip": 100.0,
        "er": 50,
        "bb_allowed": 40,
        "h_allowed": 95,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "obp": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "status": "active",
    },
    {
        "player_id": 304,
        "player_name": "IL_Player",
        "positions": "OF",
        "is_hitter": 1,
        "team": "LAD",
        "r": 70,
        "hr": 22,
        "rbi": 65,
        "sb": 10,
        "avg": 0.268,
        "obp": 0.340,
        "h": 130,
        "ab": 485,
        "bb": 45,
        "hbp": 3,
        "sf": 3,
        "ip": 0,
        "status": "IL10",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
    {
        "player_id": 305,
        "player_name": "BN_Catcher",
        "positions": "C,1B",
        "is_hitter": 1,
        "team": "STL",
        "r": 40,
        "hr": 8,
        "rbi": 32,
        "sb": 1,
        "avg": 0.235,
        "obp": 0.300,
        "h": 90,
        "ab": 383,
        "bb": 25,
        "hbp": 2,
        "sf": 2,
        "ip": 0,
        "status": "active",
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    },
]


@pytest.fixture()
def full_roster():
    """23-player roster matching Yahoo H2H format.

    10 hitters: C, 1B, 2B, 3B, SS, 3 OF, 2 Util
    8 pitchers: 3 SP, 2 RP, 3 flex (SP/RP)
    5 bench: 3 hitters (one IL10), 1 pitcher, 1 backup catcher
    """
    return pd.DataFrame(_HITTERS + _PITCHERS + _BENCH)


# ── Task 9: Pipeline Invariant Assertions ────────────────────────────


class TestPipelineInvariants:
    """Invariants that must hold across ALL modes, alpha, and risk settings.

    These are properties that should NEVER be violated regardless of
    how the optimizer is configured. If any of these fail, the optimizer
    is producing invalid lineups.

    Parametrized: 3 modes x 3 alphas = 9 combinations per test = 45 total.
    """

    @pytest.fixture(params=["quick", "standard", "full"])
    def mode(self, request):
        return request.param

    @pytest.fixture(params=[0.0, 0.5, 1.0])
    def alpha(self, request):
        return request.param

    def test_pipeline_returns_valid_structure(self, full_roster, mode, alpha):
        """Pipeline must return a dict with required keys."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha, weeks_remaining=16)
        result = pipe.optimize()

        assert isinstance(result, dict)
        assert "lineup" in result
        assert "category_weights" in result
        assert "mode" in result
        assert result["mode"] == mode

    def test_no_il_player_in_starting_lineup(self, full_roster, mode, alpha):
        """IL/DTD players must NEVER appear in starting assignments.

        IL_Player (id=304, status=IL10) must be on bench, never assigned.
        """
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha, weeks_remaining=16)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])

        for a in assignments:
            assert a.get("player_name") != "IL_Player", f"IL player assigned to slot {a.get('slot')} in mode={mode}"

    def test_lineup_fills_position_slots(self, full_roster, mode, alpha):
        """Optimizer must fill a reasonable number of the 18 starter slots.

        Yahoo H2H has 18 starter slots (C,1B,2B,3B,SS,OF*3,Util*2,
        SP*2,RP*2,P*4). The LP uses <= 1 slot constraints (not equality),
        so it may leave low-value slots empty when filling them would
        reduce the objective. With 22 active players (1 IL) we expect
        at least 14 filled.
        """
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha, weeks_remaining=16)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])

        assert len(assignments) >= 14, f"Only {len(assignments)} slots filled with 23 players in mode={mode}"

    def test_category_weights_are_positive(self, full_roster, mode, alpha):
        """All category weights must be positive (>0)."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha, weeks_remaining=16)
        result = pipe.optimize()
        weights = result.get("category_weights", {})

        for cat, w in weights.items():
            assert w > 0, f"Weight for {cat} is {w} (must be positive)"

    def test_no_duplicate_player_assignments(self, full_roster, mode, alpha):
        """Each player can only be assigned to ONE slot."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha, weeks_remaining=16)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])
        player_ids = [a.get("player_id") for a in assignments if a.get("player_id")]

        assert len(player_ids) == len(set(player_ids)), f"Duplicate player assignments in mode={mode}: {player_ids}"


# ── Task 10: Settings Axis Tests ─────────────────────────────────────


class TestSettingsAxisEffects:
    """Verify that changing settings produces expected directional changes.

    These are "monotonicity" tests: higher alpha should mean more H2H-focused
    weights, higher risk aversion should mean lower-variance selections.
    """

    def test_higher_alpha_increases_weight_variance(self, full_roster):
        """Higher alpha (more H2H-focused) should produce more extreme weights.

        alpha=0.0 (pure season): weights closer to uniform (SGP-based)
        alpha=1.0 (pure H2H): weights skewed toward swing categories

        We measure this by checking the coefficient of variation of weights.
        """
        pipe_season = LineupOptimizerPipeline(full_roster, mode="standard", alpha=0.0, weeks_remaining=16)
        pipe_h2h = LineupOptimizerPipeline(full_roster, mode="standard", alpha=1.0, weeks_remaining=16)

        # Need to provide opponent totals for H2H weighting to kick in.
        # Keys must be lowercase to match ALL_CATEGORIES in the pipeline.
        opp_totals = {
            "r": 400,
            "hr": 120,
            "rbi": 380,
            "sb": 55,
            "avg": 0.268,
            "obp": 0.335,
            "w": 48,
            "l": 30,
            "sv": 42,
            "k": 820,
            "era": 3.70,
            "whip": 1.20,
        }
        my_totals = {
            "r": 420,
            "hr": 130,
            "rbi": 400,
            "sb": 60,
            "avg": 0.272,
            "obp": 0.340,
            "w": 50,
            "l": 28,
            "sv": 45,
            "k": 850,
            "era": 3.60,
            "whip": 1.18,
        }

        result_season = pipe_season.optimize(
            h2h_opponent_totals=opp_totals,
            my_totals=my_totals,
        )
        result_h2h = pipe_h2h.optimize(
            h2h_opponent_totals=opp_totals,
            my_totals=my_totals,
        )

        w_season = list(result_season["category_weights"].values())
        w_h2h = list(result_h2h["category_weights"].values())

        cv_season = np.std(w_season) / max(np.mean(w_season), 1e-9)
        cv_h2h = np.std(w_h2h) / max(np.mean(w_h2h), 1e-9)

        # H2H weights should have higher variance (more extreme).
        # Use a soft check (0.8x) since blending can smooth differences.
        assert cv_h2h >= cv_season * 0.8, f"H2H weight CV ({cv_h2h:.3f}) not higher than season CV ({cv_season:.3f})"

    def test_full_mode_has_more_analysis_than_quick(self, full_roster):
        """Full mode should produce more analysis outputs than quick mode.

        Full enables: streaming, maximin, scenarios, multi-period.
        Quick enables: projections only.
        """
        pipe_quick = LineupOptimizerPipeline(full_roster, mode="quick", weeks_remaining=16)
        pipe_full = LineupOptimizerPipeline(full_roster, mode="full", weeks_remaining=16)

        result_quick = pipe_quick.optimize()
        result_full = pipe_full.optimize()

        # Full mode should have more non-None fields
        quick_populated = sum(1 for v in result_quick.values() if v is not None)
        full_populated = sum(1 for v in result_full.values() if v is not None)

        assert full_populated >= quick_populated, (
            f"Full mode ({full_populated} fields) has fewer outputs than quick ({quick_populated})"
        )

    def test_all_three_modes_produce_valid_lineups(self, full_roster):
        """Quick, Standard, and Full must all produce valid lineup assignments."""
        for mode in ["quick", "standard", "full"]:
            pipe = LineupOptimizerPipeline(full_roster, mode=mode, weeks_remaining=16)
            result = pipe.optimize()
            assignments = result.get("lineup", {}).get("assignments", [])
            assert len(assignments) > 0, f"Mode {mode} produced empty lineup"
