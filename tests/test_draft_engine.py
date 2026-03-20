"""Tests for the DraftRecommendationEngine.

Covers:
  - Mode presets and initialization
  - Each enhancement stage independently
  - Enhanced pick_score formula math
  - Category balance algorithm
  - FIP correction
  - Contextual factors (streaming, closer, lineup protection, flex)
  - Buy/fair/avoid classification
  - Graceful degradation when optional modules unavailable
  - End-to-end recommend() pipeline
  - Timing instrumentation
  - Edge cases
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.draft_engine import (
    ALL_CATEGORIES_UPPER,
    BASE_SGP_FLOOR,
    BAYESIAN_AVAILABLE,
    ERA_WEIGHT,
    FIP_WEIGHT,
    H2H_ENGINE_AVAILABLE,
    INJURY_PROCESS_AVAILABLE,
    MULT_CEILING,
    MULT_FLOOR,
    DraftRecommendationEngine,
)
from src.draft_state import DraftState
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def config():
    """Standard 12-team H2H league config."""
    return LeagueConfig()


@pytest.fixture
def engine(config):
    """Standard-mode engine."""
    return DraftRecommendationEngine(config, mode="standard")


@pytest.fixture
def quick_engine(config):
    """Quick-mode engine."""
    return DraftRecommendationEngine(config, mode="quick")


@pytest.fixture
def full_engine(config):
    """Full-mode engine."""
    return DraftRecommendationEngine(config, mode="full")


@pytest.fixture
def draft_state():
    """Basic DraftState with 12 teams, 23 rounds."""
    return DraftState(num_teams=12, num_rounds=23, user_team_index=0)


@pytest.fixture
def sample_pool():
    """Small player pool with realistic columns."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Mike Trout",
                "team": "LAA",
                "positions": "OF",
                "is_hitter": True,
                "is_injured": False,
                "pa": 550,
                "ab": 500,
                "h": 140,
                "r": 90,
                "hr": 35,
                "rbi": 90,
                "sb": 5,
                "avg": 0.280,
                "obp": 0.370,
                "bb": 70,
                "hbp": 5,
                "sf": 5,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 5,
                "pick_score": 8.5,
                "health_score": 0.70,
                "age": 34,
            },
            {
                "player_id": 2,
                "name": "Shohei Ohtani",
                "team": "LAD",
                "positions": "OF,Util",
                "is_hitter": True,
                "is_injured": False,
                "pa": 600,
                "ab": 540,
                "h": 160,
                "r": 100,
                "hr": 45,
                "rbi": 110,
                "sb": 15,
                "avg": 0.296,
                "obp": 0.380,
                "bb": 75,
                "hbp": 3,
                "sf": 4,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 1,
                "pick_score": 12.0,
                "health_score": 0.90,
                "age": 31,
            },
            {
                "player_id": 3,
                "name": "Corbin Burnes",
                "team": "ARI",
                "positions": "SP",
                "is_hitter": False,
                "is_injured": False,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "ip": 195,
                "w": 14,
                "l": 8,
                "sv": 0,
                "k": 210,
                "era": 3.10,
                "whip": 1.05,
                "er": 67,
                "bb_allowed": 45,
                "h_allowed": 160,
                "adp": 15,
                "pick_score": 7.2,
                "health_score": 0.85,
                "age": 31,
            },
            {
                "player_id": 4,
                "name": "Emmanuel Clase",
                "team": "CLE",
                "positions": "RP",
                "is_hitter": False,
                "is_injured": False,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "ip": 65,
                "w": 3,
                "l": 3,
                "sv": 38,
                "k": 65,
                "era": 2.50,
                "whip": 0.95,
                "er": 18,
                "bb_allowed": 15,
                "h_allowed": 47,
                "adp": 45,
                "pick_score": 5.0,
                "health_score": 0.95,
                "age": 28,
            },
            {
                "player_id": 5,
                "name": "Jazz Chisholm",
                "team": "NYY",
                "positions": "2B,3B,OF",
                "is_hitter": True,
                "is_injured": False,
                "pa": 520,
                "ab": 470,
                "h": 120,
                "r": 75,
                "hr": 22,
                "rbi": 65,
                "sb": 25,
                "avg": 0.255,
                "obp": 0.320,
                "bb": 45,
                "hbp": 3,
                "sf": 2,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 30,
                "pick_score": 6.0,
                "health_score": 0.75,
                "age": 27,
            },
            {
                "player_id": 6,
                "name": "Streaming Pitcher",
                "team": "MIA",
                "positions": "SP",
                "is_hitter": False,
                "is_injured": False,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "ip": 60,
                "w": 3,
                "l": 5,
                "sv": 0,
                "k": 55,
                "era": 4.80,
                "whip": 1.45,
                "er": 32,
                "bb_allowed": 30,
                "h_allowed": 57,
                "adp": 200,
                "pick_score": 1.5,
                "health_score": 0.90,
                "age": 26,
            },
        ]
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Initialization and Mode Tests
# ═══════════════════════════════════════════════════════════════════


class TestInitialization:
    """Test engine construction and mode presets."""

    def test_standard_mode_init(self, config):
        engine = DraftRecommendationEngine(config, mode="standard")
        assert engine.mode == "standard"
        assert engine.settings["enable_bayesian"] is True
        assert engine.settings["enable_ml"] is False

    def test_quick_mode_init(self, config):
        engine = DraftRecommendationEngine(config, mode="quick")
        assert engine.mode == "quick"
        assert engine.settings["enable_bayesian"] is False
        assert engine.settings["enable_park_factors"] is True

    def test_full_mode_init(self, config):
        engine = DraftRecommendationEngine(config, mode="full")
        assert engine.mode == "full"
        assert engine.settings["enable_ml"] is True
        assert engine.settings["enable_bayesian"] is True

    def test_invalid_mode_raises(self, config):
        with pytest.raises(ValueError, match="Unknown mode"):
            DraftRecommendationEngine(config, mode="turbo")

    def test_config_stored(self, engine, config):
        assert engine.config is config

    def test_sgp_calc_initialized(self, engine):
        assert engine.sgp_calc is not None

    def test_timing_empty_initially(self, engine):
        assert engine.timing == {}

    def test_mode_presets_keys(self):
        assert set(DraftRecommendationEngine.MODE_PRESETS.keys()) == {"quick", "standard", "full"}

    def test_all_modes_have_same_keys(self):
        keys = None
        for mode, preset in DraftRecommendationEngine.MODE_PRESETS.items():
            if keys is None:
                keys = set(preset.keys())
            else:
                assert set(preset.keys()) == keys, f"Mode '{mode}' has different keys"

    def test_quick_is_subset_of_standard(self):
        quick = DraftRecommendationEngine.MODE_PRESETS["quick"]
        standard = DraftRecommendationEngine.MODE_PRESETS["standard"]
        # Every feature enabled in quick must also be enabled in standard
        for key, val in quick.items():
            if val:
                assert standard[key], f"Quick enables {key} but standard doesn't"

    def test_standard_is_subset_of_full(self):
        standard = DraftRecommendationEngine.MODE_PRESETS["standard"]
        full = DraftRecommendationEngine.MODE_PRESETS["full"]
        for key, val in standard.items():
            if val:
                assert full[key], f"Standard enables {key} but full doesn't"


# ═══════════════════════════════════════════════════════════════════
# 2. Park Factor Tests
# ═══════════════════════════════════════════════════════════════════


class TestParkFactors:
    """Test park factor adjustment stage."""

    def test_park_factor_applied(self, engine, sample_pool, draft_state):
        pf = {"LAA": 1.05, "LAD": 1.02, "ARI": 1.06, "CLE": 0.97, "NYY": 1.02, "MIA": 0.88}
        result = engine._apply_park_factors(sample_pool.copy(), pf)
        trout = result[result["player_id"] == 1].iloc[0]
        assert trout["park_factor_adj"] == pytest.approx(1.05)

    def test_park_factor_default_one(self, engine, sample_pool):
        # Unknown team gets 1.0
        pf = {"XYZ": 1.10}
        result = engine._apply_park_factors(sample_pool.copy(), pf)
        trout = result[result["player_id"] == 1].iloc[0]
        assert trout["park_factor_adj"] == pytest.approx(1.0)

    def test_park_factor_none_dict(self, engine, sample_pool):
        result = engine._apply_park_factors(sample_pool.copy(), None)
        # Should still work with bootstrap park factors if available
        assert "park_factor_adj" in result.columns

    def test_park_factor_empty_dict(self, engine, sample_pool):
        result = engine._apply_park_factors(sample_pool.copy(), {})
        # Empty dict passed explicitly — should return pool unchanged (all 1.0)
        # because `if not factors:` short-circuits
        assert (result["park_factor_adj"] == 1.0).all()

    def test_park_factor_missing_team(self, engine):
        pool = pd.DataFrame([{"player_id": 1, "name": "Test", "team": None, "pick_score": 5.0}])
        pool["park_factor_adj"] = 1.0
        result = engine._apply_park_factors(pool, {"LAA": 1.05})
        assert result.iloc[0]["park_factor_adj"] == pytest.approx(1.0)

    def test_park_factor_case_insensitive(self, engine, sample_pool):
        pf = {"laa": 1.05}
        result = engine._apply_park_factors(sample_pool.copy(), pf)
        trout = result[result["player_id"] == 1].iloc[0]
        # team="LAA" vs key="laa" — function upper-cases
        assert trout["park_factor_adj"] == pytest.approx(1.05)


# ═══════════════════════════════════════════════════════════════════
# 3. Injury Probability Tests
# ═══════════════════════════════════════════════════════════════════


class TestInjuryProbability:
    """Test injury probability stage."""

    @pytest.mark.skipif(not INJURY_PROCESS_AVAILABLE, reason="injury_process not available")
    def test_injury_prob_applied(self, engine, sample_pool):
        result = engine._apply_injury_probability(sample_pool.copy())
        trout = result[result["player_id"] == 1].iloc[0]
        # Trout: health=0.70, age=34 — should have non-trivial injury prob
        assert trout["injury_probability"] > 0.0
        assert trout["injury_probability"] <= 0.95

    @pytest.mark.skipif(not INJURY_PROCESS_AVAILABLE, reason="injury_process not available")
    def test_injury_prob_healthy_player(self, engine, sample_pool):
        result = engine._apply_injury_probability(sample_pool.copy())
        clase = result[result["player_id"] == 4].iloc[0]
        # Clase: health=0.95, age=28 — should be low
        assert clase["injury_probability"] < 0.10

    @pytest.mark.skipif(not INJURY_PROCESS_AVAILABLE, reason="injury_process not available")
    def test_injury_prob_fragile_vs_healthy(self, engine, sample_pool):
        result = engine._apply_injury_probability(sample_pool.copy())
        trout = result[result["player_id"] == 1].iloc[0]
        ohtani = result[result["player_id"] == 2].iloc[0]
        # Trout (0.70 health, age 34) should be riskier than Ohtani (0.90, 31)
        assert trout["injury_probability"] > ohtani["injury_probability"]

    @pytest.mark.skipif(not INJURY_PROCESS_AVAILABLE, reason="injury_process not available")
    def test_injury_prob_missing_health_score(self, engine):
        pool = pd.DataFrame([{"player_id": 99, "name": "Unknown", "is_hitter": True}])
        pool["injury_probability"] = 0.0
        result = engine._apply_injury_probability(pool)
        # Should use DEFAULT_HEALTH_SCORE (0.85)
        assert result.iloc[0]["injury_probability"] >= 0.0

    @pytest.mark.skipif(not INJURY_PROCESS_AVAILABLE, reason="injury_process not available")
    def test_injury_prob_missing_age(self, engine):
        pool = pd.DataFrame([{"player_id": 99, "name": "Unknown", "is_hitter": True, "health_score": 0.60}])
        pool["injury_probability"] = 0.0
        result = engine._apply_injury_probability(pool)
        assert result.iloc[0]["injury_probability"] > 0.0


# ═══════════════════════════════════════════════════════════════════
# 4. Statcast Delta Tests
# ═══════════════════════════════════════════════════════════════════


class TestStatcastDelta:
    """Test Statcast xwOBA-wOBA delta calculation."""

    def test_statcast_delta_with_xwoba(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "Buy", "xwoba": 0.380, "avg": 0.250, "obp": 0.320},
                {"player_id": 2, "name": "Avoid", "xwoba": 0.280, "avg": 0.290, "obp": 0.350},
            ]
        )
        pool["statcast_delta"] = 0.0
        result = engine._apply_statcast_delta(pool)
        # Buy: xwoba=0.380 >> woba_approx ~0.368 → positive delta
        assert result.iloc[0]["statcast_delta"] > 0
        # Avoid: xwoba=0.280 << woba_approx ~0.4025 → negative delta
        assert result.iloc[1]["statcast_delta"] < 0

    def test_statcast_delta_no_xwoba(self, engine, sample_pool):
        # sample_pool has no xwoba column
        result = engine._apply_statcast_delta(sample_pool.copy())
        assert (result["statcast_delta"] == 0.0).all()

    def test_statcast_delta_clamped(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "Extreme", "xwoba": 1.0, "avg": 0.100, "obp": 0.100},
            ]
        )
        pool["statcast_delta"] = 0.0
        result = engine._apply_statcast_delta(pool)
        assert result.iloc[0]["statcast_delta"] <= 1.0
        assert result.iloc[0]["statcast_delta"] >= -1.0


# ═══════════════════════════════════════════════════════════════════
# 5. FIP Correction Tests
# ═══════════════════════════════════════════════════════════════════


class TestFIPCorrection:
    """Test FIP-based ERA correction."""

    def test_fip_correction_applied(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "P1", "is_hitter": False, "fip": 3.00, "era": 4.00, "ip": 180},
            ]
        )
        pool["fip_era_adj"] = 0.0
        result = engine._apply_fip_correction(pool)
        expected_era = FIP_WEIGHT * 3.00 + ERA_WEIGHT * 4.00  # 0.6*3 + 0.4*4 = 3.40
        assert result.iloc[0]["era"] == pytest.approx(expected_era, abs=0.01)
        # fip_era_adj = original_era - adjusted = 4.00 - 3.40 = 0.60
        assert result.iloc[0]["fip_era_adj"] == pytest.approx(0.60, abs=0.01)

    def test_fip_correction_hitter_skipped(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "H1", "is_hitter": True, "fip": 3.00, "era": 4.00},
            ]
        )
        pool["fip_era_adj"] = 0.0
        result = engine._apply_fip_correction(pool)
        assert result.iloc[0]["era"] == pytest.approx(4.00)

    def test_fip_correction_no_fip_column(self, engine, sample_pool):
        result = engine._apply_fip_correction(sample_pool.copy())
        # No fip column — should pass through unchanged
        assert "era" in result.columns

    def test_fip_correction_nan_fip_skipped(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "P1", "is_hitter": False, "fip": np.nan, "era": 4.00},
            ]
        )
        pool["fip_era_adj"] = 0.0
        result = engine._apply_fip_correction(pool)
        assert result.iloc[0]["era"] == pytest.approx(4.00)

    def test_fip_correction_zero_fip_skipped(self, engine):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "P1", "is_hitter": False, "fip": 0.0, "era": 4.00},
            ]
        )
        pool["fip_era_adj"] = 0.0
        result = engine._apply_fip_correction(pool)
        assert result.iloc[0]["era"] == pytest.approx(4.00)


# ═══════════════════════════════════════════════════════════════════
# 6. Contextual Factor Tests
# ═══════════════════════════════════════════════════════════════════


class TestContextualFactors:
    """Test streaming penalty, closer hierarchy, lineup protection, and flex bonus."""

    def test_streaming_penalty_low_ip(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        # Player 6 (Streaming Pitcher): IP=60 < 80 — should get penalty
        streamer = result[result["player_id"] == 6].iloc[0]
        assert streamer["streaming_penalty"] == pytest.approx(-0.3)

    def test_streaming_penalty_normal_sp(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        burnes = result[result["player_id"] == 3].iloc[0]
        # IP=195 > 80 — no penalty
        assert burnes["streaming_penalty"] == pytest.approx(0.0)

    def test_closer_hierarchy_elite(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        clase = result[result["player_id"] == 4].iloc[0]
        # SV=38 >= 20 — bonus = (38-15)*0.1 = 2.3, clamped to 2.0
        assert clase["closer_hierarchy_bonus"] == pytest.approx(2.0)

    def test_closer_hierarchy_non_closer(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        burnes = result[result["player_id"] == 3].iloc[0]
        # SV=0 — no bonus
        assert burnes["closer_hierarchy_bonus"] == pytest.approx(0.0)

    def test_flex_bonus_always_zero(self, engine, sample_pool, draft_state):
        """Flex bonus removed — VORP already includes multi-position premium."""
        pool = sample_pool.copy()
        pool["flex_bonus"] = 0.0  # initialized by enhance_player_pool
        result = engine._apply_contextual_factors(pool, draft_state)
        assert (result["flex_bonus"] == 0.0).all()

    def test_flex_bonus_multi_pos_still_zero(self, engine, draft_state):
        """Even multi-position players get flex_bonus=0.0 (VORP handles it)."""
        pool = pd.DataFrame(
            [
                {
                    "player_id": 99,
                    "name": "MultiPos",
                    "team": "LAA",
                    "positions": "SS,2B,3B",
                    "is_hitter": True,
                    "pick_score": 5.0,
                    "sv": 0,
                    "ip": 0,
                }
            ]
        )
        pool["streaming_penalty"] = 0.0
        pool["closer_hierarchy_bonus"] = 0.0
        pool["lineup_protection_bonus"] = 0.0
        pool["flex_bonus"] = 0.0
        result = engine._apply_contextual_factors(pool, draft_state)
        assert result.iloc[0]["flex_bonus"] == pytest.approx(0.0)

    def test_lineup_protection_bonus_hitter(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        # All hitters should have some lineup protection bonus >= 0
        hitters = result[result["is_hitter"] == True]  # noqa: E712
        assert (hitters["lineup_protection_bonus"] >= 0).all()

    def test_lineup_protection_pitcher_zero(self, engine, sample_pool, draft_state):
        result = engine._apply_contextual_factors(sample_pool.copy(), draft_state)
        burnes = result[result["player_id"] == 3].iloc[0]
        assert burnes["lineup_protection_bonus"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# 7. Category Balance Tests
# ═══════════════════════════════════════════════════════════════════


class TestCategoryBalance:
    """Test category balance weighting algorithm."""

    def test_median_totals_computation(self, engine):
        totals = [
            {
                "R": 100,
                "HR": 30,
                "AVG": 0.260,
                "ERA": 3.50,
                "WHIP": 1.20,
                "RBI": 80,
                "SB": 10,
                "OBP": 0.330,
                "W": 10,
                "L": 8,
                "SV": 5,
                "K": 150,
            },
            {
                "R": 80,
                "HR": 20,
                "AVG": 0.250,
                "ERA": 4.00,
                "WHIP": 1.30,
                "RBI": 70,
                "SB": 15,
                "OBP": 0.320,
                "W": 8,
                "L": 10,
                "SV": 10,
                "K": 120,
            },
            {
                "R": 120,
                "HR": 40,
                "AVG": 0.280,
                "ERA": 3.00,
                "WHIP": 1.10,
                "RBI": 100,
                "SB": 5,
                "OBP": 0.350,
                "W": 12,
                "L": 6,
                "SV": 15,
                "K": 180,
            },
        ]
        medians = engine._compute_median_totals(totals)
        assert medians["R"] == pytest.approx(100)
        assert medians["HR"] == pytest.approx(30)
        assert medians["ERA"] == pytest.approx(3.50)

    def test_median_totals_empty(self, engine):
        assert engine._compute_median_totals([]) == {}

    def test_simple_category_weights_below_median(self):
        my = {
            "R": 50,
            "HR": 10,
            "RBI": 40,
            "SB": 5,
            "AVG": 0.240,
            "OBP": 0.300,
            "W": 5,
            "L": 10,
            "SV": 2,
            "K": 80,
            "ERA": 4.50,
            "WHIP": 1.40,
        }
        med = {
            "R": 80,
            "HR": 25,
            "RBI": 70,
            "SB": 12,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 8,
            "SV": 8,
            "K": 140,
            "ERA": 3.80,
            "WHIP": 1.20,
        }
        weights = DraftRecommendationEngine._simple_category_weights(my, med)
        # Below median in R → boost (1.2 before normalization)
        # ERA: my=4.50 > med=3.80, inverse → user is worse → boost
        assert len(weights) == 12

    def test_simple_category_weights_normalized(self):
        my = {
            "R": 80,
            "HR": 25,
            "RBI": 70,
            "SB": 12,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 8,
            "SV": 8,
            "K": 140,
            "ERA": 3.80,
            "WHIP": 1.20,
        }
        med = dict(my)  # exactly at median
        weights = DraftRecommendationEngine._simple_category_weights(my, med)
        mean_w = np.mean(list(weights.values()))
        assert mean_w == pytest.approx(1.0, abs=0.01)

    def test_draft_progress_scale_early(self, engine):
        # Round 1: minimal balance influence
        scale = engine._draft_progress_scale(1)
        assert scale < 1.0

    def test_draft_progress_scale_mid(self, engine):
        scale = engine._draft_progress_scale(12)
        assert 1.0 <= scale <= 1.5

    def test_draft_progress_scale_late(self, engine):
        scale = engine._draft_progress_scale(20)
        assert scale == pytest.approx(1.5)

    def test_draft_progress_scale_round_8(self, engine):
        scale = engine._draft_progress_scale(8)
        assert scale == pytest.approx(1.0)

    def test_draft_progress_scale_round_17(self, engine):
        scale = engine._draft_progress_scale(17)
        assert scale == pytest.approx(1.5)

    def test_category_balance_multiplier_range(self, engine, sample_pool, draft_state):
        # Make a pick so user has roster totals
        draft_state.make_pick(2, "Shohei Ohtani", "OF,Util")
        result = engine._apply_category_balance(sample_pool.copy(), draft_state)
        # All multipliers should be in [0.8, 1.2]
        assert (result["category_balance_multiplier"] >= 0.8).all()
        assert (result["category_balance_multiplier"] <= 1.2).all()


# ═══════════════════════════════════════════════════════════════════
# 8. Enhanced Pick Score Formula Tests
# ═══════════════════════════════════════════════════════════════════


class TestEnhancedPickScore:
    """Test the enhanced pick_score formula math."""

    def test_baseline_score_no_adjustments(self, engine):
        row = pd.Series(
            {
                "pick_score": 5.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        assert score == pytest.approx(5.0)

    def test_park_factor_boost(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.38,  # Coors
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        assert score == pytest.approx(10.0 * 1.38)

    def test_injury_penalty(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.5,  # 50% injury risk
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # mult = 1.0 * (1 - 0.5*0.3) = 0.85
        assert score == pytest.approx(10.0 * 0.85)

    def test_statcast_delta_boost(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.5,  # strong positive signal
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # mult = (1 + 0.5*0.15) = 1.075
        assert score == pytest.approx(10.0 * 1.075)

    def test_closer_bonus_additive(self, engine):
        row = pd.Series(
            {
                "pick_score": 5.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 2.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        assert score == pytest.approx(5.0 + 2.0)

    def test_flex_bonus_additive(self, engine):
        row = pd.Series(
            {
                "pick_score": 5.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.3,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        assert score == pytest.approx(5.3)

    def test_mult_clamped_to_floor(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 0.1,  # extreme
                "park_factor_adj": 0.1,  # extreme
                "injury_probability": 0.95,
                "statcast_delta": -1.0,
                "platoon_factor": 0.5,
                "contract_year_factor": 0.5,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # mult should be clamped to 0.5
        assert score >= 10.0 * MULT_FLOOR

    def test_mult_clamped_to_ceiling(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 2.0,
                "park_factor_adj": 2.0,
                "injury_probability": 0.0,
                "statcast_delta": 1.0,
                "platoon_factor": 1.5,
                "contract_year_factor": 1.5,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # mult clamped to 1.5, no additive
        assert score == pytest.approx(10.0 * MULT_CEILING)

    def test_zero_pick_score_floor(self, engine):
        row = pd.Series(
            {
                "pick_score": 0.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # base_sgp floored to 0.01
        assert score == pytest.approx(BASE_SGP_FLOOR)

    def test_negative_pick_score_floor(self, engine):
        row = pd.Series(
            {
                "pick_score": -5.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        assert score == pytest.approx(BASE_SGP_FLOOR)

    def test_streaming_penalty_reduces_score(self, engine):
        base_row = pd.Series(
            {
                "pick_score": 5.0,
                "category_balance_multiplier": 1.0,
                "park_factor_adj": 1.0,
                "injury_probability": 0.0,
                "statcast_delta": 0.0,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.0,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.0,
                "flex_bonus": 0.0,
            }
        )
        penalized = base_row.copy()
        penalized["streaming_penalty"] = -0.5
        base_score = engine._compute_enhanced_pick_score(base_row)
        pen_score = engine._compute_enhanced_pick_score(penalized)
        assert pen_score < base_score
        assert pen_score == pytest.approx(base_score - 0.5)

    def test_combined_multiplicative_and_additive(self, engine):
        row = pd.Series(
            {
                "pick_score": 10.0,
                "category_balance_multiplier": 1.1,
                "park_factor_adj": 1.05,
                "injury_probability": 0.10,
                "statcast_delta": 0.03,
                "platoon_factor": 1.0,
                "contract_year_factor": 1.0,
                "streaming_penalty": 0.0,
                "lineup_protection_bonus": 0.2,
                "closer_hierarchy_bonus": 0.0,
                "ml_correction": 0.5,
                "flex_bonus": 0.12,
            }
        )
        score = engine._compute_enhanced_pick_score(row)
        # Hand-calculate:
        mult = 1.1 * 1.05 * (1 - 0.10 * 0.3) * (1 + 0.03 * 0.15) * 1.0 * 1.0
        mult = np.clip(mult, 0.5, 1.5)
        additive = 0.0 + 0.2 + 0.0 + 0.5 * 0.1 + 0.12
        expected = 10.0 * float(mult) + additive
        assert score == pytest.approx(expected, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# 9. Buy/Fair/Avoid Classification Tests
# ═══════════════════════════════════════════════════════════════════


class TestBuyFairAvoid:
    """Test buy/fair/avoid classification (rank-gap based)."""

    def test_buy_classification_early(self):
        """Early draft: ADP 50, enhanced rank 25 -> gap 25 >= 20 -> BUY."""
        row = pd.Series({"_enhanced_rank": 25, "_adp_rank": 50})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=10) == "BUY"

    def test_avoid_overvalued(self):
        """Enhanced rank 50, ADP 20 -> gap -30 <= -20 -> AVOID."""
        row = pd.Series({"_enhanced_rank": 50, "_adp_rank": 20})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=10) == "AVOID"

    def test_fair_small_gap(self):
        """ADP and enhanced rank close -> FAIR."""
        row = pd.Series({"_enhanced_rank": 30, "_adp_rank": 35})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=10) == "FAIR"

    def test_fair_zero_adp(self):
        """ADP of 0 (invalid) -> FAIR."""
        row = pd.Series({"_enhanced_rank": 10, "_adp_rank": 0})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=10) == "FAIR"

    def test_fair_zero_everything(self):
        """Both ranks 0 -> FAIR."""
        row = pd.Series({"_enhanced_rank": 0, "_adp_rank": 0})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row) == "FAIR"

    def test_mid_draft_lower_threshold(self):
        """Mid draft (pick 150): gap 15 >= 15 -> BUY."""
        row = pd.Series({"_enhanced_rank": 135, "_adp_rank": 150})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=150) == "BUY"

    def test_late_draft_lowest_threshold(self):
        """Late draft (pick 250): gap 10 >= 10 -> BUY."""
        row = pd.Series({"_enhanced_rank": 240, "_adp_rank": 250})
        assert DraftRecommendationEngine._classify_buy_fair_avoid(row, current_pick=250) == "BUY"


# ═══════════════════════════════════════════════════════════════════
# 10. End-to-End Enhancement Pipeline Tests
# ═══════════════════════════════════════════════════════════════════


class TestEnhancePlayerPool:
    """Test the full enhance_player_pool pipeline."""

    def test_enhance_adds_columns(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.enhance_player_pool(sample_pool, draft_state)
        expected_cols = [
            "enhanced_pick_score",
            "park_factor_adj",
            "category_balance_multiplier",
            "injury_probability",
            "statcast_delta",
            "buy_fair_avoid",
            "streaming_penalty",
            "closer_hierarchy_bonus",
            "flex_bonus",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_enhance_preserves_original_columns(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.enhance_player_pool(sample_pool, draft_state)
        for col in ["player_id", "name", "team", "positions", "pick_score"]:
            assert col in result.columns

    def test_enhance_preserves_row_count(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.enhance_player_pool(sample_pool, draft_state)
        assert len(result) == len(sample_pool)

    def test_enhance_timing_recorded(self, quick_engine, sample_pool, draft_state):
        quick_engine.enhance_player_pool(sample_pool, draft_state)
        assert "total" in quick_engine.timing

    def test_enhance_quick_mode_fast(self, quick_engine, sample_pool, draft_state):
        quick_engine.enhance_player_pool(sample_pool, draft_state)
        assert quick_engine.timing["total"] < 5.0  # generous limit

    def test_enhance_no_pick_score_column(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            [{"player_id": 1, "name": "Test", "team": "LAA", "positions": "OF", "is_hitter": True, "adp": 10}]
        )
        result = quick_engine.enhance_player_pool(pool, draft_state)
        assert "enhanced_pick_score" in result.columns
        # Should use floor value
        assert result.iloc[0]["enhanced_pick_score"] >= 0

    def test_enhance_empty_pool(self, quick_engine, draft_state):
        pool = pd.DataFrame(columns=["player_id", "name", "team", "positions", "is_hitter", "adp"])
        result = quick_engine.enhance_player_pool(pool, draft_state)
        assert len(result) == 0
        assert "enhanced_pick_score" in result.columns

    def test_enhance_standard_mode(self, engine, sample_pool, draft_state):
        result = engine.enhance_player_pool(sample_pool, draft_state)
        assert "enhanced_pick_score" in result.columns
        # Standard mode enables more features — check that injury_probability
        # is non-default for fragile players (if module available)
        if INJURY_PROCESS_AVAILABLE:
            trout = result[result["player_id"] == 1].iloc[0]
            assert trout["injury_probability"] > 0.0


# ═══════════════════════════════════════════════════════════════════
# 11. Recommend Pipeline Tests
# ═══════════════════════════════════════════════════════════════════


class TestRecommend:
    """Test end-to-end recommend() method."""

    def test_recommend_returns_dataframe(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10)
        assert isinstance(result, pd.DataFrame)

    def test_recommend_respects_top_n(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10)
        assert len(result) <= 3

    def test_recommend_has_mc_columns(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10)
        if not result.empty:
            expected = ["mc_mean_sgp", "combined_score", "urgency", "p_survive"]
            for col in expected:
                assert col in result.columns, f"Missing MC column: {col}"

    def test_recommend_has_enhanced_columns(self, quick_engine, sample_pool, draft_state):
        result = quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10)
        if not result.empty:
            assert "enhanced_pick_score" in result.columns

    def test_recommend_timing(self, quick_engine, sample_pool, draft_state):
        quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10)
        assert "recommend_total" in quick_engine.timing
        assert "mc_simulation" in quick_engine.timing

    def test_recommend_with_park_factors(self, quick_engine, sample_pool, draft_state):
        pf = {"LAA": 1.05, "LAD": 1.02}
        result = quick_engine.recommend(sample_pool, draft_state, top_n=3, n_simulations=10, park_factors=pf)
        assert isinstance(result, pd.DataFrame)

    def test_recommend_empty_pool(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "team",
                "positions",
                "is_hitter",
                "adp",
                "pick_score",
                "health_score",
                "r",
                "hr",
                "rbi",
                "sb",
                "avg",
                "obp",
                "w",
                "l",
                "sv",
                "k",
                "era",
                "whip",
                "pa",
                "ab",
                "h",
                "bb",
                "hbp",
                "sf",
                "ip",
                "er",
                "bb_allowed",
                "h_allowed",
            ]
        )
        result = quick_engine.recommend(pool, draft_state, top_n=3, n_simulations=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════
# 12. Graceful Degradation Tests
# ═══════════════════════════════════════════════════════════════════


class TestGracefulDegradation:
    """Test that engine works when optional modules are unavailable."""

    def test_bayesian_unavailable(self, config, sample_pool, draft_state):
        engine = DraftRecommendationEngine(config, mode="standard")
        with patch("src.draft_engine.BAYESIAN_AVAILABLE", False):
            result = engine._apply_bayesian_blend(sample_pool.copy())
            # Should return pool unchanged
            assert len(result) == len(sample_pool)

    def test_injury_process_unavailable(self, config, sample_pool, draft_state):
        engine = DraftRecommendationEngine(config, mode="standard")
        with patch("src.draft_engine.INJURY_PROCESS_AVAILABLE", False):
            result = engine._apply_injury_probability(sample_pool.copy())
            assert (result["injury_probability"] == 0.0).all()

    def test_h2h_engine_unavailable(self, config, sample_pool, draft_state):
        engine = DraftRecommendationEngine(config, mode="standard")
        with patch("src.draft_engine.H2H_ENGINE_AVAILABLE", False):
            draft_state.make_pick(2, "Ohtani", "OF,Util")
            result = engine._apply_category_balance(sample_pool.copy(), draft_state)
            # Should still produce multipliers using simple method
            assert "category_balance_multiplier" in result.columns

    def test_full_pipeline_with_all_disabled(self, config, sample_pool, draft_state):
        engine = DraftRecommendationEngine(config, mode="quick")
        # Quick mode disables most features — should still work
        result = engine.enhance_player_pool(sample_pool, draft_state)
        assert "enhanced_pick_score" in result.columns
        assert len(result) == len(sample_pool)


# ═══════════════════════════════════════════════════════════════════
# 13. ML Correction Placeholder Tests
# ═══════════════════════════════════════════════════════════════════


class TestMLCorrection:
    """Test ML correction placeholder."""

    def test_ml_correction_zeros(self, full_engine, sample_pool):
        result = full_engine._apply_ml_correction(sample_pool.copy())
        assert (result["ml_correction"] == 0.0).all()

    def test_ml_correction_column_exists(self, full_engine, sample_pool):
        result = full_engine._apply_ml_correction(sample_pool.copy())
        assert "ml_correction" in result.columns


# ═══════════════════════════════════════════════════════════════════
# 14. Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_player_pool(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Solo",
                    "team": "LAA",
                    "positions": "OF",
                    "is_hitter": True,
                    "is_injured": False,
                    "pa": 500,
                    "ab": 450,
                    "h": 120,
                    "r": 70,
                    "hr": 20,
                    "rbi": 60,
                    "sb": 10,
                    "avg": 0.267,
                    "obp": 0.340,
                    "bb": 50,
                    "hbp": 3,
                    "sf": 3,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 20,
                    "pick_score": 5.0,
                }
            ]
        )
        result = quick_engine.enhance_player_pool(pool, draft_state)
        assert len(result) == 1
        assert result.iloc[0]["enhanced_pick_score"] > 0

    def test_all_pitchers_pool(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            [
                {
                    "player_id": i,
                    "name": f"Pitcher {i}",
                    "team": "NYY",
                    "positions": "SP",
                    "is_hitter": False,
                    "is_injured": False,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "ip": 180,
                    "w": 12,
                    "l": 7,
                    "sv": 0,
                    "k": 190,
                    "era": 3.50,
                    "whip": 1.15,
                    "er": 70,
                    "bb_allowed": 50,
                    "h_allowed": 157,
                    "adp": 20 + i * 10,
                    "pick_score": 6.0 - i * 0.5,
                }
                for i in range(5)
            ]
        )
        result = quick_engine.enhance_player_pool(pool, draft_state)
        assert len(result) == 5

    def test_missing_optional_columns(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Minimal",
                    "team": "NYY",
                    "positions": "OF",
                    "is_hitter": True,
                    "adp": 50,
                    "pick_score": 4.0,
                }
            ]
        )
        result = quick_engine.enhance_player_pool(pool, draft_state)
        assert "enhanced_pick_score" in result.columns

    def test_none_values_in_stats(self, quick_engine, draft_state):
        pool = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Nullish",
                    "team": "NYY",
                    "positions": "OF",
                    "is_hitter": True,
                    "adp": 50,
                    "pick_score": None,
                    "health_score": None,
                    "age": None,
                    "r": None,
                    "hr": None,
                    "rbi": None,
                    "sb": None,
                    "avg": None,
                    "obp": None,
                }
            ]
        )
        result = quick_engine.enhance_player_pool(pool, draft_state)
        # Should not crash, should use floor values
        assert result.iloc[0]["enhanced_pick_score"] >= 0

    def test_draft_in_progress(self, quick_engine, sample_pool, draft_state):
        # Draft some players first
        draft_state.make_pick(1, "Mike Trout", "OF")
        draft_state.make_pick(2, "Shohei Ohtani", "OF,Util")
        result = quick_engine.enhance_player_pool(sample_pool, draft_state)
        assert len(result) == len(sample_pool)

    def test_inverse_category_weights(self):
        """ERA/WHIP are inverse — user ABOVE median is BAD."""
        my = {
            "R": 100,
            "HR": 30,
            "RBI": 80,
            "SB": 10,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 5,
            "SV": 8,
            "K": 140,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        med = {
            "R": 80,
            "HR": 25,
            "RBI": 70,
            "SB": 12,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 8,
            "SV": 8,
            "K": 140,
            "ERA": 3.80,
            "WHIP": 1.20,
        }
        weights = DraftRecommendationEngine._simple_category_weights(my, med)
        # ERA 5.00 > median 3.80 → inverse → user is worse → should boost
        assert weights["ERA"] > weights["R"]  # ERA needs more help than R (R is above median)

    def test_all_zeroes_totals(self):
        my = {cat: 0 for cat in ALL_CATEGORIES_UPPER}
        med = {cat: 0 for cat in ALL_CATEGORIES_UPPER}
        weights = DraftRecommendationEngine._simple_category_weights(my, med)
        # Should still produce valid weights
        assert len(weights) == 12
        for w in weights.values():
            assert np.isfinite(w)
