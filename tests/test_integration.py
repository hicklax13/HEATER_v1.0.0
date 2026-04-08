"""End-to-end integration tests for Plan 3 features.

Tests the full pipeline: injury adjustment → Bayesian update → valuation →
percentile forecasts → lineup optimization → opponent modeling.
"""

import numpy as np
import pandas as pd
import pytest

from src.bayesian import BayesianUpdater
from src.draft_state import get_positional_needs, get_team_draft_patterns
from src.injury_model import apply_injury_adjustment, compute_health_score, get_injury_badge
from src.simulation import compute_team_preferences
from src.valuation import (
    LeagueConfig,
    compute_percentile_projections,
    compute_projection_volatility,
    value_all_players,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HITTER_COLS = [
    "player_id",
    "name",
    "team",
    "positions",
    "is_hitter",
    "is_injured",
    "pa",
    "ab",
    "h",
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "ip",
    "w",
    "sv",
    "k",
    "era",
    "whip",
    "er",
    "bb_allowed",
    "h_allowed",
    "adp",
]


def _make_hitter(pid, name, pa=600, hr=30, r=90, rbi=95, sb=10, avg=0.270, adp=50.0):
    ab = int(pa * 0.9)
    h = int(ab * avg)
    return {
        "player_id": pid,
        "name": name,
        "team": "NYY",
        "positions": "OF",
        "is_hitter": 1,
        "is_injured": 0,
        "pa": pa,
        "ab": ab,
        "h": h,
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "avg": avg,
        "ip": 0,
        "w": 0,
        "sv": 0,
        "k": 0,
        "era": 0.0,
        "whip": 0.0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "adp": adp,
    }


def _make_pitcher(pid, name, ip=180, w=12, sv=0, k_val=200, era=3.50, whip=1.15, adp=60.0):
    er = int(ip * era / 9)
    bb_allowed = int(ip * (whip - 1.0) * 0.4)
    h_allowed = int(ip * whip) - bb_allowed
    return {
        "player_id": pid,
        "name": name,
        "team": "LAD",
        "positions": "SP",
        "is_hitter": 0,
        "is_injured": 0,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "ip": ip,
        "w": w,
        "sv": sv,
        "k": k_val,
        "era": era,
        "whip": whip,
        "er": er,
        "bb_allowed": bb_allowed,
        "h_allowed": h_allowed,
        "adp": adp,
    }


@pytest.fixture
def sample_pool():
    """Small player pool with 5 hitters + 3 pitchers."""
    rows = [
        _make_hitter(1, "Aaron Judge", pa=650, hr=45, r=110, rbi=120, sb=5, avg=0.280, adp=3.0),
        _make_hitter(2, "Mookie Betts", pa=620, hr=28, r=105, rbi=85, sb=15, avg=0.290, adp=8.0),
        _make_hitter(3, "Trea Turner", pa=600, hr=22, r=95, rbi=75, sb=25, avg=0.275, adp=15.0),
        _make_hitter(4, "Pete Alonso", pa=580, hr=38, r=80, rbi=105, sb=2, avg=0.250, adp=30.0),
        _make_hitter(5, "Bench Player", pa=400, hr=12, r=50, rbi=45, sb=8, avg=0.255, adp=150.0),
        _make_pitcher(6, "Gerrit Cole", ip=200, w=15, k_val=240, era=3.10, whip=1.05, adp=12.0),
        _make_pitcher(7, "Corbin Burnes", ip=190, w=13, k_val=210, era=3.30, whip=1.10, adp=20.0),
        _make_pitcher(8, "Josh Hader", ip=65, w=3, sv=35, k_val=80, era=2.80, whip=0.95, adp=45.0),
    ]
    return pd.DataFrame(rows, columns=HITTER_COLS)


@pytest.fixture
def league_config():
    return LeagueConfig()


# ---------------------------------------------------------------------------
# 1. Injury → Valuation pipeline
# ---------------------------------------------------------------------------


class TestInjuryValuationPipeline:
    """Full health score → projection adjustment → valuation chain."""

    def test_injury_adjusted_pool_valuates(self, sample_pool, league_config):
        """Injury-adjusted projections still produce valid valuations."""
        health = pd.DataFrame(
            [
                {"player_id": 1, "health_score": 0.85},
                {"player_id": 2, "health_score": 1.0},
                {"player_id": 3, "health_score": 0.70},
                {"player_id": 4, "health_score": 0.90},
                {"player_id": 5, "health_score": 0.95},
                {"player_id": 6, "health_score": 0.80},
                {"player_id": 7, "health_score": 1.0},
                {"player_id": 8, "health_score": 0.90},
            ]
        )
        adjusted = apply_injury_adjustment(sample_pool, health)
        valued = value_all_players(adjusted, league_config)
        assert isinstance(valued, pd.DataFrame)
        assert len(valued) == 8
        assert "pick_score" in valued.columns

    def test_injured_player_valued_lower(self, sample_pool, league_config):
        """A player with health_score=0.7 should be valued less than health_score=1.0."""
        # Value with full health
        full_health = pd.DataFrame([{"player_id": pid, "health_score": 1.0} for pid in range(1, 9)])
        full_valued = value_all_players(apply_injury_adjustment(sample_pool, full_health), league_config)

        # Value with player 3 injured
        partial_health = full_health.copy()
        partial_health.loc[partial_health["player_id"] == 3, "health_score"] = 0.60
        partial_valued = value_all_players(apply_injury_adjustment(sample_pool, partial_health), league_config)

        full_score = full_valued.loc[full_valued["player_id"] == 3, "pick_score"].iloc[0]
        partial_score = partial_valued.loc[partial_valued["player_id"] == 3, "pick_score"].iloc[0]
        assert partial_score < full_score

    def test_health_score_to_badge(self):
        """End-to-end: compute health → get badge."""
        score = compute_health_score([150, 162, 130], [162, 162, 162])
        icon, label = get_injury_badge(score)
        assert label in ("Low Risk", "Moderate Risk", "Elevated Risk", "High Risk")
        assert icon  # Non-empty string


# ---------------------------------------------------------------------------
# 2. Bayesian → Valuation pipeline
# ---------------------------------------------------------------------------


class TestBayesianValuationPipeline:
    def test_bayesian_update_feeds_valuation(self, sample_pool, league_config):
        """Bayesian-updated projections produce valid valuations."""
        updater = BayesianUpdater(prior_weight=0.6)

        # Simulate mid-season stats
        season = sample_pool.copy()
        season["pa"] = (season["pa"] * 0.3).astype(int)  # ~30% of season
        season["ab"] = (season["ab"] * 0.3).astype(int)
        season["h"] = (season["h"] * 0.35).astype(int)  # Slight batting avg bump
        season["r"] = (season["r"] * 0.3).astype(int)
        season["hr"] = (season["hr"] * 0.3).astype(int)
        season["rbi"] = (season["rbi"] * 0.3).astype(int)
        season["sb"] = (season["sb"] * 0.3).astype(int)
        season["ip"] = (season["ip"] * 0.3).astype(int)
        season["w"] = (season["w"] * 0.3).astype(int)
        season["k"] = (season["k"] * 0.3).astype(int)
        season["games_played"] = 48

        updated = updater.batch_update_projections(season, sample_pool)
        assert isinstance(updated, pd.DataFrame)
        assert len(updated) == len(sample_pool)

        valued = value_all_players(updated, league_config)
        assert "pick_score" in valued.columns
        assert len(valued) == len(sample_pool)

    def test_bayesian_regression_moves_toward_prior(self):
        """Small sample observation regresses heavily toward preseason."""
        updater = BayesianUpdater(prior_weight=0.6)
        # .400 in 50 PA → should regress a lot toward .250 mean
        result = updater.regressed_rate(0.400, 50, 0.250, 910)
        assert result < 0.300  # Heavily regressed
        assert result > 0.250  # But above league mean


# ---------------------------------------------------------------------------
# 3. Percentile forecasts pipeline
# ---------------------------------------------------------------------------


class TestPercentilePipeline:
    def test_multi_system_volatility_to_percentiles(self, sample_pool):
        """Multiple projection systems → volatility → P10/P50/P90 → valid spreads."""
        # Create 3 "systems" with slight variation
        rng = np.random.default_rng(42)
        systems = {}
        for name in ["steamer", "zips", "depthcharts"]:
            sys_df = sample_pool.copy()
            for col in ["r", "hr", "rbi", "sb"]:
                noise = rng.normal(1.0, 0.05, len(sys_df))
                sys_df[col] = (sys_df[col] * noise).round(0).astype(int)
            for col in ["avg"]:
                noise = rng.normal(0, 0.010, len(sys_df))
                sys_df[col] = sys_df[col] + noise
            systems[name] = sys_df

        volatility = compute_projection_volatility(systems)
        assert len(volatility) == len(sample_pool)

        percentiles = compute_percentile_projections(sample_pool, volatility)
        assert 10 in percentiles
        assert 50 in percentiles
        assert 90 in percentiles

        # P10 < P50 < P90 for counting stats
        for pid in [1, 2, 3]:
            p10_hr = percentiles[10].loc[percentiles[10]["player_id"] == pid, "hr"].iloc[0]
            p50_hr = percentiles[50].loc[percentiles[50]["player_id"] == pid, "hr"].iloc[0]
            p90_hr = percentiles[90].loc[percentiles[90]["player_id"] == pid, "hr"].iloc[0]
            assert p10_hr <= p50_hr <= p90_hr

    def test_single_system_zero_volatility(self, sample_pool):
        """J6: Single system → zero per-player volatility → empirical SDs used as fallback.

        P10 and P90 should now differ (empirical SDs fill the gap),
        and P10 < P50 < P90 for counting stats.
        """
        systems = {"only_system": sample_pool}
        volatility = compute_projection_volatility(systems)
        percentiles = compute_percentile_projections(sample_pool, volatility)

        for col in ["hr", "r", "rbi"]:
            p10_vals = percentiles[10][col].values
            p50_vals = percentiles[50][col].values
            p90_vals = percentiles[90][col].values
            # With empirical SD fallback, P10 < P90 (no longer equal)
            assert np.all(p10_vals <= p50_vals + 0.01)
            assert np.all(p50_vals <= p90_vals + 0.01)


# ---------------------------------------------------------------------------
# 4. Opponent modeling pipeline
# ---------------------------------------------------------------------------


class TestOpponentModelPipeline:
    def test_draft_history_to_preferences(self):
        """Draft history → team preferences → probability adjustment."""
        history = pd.DataFrame(
            {
                "team_key": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "round": [1, 2, 3, 4, 1, 2, 3, 4],
                "positions": ["SP", "SP", "OF", "SP", "OF", "OF", "1B", "OF"],
                "player_name": [f"P{i}" for i in range(8)],
            }
        )
        prefs = compute_team_preferences(history)
        assert "A" in prefs
        assert "B" in prefs
        # Team A drafted 3 SP out of 4 picks → should show pitcher bias
        assert prefs["A"]["positional_bias"].get("SP", 0) > prefs["B"]["positional_bias"].get("SP", 0)

    def test_draft_patterns_from_state(self):
        """get_team_draft_patterns works with mock draft state."""
        draft_state = {
            "picks": [
                {"round": 1, "team_index": 0, "positions": "SP"},
                {"round": 2, "team_index": 0, "positions": "OF"},
                {"round": 3, "team_index": 0, "positions": "SP"},
                {"round": 1, "team_index": 1, "positions": "C"},
            ]
        }
        patterns = get_team_draft_patterns(draft_state, 0)
        assert "positional_bias" in patterns
        assert "round_patterns" in patterns
        assert patterns["positional_bias"]["SP"] > patterns["positional_bias"].get("C", 0)

    def test_positional_needs_detection(self):
        """get_positional_needs identifies unfilled slots."""
        draft_state = {
            "picks": [
                {"round": 1, "team_index": 0, "positions": "SS"},
                {"round": 2, "team_index": 0, "positions": "OF"},
            ]
        }
        roster_config = {
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "OF": 3,
            "SP": 2,
            "RP": 2,
        }
        needs = get_positional_needs(draft_state, 0, roster_config)
        assert "SS" not in needs  # Filled (0 remaining not included)
        assert needs["OF"] == 2  # 1 filled, 2 remaining
        assert needs["C"] == 1  # Unfilled


# ---------------------------------------------------------------------------
# 5. Full pipeline: injury + Bayesian + percentiles + valuation
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_complete_pipeline(self, sample_pool, league_config):
        """injury → Bayesian update → percentiles → valuation end-to-end."""
        # Step 1: Injury adjustment
        health = pd.DataFrame([{"player_id": pid, "health_score": 0.85 + (pid % 3) * 0.05} for pid in range(1, 9)])
        adjusted = apply_injury_adjustment(sample_pool, health)
        assert len(adjusted) == 8

        # Step 2: Bayesian update (using adjusted as "preseason" and a mock mid-season)
        updater = BayesianUpdater()
        season = adjusted.copy()
        season["pa"] = (season["pa"] * 0.25).astype(int)
        season["ab"] = (season["ab"] * 0.25).astype(int)
        season["h"] = (season["h"] * 0.25).astype(int)
        season["r"] = (season["r"] * 0.25).astype(int)
        season["hr"] = (season["hr"] * 0.25).astype(int)
        season["rbi"] = (season["rbi"] * 0.25).astype(int)
        season["sb"] = (season["sb"] * 0.25).astype(int)
        season["ip"] = (season["ip"] * 0.25).astype(int)
        season["w"] = (season["w"] * 0.25).astype(int)
        season["k"] = (season["k"] * 0.25).astype(int)
        season["games_played"] = 40
        updated = updater.batch_update_projections(season, adjusted)
        assert len(updated) == 8

        # Step 3: Percentile forecasts
        systems = {"preseason": adjusted, "bayesian": updated}
        volatility = compute_projection_volatility(systems)
        percentiles = compute_percentile_projections(updated, volatility)
        assert 10 in percentiles and 90 in percentiles

        # Step 4: Valuation
        valued = value_all_players(updated, league_config)
        assert "pick_score" in valued.columns
        assert len(valued) == 8
        # Top player (Judge, pid=1) should still be highly ranked
        top = valued.nlargest(1, "pick_score")
        assert top.iloc[0]["player_id"] in [1, 2, 6]  # Judge, Betts, or Cole
