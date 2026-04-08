"""Tests for roadmap-session-3 features: T7, T8, T12, E3, J5, E7, E10, D4, E1, Q3, Q4, D7."""

import numpy as np
import pandas as pd
import pytest

# ── T7/E3: Umpire tendencies and adjustments ──────────────────────


class TestUmpireAdjustment:
    """E3: Umpire strike zone adjustment tests."""

    def test_apply_umpire_adjustment_neutral(self):
        from src.optimizer.matchup_adjustments import apply_umpire_adjustment

        k, bb, runs = apply_umpire_adjustment(8.0, 3.0, 4.5, {})
        assert k == 8.0
        assert bb == 3.0
        assert runs == 4.5

    def test_apply_umpire_adjustment_high_k(self):
        from src.optimizer.matchup_adjustments import apply_umpire_adjustment

        ump_data = {"k_pct_delta": 0.02, "bb_pct_delta": -0.01, "run_env_delta": -0.3}
        k, bb, runs = apply_umpire_adjustment(8.0, 3.0, 4.5, ump_data)
        assert k > 8.0  # Higher K umpire → more K
        assert bb < 3.0  # Lower BB umpire → fewer BB
        assert runs < 4.5  # Lower run env → fewer runs

    def test_apply_umpire_adjustment_clamped(self):
        from src.optimizer.matchup_adjustments import apply_umpire_adjustment

        # Extreme values should be clamped to ±15%
        ump_data = {"k_pct_delta": 0.10, "bb_pct_delta": 0.10, "run_env_delta": 5.0}
        k, bb, runs = apply_umpire_adjustment(8.0, 3.0, 4.5, ump_data)
        assert k <= 8.0 * 1.15 + 0.01
        assert bb <= 3.0 * 1.15 + 0.01


# ── T8/J5: Catcher framing adjustment ─────────────────────────────


class TestCatcherFramingAdjustment:
    """J5: Catcher framing pitcher adjustment tests."""

    def test_elite_framer_lowers_era(self):
        from src.optimizer.matchup_adjustments import catcher_framing_pitcher_adjustment

        era, k9 = catcher_framing_pitcher_adjustment(4.00, 9.0, 15.0)
        assert era < 4.00  # Elite framing → lower ERA
        assert k9 > 9.0  # Elite framing → more K/9

    def test_poor_framer_raises_era(self):
        from src.optimizer.matchup_adjustments import catcher_framing_pitcher_adjustment

        era, k9 = catcher_framing_pitcher_adjustment(4.00, 9.0, -15.0)
        assert era > 4.00  # Poor framing → higher ERA
        assert k9 < 9.0  # Poor framing → fewer K/9

    def test_neutral_framer_no_change(self):
        from src.optimizer.matchup_adjustments import catcher_framing_pitcher_adjustment

        era, k9 = catcher_framing_pitcher_adjustment(4.00, 9.0, 0.0)
        assert era == 4.00
        assert k9 == 9.0

    def test_era_capped(self):
        from src.optimizer.matchup_adjustments import catcher_framing_pitcher_adjustment

        # Very extreme framing should be capped at ±0.40 ERA
        era, k9 = catcher_framing_pitcher_adjustment(4.00, 9.0, 100.0)
        assert era >= 4.00 - 0.40 - 0.01
        assert era <= 4.00


# ── T12/E7: Pitcher-batter matchup history ─────────────────────────


class TestPvBMatchupAdjustment:
    """E7: PvB regression-to-mean tests."""

    def test_pvb_zero_pa_returns_generic(self):
        from src.optimizer.matchup_adjustments import pvb_matchup_adjustment

        result = pvb_matchup_adjustment(0.320, 0.400, 0)
        assert result == 0.320

    def test_pvb_full_weight_at_60_pa(self):
        from src.optimizer.matchup_adjustments import pvb_matchup_adjustment

        result = pvb_matchup_adjustment(0.320, 0.400, 60)
        assert abs(result - 0.400) < 0.001

    def test_pvb_half_weight_at_30_pa(self):
        from src.optimizer.matchup_adjustments import pvb_matchup_adjustment

        result = pvb_matchup_adjustment(0.320, 0.400, 30)
        expected = 0.5 * 0.400 + 0.5 * 0.320  # 0.360
        assert abs(result - expected) < 0.001

    def test_pvb_small_sample_mostly_generic(self):
        from src.optimizer.matchup_adjustments import pvb_matchup_adjustment

        result = pvb_matchup_adjustment(0.320, 0.500, 6)
        # 10% weight on PvB → close to generic
        assert result < 0.350


# ── E1: xBABIP model ──────────────────────────────────────────────


class TestXBABIP:
    """E1: xBABIP model tests."""

    def test_compute_xbabip_average_player(self):
        from src.ml_ensemble import compute_xbabip

        player = {
            "ld_pct": 0.22,
            "hitter_gb_pct": 0.43,
            "hitter_fb_pct": 0.35,
            "sprint_speed": 27.0,
            "barrel_pct": 0.07,
        }
        result = compute_xbabip(player)
        assert result is not None
        assert abs(result - 0.300) < 0.01  # Average player → ~.300 xBABIP

    def test_compute_xbabip_high_ld(self):
        from src.ml_ensemble import compute_xbabip

        player = {
            "ld_pct": 0.30,  # Very high line drive rate
            "hitter_gb_pct": 0.35,
            "hitter_fb_pct": 0.35,
            "sprint_speed": 27.0,
            "barrel_pct": 0.07,
        }
        result = compute_xbabip(player)
        assert result is not None
        assert result > 0.320  # High LD% → higher xBABIP

    def test_compute_xbabip_insufficient_data(self):
        from src.ml_ensemble import compute_xbabip

        # Only 1 feature → returns None
        player = {"ld_pct": 0.22}
        result = compute_xbabip(player)
        assert result is None

    def test_xbabip_regression_flag_buy_low(self):
        from src.ml_ensemble import compute_xbabip_regression_flag

        player = {
            "pa": 100,
            "babip": 0.250,  # Below expected
            "ld_pct": 0.25,
            "hitter_gb_pct": 0.40,
            "hitter_fb_pct": 0.35,
            "sprint_speed": 28.0,
            "barrel_pct": 0.10,
        }
        result = compute_xbabip_regression_flag(player)
        assert result["flag"] == "BUY_LOW"
        assert result["gap"] > 0.03

    def test_xbabip_regression_flag_insufficient_pa(self):
        from src.ml_ensemble import compute_xbabip_regression_flag

        player = {"pa": 20, "babip": 0.250, "ld_pct": 0.25, "sprint_speed": 28.0}
        result = compute_xbabip_regression_flag(player, min_pa=50)
        assert result["flag"] == ""

    def test_xbabip_clamped(self):
        from src.ml_ensemble import compute_xbabip

        # Extreme values should be clamped
        player = {
            "ld_pct": 0.50,  # Impossibly high
            "hitter_gb_pct": 0.10,
            "hitter_fb_pct": 0.40,
            "sprint_speed": 32.0,
            "barrel_pct": 0.30,
        }
        result = compute_xbabip(player)
        assert result is not None
        assert result <= 0.450


# ── D4: Opponent behavior learning ─────────────────────────────────


class TestOpponentBehaviorLearning:
    """D4: Opponent behavior learning tests."""

    def test_learn_with_no_data(self):
        from src.opponent_trade_analysis import learn_opponent_trade_behavior

        result = learn_opponent_trade_behavior("Team A")
        assert result["acceptance_modifier"] == 1.0
        assert result["sample_size"] == 0
        assert result["trade_frequency"] == 0.0

    def test_learn_with_transactions(self):
        from src.opponent_trade_analysis import learn_opponent_trade_behavior

        txns = pd.DataFrame(
            {
                "type": ["trade", "add", "trade", "drop", "trade"],
                "team_from": ["Team A", "Team B", "Team A", "Team C", "Team A"],
                "team_to": ["Team B", "", "Team C", "", "Team D"],
                "timestamp": pd.date_range("2026-04-01", periods=5, freq="D"),
            }
        )
        result = learn_opponent_trade_behavior("Team A", txns)
        assert result["trade_frequency"] > 0  # Team A made 3 trades


# ── Q3: Per-player ADP sigma ──────────────────────────────────────


class TestPlayerADPSigma:
    """Q3: Per-player ADP standard deviation tests."""

    def test_compute_sigma_multiple_sources(self):
        from src.simulation import compute_player_adp_sigma

        pool = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "adp": [10.0, 50.0, 200.0],
                "nfbc_adp": [15.0, 45.0, 210.0],
                "consensus_rank": [12.0, 48.0, 195.0],
            }
        )
        sigma = compute_player_adp_sigma(pool)
        assert len(sigma) == 3
        assert all(sigma > 0)
        # Early picks should have tighter distributions
        assert sigma.iloc[0] < sigma.iloc[2]

    def test_compute_sigma_injury_adjustment(self):
        from src.simulation import compute_player_adp_sigma

        pool = pd.DataFrame(
            {
                "player_id": [1, 2],
                "adp": [50.0, 50.0],
                "nfbc_adp": [55.0, 55.0],
                "health_score": [0.50, 0.95],  # Injured vs healthy
            }
        )
        sigma = compute_player_adp_sigma(pool)
        # Injured player should have wider distribution
        assert sigma.iloc[0] > sigma.iloc[1]

    def test_compute_sigma_single_source(self):
        from src.simulation import compute_player_adp_sigma

        pool = pd.DataFrame(
            {
                "player_id": [1],
                "adp": [100.0],
            }
        )
        sigma = compute_player_adp_sigma(pool)
        assert sigma.iloc[0] >= 2.0  # Minimum floor


# ── Q4: Draft standings impact ─────────────────────────────────────


class TestDraftStandingsImpact:
    """Q4: Draft value standings impact tests."""

    def test_weak_category_boost(self):
        from src.simulation import evaluate_pick_standings_impact
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        candidate = pd.Series({"hr": 30, "rbi": 80, "r": 70, "sb": 5, "avg": 0.260, "obp": 0.330})
        roster_sgp = {"hr": -1.0, "rbi": -1.0, "r": 0.0, "sb": 0.0, "avg": 0.0, "obp": 0.0}
        league_avg = {"hr": 0.0, "rbi": 0.0, "r": 0.0, "sb": 0.0, "avg": 0.0, "obp": 0.0}

        result = evaluate_pick_standings_impact(candidate, roster_sgp, league_avg, config)
        assert result["weak_cats_helped"] > 0
        assert result["standings_impact"] != 0

    def test_strong_category_diminishing(self):
        from src.simulation import evaluate_pick_standings_impact
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        candidate = pd.Series({"hr": 30, "rbi": 80, "r": 70, "sb": 5, "avg": 0.260, "obp": 0.330})
        # Already strong in all categories
        roster_sgp = {"hr": 2.0, "rbi": 2.0, "r": 2.0, "sb": 2.0, "avg": 2.0, "obp": 2.0}
        league_avg = {"hr": 0.0, "rbi": 0.0, "r": 0.0, "sb": 0.0, "avg": 0.0, "obp": 0.0}

        result = evaluate_pick_standings_impact(candidate, roster_sgp, league_avg, config)
        assert result["strong_cats_padded"] > 0


# ── D7: Lineup RL ──────────────────────────────────────────────────


class TestLineupRL:
    """D7: Category-aware lineup RL tests."""

    def test_bandit_creation(self):
        from src.lineup_rl import LineupContextualBandit

        bandit = LineupContextualBandit()
        assert bandit.categories is not None
        assert len(bandit.categories) == 12
        assert not bandit.is_ready
        assert bandit.weeks_of_data == 0

    def test_bandit_recommend_weights_no_data(self):
        from src.lineup_rl import LineupContextualBandit

        bandit = LineupContextualBandit()
        base = {"r": 1.0, "hr": 1.5, "rbi": 1.0}
        weights = bandit.recommend_weights({"wins": 3, "losses": 2, "ties": 1}, base)
        # With no data, should return base weights
        assert weights == base

    def test_bandit_record_and_learn(self):
        from src.lineup_rl import LineupContextualBandit

        bandit = LineupContextualBandit(categories=["r", "hr", "rbi"])
        # Record 10 weeks of data
        for i in range(10):
            state = {"wins": 3, "losses": 2, "ties": 1}
            weights = {"r": 1.5, "hr": 1.2, "rbi": 0.8}
            results = {"r": "W", "hr": "W", "rbi": "L"}
            bandit.record_outcome(state, weights, results)

        assert bandit.weeks_of_data == 10
        assert bandit.is_ready

        # Now recommendations should use learned data
        weights = bandit.recommend_weights({"wins": 3, "losses": 2, "ties": 1})
        assert all(w > 0 for w in weights.values())

    def test_context_key_categories(self):
        from src.lineup_rl import LineupContextualBandit

        bandit = LineupContextualBandit()
        assert bandit._context_key({"wins": 8, "losses": 2, "ties": 2}) == "winning_big"
        assert bandit._context_key({"wins": 4, "losses": 3, "ties": 5}) == "winning"
        assert bandit._context_key({"wins": 3, "losses": 8, "ties": 1}) == "losing_big"
        assert bandit._context_key({"wins": 3, "losses": 4, "ties": 5}) == "losing"
        assert bandit._context_key({"wins": 4, "losses": 4, "ties": 4}) == "tied"

    def test_get_category_win_rates(self):
        from src.lineup_rl import LineupContextualBandit

        bandit = LineupContextualBandit(categories=["r", "hr"])
        for _ in range(5):
            bandit.record_outcome(
                {"wins": 3, "losses": 3, "ties": 0},
                {"r": 1.5, "hr": 1.0},
                {"r": "W", "hr": "L"},
            )

        rates = bandit.get_category_win_rates()
        assert rates["r"] == 1.0  # Always won R
        assert rates["hr"] == 0.0  # Always lost HR

    def test_singleton_accessor(self):
        from src.lineup_rl import get_lineup_bandit

        b1 = get_lineup_bandit()
        b2 = get_lineup_bandit()
        assert b1 is b2


# ── Database schema tests ──────────────────────────────────────────


class TestNewDatabaseTables:
    """Verify new tables are created properly."""

    def test_new_tables_in_valid_names(self):
        from src.database import _VALID_TABLE_NAMES

        assert "umpire_tendencies" in _VALID_TABLE_NAMES
        assert "catcher_framing" in _VALID_TABLE_NAMES
        assert "pvb_splits" in _VALID_TABLE_NAMES
        assert "opponent_trade_history" in _VALID_TABLE_NAMES

    def test_tables_created_on_init(self):
        from src.database import get_connection, init_db

        init_db()
        conn = get_connection()
        try:
            for table in ["umpire_tendencies", "catcher_framing", "pvb_splits", "opponent_trade_history"]:
                result = conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ).fetchone()
                assert result is not None, f"Table {table} should exist"
        finally:
            conn.close()


# ── Catcher framing data accessor test ─────────────────────────────


class TestCatcherFramingData:
    """E10: Catcher framing data accessor tests."""

    def test_get_catcher_framing_data_empty(self):
        from src.optimizer.matchup_adjustments import get_catcher_framing_data

        # Should return empty dict when no data (graceful fallback)
        result = get_catcher_framing_data()
        assert isinstance(result, dict)


# ── PvB data accessor test ─────────────────────────────────────────


class TestPvBMatchupData:
    """E7: PvB matchup data accessor tests."""

    def test_get_pvb_matchup_data_empty(self):
        from src.optimizer.matchup_adjustments import get_pvb_matchup_data

        result = get_pvb_matchup_data()
        assert isinstance(result, dict)
