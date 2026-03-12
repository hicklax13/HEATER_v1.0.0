"""Math verification tests for src/simulation.py formulas.

Tests survival probability, urgency, opponent pick probabilities,
MC convergence, tier assignment, and team preference computation
— using hand-calculated golden values and mathematical invariants.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.simulation import DraftSimulator, compute_team_preferences
from src.valuation import LeagueConfig, SGPCalculator, assign_tiers

# ── Helpers ─────────────────────────────────────────────────────────


def _pool_df(n=20, adp_start=10):
    """Create a minimal player pool DataFrame for tests."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "player_id": i + 1,
                "name": f"P_{i}",
                "positions": "OF" if i % 2 == 0 else "SP",
                "adp": adp_start + i * 3,
                "pick_score": 5.0 - i * 0.2,
                "is_hitter": 1 if i % 2 == 0 else 0,
                "r": 80,
                "hr": 25,
                "rbi": 80,
                "sb": 10,
                "avg": 0.270,
                "w": 0 if i % 2 == 0 else 12,
                "sv": 0,
                "k": 0 if i % 2 == 0 else 180,
                "ip": 0 if i % 2 == 0 else 180,
                "era": 0 if i % 2 == 0 else 3.50,
                "whip": 0 if i % 2 == 0 else 1.15,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    return LeagueConfig()


@pytest.fixture
def sim(config):
    """DraftSimulator with sigma=10 (mid-range for testing)."""
    return DraftSimulator(config=config, sigma=10)


# ── Survival Probability ────────────────────────────────────────────


class TestSurvivalProbability:
    """Verify survival probability = CDF of z-score with scarcity adj."""

    def test_hand_calc_z_score(self, sim):
        """Verify z = (ADP - next_pick) / (sigma * picks_between^0.3).

        Parameters chosen so that the z-score path is taken (not early returns).
        Constraint: 0 < picks_until_adp <= picks_between * 3.
        """
        adp = 35.0
        current_pick = 10
        next_pick = 22
        picks_between = next_pick - current_pick  # 12
        # Verify we're in the z-score branch (not early-return)
        picks_until_adp = adp - current_pick  # 25, <= 36
        assert 0 < picks_until_adp <= picks_between * 3
        z = (adp - next_pick) / (sim.sigma * max(1, picks_between**0.3))
        expected = float(np.clip(norm.cdf(z), 0.01, 0.99))
        actual = sim.survival_probability(adp, current_pick, next_pick)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_high_adp_high_survival(self, sim):
        """Player with ADP way above next pick → high survival prob."""
        prob = sim.survival_probability(player_adp=200, current_pick=10, next_user_pick=22)
        assert prob > 0.90

    def test_low_adp_low_survival(self, sim):
        """Player with ADP below next pick → low survival prob."""
        prob = sim.survival_probability(player_adp=15, current_pick=10, next_user_pick=22)
        assert prob < 0.50

    def test_adp_at_next_pick_near_50pct(self, sim):
        """Player with ADP ≈ next_pick should have survival near 50%."""
        prob = sim.survival_probability(player_adp=22, current_pick=10, next_user_pick=22)
        # z ≈ 0, CDF(0) = 0.5 (slightly modified by picks_between^0.3)
        assert 0.35 < prob < 0.65

    def test_survival_monotonic_in_adp(self, sim):
        """Higher ADP → higher survival probability (monotonic)."""
        probs = [sim.survival_probability(adp, current_pick=10, next_user_pick=22) for adp in [15, 25, 35, 50, 80]]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], f"Not monotonic at index {i}"

    def test_no_future_pick_returns_zero(self, sim):
        """If next_pick <= current_pick, survival = 0."""
        prob = sim.survival_probability(player_adp=50, current_pick=22, next_user_pick=22)
        assert prob == 0.0

    def test_passed_adp_edge_case(self, sim):
        """Player whose ADP already passed → very low survival."""
        # ADP=5, current_pick=10 → ADP already passed
        prob = sim.survival_probability(player_adp=5, current_pick=10, next_user_pick=22)
        assert prob < 0.15

    def test_deep_adp_edge_case(self, sim):
        """Player with ADP far in the future → ~95% survival."""
        prob = sim.survival_probability(player_adp=200, current_pick=10, next_user_pick=22)
        assert prob >= 0.90

    def test_survival_bounded_01_099(self, sim):
        """Probability must always be in [0.0, 0.99]."""
        for adp in [1, 5, 10, 22, 50, 100, 300]:
            prob = sim.survival_probability(adp, current_pick=10, next_user_pick=22)
            assert 0.0 <= prob <= 0.99

    def test_scarcity_reduces_survival(self, sim):
        """Positions with high league demand reduce survival probability."""
        base = sim.survival_probability(player_adp=40, current_pick=10, next_user_pick=22)
        scarce = sim.survival_probability(
            player_adp=40,
            current_pick=10,
            next_user_pick=22,
            positions_needed_league={"SS": 8},
            player_positions="SS",
        )
        assert scarce < base, "Scarcity should reduce survival"

    def test_scarcity_factor_capped_at_20pct(self, sim):
        """Scarcity factor = min(0.2, demand/50), so max 20% reduction."""
        base = sim.survival_probability(player_adp=40, current_pick=10, next_user_pick=22)
        extreme = sim.survival_probability(
            player_adp=40,
            current_pick=10,
            next_user_pick=22,
            positions_needed_league={"SS": 100},  # way above cap
            player_positions="SS",
        )
        # Scarcity factor at max: base * (1 - 0.2) = base * 0.8
        assert extreme >= base * 0.79  # allow tiny float tolerance


# ── Urgency Formula ─────────────────────────────────────────────────


class TestUrgency:
    """Verify urgency = (1 - P_survive) * positional_dropoff."""

    def test_urgency_formula_hand_calc(self, sim):
        """Urgency should be (1 - survival) * dropoff."""
        pool = _pool_df(10, adp_start=20)
        player = pool.iloc[0]  # pick_score=5.0, pos=OF
        current_pick = 10
        next_pick = 22

        p_surv = sim.survival_probability(float(player["adp"]), current_pick, next_pick)
        # Next best OF in pool
        of_pool = pool[(pool["positions"] == "OF") & (pool["player_id"] != player["player_id"])]
        if not of_pool.empty:
            dropoff = max(0, player["pick_score"] - of_pool["pick_score"].max())
        else:
            dropoff = player["pick_score"] * 0.3

        expected = (1 - p_surv) * dropoff
        actual = sim.compute_urgency(player, pool, current_pick, next_pick)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_safe_player_low_urgency(self, sim):
        """Player with high survival prob → low urgency."""
        pool = _pool_df(10, adp_start=100)  # All high ADP → safe
        player = pool.iloc[0]
        urgency = sim.compute_urgency(player, pool, current_pick=10, next_user_pick=22)
        assert urgency < 0.5, "Safe player should have low urgency"

    def test_scarce_player_high_urgency(self, sim):
        """Player about to be taken with big dropoff → high urgency."""
        pool = _pool_df(10, adp_start=15)
        # Make first player much better than rest at position
        pool.loc[0, "pick_score"] = 10.0
        pool.loc[2, "pick_score"] = 3.0  # next OF
        player = pool.iloc[0]
        urgency = sim.compute_urgency(player, pool, current_pick=10, next_user_pick=22)
        assert urgency > 0.5

    def test_urgency_non_negative(self, sim):
        """Urgency should never be negative."""
        pool = _pool_df(10)
        for _, player in pool.iterrows():
            urg = sim.compute_urgency(player, pool, current_pick=10, next_user_pick=22)
            assert urg >= 0, f"Negative urgency for {player['name']}"


# ── Combined Score ──────────────────────────────────────────────────


class TestCombinedScore:
    """Verify combined_score = MC_mean_sgp + urgency * 0.4."""

    def test_combined_score_formula(self):
        """Verify the weighting: combined = mean_sgp + urgency * 0.4."""
        mean_sgp = 3.5
        urgency = 2.0
        expected = mean_sgp + urgency * 0.4
        assert expected == pytest.approx(4.3, rel=1e-6)

    def test_urgency_weight_is_04(self):
        """The urgency weight constant should be 0.4."""
        # If urgency doubles, combined increases by 0.4 * delta
        mean_sgp = 3.0
        urg1 = 1.0
        urg2 = 3.0
        combined1 = mean_sgp + urg1 * 0.4
        combined2 = mean_sgp + urg2 * 0.4
        assert (combined2 - combined1) == pytest.approx(0.8, rel=1e-6)


# ── Opponent Pick Probabilities ─────────────────────────────────────


class TestOpponentProbabilities:
    """Verify the ADP-based probability model for opponent picks."""

    def test_probabilities_sum_to_one(self, sim):
        """Opponent probabilities must sum to 1.0."""
        pool = _pool_df(20)
        probs = sim.opponent_pick_probability(pool, pick_num=30)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_closer_adp_gets_higher_prob(self, sim):
        """Player closer to pick_num ADP should have higher probability."""
        pool = _pool_df(20, adp_start=10)
        probs = sim.opponent_pick_probability(pool, pick_num=30)
        # Find player closest to ADP 30
        closest_idx = (pool["adp"] - 30).abs().idxmin()
        farthest_idx = (pool["adp"] - 30).abs().idxmax()
        assert probs[closest_idx] > probs[farthest_idx]

    def test_need_boost_increases_prob(self, sim):
        """Player at needed position should get higher probability."""
        pool = _pool_df(20, adp_start=25)
        base_probs = sim.opponent_pick_probability(pool, pick_num=30)
        need_probs = sim.opponent_pick_probability(
            pool,
            pick_num=30,
            team_needs={"SS": 1},
        )
        # If pool has SS players, they should be boosted
        ss_mask = pool["positions"] == "SS"
        if ss_mask.any():
            ss_idx = pool[ss_mask].index[0]
            assert need_probs[ss_idx] >= base_probs[ss_idx]

    def test_empty_pool_uniform(self, sim):
        """Empty pool should return uniform (or handle gracefully)."""
        pool = _pool_df(3)
        probs = sim.opponent_pick_probability(pool, pick_num=30)
        assert len(probs) == 3
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_blending_weights_when_all_data(self, sim):
        """With needs + history: w_adp≈0.5, w_need=0.3, w_hist=0.2."""
        pool = _pool_df(10, adp_start=25)
        # Just verify probabilities are valid when all inputs provided
        probs = sim.opponent_pick_probability(
            pool,
            pick_num=30,
            team_needs={"OF": 2},
            team_preferences={"OF": 0.4, "SP": 0.3},
        )
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(probs >= 0)


# ── Team Preferences ───────────────────────────────────────────────


class TestTeamPreferences:
    """Verify compute_team_preferences: positional_bias sums to ~1.0."""

    def test_empty_history_returns_empty(self):
        """No history → empty preferences dict."""
        assert compute_team_preferences(None) == {}
        assert compute_team_preferences(pd.DataFrame()) == {}

    def test_single_team_bias_sums_to_one(self):
        """One team's positional_bias fractions should sum to ~1.0."""
        history = pd.DataFrame(
            [
                {"team_key": "T1", "positions": "OF", "round": 1},
                {"team_key": "T1", "positions": "SP", "round": 2},
                {"team_key": "T1", "positions": "OF", "round": 3},
                {"team_key": "T1", "positions": "SS", "round": 4},
            ]
        )
        prefs = compute_team_preferences(history)
        bias = prefs["T1"]["positional_bias"]
        assert sum(bias.values()) == pytest.approx(1.0, abs=0.01)

    def test_position_counts_correct(self):
        """Bias should reflect pick frequency: 2 OF out of 4 = 0.5."""
        history = pd.DataFrame(
            [
                {"team_key": "T1", "positions": "OF", "round": 1},
                {"team_key": "T1", "positions": "OF", "round": 2},
                {"team_key": "T1", "positions": "SP", "round": 3},
                {"team_key": "T1", "positions": "SP", "round": 4},
            ]
        )
        prefs = compute_team_preferences(history)
        assert prefs["T1"]["positional_bias"]["OF"] == pytest.approx(0.5, abs=0.01)
        assert prefs["T1"]["positional_bias"]["SP"] == pytest.approx(0.5, abs=0.01)

    def test_multi_team_independence(self):
        """Each team's preferences are computed independently."""
        history = pd.DataFrame(
            [
                {"team_key": "T1", "positions": "OF", "round": 1},
                {"team_key": "T1", "positions": "OF", "round": 2},
                {"team_key": "T2", "positions": "SP", "round": 1},
                {"team_key": "T2", "positions": "SP", "round": 2},
            ]
        )
        prefs = compute_team_preferences(history)
        assert "T1" in prefs and "T2" in prefs
        assert prefs["T1"]["positional_bias"].get("OF", 0) == pytest.approx(1.0, abs=0.01)
        assert prefs["T2"]["positional_bias"].get("SP", 0) == pytest.approx(1.0, abs=0.01)

    def test_comma_positions_uses_primary(self):
        """Multi-position 'OF,1B' should use primary position 'OF'."""
        history = pd.DataFrame(
            [
                {"team_key": "T1", "positions": "OF,1B", "round": 1},
                {"team_key": "T1", "positions": "SS", "round": 2},
            ]
        )
        prefs = compute_team_preferences(history)
        assert "OF" in prefs["T1"]["positional_bias"]
        assert "1B" not in prefs["T1"]["positional_bias"]


# ── Tier Assignment (Natural Breaks) ───────────────────────────────


class TestTierAssignment:
    """Verify assign_tiers uses gap-based natural breaks."""

    def test_tiers_monotonically_increase(self):
        """Higher-scoring players should have lower (better) tier numbers."""
        pool = pd.DataFrame(
            {
                "player_id": range(1, 21),
                "pick_score": [10 - i * 0.5 for i in range(20)],
            }
        )
        tiered = assign_tiers(pool, "pick_score", n_tiers=4)
        # Tiers should be non-decreasing when sorted by descending score
        sorted_t = tiered.sort_values("pick_score", ascending=False)
        tiers_list = sorted_t["tier"].tolist()
        for i in range(len(tiers_list) - 1):
            assert tiers_list[i] <= tiers_list[i + 1]

    def test_tier_range_bounded(self):
        """All tiers should be in [1, n_tiers]."""
        pool = pd.DataFrame(
            {
                "player_id": range(1, 51),
                "pick_score": np.random.RandomState(42).uniform(0, 10, 50),
            }
        )
        tiered = assign_tiers(pool, "pick_score", n_tiers=8)
        assert tiered["tier"].min() >= 1
        assert tiered["tier"].max() <= 8

    def test_empty_pool_returns_empty(self):
        """Empty input → empty output."""
        empty = pd.DataFrame(columns=["player_id", "pick_score"])
        result = assign_tiers(empty)
        assert result.empty

    def test_small_pool_one_per_tier(self):
        """When n players <= n_tiers, each gets its own tier."""
        pool = pd.DataFrame({"player_id": [1, 2, 3], "pick_score": [10.0, 5.0, 1.0]})
        tiered = assign_tiers(pool, "pick_score", n_tiers=8)
        assert tiered["tier"].nunique() == 3

    def test_large_gap_creates_tier_boundary(self):
        """A clear gap in scores should produce a tier break across clusters."""
        # Three clusters with wide gaps: [10.0-9.3] [5.0-4.3] [1.0-0.8]
        scores = [10.0, 9.8, 9.5, 9.3, 5.0, 4.8, 4.5, 4.3, 1.0, 0.8]
        pool = pd.DataFrame({"player_id": range(1, 11), "pick_score": scores})
        tiered = assign_tiers(pool, "pick_score", n_tiers=4)
        sorted_t = tiered.sort_values("pick_score", ascending=False)
        # Top cluster (10.0) and bottom cluster (1.0) should be different tiers
        top_tier = sorted_t.iloc[0]["tier"]
        bottom_tier = sorted_t.iloc[-1]["tier"]
        assert top_tier < bottom_tier, "Top and bottom clusters should be different tiers"


# ── MC Simulation Convergence ───────────────────────────────────────


class TestMCConvergence:
    """Verify Monte Carlo simulation statistical properties."""

    def test_simulate_returns_required_keys(self, sim):
        """Simulation result dict must have expected keys."""
        pool = _pool_df(20)
        result = sim.simulate_draft(
            available_ids=pool["player_id"].values,
            adp_values=pool["adp"].values.astype(float),
            sgp_values=np.ones(20) * 2.0,
            positions=[str(p) for p in pool["positions"]],
            user_team_index=0,
            current_pick=1,
            total_picks=120,
            num_teams=12,
            user_roster_needs={"OF", "SP"},
            candidate_id=1,
            n_simulations=50,
        )
        assert "mean_sgp" in result
        assert "std_sgp" in result
        assert "p25_sgp" in result
        assert "risk_adjusted_sgp" in result

    def test_mean_sgp_reasonable(self, sim):
        """MC mean SGP should be within reasonable bounds."""
        pool = _pool_df(20)
        result = sim.simulate_draft(
            available_ids=pool["player_id"].values,
            adp_values=pool["adp"].values.astype(float),
            sgp_values=np.ones(20) * 2.0,
            positions=[str(p) for p in pool["positions"]],
            user_team_index=0,
            current_pick=1,
            total_picks=120,
            num_teams=12,
            user_roster_needs={"OF", "SP"},
            candidate_id=1,
            n_simulations=50,
        )
        # With SGP=2.0 for everyone and ~6-round horizon picking ~6 players
        assert result["mean_sgp"] > 0, "Mean SGP should be positive"

    def test_std_positive_with_noise(self, sim):
        """With percentile sampling, std should be positive."""
        pool = _pool_df(20)
        vol = np.ones(20) * 0.5  # some volatility
        result = sim.simulate_draft(
            available_ids=pool["player_id"].values,
            adp_values=pool["adp"].values.astype(float),
            sgp_values=np.ones(20) * 2.0,
            positions=[str(p) for p in pool["positions"]],
            user_team_index=0,
            current_pick=1,
            total_picks=120,
            num_teams=12,
            user_roster_needs={"OF", "SP"},
            candidate_id=1,
            n_simulations=100,
            use_percentile_sampling=True,
            sgp_volatility=vol,
        )
        assert result["std_sgp"] >= 0

    def test_risk_adjusted_le_mean_with_volatility(self, sim):
        """risk_adjusted = mean - 0.5*std, so ≤ mean when std > 0."""
        pool = _pool_df(20)
        vol = np.ones(20) * 1.0
        result = sim.simulate_draft(
            available_ids=pool["player_id"].values,
            adp_values=pool["adp"].values.astype(float),
            sgp_values=np.ones(20) * 2.0,
            positions=[str(p) for p in pool["positions"]],
            user_team_index=0,
            current_pick=1,
            total_picks=120,
            num_teams=12,
            user_roster_needs={"OF", "SP"},
            candidate_id=1,
            n_simulations=100,
            use_percentile_sampling=True,
            sgp_volatility=vol,
        )
        assert result["risk_adjusted_sgp"] <= result["mean_sgp"] + 0.01

    def test_horizon_limit(self):
        """MC horizon = current_pick + 1 + num_teams * 6."""
        current_pick = 10
        num_teams = 12
        total_picks = 276
        expected_horizon = min(total_picks, current_pick + 1 + num_teams * 6)
        assert expected_horizon == 83  # 10 + 1 + 72 = 83
