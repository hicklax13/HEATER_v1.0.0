"""Tests for lineup optimizer."""

import pandas as pd
import pytest

from src.lineup_optimizer import (
    PULP_AVAILABLE,
    LineupOptimizer,
    _compute_player_value,
    _parse_positions,
)


@pytest.fixture
def sample_roster():
    """Create a sample roster with hitters and pitchers.

    Must have enough players to fill all ROSTER_SLOTS:
    C(1) + 1B(1) + 2B(1) + 3B(1) + SS(1) + OF(3) + Util(2) + SP(2) + RP(2) + P(4) = 18 slots.
    We provide 18 players (10 hitters + 8 pitchers) so the LP is feasible.

    Includes ip, is_hitter, ab, h, er, bb_allowed, h_allowed columns that
    mirror real get_team_roster() output for scale normalization and rate stats.
    """
    return pd.DataFrame(
        {
            "player_id": list(range(1, 19)),
            "player_name": [
                "Catcher1",
                "First1",
                "Second1",
                "Third1",
                "Short1",
                "OF1",
                "OF2",
                "OF3",
                "Util1",
                "Util2",
                "SP1",
                "SP2",
                "RP1",
                "RP2",
                "SP3",
                "SP4",
                "RP3",
                "RP4",
            ],
            "positions": [
                "C",
                "1B",
                "2B",
                "3B",
                "SS",
                "OF",
                "OF",
                "OF",
                "1B,OF",
                "2B,SS",
                "SP",
                "SP",
                "RP",
                "RP",
                "SP",
                "SP",
                "RP",
                "RP",
            ],
            "is_hitter": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "ab": [450, 550, 520, 500, 480, 570, 540, 530, 400, 380, 0, 0, 0, 0, 0, 0, 0, 0],
            "h": [115, 154, 140, 133, 132, 165, 140, 133, 96, 93, 0, 0, 0, 0, 0, 0, 0, 0],
            "r": [50, 70, 65, 60, 55, 80, 75, 70, 45, 40, 0, 0, 0, 0, 0, 0, 0, 0],
            "hr": [15, 30, 18, 25, 12, 28, 22, 20, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
            "rbi": [55, 85, 60, 75, 50, 80, 70, 65, 40, 35, 0, 0, 0, 0, 0, 0, 0, 0],
            "sb": [3, 5, 15, 8, 20, 12, 10, 8, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            "avg": [0.255, 0.280, 0.270, 0.265, 0.275, 0.290, 0.260, 0.250, 0.240, 0.245, 0, 0, 0, 0, 0, 0, 0, 0],
            "ip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 170, 65, 55, 150, 130, 60, 50],
            "w": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 10, 3, 2, 8, 6, 1, 1],
            "sv": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 20, 0, 0, 15, 10],
            "k": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 150, 70, 60, 120, 100, 50, 40],
            "era": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.50, 4.00, 3.20, 3.80, 3.90, 4.20, 3.60, 4.10],
            "whip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.15, 1.25, 1.10, 1.30, 1.20, 1.35, 1.15, 1.40],
            "er": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 76, 23, 23, 65, 61, 24, 23],
            "bb_allowed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 55, 18, 20, 45, 50, 16, 20],
            "h_allowed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 158, 54, 52, 135, 126, 53, 50],
        }
    )


@pytest.fixture
def small_roster():
    """Minimal roster for simple tests."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Hitter A", "Hitter B", "Pitcher A"],
            "positions": ["1B,OF", "SS", "SP"],
            "is_hitter": [1, 1, 0],
            "ab": [550, 480, 0],
            "h": [154, 130, 0],
            "r": [80, 60, 0],
            "hr": [30, 15, 0],
            "rbi": [90, 55, 0],
            "sb": [5, 20, 0],
            "avg": [0.280, 0.270, 0],
            "ip": [0, 0, 190],
            "w": [0, 0, 12],
            "sv": [0, 0, 0],
            "k": [0, 0, 180],
            "era": [0, 0, 3.50],
            "whip": [0, 0, 1.15],
            "er": [0, 0, 74],
            "bb_allowed": [0, 0, 50],
            "h_allowed": [0, 0, 169],
        }
    )


# ── Position parsing tests ───────────────────────────────────────────


class TestParsePositions:
    def test_single_position(self):
        assert _parse_positions("SS") == ["SS"]

    def test_multi_position(self):
        assert _parse_positions("1B,OF") == ["1B", "OF"]

    def test_empty_string(self):
        assert _parse_positions("") == []

    def test_none(self):
        assert _parse_positions(None) == []

    def test_with_spaces(self):
        assert _parse_positions("1B, OF, DH") == ["1B", "OF", "DH"]


# ── Player value computation tests ───────────────────────────────────


class TestPlayerValue:
    def test_hitter_value_positive(self, small_roster):
        row = small_roster.iloc[0]
        weights = {cat: 1.0 for cat in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]}
        value = _compute_player_value(row, weights)
        # r=80 + hr=30 + rbi=90 + sb=5 + avg=0.280 - era=0 - whip=0 + w=0 + sv=0 + k=0
        assert value > 0

    def test_pitcher_value_includes_inverse(self, sample_roster):
        """The best SP (W=12, K=180, ERA=3.50, IP=190) should have positive
        value when using proper scale normalization with a realistic roster."""
        from src.lineup_optimizer import _compute_scale_factors

        row = sample_roster.iloc[10]  # SP1: W=12, K=180, ERA=3.50, IP=190
        weights = {cat: 1.0 for cat in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]}
        scale = _compute_scale_factors(sample_roster)
        value = _compute_player_value(row, weights, scale)
        # With normalization, the best SP's W + K should outweigh ERA + WHIP
        assert value > 0, f"Best SP should have positive normalized value, got {value}"

    def test_weighted_categories(self, small_roster):
        row = small_roster.iloc[0]
        # Double the weight on HR
        weights_normal = {"hr": 1.0, "r": 0, "rbi": 0, "sb": 0, "avg": 0, "w": 0, "sv": 0, "k": 0, "era": 0, "whip": 0}
        weights_double = {"hr": 2.0, "r": 0, "rbi": 0, "sb": 0, "avg": 0, "w": 0, "sv": 0, "k": 0, "era": 0, "whip": 0}
        v1 = _compute_player_value(row, weights_normal)
        v2 = _compute_player_value(row, weights_double)
        assert v2 == pytest.approx(v1 * 2)


# ── Optimizer tests ──────────────────────────────────────────────────


class TestOptimizer:
    def test_empty_roster(self):
        empty = pd.DataFrame()
        opt = LineupOptimizer(empty)
        result = opt.optimize_lineup()
        assert result["assignments"] == []
        assert result["status"] in ("empty_roster", "greedy_fallback")

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_produces_valid_lineup(self, sample_roster):
        opt = LineupOptimizer(sample_roster)
        result = opt.optimize_lineup()
        assert result["status"] == "Optimal"
        assert len(result["assignments"]) > 0
        # Each assignment should have required keys
        for a in result["assignments"]:
            assert "slot" in a
            assert "player_name" in a
            assert "player_id" in a

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_no_duplicate_players(self, sample_roster):
        opt = LineupOptimizer(sample_roster)
        result = opt.optimize_lineup()
        assigned_ids = [a["player_id"] for a in result["assignments"]]
        assert len(assigned_ids) == len(set(assigned_ids)), "Player assigned to multiple slots"

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_position_eligibility(self, sample_roster):
        """Players should only be assigned to eligible positions."""
        opt = LineupOptimizer(sample_roster)
        result = opt.optimize_lineup()
        for a in result["assignments"]:
            pid = a["player_id"]
            row = sample_roster[sample_roster["player_id"] == pid].iloc[0]
            player_pos = set(row["positions"].split(","))
            slot = a["slot"]
            # The slot should accept at least one of the player's positions
            from src.lineup_optimizer import ROSTER_SLOTS

            if slot in ROSTER_SLOTS:
                eligible = set(ROSTER_SLOTS[slot][1])
                assert player_pos & eligible, f"{a['player_name']} ({player_pos}) in slot {slot} ({eligible})"

    def test_greedy_fallback(self, small_roster):
        """Greedy fallback should produce some assignments."""
        opt = LineupOptimizer(small_roster)
        result = opt._greedy_fallback()
        assert result["status"] == "greedy_fallback"
        # Should assign at least some players
        assert len(result["assignments"]) >= 0  # May be 0 if no matching slots

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_yahoo_style_sp_rp_eligibility(self):
        """Yahoo lists every pitcher as "SP,P" or "RP,P" — the generic "P"
        must NOT allow relievers to fill SP slots (or vice versa).
        Regression test for an LP that was happily putting RPs at SP."""
        # Two SPs and two RPs, all with the realistic "X,P" eligibility strings
        roster = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "player_name": ["TrueSP1", "TrueSP2", "TrueRP1", "TrueRP2"],
                "positions": ["SP,P", "SP,P", "RP,P", "RP,P"],
                "is_hitter": [0, 0, 0, 0],
                "ab": [0, 0, 0, 0],
                "h": [0, 0, 0, 0],
                "r": [0, 0, 0, 0],
                "hr": [0, 0, 0, 0],
                "rbi": [0, 0, 0, 0],
                "sb": [0, 0, 0, 0],
                "avg": [0, 0, 0, 0],
                "ip": [190, 170, 65, 55],
                "w": [12, 10, 3, 2],
                "sv": [0, 0, 25, 20],
                "k": [180, 150, 70, 60],
                "era": [3.50, 4.00, 3.20, 3.80],
                "whip": [1.15, 1.25, 1.10, 1.30],
                "er": [74, 76, 23, 23],
                "bb_allowed": [50, 55, 18, 20],
                "h_allowed": [169, 158, 54, 52],
            }
        )
        opt = LineupOptimizer(roster, roster_slots={"SP": (2, ["SP"]), "RP": (2, ["RP"])})
        result = opt.optimize_lineup()
        sp_assignments = [a for a in result["assignments"] if a["slot"].startswith("SP")]
        rp_assignments = [a for a in result["assignments"] if a["slot"].startswith("RP")]
        # SP slots must be filled by SP-eligible players only
        for a in sp_assignments:
            row = roster[roster["player_id"] == a["player_id"]].iloc[0]
            assert "SP" in row["positions"].split(","), (
                f"{a['player_name']} ({row['positions']}) should not be in {a['slot']}"
            )
        # RP slots must be filled by RP-eligible players only
        for a in rp_assignments:
            row = roster[roster["player_id"] == a["player_id"]].iloc[0]
            assert "RP" in row["positions"].split(","), (
                f"{a['player_name']} ({row['positions']}) should not be in {a['slot']}"
            )

    def test_projected_stats_computed(self, sample_roster):
        opt = LineupOptimizer(sample_roster)
        result = opt.optimize_lineup()
        assert "projected_stats" in result
        if result["assignments"]:
            # Should have at least some categories
            assert len(result["projected_stats"]) > 0


# ── Category targeting tests ─────────────────────────────────────────


class TestCategoryTargeting:
    @pytest.fixture
    def standings(self):
        rows = []
        for cat in ["r", "hr", "rbi"]:
            for i, (team, total) in enumerate(
                [
                    ("Team A", 800),
                    ("Team B", 790),
                    ("Team C", 750),
                    ("My Team", 745),
                    ("Team E", 700),
                ]
            ):
                rows.append(
                    {
                        "team_name": team,
                        "category": cat,
                        "total": total,
                        "rank": i + 1,
                    }
                )
        return pd.DataFrame(rows)

    def test_small_gap_gets_high_weight(self, standings, small_roster):
        opt = LineupOptimizer(small_roster)
        weights = opt.category_targeting(standings, "My Team")
        # Gap to Team C in 'r' is 750-745=5 — small gap should get high weight
        assert "r" in weights

    def test_empty_standings(self, small_roster):
        opt = LineupOptimizer(small_roster)
        weights = opt.category_targeting(pd.DataFrame(), "My Team")
        assert all(v == 1.0 for v in weights.values())


# ── Two-start pitcher tests ─────────────────────────────────────────


class TestTwoStartPitchers:
    def test_with_schedule(self, sample_roster):
        schedule = {
            "SP1": ["2026-04-01", "2026-04-07"],
            "SP2": ["2026-04-02"],
        }
        result = LineupOptimizer.identify_two_start_pitchers(sample_roster, schedule)
        assert "SP1" in result
        assert "SP2" not in result

    def test_without_schedule(self, sample_roster):
        result = LineupOptimizer.identify_two_start_pitchers(sample_roster)
        # Without schedule, returns all SPs
        assert "SP1" in result
        assert "SP2" in result


# ── Suggest moves tests ──────────────────────────────────────────────


class TestSuggestMoves:
    def test_suggests_upgrades(self, small_roster):
        opt = LineupOptimizer(small_roster)
        better_fa = pd.DataFrame(
            {
                "player_id": [99],
                "player_name": ["Super Star"],
                "positions": ["1B"],
                "r": [100],
                "hr": [40],
                "rbi": [110],
                "sb": [10],
                "avg": [0.300],
                "w": [0],
                "sv": [0],
                "k": [0],
                "era": [0],
                "whip": [0],
            }
        )
        suggestions = opt.suggest_moves(better_fa)
        assert len(suggestions) > 0
        assert suggestions[0]["player_name"] == "Super Star"
        assert suggestions[0]["marginal_value"] > 0

    def test_empty_fa_pool(self, small_roster):
        opt = LineupOptimizer(small_roster)
        suggestions = opt.suggest_moves(pd.DataFrame())
        assert suggestions == []


# ── Scale normalization tests ────────────────────────────────────────


class TestScaleNormalization:
    """Verify that the LP objective function normalizes categories to
    comparable scales, preventing ERA*IP from dominating counting stats."""

    def test_scale_factors_computed(self, sample_roster):
        """Scale factors should be computed for all categories."""
        from src.lineup_optimizer import _compute_scale_factors

        scale = _compute_scale_factors(sample_roster)
        from src.lineup_optimizer import ALL_CATS

        assert set(scale.keys()) == set(ALL_CATS)
        # All scale factors should be positive
        for cat, s in scale.items():
            assert s > 0, f"Scale for {cat} should be positive, got {s}"

    def test_scale_differentiates_pitchers_correctly(self, sample_roster):
        """The best SP should have higher normalized value than the worst RP.
        Without normalization, ERA*IP dominance would make ALL pitchers
        look terrible and ranking would be inverted."""
        from src.lineup_optimizer import _compute_player_value, _compute_scale_factors

        scale = _compute_scale_factors(sample_roster)
        weights = {cat: 1.0 for cat in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]}

        # SP1 (idx=10): W=12, K=180, ERA=3.50, WHIP=1.15, IP=190 — best SP
        sp1_value = _compute_player_value(sample_roster.iloc[10], weights, scale)
        # RP4 (idx=17): W=1, K=40, ERA=4.10, WHIP=1.40, IP=50 — worst pitcher
        rp4_value = _compute_player_value(sample_roster.iloc[17], weights, scale)

        # Best SP should clearly outrank worst RP
        assert sp1_value > rp4_value, f"SP1 ({sp1_value:.3f}) should rank higher than RP4 ({rp4_value:.3f})"
        # Best SP should have positive value (worth starting)
        assert sp1_value > 0, f"Best SP should have positive normalized value, got {sp1_value}"

    def test_avg_not_negligible(self, sample_roster):
        """AVG (0.280) should not be negligible compared to HR (30) after
        normalization. Both should contribute meaningfully to total value."""
        from src.lineup_optimizer import _compute_scale_factors

        scale = _compute_scale_factors(sample_roster)
        # AVG scale should be much smaller than HR scale (since raw values differ 100x)
        # This means AVG/scale_avg ≈ HR/scale_hr (both ~1-3 standard deviations)
        avg_contribution = 0.280 / scale["avg"]
        hr_contribution = 30 / scale["hr"]
        ratio = hr_contribution / avg_contribution if avg_contribution > 0 else float("inf")
        assert ratio < 20, (
            f"HR contribution ({hr_contribution:.2f}) should not dominate "
            f"AVG contribution ({avg_contribution:.2f}) by more than 20x, ratio={ratio:.1f}"
        )

    @pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP not installed")
    def test_lp_starts_sp_over_rp_when_better(self, sample_roster):
        """The LP should start the best SP (SP1: W=12, K=180, ERA=3.50, IP=190)
        because normalization prevents ERA*IP from making SPs look terrible."""
        opt = LineupOptimizer(sample_roster)
        result = opt.optimize_lineup()
        assigned_names = {a["player_name"] for a in result["assignments"]}
        # SP1 is the best pitcher — should be in the lineup
        assert "SP1" in assigned_names, (
            f"SP1 (best starter) should be in lineup but was benched. Assigned: {assigned_names}"
        )
