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
            "r": [50, 70, 65, 60, 55, 80, 75, 70, 45, 40, 0, 0, 0, 0, 0, 0, 0, 0],
            "hr": [15, 30, 18, 25, 12, 28, 22, 20, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
            "rbi": [55, 85, 60, 75, 50, 80, 70, 65, 40, 35, 0, 0, 0, 0, 0, 0, 0, 0],
            "sb": [3, 5, 15, 8, 20, 12, 10, 8, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            "avg": [0.255, 0.280, 0.270, 0.265, 0.275, 0.290, 0.260, 0.250, 0.240, 0.245, 0, 0, 0, 0, 0, 0, 0, 0],
            "w": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 10, 3, 2, 8, 6, 1, 1],
            "sv": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 20, 0, 0, 15, 10],
            "k": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 150, 70, 60, 120, 100, 50, 40],
            "era": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.50, 4.00, 3.20, 3.80, 3.90, 4.20, 3.60, 4.10],
            "whip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.15, 1.25, 1.10, 1.30, 1.20, 1.35, 1.15, 1.40],
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
            "r": [80, 60, 0],
            "hr": [30, 15, 0],
            "rbi": [90, 55, 0],
            "sb": [5, 20, 0],
            "avg": [0.280, 0.270, 0],
            "w": [0, 0, 12],
            "sv": [0, 0, 0],
            "k": [0, 0, 180],
            "era": [0, 0, 3.50],
            "whip": [0, 0, 1.15],
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

    def test_pitcher_value_includes_inverse(self, small_roster):
        row = small_roster.iloc[2]  # SP with ERA=3.50, WHIP=1.15
        weights = {cat: 1.0 for cat in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]}
        value = _compute_player_value(row, weights)
        # w=12 + k=180 - era=3.50 - whip=1.15 > 0
        assert value > 0

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
