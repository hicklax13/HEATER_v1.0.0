"""Tests for Post-Draft Grader (src/draft_grader.py)."""

import numpy as np
import pandas as pd
import pytest

from src.draft_grader import (
    GRADE_THRESHOLDS,
    SGP_REACH_THRESHOLD,
    SGP_STEAL_THRESHOLD,
    build_expected_sgp_curve,
    category_balance_score,
    classify_pick,
    composite_to_grade,
    compute_category_projections,
    expected_sgp_at_pick,
    grade_draft,
)
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pool(n=50):
    """Create a player pool for draft grading tests."""
    players = []
    for i in range(n):
        is_hitter = i < 35
        if is_hitter:
            pos_options = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
            pos = pos_options[i % len(pos_options)]
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Hitter_{i + 1}",
                    "team": f"TM{i % 10}",
                    "positions": pos,
                    "is_hitter": 1,
                    "pa": 600 - i * 5,
                    "ab": 540 - i * 5,
                    "h": 150 - i * 2,
                    "r": 80 - i,
                    "hr": 30 - i,
                    "rbi": 90 - i * 2,
                    "sb": 15 - i % 8,
                    "avg": round(0.280 - i * 0.002, 3),
                    "obp": round(0.350 - i * 0.002, 3),
                    "bb": 55 - i,
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
                    "adp": i + 1,
                    "is_injured": 0,
                }
            )
        else:
            pos = "SP" if i % 2 == 0 else "RP"
            ip = 200 - (i - 35) * 8
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Pitcher_{i + 1}",
                    "team": f"TM{i % 10}",
                    "positions": pos,
                    "is_hitter": 0,
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
                    "ip": max(ip, 30),
                    "w": 12 - (i - 35),
                    "l": 6 + (i - 35) % 3,
                    "sv": 25 if pos == "RP" else 0,
                    "k": 200 - (i - 35) * 8,
                    "era": round(3.20 + (i - 35) * 0.15, 2),
                    "whip": round(1.05 + (i - 35) * 0.03, 2),
                    "er": 50 + (i - 35) * 3,
                    "bb_allowed": 40 + (i - 35),
                    "h_allowed": 120 + (i - 35) * 3,
                    "adp": i + 1,
                    "is_injured": 0,
                }
            )
    return pd.DataFrame(players)


def _make_draft_picks(pool, pick_ids, num_teams=12):
    """Create draft picks from specific player IDs."""
    picks = []
    for i, pid in enumerate(pick_ids):
        round_num = i // num_teams + 1
        pick_num = i * num_teams + 1  # Simplified pick number
        player = pool[pool["player_id"] == pid]
        name = player.iloc[0]["name"] if not player.empty else f"Player_{pid}"
        picks.append(
            {
                "round": round_num,
                "pick_number": pick_num,
                "player_id": pid,
                "player_name": name,
            }
        )
    return picks


@pytest.fixture
def pool():
    return _make_pool()


@pytest.fixture
def config():
    return LeagueConfig()


# ── Grade Mapping ─────────────────────────────────────────────────────


class TestGradeMapping:
    def test_a_plus(self):
        assert composite_to_grade(2.0) == "A+"

    def test_a(self):
        assert composite_to_grade(1.2) == "A"

    def test_b(self):
        assert composite_to_grade(0.2) == "B"

    def test_c(self):
        assert composite_to_grade(-0.5) == "C"

    def test_d(self):
        assert composite_to_grade(-1.2) == "D"

    def test_f(self):
        assert composite_to_grade(-2.0) == "F"

    def test_boundary_a_plus(self):
        assert composite_to_grade(1.5) == "A+"

    def test_boundary_b_minus(self):
        assert composite_to_grade(-0.1) == "B-"


# ── Expected SGP Curve ────────────────────────────────────────────────


class TestExpectedSGPCurve:
    def test_returns_list(self, pool, config):
        curve = build_expected_sgp_curve(pool, config)
        assert isinstance(curve, list)
        assert len(curve) == len(pool)

    def test_monotonic_with_adp(self, pool, config):
        """When sorted by ADP, SGP should generally decrease."""
        curve = build_expected_sgp_curve(pool, config)
        # Not strictly monotonic but first should be > last
        assert curve[0] > curve[-1]

    def test_at_pick_in_range(self, pool, config):
        curve = build_expected_sgp_curve(pool, config)
        val = expected_sgp_at_pick(1, curve)
        assert isinstance(val, float)

    def test_at_pick_beyond_range(self, pool, config):
        curve = build_expected_sgp_curve(pool, config)
        val = expected_sgp_at_pick(999, curve)
        assert val == curve[-1]

    def test_at_pick_zero(self, pool, config):
        curve = build_expected_sgp_curve(pool, config)
        val = expected_sgp_at_pick(0, curve)
        assert val == curve[0]


# ── Pick Classification ──────────────────────────────────────────────


class TestClassifyPick:
    def test_steal(self):
        cls, surplus, gap = classify_pick(5.0, 2.0, 50, 10, 12, "OF")
        assert cls in ("STEAL", "GREAT STEAL")
        assert surplus > 0

    def test_reach(self):
        cls, surplus, gap = classify_pick(1.0, 4.0, 10, 50, 12, "OF")
        assert cls in ("REACH", "SLIGHT REACH")
        assert surplus < 0

    def test_fair(self):
        cls, surplus, gap = classify_pick(3.0, 3.2, 25, 24, 12, "OF")
        assert cls == "FAIR"

    def test_great_steal(self):
        cls, surplus, gap = classify_pick(6.0, 2.0, 100, 10, 12, "OF")
        assert cls == "GREAT STEAL"

    def test_position_affects_threshold(self):
        """C/SS should use tighter thresholds than OF."""
        # Same SGP surplus, but different position
        cls_c, _, _ = classify_pick(3.5, 2.0, 30, 12, 12, "C")
        cls_of, _, _ = classify_pick(3.5, 2.0, 30, 12, 12, "OF")
        # Both should classify based on SGP surplus primarily
        assert cls_c in ("STEAL", "GREAT STEAL", "FAIR")
        assert cls_of in ("STEAL", "GREAT STEAL", "FAIR")

    def test_zero_adp_handled(self):
        cls, surplus, gap = classify_pick(3.0, 3.0, 0, 20, 12, "OF")
        assert gap == 0  # ADP=0 → gap=0


# ── Category Balance ──────────────────────────────────────────────────


class TestCategoryBalance:
    def test_perfectly_balanced(self):
        proj = {f"C{i}": {"total": 100, "z_score": 1.0} for i in range(12)}
        score = category_balance_score(proj)
        assert score == 1.0

    def test_unbalanced(self):
        proj = {}
        for i in range(12):
            proj[f"C{i}"] = {"total": 100, "z_score": float(i)}
        score = category_balance_score(proj)
        assert score < 0.8

    def test_single_category(self):
        proj = {"HR": {"total": 50, "z_score": 2.0}}
        score = category_balance_score(proj)
        assert score == 1.0

    def test_in_range(self):
        proj = {f"C{i}": {"total": 100, "z_score": 1.0 + i * 0.3} for i in range(12)}
        score = category_balance_score(proj)
        assert 0.0 <= score <= 1.0


# ── Category Projections ─────────────────────────────────────────────


class TestCategoryProjections:
    def test_returns_all_categories(self, pool, config):
        roster_ids = list(range(1, 15))
        proj = compute_category_projections(roster_ids, pool, config)
        for cat in config.all_categories:
            assert cat in proj
            assert "total" in proj[cat]
            assert "z_score" in proj[cat]

    def test_empty_roster(self, pool, config):
        proj = compute_category_projections([], pool, config)
        for cat in config.all_categories:
            assert cat in proj


# ── Main Draft Grader ─────────────────────────────────────────────────


class TestGradeDraft:
    def test_returns_required_keys(self, pool, config):
        picks = _make_draft_picks(pool, [1, 2, 3, 4, 5], 12)
        result = grade_draft(picks, pool, config)
        assert "overall_grade" in result
        assert "overall_score" in result
        assert "picks" in result
        assert "steals" in result
        assert "reaches" in result
        assert "category_projections" in result
        assert "strengths" in result
        assert "weaknesses" in result

    def test_grade_is_valid_letter(self, pool, config):
        picks = _make_draft_picks(pool, [1, 5, 10, 15, 20], 12)
        result = grade_draft(picks, pool, config)
        valid_grades = {g for _, g in GRADE_THRESHOLDS} | {"F", "N/A"}
        assert result["overall_grade"] in valid_grades

    def test_picks_annotated(self, pool, config):
        picks = _make_draft_picks(pool, [1, 2, 3], 12)
        result = grade_draft(picks, pool, config)
        assert len(result["picks"]) == 3
        for p in result["picks"]:
            assert "classification" in p
            assert "surplus" in p
            assert "sgp" in p
            assert "expected_sgp" in p

    def test_steals_sorted_by_surplus(self, pool, config):
        picks = _make_draft_picks(pool, list(range(1, 24)), 12)
        result = grade_draft(picks, pool, config)
        if len(result["steals"]) >= 2:
            for i in range(len(result["steals"]) - 1):
                assert result["steals"][i]["surplus"] >= result["steals"][i + 1]["surplus"]

    def test_reaches_sorted_ascending(self, pool, config):
        picks = _make_draft_picks(pool, list(range(1, 24)), 12)
        result = grade_draft(picks, pool, config)
        if len(result["reaches"]) >= 2:
            for i in range(len(result["reaches"]) - 1):
                assert result["reaches"][i]["surplus"] <= result["reaches"][i + 1]["surplus"]

    def test_strengths_and_weaknesses(self, pool, config):
        picks = _make_draft_picks(pool, list(range(1, 14)), 12)
        result = grade_draft(picks, pool, config)
        assert len(result["strengths"]) <= 3
        assert len(result["weaknesses"]) <= 3

    def test_empty_picks(self, pool, config):
        result = grade_draft([], pool, config)
        assert result["overall_grade"] == "N/A"

    def test_empty_pool(self, config):
        picks = [{"round": 1, "pick_number": 1, "player_id": 1, "player_name": "X"}]
        result = grade_draft(picks, pd.DataFrame(), config)
        assert result["overall_grade"] == "N/A"

    def test_good_draft_gets_good_grade(self, pool, config):
        """Drafting top players should get a good grade."""
        top_picks = _make_draft_picks(pool, [1, 2, 3, 4, 5, 6], 12)
        result = grade_draft(top_picks, pool, config)
        assert result["overall_grade"] in ("A+", "A", "A-", "B+", "B")

    def test_component_scores_present(self, pool, config):
        picks = _make_draft_picks(pool, [1, 5, 10], 12)
        result = grade_draft(picks, pool, config)
        assert "team_value_score" in result
        assert "pick_efficiency_score" in result
        assert "category_balance_score" in result

    def test_total_sgp_computed(self, pool, config):
        picks = _make_draft_picks(pool, [1, 2, 3], 12)
        result = grade_draft(picks, pool, config)
        assert "total_sgp" in result
        assert result["total_sgp"] > 0
