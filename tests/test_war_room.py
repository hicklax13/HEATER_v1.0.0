"""Tests for War Room modules: war_room, war_room_actions, war_room_hotcold."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.war_room import compute_matchup_pulse, get_flippable_categories
from src.war_room_actions import compute_todays_actions
from src.war_room_hotcold import (
    _build_hitter_headline,
    _build_pitcher_headline,
    _evaluate_hitter,
    _evaluate_pitcher,
    compute_hot_cold_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_matchup(
    wins: int = 5,
    losses: int = 4,
    ties: int = 3,
    week: int = 5,
    opp_name: str = "Team Rival",
    categories: list[dict] | None = None,
) -> dict:
    """Build a mock matchup dict mirroring yds.get_matchup() output."""
    if categories is None:
        categories = [
            {"cat": "R", "result": "WIN", "you": "45", "opp": "40"},
            {"cat": "HR", "result": "WIN", "you": "12", "opp": "10"},
            {"cat": "RBI", "result": "LOSS", "you": "38", "opp": "42"},
            {"cat": "SB", "result": "TIE", "you": "5", "opp": "5"},
            {"cat": "AVG", "result": "WIN", "you": ".285", "opp": ".270"},
            {"cat": "OBP", "result": "LOSS", "you": ".330", "opp": ".340"},
            {"cat": "W", "result": "WIN", "you": "5", "opp": "3"},
            {"cat": "L", "result": "LOSS", "you": "4", "opp": "2"},
            {"cat": "SV", "result": "WIN", "you": "6", "opp": "3"},
            {"cat": "K", "result": "LOSS", "you": "60", "opp": "68"},
            {"cat": "ERA", "result": "WIN", "you": "3.20", "opp": "4.10"},
            {"cat": "WHIP", "result": "TIE", "you": "1.15", "opp": "1.15"},
        ]
    return {
        "week": week,
        "opp_name": opp_name,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "categories": categories,
    }


def _make_roster(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a mock roster DataFrame."""
    if rows is None:
        rows = [
            {
                "name": "Mike Trout",
                "team": "LAA",
                "positions": "CF",
                "roster_slot": "OF",
                "is_hitter": True,
                "status": "",
                "mlb_id": 545361,
                "player_id": 1,
                "avg": 0.280,
                "obp": 0.350,
                "hr": 10,
                "rbi": 30,
                "sb": 3,
                "r": 25,
                "era": 0,
                "whip": 0,
                "k": 0,
                "w": 0,
                "sv": 0,
                "ip": 0,
                "l": 0,
            },
            {
                "name": "Aaron Judge",
                "team": "NYY",
                "positions": "RF",
                "roster_slot": "BN",
                "is_hitter": True,
                "status": "",
                "mlb_id": 592450,
                "player_id": 2,
                "avg": 0.300,
                "obp": 0.400,
                "hr": 15,
                "rbi": 40,
                "sb": 2,
                "r": 30,
                "era": 0,
                "whip": 0,
                "k": 0,
                "w": 0,
                "sv": 0,
                "ip": 0,
                "l": 0,
            },
            {
                "name": "Gerrit Cole",
                "team": "NYY",
                "positions": "SP",
                "roster_slot": "SP",
                "is_hitter": False,
                "status": "",
                "mlb_id": 543037,
                "player_id": 3,
                "avg": 0,
                "obp": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "r": 0,
                "era": 3.10,
                "whip": 1.05,
                "k": 80,
                "w": 5,
                "sv": 0,
                "ip": 60,
                "l": 2,
            },
            {
                "name": "IL Player",
                "team": "BOS",
                "positions": "SS",
                "roster_slot": "IL",
                "is_hitter": True,
                "status": "il10",
                "mlb_id": 111111,
                "player_id": 4,
                "avg": 0.250,
                "obp": 0.320,
                "hr": 5,
                "rbi": 15,
                "sb": 1,
                "r": 12,
                "era": 0,
                "whip": 0,
                "k": 0,
                "w": 0,
                "sv": 0,
                "ip": 0,
                "l": 0,
            },
        ]
    return pd.DataFrame(rows)


# ===========================================================================
# war_room.py — compute_matchup_pulse
# ===========================================================================


class TestComputeMatchupPulse:
    def test_none_matchup_returns_unavailable(self):
        result = compute_matchup_pulse(None)
        assert result["available"] is False
        assert result["score"] == "0-0-0"
        assert result["verdict"] == "Tied"

    def test_winning_matchup(self):
        matchup = _make_matchup(wins=7, losses=3, ties=2)
        result = compute_matchup_pulse(matchup)
        assert result["available"] is True
        assert result["verdict"] == "Leading"
        assert result["score"] == "7-3-2"
        assert result["margin"] == 4
        assert result["week"] == 5
        assert result["opponent"] == "Team Rival"

    def test_losing_matchup(self):
        matchup = _make_matchup(wins=3, losses=8, ties=1)
        result = compute_matchup_pulse(matchup)
        assert result["verdict"] == "Trailing"
        assert result["margin"] == -5

    def test_tied_matchup(self):
        matchup = _make_matchup(wins=6, losses=6, ties=0)
        result = compute_matchup_pulse(matchup)
        assert result["verdict"] == "Tied"
        assert result["margin"] == 0

    def test_category_classification(self):
        matchup = _make_matchup()
        result = compute_matchup_pulse(matchup)
        assert "R" in result["winning_cats"]
        assert "RBI" in result["losing_cats"]
        assert "SB" in result["tied_cats"]

    def test_empty_categories_list(self):
        matchup = _make_matchup(categories=[])
        result = compute_matchup_pulse(matchup)
        assert result["available"] is True
        assert result["winning_cats"] == []
        assert result["losing_cats"] == []
        assert result["tied_cats"] == []


# ===========================================================================
# war_room.py — get_flippable_categories
# ===========================================================================


class TestGetFlippableCategories:
    def test_none_matchup_returns_empty(self):
        assert get_flippable_categories(None) == []

    def test_no_categories_returns_empty(self):
        matchup = _make_matchup(categories=[])
        assert get_flippable_categories(matchup) == []

    def test_close_loss_is_flippable(self):
        """A category we're losing by a small gap should appear as flip_to_win."""
        categories = [
            {"cat": "HR", "result": "LOSS", "you": "10", "opp": "11"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert result[0]["category"] == "HR"
        assert result[0]["direction"] == "flip_to_win"
        assert result[0]["gap"] == 1.0

    def test_close_win_is_at_risk(self):
        """A category we're winning by a small gap should appear as at_risk."""
        categories = [
            {"cat": "RBI", "result": "WIN", "you": "42", "opp": "40"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert result[0]["category"] == "RBI"
        assert result[0]["direction"] == "at_risk"
        assert result[0]["gap"] == 2.0

    def test_tie_is_flippable(self):
        categories = [
            {"cat": "SB", "result": "TIE", "you": "5", "opp": "5"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert result[0]["direction"] == "flip_to_win"
        assert result[0]["gap"] == 0

    def test_large_gap_not_flippable(self):
        """A category with a gap beyond the weekly SD threshold should NOT appear."""
        categories = [
            {"cat": "K", "result": "LOSS", "you": "40", "opp": "70"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 0

    def test_max_three_returned(self):
        """At most 3 flippable categories should be returned."""
        categories = [
            {"cat": "HR", "result": "LOSS", "you": "10", "opp": "11"},
            {"cat": "RBI", "result": "LOSS", "you": "38", "opp": "40"},
            {"cat": "R", "result": "LOSS", "you": "44", "opp": "46"},
            {"cat": "SB", "result": "TIE", "you": "5", "opp": "5"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) <= 3

    def test_inverse_cat_loss_gap_direction(self):
        """For inverse cat (ERA), losing means our value is higher (worse)."""
        categories = [
            {"cat": "ERA", "result": "LOSS", "you": "3.80", "opp": "3.50"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert result[0]["direction"] == "flip_to_win"
        assert abs(result[0]["gap"] - 0.30) < 0.01

    def test_inverse_cat_win_at_risk(self):
        """For inverse cat (WHIP), winning means our value is lower (better)."""
        categories = [
            {"cat": "WHIP", "result": "WIN", "you": "1.10", "opp": "1.15"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert result[0]["direction"] == "at_risk"

    def test_suggestion_text_present(self):
        categories = [
            {"cat": "SV", "result": "LOSS", "you": "3", "opp": "4"},
        ]
        matchup = _make_matchup(categories=categories)
        result = get_flippable_categories(matchup)
        assert len(result) == 1
        assert len(result[0]["suggestion"]) > 0


# ===========================================================================
# war_room_actions.py — compute_todays_actions
# ===========================================================================


class TestComputeTodaysActions:
    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_empty_roster_returns_empty(self, _mock_sched):
        result = compute_todays_actions(pd.DataFrame(), teams_playing=["NYY"])
        assert result == []

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_no_games_today_returns_empty(self, _mock_sched):
        roster = _make_roster()
        result = compute_todays_actions(roster, teams_playing=[])
        assert result == []

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_offday_starter_generates_bench_action(self, _mock_sched):
        """A starter whose team is NOT playing should generate a bench action."""
        roster = _make_roster()
        # LAA not in teams_playing -> Trout is off
        result = compute_todays_actions(roster, teams_playing=["NYY", "BOS"])
        bench_actions = [a for a in result if a["action_type"] == "bench"]
        assert len(bench_actions) >= 1
        assert bench_actions[0]["player"] == "Mike Trout"
        assert bench_actions[0]["urgency"] == "high"

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_bench_player_start_for_losing_cat(self, _mock_sched):
        """A bench hitter whose team is playing should be suggested for losing cats."""
        roster = _make_roster()
        # NYY is playing, Judge is on bench, and we're losing HR
        result = compute_todays_actions(roster, teams_playing=["NYY", "LAA"], losing_cats=["HR"])
        start_actions = [a for a in result if a["action_type"] == "start" and a["player"] == "Aaron Judge"]
        assert len(start_actions) >= 1
        assert "HR" in start_actions[0]["category_impact"]

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_max_five_actions(self, _mock_sched):
        """Actions should be capped at 5."""
        # Build a large roster with many bench players
        rows = []
        for i in range(15):
            rows.append(
                {
                    "name": f"Player {i}",
                    "team": "NYY",
                    "positions": "OF",
                    "roster_slot": "BN" if i < 8 else "OF",
                    "is_hitter": True,
                    "status": "",
                    "mlb_id": 100000 + i,
                    "player_id": i,
                    "avg": 0.260,
                    "obp": 0.330,
                    "hr": 5,
                    "rbi": 15,
                    "sb": 2,
                    "r": 12,
                    "era": 0,
                    "whip": 0,
                    "k": 0,
                    "w": 0,
                    "sv": 0,
                    "ip": 0,
                    "l": 0,
                }
            )
        roster = pd.DataFrame(rows)
        result = compute_todays_actions(
            roster,
            teams_playing=["NYY"],
            losing_cats=["HR", "RBI", "SB", "R", "AVG", "OBP"],
        )
        assert len(result) <= 5

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_il_player_not_suggested(self, _mock_sched):
        """IL players should never appear in actions."""
        roster = _make_roster()
        result = compute_todays_actions(roster, teams_playing=["BOS", "NYY", "LAA"], losing_cats=["HR"])
        il_actions = [a for a in result if a["player"] == "IL Player"]
        assert len(il_actions) == 0

    @patch("src.war_room_actions._build_schedule_context", return_value=({"NYY": "BOS", "BOS": "NYY"}, {"cole"}))
    def test_sp_matchup_action_generated(self, _mock_sched):
        """SP who is a probable starter with schedule pairings should generate an action."""
        roster = _make_roster()
        result = compute_todays_actions(roster, teams_playing=["NYY", "LAA", "BOS"])
        sp_actions = [a for a in result if a["player"] == "Gerrit Cole" and a["priority"] == 3]
        # Cole is SP, NYY is playing, and he's in probable starters
        assert len(sp_actions) >= 1

    @patch("src.war_room_actions._build_schedule_context", return_value=({}, set()))
    def test_actions_sorted_by_priority(self, _mock_sched):
        """Actions should be sorted by priority (lowest first)."""
        roster = _make_roster()
        # LAA not playing -> bench action for Trout (priority 1)
        # NYY playing, Judge on bench with losing cats -> start action (priority 2)
        result = compute_todays_actions(roster, teams_playing=["NYY", "BOS"], losing_cats=["HR"])
        if len(result) >= 2:
            priorities = [a["priority"] for a in result]
            assert priorities == sorted(priorities)


# ===========================================================================
# war_room_hotcold.py — compute_hot_cold_report
# ===========================================================================


class TestComputeHotColdReport:
    def test_empty_roster_returns_empty(self):
        result = compute_hot_cold_report(pd.DataFrame())
        assert result == []

    def test_none_roster_returns_empty(self):
        result = compute_hot_cold_report(None)
        assert result == []

    @patch("src.game_day.get_player_recent_form_cached")
    def test_hot_hitter_detected(self, mock_form):
        """A hitter with L7 AVG much higher than season should be marked hot."""
        mock_form.return_value = {
            "player_type": "hitter",
            "l7": {"games": 7, "avg": 0.420, "hr": 3, "rbi": 8, "sb": 2, "r": 6},
        }
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        hot = [e for e in result if e["status"] == "hot" and e["player_type"] == "hitter"]
        assert len(hot) >= 1
        assert hot[0]["deviation_score"] > 0

    @patch("src.game_day.get_player_recent_form_cached")
    def test_cold_hitter_detected(self, mock_form):
        """A hitter with L7 AVG much lower than season should be marked cold."""
        mock_form.return_value = {
            "player_type": "hitter",
            "l7": {"games": 7, "avg": 0.100, "hr": 0, "rbi": 1, "sb": 0, "r": 0},
        }
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        cold = [e for e in result if e["status"] == "cold" and e["player_type"] == "hitter"]
        assert len(cold) >= 1
        assert cold[0]["deviation_score"] < 0

    @patch("src.game_day.get_player_recent_form_cached")
    def test_hot_pitcher_detected(self, mock_form):
        """A pitcher with L7 ERA much lower than season should be marked hot."""

        def _form_side_effect(mlb_id):
            if mlb_id == 543037:  # Cole
                return {
                    "player_type": "pitcher",
                    "l7": {"games": 3, "era": 1.00, "whip": 0.80, "k": 25, "ip": 18},
                }
            return None

        mock_form.side_effect = _form_side_effect
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        hot_pitchers = [e for e in result if e["status"] == "hot" and e["player_type"] == "pitcher"]
        assert len(hot_pitchers) >= 1

    @patch("src.game_day.get_player_recent_form_cached")
    def test_cold_pitcher_detected(self, mock_form):
        """A pitcher with L7 ERA much higher than season should be marked cold."""

        def _form_side_effect(mlb_id):
            if mlb_id == 543037:  # Cole
                return {
                    "player_type": "pitcher",
                    "l7": {"games": 4, "era": 8.50, "whip": 1.90, "k": 5, "ip": 12},
                }
            return None

        mock_form.side_effect = _form_side_effect
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        cold_pitchers = [e for e in result if e["status"] == "cold" and e["player_type"] == "pitcher"]
        assert len(cold_pitchers) >= 1
        assert cold_pitchers[0]["deviation_score"] < 0

    @patch("src.game_day.get_player_recent_form_cached")
    def test_il_players_skipped(self, mock_form):
        """IL players should not appear in the hot/cold report."""
        mock_form.return_value = {
            "player_type": "hitter",
            "l7": {"games": 7, "avg": 0.500, "hr": 5, "rbi": 12, "sb": 3, "r": 8},
        }
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        il_entries = [e for e in result if e["player"] == "IL Player"]
        assert len(il_entries) == 0

    @patch("src.game_day.get_player_recent_form_cached")
    def test_too_few_games_skipped(self, mock_form):
        """Players with fewer than 3 L7 games should be skipped."""
        mock_form.return_value = {
            "player_type": "hitter",
            "l7": {"games": 2, "avg": 0.500, "hr": 3, "rbi": 8, "sb": 2, "r": 4},
        }
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=10)
        assert len(result) == 0

    @patch("src.game_day.get_player_recent_form_cached")
    def test_max_entries_respected(self, mock_form):
        """Report should respect the max_entries limit."""
        mock_form.return_value = {
            "player_type": "hitter",
            "l7": {"games": 7, "avg": 0.450, "hr": 4, "rbi": 10, "sb": 3, "r": 7},
        }
        roster = _make_roster()
        result = compute_hot_cold_report(roster, max_entries=2)
        assert len(result) <= 2


# ===========================================================================
# war_room_hotcold.py — _evaluate_hitter / _evaluate_pitcher (critical math)
# ===========================================================================


class TestEvaluateHitter:
    def test_hot_hitter_positive_score(self):
        row = pd.Series({"avg": 0.260, "hr": 10})
        l7 = {"avg": 0.400, "hr": 3, "rbi": 8, "sb": 2, "r": 5, "games": 7}
        result = _evaluate_hitter(row, l7, "Test Player", "NYY")
        assert result is not None
        assert result["status"] == "hot"
        assert result["deviation_score"] > 0

    def test_cold_hitter_negative_score(self):
        row = pd.Series({"avg": 0.300, "hr": 20})
        l7 = {"avg": 0.100, "hr": 0, "rbi": 0, "sb": 0, "r": 0, "games": 7}
        result = _evaluate_hitter(row, l7, "Cold Batter", "BOS")
        assert result is not None
        assert result["status"] == "cold"
        assert result["deviation_score"] < 0

    def test_moderate_deviation_returns_none(self):
        """Deviation below threshold should return None (no signal)."""
        row = pd.Series({"avg": 0.265, "hr": 12})
        l7 = {"avg": 0.270, "hr": 1, "rbi": 4, "sb": 1, "r": 3, "games": 7}
        result = _evaluate_hitter(row, l7, "Average Joe", "CHC")
        # Small difference -- should be below moderate threshold
        assert result is None


class TestEvaluatePitcher:
    def test_hot_pitcher_low_era(self):
        row = pd.Series({"era": 4.50, "whip": 1.30, "k": 50, "ip": 50})
        l7 = {"era": 1.50, "whip": 0.80, "k": 20, "ip": 14, "games": 3}
        result = _evaluate_pitcher(row, l7, "Ace Pitcher", "LAD")
        assert result is not None
        assert result["status"] == "hot"
        assert result["deviation_score"] > 0

    def test_cold_pitcher_high_era(self):
        row = pd.Series({"era": 3.50, "whip": 1.10, "k": 80, "ip": 70})
        l7 = {"era": 9.00, "whip": 2.00, "k": 3, "ip": 10, "games": 3}
        result = _evaluate_pitcher(row, l7, "Struggling Arm", "CWS")
        assert result is not None
        assert result["status"] == "cold"
        assert result["deviation_score"] < 0


# ===========================================================================
# war_room_hotcold.py — headline builders
# ===========================================================================


class TestBuildHitterHeadline:
    def test_hot_headline_with_hr(self):
        headline = _build_hitter_headline(0.350, 3, 8, 2, 5, 7, is_hot=True)
        assert ".350 AVG" in headline
        assert "3 HR" in headline
        assert "7 games" in headline

    def test_cold_headline_zero_runs(self):
        headline = _build_hitter_headline(0.100, 0, 1, 0, 0, 7, is_hot=False)
        assert ".100 AVG" in headline
        assert "0 HR" in headline
        assert "0 R" in headline


class TestBuildPitcherHeadline:
    def test_hot_headline_with_k(self):
        headline = _build_pitcher_headline(1.50, 0.90, 20, 3, is_hot=True)
        assert "1.50 ERA" in headline
        assert "20 K" in headline
        assert "3 games" in headline

    def test_cold_headline_high_whip(self):
        headline = _build_pitcher_headline(7.20, 1.80, 0, 3, is_hot=False)
        assert "7.20 ERA" in headline
        assert "1.80 WHIP" in headline
