"""Tests for src/weekly_report.py — Monday reports, Thursday checkpoints, daily lineup checks."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_roster():
    """Minimal roster DataFrame with a few stat columns."""
    return pd.DataFrame(
        {
            "name": ["Player A", "Player B", "Player C"],
            "team": ["NYY", "BOS", "LAD"],
            "positions": ["1B,OF", "SS", "SP"],
            "roster_slot": ["1B", "SS", "SP"],
            "is_hitter": [1, 1, 0],
            "r": [10, 8, 0],
            "hr": [5, 3, 0],
            "rbi": [12, 6, 0],
            "sb": [1, 4, 0],
            "avg": [0.280, 0.310, 0.0],
            "obp": [0.350, 0.380, 0.0],
            "w": [0, 0, 3],
            "l": [0, 0, 1],
            "sv": [0, 0, 0],
            "k": [0, 0, 25],
            "era": [0.0, 0.0, 3.50],
            "whip": [0.0, 0.0, 1.10],
        }
    )


@pytest.fixture
def opponent_profile():
    return {
        "name": "Twigs",
        "tier": 1,
        "threat": "High",
        "manager": "Nick",
        "strengths": ["HR", "R", "SB", "K"],
        "weaknesses": ["SV"],
        "notes": "Judge + Duran.",
    }


# ---------------------------------------------------------------------------
# generate_monday_report
# ---------------------------------------------------------------------------


class TestGenerateMondayReport:
    def test_basic_structure(self, user_roster, opponent_profile):
        from src.weekly_report import generate_monday_report

        report = generate_monday_report(user_roster, None, opponent_profile, week=7)
        assert report["week"] == 7
        assert report["opponent"] == "Twigs"
        assert report["tier"] == 1
        assert report["threat"] == "High"
        assert "category_projections" in report
        assert "exploit_weaknesses" in report
        assert "protect_floor" in report
        assert "streaming_guidance" in report

    def test_category_projections_populated(self, user_roster, opponent_profile):
        from src.weekly_report import generate_monday_report

        report = generate_monday_report(user_roster, None, opponent_profile, week=7)
        cats = report["category_projections"]
        assert len(cats) > 0
        cat_names = [c["category"] for c in cats]
        # All 12 standard categories should appear
        for expected in ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]:
            assert expected in cat_names

    def test_outlook_labels(self, user_roster, opponent_profile):
        from src.weekly_report import generate_monday_report

        report = generate_monday_report(user_roster, None, opponent_profile, week=7)
        cats = {c["category"]: c["outlook"] for c in report["category_projections"]}
        # SV is an opponent weakness -> LIKELY WIN
        assert cats["SV"] == "LIKELY WIN"
        # HR is an opponent strength -> LIKELY LOSS
        assert cats["HR"] == "LIKELY LOSS"
        # W is neither -> TOSS-UP
        assert cats["W"] == "TOSS-UP"

    def test_empty_roster(self, opponent_profile):
        from src.weekly_report import generate_monday_report

        empty = pd.DataFrame()
        report = generate_monday_report(empty, None, opponent_profile, week=1)
        assert report["category_projections"] == []
        assert report["opponent"] == "Twigs"

    def test_exploit_weaknesses_populated(self, user_roster, opponent_profile):
        from src.weekly_report import generate_monday_report

        report = generate_monday_report(user_roster, None, opponent_profile, week=7)
        assert len(report["exploit_weaknesses"]) == 1
        assert "SV" in report["exploit_weaknesses"][0]

    def test_streaming_guidance_sv_weakness(self, user_roster, opponent_profile):
        from src.weekly_report import generate_monday_report

        report = generate_monday_report(user_roster, None, opponent_profile, week=7)
        assert any("RP" in s or "closers" in s for s in report["streaming_guidance"])

    def test_streaming_guidance_no_edge(self, user_roster):
        """When opponent has no exploitable weaknesses for streaming, fallback message."""
        from src.weekly_report import generate_monday_report

        profile = {
            "name": "Neutral",
            "tier": 3,
            "threat": "Medium",
            "manager": "X",
            "strengths": [],
            "weaknesses": ["HR"],  # HR not in streaming guidance triggers
            "notes": "",
        }
        report = generate_monday_report(user_roster, None, profile, week=1)
        assert any("Stick with" in s or "No clear" in s for s in report["streaming_guidance"])

    def test_missing_profile_fields(self, user_roster):
        """Opponent profile with missing keys should not crash."""
        from src.weekly_report import generate_monday_report

        sparse_profile = {}
        report = generate_monday_report(user_roster, None, sparse_profile, week=1)
        assert report["opponent"] == "Unknown"
        assert report["tier"] == 3
        assert report["threat"] == "Unknown"


# ---------------------------------------------------------------------------
# generate_thursday_checkpoint
# ---------------------------------------------------------------------------


class TestGenerateThursdayCheckpoint:
    def test_low_ip_danger(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        cp = generate_thursday_checkpoint(user_roster, ip_projected=12.0)
        assert "DANGER" in cp["ip_status"]
        assert any("Stream" in r for r in cp["recommendations"])

    def test_sufficient_ip(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        cp = generate_thursday_checkpoint(user_roster, ip_projected=25.0)
        assert "On pace" in cp["ip_status"]

    def test_matchup_score_close_categories(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        score = {
            "HR": {"margin": 2.0},
            "SB": {"margin": -0.1},
            "ERA": {"margin": 0.3},
        }
        cp = generate_thursday_checkpoint(user_roster, matchup_score=score, ip_projected=25.0)
        assert "SB" in cp["categories_at_risk"]
        assert "ERA" in cp["categories_at_risk"]

    def test_era_whip_protection_advice(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        score = {
            "ERA": {"margin": 1.0},
            "WHIP": {"margin": 0.8},
        }
        cp = generate_thursday_checkpoint(user_roster, matchup_score=score, ip_projected=25.0)
        assert any("benching" in r.lower() or "protect" in r.lower() for r in cp["recommendations"])

    def test_no_matchup_score(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        cp = generate_thursday_checkpoint(user_roster, matchup_score=None, ip_projected=25.0)
        assert cp["categories_at_risk"] == []

    def test_losing_k_or_w(self, user_roster):
        from src.weekly_report import generate_thursday_checkpoint

        score = {"K": {"margin": -1.5}, "W": {"margin": -2.0}}
        cp = generate_thursday_checkpoint(user_roster, matchup_score=score, ip_projected=25.0)
        assert any("K" in r or "W" in r for r in cp["recommendations"])


# ---------------------------------------------------------------------------
# check_daily_lineup
# ---------------------------------------------------------------------------


class TestCheckDailyLineup:
    def test_off_day_starter_flagged(self):
        from src.weekly_report import check_daily_lineup

        roster = pd.DataFrame(
            {
                "name": ["Starter", "Bencher"],
                "team": ["NYY", "BOS"],
                "roster_slot": ["1B", "BN"],
            }
        )
        # Only BOS is playing — NYY starter should be flagged
        alerts = check_daily_lineup(roster, todays_games=["BOS"])
        assert len(alerts) == 2  # one warning (off-day) + one info (benched active)
        warning = [a for a in alerts if a["severity"] == "warning"]
        assert len(warning) == 1
        assert warning[0]["player"] == "Starter"

    def test_benched_player_with_game_flagged(self):
        from src.weekly_report import check_daily_lineup

        roster = pd.DataFrame(
            {
                "name": ["Active"],
                "team": ["NYY"],
                "roster_slot": ["BN"],
            }
        )
        alerts = check_daily_lineup(roster, todays_games=["NYY"])
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"

    def test_il_player_not_flagged_as_benchable(self):
        from src.weekly_report import check_daily_lineup

        roster = pd.DataFrame(
            {
                "name": ["Hurt Guy"],
                "team": ["NYY"],
                "roster_slot": ["IL"],
            }
        )
        alerts = check_daily_lineup(roster, todays_games=["NYY"])
        # IL players should not get info alerts
        assert len(alerts) == 0

    def test_no_games_returns_empty(self):
        from src.weekly_report import check_daily_lineup

        roster = pd.DataFrame({"name": ["A"], "team": ["NYY"], "roster_slot": ["1B"]})
        alerts = check_daily_lineup(roster, todays_games=None)
        assert alerts == []

    def test_empty_roster(self):
        from src.weekly_report import check_daily_lineup

        roster = pd.DataFrame(columns=["name", "team", "roster_slot"])
        alerts = check_daily_lineup(roster, todays_games=["NYY"])
        assert alerts == []


# ---------------------------------------------------------------------------
# _position_eligible
# ---------------------------------------------------------------------------


class TestPositionEligible:
    def test_util_accepts_any(self):
        from src.weekly_report import _position_eligible

        assert _position_eligible("1B,OF", "Util") is True
        assert _position_eligible("SP", "UT") is True

    def test_p_slot_accepts_sp_rp(self):
        from src.weekly_report import _position_eligible

        assert _position_eligible("SP", "P") is True
        assert _position_eligible("RP", "P") is True
        assert _position_eligible("1B", "P") is False

    def test_of_slot_accepts_outfield(self):
        from src.weekly_report import _position_eligible

        assert _position_eligible("LF,CF", "OF") is True
        assert _position_eligible("RF", "OF") is True
        assert _position_eligible("SS", "OF") is False

    def test_direct_match(self):
        from src.weekly_report import _position_eligible

        assert _position_eligible("1B,DH", "1B") is True
        assert _position_eligible("2B", "SS") is False


# ---------------------------------------------------------------------------
# validate_daily_lineup (enriched alerts)
# ---------------------------------------------------------------------------


class TestValidateDailyLineup:
    def test_replacement_suggestions(self):
        from src.weekly_report import validate_daily_lineup

        roster = pd.DataFrame(
            {
                "name": ["Off Day Guy", "Bench Hit"],
                "team": ["NYY", "BOS"],
                "positions": ["1B,OF", "1B,DH"],
                "roster_slot": ["1B", "BN"],
                "is_hitter": [1, 1],
            }
        )
        alerts = validate_daily_lineup(roster, todays_games=["BOS"])
        warning = [a for a in alerts if a["severity"] == "warning"]
        assert len(warning) == 1
        assert "Bench Hit" in warning[0]["replacements"]

    def test_no_games_passes_through(self):
        from src.weekly_report import validate_daily_lineup

        roster = pd.DataFrame(
            {
                "name": ["A"],
                "team": ["NYY"],
                "positions": ["1B"],
                "roster_slot": ["1B"],
                "is_hitter": [1],
            }
        )
        alerts = validate_daily_lineup(roster, todays_games=None)
        assert alerts == []
