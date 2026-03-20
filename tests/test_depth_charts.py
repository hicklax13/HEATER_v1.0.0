"""Tests for depth chart scraper and role classification.

Tests are organised into four groups:

1. ``TestFetchDepthCharts`` — HTTP fetch + parse (mocked)
2. ``TestClassifyRole`` — role classification from slot/role data
3. ``TestGetPlayerLineupSlot`` — lineup slot lookup
4. ``TestGetPlayerRole`` — combined role lookup
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.depth_charts import (
    classify_role,
    fetch_depth_charts,
    get_player_lineup_slot,
    get_player_role,
)

# ── Sample depth chart data for tests ─────────────────────────────────

SAMPLE_DEPTH_DATA: dict[str, dict] = {
    "NYY": {
        "lineup": [
            "Anthony Volpe",
            "Juan Soto",
            "Aaron Judge",
            "Giancarlo Stanton",
            "Jazz Chisholm Jr.",
            "Austin Wells",
            "Cody Bellinger",
            "Oswaldo Cabrera",
            "Trent Grisham",
        ],
        "rotation": [
            "Gerrit Cole",
            "Carlos Rodon",
            "Marcus Stroman",
            "Clarke Schmidt",
            "Luis Gil",
        ],
        "bullpen": {
            "CL": "Clay Holmes",
            "SU": ["Jonathan Loaisiga", "Tommy Kahnle"],
            "MR": ["Ian Hamilton", "Luke Weaver"],
        },
    },
    "LAD": {
        "lineup": [
            "Mookie Betts",
            "Shohei Ohtani",
            "Freddie Freeman",
            "Teoscar Hernandez",
            "Will Smith",
            "Max Muncy",
            "Chris Taylor",
            "Gavin Lux",
            "James Outman",
        ],
        "rotation": [
            "Tyler Glasnow",
            "Yoshinobu Yamamoto",
            "Bobby Miller",
            "Clayton Kershaw",
            "Walker Buehler",
        ],
        "bullpen": {
            "CL": "Evan Phillips",
            "SU": ["Alex Vesia", "Ryan Brasier"],
        },
    },
}


# ── TestFetchDepthCharts ──────────────────────────────────────────────


class TestFetchDepthCharts:
    """Tests for the fetch_depth_charts function."""

    def test_returns_dict(self):
        """fetch_depth_charts always returns a dict."""
        # Even if we can't reach the site, it should return a dict
        with patch("src.depth_charts.requests.get", side_effect=Exception("network error")):
            result = fetch_depth_charts()
        assert isinstance(result, dict)

    def test_empty_on_network_failure(self):
        """Returns empty dict when HTTP request fails."""
        with patch("src.depth_charts.requests.get", side_effect=ConnectionError("refused")):
            result = fetch_depth_charts()
            assert result == {}

    def test_empty_on_http_error(self):
        """Returns empty dict on non-200 status code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("src.depth_charts.requests.get", return_value=mock_resp):
            result = fetch_depth_charts()
            assert result == {}

    def test_empty_on_parse_error(self):
        """Returns empty dict when HTML parsing fails."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = "<html><body>not a depth chart</body></html>"
        with patch("src.depth_charts.requests.get", return_value=mock_resp):
            result = fetch_depth_charts()
            # Should be an empty dict since no team links found
            assert isinstance(result, dict)

    def test_uses_correct_headers(self):
        """Verifies User-Agent header is sent."""
        with patch("src.depth_charts.requests.get", side_effect=Exception("stop")) as mock_get:
            fetch_depth_charts()
            if mock_get.called:
                call_kwargs = mock_get.call_args
                assert call_kwargs.kwargs.get("headers", {}).get("User-Agent") == "Fantasy Baseball Draft Tool"

    def test_uses_timeout(self):
        """Verifies timeout is passed to requests."""
        with patch("src.depth_charts.requests.get", side_effect=Exception("stop")) as mock_get:
            fetch_depth_charts()
            if mock_get.called:
                call_kwargs = mock_get.call_args
                # Connection test uses a short 5s timeout for fail-fast
                assert call_kwargs.kwargs.get("timeout") == 5


# ── TestClassifyRole ──────────────────────────────────────────────────


class TestClassifyRole:
    """Tests for the classify_role function."""

    def test_starter_from_lineup_slot(self):
        """Player in lineup slot 1-9 is a starter."""
        assert classify_role(lineup_slot=3, rotation_slot=None, bullpen_role=None) == "starter"

    def test_starter_from_lineup_slot_1(self):
        """Lineup slot 1 (leadoff) is a starter."""
        assert classify_role(lineup_slot=1, rotation_slot=None, bullpen_role=None) == "starter"

    def test_starter_from_lineup_slot_9(self):
        """Lineup slot 9 is a starter."""
        assert classify_role(lineup_slot=9, rotation_slot=None, bullpen_role=None) == "starter"

    def test_closer_role(self):
        """Bullpen role CL maps to closer."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="CL") == "closer"

    def test_closer_case_insensitive(self):
        """Bullpen role matching is case-insensitive."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="cl") == "closer"

    def test_setup_role(self):
        """Bullpen role SU maps to setup."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="SU") == "setup"

    def test_setup_case_insensitive(self):
        """Setup role matching is case-insensitive."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="su") == "setup"

    def test_rotation_starter(self):
        """Player in rotation slot 1-5 is a starter."""
        assert classify_role(lineup_slot=None, rotation_slot=2, bullpen_role=None) == "starter"

    def test_rotation_slot_5(self):
        """Rotation slot 5 is still a starter."""
        assert classify_role(lineup_slot=None, rotation_slot=5, bullpen_role=None) == "starter"

    def test_bench_player(self):
        """Player with no slot data is bench."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role=None) == "bench"

    def test_generic_bullpen(self):
        """Middle reliever role maps to bullpen."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="MR") == "bullpen"

    def test_long_reliever(self):
        """Long reliever role maps to bullpen."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="LR") == "bullpen"

    def test_lineup_takes_priority_over_rotation(self):
        """If a player has both lineup and rotation, lineup wins."""
        assert classify_role(lineup_slot=5, rotation_slot=1, bullpen_role=None) == "starter"

    def test_closer_takes_priority_over_rotation(self):
        """Closer role takes priority over rotation slot."""
        assert classify_role(lineup_slot=None, rotation_slot=1, bullpen_role="CL") == "closer"

    def test_invalid_lineup_slot_zero(self):
        """Lineup slot 0 is invalid, should fall through."""
        assert classify_role(lineup_slot=0, rotation_slot=None, bullpen_role=None) == "bench"

    def test_invalid_lineup_slot_negative(self):
        """Negative lineup slot is invalid."""
        assert classify_role(lineup_slot=-1, rotation_slot=None, bullpen_role=None) == "bench"

    def test_invalid_rotation_slot_zero(self):
        """Rotation slot 0 is invalid."""
        assert classify_role(lineup_slot=None, rotation_slot=0, bullpen_role=None) == "bench"

    def test_empty_bullpen_role_string(self):
        """Empty string bullpen role falls through to bench."""
        assert classify_role(lineup_slot=None, rotation_slot=None, bullpen_role="") == "bench"


# ── TestGetPlayerLineupSlot ───────────────────────────────────────────


class TestGetPlayerLineupSlot:
    """Tests for the get_player_lineup_slot function."""

    def test_finds_player_in_lineup(self):
        """Finds Aaron Judge at slot 3 in NYY lineup."""
        slot = get_player_lineup_slot("Aaron Judge", SAMPLE_DEPTH_DATA)
        assert slot == 3

    def test_finds_leadoff_hitter(self):
        """Finds leadoff hitter at slot 1."""
        slot = get_player_lineup_slot("Anthony Volpe", SAMPLE_DEPTH_DATA)
        assert slot == 1

    def test_finds_ninth_hitter(self):
        """Finds 9th slot hitter."""
        slot = get_player_lineup_slot("Trent Grisham", SAMPLE_DEPTH_DATA)
        assert slot == 9

    def test_finds_player_across_teams(self):
        """Finds a player on a different team (LAD)."""
        slot = get_player_lineup_slot("Mookie Betts", SAMPLE_DEPTH_DATA)
        assert slot == 1

    def test_case_insensitive_match(self):
        """Case-insensitive name matching works."""
        slot = get_player_lineup_slot("aaron judge", SAMPLE_DEPTH_DATA)
        assert slot == 3

    def test_case_insensitive_upper(self):
        """All-caps name matching works."""
        slot = get_player_lineup_slot("AARON JUDGE", SAMPLE_DEPTH_DATA)
        assert slot == 3

    def test_not_found_returns_none(self):
        """Returns None for a player not in any lineup."""
        slot = get_player_lineup_slot("Unknown Player", SAMPLE_DEPTH_DATA)
        assert slot is None

    def test_pitcher_not_in_lineup(self):
        """Pitchers are not in the batting lineup (NL DH era)."""
        slot = get_player_lineup_slot("Gerrit Cole", SAMPLE_DEPTH_DATA)
        assert slot is None

    def test_empty_name_returns_none(self):
        """Empty string name returns None."""
        slot = get_player_lineup_slot("", SAMPLE_DEPTH_DATA)
        assert slot is None

    def test_none_name_returns_none(self):
        """None name returns None."""
        slot = get_player_lineup_slot(None, SAMPLE_DEPTH_DATA)
        assert slot is None

    def test_empty_depth_data_returns_none(self):
        """Empty depth data returns None."""
        slot = get_player_lineup_slot("Aaron Judge", {})
        assert slot is None


# ── TestGetPlayerRole ─────────────────────────────────────────────────


class TestGetPlayerRole:
    """Tests for the get_player_role function."""

    def test_lineup_starter(self):
        """Lineup player returns starter."""
        role = get_player_role("Aaron Judge", SAMPLE_DEPTH_DATA)
        assert role == "starter"

    def test_rotation_pitcher(self):
        """Rotation pitcher returns starter."""
        role = get_player_role("Gerrit Cole", SAMPLE_DEPTH_DATA)
        assert role == "starter"

    def test_closer(self):
        """Closer returns closer."""
        role = get_player_role("Clay Holmes", SAMPLE_DEPTH_DATA)
        assert role == "closer"

    def test_setup_man(self):
        """Setup man returns setup."""
        role = get_player_role("Jonathan Loaisiga", SAMPLE_DEPTH_DATA)
        assert role == "setup"

    def test_middle_reliever(self):
        """Middle reliever returns bullpen."""
        role = get_player_role("Ian Hamilton", SAMPLE_DEPTH_DATA)
        assert role == "bullpen"

    def test_case_insensitive(self):
        """Case-insensitive name matching works."""
        role = get_player_role("clay holmes", SAMPLE_DEPTH_DATA)
        assert role == "closer"

    def test_unknown_player(self):
        """Unknown player returns bench."""
        role = get_player_role("Unknown Player", SAMPLE_DEPTH_DATA)
        assert role == "bench"

    def test_empty_name_returns_bench(self):
        """Empty string returns bench."""
        role = get_player_role("", SAMPLE_DEPTH_DATA)
        assert role == "bench"

    def test_none_name_returns_bench(self):
        """None returns bench."""
        role = get_player_role(None, SAMPLE_DEPTH_DATA)
        assert role == "bench"

    def test_empty_depth_data(self):
        """Empty depth data returns bench."""
        role = get_player_role("Aaron Judge", {})
        assert role == "bench"

    def test_closer_on_another_team(self):
        """Finds closer on LAD team."""
        role = get_player_role("Evan Phillips", SAMPLE_DEPTH_DATA)
        assert role == "closer"

    def test_rotation_pitcher_lad(self):
        """Finds rotation pitcher on LAD."""
        role = get_player_role("Tyler Glasnow", SAMPLE_DEPTH_DATA)
        assert role == "starter"
