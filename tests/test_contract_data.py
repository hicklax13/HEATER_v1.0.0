"""Tests for src/contract_data.py — contract year data from BB-Ref."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.contract_data import (
    fetch_contract_year_players,
    is_contract_year,
    mark_contract_years,
)

# ── Sample HTML mimicking BB-Ref free agent table ───────────────────

_SAMPLE_HTML = """
<html><body>
<table id="fa_2027">
  <thead><tr><th>Name</th><th>Team</th></tr></thead>
  <tbody>
    <tr><td><a href="/players/t/troutmi01.shtml">Mike Trout</a></td><td>LAA</td></tr>
    <tr><td><a href="/players/s/sotoju01.shtml">Juan Soto</a></td><td>NYM</td></tr>
    <tr><td><a href="/players/a/acunaro01.shtml">Ronald Acuna Jr.</a></td><td>ATL</td></tr>
  </tbody>
</table>
</body></html>
"""


# ── fetch_contract_year_players ─────────────────────────────────────


class TestFetchContractYearPlayers:
    """Tests for the BB-Ref scraper."""

    @patch("src.contract_data.requests.get")
    def test_successful_scrape(self, mock_get):
        """Parses player names from a valid HTML table."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_contract_year_players(fa_year=2027)

        assert isinstance(result, set)
        assert len(result) == 3
        assert "mike trout" in result
        assert "juan soto" in result
        assert "ronald acuna jr." in result

    @patch("src.contract_data.requests.get")
    def test_network_error_returns_empty_set(self, mock_get):
        """Network failures return empty set, not an exception."""
        mock_get.side_effect = ConnectionError("No internet")

        result = fetch_contract_year_players(fa_year=2027)

        assert result == set()

    @patch("src.contract_data.requests.get")
    def test_http_error_returns_empty_set(self, mock_get):
        """HTTP 404 / 500 returns empty set."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_resp

        result = fetch_contract_year_players(fa_year=2027)

        assert result == set()

    @patch("src.contract_data.requests.get")
    def test_empty_table_returns_empty_set(self, mock_get):
        """Page with no player links returns empty set."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><table><tr><td>No data</td></tr></table></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_contract_year_players(fa_year=2027)

        assert result == set()

    @patch("src.contract_data.requests.get")
    def test_uses_correct_url_and_headers(self, mock_get):
        """Verify URL pattern and User-Agent header."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_contract_year_players(fa_year=2028)

        mock_get.assert_called_once_with(
            "https://www.baseball-reference.com/leagues/majors/2028-free-agents.shtml",
            headers={"User-Agent": "Fantasy Baseball Draft Tool"},
            timeout=15,
        )

    @patch("src.contract_data.requests.get")
    def test_names_are_lowercased(self, mock_get):
        """All returned names must be lowercase for case-insensitive matching."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_contract_year_players(fa_year=2027)

        for name in result:
            assert name == name.lower(), f"Name not lowercased: {name}"


# ── is_contract_year ────────────────────────────────────────────────


class TestIsContractYear:
    """Tests for the case-insensitive contract year check."""

    def test_exact_match(self):
        names = {"mike trout", "juan soto"}
        assert is_contract_year("mike trout", names) is True

    def test_case_insensitive(self):
        names = {"mike trout", "juan soto"}
        assert is_contract_year("Mike Trout", names) is True
        assert is_contract_year("JUAN SOTO", names) is True

    def test_not_in_set(self):
        names = {"mike trout", "juan soto"}
        assert is_contract_year("Shohei Ohtani", names) is False

    def test_empty_set(self):
        assert is_contract_year("Mike Trout", set()) is False

    def test_empty_name(self):
        names = {"mike trout"}
        assert is_contract_year("", names) is False

    def test_none_name(self):
        names = {"mike trout"}
        assert is_contract_year(None, names) is False

    def test_whitespace_stripped(self):
        names = {"mike trout"}
        assert is_contract_year("  Mike Trout  ", names) is True


# ── mark_contract_years ─────────────────────────────────────────────


class TestMarkContractYears:
    """Tests for adding contract_year column to player pool."""

    def test_marks_matching_players(self):
        pool = pd.DataFrame({"name": ["Mike Trout", "Juan Soto", "Shohei Ohtani"], "team": ["LAA", "NYM", "LAD"]})
        cy_set = {"mike trout", "juan soto"}

        result = mark_contract_years(pool, cy_set)

        assert "contract_year" in result.columns
        assert result.loc[result["name"] == "Mike Trout", "contract_year"].iloc[0] == True  # noqa: E712
        assert result.loc[result["name"] == "Juan Soto", "contract_year"].iloc[0] == True  # noqa: E712
        assert result.loc[result["name"] == "Shohei Ohtani", "contract_year"].iloc[0] == False  # noqa: E712

    def test_empty_set_all_false(self):
        pool = pd.DataFrame({"name": ["Mike Trout", "Juan Soto"]})

        result = mark_contract_years(pool, set())

        assert (result["contract_year"] == False).all()  # noqa: E712

    def test_missing_name_column(self):
        pool = pd.DataFrame({"player_name": ["Mike Trout"], "team": ["LAA"]})

        result = mark_contract_years(pool, {"mike trout"})

        assert "contract_year" in result.columns
        assert (result["contract_year"] == False).all()  # noqa: E712

    def test_empty_dataframe(self):
        pool = pd.DataFrame({"name": [], "team": []})
        cy_set = {"mike trout"}

        result = mark_contract_years(pool, cy_set)

        assert "contract_year" in result.columns
        assert len(result) == 0
