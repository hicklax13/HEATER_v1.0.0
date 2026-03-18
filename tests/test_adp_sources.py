"""Tests for the multi-source ADP module."""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adp_sources import (
    compute_composite_adp,
    fetch_fantasypros_ecr,
    fetch_nfbc_adp,
)

# ── Sample HTML responses ──────────────────────────────────────────

FANTASYPROS_HTML = """
<html><body>
<table id="ranking-table">
  <thead><tr><th>Rank</th><th>Player</th><th>Pos</th></tr></thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>
        <a class="player-name" href="/player/123">Aaron Judge</a>
        <small>NYY - OF</small>
      </td>
      <td>OF</td>
    </tr>
    <tr>
      <td>2</td>
      <td>
        <a class="player-name" href="/player/456">Shohei Ohtani</a>
        <small>LAD - DH</small>
      </td>
      <td>DH</td>
    </tr>
    <tr>
      <td>3</td>
      <td>
        <a class="player-name" href="/player/789">Mookie Betts</a>
        <small>LAD - SS</small>
      </td>
      <td>SS</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

NFBC_HTML = """
<html><body>
<table>
  <thead><tr><th>Rank</th><th>Player</th><th>ADP</th><th>Min</th><th>Max</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>Aaron Judge</td><td>1.5</td><td>1</td><td>3</td></tr>
    <tr><td>2</td><td>Shohei Ohtani</td><td>2.3</td><td>1</td><td>5</td></tr>
    <tr><td>3</td><td>Mookie Betts</td><td>4.8</td><td>2</td><td>8</td></tr>
  </tbody>
</table>
</body></html>
"""

EMPTY_HTML = "<html><body><div>No data available</div></body></html>"


# ── FantasyPros ECR tests ──────────────────────────────────────────


@patch("src.adp_sources.requests.get")
def test_fantasypros_ecr_success(mock_get):
    """FantasyPros ECR returns correct player data from mocked HTML."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = FANTASYPROS_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = fetch_fantasypros_ecr()

    assert not df.empty
    assert len(df) == 3
    assert list(df.columns) == ["player_name", "ecr_rank", "team", "position"]
    assert df.iloc[0]["player_name"] == "Aaron Judge"
    assert df.iloc[0]["ecr_rank"] == 1
    assert df.iloc[0]["team"] == "NYY"
    assert df.iloc[0]["position"] == "OF"
    assert df.iloc[1]["player_name"] == "Shohei Ohtani"
    assert df.iloc[2]["ecr_rank"] == 3

    # Verify correct headers and timeout
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["headers"]["User-Agent"] == "Fantasy Baseball Draft Tool"
    assert call_kwargs.kwargs["timeout"] == 15


@patch("src.adp_sources.requests.get")
def test_fantasypros_ecr_http_error(mock_get):
    """FantasyPros ECR returns empty DataFrame on HTTP error."""
    mock_get.side_effect = Exception("Connection refused")

    df = fetch_fantasypros_ecr()

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["player_name", "ecr_rank", "team", "position"]


@patch("src.adp_sources.requests.get")
def test_fantasypros_ecr_no_table(mock_get):
    """FantasyPros ECR returns empty DataFrame when no table found."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = EMPTY_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = fetch_fantasypros_ecr()

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── NFBC ADP tests ─────────────────────────────────────────────────


@patch("src.adp_sources.requests.get")
def test_nfbc_adp_success(mock_get):
    """NFBC ADP returns correct player data from mocked HTML."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = NFBC_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = fetch_nfbc_adp()

    assert not df.empty
    assert len(df) == 3
    assert list(df.columns) == ["player_name", "nfbc_adp"]
    assert df.iloc[0]["player_name"] == "Aaron Judge"
    assert df.iloc[0]["nfbc_adp"] == 1.5
    assert df.iloc[1]["player_name"] == "Shohei Ohtani"
    assert df.iloc[1]["nfbc_adp"] == 2.3

    # Verify correct headers and timeout
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["headers"]["User-Agent"] == "Fantasy Baseball Draft Tool"
    assert call_kwargs.kwargs["timeout"] == 15


@patch("src.adp_sources.requests.get")
def test_nfbc_adp_http_error(mock_get):
    """NFBC ADP returns empty DataFrame on HTTP error."""
    mock_get.side_effect = Exception("Timeout")

    df = fetch_nfbc_adp()

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["player_name", "nfbc_adp"]


@patch("src.adp_sources.requests.get")
def test_nfbc_adp_no_table(mock_get):
    """NFBC ADP returns empty DataFrame when no table found."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = EMPTY_HTML
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = fetch_nfbc_adp()

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── Composite ADP tests ───────────────────────────────────────────


def test_composite_adp_yahoo_priority():
    """Composite ADP uses Yahoo ADP when available (highest priority)."""
    row = {"yahoo_adp": 5.0, "fantasypros_adp": 10.0, "nfbc_adp": 8.0, "adp": 12.0}
    assert compute_composite_adp(row) == 5.0


def test_composite_adp_fantasypros_fallback():
    """Composite ADP falls back to FantasyPros when Yahoo missing."""
    row = {"yahoo_adp": None, "fantasypros_adp": 10.0, "nfbc_adp": 8.0, "adp": 12.0}
    assert compute_composite_adp(row) == 10.0


def test_composite_adp_nfbc_fallback():
    """Composite ADP falls back to NFBC when Yahoo and FantasyPros missing."""
    row = {"yahoo_adp": None, "fantasypros_adp": None, "nfbc_adp": 8.0, "adp": 12.0}
    assert compute_composite_adp(row) == 8.0


def test_composite_adp_base_fallback():
    """Composite ADP falls back to base adp when all sources missing."""
    row = {"yahoo_adp": None, "fantasypros_adp": None, "nfbc_adp": None, "adp": 12.0}
    assert compute_composite_adp(row) == 12.0


def test_composite_adp_all_none():
    """Composite ADP returns 999.0 when all sources are None."""
    row = {"yahoo_adp": None, "fantasypros_adp": None, "nfbc_adp": None, "adp": None}
    assert compute_composite_adp(row) == 999.0


def test_composite_adp_empty_dict():
    """Composite ADP returns 999.0 for empty dict."""
    assert compute_composite_adp({}) == 999.0


def test_composite_adp_skips_nan():
    """Composite ADP skips NaN values and falls through to next source."""
    row = {"yahoo_adp": float("nan"), "fantasypros_adp": 10.0}
    assert compute_composite_adp(row) == 10.0


def test_composite_adp_skips_zero():
    """Composite ADP skips zero values (invalid ADP)."""
    row = {"yahoo_adp": 0.0, "fantasypros_adp": 0, "nfbc_adp": 15.0}
    assert compute_composite_adp(row) == 15.0


def test_composite_adp_skips_negative():
    """Composite ADP skips negative values."""
    row = {"yahoo_adp": -5.0, "fantasypros_adp": 10.0}
    assert compute_composite_adp(row) == 10.0


def test_composite_adp_partial_keys():
    """Composite ADP works with only some keys present."""
    row = {"nfbc_adp": 22.5}
    assert compute_composite_adp(row) == 22.5
