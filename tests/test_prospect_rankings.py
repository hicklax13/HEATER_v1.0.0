"""Tests for prospect rankings."""

from __future__ import annotations

from src.ecr import fetch_prospect_rankings, filter_prospects_by_position


def test_fetch_returns_dataframe():
    df = fetch_prospect_rankings()
    assert len(df) > 0
    assert list(df.columns) == ["rank", "name", "team", "position", "eta", "fv"]


def test_top_n_limit():
    df = fetch_prospect_rankings(top_n=5)
    assert len(df) == 5


def test_top_n_exceeds_available():
    df = fetch_prospect_rankings(top_n=500)
    assert len(df) == 20  # Only 20 in the static list


def test_position_filter_sp():
    df = fetch_prospect_rankings()
    filtered = filter_prospects_by_position(df, "SP")
    assert len(filtered) > 0
    assert all("SP" in pos for pos in filtered["position"])


def test_position_filter_empty_string():
    df = fetch_prospect_rankings()
    filtered = filter_prospects_by_position(df, "")
    assert len(filtered) == len(df)


def test_position_filter_no_match():
    df = fetch_prospect_rankings()
    filtered = filter_prospects_by_position(df, "DH")
    assert len(filtered) == 0
