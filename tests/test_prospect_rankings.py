# tests/test_prospect_rankings.py
"""Tests for prospect rankings (updated for prospect_engine)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.prospect_engine import get_prospect_rankings


def test_fetch_returns_dataframe():
    df = get_prospect_rankings()
    assert len(df) > 0
    assert "name" in df.columns


def test_top_n_limit():
    df = get_prospect_rankings(top_n=5)
    assert len(df) <= 5


def test_top_n_exceeds_available():
    df = get_prospect_rankings(top_n=500)
    assert len(df) <= 500


def test_position_filter_sp():
    df = get_prospect_rankings(position="SP")
    if len(df) > 0:
        assert all("SP" in pos for pos in df["position"])


def test_position_filter_no_match():
    df = get_prospect_rankings(position="ZZZZZ")
    assert len(df) == 0


def test_org_filter():
    df = get_prospect_rankings(org="BOS")
    if len(df) > 0:
        assert all(t == "BOS" for t in df["team"])
