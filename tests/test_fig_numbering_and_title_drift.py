"""Structural guard: FIG numbering + nav/title drift (Task 5.2).

Checks three pages for three separate invariants:

1. pages/4_Pitcher_Streaming.py  — eyebrow must use two-digit zero-padded
   `FIG.04` (not the bare `FIG.4` that existed before the fix).

2. pages/5_Matchup_Planner.py   — no stray `FIG.02` sub-caption on a page
   whose own eyebrow is `FIG.05`.

3. pages/10_Punt_Analyzer.py    — the page-header title must be "Punt Analyzer"
   (matching the sidebar nav label), not the old "Punt Strategy Simulator".
"""

from __future__ import annotations

from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def _text(stem: str) -> str:
    return (PAGES_DIR / stem).read_text(encoding="utf-8")


def test_pitcher_streaming_fig_zero_padded():
    """FIG.04 (two-digit) must appear; bare FIG.4 must not."""
    text = _text("4_Pitcher_Streaming.py")
    assert "FIG.04" in text, "pages/4_Pitcher_Streaming.py: expected 'FIG.04' but not found"
    assert "FIG.4 " not in text and '"FIG.4 ' not in text and "'FIG.4 " not in text, (
        "pages/4_Pitcher_Streaming.py: bare 'FIG.4 ' (unpadded) still present — should have been replaced with 'FIG.04'"
    )


def test_matchup_planner_no_stale_fig02_subcaption():
    """No fig_label containing 'FIG.02' on the FIG.05 Matchup Planner page."""
    text = _text("5_Matchup_Planner.py")
    assert "FIG.02" not in text, (
        "pages/5_Matchup_Planner.py: stale 'FIG.02' sub-caption found — "
        "should have been updated to 'FIG.05' to match the page eyebrow"
    )


def test_punt_analyzer_title_matches_nav():
    """Page-header title must be 'Punt Analyzer', not 'Punt Strategy Simulator'."""
    text = _text("10_Punt_Analyzer.py")
    assert "Punt Strategy Simulator" not in text, (
        "pages/10_Punt_Analyzer.py: old title 'Punt Strategy Simulator' still present — "
        "should be 'Punt Analyzer' to match the sidebar nav label"
    )
    assert "Punt Analyzer" in text, "pages/10_Punt_Analyzer.py: expected title 'Punt Analyzer' not found"
