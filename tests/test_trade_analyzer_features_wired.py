"""Structural-invariant guard: Trade Analyzer page wires Features 1/2/3.

UI wiring task (2026-05-23): pages/11_Trade_Analyzer.py must invoke
evaluate_trade() with the new opt-in flags for the IP-floor penalty
(Feature 1, always-on as a passive engine output), weekly H2H matrix
(Feature 2), and playoff/championship sim (Feature 3).

This test guards the wiring contract — if the flags are removed or
renamed, the new outputs silently disappear from the UI and the user
loses the report's primary objective (Δ title-odds).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


@pytest.fixture(scope="module")
def page_source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_calls_evaluate_trade_with_enable_weekly_matrix(page_source: str) -> None:
    """evaluate_trade(... enable_weekly_matrix=...) must be present."""
    assert "enable_weekly_matrix=" in page_source, (
        "Trade Analyzer page must opt into the weekly H2H matrix via "
        "enable_weekly_matrix=. Removed wiring drops Feature 2 from the UI."
    )


def test_page_calls_evaluate_trade_with_enable_playoff_sim(page_source: str) -> None:
    """evaluate_trade(... enable_playoff_sim=...) must be present."""
    assert "enable_playoff_sim=" in page_source, (
        "Trade Analyzer page must opt into the playoff/championship sim via "
        "enable_playoff_sim=. Removed wiring drops Feature 3 — the report's "
        "PRIMARY objective per Q(a) — from the UI."
    )


def test_page_loads_weekly_schedule(page_source: str) -> None:
    """The page must source weekly_schedule from load_league_schedule (Yahoo)."""
    assert "load_league_schedule" in page_source, (
        "Trade Analyzer must call load_league_schedule() to wire the real "
        "Yahoo schedule into Feature 2 + 3. Without it, the matrix and "
        "playoff sim run on neutral defaults."
    )
    assert "weekly_schedule=" in page_source, "evaluate_trade() must receive weekly_schedule= for Features 2/3 to fire."


def test_page_loads_league_rosters(page_source: str) -> None:
    """The page must load league_rosters from the DB for opponent strength."""
    assert re.search(r"FROM\s+league_rosters", page_source, re.IGNORECASE), (
        "Trade Analyzer must query league_rosters table for Feature 3 opponents."
    )
    assert "league_rosters=" in page_source, "evaluate_trade() must receive league_rosters= for Features 2/3."


def test_page_loads_current_wins(page_source: str) -> None:
    """The page must build current_wins from standings for Feature 3."""
    assert "current_wins=" in page_source, (
        "evaluate_trade() must receive current_wins= for Feature 3 "
        "(playoff sim starts from current standings position)."
    )


def test_page_renders_playoff_prob_metric(page_source: str) -> None:
    """The page must surface delta_playoff_prob in the UI."""
    assert "delta_playoff_prob" in page_source, (
        "Trade Analyzer must render delta_playoff_prob — the report's PRIMARY objective."
    )


def test_page_renders_champ_prob_metric(page_source: str) -> None:
    """The page must surface delta_champ_prob in the UI."""
    assert "delta_champ_prob" in page_source, (
        "Trade Analyzer must render delta_champ_prob — the report's championship-odds metric."
    )


def test_page_renders_ip_floor_status(page_source: str) -> None:
    """The page must surface ip_floor_detail / IP floor warnings."""
    assert "ip_floor_detail" in page_source or "ip_floor_penalty" in page_source, (
        "Trade Analyzer must surface IP-floor status — without it, users miss the Yahoo 20 IP/week forfeit risk."
    )


def test_page_renders_weekly_matrix(page_source: str) -> None:
    """The page must surface the weekly_matrix output (table or heatmap)."""
    assert "weekly_matrix" in page_source, (
        "Trade Analyzer must render the weekly_matrix output — without it, "
        "users miss the per-week win-probability breakdown."
    )


def test_page_syntax_valid(page_source: str) -> None:
    """The page must parse as valid Python."""
    import ast

    ast.parse(page_source)
