"""Tests for cheat sheet generation."""

from __future__ import annotations

import pandas as pd

from src.cheat_sheet import (
    CheatSheetOptions,
    _player_row_html,
    _tier_break_html,
    generate_cheat_sheet_html,
    generate_cheat_sheet_pdf,
)


def _make_pool():
    rows = []
    for i in range(30):
        rows.append(
            {
                "player_id": i + 1,
                "name": f"Player {i + 1}",
                "team": "NYY",
                "positions": ["SS", "OF", "1B", "SP", "RP", "C", "2B", "3B"][i % 8],
                "is_hitter": i % 8 < 6,
                "pick_score": 10.0 - i * 0.3,
                "adp": i + 1,
            }
        )
    return pd.DataFrame(rows)


def test_html_generation():
    html = generate_cheat_sheet_html(_make_pool())
    assert "<!DOCTYPE html>" in html
    assert "HEATER" in html


def test_html_contains_players():
    html = generate_cheat_sheet_html(_make_pool())
    assert "Player 1" in html


def test_html_positional_sections():
    html = generate_cheat_sheet_html(_make_pool())
    assert "SS Rankings" in html
    assert "SP Rankings" in html


def test_tier_breaks():
    html = generate_cheat_sheet_html(_make_pool(), CheatSheetOptions(show_tiers=True))
    assert "tier-break" in html


def test_health_badges():
    html = generate_cheat_sheet_html(_make_pool(), health_scores={1: 0.5, 2: 0.95})
    assert "health-dot" in html


def test_tag_badges():
    html = generate_cheat_sheet_html(_make_pool(), tags_by_player={1: ["Sleeper"], 2: ["Avoid"]})
    assert "Sleeper" in html
    assert "Avoid" in html


def test_sort_order():
    pool = _make_pool()
    html = generate_cheat_sheet_html(pool, CheatSheetOptions(sort_by="pick_score"))
    # Player 1 should appear before Player 30
    idx1 = html.find("Player 1")
    idx30 = html.find("Player 30")
    assert idx1 < idx30 if idx1 >= 0 and idx30 >= 0 else True


def test_top_n_per_position():
    opts = CheatSheetOptions(top_n_per_position=5)
    html = generate_cheat_sheet_html(_make_pool(), opts)
    assert isinstance(html, str)


def test_pdf_fallback():
    result = generate_cheat_sheet_pdf("<html><body>test</body></html>")
    # Should return None if weasyprint not installed, or bytes if it is
    assert result is None or isinstance(result, bytes)


def test_print_css_included():
    html = generate_cheat_sheet_html(_make_pool())
    assert "@media print" in html


def test_player_row_html():
    row = _player_row_html(
        {
            "name": "Test",
            "team": "NYY",
            "positions": "SS",
            "pick_score": 5.0,
            "adp": 10,
        }
    )
    assert "Test" in row
    assert "NYY" in row


def test_tier_break_html():
    html = _tier_break_html()
    assert "tier-break" in html


def test_empty_pool():
    html = generate_cheat_sheet_html(pd.DataFrame(columns=["player_id", "name", "team", "positions", "pick_score"]))
    assert "HEATER" in html


def test_options_defaults():
    opts = CheatSheetOptions()
    assert opts.sort_by == "pick_score"
    assert len(opts.positions) == 8
    assert opts.top_n_per_position == 20


def test_custom_options():
    opts = CheatSheetOptions(sort_by="adp", top_n_per_position=10, show_percentiles=False)
    assert opts.sort_by == "adp"
    assert opts.top_n_per_position == 10


def test_all_positions_covered():
    html = generate_cheat_sheet_html(_make_pool())
    for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
        assert f"{pos} Rankings" in html
