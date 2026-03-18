"""Tests for category heatmap grid visualization (Gap 8).

Covers: 12-row output, inverse stat color flipping, green for top 3,
red for bottom 3, and rendering without errors.
"""

import re

import pytest


def _make_all_totals(n=12):
    """Create mock all-team totals for n teams.

    Team 0 is the 'user' (best in counting, worst in inverse).
    Teams are spread so ranking is deterministic.
    """
    totals = []
    for i in range(n):
        totals.append(
            {
                "R": 100 - i * 5,  # Team 0 = 100 (best), Team 11 = 45 (worst)
                "HR": 40 - i * 2,
                "RBI": 120 - i * 6,
                "SB": 30 - i * 2,
                "AVG": 0.300 - i * 0.005,
                "OBP": 0.380 - i * 0.005,
                "W": 15 - i,
                "L": 5 + i,  # Inverse: Team 0 = 5 (best/lowest), Team 11 = 16
                "SV": 20 - i,
                "K": 200 - i * 10,
                "ERA": 3.00 + i * 0.20,  # Inverse: Team 0 = 3.00 (best)
                "WHIP": 1.00 + i * 0.05,  # Inverse: Team 0 = 1.00 (best)
            }
        )
    return totals


class TestHeatmapHas12Rows:
    """Heatmap HTML contains exactly 12 data rows (one per category)."""

    def test_heatmap_html_has_12_rows(self):
        from src.ui_shared import build_category_heatmap_html

        user_totals = _make_all_totals(12)[0]
        all_totals = _make_all_totals(12)

        html = build_category_heatmap_html(user_totals, all_totals)

        # Count <tr> tags — should be 1 header + 12 data rows = 13 total
        tr_count = html.count("<tr")
        assert tr_count == 13, f"Expected 13 <tr> tags (1 header + 12 data), got {tr_count}"

        # All 12 categories should appear
        for cat_name in [
            "Runs",
            "Home Runs",
            "Runs Batted In",
            "Stolen Bases",
            "Batting Average",
            "On-Base Percentage",
            "Wins",
            "Losses",
            "Saves",
            "Strikeouts",
            "Earned Run Average",
            "Walks + Hits per Inning Pitched",
        ]:
            assert cat_name in html, f"Missing category: {cat_name}"


class TestHeatmapInverseColorFlip:
    """Inverse stats (L, ERA, WHIP) flip color logic — low = green."""

    def test_heatmap_inverse_color_flip(self):
        from src.ui_shared import build_category_heatmap_html

        all_totals = _make_all_totals(12)
        # User team has best inverse stats (lowest L, ERA, WHIP)
        user_totals = all_totals[0]

        html = build_category_heatmap_html(user_totals, all_totals)

        # ERA row should be "Strong" (green) because user has lowest ERA
        era_match = re.search(r"Earned Run Average.*?</tr>", html, re.DOTALL)
        assert era_match is not None
        assert "Strong" in era_match.group()

        # Now test with worst inverse stats
        worst_totals = all_totals[-1]
        html_worst = build_category_heatmap_html(worst_totals, all_totals)

        era_worst = re.search(r"Earned Run Average.*?</tr>", html_worst, re.DOTALL)
        assert era_worst is not None
        assert "Weak" in era_worst.group()


class TestHeatmapGreenForTop3:
    """Categories ranked in top 25% (top 3 of 12) should show 'Strong'."""

    def test_heatmap_green_for_top_3(self):
        from src.ui_shared import build_category_heatmap_html

        all_totals = _make_all_totals(12)
        # Team 0 is best in all counting stats
        user_totals = all_totals[0]

        html = build_category_heatmap_html(user_totals, all_totals)

        # HR: user has 40, rank 1 — should be Strong (green)
        hr_match = re.search(r"Home Runs.*?</tr>", html, re.DOTALL)
        assert hr_match is not None
        assert "Strong" in hr_match.group()
        assert "#2d6a4f" in hr_match.group()  # green color


class TestHeatmapRedForBottom3:
    """Categories ranked in bottom 25% (bottom 3 of 12) should show 'Weak'."""

    def test_heatmap_red_for_bottom_3(self):
        from src.ui_shared import build_category_heatmap_html

        all_totals = _make_all_totals(12)
        # Team 11 is worst in all counting stats
        user_totals = all_totals[11]

        html = build_category_heatmap_html(user_totals, all_totals)

        # HR: user has 18, rank 12 — should be Weak (red)
        hr_match = re.search(r"Home Runs.*?</tr>", html, re.DOTALL)
        assert hr_match is not None
        assert "Weak" in hr_match.group()
        assert "#e63946" in hr_match.group()  # primary/red color


class TestHeatmapRendersWithoutError:
    """Heatmap generates valid HTML for edge cases."""

    def test_heatmap_renders_without_error(self):
        from src.ui_shared import build_category_heatmap_html

        # Empty all_totals — should return empty string
        html_empty = build_category_heatmap_html({}, [])
        assert html_empty == ""

        # Single team
        single = [
            {
                "R": 80,
                "HR": 25,
                "RBI": 90,
                "SB": 15,
                "AVG": 0.270,
                "OBP": 0.340,
                "W": 10,
                "L": 8,
                "SV": 12,
                "K": 150,
                "ERA": 3.80,
                "WHIP": 1.20,
            }
        ]
        html_single = build_category_heatmap_html(single[0], single)
        assert "heatmap-grid" in html_single
        assert "<table" in html_single

        # Normal 12-team case
        all_totals = _make_all_totals(12)
        html_normal = build_category_heatmap_html(all_totals[5], all_totals)
        assert "heatmap-grid" in html_normal
        assert html_normal.count("<tr") == 13
