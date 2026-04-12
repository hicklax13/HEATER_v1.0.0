"""Tests for S1 (Trade Finder tab merge), S5 (standings_utils), V6 (format_stat)."""

from __future__ import annotations

import pandas as pd

# ── S1: Trade Finder has 4 tabs, not 5 ──────────────────────────────


class TestTradeFinderTabStructure:
    """Verify the Trade Finder page defines exactly 4 tabs."""

    def test_trade_finder_has_four_tabs(self):
        """The st.tabs call should list exactly 4 tab labels."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Finder.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        tab_labels: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match st.tabs(...)
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "tabs"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "st"
                ):
                    if node.args and isinstance(node.args[0], ast.List):
                        for elt in node.args[0].elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                tab_labels.append(elt.value)

        assert len(tab_labels) == 4, f"Expected 4 tabs, got {len(tab_labels)}: {tab_labels}"

    def test_tab_names(self):
        """Verify the exact tab names after merge."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Finder.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        tab_labels: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "tabs"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "st"
                ):
                    if node.args and isinstance(node.args[0], ast.List):
                        for elt in node.args[0].elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                tab_labels.append(elt.value)

        expected = ["Trade Recommendations", "Target a Player", "Browse Partners", "Trade Readiness"]
        assert tab_labels == expected, f"Tab labels mismatch: {tab_labels}"

    def test_no_smart_recommendations_tab(self):
        """The old 'Smart Recommendations' tab name should not appear in st.tabs."""
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Finder.py"
        content = src.read_text(encoding="utf-8")

        # Should NOT appear as a tab label (it can appear in comments/docstrings)
        assert '"Smart Recommendations"' not in content or "st.tabs" not in content.split('"Smart Recommendations"')[0]

    def test_no_by_value_tab(self):
        """The old 'By Value' tab name should not appear in st.tabs."""
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Finder.py"
        content = src.read_text(encoding="utf-8")

        assert '"By Value"' not in content


# ── S5: standings_utils caching ─────────────────────────────────────


class TestStandingsUtils:
    """Test get_all_team_totals caching behavior."""

    def _make_pool(self) -> pd.DataFrame:
        """Create a minimal player pool for testing."""
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Player A",
                    "is_hitter": 1,
                    "r": 80,
                    "hr": 25,
                    "rbi": 70,
                    "sb": 10,
                    "avg": 0.280,
                    "obp": 0.350,
                    "ab": 500,
                    "h": 140,
                    "bb": 50,
                    "hbp": 5,
                    "sf": 3,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "ip": 0,
                    "er": 0,
                    "era": 0,
                    "whip": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "name": "Player B",
                    "is_hitter": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "ab": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "w": 12,
                    "l": 5,
                    "sv": 0,
                    "k": 180,
                    "ip": 180,
                    "er": 60,
                    "era": 3.00,
                    "whip": 1.10,
                    "bb_allowed": 50,
                    "h_allowed": 150,
                },
            ]
        )

    def test_returns_dict_for_all_teams(self):
        from src.standings_utils import clear_cache, get_all_team_totals

        clear_cache()
        pool = self._make_pool()
        rosters = {"Team A": [1], "Team B": [2]}
        result = get_all_team_totals(rosters, pool)

        assert isinstance(result, dict)
        assert "Team A" in result
        assert "Team B" in result
        assert result["Team A"]["HR"] == 25
        assert result["Team B"]["K"] == 180
        clear_cache()

    def test_caching_returns_same_object(self):
        from src.standings_utils import clear_cache, get_all_team_totals

        clear_cache()
        pool = self._make_pool()
        rosters = {"Team A": [1]}
        first = get_all_team_totals(rosters, pool)
        second = get_all_team_totals(rosters, pool)
        assert first is second  # Same object, not a copy
        clear_cache()

    def test_clear_cache_forces_recomputation(self):
        from src.standings_utils import clear_cache, get_all_team_totals

        clear_cache()
        pool = self._make_pool()
        rosters = {"Team A": [1]}
        first = get_all_team_totals(rosters, pool)
        clear_cache()
        second = get_all_team_totals(rosters, pool)
        assert first is not second  # Different object after cache clear
        assert first == second  # But same values
        clear_cache()

    def test_force_refresh_recomputes(self):
        from src.standings_utils import clear_cache, get_all_team_totals

        clear_cache()
        pool = self._make_pool()
        rosters = {"Team A": [1]}
        first = get_all_team_totals(rosters, pool)
        second = get_all_team_totals(rosters, pool, force_refresh=True)
        assert first is not second
        clear_cache()


# ── V6: format_stat ─────────────────────────────────────────────────


class TestFormatStat:
    """Test unified stat formatting."""

    def test_avg_formats_3_decimals(self):
        from src.ui_shared import format_stat

        assert format_stat(0.289, "AVG") == ".289"

    def test_obp_formats_3_decimals(self):
        from src.ui_shared import format_stat

        assert format_stat(0.350, "OBP") == ".350"

    def test_avg_above_1(self):
        from src.ui_shared import format_stat

        # Edge case: AVG > 1 (shouldn't happen but test the branch)
        result = format_stat(1.000, "AVG")
        assert result == "1.000"

    def test_era_formats_2_decimals(self):
        from src.ui_shared import format_stat

        assert format_stat(3.5, "ERA") == "3.50"

    def test_whip_formats_2_decimals(self):
        from src.ui_shared import format_stat

        assert format_stat(1.1, "WHIP") == "1.10"

    def test_sgp_shows_positive_sign(self):
        from src.ui_shared import format_stat

        assert format_stat(1.5, "SGP") == "+1.50"

    def test_sgp_shows_negative_sign(self):
        from src.ui_shared import format_stat

        assert format_stat(-0.75, "SGP") == "-0.75"

    def test_counting_stat_integer(self):
        from src.ui_shared import format_stat

        assert format_stat(25.0, "HR") == "25"

    def test_counting_stat_fractional(self):
        from src.ui_shared import format_stat

        assert format_stat(25.5, "IP") == "25.5"

    def test_percentage_format(self):
        from src.ui_shared import format_stat

        assert format_stat(22.5, "PCT") == "22.5%"

    def test_case_insensitive_avg(self):
        from src.ui_shared import format_stat

        assert format_stat(0.300, "avg") == ".300"

    def test_case_insensitive_era(self):
        from src.ui_shared import format_stat

        assert format_stat(4.00, "era") == "4.00"
