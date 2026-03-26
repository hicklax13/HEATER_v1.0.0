"""Tests for build_compact_table_html() — ESPN-style compact table rendering."""

import pandas as pd
import pytest

from src.ui_shared import (
    HITTING_STAT_COLS,
    PITCHING_STAT_COLS,
    THEME,
    build_compact_table_html,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def hitter_df():
    """A small hitter DataFrame with hitting stat columns."""
    return pd.DataFrame(
        {
            "Name": ["Mike Trout", "Aaron Judge"],
            "R": [80, 95],
            "HR": [30, 45],
            "RBI": [75, 110],
            "SB": [5, 3],
            "AVG": [0.280, 0.305],
            "OBP": [0.370, 0.410],
        }
    )


@pytest.fixture
def pitcher_df():
    """A small pitcher DataFrame with pitching stat columns."""
    return pd.DataFrame(
        {
            "Name": ["Gerrit Cole", "Spencer Strider"],
            "W": [15, 12],
            "L": [5, 4],
            "SV": [0, 0],
            "K": [220, 260],
            "ERA": [2.95, 3.10],
            "WHIP": [1.05, 0.98],
        }
    )


@pytest.fixture
def mixed_df():
    """A DataFrame with both hitting and pitching columns."""
    return pd.DataFrame(
        {
            "Name": ["Shohei Ohtani"],
            "R": [100],
            "HR": [40],
            "W": [10],
            "K": [180],
            "ERA": [3.00],
            "AVG": [0.290],
        }
    )


@pytest.fixture
def health_df():
    """A DataFrame with a health status column."""
    return pd.DataFrame(
        {
            "Name": ["Player A", "Player B", "Player C", "Player D"],
            "Health": ["Healthy", "Day-to-Day", "IL", "Out"],
            "HR": [20, 15, 10, 5],
        }
    )


# ── Basic Structure ───────────────────────────────────────────────


class TestBasicStructure:
    def test_returns_string(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert isinstance(result, str)

    def test_contains_wrapper_div(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert 'class="compact-table-wrap"' in result

    def test_contains_table(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert 'class="compact-table"' in result

    def test_contains_thead_and_tbody(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "<thead>" in result
        assert "<tbody>" in result

    def test_correct_row_count(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        # 2 data rows in tbody
        assert result.count("<tr>") == 3  # 1 header + 2 data
        assert result.count("</tr>") == 3

    def test_correct_column_count(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        # 7 columns: Name + R + HR + RBI + SB + AVG + OBP
        # Use "</th>" to avoid matching "<thead>"
        assert result.count("</th>") == len(hitter_df.columns)

    def test_max_height_applied(self, hitter_df):
        result = build_compact_table_html(hitter_df, max_height=300)
        assert "max-height:300px" in result

    def test_max_height_zero_no_style(self, hitter_df):
        result = build_compact_table_html(hitter_df, max_height=0)
        assert "max-height" not in result


# ── Empty / None Handling ─────────────────────────────────────────


class TestEmptyHandling:
    def test_empty_dataframe(self):
        result = build_compact_table_html(pd.DataFrame())
        assert "No data available" in result
        assert 'class="compact-table-wrap"' in result

    def test_none_dataframe(self):
        result = build_compact_table_html(None)
        assert "No data available" in result

    def test_empty_still_has_wrapper(self):
        result = build_compact_table_html(pd.DataFrame())
        assert 'class="compact-table-wrap"' in result


# ── Header Classes (th-hit / th-pit) ─────────────────────────────


class TestHeaderClasses:
    def test_hitting_columns_get_th_hit(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "th-hit" in result

    def test_pitching_columns_get_th_pit(self, pitcher_df):
        result = build_compact_table_html(pitcher_df)
        assert "th-pit" in result

    def test_mixed_has_both_classes(self, mixed_df):
        result = build_compact_table_html(mixed_df)
        assert "th-hit" in result
        assert "th-pit" in result

    def test_name_column_no_stat_class(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        # The Name column header should not have th-hit or th-pit
        # It should have col-name instead
        # Check that the first th has col-name
        assert 'class="col-name"' in result

    def test_custom_highlight_cols(self, hitter_df):
        custom = {"R": "th-pit", "HR": "th-pit"}
        result = build_compact_table_html(hitter_df, highlight_cols=custom)
        # R and HR should be th-pit, not th-hit
        assert "th-pit" in result
        # AVG should NOT have any class (custom overrides auto-detect)
        # since highlight_cols was fully specified, only R and HR have classes

    def test_auto_detect_case_insensitive(self):
        df = pd.DataFrame({"Name": ["A"], "r": [10], "era": [3.0]})
        result = build_compact_table_html(df)
        assert "th-hit" in result
        assert "th-pit" in result


# ── Sticky Name Column ───────────────────────────────────────────


class TestStickyColumn:
    def test_first_column_is_sticky(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "col-name" in result

    def test_col_name_in_header_and_body(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        # Should appear in th (header) and td (body cells)
        count = result.count("col-name")
        # 1 in header + 2 in body rows = 3
        assert count >= 3

    def test_first_col_value_appears(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "Mike Trout" in result
        assert "Aaron Judge" in result


# ── Health Dot ────────────────────────────────────────────────────


class TestHealthDot:
    def test_health_dot_rendered(self, health_df):
        result = build_compact_table_html(health_df, health_col="Health")
        assert 'class="health-dot"' in result

    def test_healthy_gets_green_dot(self, health_df):
        result = build_compact_table_html(health_df, health_col="Health")
        assert THEME["green"] in result

    def test_day_to_day_gets_warn_dot(self, health_df):
        result = build_compact_table_html(health_df, health_col="Health")
        assert THEME["warn"] in result

    def test_il_gets_danger_dot(self, health_df):
        result = build_compact_table_html(health_df, health_col="Health")
        assert THEME["danger"] in result

    def test_four_dots_for_four_players(self, health_df):
        result = build_compact_table_html(health_df, health_col="Health")
        assert result.count("health-dot") == 4

    def test_no_health_col_no_dots(self, health_df):
        result = build_compact_table_html(health_df)
        assert "health-dot" not in result


# ── Row Classes ───────────────────────────────────────────────────


class TestRowClasses:
    def test_row_start_class(self, hitter_df):
        classes = {0: "row-start", 1: "row-bench"}
        result = build_compact_table_html(hitter_df, row_classes=classes)
        assert "row-start" in result
        assert "row-bench" in result

    def test_no_row_classes_by_default(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "row-start" not in result
        assert "row-bench" not in result

    def test_partial_row_classes(self, hitter_df):
        classes = {0: "row-start"}
        result = build_compact_table_html(hitter_df, row_classes=classes)
        assert "row-start" in result
        # Second row should NOT have a class attr
        # Count class="row- occurrences
        assert result.count('class="row-') == 1


# ── Numeric Formatting ────────────────────────────────────────────


class TestNumericFormatting:
    def test_rate_stat_three_decimals(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        assert "0.280" in result
        assert "0.305" in result

    def test_counting_stat_integers(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        # HR=30 should appear as "30", not "30.0"
        assert ">30<" in result
        assert ">45<" in result

    def test_era_three_decimals(self, pitcher_df):
        result = build_compact_table_html(pitcher_df)
        assert "2.950" in result
        assert "3.100" in result

    def test_whip_three_decimals(self, pitcher_df):
        result = build_compact_table_html(pitcher_df)
        assert "1.050" in result
        assert "0.980" in result

    def test_non_numeric_passthrough(self):
        df = pd.DataFrame({"Name": ["A"], "Notes": ["Some text"]})
        result = build_compact_table_html(df)
        assert "Some text" in result


# ── Data Integrity ────────────────────────────────────────────────


class TestDataIntegrity:
    def test_all_player_names_present(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        for name in hitter_df["Name"]:
            assert name in result

    def test_all_column_headers_present(self, hitter_df):
        result = build_compact_table_html(hitter_df)
        for col in hitter_df.columns:
            assert f">{col}<" in result

    def test_single_row_df(self):
        df = pd.DataFrame({"Name": ["Solo"], "HR": [10]})
        result = build_compact_table_html(df)
        assert "Solo" in result
        assert result.count("<tr>") == 2  # 1 header + 1 data

    def test_many_columns(self):
        data = {"Name": ["X"]}
        for col in list(HITTING_STAT_COLS) + list(PITCHING_STAT_COLS):
            data[col] = [10]
        df = pd.DataFrame(data)
        result = build_compact_table_html(df)
        # Name + 6 hitting + 6 pitching = 13 columns
        assert result.count("</th>") == len(df.columns)

    def test_none_values_handled(self):
        df = pd.DataFrame({"Name": ["A"], "HR": [None], "AVG": [None]})
        result = build_compact_table_html(df)
        assert "A" in result
        # Should not crash

    def test_nan_values_render_empty(self):
        df = pd.DataFrame({"Name": ["A"], "HR": [float("nan")], "AVG": [float("nan")]})
        result = build_compact_table_html(df)
        assert "nan" not in result.lower()
        assert "A" in result

    def test_inf_values_render_empty(self):
        df = pd.DataFrame({"Name": ["A"], "ERA": [float("inf")]})
        result = build_compact_table_html(df)
        assert "inf" not in result.lower()

    def test_highlight_cols_as_list(self):
        df = pd.DataFrame({"Name": ["A"], "Value": [10.5], "Impact": [3.2]})
        result = build_compact_table_html(df, highlight_cols=["Value", "Impact"])
        assert "th-hit" in result
        # Should not crash (list is normalized to dict)

    def test_highlight_cols_as_set(self):
        df = pd.DataFrame({"Name": ["A"], "R": [10], "W": [5]})
        result = build_compact_table_html(df, highlight_cols={"R", "W"})
        assert "th-hit" in result

    def test_health_col_substring_no_false_match(self):
        """Column named 'H' should still be formatted numerically when health_col='Health'."""
        df = pd.DataFrame({"Name": ["A"], "H": [150], "Health": ["Healthy"]})
        result = build_compact_table_html(df, health_col="Health")
        assert ">150<" in result


# ── Headshot Thumbnails ──────────────────────────────────────────


class TestHeadshotThumbnails:
    def test_mlb_id_column_renders_headshots(self):
        """When mlb_id column exists with valid ID, MLB headshot URL should appear."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "mlb_id": [660271]})
        result = build_compact_table_html(df)
        assert "img.mlbstatic.com" in result
        assert "660271" in result

    def test_mlb_id_column_hidden_from_display(self):
        """mlb_id column should not appear as a visible header."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "mlb_id": [660271]})
        result = build_compact_table_html(df)
        assert ">mlb_id<" not in result

    def test_player_id_column_hidden_from_display(self):
        """player_id column should not appear as a visible header."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "player_id": [123]})
        result = build_compact_table_html(df)
        assert ">player_id<" not in result

    def test_no_mlb_id_no_headshots(self):
        """Without mlb_id column and show_avatars=False, no avatars should render."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40]})
        result = build_compact_table_html(df)
        assert "img.mlbstatic.com" not in result
        assert "<img" not in result

    def test_null_mlb_id_shows_fallback_avatar(self):
        """Null mlb_id values should show the fallback avatar SVG, not MLB headshot."""
        df = pd.DataFrame({"Name": ["Unknown"], "HR": [5], "mlb_id": [None]})
        result = build_compact_table_html(df)
        assert "img.mlbstatic.com" not in result
        # Should still show an <img> with the fallback SVG
        assert "<img" in result

    def test_zero_mlb_id_shows_fallback_avatar(self):
        """Zero mlb_id should show fallback avatar, not MLB headshot."""
        df = pd.DataFrame({"Name": ["Unknown"], "HR": [5], "mlb_id": [0]})
        result = build_compact_table_html(df)
        assert "img.mlbstatic.com" not in result
        assert "<img" in result

    def test_headshot_has_onerror_fallback(self):
        """Headshot images should swap to fallback SVG on load error."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "mlb_id": [660271]})
        result = build_compact_table_html(df)
        assert "onerror" in result

    def test_headshot_with_multiple_players(self):
        """Each player row should get its own headshot."""
        df = pd.DataFrame(
            {
                "Name": ["Ohtani", "Judge", "Acuna"],
                "HR": [40, 45, 35],
                "mlb_id": [660271, 592450, 660670],
            }
        )
        result = build_compact_table_html(df)
        assert "660271" in result
        assert "592450" in result
        assert "660670" in result

    def test_player_names_still_visible_with_headshots(self):
        """Player names should still be in the output alongside headshots."""
        df = pd.DataFrame({"Name": ["Ohtani", "Judge"], "HR": [40, 45], "mlb_id": [660271, 592450]})
        result = build_compact_table_html(df)
        assert "Ohtani" in result
        assert "Judge" in result

    def test_headshot_uses_lazy_loading(self):
        """Headshot images should use lazy loading for performance."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "mlb_id": [660271]})
        result = build_compact_table_html(df)
        assert 'loading="lazy"' in result

    def test_show_avatars_explicit_true(self):
        """show_avatars=True should render avatar circles even without mlb_id."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40]})
        result = build_compact_table_html(df, show_avatars=True)
        assert "<img" in result

    def test_show_avatars_explicit_false(self):
        """show_avatars=False should suppress avatars even with mlb_id."""
        df = pd.DataFrame({"Name": ["Ohtani"], "HR": [40], "mlb_id": [660271]})
        result = build_compact_table_html(df, show_avatars=False)
        assert "<img" not in result
