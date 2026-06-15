"""Unit tests for src/lineup_optimizer_helpers.py — pure helpers extracted from
pages/2_Line-up_Optimizer.py.

Each helper is tested in isolation without Streamlit, DB, or Yahoo dependencies.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.lineup_optimizer_helpers import (
    decision_row_classes,
    expand_skeleton,
    fmt_deltas,
    slot_sort_key,
)

# ── fmt_deltas ────────────────────────────────────────────────────────────────


class TestFmtDeltas:
    def test_empty_dict_returns_em_dash(self):
        assert fmt_deltas({}) == "—"

    def test_single_positive_entry(self):
        result = fmt_deltas({"hr": 2.0})
        assert "HR" in result
        assert "+2.00" in result

    def test_single_negative_entry(self):
        result = fmt_deltas({"era": -0.5})
        assert "ERA" in result
        assert "-0.50" in result

    def test_sorted_by_abs_magnitude_descending(self):
        d = {"sb": 0.1, "hr": 3.0, "avg": 0.5}
        result = fmt_deltas(d)
        # HR should appear before AVG before SB
        assert result.index("HR") < result.index("AVG") < result.index("SB")

    def test_multiple_entries_comma_separated(self):
        d = {"k": 1.0, "w": 2.0}
        result = fmt_deltas(d)
        assert "," in result

    def test_keys_uppercased(self):
        result = fmt_deltas({"avg": 0.01})
        assert "AVG" in result
        assert "avg" not in result

    def test_format_precision_two_decimal_places(self):
        result = fmt_deltas({"rbi": 1.234})
        # Should round to 2 decimal places
        assert "1.23" in result or "1.23" in result

    def test_zero_value_formatted(self):
        result = fmt_deltas({"sv": 0.0})
        assert "SV" in result
        assert "+0.00" in result


# ── decision_row_classes ──────────────────────────────────────────────────────


class TestDecisionRowClasses:
    def _make_df(self, decisions: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"Decision": decisions})

    def test_start_maps_to_row_start(self):
        df = self._make_df(["START"])
        result = decision_row_classes(df)
        assert result[0] == "row-start"

    def test_start_warning_maps_to_row_start_forced(self):
        df = self._make_df(["START ⚠"])
        result = decision_row_classes(df)
        assert result[0] == "row-start-forced"

    def test_bench_maps_to_row_bench(self):
        df = self._make_df(["BENCH"])
        result = decision_row_classes(df)
        assert result[0] == "row-bench"

    def test_il_maps_to_row_il(self):
        df = self._make_df(["IL"])
        result = decision_row_classes(df)
        assert result[0] == "row-il"

    def test_leave_empty_maps_to_row_empty(self):
        df = self._make_df(["LEAVE EMPTY"])
        result = decision_row_classes(df)
        assert result[0] == "row-empty"

    def test_unknown_maps_to_row_bench(self):
        df = self._make_df(["SOMETHING_ELSE"])
        result = decision_row_classes(df)
        assert result[0] == "row-bench"

    def test_multiple_rows_all_mapped(self):
        decisions = ["START", "BENCH", "IL", "LEAVE EMPTY", "START ⚠"]
        df = self._make_df(decisions)
        result = decision_row_classes(df)
        assert len(result) == 5
        assert result[0] == "row-start"
        assert result[1] == "row-bench"
        assert result[2] == "row-il"
        assert result[3] == "row-empty"
        assert result[4] == "row-start-forced"

    def test_empty_df_returns_empty_dict(self):
        df = pd.DataFrame({"Decision": []})
        assert decision_row_classes(df) == {}

    def test_missing_decision_col_defaults_to_bench(self):
        """When the DataFrame has no Decision column, each row gets row-bench."""
        df = pd.DataFrame({"Player": ["Alice", "Bob"]})
        result = decision_row_classes(df)
        assert result == {0: "row-bench", 1: "row-bench"}

    def test_returns_dict_with_int_keys(self):
        df = self._make_df(["START", "BENCH"])
        result = decision_row_classes(df)
        assert all(isinstance(k, int) for k in result)

    def test_all_start_variants_get_row_start(self):
        # Any string starting with "START" but without ⚠ → row-start
        df = self._make_df(["START", "STARTS", "START_ALT"])
        result = decision_row_classes(df)
        for k in result:
            assert result[k] == "row-start"


# ── slot_sort_key ─────────────────────────────────────────────────────────────


class TestSlotSortKey:
    _ORDER = {"C": 0, "1B": 1, "OF": 2, "BN": 3}

    def _make_df(self, slots: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"selected_position": slots})

    def test_known_slot_returns_correct_priority(self):
        df = self._make_df(["C"])
        result = slot_sort_key(df, self._ORDER)
        assert result.iloc[0] == 0

    def test_unknown_slot_returns_max_plus_one(self):
        df = self._make_df(["ZZ"])
        result = slot_sort_key(df, self._ORDER)
        assert result.iloc[0] == max(self._ORDER.values()) + 1

    def test_multiple_rows_sorted_correctly(self):
        df = self._make_df(["OF", "C", "BN"])
        result = slot_sort_key(df, self._ORDER)
        assert list(result) == [2, 0, 3]

    def test_returns_series_with_same_length_as_input(self):
        df = self._make_df(["C", "1B", "OF", "BN"])
        result = slot_sort_key(df, self._ORDER)
        assert len(result) == 4

    def test_empty_df_returns_empty_series(self):
        df = self._make_df([])
        result = slot_sort_key(df, self._ORDER)
        assert len(result) == 0

    def test_sort_using_key_gives_correct_order(self):
        df = self._make_df(["BN", "C", "OF", "1B"])
        key_series = slot_sort_key(df, self._ORDER)
        df["_key"] = key_series
        sorted_df = df.sort_values("_key")
        assert list(sorted_df["selected_position"]) == ["C", "1B", "OF", "BN"]


# ── expand_skeleton ───────────────────────────────────────────────────────────


class TestExpandSkeleton:
    def test_single_slot_count_one(self):
        result = expand_skeleton({"C": (1, ["C"])})
        assert result == ["C"]

    def test_multi_count_slot_repeated(self):
        result = expand_skeleton({"OF": (3, ["OF", "LF", "CF", "RF"])})
        assert result == ["OF", "OF", "OF"]

    def test_multiple_slots(self):
        slots = {"C": (1, ["C"]), "1B": (1, ["1B"]), "OF": (2, ["OF"])}
        result = expand_skeleton(slots)
        assert result == ["C", "1B", "OF", "OF"]

    def test_empty_dict_returns_empty_list(self):
        assert expand_skeleton({}) == []

    def test_count_zero_slot_not_included(self):
        result = expand_skeleton({"SP": (0, ["SP"])})
        assert result == []

    def test_typical_hitter_slots(self):
        hitter_slots = {
            "C": (1, ["C"]),
            "1B": (1, ["1B"]),
            "2B": (1, ["2B"]),
            "3B": (1, ["3B"]),
            "SS": (1, ["SS"]),
            "OF": (3, ["OF"]),
            "Util": (2, ["C", "1B", "2B", "3B", "SS", "OF"]),
        }
        result = expand_skeleton(hitter_slots)
        assert result.count("C") == 1
        assert result.count("OF") == 3
        assert result.count("Util") == 2
        assert len(result) == 10

    def test_typical_pitcher_slots(self):
        pitcher_slots = {"SP": (2, ["SP"]), "RP": (2, ["RP"]), "P": (4, ["SP", "RP", "P"])}
        result = expand_skeleton(pitcher_slots)
        assert result.count("SP") == 2
        assert result.count("RP") == 2
        assert result.count("P") == 4
        assert len(result) == 8

    def test_eligible_codes_not_included_in_output(self):
        """expand_skeleton must only return slot names, not eligible position codes."""
        result = expand_skeleton({"OF": (2, ["LF", "CF", "RF"])})
        assert result == ["OF", "OF"]
        assert "LF" not in result
        assert "CF" not in result
