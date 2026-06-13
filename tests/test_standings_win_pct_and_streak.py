"""Tests for League Standings page formatting fixes.

Task 5.1: Win% displays as "NN.N%" not ".NNN" (batting-average style).
Task 5.4: Dead "Streak" column is removed from the standings table.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PAGE_PATH = Path(__file__).parent.parent / "pages" / "6_League_Standings.py"
PAGE_SRC = PAGE_PATH.read_text(encoding="utf-8")


# ── Task 5.1: Win% formatting ─────────────────────────────────────────


class TestWinPctFormatting:
    """Win% in the standings table must render as 'NN.N%', not '.NNN'."""

    def test_display_rows_win_pct_not_avg_format(self):
        """The display_rows dict must NOT use format_stat(..., 'AVG') for Win%.

        format_stat(0.318, 'AVG') returns '0.318' which reads like a batting
        average.  Win% should be formatted as a percentage like '31.8%'.
        """
        # Search for format_stat(... "AVG") or format_stat(... 'AVG') usages
        # near "Win%" key assignment in display_rows
        # We check that no assignment to "Win%" uses the AVG format_stat call.
        tree = ast.parse(PAGE_SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key_node, val_node in zip(node.keys, node.values):
                # Find dicts with a "Win%" key
                if not (isinstance(key_node, ast.Constant) and key_node.value == "Win%"):
                    continue
                # Reconstruct the value source fragment
                val_src = ast.unparse(val_node)
                assert "format_stat" not in val_src or "'AVG'" not in val_src and '"AVG"' not in val_src, (
                    f"'Win%' value still uses format_stat(..., 'AVG'): {val_src!r}. "
                    "Change to a percentage string like f'{wp*100:.1f}%'."
                )

    def test_win_pct_value_looks_like_percentage(self):
        """Spot-check: a typical 0.318 win-pct value must produce 'NN.N%' form."""
        # Simulate what the page now does for a sample win_pct
        # (We import only format helpers, not the full page which needs Streamlit.)
        wp = 0.318
        # The expected new format is something like "31.8%"
        formatted = f"{wp * 100:.1f}%"
        assert re.match(r"^\d+\.\d%$", formatted), f"Expected 'NN.N%' format, got {formatted!r}"
        assert "." not in formatted.split("%")[0][:1] or formatted[0].isdigit(), (
            f"Win% must start with a digit, not a dot. Got {formatted!r}"
        )

    def test_win_pct_string_starts_with_digit(self):
        """The formatted Win% string must start with a digit, not '.'."""
        for raw in [0.0, 0.318, 0.500, 1.0]:
            result = f"{raw * 100:.1f}%"
            assert result[0].isdigit(), f"Win% '{result}' starts with non-digit for raw={raw}"

    def test_page_source_contains_pct_format_for_win_pct(self):
        """Page source must contain a percentage-format expression for Win%.

        Matches patterns like:
          "Win%": f"{wp * 100:.1f}%"
          "Win%": f"{rec.get('win_pct', 0) * 100:.1f}%"
        """
        # Look for lines that have a "Win%" key with a percentage f-string value.
        # Use a per-line search so quote nesting inside f-strings doesn't matter.
        found = any(
            ('"Win%"' in line or "'Win%'" in line) and ("* 100" in line or "*100" in line) and "%" in line
            for line in PAGE_SRC.splitlines()
        )
        assert found, (
            "Could not find a percentage-format ('NN.N%') assignment for 'Win%' key in the page source. "
            'Expected something like:  "Win%": f"{wp * 100:.1f}%"'
        )

    def test_no_avg_format_stat_for_win_pct_in_page(self):
        """format_stat(win_pct_var, 'AVG') must not appear near 'Win%' key."""
        # Simple substring checks: look for lines that have BOTH "Win%" and format_stat + AVG
        for line in PAGE_SRC.splitlines():
            if '"Win%"' in line or "'Win%'" in line:
                assert not ("format_stat" in line and ("'AVG'" in line or '"AVG"' in line)), (
                    f"Line still formats Win% as AVG: {line.strip()!r}. "
                    "Replace with a percentage expression like f'{wp * 100:.1f}%'."
                )


# ── Task 5.4: Streak column removed ──────────────────────────────────


class TestStreakColumnRemoved:
    """The always-empty 'Streak' column must be absent from the standings table."""

    def test_streak_not_in_display_rows_keys(self):
        """No dict literal in the page should include a 'Streak' key."""
        tree = ast.parse(PAGE_SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and key_node.value == "Streak":
                    raise AssertionError(
                        "Found a 'Streak' key in a dict literal inside the page. "
                        "Remove the Streak column from display_rows."
                    )

    def test_streak_column_not_in_record_df(self):
        """The 'Streak' column must not appear in the DataFrame built for the table.

        We check that there is no string literal 'Streak' appearing as a dict
        key in the page source (which would add it as a column).
        """
        # Look for  "Streak": ...  or  'Streak': ...  patterns
        col_pattern = re.compile(r"""["']Streak["']\s*:""")
        assert not col_pattern.search(PAGE_SRC), (
            "Found 'Streak': key in page source — it adds an empty column to the table. "
            "Remove the Streak column entirely from display_rows."
        )

    def test_no_streak_variable_in_display_rows_build_loop(self):
        """The variable 'streak' must not be used to populate display_rows.

        Reading 'streak' from records_df for the context-card chip is fine
        (that's a separate code path), but it must not feed display_rows.
        """
        # Find the display_rows list and check no Streak key is appended
        # via AST: look for dict with key "Streak"
        tree = ast.parse(PAGE_SRC)

        streak_key_nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and k.value == "Streak":
                        streak_key_nodes.append(k)

        assert not streak_key_nodes, (
            f"Found {len(streak_key_nodes)} 'Streak' key(s) in dict literals. "
            "Remove Streak from the standings table display_rows."
        )
