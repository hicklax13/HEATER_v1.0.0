"""TDD tests for Task 5.1 (My Team slice):
Counting stats in Marcel projections must render as integers ("15"), not floats ("15.00").
Rate stats (AVG, OBP, ERA, WHIP) keep their format_stat precision.
"""

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "1_My_Team.py"
PAGE_TEXT = PAGE.read_text(encoding="utf-8")


def test_no_float_format_for_counting_stats_in_bayes():
    """HR/RBI/SB/K must NOT be formatted with :.2f in the Marcel/Bayesian block.

    The line:
        bayes_df[c] = bayes_df[c].map(lambda x: f"{x:.2f}")
    applied to counting stats (HR, RBI, SB, K) must be replaced with integer
    formatting so "15" renders instead of "15.00".
    """
    # Find the bayes section: from the 'Marcel-Adjusted Projections' heading
    # onward, search for any :.2f map applied to HR/RBI/SB/K columns
    bayes_start = PAGE_TEXT.find("Marcel-Adjusted Projections")
    if bayes_start == -1:
        # Acceptable fallback: search whole page for the problematic pattern
        bayes_start = 0

    bayes_section = PAGE_TEXT[bayes_start : bayes_start + 3000]

    # The old bug: looping over ["HR", "RBI", "SB", "K", "ID"] with f"{x:.2f}"
    # Check that this specific float format is NOT applied to those counting stat cols
    bad_pattern = re.search(
        r'["\'](HR|RBI|SB|K)["\'].*?\.2f',
        bayes_section,
        re.DOTALL,
    )
    assert bad_pattern is None, (
        "Task 5.1: Found :.2f format applied to a counting stat (HR/RBI/SB/K) "
        "in the Marcel/Bayesian block. Use int() instead so '15' renders, not '15.00'."
    )


def test_counting_stats_use_integer_format_in_bayes():
    """The Marcel/Bayesian block must format HR/RBI/SB/K as integers.

    Acceptable forms:
      int(x)          str(int(x))      int(round(x))
      f"{int(x)}"     f"{int(x):d}"    str(round(x))
    """
    bayes_start = PAGE_TEXT.find("Marcel-Adjusted Projections")
    if bayes_start == -1:
        bayes_start = 0

    bayes_section = PAGE_TEXT[bayes_start : bayes_start + 3000]

    # Must contain some int() or :d conversion near counting stat handling
    has_int_format = bool(
        re.search(r"int\(", bayes_section)
        or re.search(r'f"\{[^}]*:d\}"', bayes_section)
        or re.search(r"str\(int\(", bayes_section)
        or re.search(r"int\(round\(", bayes_section)
    )
    assert has_int_format, (
        "Task 5.1: No integer formatting (int(), :d, str(int(...))) found in the "
        "Marcel/Bayesian block for counting stats. HR/RBI/SB/K must render as '15', not '15.00'."
    )


def test_rate_stats_still_use_format_stat_in_bayes():
    """AVG, ERA, WHIP must still use format_stat() in the Marcel/Bayesian block."""
    bayes_start = PAGE_TEXT.find("Marcel-Adjusted Projections")
    if bayes_start == -1:
        bayes_start = 0

    bayes_section = PAGE_TEXT[bayes_start : bayes_start + 3000]

    assert "format_stat" in bayes_section, (
        "Task 5.1: format_stat() must still be called for rate stats (AVG/ERA/WHIP) "
        "in the Marcel/Bayesian block — do not remove it."
    )


def test_counting_stat_columns_not_in_float_loop():
    """The :.2f map must NOT include HR, RBI, SB, K in its column list.

    Old code: for c in ["HR", "RBI", "SB", "K", "ID"]: ... f"{x:.2f}"
    New code: counting stats use int(); only non-category floats (if any) use :.2f.
    """
    bayes_start = PAGE_TEXT.find("Marcel-Adjusted Projections")
    if bayes_start == -1:
        bayes_start = 0

    bayes_section = PAGE_TEXT[bayes_start : bayes_start + 3000]

    # Check that the .2f map loop does NOT list the counting stat columns
    float_loop_match = re.search(
        r"for\s+\w+\s+in\s+\[([^\]]*)\][^:]*:\s*[^\n]*\.2f",
        bayes_section,
        re.DOTALL,
    )
    if float_loop_match:
        loop_cols = float_loop_match.group(1)
        counting_stats = {"HR", "RBI", "SB", "K"}
        for stat in counting_stats:
            assert f'"{stat}"' not in loop_cols and f"'{stat}'" not in loop_cols, (
                f"Task 5.1: '{stat}' must not appear in the :.2f formatting loop. "
                f"Loop column list was: [{loop_cols.strip()}]"
            )
