"""TDD tests for Player Compare rendering fixes (Task 4.6).

4.6-A  Injury health dots: health table uses render_compact_table with html_cols
        (not render_styled_table) so the colored dot spans render as HTML.
4.6-B  Same-type radar: two hitters (or two pitchers) must not plot the
        opposite-type axes (dead flat zeros).
4.6-C  Cross-type warning: hitter vs pitcher shows a warning / render_empty_state
        and must not show a composite verdict.
"""

import ast
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "16_Player_Compare.py"
PAGE_TEXT = PAGE.read_text(encoding="utf-8")
# Strip comments and docstrings for structural checks
PAGE_NO_COMMENTS = re.sub(r"#.*$", "", PAGE_TEXT, flags=re.MULTILINE)
PAGE_NO_STRINGS = re.sub(r'"""[\s\S]*?"""', "", PAGE_NO_COMMENTS)
PAGE_NO_STRINGS = re.sub(r"'[^']*'", "", PAGE_NO_STRINGS)


# ── 4.6-A: Injury badge HTML rendering ───────────────────────────────────────


def test_health_table_uses_render_compact_table():
    """The health/confidence section must use render_compact_table, not render_styled_table."""
    # Find the health section — look for 'Health & Confidence' then scan forward
    health_section_start = PAGE_TEXT.find("Health & Confidence")
    assert health_section_start != -1, "Expected 'Health & Confidence' section header in page"

    # The text after the subheader must contain render_compact_table
    after_health = PAGE_TEXT[health_section_start:]
    # Next major section starts at the next st.subheader or end of comparison block
    assert "render_compact_table" in after_health, (
        "Task 4.6-A: Health & Confidence section must use render_compact_table() "
        "so HTML dot spans in the Health column are rendered — render_styled_table "
        "is also fine but must NOT be the sole call for the health rows"
    )


def test_health_table_uses_html_cols():
    """render_compact_table for health rows must pass html_cols to allow span rendering."""
    health_section_start = PAGE_TEXT.find("Health & Confidence")
    assert health_section_start != -1

    after_health = PAGE_TEXT[health_section_start:]
    assert "html_cols" in after_health, (
        "Task 4.6-A: render_compact_table call for health rows must pass html_cols={...} "
        "so the colored dot <span> in the Health column is rendered as HTML, not escaped text"
    )


def test_health_column_in_html_cols():
    """The 'Health' column name must appear in the html_cols argument."""
    # Look for render_compact_table call that includes html_cols with "Health".
    # We search for the html_cols={"Health"} pattern anywhere on the page,
    # then verify it is part of a render_compact_table call.
    has_html_cols_health = bool(re.search(r'html_cols\s*=\s*\{[^}]*["\']Health["\']', PAGE_TEXT))
    # Also check that render_compact_table and html_cols appear on the same logical line
    # (within 200 chars of each other, allowing for the DataFrame(...) argument)
    has_compact_with_health = False
    for m in re.finditer(r"render_compact_table", PAGE_TEXT):
        nearby = PAGE_TEXT[m.start() : m.start() + 200]
        if "html_cols" in nearby and "Health" in nearby:
            has_compact_with_health = True
            break
    assert has_html_cols_health or has_compact_with_health, (
        "Task 4.6-A: The 'Health' column must be listed in html_cols when calling "
        "render_compact_table for the health rows, e.g. html_cols={'Health'}"
    )


def test_health_section_does_not_use_render_styled_table():
    """render_styled_table must not be the renderer for the health_rows table."""
    # After 'Health & Confidence', the next render_styled_table (if any) should
    # NOT be rendering health_rows — health_rows must flow into render_compact_table.
    # We check that `render_styled_table(pd.DataFrame(health_rows))` is NOT present.
    assert "render_styled_table(pd.DataFrame(health_rows))" not in PAGE_TEXT, (
        "Task 4.6-A: render_styled_table(pd.DataFrame(health_rows)) still present. "
        "Switch the health table to render_compact_table(..., html_cols={'Health'})"
    )


# ── 4.6-B: Same-type radar axes ──────────────────────────────────────────────


def test_radar_same_type_check_present():
    """Page must determine is_hitter for both players near the radar chart."""
    radar_start = PAGE_TEXT.find("Scatterpolar")
    assert radar_start != -1, "Expected Scatterpolar (radar chart) in page"

    # Look for is_hitter variable assignment in a 1500-char window before the radar
    window = PAGE_TEXT[max(0, radar_start - 1500) : radar_start + 100]
    has_type_check = (
        "is_hitter_a" in window
        or "is_hitter_b" in window
        or "_is_hitter_a" in window
        or "_is_hitter_b" in window
        or "same_type" in window
    )
    assert has_type_check, (
        "Task 4.6-B: Page must determine is_hitter_a and is_hitter_b before the radar "
        "chart so it can suppress dead axes in same-type comparisons"
    )


def test_radar_uses_type_aware_categories():
    """Radar chart section must select category subset based on player type."""
    radar_start = PAGE_TEXT.find("Scatterpolar")
    assert radar_start != -1, "Expected Scatterpolar (radar chart) in page"

    # Look 1500 chars before the radar for HITTING_CATEGORIES or PITCHING_CATEGORIES
    window = PAGE_TEXT[max(0, radar_start - 1500) : radar_start + 100]
    has_type_aware = (
        "HITTING_CATEGORIES" in window
        or "PITCHING_CATEGORIES" in window
        or "hitting_categories" in window
        or "pitching_categories" in window
    )
    assert has_type_aware, (
        "Task 4.6-B: Radar chart must reference HITTING_CATEGORIES / PITCHING_CATEGORIES "
        "in the window before Scatterpolar to build a type-filtered axis list. "
        "Two hitters → only 6 hitting axes; two pitchers → only 6 pitching axes."
    )


def test_radar_only_relevant_cats_comment_or_logic():
    """Page source must import or reference HITTING_CATEGORIES or PITCHING_CATEGORIES."""
    has_relevant = (
        "HITTING_CATEGORIES" in PAGE_TEXT
        or "PITCHING_CATEGORIES" in PAGE_TEXT
        or "hitting_categories" in PAGE_TEXT
        or "pitching_categories" in PAGE_TEXT
    )
    assert has_relevant, (
        "Task 4.6-B: No evidence of HITTING_CATEGORIES / PITCHING_CATEGORIES in page. "
        "Import them from src.ui_shared and use them to filter radar axes by player type."
    )


# ── 4.6-C: Cross-type comparison warning ─────────────────────────────────────


def test_cross_type_warning_present():
    """Page must show a dedicated warning when comparing a hitter to a pitcher."""
    # Must be a string literal containing both 'hitter' and 'pitcher' together
    # (not just separate occurrences from unrelated code)
    pattern = r'["\'].*?hitter.*?pitcher.*?["\']|["\'].*?pitcher.*?hitter.*?["\']'
    match = re.search(pattern, PAGE_TEXT, re.IGNORECASE)
    assert match, (
        "Task 4.6-C: No cross-type warning string found. Page must include a string "
        "literal mentioning both 'hitter' and 'pitcher' together (e.g. in st.warning(...) "
        "or render_empty_state(...)). Comparing a hitter and a pitcher is not meaningful."
    )


def test_cross_type_warning_uses_st_warning_or_render_empty_state():
    """The cross-type guard must call st.warning() near the hitter+pitcher message."""
    # Find a st.warning call that includes both hitter and pitcher
    # Look for st.warning(...hitter...pitcher...) or st.warning(...pitcher...hitter...)
    pattern = r"st\.warning\s*\([^)]*(?:hitter[^)]*pitcher|pitcher[^)]*hitter)[^)]*\)"
    match = re.search(pattern, PAGE_TEXT, re.IGNORECASE | re.DOTALL)
    has_warning = bool(match) or (
        "render_empty_state" in PAGE_TEXT and re.search(r"(hitter.*pitcher|pitcher.*hitter)", PAGE_TEXT, re.IGNORECASE)
    )
    assert has_warning, (
        "Task 4.6-C: Cross-type guard must call st.warning(...) with a message "
        "mentioning both 'hitter' and 'pitcher', or render_empty_state() with "
        "similar content. The composite comparison is not meaningful cross-type."
    )


def test_cross_type_guard_near_composite_section():
    """Cross-type check must appear within the comparison results block (before Category Edge)."""
    # The guard should be inside the `if "error" not in result:` block that
    # renders the identity strip, radar, Category Edge, etc.
    # We look in the full comparison result block (up to 7000 chars before Category Edge).
    edge_start = PAGE_TEXT.find("Category Edge")
    assert edge_start != -1, "Expected 'Category Edge' panel in page"

    # Look in a broad window that covers the entire comparison result block
    window = PAGE_TEXT[max(0, edge_start - 7000) : edge_start + 500]
    has_cross_guard = (
        "_cross_type" in window
        or re.search(r"(hitter.*pitcher|pitcher.*hitter)", window, re.IGNORECASE) is not None
        or "same_type" in window
    )
    assert has_cross_guard, (
        "Task 4.6-C: Cross-type guard not found within the comparison results block. "
        "Expected '_cross_type' or a hitter+pitcher string near the 'Category Edge' panel."
    )


def test_cross_type_hitter_pitcher_message_text():
    """The cross-type warning message must mention both hitter and pitcher explicitly."""
    pattern = r'["\'][^"\']*(?:hitter[^"\']*pitcher|pitcher[^"\']*hitter)[^"\']*["\']'
    match = re.search(pattern, PAGE_TEXT, re.IGNORECASE)
    assert match, (
        "Task 4.6-C: Warning message must explicitly mention both 'hitter' and 'pitcher' "
        "in the same string literal so users understand why the comparison is limited."
    )
