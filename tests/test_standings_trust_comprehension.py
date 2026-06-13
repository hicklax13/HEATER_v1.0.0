"""Trust & comprehension additions — League Standings page (Tasks 3.1, 3.3, 3.5).

Tests are AST/source-text based so they run without Streamlit or a real DB.

Task 3.1  — render_data_freshness_chip("standings") is called near the header.
Task 3.3  — jargon_help is used for Magic#, SOS, and category-rank badges;
            render_glossary_expander([...]) appears once.
Task 3.5  — Viewer's own team row is highlighted via is_mine flag / row class
            and a badge-legend line is present (explains what rank=1 means).
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "6_League_Standings.py"


def _page_text() -> str:
    assert PAGE.exists(), f"Page not found: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


def _page_ast() -> ast.Module:
    return ast.parse(_page_text())


# ── Task 3.1: Data freshness chip ────────────────────────────────────────────


def test_standings_imports_render_data_freshness_chip():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    text = _page_text()
    assert "render_data_freshness_chip" in text, "Task 3.1: render_data_freshness_chip not found in page imports"


def test_standings_freshness_chip_called_with_standings_source():
    """render_data_freshness_chip('standings') must be called on the page."""
    text = _page_text()
    # Accept both single- and double-quoted string
    pat = re.compile(r"render_data_freshness_chip\(\s*['\"]standings['\"]")
    assert pat.search(text), "Task 3.1: render_data_freshness_chip('standings') not found in page"


# ── Task 3.3: Jargon tooltips & glossary ─────────────────────────────────────


def test_standings_imports_jargon_help():
    """jargon_help must be imported from src.ui_shared."""
    text = _page_text()
    assert "jargon_help" in text, "Task 3.3: jargon_help not imported on standings page"


def test_standings_imports_render_glossary_expander():
    """render_glossary_expander must be imported from src.ui_shared."""
    text = _page_text()
    assert "render_glossary_expander" in text, "Task 3.3: render_glossary_expander not imported on standings page"


def test_standings_jargon_help_used_for_magic_number():
    """jargon_help('Magic#') (or equivalent) must be called on the page."""
    text = _page_text()
    pat = re.compile(r"jargon_help\(\s*['\"]Magic#['\"]")
    assert pat.search(text), "Task 3.3: jargon_help('Magic#') not found in standings page"


def test_standings_jargon_help_used_for_sos():
    """jargon_help('SOS') must be called on the page."""
    text = _page_text()
    pat = re.compile(r"jargon_help\(\s*['\"]SOS['\"]")
    assert pat.search(text), "Task 3.3: jargon_help('SOS') not found in standings page"


def test_standings_glossary_expander_called_once():
    """render_glossary_expander must be called exactly once on the page."""
    text = _page_text()
    count = text.count("render_glossary_expander(")
    assert count == 1, f"Task 3.3: expected render_glossary_expander called once, found {count} times"


def test_standings_glossary_includes_magic_and_sos():
    """The glossary call must include both 'Magic#' and 'SOS' in the terms list."""
    text = _page_text()
    # Find the render_glossary_expander(...) call and check it mentions Magic# and SOS
    pat = re.compile(r"render_glossary_expander\(([^)]*)\)", re.DOTALL)
    m = pat.search(text)
    assert m, "render_glossary_expander call not found"
    call_args = m.group(1)
    assert "Magic#" in call_args, "Task 3.3: 'Magic#' not in glossary term list"
    assert "SOS" in call_args, "Task 3.3: 'SOS' not in glossary term list"


# ── Task 3.5: Highlight viewer's own team ────────────────────────────────────


def test_standings_has_is_mine_flag_or_row_start_class():
    """The page must use 'is_mine' or 'row-start' to mark the viewer's row."""
    text = _page_text()
    assert "is_mine" in text or "row-start" in text, (
        "Task 3.5: No viewer-team highlight class (is_mine / row-start) found in standings page"
    )


def test_standings_user_team_row_class_set_via_resolver():
    """User team is resolved via _get_user_team_name or resolve_viewer_team_name
    and then used to mark the highlight row."""
    text = _page_text()
    # The page already uses _get_user_team_name which calls resolve_viewer_team_name.
    # Additionally the row_classes dict should be populated for the user row.
    assert "row_classes" in text or "row-start" in text, (
        "Task 3.5: row_classes dict or row-start class not found — user row not highlighted"
    )
    # Both the H2H table and the category grid should mark the user row
    assert text.count("row-start") >= 2, (
        "Task 3.5: expected at least 2 row-start assignments (H2H table + category grid)"
    )


def test_standings_badge_legend_present():
    """A one-line legend must explain the colored category-rank badges.

    The legend must mention rank 1 / best (or equivalent) so users understand
    what the green/yellow/red coloring means.
    """
    text = _page_text()
    # Accept any line that mentions "1" and one of: best, top, rank, green
    pat = re.compile(
        r"(rank.{0,30}1.{0,30}best|1.{0,30}best.{0,30}rank|badge.{0,50}best|best.{0,50}badge|1\s*=\s*best|rank\s*1\s*=|green.{0,30}best|best.{0,30}green)",
        re.IGNORECASE,
    )
    assert pat.search(text), (
        "Task 3.5: No badge legend explaining that rank 1 = best found in standings page. "
        "Add a one-line caption/markdown explaining the color coding."
    )
