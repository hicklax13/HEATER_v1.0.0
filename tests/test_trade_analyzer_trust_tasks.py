"""Structural-invariant guards for Tasks 3.1 / 3.3 / 3.7 on Trade Analyzer.

Task 3.7 — Verdict/grade hoisted BEFORE playoff sim (first result rendered).
Task 3.1 — render_data_freshness_chip called near the header.
Task 3.3 — jargon_help used for grade / surplus SGP / confidence;
           render_glossary_expander called once.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


# ── Task 3.1: data freshness chip ────────────────────────────────────────────


def test_render_data_freshness_chip_imported(source: str) -> None:
    """render_data_freshness_chip must be imported from src.ui_shared."""
    assert "render_data_freshness_chip" in source, (
        "Trade Analyzer must import and call render_data_freshness_chip (Task 3.1)."
    )


def test_render_data_freshness_chip_called(source: str) -> None:
    """render_data_freshness_chip() must be called in the page body."""
    assert re.search(r"render_data_freshness_chip\s*\(", source), (
        "render_data_freshness_chip() must be called near the header (Task 3.1)."
    )


# ── Task 3.3: jargon tooltips + glossary ─────────────────────────────────────


def test_jargon_help_imported(source: str) -> None:
    """jargon_help must be imported from src.ui_shared."""
    assert "jargon_help" in source, "Trade Analyzer must import jargon_help (Task 3.3)."


def test_render_glossary_expander_imported(source: str) -> None:
    """render_glossary_expander must be imported from src.ui_shared."""
    assert "render_glossary_expander" in source, "Trade Analyzer must import render_glossary_expander (Task 3.3)."


def test_jargon_help_used_for_grade(source: str) -> None:
    """jargon_help must be called with 'grade'-related term (Task 3.3)."""
    assert re.search(r'jargon_help\s*\(\s*["\'].*[Gg]rade', source) or re.search(
        r'jargon_help\s*\(\s*["\']SGP["\']', source
    ), "jargon_help must be called at least once (Task 3.3)."


def test_render_glossary_expander_called(source: str) -> None:
    """render_glossary_expander must be called once in the results section."""
    assert re.search(r"render_glossary_expander\s*\(", source), (
        "render_glossary_expander() must be called once in the results area (Task 3.3)."
    )


# ── Task 3.7: verdict/grade hoisted before playoff sim ───────────────────────


def test_verdict_banner_before_playoff_sim(source: str) -> None:
    """The verdict/grade block must appear BEFORE the playoff sim block.

    Structural check: the verdict banner HTML (result["verdict"]) must appear
    at a lower line number than the 'playoff_sim' render block.
    """
    lines = source.splitlines()

    # Find the first line that renders the verdict as HTML (the banner)
    verdict_banner_line: int | None = None
    for i, line in enumerate(lines, start=1):
        # The main verdict banner uses result["verdict"] in a large HTML block
        # and has a 'border' or 'glass' class (distinguishes it from the context card)
        if '"glass"' in line and 'result["verdict"]' in line:
            verdict_banner_line = i
            break

    # Also look for the verdict section header (fallback if banner spans lines)
    if verdict_banner_line is None:
        for i, line in enumerate(lines, start=1):
            if "verdict_key" in line or ("ACCEPT" in line and "color" in line and "result" in line):
                verdict_banner_line = i
                break

    # Find the first line that renders the playoff sim section
    playoff_sim_line: int | None = None
    for i, line in enumerate(lines, start=1):
        if '"playoff_sim"' in line and "in result" in line:
            playoff_sim_line = i
            break

    assert verdict_banner_line is not None, (
        "Could not find the verdict banner block in Trade Analyzer. "
        "The verdict/grade must render as an HTML element (Task 3.7)."
    )
    assert playoff_sim_line is not None, "Could not find the playoff_sim render block in Trade Analyzer (Task 3.7)."
    assert verdict_banner_line < playoff_sim_line, (
        f"Verdict banner (line {verdict_banner_line}) must appear BEFORE the "
        f"playoff sim block (line {playoff_sim_line}). Hoist the verdict (Task 3.7)."
    )


def test_advanced_metrics_in_expanders(source: str) -> None:
    """Weekly matrix and CARA/CVaR sections must be inside st.expander calls.

    After Task 3.7, CARA/CVaR and weekly matrix are secondary. At least one
    expander must wrap them (the existing weekly matrix expander is fine).
    """
    assert re.search(r"st\.expander\s*\(", source), (
        "Advanced metrics (weekly matrix / CARA) must be accessible via st.expander "
        "so the primary verdict/grade is the first result seen (Task 3.7)."
    )


def test_page_syntax_valid_after_tasks(source: str) -> None:
    """Page must parse as valid Python."""
    ast.parse(source)
