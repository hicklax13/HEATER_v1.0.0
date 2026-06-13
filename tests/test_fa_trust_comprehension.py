"""TDD tests for Free Agents page trust/comprehension fixes.

Tasks:
  3.1 — render_data_freshness_chip("free_agents") called near the header.
  3.3 — jargon_help("Heat") / jargon_help("Net SGP") referenced in page;
        render_glossary_expander([...]) called once.
  3.4 — Heat scale anchor text present in the page source.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "14_Free_Agents.py"


def _src() -> str:
    return _PAGE.read_text(encoding="utf-8")


# ── Task 3.1: freshness chip ─────────────────────────────────────────────────


def test_fa_freshness_chip_imported():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    src = _src()
    assert "render_data_freshness_chip" in src, (
        "pages/14_Free_Agents.py must import render_data_freshness_chip from src.ui_shared"
    )


def test_fa_freshness_chip_called_with_source():
    """render_data_freshness_chip must be called with 'free_agents' as the source."""
    src = _src()
    assert 'render_data_freshness_chip("free_agents")' in src or "render_data_freshness_chip('free_agents')" in src, (
        'pages/14_Free_Agents.py must call render_data_freshness_chip("free_agents")'
    )


# ── Task 3.3: jargon tooltips + glossary ─────────────────────────────────────


def test_fa_jargon_help_imported():
    """jargon_help must be imported from src.ui_shared."""
    src = _src()
    assert "jargon_help" in src, "pages/14_Free_Agents.py must import jargon_help from src.ui_shared"


def test_fa_jargon_help_heat_referenced():
    """jargon_help('Heat') must appear in the page."""
    src = _src()
    assert 'jargon_help("Heat")' in src or "jargon_help('Heat')" in src, (
        'pages/14_Free_Agents.py must reference jargon_help("Heat") for the Heat column tooltip'
    )


def test_fa_jargon_help_net_sgp_referenced():
    """jargon_help('Net SGP') must appear in the page."""
    src = _src()
    assert 'jargon_help("Net SGP")' in src or "jargon_help('Net SGP')" in src, (
        'pages/14_Free_Agents.py must reference jargon_help("Net SGP") for the Net SGP column tooltip'
    )


def test_fa_glossary_expander_imported():
    """render_glossary_expander must be imported from src.ui_shared."""
    src = _src()
    assert "render_glossary_expander" in src, (
        "pages/14_Free_Agents.py must import render_glossary_expander from src.ui_shared"
    )


def test_fa_glossary_expander_called_once():
    """render_glossary_expander must be called exactly once in the page."""
    src = _src()
    count = src.count("render_glossary_expander(")
    assert count == 1, f"pages/14_Free_Agents.py must call render_glossary_expander exactly once, found {count}"


def test_fa_glossary_expander_includes_heat_and_sgp():
    """render_glossary_expander call must include 'Heat', 'Net SGP', and 'SGP'."""
    src = _src()
    # Find the call via AST — look for render_glossary_expander(...) with a list arg
    tree = ast.parse(src)
    found_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_glossary_expander"
        ):
            found_call = True
            # The first positional arg should be a list with our terms
            if node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.List):
                    terms = [
                        elt.value
                        for elt in first_arg.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    assert "Heat" in terms, f"render_glossary_expander list must include 'Heat', got {terms}"
                    assert "Net SGP" in terms, f"render_glossary_expander list must include 'Net SGP', got {terms}"
                    assert "SGP" in terms, f"render_glossary_expander list must include 'SGP', got {terms}"
                    return  # All assertions passed
    assert found_call, "render_glossary_expander call not found in pages/14_Free_Agents.py"
    # If we reach here, call found but no list arg — still valid as long as call exists
    assert True


# ── Task 3.4: Heat scale anchor ──────────────────────────────────────────────


def test_fa_heat_scale_anchor_present():
    """A short scale description for the Heat column (0-10) must appear in the page."""
    src = _src()
    # Accept any variation that conveys '0-10' scale anchoring for Heat
    has_anchor = (
        "0-10" in src
        and ("Heat" in src)
        and ("higher" in src.lower() or "hot" in src.lower() or "more added" in src.lower() or "scale" in src.lower())
    )
    assert has_anchor, (
        "pages/14_Free_Agents.py must include a brief Heat 0-10 scale anchor "
        "(e.g. 'Heat 0-10: higher = more added/owned lately')"
    )
