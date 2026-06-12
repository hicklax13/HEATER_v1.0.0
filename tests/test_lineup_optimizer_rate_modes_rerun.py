"""Regression guard: the 2026-06-10 "Refresh Yahoo Data" NameError.

``_dcv_rate_modes`` was bound only inside the ``if optimize_clicked:`` branch
of pages/2, but consumed in the persisted-results RENDER branch — which
re-executes on every rerun (any widget click, including "Refresh Yahoo
Data"), where the optimize-click local is unbound ⇒ NameError. The fix
persists it to ``st.session_state["_dcv_rate_modes"]`` at optimize time and
reads it back in the render branch. This test locks both halves, and
verifies no OTHER optimize-click local is consumed unbound in the render
branch (same bug DNA).
"""

from __future__ import annotations

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "2_Line-up_Optimizer.py"


def test_rate_modes_persisted_and_read_from_session_state():
    src = _PAGE.read_text(encoding="utf-8")
    assert 'st.session_state["_dcv_rate_modes"]' in src, (
        "optimize-click branch must persist _dcv_rate_modes to session state"
    )
    assert 'st.session_state.get("_dcv_rate_modes")' in src, (
        "the render branch must read _dcv_rate_modes back from session state — "
        "the optimize-click local is unbound on plain reruns"
    )


def _branch_ranges(tree: ast.Module) -> tuple[tuple[int, int], tuple[int, int]]:
    """(click_branch_range, render_branch_range) located structurally:
    the ``if optimize_clicked:`` If node and the next sibling If whose test
    references ``result`` — robust to line drift."""
    click = render = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "optimize_clicked" in names and click is None:
            click = node
        elif "result" in names and click is not None and render is None and node.lineno > click.end_lineno:
            render = node
    assert click is not None, "optimize_clicked branch not found"
    assert render is not None, "results render branch not found"
    return (click.lineno, click.end_lineno), (render.lineno, render.end_lineno)


def test_no_click_branch_locals_consumed_in_render_branch():
    tree = ast.parse(_PAGE.read_text(encoding="utf-8"))
    (c_lo, c_hi), (r_lo, r_hi) = _branch_ranges(tree)

    click_assigned: set[str] = set()
    render_assigned: set[str] = set()
    render_loaded: dict[str, int] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        line = node.lineno
        if isinstance(node.ctx, ast.Store):
            if c_lo <= line <= c_hi:
                click_assigned.add(node.id)
            if r_lo <= line <= r_hi:
                render_assigned.add(node.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        line = node.lineno
        if r_lo <= line <= r_hi and node.id.startswith("_"):
            if node.id in click_assigned and node.id not in render_assigned:
                render_loaded.setdefault(node.id, line)

    assert not render_loaded, (
        "render-branch reads of optimize-click locals (NameError on rerun — "
        f"the Refresh Yahoo Data bug DNA): {render_loaded}"
    )
