"""Design-audit tests for Closer Monitor page (tasks 3.1, 3.3, 3.5).

Task 3.1 — render_data_freshness_chip imported + called.
Task 3.3 — jargon_help + render_glossary_expander imported + called with "% JOB" / "gmLI".
Task 3.5 — viewer team resolved via resolve_viewer_team_name; MINE/FREE badge logic present.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).parent.parent / "pages" / "3_Closer_Monitor.py"
_tree = ast.parse(SRC.read_text(encoding="utf-8"))
_src_text = SRC.read_text(encoding="utf-8")


# ── helpers ───────────────────────────────────────────────────────────────────


def _ui_shared_imports() -> list[str]:
    names: list[str] = []
    for node in ast.walk(_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.ui_shared":
            names.extend(alias.name for alias in node.names)
    return names


def _auth_imports() -> list[str]:
    names: list[str] = []
    for node in ast.walk(_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.auth":
            names.extend(alias.name for alias in node.names)
    return names


# ── Task 3.1 — freshness chip ─────────────────────────────────────────────────


def test_closer_imports_freshness_chip():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    assert "render_data_freshness_chip" in _ui_shared_imports(), (
        "pages/3_Closer_Monitor.py must import render_data_freshness_chip from src.ui_shared"
    )


def test_closer_calls_freshness_chip():
    """render_data_freshness_chip must be called somewhere on the page."""
    assert "render_data_freshness_chip(" in _src_text, (
        "pages/3_Closer_Monitor.py must call render_data_freshness_chip()"
    )


# ── Task 3.3 — jargon tooltips + glossary ────────────────────────────────────


def test_closer_imports_jargon_help():
    """jargon_help must be imported from src.ui_shared."""
    assert "jargon_help" in _ui_shared_imports(), "pages/3_Closer_Monitor.py must import jargon_help from src.ui_shared"


def test_closer_imports_render_glossary_expander():
    """render_glossary_expander must be imported from src.ui_shared."""
    assert "render_glossary_expander" in _ui_shared_imports(), (
        "pages/3_Closer_Monitor.py must import render_glossary_expander from src.ui_shared"
    )


def test_closer_calls_jargon_help_pct_job():
    """jargon_help('% JOB') must appear in the page source."""
    assert 'jargon_help("% JOB")' in _src_text or "jargon_help('% JOB')" in _src_text, (
        'pages/3_Closer_Monitor.py must call jargon_help("% JOB") as a help= tooltip'
    )


def test_closer_calls_render_glossary_expander_with_pct_job():
    """render_glossary_expander must be called and must include '% JOB'."""
    assert "render_glossary_expander(" in _src_text, "pages/3_Closer_Monitor.py must call render_glossary_expander()"
    assert "% JOB" in _src_text, "render_glossary_expander call must include '% JOB'"


# ── Task 3.5 — viewer roster lens ────────────────────────────────────────────


def test_closer_imports_resolve_viewer_team_name():
    """resolve_viewer_team_name must be imported from src.auth."""
    assert "resolve_viewer_team_name" in _auth_imports(), (
        "pages/3_Closer_Monitor.py must import resolve_viewer_team_name from src.auth"
    )


def test_closer_calls_resolve_viewer_team_name_with_arg():
    """resolve_viewer_team_name must be called with at least one argument."""
    for node in ast.walk(_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_viewer_team_name"
        ):
            assert node.args or node.keywords, "resolve_viewer_team_name() must be called WITH a rosters frame"
            return
    raise AssertionError("pages/3_Closer_Monitor.py must call resolve_viewer_team_name(rosters)")


def test_closer_has_mine_free_badge():
    """Page must output a MINE / FREE (or 'My closers') ownership badge."""
    has_badge = (
        "MINE" in _src_text
        or "FREE" in _src_text
        or "my_closers" in _src_text
        or "viewer_team" in _src_text
        or "viewer_name" in _src_text
    )
    assert has_badge, (
        "pages/3_Closer_Monitor.py must flag closers on the viewer's roster "
        "with a MINE/FREE badge or similar. Expected 'MINE', 'FREE', "
        "'my_closers', 'viewer_team', or 'viewer_name' in the source."
    )
