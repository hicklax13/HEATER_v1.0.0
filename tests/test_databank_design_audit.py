"""Design-audit tests for Player Databank page (tasks 3.1 and 3.5).

Task 3.1 — render_data_freshness_chip imported + called.
Task 3.5 — viewer team resolved via resolve_viewer_team_name; roster-lens UI present.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).parent.parent / "pages" / "19_Player_Databank.py"
_tree = ast.parse(SRC.read_text(encoding="utf-8"))
_src_text = SRC.read_text(encoding="utf-8")


# ── Task 3.1 — freshness chip ─────────────────────────────────────────────────


def test_databank_imports_freshness_chip():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    imports = []
    for node in ast.walk(_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.ui_shared":
            imports.extend(alias.name for alias in node.names)
    assert "render_data_freshness_chip" in imports, (
        "pages/19_Player_Databank.py must import render_data_freshness_chip from src.ui_shared"
    )


def test_databank_calls_freshness_chip():
    """render_data_freshness_chip must be called somewhere on the page."""
    assert "render_data_freshness_chip(" in _src_text, (
        "pages/19_Player_Databank.py must call render_data_freshness_chip()"
    )


# ── Task 3.5 — viewer roster lens ────────────────────────────────────────────


def test_databank_imports_resolve_viewer_team_name():
    """resolve_viewer_team_name must be imported from src.auth."""
    imports = []
    for node in ast.walk(_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.auth":
            imports.extend(alias.name for alias in node.names)
    assert "resolve_viewer_team_name" in imports, (
        "pages/19_Player_Databank.py must import resolve_viewer_team_name from src.auth"
    )


def test_databank_calls_resolve_viewer_team_name_with_arg():
    """resolve_viewer_team_name must be called with at least one argument (the rosters frame)."""
    for node in ast.walk(_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_viewer_team_name"
        ):
            # Must have at least one positional or keyword arg
            assert node.args or node.keywords, "resolve_viewer_team_name() must be called WITH a rosters frame"
            return
    raise AssertionError("pages/19_Player_Databank.py must call resolve_viewer_team_name(rosters)")


def test_databank_has_roster_lens_filter():
    """Page must have a roster-lens option: 'My Roster' or similar radio/selectbox."""
    # Accept either a st.radio or st.selectbox with a 'My Roster' / 'Mine' option,
    # or a 'viewer_team' variable assignment, or the "My Roster" label string.
    has_lens = (
        "My Roster" in _src_text or "MINE" in _src_text or "viewer_team" in _src_text or "roster_lens" in _src_text
    )
    assert has_lens, (
        "pages/19_Player_Databank.py must expose a 'My Roster / Free Agents / All' "
        "lens or a viewer-team highlight. Expected 'My Roster', 'MINE', 'viewer_team', "
        "or 'roster_lens' to appear in the page source."
    )
