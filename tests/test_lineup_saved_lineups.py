"""TDD tests for R-3: save/load named lineup configs on the Lineup Optimizer page.

Tests cover:
1. Page imports save_view / load_view / list_views / delete_view from src.user_data
   with kind="lineup".
2. The _build_lineup_payload() helper returns a JSON-serializable dict with the
   required keys (mode, scope, risk_aversion, starter_ids).
3. Structural checks: the page renders the "Saved lineups" expander/section with
   a name text input, Save button, load selectbox, Load button, and Delete button.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "2_Line-up_Optimizer.py"
_PAGE_TEXT: str = _PAGE.read_text(encoding="utf-8")


# ── 1. Import checks ─────────────────────────────────────────────────


def test_page_imports_save_view_from_user_data():
    assert "save_view" in _PAGE_TEXT, "page must import save_view from src.user_data"


def test_page_imports_load_view_from_user_data():
    assert "load_view" in _PAGE_TEXT, "page must import load_view from src.user_data"


def test_page_imports_list_views_from_user_data():
    assert "list_views" in _PAGE_TEXT, "page must import list_views from src.user_data"


def test_page_imports_delete_view_from_user_data():
    assert "delete_view" in _PAGE_TEXT, "page must import delete_view from src.user_data"


def test_page_uses_lineup_kind_for_save_view():
    """All save_view / load_view / list_views / delete_view calls must use kind="lineup"."""
    assert 'save_view("lineup"' in _PAGE_TEXT or "save_view('lineup'" in _PAGE_TEXT, (
        'page must call save_view("lineup", ...) with kind="lineup"'
    )


def test_page_uses_lineup_kind_for_load_view():
    assert 'load_view("lineup"' in _PAGE_TEXT or "load_view('lineup'" in _PAGE_TEXT, (
        'page must call load_view("lineup", ...) with kind="lineup"'
    )


def test_page_uses_lineup_kind_for_list_views():
    assert 'list_views("lineup"' in _PAGE_TEXT or "list_views('lineup'" in _PAGE_TEXT, (
        'page must call list_views("lineup") with kind="lineup"'
    )


def test_page_uses_lineup_kind_for_delete_view():
    assert 'delete_view("lineup"' in _PAGE_TEXT or "delete_view('lineup'" in _PAGE_TEXT, (
        'page must call delete_view("lineup", ...) with kind="lineup"'
    )


# ── 2. _build_lineup_payload helper ──────────────────────────────────


def test_build_lineup_payload_defined_in_page():
    assert "_build_lineup_payload" in _PAGE_TEXT, "page must define a _build_lineup_payload helper function"


def test_build_lineup_payload_returns_json_serializable_dict():
    """Import and call the helper with realistic args; verify output shape."""
    import importlib.util
    import sys
    import types
    import unittest.mock

    # Build a minimal stub environment so the page module can be imported
    # without Streamlit / DB / Yahoo dependencies.
    st_mock = types.ModuleType("streamlit")
    for _attr in (
        "session_state",
        "radio",
        "slider",
        "button",
        "selectbox",
        "text_input",
        "spinner",
        "success",
        "info",
        "warning",
        "error",
        "stop",
        "toast",
        "tabs",
        "columns",
        "expander",
        "metric",
        "caption",
        "divider",
        "progress",
        "markdown",
        "rerun",
        "fragment",
        "dialog",
        "subheader",
        "write",
        "empty",
        "set_page_config",
        "chat_message",
        "chat_input",
    ):
        setattr(st_mock, _attr, unittest.mock.MagicMock())
    st_mock.session_state = {}

    # We extract and exec only the helper function, not the full module.
    # This is simpler and more reliable than full module import for a page.
    src = _PAGE_TEXT

    # Find the function definition via AST
    tree = ast.parse(src)
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_lineup_payload":
            func_node = node
            break

    assert func_node is not None, "_build_lineup_payload must be an ast.FunctionDef"

    # Exec just the function in an isolated namespace
    func_src_lines = src.splitlines()[func_node.lineno - 1 : func_node.end_lineno]
    func_src = "\n".join(func_src_lines)
    ns: dict = {}
    exec(compile(func_src, "<_build_lineup_payload>", "exec"), ns)  # noqa: S102

    fn = ns["_build_lineup_payload"]

    payload = fn(
        mode="standard",
        scope="rest_of_week",
        risk_aversion=0.15,
        starter_ids=[101, 202, 303],
    )

    # Must be a dict
    assert isinstance(payload, dict), "_build_lineup_payload must return a dict"

    # Must be JSON-serializable (no DataFrames, sets, or non-primitive types)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)

    # Required keys
    assert "mode" in payload, "payload must contain 'mode'"
    assert "scope" in payload, "payload must contain 'scope'"
    assert "risk_aversion" in payload, "payload must contain 'risk_aversion'"
    assert "starter_ids" in payload, "payload must contain 'starter_ids'"

    # Correct values
    assert payload["mode"] == "standard"
    assert payload["scope"] == "rest_of_week"
    assert abs(payload["risk_aversion"] - 0.15) < 1e-9
    assert payload["starter_ids"] == [101, 202, 303]


def test_build_lineup_payload_empty_starter_ids_is_valid():
    """No optimization result yet — starter_ids=[] is fine."""
    tree = ast.parse(_PAGE_TEXT)
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_lineup_payload":
            func_node = node
            break
    assert func_node is not None
    func_src = "\n".join(_PAGE_TEXT.splitlines()[func_node.lineno - 1 : func_node.end_lineno])
    ns: dict = {}
    exec(compile(func_src, "<_build_lineup_payload>", "exec"), ns)  # noqa: S102
    payload = ns["_build_lineup_payload"](mode="quick", scope="today", risk_aversion=0.0, starter_ids=[])
    assert payload["starter_ids"] == []
    assert json.dumps(payload)  # must be serializable


# ── 3. Structural: UI elements present in page source ────────────────


def test_page_has_saved_lineups_section():
    """Page must contain a recognisable 'saved lineups' heading / expander label."""
    lower = _PAGE_TEXT.lower()
    assert "saved lineup" in lower, "page must include a 'Saved lineups' section (expander label or heading)"


def test_page_has_lineup_save_name_text_input():
    """Page must include a st.text_input for naming the lineup to save."""
    assert "lineup_save_name" in _PAGE_TEXT, (
        'page must contain a st.text_input with key="lineup_save_name" (or similar) '
        "so the user can type a name before saving"
    )


def test_page_has_save_lineup_button():
    """Page must include a Save button wired to save_view."""
    assert "lineup_save_btn" in _PAGE_TEXT or "Save lineup" in _PAGE_TEXT or "Save current lineup" in _PAGE_TEXT, (
        "page must contain a Save button for the named lineup (key lineup_save_btn or "
        "label 'Save lineup' / 'Save current lineup')"
    )


def test_page_has_load_lineup_selectbox():
    """Page must include a st.selectbox for picking a saved lineup to load."""
    assert "lineup_load_select" in _PAGE_TEXT, (
        "page must contain a st.selectbox with key=\"lineup_load_select\" populated from list_views('lineup')"
    )


def test_page_has_load_lineup_button():
    """Page must include a Load button wired to load_view."""
    assert "lineup_load_btn" in _PAGE_TEXT or "Load lineup" in _PAGE_TEXT or "Load saved lineup" in _PAGE_TEXT, (
        "page must contain a Load button for the selected lineup"
    )


def test_page_has_delete_lineup_button():
    """Page must include a Delete button wired to delete_view."""
    assert "lineup_delete_btn" in _PAGE_TEXT or "Delete lineup" in _PAGE_TEXT or "Delete" in _PAGE_TEXT, (
        "page must contain a Delete button for the selected lineup"
    )
