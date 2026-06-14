"""R-4 CSV export — structural tests for download buttons on two pages.

Each test uses AST analysis to verify that the page calls st.download_button(...)
with a ``file_name`` ending in ``.csv`` and ``mime="text/csv"``.  Tests are
written first (TDD); they will fail until the implementation is added.
"""

from __future__ import annotations

import ast
import pathlib

_PAGES_DIR = pathlib.Path(__file__).parent.parent / "pages"

_DATABANK_SRC = _PAGES_DIR / "19_Player_Databank.py"
_STANDINGS_SRC = _PAGES_DIR / "6_League_Standings.py"


# ── helpers ───────────────────────────────────────────────────────────────────


def _download_button_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ast.Call whose function resolves to st.download_button."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `st.download_button(...)` (Attribute on Name "st")
        if isinstance(func, ast.Attribute) and func.attr == "download_button":
            if isinstance(func.value, ast.Name) and func.value.id == "st":
                calls.append(node)
    return calls


def _keyword_value(call: ast.Call, kwname: str) -> ast.expr | None:
    """Return the AST expression for a keyword argument, or None if absent."""
    for kw in call.keywords:
        if kw.arg == kwname:
            return kw.value
    return None


def _literal_str(node: ast.expr | None) -> str | None:
    """If *node* is a string constant, return its value; else None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _var_csv_names(tree: ast.AST) -> set[str]:
    """Return the set of variable names whose assigned value ends with '.csv'.

    Handles both string constants and f-strings (JoinedStr) where the
    trailing part of the format string is a literal '.csv' suffix.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        is_csv = False
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            is_csv = val.value.endswith(".csv")
        elif isinstance(val, ast.JoinedStr) and val.values:
            last = val.values[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                is_csv = last.value.endswith(".csv")
        if is_csv:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _ends_with_csv(node: ast.expr | None, csv_var_names: set[str] | None = None) -> bool:
    """Return True if the node represents a value that ends with '.csv'.

    Handles:
    - Plain string constants (``"foo.csv"``)
    - f-strings (``JoinedStr``) whose last part ends in '.csv'
    - Variable references (``Name``) whose assignment ends in '.csv'
      (requires *csv_var_names* to be pre-computed via ``_var_csv_names``).
    """
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.endswith(".csv")
    if isinstance(node, ast.JoinedStr):
        if node.values and isinstance(node.values[-1], ast.Constant):
            last = node.values[-1].value
            if isinstance(last, str) and last.endswith(".csv"):
                return True
    if isinstance(node, ast.Name) and csv_var_names is not None:
        return node.id in csv_var_names
    return False


def _has_csv_download_button(tree: ast.AST) -> bool:
    """Return True if the tree contains a st.download_button with .csv file_name
    AND mime='text/csv'.
    """
    csv_vars = _var_csv_names(tree)
    for call in _download_button_calls(tree):
        file_name_node = _keyword_value(call, "file_name")
        mime_node = _keyword_value(call, "mime")
        mime = _literal_str(mime_node)
        if _ends_with_csv(file_name_node, csv_vars) and mime == "text/csv":
            return True
    return False


# ── Player Databank ───────────────────────────────────────────────────────────


def test_databank_has_csv_download_button():
    """pages/19_Player_Databank.py must call st.download_button with a .csv
    file_name and mime='text/csv'."""
    tree = ast.parse(_DATABANK_SRC.read_text(encoding="utf-8"))
    assert _has_csv_download_button(tree), (
        "pages/19_Player_Databank.py must contain a call to st.download_button(..., "
        "file_name=<something>.csv, mime='text/csv')"
    )


def test_databank_csv_button_has_unique_key():
    """The CSV download_button in the Databank page must have a key= argument
    (to avoid key-collision with the existing Excel export button)."""
    tree = ast.parse(_DATABANK_SRC.read_text(encoding="utf-8"))
    csv_vars = _var_csv_names(tree)
    for call in _download_button_calls(tree):
        file_name_node = _keyword_value(call, "file_name")
        mime_node = _keyword_value(call, "mime")
        mime = _literal_str(mime_node)
        if _ends_with_csv(file_name_node, csv_vars) and mime == "text/csv":
            key_node = _keyword_value(call, "key")
            assert key_node is not None, "The CSV download_button in 19_Player_Databank.py must have a key= argument."
            return
    assert False, "No CSV download_button found (run test_databank_has_csv_download_button first)."


# ── League Standings ──────────────────────────────────────────────────────────


def test_standings_has_csv_download_button():
    """pages/6_League_Standings.py must call st.download_button with a .csv
    file_name and mime='text/csv'."""
    tree = ast.parse(_STANDINGS_SRC.read_text(encoding="utf-8"))
    assert _has_csv_download_button(tree), (
        "pages/6_League_Standings.py must contain a call to st.download_button(..., "
        "file_name=<something>.csv, mime='text/csv')"
    )


def test_standings_csv_button_has_unique_key():
    """The CSV download_button in the Standings page must have a key= argument."""
    tree = ast.parse(_STANDINGS_SRC.read_text(encoding="utf-8"))
    for call in _download_button_calls(tree):
        file_name_node = _keyword_value(call, "file_name")
        mime_node = _keyword_value(call, "mime")
        mime = _literal_str(mime_node)
        if _ends_with_csv(file_name_node) and mime == "text/csv":
            key_node = _keyword_value(call, "key")
            assert key_node is not None, "The CSV download_button in 6_League_Standings.py must have a key= argument."
            return
    assert False, "No CSV download_button found (run test_standings_has_csv_download_button first)."
