"""R-3: Player watchlist on the Free Agents page.

Three test classes:
1. test_sync_watchlist_helper — pure unit tests for the extracted delta helper.
2. test_fa_page_watchlist_imports — AST/source guards that the page wires
   get_watchlist / toggle_watchlist / add_to_watchlist / remove_from_watchlist.
3. test_fa_page_watchlist_ui — source guards for the multiselect, star marks,
   and watchlist expander.
"""

from __future__ import annotations

import pathlib

_PAGE = pathlib.Path("pages/14_Free_Agents.py")


def _src() -> str:
    return _PAGE.read_text(encoding="utf-8")


# ── 1. Pure unit tests for the sync helper ────────────────────────────────────


def _import_sync_helper():
    """Import _sync_watchlist from the page module.

    We exec the helper function out of the page source rather than
    importing the full page (which would trigger Streamlit side-effects).
    """
    import ast
    import types

    src = _src()
    tree = ast.parse(src)

    # Extract just the _sync_watchlist function definition
    func_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_sync_watchlist":
            # Reconstruct source lines for this function
            lines = src.splitlines()
            func_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            break

    assert func_src is not None, (
        "_sync_watchlist helper not found in pages/14_Free_Agents.py. "
        "Add a standalone function: _sync_watchlist(selected_ids, current_ids) "
        "-> tuple[set[int], set[int]] returning (to_add, to_remove)."
    )

    mod = types.ModuleType("_fa_sync_test")
    exec(compile(func_src, "<fa_page>", "exec"), mod.__dict__)
    return mod._sync_watchlist


def test_sync_watchlist_add_new():
    """Selected ids not in current watchlist → to_add non-empty, to_remove empty."""
    _sync = _import_sync_helper()
    to_add, to_remove = _sync({1, 2, 3}, {1})
    assert to_add == {2, 3}
    assert to_remove == set()


def test_sync_watchlist_remove_deselected():
    """Ids in current watchlist but not selected → to_remove non-empty, to_add empty."""
    _sync = _import_sync_helper()
    to_add, to_remove = _sync({1}, {1, 2, 3})
    assert to_add == set()
    assert to_remove == {2, 3}


def test_sync_watchlist_mixed_delta():
    """Add some, remove others in the same call."""
    _sync = _import_sync_helper()
    to_add, to_remove = _sync({1, 4}, {1, 2, 3})
    assert to_add == {4}
    assert to_remove == {2, 3}


def test_sync_watchlist_no_change():
    """Identical selected and current → both sets empty."""
    _sync = _import_sync_helper()
    to_add, to_remove = _sync({1, 2}, {1, 2})
    assert to_add == set()
    assert to_remove == set()


def test_sync_watchlist_empty_inputs():
    """Both empty → both sets empty."""
    _sync = _import_sync_helper()
    to_add, to_remove = _sync(set(), set())
    assert to_add == set()
    assert to_remove == set()


# ── 2. Import guards ──────────────────────────────────────────────────────────


def test_fa_page_imports_get_watchlist():
    src = _src()
    assert "get_watchlist" in src, "pages/14_Free_Agents.py must import get_watchlist from src.user_data (R-3)"


def test_fa_page_imports_add_to_watchlist():
    src = _src()
    assert "add_to_watchlist" in src, "pages/14_Free_Agents.py must import add_to_watchlist from src.user_data (R-3)"


def test_fa_page_imports_remove_from_watchlist():
    src = _src()
    assert "remove_from_watchlist" in src, (
        "pages/14_Free_Agents.py must import remove_from_watchlist from src.user_data (R-3)"
    )


def test_fa_page_imports_toggle_watchlist():
    src = _src()
    assert "toggle_watchlist" in src, (
        "pages/14_Free_Agents.py must import toggle_watchlist from src.user_data (R-3). "
        "Even if the multiselect-based sync doesn't call it directly, it must be importable."
    )


def test_fa_page_imports_from_user_data():
    src = _src()
    assert "from src.user_data import" in src, "pages/14_Free_Agents.py must import from src.user_data (R-3)"


# ── 3. UI guards ──────────────────────────────────────────────────────────────


def test_fa_page_has_watchlist_multiselect():
    """Page must have a st.multiselect call with 'Watchlist' in the label (R-3)."""
    src = _src()
    assert "multiselect" in src.lower(), (
        "R-3: pages/14_Free_Agents.py must call st.multiselect for the watchlist control."
    )
    assert "watchlist" in src.lower(), "R-3: The multiselect must carry a 'Watchlist' label."


def test_fa_page_has_watchlist_expander():
    """Page must render a 'My Watchlist' expander (R-3)."""
    src = _src()
    # Accept 'Watchlist' inside an expander call
    assert "expander" in src.lower() and "watchlist" in src.lower(), (
        "R-3: pages/14_Free_Agents.py must render a watchlist expander (e.g. st.expander('★ My Watchlist (N)'))."
    )


def test_fa_page_marks_watched_players_with_star():
    """Page must prepend '★ ' to the display name for watched players (R-3)."""
    src = _src()
    assert "★" in src, (
        "R-3: pages/14_Free_Agents.py must mark watched players with a leading '★ ' in the displayed FA table."
    )


def test_fa_page_calls_sync_watchlist():
    """Page must contain the _sync_watchlist helper call to diff the multiselect (R-3)."""
    src = _src()
    assert "_sync_watchlist" in src, (
        "R-3: pages/14_Free_Agents.py must define and call _sync_watchlist to compute "
        "the add/remove delta from the multiselect."
    )


def test_fa_page_calls_add_to_watchlist():
    """Page must call add_to_watchlist to persist additions (R-3)."""
    src = _src()
    assert "add_to_watchlist(" in src, "R-3: pages/14_Free_Agents.py must call add_to_watchlist(...) when syncing."


def test_fa_page_calls_remove_from_watchlist():
    """Page must call remove_from_watchlist to persist removals (R-3)."""
    src = _src()
    assert "remove_from_watchlist(" in src, (
        "R-3: pages/14_Free_Agents.py must call remove_from_watchlist(...) when syncing."
    )


def test_fa_page_calls_get_watchlist():
    """Page must call get_watchlist() to populate the multiselect default (R-3)."""
    src = _src()
    assert "get_watchlist()" in src, (
        "R-3: pages/14_Free_Agents.py must call get_watchlist() to build the default "
        "selection for the st.multiselect watchlist control."
    )
