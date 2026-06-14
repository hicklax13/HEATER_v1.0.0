"""R-3 — Closer Watchlist feature tests (TDD).

Three structural guards (AST-based) verify the page wires the watchlist
without executing Streamlit or hitting the DB.

1. Page imports watchlist helpers from src.user_data.
2. A st.multiselect with a "Watch" label exists on the page.
3. The card-build marks watched closers with a WATCHED chip.
"""

from __future__ import annotations

import ast
import pathlib

PAGE = pathlib.Path(__file__).parent.parent / "pages" / "3_Closer_Monitor.py"
_src = PAGE.read_text(encoding="utf-8")
_tree = ast.parse(_src)


# ── helper ────────────────────────────────────────────────────────────────────


def _imports_from(module: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(_tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.extend(alias.name for alias in node.names)
    return names


# ── Test 1 — watchlist helpers imported from src.user_data ───────────────────


def test_page_imports_get_watchlist():
    """Page must import get_watchlist from src.user_data."""
    assert "get_watchlist" in _imports_from("src.user_data"), (
        "pages/3_Closer_Monitor.py must import get_watchlist from src.user_data"
    )


def test_page_imports_add_to_watchlist():
    """Page must import add_to_watchlist from src.user_data."""
    assert "add_to_watchlist" in _imports_from("src.user_data"), (
        "pages/3_Closer_Monitor.py must import add_to_watchlist from src.user_data"
    )


def test_page_imports_remove_from_watchlist():
    """Page must import remove_from_watchlist from src.user_data."""
    assert "remove_from_watchlist" in _imports_from("src.user_data"), (
        "pages/3_Closer_Monitor.py must import remove_from_watchlist from src.user_data"
    )


# ── Test 2 — st.multiselect with "Watch" label ───────────────────────────────


def test_page_has_watchlist_multiselect():
    """Page must call st.multiselect with a label containing 'Watch'.

    The widget lets users pick closers to track.  The label must contain
    the word 'Watch' (case-insensitive) so users can identify it.
    """
    for node in ast.walk(_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "multiselect"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        ):
            # Check first positional arg or 'label' keyword
            label_val: str | None = None
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    label_val = arg0.value
            if label_val is None:
                for kw in node.keywords:
                    if kw.arg == "label" and isinstance(kw.value, ast.Constant):
                        label_val = str(kw.value.value)
                        break
            if label_val is not None and "watch" in label_val.lower():
                return  # found it
    raise AssertionError(
        "pages/3_Closer_Monitor.py must call st.multiselect(...) "
        "with a label containing 'Watch' (e.g. '★ Watch closers')"
    )


# ── Test 3 — WATCHED chip present in card HTML ───────────────────────────────


def test_card_marks_watched_closers():
    """The card-build block must emit a WATCHED indicator for watched closers.

    The existing MINE/FREE chip pattern is: build a _badge_html string then
    inject it into the card markdown.  The WATCHED chip must follow the same
    pattern and the literal string 'WATCHED' must appear in the page source
    (inside the chip HTML or nearby variable name).
    """
    assert "WATCHED" in _src, (
        "pages/3_Closer_Monitor.py must render a 'WATCHED' chip on cards "
        "for closers whose player_id is in get_watchlist().  "
        "Expected the string 'WATCHED' in the page source."
    )


# ── Test 4 — watching-count caption present ──────────────────────────────────


def test_page_has_watching_count_caption():
    """Page must render a caption that surfaces how many closers are watched.

    Expected pattern: 'Watching N closers' or '★ Watching N closers' via
    st.caption() or an inline HTML caption.
    """
    assert "Watching" in _src and ("closer" in _src.lower() or "watch" in _src.lower()), (
        "pages/3_Closer_Monitor.py must render a 'Watching N closers' caption "
        "so the user can see how many are on their watchlist at a glance."
    )
