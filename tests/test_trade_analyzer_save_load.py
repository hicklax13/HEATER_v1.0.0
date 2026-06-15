"""TDD tests for R-3: save/load named trade proposals in Trade Analyzer.

Tests are written RED first. They check:
1. The page imports save_view / load_view / list_views / delete_view from src.user_data.
2. The page defines a _save_trade helper that serialises the current proposal.
3. Loading a saved trade writes _tf_prefill (or sets giving/receiving) in session
   state, reusing the existing _consume_trade_prefill mechanism.
4. Defensive cases: blank name → no-op, missing name → friendly no-op.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


# ── 1. Import checks ──────────────────────────────────────────────────────────


def test_page_imports_save_view(source: str) -> None:
    """pages/11_Trade_Analyzer.py must import save_view from src.user_data."""
    assert "save_view" in source, "Trade Analyzer must import save_view from src.user_data (R-3)."


def test_page_imports_load_view(source: str) -> None:
    """pages/11_Trade_Analyzer.py must import load_view from src.user_data."""
    assert "load_view" in source, "Trade Analyzer must import load_view from src.user_data (R-3)."


def test_page_imports_list_views(source: str) -> None:
    """pages/11_Trade_Analyzer.py must import list_views from src.user_data."""
    assert "list_views" in source, "Trade Analyzer must import list_views from src.user_data (R-3)."


def test_page_imports_delete_view(source: str) -> None:
    """pages/11_Trade_Analyzer.py must import delete_view from src.user_data."""
    assert "delete_view" in source, "Trade Analyzer must import delete_view from src.user_data (R-3)."


def test_imports_use_kind_trade(source: str) -> None:
    """The string 'trade' must appear as the kind argument near the user_data calls."""
    assert '"trade"' in source or "'trade'" in source, (
        'Trade Analyzer must call save/load/list/delete_view with kind="trade" (R-3).'
    )


# ── 2. _save_trade_proposal helper ───────────────────────────────────────────


def _extract_function(source: str, name: str) -> str | None:
    """Return the raw source text of a top-level function *name*, or None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def test_save_trade_proposal_helper_exists(source: str) -> None:
    """Page must define _save_trade_proposal(name, giving_ids, receiving_ids, result) helper."""
    fn = _extract_function(source, "_save_trade_proposal")
    assert fn is not None, "pages/11_Trade_Analyzer.py must define _save_trade_proposal() helper (R-3)."


def test_save_trade_calls_save_view(source: str) -> None:
    """_save_trade_proposal must call save_view."""
    fn = _extract_function(source, "_save_trade_proposal")
    assert fn is not None, "_save_trade_proposal not found."
    assert "save_view" in fn, "_save_trade_proposal must call save_view() (R-3)."


def _exec_save_helper(mock_save_fn=None, mock_load_fn=None) -> Any:
    """Exec _save_trade_proposal with save_view/load_view pre-injected in scope."""
    fn_src = _extract_function(PAGE_PATH.read_text(encoding="utf-8"), "_save_trade_proposal")
    assert fn_src is not None, "_save_trade_proposal not found."
    # Inject dependencies so the exec'd function can call them.
    import src.user_data as _ud

    ns: dict[str, Any] = {
        "save_view": mock_save_fn if mock_save_fn is not None else _ud.save_view,
        "load_view": mock_load_fn if mock_load_fn is not None else _ud.load_view,
        "list_views": _ud.list_views,
        "delete_view": _ud.delete_view,
    }
    exec(compile(fn_src, PAGE_PATH.name, "exec"), ns)  # noqa: S102
    return ns["_save_trade_proposal"]


def _exec_load_helper(mock_save_fn=None, mock_load_fn=None) -> Any:
    """Exec _load_trade_proposal with save_view/load_view pre-injected in scope."""
    fn_src = _extract_load_helper(PAGE_PATH.read_text(encoding="utf-8"))
    assert fn_src is not None, "_load_trade_proposal not found."
    import src.user_data as _ud

    ns: dict[str, Any] = {
        "save_view": mock_save_fn if mock_save_fn is not None else _ud.save_view,
        "load_view": mock_load_fn if mock_load_fn is not None else _ud.load_view,
        "list_views": _ud.list_views,
        "delete_view": _ud.delete_view,
    }
    exec(compile(fn_src, PAGE_PATH.name, "exec"), ns)  # noqa: S102
    return ns["_load_trade_proposal"]


def test_save_trade_rejects_blank_name() -> None:
    """_save_trade_proposal with blank/whitespace name must be a no-op (no DB call)."""
    mock_save = MagicMock()
    _save = _exec_save_helper(mock_save_fn=mock_save)
    _save("   ", [1], [2], None)  # blank name — should not call save_view
    mock_save.assert_not_called()


def test_save_trade_serialises_giving_receiving() -> None:
    """_save_trade_proposal must include giving_ids and receiving_ids in the payload."""
    captured: list[dict] = []

    def _mock_save(kind: str, name: str, payload: dict) -> None:  # noqa: ARG001
        captured.append({"kind": kind, "name": name, "payload": payload})

    _save = _exec_save_helper(mock_save_fn=_mock_save)
    _save("My Trade", [10, 11], [20], None)

    assert len(captured) == 1, "save_view must be called exactly once."
    call = captured[0]
    assert call["kind"] == "trade", "kind must be 'trade'."
    assert call["name"] == "My Trade", "name must pass through."
    payload = call["payload"]
    assert payload.get("giving_ids") == [10, 11], "giving_ids must be in payload."
    assert payload.get("receiving_ids") == [20], "receiving_ids must be in payload."


def test_save_trade_payload_is_json_serialisable() -> None:
    """Payload produced by _save_trade_proposal must be JSON-serialisable."""
    captured: list[dict] = []

    def _mock_save(kind: str, name: str, payload: dict) -> None:  # noqa: ARG001
        captured.append(payload)

    _save = _exec_save_helper(mock_save_fn=_mock_save)
    _save("Test", [1, 2], [3], {"grade": "A", "verdict": "ACCEPT"})

    assert captured, "save_view was not called."
    # This must not raise
    json.dumps(captured[0])


def test_save_trade_includes_grade_when_result_present() -> None:
    """When a result dict is provided, grade/verdict should be captured in payload."""
    captured: list[dict] = []

    def _mock_save(kind: str, name: str, payload: dict) -> None:  # noqa: ARG001
        captured.append(payload)

    _save = _exec_save_helper(mock_save_fn=_mock_save)
    _save("Trade A", [5], [6], {"grade": "B+", "verdict": "ACCEPT"})

    assert captured, "save_view was not called."
    payload = captured[0]
    # grade/verdict captured — either at top level or nested under "result"
    has_grade = "grade" in payload or ("result" in payload and "grade" in payload.get("result", {}))
    assert has_grade, "grade must be saved in payload when result is provided."


# ── 3. Load path sets _tf_prefill ────────────────────────────────────────────


def _extract_load_helper(source: str) -> str | None:
    """Return source of _load_trade_proposal or None."""
    return _extract_function(source, "_load_trade_proposal")


def test_load_trade_proposal_helper_exists(source: str) -> None:
    """Page must define _load_trade_proposal(name, session_state) helper."""
    fn = _extract_load_helper(source)
    assert fn is not None, "pages/11_Trade_Analyzer.py must define _load_trade_proposal() helper (R-3)."


def test_load_trade_calls_load_view(source: str) -> None:
    """_load_trade_proposal must call load_view."""
    fn = _extract_load_helper(source)
    assert fn is not None, "_load_trade_proposal not found."
    assert "load_view" in fn, "_load_trade_proposal must call load_view() (R-3)."


def test_load_trade_sets_tf_prefill() -> None:
    """Loading a saved trade must write _tf_prefill to session state."""
    saved_payload = {"giving_ids": [10, 11], "receiving_ids": [20]}

    def _mock_load(kind: str, name: str) -> dict | None:  # noqa: ARG001
        return saved_payload

    _load = _exec_load_helper(mock_load_fn=_mock_load)
    ss: dict = {}
    _load("My Trade", ss)

    assert "_tf_prefill" in ss, "_load_trade_proposal must set _tf_prefill in session state."
    prefill = ss["_tf_prefill"]
    assert prefill.get("giving_ids") == [10, 11]
    assert prefill.get("receiving_ids") == [20]


def test_load_trade_no_op_on_missing() -> None:
    """If load_view returns None (no such saved trade), session state is unchanged."""

    def _mock_load(kind: str, name: str) -> dict | None:  # noqa: ARG001
        return None

    _load = _exec_load_helper(mock_load_fn=_mock_load)
    ss: dict = {}
    _load("Nonexistent", ss)

    assert "_tf_prefill" not in ss, "Session state must be unchanged when load_view returns None."


def test_load_trade_no_op_on_blank_name() -> None:
    """Blank/whitespace name must not call load_view and leave session state untouched."""
    mock_load = MagicMock(return_value=None)
    _load = _exec_load_helper(mock_load_fn=mock_load)
    ss: dict = {}
    _load("  ", ss)

    mock_load.assert_not_called()
    assert "_tf_prefill" not in ss


# ── 4. Delete helper ──────────────────────────────────────────────────────────


def test_delete_view_called_in_page(source: str) -> None:
    """delete_view must be called somewhere in the page (for the Delete button)."""
    assert "delete_view" in source, "delete_view must be called in Trade Analyzer for the Delete button (R-3)."


# ── 5. Structural: list_views("trade") appears in the page ───────────────────


def test_list_views_called_with_trade_kind(source: str) -> None:
    """list_views(\"trade\") must appear in the page for populating the Load selectbox."""
    assert 'list_views("trade")' in source or "list_views('trade')" in source, (
        'list_views("trade") must be called in the Trade Analyzer page (R-3).'
    )


# ── 6. Integration: save then load round-trip via helpers ────────────────────


def test_save_load_roundtrip() -> None:
    """Save a trade then load it back; _tf_prefill should match the original ids."""
    store: dict[str, dict] = {}

    def _mock_save(kind: str, name: str, payload: dict) -> None:
        store[name] = payload

    def _mock_load(kind: str, name: str) -> dict | None:
        return store.get(name)

    _save = _exec_save_helper(mock_save_fn=_mock_save)
    _load = _exec_load_helper(mock_load_fn=_mock_load)

    _save("Round Trip", [7, 8], [9], None)

    ss: dict = {}
    _load("Round Trip", ss)

    assert ss.get("_tf_prefill", {}).get("giving_ids") == [7, 8]
    assert ss.get("_tf_prefill", {}).get("receiving_ids") == [9]
