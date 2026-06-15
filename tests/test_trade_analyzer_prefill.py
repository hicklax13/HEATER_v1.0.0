"""Tests for Trade Finder → Trade Analyzer prefill hand-off.

Task: when `_tf_prefill` is in session state, the Trade Analyzer must
pre-populate the 'giving' / 'receiving' multiselect keys and then clear
`_tf_prefill` so it doesn't re-apply on subsequent reruns.

Coverage strategy:
1. Pure unit-tests of `_consume_trade_prefill` helper (no Streamlit needed).
2. Structural AST check that the page wires the helper before the multiselects.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


# ── Fixture: import the helper without importing the full page ────────────────
# The page contains top-level Streamlit calls, so we can't `import` it normally.
# Instead we exec only the helper function from the source.


def _load_helper():
    """Extract and exec `_consume_trade_prefill` from the page source."""
    source = PAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Find the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_consume_trade_prefill":
            # Compile just this function
            func_source = ast.get_source_segment(source, node)
            if func_source is None:
                # Fallback: slice by line numbers
                lines = source.splitlines()
                func_lines = lines[node.lineno - 1 : node.end_lineno]
                func_source = "\n".join(func_lines)
            ns: dict = {}
            exec(compile(func_source, PAGE_PATH.name, "exec"), ns)  # noqa: S102
            return ns["_consume_trade_prefill"]
    return None


@pytest.fixture(scope="module")
def consume_prefill():
    fn = _load_helper()
    assert fn is not None, (
        "_consume_trade_prefill not found in pages/11_Trade_Analyzer.py — add the helper before implementing."
    )
    return fn


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


# ── Unit tests of the pure helper ────────────────────────────────────────────


def test_prefill_populates_giving_and_receiving(consume_prefill):
    """Valid prefill: giving/receiving keys set from id→name maps."""
    ss = {
        "_tf_prefill": {"giving_ids": [1, 2], "receiving_ids": [3]},
    }
    id_to_name_give = {1: "Aaron Judge", 2: "Manny Machado", 4: "Shohei Ohtani"}
    id_to_name_receive = {3: "Freddie Freeman", 5: "Jose Ramirez"}

    consume_prefill(ss, id_to_name_give, id_to_name_receive)

    assert ss.get("giving") == ["Aaron Judge", "Manny Machado"]
    assert ss.get("receiving") == ["Freddie Freeman"]


def test_prefill_clears_tf_prefill_key(consume_prefill):
    """After consuming, `_tf_prefill` must be removed from session state."""
    ss = {
        "_tf_prefill": {"giving_ids": [10], "receiving_ids": [20]},
    }
    id_to_name_give = {10: "Player A"}
    id_to_name_receive = {20: "Player B"}

    consume_prefill(ss, id_to_name_give, id_to_name_receive)

    assert "_tf_prefill" not in ss, "_tf_prefill must be popped after consuming"


def test_prefill_ignores_invalid_ids(consume_prefill):
    """IDs not present in the option maps are silently skipped."""
    ss = {
        "_tf_prefill": {"giving_ids": [1, 999], "receiving_ids": [3, 888]},
    }
    id_to_name_give = {1: "Aaron Judge"}
    id_to_name_receive = {3: "Freddie Freeman"}

    consume_prefill(ss, id_to_name_give, id_to_name_receive)

    assert ss.get("giving") == ["Aaron Judge"]
    assert ss.get("receiving") == ["Freddie Freeman"]
    assert "_tf_prefill" not in ss


def test_prefill_no_op_when_absent(consume_prefill):
    """When `_tf_prefill` is not in session state, nothing changes."""
    ss: dict = {}
    consume_prefill(ss, {1: "Player A"}, {2: "Player B"})

    assert "giving" not in ss
    assert "receiving" not in ss
    assert "_tf_prefill" not in ss


def test_prefill_malformed_dict_ignored(consume_prefill):
    """Malformed prefill (missing keys) doesn't crash and leaves no side-effects."""
    ss = {"_tf_prefill": {"bad_key": [1, 2]}}
    consume_prefill(ss, {1: "Player A"}, {2: "Player B"})
    # Should not raise and should clear the bad prefill
    assert "_tf_prefill" not in ss


def test_prefill_non_dict_ignored(consume_prefill):
    """Non-dict prefill value is discarded silently."""
    ss = {"_tf_prefill": "garbage"}
    consume_prefill(ss, {1: "Player A"}, {2: "Player B"})
    assert "_tf_prefill" not in ss
    assert "giving" not in ss
    assert "receiving" not in ss


def test_prefill_empty_lists(consume_prefill):
    """Empty id lists produce empty selections."""
    ss = {"_tf_prefill": {"giving_ids": [], "receiving_ids": []}}
    consume_prefill(ss, {1: "Player A"}, {2: "Player B"})

    assert ss.get("giving") == []
    assert ss.get("receiving") == []
    assert "_tf_prefill" not in ss


# ── Structural: helper is called before the multiselects ─────────────────────


def test_consume_prefill_defined_in_page(source: str):
    """Page must define `_consume_trade_prefill` function."""
    assert "_consume_trade_prefill" in source, "pages/11_Trade_Analyzer.py must define _consume_trade_prefill()."


def test_consume_prefill_called_before_multiselect(source: str):
    """_consume_trade_prefill() must be called before st.multiselect(key='giving')."""
    prefill_pos = source.find("_consume_trade_prefill(")
    giving_pos = source.find('key="giving"')
    assert prefill_pos != -1, "_consume_trade_prefill() call not found in page."
    assert giving_pos != -1, 'st.multiselect(key="giving") not found in page.'
    assert prefill_pos < giving_pos, "_consume_trade_prefill() must be called BEFORE the 'giving' multiselect."


def test_tf_prefill_not_persisted_after_call(source: str):
    """The helper source must contain 'pop' (or 'del') to clear _tf_prefill."""
    # Extract the raw source of _consume_trade_prefill via AST line offsets.
    tree = ast.parse(source)
    fn_source = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_consume_trade_prefill":
            lines = source.splitlines()
            fn_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            break

    assert fn_source is not None, "_consume_trade_prefill not found in page."
    has_removal = ".pop(" in fn_source or "del ss[" in fn_source or "del session_state[" in fn_source
    assert has_removal, "_consume_trade_prefill must remove '_tf_prefill' via .pop() or del."
