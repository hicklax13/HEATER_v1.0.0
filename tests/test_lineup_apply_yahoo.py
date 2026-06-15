"""TDD tests for R-1: confirm-gated "Apply lineup to Yahoo" action.

Covers:
1. _build_lineup_assignments(result, roster) pure helper — unit-tested directly.
2. Page imports set_lineup via yds._client (structural check).
3. Two-step confirm flow: pending flag + Confirm control distinct from initial Apply.
4. Failure path renders manual-moves fallback.
5. Button hidden / caption shown when Yahoo write is unavailable.
6. Button hidden when no optimizer result yet.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "2_Line-up_Optimizer.py"
_PAGE_TEXT: str = _PAGE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper: extract the _build_lineup_assignments function from the page src
# ---------------------------------------------------------------------------


def _extract_and_exec_helper(func_name: str) -> object:
    """Parse the page AST, find func_name, exec it, return the callable."""
    tree = ast.parse(_PAGE_TEXT)
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            break
    assert func_node is not None, f"{func_name} must be defined in the page"
    lines = _PAGE_TEXT.splitlines()[func_node.lineno - 1 : func_node.end_lineno]
    src = "\n".join(lines)
    ns: dict = {}
    exec(compile(src, f"<{func_name}>", "exec"), ns)  # noqa: S102
    return ns[func_name]


# ---------------------------------------------------------------------------
# 1. _build_lineup_assignments helper — unit tests
# ---------------------------------------------------------------------------


def test_build_lineup_assignments_defined_in_page():
    assert "_build_lineup_assignments" in _PAGE_TEXT, "page must define a _build_lineup_assignments helper"


def test_build_lineup_assignments_returns_list_of_dicts():
    fn = _extract_and_exec_helper("_build_lineup_assignments")

    # Minimal result with starter player_ids and their LP-recommended slots
    result = {
        "lp_slot_map": {1: "C", 2: "SP"},
    }
    # Roster has yahoo_player_key for each player_id
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "yahoo_player_key": ["469.p.1001", "469.p.1002", "469.p.1003"],
        }
    )
    assignments = fn(result, roster)
    assert isinstance(assignments, list), "must return a list"
    assert len(assignments) == 2
    for a in assignments:
        assert "player_key" in a
        assert "position" in a

    keys = {a["player_key"] for a in assignments}
    assert "469.p.1001" in keys
    assert "469.p.1002" in keys


def test_build_lineup_assignments_skips_missing_yahoo_key():
    fn = _extract_and_exec_helper("_build_lineup_assignments")

    result = {"lp_slot_map": {1: "OF", 2: "BN"}}
    roster = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["Alice", "Bob"],
            # player 2 has no yahoo_player_key
            "yahoo_player_key": ["469.p.1001", ""],
        }
    )
    assignments = fn(result, roster)
    # Only player 1 should be included (player 2 has empty key)
    assert len(assignments) == 1
    assert assignments[0]["player_key"] == "469.p.1001"
    assert assignments[0]["position"] == "OF"


def test_build_lineup_assignments_empty_result_returns_empty():
    fn = _extract_and_exec_helper("_build_lineup_assignments")

    assert fn({}, pd.DataFrame()) == []
    assert fn({"lp_slot_map": {}}, pd.DataFrame({"player_id": [], "yahoo_player_key": []})) == []


def test_build_lineup_assignments_no_lp_slot_map_key_returns_empty():
    fn = _extract_and_exec_helper("_build_lineup_assignments")
    roster = pd.DataFrame({"player_id": [1], "yahoo_player_key": ["469.p.1"]})
    # result has starter_ids but no lp_slot_map
    result = {"starter_ids": [1]}
    assignments = fn(result, roster)
    assert assignments == [], "without lp_slot_map, helper must return []"


# ---------------------------------------------------------------------------
# 2. Page accesses set_lineup via yds._client (structural)
# ---------------------------------------------------------------------------


def test_page_calls_set_lineup_via_client():
    """Page must call set_lineup through the yds._client attribute."""
    assert "set_lineup" in _PAGE_TEXT, "page must call set_lineup to apply the lineup to Yahoo"


def test_page_accesses_yds_client_for_write():
    """Page must access yds._client (or equivalent) to reach the Yahoo write client."""
    assert "yds._client" in _PAGE_TEXT or "_client.set_lineup" in _PAGE_TEXT, (
        "page must reach the Yahoo write client via yds._client"
    )


# ---------------------------------------------------------------------------
# 3. Two-step confirm flow
# ---------------------------------------------------------------------------


def test_page_has_apply_lineup_pending_flag():
    """Page must use a session-state pending flag to gate the confirm step."""
    assert "_apply_lineup_pending" in _PAGE_TEXT, (
        'page must store a pending flag in st.session_state["_apply_lineup_pending"] '
        "to implement the two-step confirm flow"
    )


def test_page_has_apply_lineup_button():
    """Page must render the initial Apply button."""
    assert "Apply" in _PAGE_TEXT and "Yahoo" in _PAGE_TEXT, (
        'page must render an "Apply this lineup to Yahoo" (or similar) button'
    )
    # More specific: there must be a button whose label contains both "Apply" and "Yahoo"
    # (case-insensitive substring on the label strings in the source)
    lower = _PAGE_TEXT.lower()
    # Find "apply" within 60 chars of "yahoo" — both words near each other
    idx_apply = lower.find("apply")
    while idx_apply != -1:
        snippet = lower[max(0, idx_apply - 20) : idx_apply + 80]
        if "yahoo" in snippet:
            break
        idx_apply = lower.find("apply", idx_apply + 1)
    assert idx_apply != -1, 'page must include a button/label combining "Apply" and "Yahoo" near each other'


def test_page_has_confirm_apply_button_distinct_from_initial():
    """The Confirm button must be a DIFFERENT widget from the initial Apply button."""
    # The confirm button should have a different key from the initial apply button.
    assert "Confirm" in _PAGE_TEXT and "apply" in _PAGE_TEXT.lower(), (
        'page must have a "Confirm & apply" button separate from the initial Apply trigger'
    )
    # Both "apply_lineup_confirm" (or similar) and the initial trigger key must exist
    assert "apply_lineup_confirm" in _PAGE_TEXT or "_confirm_apply" in _PAGE_TEXT, (
        "the Confirm button must have a distinct session-state key (apply_lineup_confirm or _confirm_apply)"
    )


def test_page_has_cancel_button_for_pending_apply():
    """Page must render a Cancel button that clears the pending flag."""
    assert "Cancel" in _PAGE_TEXT, "page must include a Cancel button to abort the pending apply"
    assert "_apply_lineup_pending" in _PAGE_TEXT, (
        "the Cancel action must clear st.session_state['_apply_lineup_pending']"
    )


def test_pending_flag_cleared_on_cancel_in_source():
    """The source must set _apply_lineup_pending to False (or delete it) somewhere."""
    # Either del st.session_state["_apply_lineup_pending"] or set to False
    lower = _PAGE_TEXT
    cleared = (
        'st.session_state["_apply_lineup_pending"] = False' in lower
        or 'del st.session_state["_apply_lineup_pending"]' in lower
        or "st.session_state['_apply_lineup_pending'] = False" in lower
    )
    assert cleared, "page must clear st.session_state['_apply_lineup_pending'] on Cancel (set to False or del)"


# ---------------------------------------------------------------------------
# 4. Failure path: manual-moves fallback rendered
# ---------------------------------------------------------------------------


def test_page_renders_manual_fallback_on_failure():
    """When set_lineup returns ok=False, page must list moves as text."""
    # The page must have a branch for the not-ok case that lists assignments.
    # We check for the error rendering and a text fallback like "Set manually".
    assert 'result["ok"]' in _PAGE_TEXT or 'result.get("ok")' in _PAGE_TEXT or "_apply_result" in _PAGE_TEXT, (
        "page must inspect the set_lineup result dict for the 'ok' key"
    )
    lower = _PAGE_TEXT.lower()
    assert "manually" in lower or "manual" in lower, "failure path must suggest the user apply moves manually in Yahoo"


def test_page_shows_error_on_set_lineup_failure():
    """When set_lineup returns ok=False, page must call st.error with the error message."""
    assert "st.error" in _PAGE_TEXT, "page must call st.error when set_lineup fails"


# ---------------------------------------------------------------------------
# 5. Write-gate: button hidden / caption shown when no writable client
# ---------------------------------------------------------------------------


def test_page_gates_apply_button_on_is_connected():
    """Apply button must only show when yds.is_connected() (and viewer_can_write)."""
    # Both gating calls must appear in the page.
    assert "is_connected" in _PAGE_TEXT, "Apply button must be gated on yds.is_connected()"
    assert "viewer_can_write" in _PAGE_TEXT, "Apply button must be gated on viewer_can_write()"
    # The gate and the apply button label must both appear in the source.
    # They are guaranteed to coexist because the button is nested under the gate.
    assert "Apply this lineup to Yahoo" in _PAGE_TEXT or "apply_lineup_to_yahoo" in _PAGE_TEXT, (
        "Apply button must be present in the page"
    )
    # Confirm both the gate AND the button label appear — their proximity is
    # enforced by the is_connected block nesting the button.
    lower = _PAGE_TEXT.lower()
    # Find the is_connected closest to "apply_lineup_to_yahoo"
    apply_idx = lower.find("apply_lineup_to_yahoo")
    assert apply_idx != -1, "apply_lineup_to_yahoo key not found"
    # Search for is_connected within 500 chars before the button key
    nearby = lower[max(0, apply_idx - 500) : apply_idx + 100]
    assert "is_connected" in nearby, "is_connected() check must appear shortly before the Apply button key"


def test_page_shows_caption_when_yahoo_not_connected():
    """Page must show a caption/message when Yahoo write is not available."""
    lower = _PAGE_TEXT.lower()
    # Should mention "connect" or "owner" near the apply button section
    assert "connect yahoo" in lower or "team owner" in lower or "yahoo (as" in lower, (
        "page must show a 'Connect Yahoo (as the team owner) to apply lineups' (or similar) caption when not connected"
    )


# ---------------------------------------------------------------------------
# 6. Button hidden when no result yet
# ---------------------------------------------------------------------------


def test_apply_button_guarded_by_optimizer_result():
    """Apply button must not render unless there is an optimizer result."""
    # The apply button must be inside the result-display branch, not at top level.
    tree = ast.parse(_PAGE_TEXT)

    # Find If nodes that reference "lineup_optimizer_result" in session state
    result_if_lines: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Check if the test references lineup_optimizer_result
        src_snippet = ast.unparse(node.test)
        if "lineup_optimizer_result" in src_snippet or "result" in src_snippet:
            result_if_lines.append((node.lineno, node.end_lineno))

    # Find the line where the Apply button KEY is defined — more stable than
    # searching for "Apply" + "Yahoo" + "button" on the same line.
    apply_line = None
    for i, line in enumerate(_PAGE_TEXT.splitlines(), start=1):
        if "apply_lineup_to_yahoo" in line:
            apply_line = i
            break

    assert apply_line is not None, (
        "could not locate the Apply button line — must contain the key 'apply_lineup_to_yahoo'"
    )

    # The apply button line must fall inside at least one result-gating if block
    inside = any(lo <= apply_line <= hi for lo, hi in result_if_lines)
    assert inside, (
        f"Apply button at line {apply_line} must be inside an if-block that "
        "checks for an optimizer result (guarded by lineup_optimizer_result)"
    )
