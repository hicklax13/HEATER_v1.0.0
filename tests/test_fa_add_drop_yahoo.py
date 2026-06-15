"""TDD tests for R-1: confirm-gated "Add/Drop on Yahoo" action on Free Agents page.

Covers:
  1. `resolve_add_drop_keys` helper — FA player_key + roster yahoo_player_key resolution.
  2. Page structural guards — client gate (is_connected only), two-step confirm
     pattern, add_drop import/call, failure path with manual fallback.
  3. Unit tests for the key-resolver with missing-key graceful cases.

No real network calls; no Streamlit runtime needed for the structural tests.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

_PAGE_PATH = pathlib.Path("pages/14_Free_Agents.py")


def _src() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Key-resolver helper (unit tests)
# ---------------------------------------------------------------------------


class TestResolveAddDropKeys:
    """Unit tests for the `resolve_add_drop_keys` helper extracted to the page."""

    def test_helper_importable_from_page_module(self):
        """The helper must be defined in the page source (importable via AST)."""
        src = _src()
        tree = ast.parse(src)
        fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "resolve_add_drop_keys" in fn_names, (
            "pages/14_Free_Agents.py must define a `resolve_add_drop_keys` function "
            "to keep the key-resolution logic testable in isolation."
        )

    def test_helper_returns_add_key_from_rec_player_key(self):
        """When the rec carries a `player_key`, that value is returned as the add key."""
        # Import the helper directly from the module source via exec
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        # Rec with a player_key field (Yahoo FA data carries this)
        rec = {"add_player_id": 42, "player_key": "469.p.99999", "add_name": "Test Player"}
        drop_pid = 7
        roster_df = pd.DataFrame({"player_id": [7], "yahoo_player_key": ["469.p.77777"], "name": ["Drop Guy"]})
        fa_df = pd.DataFrame(
            {
                "player_id": [42],
                "player_key": ["469.p.99999"],
                "player_name": ["Test Player"],
            }
        )

        add_key, drop_key, err = fn(rec, drop_pid, fa_df, roster_df)
        assert add_key == "469.p.99999", f"Expected add key '469.p.99999', got {add_key!r}"
        assert drop_key == "469.p.77777", f"Expected drop key '469.p.77777', got {drop_key!r}"
        assert err is None, f"Expected no error, got {err!r}"

    def test_helper_returns_friendly_error_when_add_key_missing(self):
        """When no player_key is resolvable for the add player, err must be a non-empty string."""
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        rec = {"add_player_id": 42, "add_name": "Mystery Player"}  # No player_key
        drop_pid = 7
        # FA pool has no player_key for this player
        fa_df = pd.DataFrame({"player_id": [42], "player_name": ["Mystery Player"]})
        roster_df = pd.DataFrame({"player_id": [7], "yahoo_player_key": ["469.p.77777"], "name": ["Drop Guy"]})

        add_key, drop_key, err = fn(rec, drop_pid, fa_df, roster_df)
        assert add_key is None, f"Expected add_key=None when unresolvable, got {add_key!r}"
        assert err is not None and len(err) > 0, "Expected a non-empty error string"
        # The error must mention what the user should do manually
        assert "manually" in err.lower() or "yahoo" in err.lower(), (
            "Error must tell user to apply move manually in Yahoo. Got: {err!r}"
        )

    def test_helper_returns_friendly_error_when_drop_key_missing(self):
        """When the roster player has no yahoo_player_key, err is a non-empty string."""
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        rec = {"add_player_id": 42, "player_key": "469.p.99999", "add_name": "Test Player"}
        drop_pid = 7
        fa_df = pd.DataFrame({"player_id": [42], "player_key": ["469.p.99999"], "player_name": ["Test Player"]})
        # Roster row has no yahoo_player_key (or empty string)
        roster_df = pd.DataFrame({"player_id": [7], "yahoo_player_key": [""], "name": ["Drop Guy"]})

        add_key, drop_key, err = fn(rec, drop_pid, fa_df, roster_df)
        assert drop_key is None, f"Expected drop_key=None when unresolvable, got {drop_key!r}"
        assert err is not None and len(err) > 0, "Expected a non-empty error string"

    def test_helper_handles_empty_roster_df(self):
        """When the roster DataFrame is empty, the function must return a graceful error."""
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        rec = {"add_player_id": 42, "player_key": "469.p.99999", "add_name": "Test Player"}
        drop_pid = 7
        fa_df = pd.DataFrame({"player_id": [42], "player_key": ["469.p.99999"], "player_name": ["Test Player"]})
        roster_df = pd.DataFrame()  # empty

        # Must not raise
        add_key, drop_key, err = fn(rec, drop_pid, fa_df, roster_df)
        assert err is not None  # can't resolve drop from empty roster

    def test_helper_fa_df_lookup_falls_back_to_rec_player_key(self):
        """When fa_df has no player_key column but rec itself has player_key, use rec."""
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        rec = {"add_player_id": 42, "player_key": "469.p.99999", "add_name": "Test Player"}
        drop_pid = 7
        # fa_df has no player_key column at all
        fa_df = pd.DataFrame({"player_id": [42], "player_name": ["Test Player"]})
        roster_df = pd.DataFrame({"player_id": [7], "yahoo_player_key": ["469.p.77777"], "name": ["Drop Guy"]})

        add_key, drop_key, err = fn(rec, drop_pid, fa_df, roster_df)
        # Should use rec["player_key"] directly
        assert add_key == "469.p.99999"
        assert drop_key == "469.p.77777"
        assert err is None


# ---------------------------------------------------------------------------
# 2. Structural page guards
# ---------------------------------------------------------------------------


class TestAddDropPageStructure:
    """Source-level guards ensuring the two-step confirm flow exists in the page."""

    def test_page_imports_add_drop_aware_client_access(self):
        """The page must use is_connected() to gate the Add/Drop button.

        The function is_connected() is the canonical check for whether the
        token-owner's client is available for writes.
        """
        src = _src()
        assert "is_connected" in src, (
            "pages/14_Free_Agents.py must call yds.is_connected() (or similar) to "
            "gate the Add/Drop on Yahoo button so read-only members never see it."
        )

    def test_page_has_add_drop_button_label(self):
        """The page must render an 'Add/Drop on Yahoo' button."""
        src = _src()
        # Accept the label with or without emoji
        assert "Add/Drop on Yahoo" in src, (
            "pages/14_Free_Agents.py must render an 'Add/Drop on Yahoo' button "
            "inside the Recommended Adds/Drops section."
        )

    def test_page_has_two_step_confirm_pattern(self):
        """The two-step confirm must use a session_state pending flag.

        Acceptable patterns:
          st.session_state["fa_add_drop_pending_..."]  or
          "fa_add_drop_pending" in st.session_state
        """
        src = _src()
        assert "fa_add_drop_pending" in src, (
            "The Add/Drop flow must use a session_state key named "
            "'fa_add_drop_pending' (or similar) to store the pending move so "
            "the mandatory two-step confirm is enforced across reruns."
        )

    def test_page_has_confirm_and_cancel_buttons(self):
        """Both 'Confirm & apply to Yahoo' and 'Cancel' buttons must exist."""
        src = _src()
        assert "Confirm" in src and "apply to Yahoo" in src, (
            "The two-step confirm must include a 'Confirm & apply to Yahoo' button. "
            "This is the ONLY path that triggers the actual write — clicking the "
            "initial 'Add/Drop on Yahoo' button must NOT immediately submit."
        )
        # Cancel button (label or keyword)
        assert "Cancel" in src, (
            "The two-step confirm must include a 'Cancel' button that clears "
            "the pending state without submitting to Yahoo."
        )

    def test_page_calls_add_drop_on_client(self):
        """The page must call .add_drop() (or client.add_drop()) for the write."""
        src = _src()
        assert ".add_drop(" in src, (
            "pages/14_Free_Agents.py must call <client>.add_drop(add_key, drop_key) "
            "on Confirm. This is the only write path to Yahoo."
        )

    def test_page_handles_ok_false_with_manual_fallback(self):
        """When add_drop returns ok=False, the page must render both an error AND
        a manual fallback message telling the user the player names."""
        src = _src()
        # The page must handle the not-ok case
        assert 'result.get("ok")' in src or '"ok"' in src or "result[" in src, (
            "The page must inspect the add_drop result dict for 'ok' key."
        )
        # Check for the manual fallback message guidance
        assert "manually" in src.lower(), (
            "When Yahoo add_drop fails, the page must tell the user to apply "
            "the move manually in Yahoo (include the word 'manually' in the "
            "fallback message)."
        )

    def test_page_shows_connect_caption_when_not_connected(self):
        """When yds.is_connected() is False, show a caption instead of the button."""
        src = _src()
        # Look for a caption/info near is_connected that mentions connecting
        assert "Connect Yahoo" in src or "connect Yahoo" in src or "token" in src.lower(), (
            "When Yahoo is not connected, the page must show a caption explaining "
            "that the token owner must connect Yahoo to apply moves. "
            "Expected text like 'Connect Yahoo (as the team owner) to apply moves.'"
        )

    def test_add_drop_not_called_outside_confirm_branch(self):
        """The .add_drop() call must only appear AFTER the Confirm button check.

        We verify it's not at module top-level (i.e., the call is inside a
        conditional block that corresponds to the Confirm step).
        """
        src = _src()
        tree = ast.parse(src)
        # Find all Call nodes for .add_drop()
        add_drop_call_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_drop":
                add_drop_call_lines.append(node.lineno)

        # All .add_drop() calls must be inside an If block (not at module top-level)
        # We use a simple heuristic: check indentation in source lines
        source_lines = src.splitlines()
        for lineno in add_drop_call_lines:
            line = source_lines[lineno - 1]
            indent = len(line) - len(line.lstrip())
            assert indent >= 8, (  # at least 2 levels deep (if + if)
                f"Line {lineno}: `.add_drop(...)` call appears at indent={indent} "
                "(expected >= 8, i.e. inside at least two nested if-blocks). "
                "The write must only happen inside the Confirm conditional."
            )

    def test_preview_shows_add_and_drop_names(self):
        """The confirm preview must render both the add and drop player names."""
        src = _src()
        # The preview step should mention both players before the write
        # We look for patterns like "Add: {name}" or "Drop: {name}" in the confirm UI
        has_add_preview = "Add:" in src or "add_name" in src
        has_drop_preview = "Drop:" in src or "drop_name" in src
        assert has_add_preview, (
            "The two-step confirm preview must show 'Add: <player name>' so "
            "the user knows exactly what they're confirming."
        )
        assert has_drop_preview, (
            "The two-step confirm preview must show 'Drop: <player name>' so "
            "the user knows exactly what they're confirming."
        )


# ---------------------------------------------------------------------------
# 3. Integration-style unit test: add_drop result handling
# ---------------------------------------------------------------------------


class TestAddDropResultHandling:
    """Tests that the page correctly handles both ok=True and ok=False results."""

    def test_resolve_add_drop_keys_signature(self):
        """Helper must accept (rec, drop_player_id, fa_df, roster_df) and return
        a 3-tuple (add_key, drop_key, error_msg)."""
        globs: dict = {}
        exec(compile(_src(), str(_PAGE_PATH), "exec"), globs)  # noqa: S102
        fn = globs.get("resolve_add_drop_keys")
        if fn is None:
            pytest.skip("resolve_add_drop_keys not yet implemented")

        import inspect

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert len(params) == 4, (
            f"resolve_add_drop_keys must have exactly 4 parameters "
            f"(rec, drop_player_id, fa_df, roster_df), got: {params}"
        )

    def test_page_uses_st_success_on_ok(self):
        """On a successful add_drop (ok=True), the page must call st.success(...)."""
        src = _src()
        # Look for st.success with a message about Yahoo submission
        assert "st.success" in src, (
            "The page must call st.success(...) after a successful add_drop. "
            "This is the user feedback for a completed transaction."
        )
        # The success message should reference Yahoo or submission
        idx = src.find("st.success")
        while idx != -1:
            snippet = src[idx : idx + 150]
            if "Yahoo" in snippet or "submitted" in snippet or "apply" in snippet.lower():
                break
            idx = src.find("st.success", idx + 1)
        else:
            # No st.success with Yahoo/submitted found — fail
            assert False, (
                "st.success message must mention Yahoo or 'submitted' so the user "
                "knows the transaction was sent to Yahoo (not just saved locally)."
            )

    def test_page_uses_st_error_on_not_ok(self):
        """On a failed add_drop (ok=False), the page must call st.error(...)
        with the error message from the result dict."""
        src = _src()
        assert "st.error" in src, "The page must call st.error(...) when add_drop returns ok=False."
        # The error rendering should use the result's error field
        assert 'result["error"]' in src or "result.get" in src, (
            "The st.error call must display result['error'] (the Yahoo error message), not a hardcoded string."
        )

    def test_page_reads_client_from_yds(self):
        """The page must access the Yahoo client via the existing yds service object,
        not create a new YahooFantasyClient."""
        src = _src()
        # The page already has `yds = get_yahoo_data_service()` — it should
        # use yds._client or a thin wrapper, never instantiate YahooFantasyClient directly.
        assert "YahooFantasyClient()" not in src, (
            "pages/14_Free_Agents.py must NOT instantiate YahooFantasyClient() directly. "
            "Reach the client via the existing yds service object."
        )
        # Must use yds for the client
        has_client_access = "yds._client" in src or "yds.get_client" in src or "_client" in src
        assert has_client_access, (
            "The page must access the Yahoo client through the existing yds service. "
            "Use yds._client.add_drop(...) or add a thin accessor."
        )
