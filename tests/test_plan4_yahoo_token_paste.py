"""Plan 4 guard: the admin Yahoo-token paste path. The helper validates pasted
JSON and persists it to the volume (_AUTH_DIR); the page is a thin admin-gated
caller that audit-logs only the ACTION, never the token contents."""

import ast
import json
from pathlib import Path

from src.yahoo_api import save_yahoo_token_json

_PAGE = Path(__file__).parent.parent / "pages" / "_admin_controls.py"


def test_valid_token_written_to_auth_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    payload = {"refresh_token": "abc123", "access_token": "xyz", "token_type": "bearer"}
    ok, msg = save_yahoo_token_json(json.dumps(payload))
    assert ok is True
    written = tmp_path / "yahoo_token.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["refresh_token"] == "abc123"


def test_non_json_rejected_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json("this is not json {")
    assert ok is False
    assert "JSON" in msg
    assert not (tmp_path / "yahoo_token.json").exists()


def test_missing_refresh_token_rejected_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json(json.dumps({"access_token": "only-access"}))
    assert ok is False
    assert "refresh_token" in msg
    assert not (tmp_path / "yahoo_token.json").exists()


def test_non_object_json_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json(json.dumps(["not", "a", "dict"]))
    assert ok is False
    assert not (tmp_path / "yahoo_token.json").exists()


def test_page_admin_gated_and_logs_action_not_contents():
    """AST guard: page calls require_admin() + save_yahoo_token_json, logs the
    action string 'yahoo_token_update', and never passes the pasted text var to
    log_action (token contents must never be audit-logged)."""
    src = _PAGE.read_text(encoding="utf-8")
    assert "require_admin()" in src
    assert "save_yahoo_token_json" in src
    assert "yahoo_token_update" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "log_action":
            for arg in node.args:
                # No positional arg may be the pasted-text variable.
                assert getattr(arg, "id", None) != "_yahoo_token_text"


def test_admin_smoke_renders_yahoo_section(monkeypatch):
    from streamlit.testing.v1 import AppTest

    # The admin Yahoo-token section only renders under MULTI_USER. Set it
    # explicitly so the test does not depend on ambient env state left by another
    # test on the same xdist worker — the intermittent `-n auto` failure observed
    # 2026-06-02 (the "Yahoo" subheader was absent whenever MULTI_USER was off).
    monkeypatch.setenv("MULTI_USER", "1")

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "user_id": 1,
        "username": "admin",
        "is_admin": 1,
        "status": "active",
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run(timeout=60)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Yahoo" in s for s in subheaders)
