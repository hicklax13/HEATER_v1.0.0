"""Plan 4 guard: flag-on Home suppresses the v1 splash-screen bootstrap and
shows a 'warming up' gate driven by the single authoritative freshness signal
(_latest_successful_refresh). Flag-off Home is unchanged (covered in backcompat)."""

import ast
from pathlib import Path

_APP = Path(__file__).parent.parent / "app.py"


def _func_source(name: str) -> str:
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_APP.read_text(encoding="utf-8"), node)
    raise AssertionError(f"{name} not found in app.py")


def test_app_imports_refresh_snapshot():
    # The scheduler import is added in Task 6 (imported where used, to avoid an
    # F401 between tasks); the scheduler-wiring test guards that one.
    src = _APP.read_text(encoding="utf-8")
    assert "get_refresh_log_snapshot" in src


def test_render_single_user_app_branches_on_flag():
    body = _func_source("render_single_user_app")
    assert "multi_user_enabled()" in body, "Home must branch on the flag"
    assert "_render_multiuser_home_gate" in body, "flag-on path calls the gate"


def test_flag_on_path_suppresses_splash():
    """In render_single_user_app, the multi_user_enabled() branch must call the
    gate; render_splash_screen stays on the flag-off (else) side only."""
    body = _func_source("render_single_user_app")
    tree = ast.parse(body)
    # Find the `if multi_user_enabled():` node and assert render_splash_screen is
    # NOT called inside its body, but _render_multiuser_home_gate IS.
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(body, node.test) or ""
            if "multi_user_enabled" in test_src:
                found = True
                if_body_src = "\n".join(ast.get_source_segment(body, n) or "" for n in node.body)
                assert "_render_multiuser_home_gate" in if_body_src
                assert "render_splash_screen" not in if_body_src
    assert found, "no `if multi_user_enabled():` branch in render_single_user_app"


def test_latest_successful_refresh_none_when_empty(monkeypatch):
    import app

    monkeypatch.setattr(app, "get_refresh_log_snapshot", lambda: [])
    assert app._latest_successful_refresh() is None


def test_latest_successful_refresh_none_when_only_errors(monkeypatch):
    import app

    monkeypatch.setattr(
        app,
        "get_refresh_log_snapshot",
        lambda: [{"status": "error", "last_refresh": "2026-05-31T10:00:00"}],
    )
    assert app._latest_successful_refresh() is None


def test_latest_successful_refresh_picks_latest_success(monkeypatch):
    import app

    monkeypatch.setattr(
        app,
        "get_refresh_log_snapshot",
        lambda: [
            {"status": "success", "last_refresh": "2026-05-31T09:00:00"},
            {"status": "error", "last_refresh": "2026-05-31T11:00:00"},
            {"status": "success", "last_refresh": "2026-05-31T10:30:00"},
        ],
    )
    latest = app._latest_successful_refresh()
    assert latest is not None
    assert latest.hour == 10 and latest.minute == 30
