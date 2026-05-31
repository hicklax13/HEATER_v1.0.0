"""Plan 4 guard: MULTI_USER off ⇒ main() renders single-user v1 and NEVER starts
the scheduler. Mirrors the test_admin_backcompat.py pattern."""

import ast
from pathlib import Path

import pytest

_APP = Path(__file__).parent.parent / "app.py"


@pytest.fixture
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


def test_main_flag_off_renders_single_user_and_no_scheduler(_flag_off, monkeypatch):
    import app

    monkeypatch.setattr(app, "multi_user_enabled", lambda: False)

    calls = {"single_user": 0, "scheduler": 0}
    monkeypatch.setattr(app, "init_session", lambda: None)
    monkeypatch.setattr(app, "inject_custom_css", lambda: None)
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(
        app,
        "render_single_user_app",
        lambda: calls.__setitem__("single_user", calls["single_user"] + 1),
    )
    monkeypatch.setattr(
        app,
        "start_background_refresh",
        lambda: calls.__setitem__("scheduler", calls["scheduler"] + 1),
    )
    # require_auth must never be reached on the flag-off path; make it explode if it is.
    monkeypatch.setattr(
        app,
        "require_auth",
        lambda *a, **k: pytest.fail("require_auth reached on flag-off path"),
    )

    app.main()

    assert calls["single_user"] == 1, "flag-off must render the single-user app once"
    assert calls["scheduler"] == 0, "flag-off must NOT start the scheduler"


def test_flag_off_home_still_renders_splash():
    """AST: render_single_user_app keeps render_splash_screen on the flag-off
    (else) side — v1 splash bootstrap preserved."""
    src = _APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render_single_user_app")
    fn_src = ast.get_source_segment(src, fn)
    assert "render_splash_screen" in fn_src, "v1 splash must remain in Home"
