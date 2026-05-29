"""Structural guard: app.py main() gates auth between init_db and splash.

The AST tests lock the *placement* of require_auth() in main(); the AppTest
smoke at the bottom locks its *runtime effect*: with MULTI_USER on and no
session, app.py must render the login screen and stop BEFORE the splash /
bootstrap ever runs. (The stop sitting above render_splash_screen() is what
keeps this smoke cheap — the 33-phase bootstrap is never reached.)
"""

import ast
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _main_node(src: str):
    """The ast.FunctionDef node for main()."""
    tree = ast.parse(src)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _main_body_calls():
    """Ordered list of top-level call-expression names inside main()."""
    main = _main_node(_APP.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def _main_src():
    """Source text of main() only, so substring checks ignore defs elsewhere.

    A whole-file ``str.index`` collides on names like ``render_splash_screen()``
    whose ``def ...():`` line contains the same substring as the call site.
    """
    src = _APP.read_text(encoding="utf-8")
    return ast.get_source_segment(src, _main_node(src))


def test_require_auth_called_in_main():
    assert "require_auth" in _main_body_calls()


def test_require_auth_after_init_db_before_splash():
    body = _main_src()
    i_init = body.index("init_db()")
    i_auth = body.index("require_auth()")
    i_splash = body.index("render_splash_screen()")
    assert i_init < i_auth < i_splash, "require_auth() must sit between init_db() and splash"


def test_require_auth_imported():
    src = _APP.read_text(encoding="utf-8")
    assert "from src.auth import" in src and "require_auth" in src


@pytest.fixture
def _temp_db(tmp_path, monkeypatch):
    """Isolate app.py's init_db() onto a throwaway SQLite file."""
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "app_auth_gate.db")


def test_flag_on_no_session_renders_login_and_stops(_temp_db, monkeypatch):
    """MULTI_USER on + no session ⇒ login screen shown, splash never reached."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)  # no bootstrap admin
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    at = AppTest.from_file(str(_APP))
    # Generous timeout: app.py's first import pulls heavy ML/data deps
    # (pybaseball, statsapi, PyMC). The gate itself halts in milliseconds.
    at.run(timeout=60)

    assert not at.exception, [str(e) for e in at.exception]
    # require_auth() rendered the sign-in title and st.stop()'d the script.
    assert any("sign in" in t.value.lower() for t in at.title), [t.value for t in at.title]
    # The splash/bootstrap path was never entered — proof the gate halted early.
    # (AppTest session_state routes .get() through __getattr__, so use `in`.)
    assert "bootstrap_complete" not in at.session_state
