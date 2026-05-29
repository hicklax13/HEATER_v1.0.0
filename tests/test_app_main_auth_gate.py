"""Structural guard: app.py main() gates auth between init_db and splash."""

import ast
from pathlib import Path

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
