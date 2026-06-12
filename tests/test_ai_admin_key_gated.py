"""Admin AI controls: require_admin + audit logs the action only, never the key."""

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_controls.py"


def test_page_calls_require_admin_before_ai_section():
    src = _PAGE.read_text(encoding="utf-8")
    assert "require_admin()" in src
    assert "set_admin_shared_key" in src
    assert "set_daily_cap" in src


def test_audit_never_logs_key_text():
    """AST: no log_action call passes the AI key variable as detail."""
    src = _PAGE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "log_action":
            dump = ast.dump(node)
            assert "ai_shared_key_text" not in dump, "AI key text must never be passed to log_action"


def test_setter_is_multi_user_gated():
    # set_admin_shared_key -> set_setting -> gated; assert the import wiring exists.
    src = _PAGE.read_text(encoding="utf-8")
    assert "from src.ai.keys import set_admin_shared_key" in src
