"""Plan 4 guard: main() starts the single-writer scheduler on the flag-ON path
only (after the flag-off early return), and start_background_refresh is
idempotent (one thread no matter how many times main() re-runs)."""

import ast
import threading
from pathlib import Path

_APP = Path(__file__).parent.parent / "app.py"


def test_app_imports_start_background_refresh():
    assert "from src.scheduler import start_background_refresh" in _APP.read_text(encoding="utf-8")


def test_main_starts_scheduler_after_flag_off_return():
    """AST: within main(), the line that calls start_background_refresh() must
    come AFTER the `return` that ends the flag-off (not multi_user_enabled())
    fast path — so v1 never starts a thread."""
    src = _APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")

    # Line of the flag-off early return: the `return` inside an
    # `if not multi_user_enabled():` block.
    flag_off_return_line = None
    for node in ast.walk(main):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(src, node.test) or ""
            if "multi_user_enabled" in test_src and ("not " in test_src or "is False" in test_src):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return):
                        flag_off_return_line = inner.lineno
    assert flag_off_return_line is not None, "no flag-off early return in main()"

    sched_call_line = None
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "start_background_refresh":
            sched_call_line = node.lineno
    assert sched_call_line is not None, "main() never calls start_background_refresh()"
    assert sched_call_line > flag_off_return_line, "start_background_refresh() must be AFTER the flag-off early return"


def test_start_background_refresh_idempotent(monkeypatch):
    """Calling start twice yields exactly one 'heater-refresh' thread."""
    import src.scheduler as scheduler

    # Make the refresh loop a no-op so the test doesn't touch the DB/network.
    monkeypatch.setattr("src.data_bootstrap.bootstrap_all_data", lambda *a, **k: {}, raising=False)
    try:
        scheduler.start_background_refresh()
        scheduler.start_background_refresh()
        named = [t for t in threading.enumerate() if t.name == "heater-refresh"]
        assert len(named) == 1
    finally:
        scheduler.stop_background_refresh()
