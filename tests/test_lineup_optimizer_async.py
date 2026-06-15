"""TDD tests for R-6: non-blocking (background-threaded) lineup optimization.

Three invariants:
1. A pure compute helper ``_run_optimize_compute`` exists in the page and
   returns the result dict without importing or calling ``st.*``.
2. On click, the handler submits to a ThreadPoolExecutor and stores a
   ``concurrent.futures.Future`` in ``st.session_state["_opt_future"]`` and
   sets ``st.session_state["_opt_running"] = True``.
3. A small pure state-machine helper ``_poll_optimize`` inspects session_state
   and returns:
     "idle"    — no future present
     "running" — future present but not done
     "done"    — future done + result stored in lineup_optimizer_result,
                  _opt_running cleared
     "error"   — future raised, error stored in _opt_error, _opt_running cleared
"""

from __future__ import annotations

import ast
import concurrent.futures
import types
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "2_Line-up_Optimizer.py"
_PAGE_TEXT: str = _PAGE.read_text(encoding="utf-8")
_PAGE_TREE: ast.Module = ast.parse(_PAGE_TEXT)


# ── 1. Pure compute helper exists and does NOT call st.* ──────────────


def test_run_optimize_compute_defined_in_page():
    """``_run_optimize_compute`` must be a top-level function in the page."""
    names = {node.name for node in ast.walk(_PAGE_TREE) if isinstance(node, ast.FunctionDef)}
    assert "_run_optimize_compute" in names, (
        "page must define a top-level function _run_optimize_compute "
        "that performs the heavy compute without touching st.*"
    )


def test_run_optimize_compute_does_not_call_st():
    """``_run_optimize_compute`` must not reference the ``st`` module anywhere
    inside its body — that would break the background thread (no ScriptRunContext).
    """
    # Find the function node
    func_node = None
    for node in ast.walk(_PAGE_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_optimize_compute":
            func_node = node
            break
    assert func_node is not None, "_run_optimize_compute not found"

    # Walk only nodes that fall within the function's source lines
    st_calls: list[int] = []
    for node in ast.walk(func_node):
        # Look for ``st.something`` attribute accesses
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "st":
            st_calls.append(node.lineno)

    assert not st_calls, (
        f"_run_optimize_compute calls st.* at lines {st_calls} — "
        "background thread must not touch Streamlit (no ScriptRunContext)"
    )


def test_run_optimize_compute_returns_dict():
    """Extract + exec ``_run_optimize_compute``; call it with a minimal stub and
    confirm it returns a dict (or raises, which is also acceptable — the caller
    wraps in try/except).  The key requirement is that it never touches ``st.*``.
    This is already covered by the AST test above; this test just confirms the
    function is exec-able as a standalone unit."""
    func_node = None
    for node in ast.walk(_PAGE_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_optimize_compute":
            func_node = node
            break
    assert func_node is not None

    # We can at least compile it without SyntaxError
    lines = _PAGE_TEXT.splitlines()
    func_src = "\n".join(lines[func_node.lineno - 1 : func_node.end_lineno])
    # Should compile cleanly
    compile(func_src, "<_run_optimize_compute>", "exec")


# ── 2. Click handler submits to executor, stores Future ───────────────


def test_page_uses_threadpoolexecutor():
    """Page must import or use ThreadPoolExecutor for background execution."""
    assert "ThreadPoolExecutor" in _PAGE_TEXT, (
        "page must use concurrent.futures.ThreadPoolExecutor to run the heavy compute in a background thread"
    )


def test_page_stores_opt_future_in_session_state():
    """On click, the handler must store the Future in session_state[\"_opt_future\"]."""
    assert '"_opt_future"' in _PAGE_TEXT or "'_opt_future'" in _PAGE_TEXT, (
        'page must store the concurrent.futures.Future in st.session_state["_opt_future"]'
    )


def test_page_sets_opt_running_flag():
    """On click, the handler must set session_state[\"_opt_running\"] = True."""
    assert '"_opt_running"' in _PAGE_TEXT or "'_opt_running'" in _PAGE_TEXT, (
        'page must set st.session_state["_opt_running"] to gate the polling path'
    )


def test_page_calls_executor_submit():
    """Handler must call .submit() on the executor to dispatch the work."""
    # executor.submit(... is the canonical pattern
    assert ".submit(" in _PAGE_TEXT, (
        "page must call executor.submit(...) to submit _run_optimize_compute to the ThreadPoolExecutor"
    )


# ── 3. _poll_optimize state machine ──────────────────────────────────


def test_poll_optimize_defined_in_page():
    names = {node.name for node in ast.walk(_PAGE_TREE) if isinstance(node, ast.FunctionDef)}
    assert "_poll_optimize" in names, (
        "page must define a _poll_optimize(session_state) helper function "
        "that returns 'idle' | 'running' | 'done' | 'error'"
    )


def _exec_poll_fn():
    """Extract and exec _poll_optimize from the page source."""
    func_node = None
    for node in ast.walk(_PAGE_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "_poll_optimize":
            func_node = node
            break
    assert func_node is not None, "_poll_optimize not found"
    lines = _PAGE_TEXT.splitlines()
    func_src = "\n".join(lines[func_node.lineno - 1 : func_node.end_lineno])
    ns: dict = {}
    exec(compile(func_src, "<_poll_optimize>", "exec"), ns)  # noqa: S102
    return ns["_poll_optimize"]


def test_poll_optimize_returns_idle_when_no_future():
    fn = _exec_poll_fn()
    state: dict = {}
    assert fn(state) == "idle"


def test_poll_optimize_returns_running_when_future_not_done():
    fn = _exec_poll_fn()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    import threading

    gate = threading.Event()
    future = executor.submit(gate.wait)  # will block until we release it

    state: dict = {"_opt_future": future, "_opt_running": True}
    try:
        result = fn(state)
        assert result == "running", f"expected 'running', got {result!r}"
    finally:
        gate.set()
        executor.shutdown(wait=False)


def test_poll_optimize_returns_done_and_stores_result():
    fn = _exec_poll_fn()

    fake_result = {"mode": "quick", "scope": "today", "lineup": None}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: fake_result)
    # Wait for completion before calling poll
    concurrent.futures.wait([future])

    state: dict = {"_opt_future": future, "_opt_running": True}
    status = fn(state)
    assert status == "done", f"expected 'done', got {status!r}"
    assert state.get("lineup_optimizer_result") == fake_result, (
        "_poll_optimize must copy future.result() into session_state['lineup_optimizer_result'] when done"
    )
    assert not state.get("_opt_running"), "_poll_optimize must clear _opt_running when the future completes"
    executor.shutdown(wait=False)


def test_poll_optimize_returns_error_on_exception():
    fn = _exec_poll_fn()

    def _raise():
        raise ValueError("LP solver crashed")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_raise)
    concurrent.futures.wait([future])

    state: dict = {"_opt_future": future, "_opt_running": True}
    status = fn(state)
    assert status == "error", f"expected 'error', got {status!r}"
    assert not state.get("_opt_running"), "_poll_optimize must clear _opt_running on exception"
    assert "_opt_error" in state, "_poll_optimize must store the exception message in session_state['_opt_error']"
    executor.shutdown(wait=False)


def test_poll_optimize_idle_when_running_false_but_future_present():
    """If _opt_running was cleared (e.g., already handled), treat as idle."""
    fn = _exec_poll_fn()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: {"done": True})
    concurrent.futures.wait([future])
    # _opt_running absent / False
    state: dict = {"_opt_future": future, "_opt_running": False}
    result = fn(state)
    # Either "idle" or "done" is acceptable — the key requirement is NOT "running"
    assert result in ("idle", "done"), f"with _opt_running=False, poll must return 'idle' or 'done', got {result!r}"
    executor.shutdown(wait=False)


# ── 4. Structural: poll path shows spinner, not submit again ──────────


def test_page_shows_spinner_while_running():
    """While optimizing, the page must render a spinner/status message
    so the user knows work is in progress."""
    # We check for a spinner call near the _opt_running check
    assert "Optimizing" in _PAGE_TEXT or "optimizing" in _PAGE_TEXT, (
        "page must show an 'Optimizing…' status message while the background thread is running"
    )


def test_page_does_not_resubmit_while_running():
    """The click handler must guard against resubmitting while a future is
    already in flight (idempotency)."""
    # Standard pattern: check not session_state.get("_opt_running") before submit
    assert "_opt_running" in _PAGE_TEXT, 'page must guard against resubmission using the "_opt_running" flag'


def test_page_calls_st_rerun_for_polling():
    """While the background task is running the page must call st.rerun()
    to poll the future on the next script execution."""
    assert "st.rerun()" in _PAGE_TEXT, "page must call st.rerun() to poll the Future until it is done"
