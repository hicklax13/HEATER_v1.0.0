"""BUG-023 fix: try_reconnect_yahoo lives in src/yahoo_api.py for headless callers."""

import inspect
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_try_reconnect_yahoo_callable_from_yahoo_api():
    """The reconnect helper must be importable from src.yahoo_api so
    non-Streamlit callers (CI, ops scripts) can also reconnect."""
    from src.yahoo_api import try_reconnect_yahoo

    assert callable(try_reconnect_yahoo)
    sig = inspect.signature(try_reconnect_yahoo)
    # No required positional args (reads env + token file)
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    ]
    assert not required, (
        f"BUG-023: try_reconnect_yahoo should not require positional args (it reads env + token). Required: {required}"
    )


def test_app_does_not_duplicate_reconnect_logic():
    """app.py should NOT define a fat _try_reconnect_yahoo with the full
    token-loading logic. Either import from src.yahoo_api, or keep a thin
    wrapper (<=6 non-comment body lines) that delegates."""
    app_text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")

    # Check if app.py imports try_reconnect_yahoo from src.yahoo_api
    if re.search(r"from src\.yahoo_api import [^\n]*\btry_reconnect_yahoo\b", app_text):
        return  # Direct import --- preferred

    # Otherwise verify the local function is a thin wrapper
    m = re.search(
        r"def _try_reconnect_yahoo[^\n]*\n((?:[ \t][^\n]*\n)+)",
        app_text,
    )
    if m:
        body = m.group(1)
        non_trivial_lines = [
            ln
            for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith('"""')
        ]
        assert len(non_trivial_lines) <= 6, (
            f"BUG-023 regression: app.py defines _try_reconnect_yahoo with "
            f"{len(non_trivial_lines)} non-trivial body lines; should be a thin "
            f"wrapper (<=6 lines) that calls src.yahoo_api.try_reconnect_yahoo "
            f"so headless callers can reuse the logic. Body:\n{body[:500]}"
        )


def test_yahoo_api_reconnect_does_not_import_streamlit():
    """The yahoo_api.try_reconnect_yahoo function must not depend on streamlit
    (so headless callers can use it). Check that src/yahoo_api.py either
    doesn't import streamlit at all, OR if it does, the import is guarded
    (e.g., inside a try/except or behind TYPE_CHECKING)."""
    text = (REPO_ROOT / "src" / "yahoo_api.py").read_text(encoding="utf-8")
    # Top-level (column 0) `import streamlit` or `from streamlit` is the bug
    bad = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("import streamlit") or line.startswith("from streamlit"):
            bad.append((lineno, line.strip()))
    assert not bad, (
        f"BUG-023 regression: src/yahoo_api.py has top-level streamlit import; "
        f"makes try_reconnect_yahoo unusable from headless contexts. Offenders: {bad}"
    )
