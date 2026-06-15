"""Structural + runtime guards for the R-8 first-run onboarding guide.

TDD: these tests are written BEFORE the implementation and must fail red
until app.py is updated.

Guards:
  1. app.py imports save_view / load_view from src.user_data
  2. A pure helper _onboarding_dismissed() reads load_view("ui","onboarding")
  3. The dismiss action calls save_view("ui","onboarding",{"dismissed":True})
  4. The welcome copy mentions the four key pages
  5. A "re-open" affordance exists (small button / expander) after dismissal
  6. Defensiveness: exceptions from load_view / save_view are swallowed
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"
_SRC = _APP.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


# ── helpers ───────────────────────────────────────────────────────────


def _app_imports_names() -> list[str]:
    """All names imported in app.py at top-level."""
    names: list[str] = []
    for node in _TREE.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.append(alias.asname or alias.name)
    return names


def _all_call_names(tree: ast.AST) -> list[str]:
    """All function-call names (both Name and Attribute) in the whole file."""
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls


def _fn_src(name: str) -> str:
    """Source text of a named function defined at module level."""
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SRC, node) or ""
    return ""


# ── 1. imports ────────────────────────────────────────────────────────


def test_app_imports_save_view():
    """app.py must import save_view from src.user_data."""
    assert "save_view" in _app_imports_names(), "app.py must import save_view from src.user_data"


def test_app_imports_load_view():
    """app.py must import load_view from src.user_data."""
    assert "load_view" in _app_imports_names(), "app.py must import load_view from src.user_data"


def test_app_imports_user_data_module():
    """The import must come from src.user_data (not some other module)."""
    for node in _TREE.body:
        if isinstance(node, ast.ImportFrom) and node.module == "src.user_data":
            imported = {alias.asname or alias.name for alias in node.names}
            if "save_view" in imported or "load_view" in imported:
                return
    pytest.fail("No 'from src.user_data import ... save_view/load_view ...' found in app.py")


# ── 2. _onboarding_dismissed helper ──────────────────────────────────


def test_onboarding_dismissed_helper_exists():
    """A function _onboarding_dismissed() must be defined in app.py."""
    fn_names = {n.name for n in _TREE.body if isinstance(n, ast.FunctionDef)}
    assert "_onboarding_dismissed" in fn_names, "_onboarding_dismissed() not found in app.py"


def test_onboarding_dismissed_calls_load_view():
    """_onboarding_dismissed() must call load_view."""
    src = _fn_src("_onboarding_dismissed")
    assert src, "_onboarding_dismissed() not found"
    tree = ast.parse(src)
    calls = _all_call_names(tree)
    assert "load_view" in calls, "_onboarding_dismissed() must call load_view()"


def test_onboarding_dismissed_uses_correct_keys():
    """_onboarding_dismissed() must pass kind='ui' and name='onboarding'."""
    src = _fn_src("_onboarding_dismissed")
    assert '"ui"' in src or "'ui'" in src, "_onboarding_dismissed() must use kind='ui'"
    assert '"onboarding"' in src or "'onboarding'" in src, "_onboarding_dismissed() must use name='onboarding'"


def test_onboarding_dismissed_returns_bool():
    """_onboarding_dismissed() must return a boolean (bool() call or True/False literal)."""
    src = _fn_src("_onboarding_dismissed")
    assert src, "_onboarding_dismissed() not found"
    # accept: explicit True/False, bool(), or a .get() comparison
    has_bool = "True" in src or "False" in src or "bool(" in src or ".get(" in src or "== True" in src
    assert has_bool, "_onboarding_dismissed() must return a bool-ish value"


# ── 3. dismiss action calls save_view ─────────────────────────────────


def test_dismiss_action_calls_save_view():
    """app.py must call save_view('ui','onboarding',{'dismissed':True}) somewhere."""
    # Check raw source for the call — the exact quoting may vary.
    assert "save_view" in _SRC, "save_view never called in app.py"
    # Must pass the onboarding key
    assert '"onboarding"' in _SRC or "'onboarding'" in _SRC, "save_view call must reference 'onboarding' as the name"
    # Must pass dismissed=True
    assert '"dismissed"' in _SRC or "'dismissed'" in _SRC, "save_view call must include 'dismissed' key"


# ── 4. welcome copy mentions key pages ────────────────────────────────


@pytest.mark.parametrize(
    "page_keyword",
    ["My Team", "Lineup", "Free Agent", "Trade"],
)
def test_onboarding_copy_mentions_page(page_keyword):
    """The source must contain copy that mentions each in-season page."""
    assert page_keyword in _SRC, f"onboarding copy must mention '{page_keyword}'"


# ── 5. re-open affordance exists ──────────────────────────────────────


def test_reopen_affordance_exists():
    """There must be a 'show' or 'getting-started' re-open affordance in app.py."""
    # Accept: a button/expander labelled with 'getting' or 'guide' or 'started'
    lower = _SRC.lower()
    has_reopen = (
        "getting-started" in lower
        or "getting started" in lower
        or "show guide" in lower
        or "show getting" in lower
        or "reopen" in lower
        or "re-open" in lower
    )
    assert has_reopen, "No re-open affordance ('getting started' / 'show guide') found in app.py"


# ── 6. defensive error-handling around user_data calls ───────────────


def test_onboarding_dismissed_wraps_load_view_defensively():
    """_onboarding_dismissed() must swallow exceptions from load_view (try/except)."""
    src = _fn_src("_onboarding_dismissed")
    assert src, "_onboarding_dismissed() not found"
    # Accept a try/except block OR a default-return on None
    has_guard = (
        "try:" in src or "except" in src or "if " in src  # None-check pattern: if data is None: return False
    )
    assert has_guard, "_onboarding_dismissed() must handle exceptions / None from load_view"


# ── 7. unit-testable pure-helper behaviour ────────────────────────────


def test_onboarding_dismissed_returns_false_when_no_view(tmp_path, monkeypatch):
    """_onboarding_dismissed() returns False when load_view returns None."""
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "od_test.db")
    # Patch load_view to return None (nothing saved yet)
    with patch("app.load_view", return_value=None):
        import importlib
        import sys

        # Re-import app.py safely: only need the helper function, not the full
        # Streamlit render tree.  We pull it directly via importlib so we don't
        # trigger st.set_page_config twice.
        if "app" in sys.modules:
            app_mod = sys.modules["app"]
        else:
            app_mod = importlib.import_module("app")

        result = app_mod._onboarding_dismissed()
    assert result is False, "_onboarding_dismissed() must return False when no view is saved"


def test_onboarding_dismissed_returns_true_when_dismissed(tmp_path, monkeypatch):
    """_onboarding_dismissed() returns True when load_view returns {'dismissed': True}."""
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "od_test2.db")
    with patch("app.load_view", return_value={"dismissed": True}):
        import sys

        app_mod = sys.modules.get("app")
        if app_mod is None:
            import importlib

            app_mod = importlib.import_module("app")

        result = app_mod._onboarding_dismissed()
    assert result is True, "_onboarding_dismissed() must return True when {'dismissed': True} saved"


def test_onboarding_dismissed_returns_false_when_load_view_raises(tmp_path, monkeypatch):
    """_onboarding_dismissed() must return False (not crash) when load_view raises."""
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "od_test3.db")

    def _boom(*_a, **_kw):
        raise RuntimeError("DB locked")

    with patch("app.load_view", side_effect=_boom):
        import sys

        app_mod = sys.modules.get("app")
        if app_mod is None:
            import importlib

            app_mod = importlib.import_module("app")

        result = app_mod._onboarding_dismissed()
    assert result is False, "_onboarding_dismissed() must return False when load_view raises"
