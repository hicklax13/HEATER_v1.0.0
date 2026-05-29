"""Structural invariant: every interactive page calls require_auth().

Streamlit executes each pages/*.py independently, so the auth gate from app.py
does not protect deep-linked pages. Any page that renders UI (detected via its
inject_custom_css() call) MUST import and call require_auth().

The admin console (00_Admin_Console.py) is exempt: it uses the stricter
require_admin() guard instead.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and p.name != "00_Admin_Console.py"
)


def test_found_the_pages():
    # Sanity: we should be guarding all 13 in-season pages.
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_require_auth(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.auth import" in src and "require_auth" in src, (
        f"{page.name} must import require_auth from src.auth"
    )


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_calls_require_auth_after_css(page):
    src = page.read_text(encoding="utf-8")
    i_css = src.index("inject_custom_css()")
    i_auth = src.index("require_auth()")
    assert i_auth > i_css, f"{page.name}: require_auth() must follow inject_custom_css()"
