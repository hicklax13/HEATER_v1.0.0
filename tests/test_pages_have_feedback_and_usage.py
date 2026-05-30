"""Structural invariant: every interactive page logs usage + offers feedback.

Mirrors test_pages_have_auth_guard.py. Streamlit runs each pages/*.py top to
bottom on every interaction, so the usage/feedback wiring is per-page, not
global. log_page_view() must sit after the auth gate; the feedback widget is
rendered on the page (appended at EOF).

Underscore-prefixed admin pages (e.g. _admin_console.py) are exempt — they have
their own surfaces.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and not p.name.startswith("_")
)


def test_found_the_pages():
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_usage_and_feedback(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.usage import log_page_view" in src, f"{page.name} must import log_page_view"
    assert "from src.feedback import render_feedback_widget" in src, f"{page.name} must import render_feedback_widget"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_log_page_view_after_require_auth(page):
    src = page.read_text(encoding="utf-8")
    i_auth = src.index("require_auth()")
    i_log = src.index("log_page_view(")
    assert i_log > i_auth, f"{page.name}: log_page_view() must follow require_auth()"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_feedback_widget_called(page):
    src = page.read_text(encoding="utf-8")
    assert "render_feedback_widget(" in src, f"{page.name} must call render_feedback_widget()"
