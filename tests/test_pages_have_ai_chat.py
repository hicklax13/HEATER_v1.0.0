"""Structural invariant: every interactive page mounts the AI chat widget.

Mirrors test_pages_have_feedback_and_usage.py. Streamlit runs each page top to
bottom on every interaction, so the chat mount is per-page, not global. The call
must sit after the auth gate. Underscore-prefixed admin pages are exempt.
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
def test_page_imports_chat(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.ai.chat import render_chat_widget" in src, f"{page.name} must import render_chat_widget"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_chat_widget_called_after_auth(page):
    src = page.read_text(encoding="utf-8")
    assert "render_chat_widget(" in src, f"{page.name} must call render_chat_widget()"
    i_auth = src.index("require_auth()")
    i_chat = src.index("render_chat_widget(")
    assert i_chat > i_auth, f"{page.name}: render_chat_widget() must follow require_auth()"
