"""Structural guard: each interactive season page enforces its feature flag
via require_page_enabled("page:<stem>") AFTER require_auth() and BEFORE
log_page_view(). A disabled page must hard-stop non-admins even on direct nav.

Mirrors test_pages_have_auth_guard.py / test_pages_have_feedback_and_usage.py
discovery rules (calls inject_custom_css(); name does not start with "_").
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
    assert len(_INTERACTIVE_PAGES) == 14, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_require_page_enabled(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.feature_flags import require_page_enabled" in src, f"{page.name} must import require_page_enabled"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_calls_require_page_enabled_with_correct_key(page):
    src = page.read_text(encoding="utf-8")
    stem = page.stem
    assert f'require_page_enabled("page:{stem}")' in src, f'{page.name} must call require_page_enabled("page:{stem}")'


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_require_page_enabled_between_auth_and_log(page):
    src = page.read_text(encoding="utf-8")
    i_auth = src.index("require_auth()")
    i_gate = src.index("require_page_enabled(")
    i_log = src.index("log_page_view(")
    assert i_auth < i_gate < i_log, (
        f"{page.name}: require_page_enabled() must sit after require_auth() and before log_page_view()"
    )
