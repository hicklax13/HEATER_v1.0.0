"""Structural guard: interactive pages call st.set_page_config only when
MULTI_USER is OFF, so st.navigation() (flag ON) owns page config via app.py.

Under st.navigation() Streamlit expects exactly one set_page_config (app.py's).
Each page therefore guards its own behind `if not multi_user_enabled():`.
Mirrors test_pages_have_auth_guard.py discovery rules.
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
def test_page_imports_multi_user_enabled(page):
    src = page.read_text(encoding="utf-8")
    assert "multi_user_enabled" in src, f"{page.name} must import multi_user_enabled"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_set_page_config_guarded(page):
    lines = page.read_text(encoding="utf-8").splitlines()
    cfg_idx = next((i for i, ln in enumerate(lines) if "st.set_page_config(" in ln), None)
    assert cfg_idx is not None, f"{page.name}: no st.set_page_config( call found"
    # Walk back to the previous non-blank line; it must be the flag guard.
    j = cfg_idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    assert lines[j].strip() == "if not multi_user_enabled():", (
        f"{page.name}: st.set_page_config must be guarded by "
        f"'if not multi_user_enabled():' (found {lines[j].strip()!r})"
    )
