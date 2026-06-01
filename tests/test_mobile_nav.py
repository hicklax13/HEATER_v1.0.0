"""The Streamlit header bar must be hidden only on desktop widths.

On phones the sidebar auto-collapses and the only control to re-open it lives in
that header bar. Hiding the header unconditionally (the original behavior)
stranded mobile users on the landing page with no way to navigate. The hide must
be wrapped in a desktop-only media query so phones keep the menu toggle.
2026-06-01 pre-launch audit (mobile).
"""

from pathlib import Path


def test_header_hide_is_desktop_only():
    src = (Path(__file__).resolve().parent.parent / "src" / "ui_shared.py").read_text(encoding="utf-8")
    assert "@media (min-width: 768px)" in src, (
        "header[data-testid='stHeader'] + stToolbar must be hidden only inside a "
        "@media (min-width: 768px) query so phone users keep the sidebar menu toggle"
    )


def test_mobile_keeps_sidebar_expand_button():
    """The collapsed-sidebar open toggle (stExpandSidebarButton) lives INSIDE
    stToolbar. Hiding stToolbar unconditionally hid the toggle on mobile and
    stranded phone users with no way to open the nav. ui_shared.py must keep the
    toolbar + expand button reachable on phone widths. 2026-06-01 mobile fix."""
    src = (Path(__file__).resolve().parent.parent / "src" / "ui_shared.py").read_text(encoding="utf-8")
    assert "stExpandSidebarButton" in src, (
        "ui_shared.py must surface the collapsed-sidebar open toggle ([data-testid='stExpandSidebarButton']) on mobile"
    )
    assert "@media (max-width: 767px)" in src, "must have a phone-width media query that keeps the toolbar/toggle"
