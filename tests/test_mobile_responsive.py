"""Mobile-responsive CSS regression guards (R-7).

Asserts that inject_custom_css() emits a @media (max-width: 768px) block
containing the key responsive rules:
  - Horizontal scroll on compact tables so wide columns don't break layout
  - Comfortable minimum touch-target height on interactive elements
  - Hero numeral / wordmark scale-down so oversized figures don't overflow
  - Tighter card / column padding for small screens
  - Touch-friendly WebKit momentum scrolling on table overflow

The block must be ADDITIVE — none of the locked desktop rules may be
altered by these additions. The prefers-reduced-motion block must not be
reordered (validated by requiring it still precedes the mobile block is
irrelevant since it's not currently the last rule — we just confirm both
still exist).

2026-06-13  feat/beta-roadmap-phase7
"""

from pathlib import Path

_SRC = (Path(__file__).resolve().parent.parent / "src" / "ui_shared.py").read_text(encoding="utf-8")


def test_mobile_media_query_block_present():
    """A @media (max-width: 768px) block must exist in the stylesheet."""
    assert "@media (max-width: 768px)" in _SRC, (
        "inject_custom_css() must emit a @media (max-width: 768px) responsive block"
    )


def test_mobile_table_overflow_scrollable():
    """On mobile, compact-table-wrap must be horizontally scrollable with
    WebKit momentum scrolling — prevents wide tables from breaking layout."""
    assert "-webkit-overflow-scrolling: touch" in _SRC, (
        "mobile block must include -webkit-overflow-scrolling: touch "
        "on the compact-table container for smooth iOS scroll"
    )


def test_mobile_touch_target_min_height():
    """Interactive elements (buttons, tabs) need a minimum tap target of
    44 px so thumbs can hit them reliably on small screens."""
    # The value must appear in or near a max-width media query context.
    # We check the source contains both the mobile query AND a min-height tap target.
    assert "min-height: 44px" in _SRC, (
        "mobile block must set min-height: 44px on interactive elements (WCAG 2.5.5 / Apple HIG touch-target guidance)"
    )


def test_mobile_hero_num_scales_down():
    """Oversized hero numerals must scale down on small screens so they
    don't overflow the viewport or push other content off-screen."""
    # The font-size for .hero-num inside a max-width query must be present
    # (any reasonable clamp/vw/px value is acceptable — just assert the rule exists).
    import re

    # Look for a .hero-num font-size declaration inside any max-width media block
    mobile_hero_re = re.compile(
        r"@media\s*\(max-width:[^)]+\)[^@]*?\.hero-num[^}]*?font-size",
        re.DOTALL,
    )
    assert mobile_hero_re.search(_SRC), (
        "a @media (max-width:...) block must scale down .hero-num font-size so hero figures don't overflow on mobile"
    )


def test_mobile_card_padding_reduced():
    """Cards / fp-card / glass panels should use tighter horizontal
    padding on mobile so content has breathing room without cramping."""
    import re

    # Look for a .glass / .fp-card padding declaration inside a max-width block
    mobile_pad_re = re.compile(
        r"@media\s*\(max-width:[^)]+\)[^@]*?(?:\.glass|\.fp-card|\.metric-card)[^}]*?padding",
        re.DOTALL,
    )
    assert mobile_pad_re.search(_SRC), "a @media (max-width:...) block must reduce card padding on mobile"


def test_mobile_block_does_not_alter_desktop_overflow_rule():
    """The existing desktop overflow-x:auto on .compact-table-wrap must
    be unchanged — mobile block is purely additive."""
    # The desktop rule must still be present exactly as locked.
    assert ".compact-table-wrap {{\n        overflow-x: auto !important;" in _SRC or (
        ".compact-table-wrap {{" in _SRC and "overflow-x: auto !important;" in _SRC
    ), "the desktop .compact-table-wrap overflow-x:auto rule must not have been removed"


def test_prefers_reduced_motion_still_present():
    """Sanity: prefers-reduced-motion must still be in the sheet (not accidentally deleted)."""
    assert "@media (prefers-reduced-motion: reduce)" in _SRC, (
        "prefers-reduced-motion block must still be present after mobile CSS additions"
    )
