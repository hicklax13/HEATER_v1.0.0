"""Lock the FantasyPros-inspired UI/UX revamp (2026-06-08).

Regression guard over the design-system changes in ``src/ui_shared.py``:
- Section headers are Figtree title-case (Bebas all-caps retired).
- Bebas Neue survives ONLY on the brand marks (splash title + logo wordmarks).
- The sidebar is a thin dark icon rail (narrow width + dark token).
- The FP design tokens are emitted as ``--fp-*`` CSS custom properties.

These lock the "feel" of the revamp so a future edit can't silently bring back
the shouty Bebas all-caps headers, the wide orange sidebar, or dark-navy chrome.
"""

import re
from pathlib import Path

_UI_PATH = Path(__file__).resolve().parent.parent / "src" / "ui_shared.py"
_SRC = _UI_PATH.read_text(encoding="utf-8")


def _css_block(selector: str) -> str:
    """Return the CSS rule body for ``selector`` from ui_shared.

    CSS braces in the source are f-string-escaped as ``{{``/``}}``.
    """
    needle = f"{selector} {{"
    idx = _SRC.find(needle)
    assert idx != -1, f"{selector!r} CSS rule not found in ui_shared.py"
    end = _SRC.find("}}", idx)
    assert end != -1, f"{selector!r} CSS rule has no closing brace"
    return _SRC[idx:end]


def test_sec_head_is_figtree_title_case():
    """`.sec-head` must be Figtree title-case — Bebas all-caps retired."""
    block = _css_block(".sec-head")
    assert "text-transform: none" in block, ".sec-head must set text-transform:none (no all-caps)"
    assert "Bebas" not in block, ".sec-head must not use Bebas Neue"
    assert "var(--font-body)" in block, ".sec-head must use the Figtree --font-body token"


def test_bebas_retired_except_brand_marks():
    """Bebas Neue may only remain on the splash title + logo wordmarks."""
    count = _SRC.count("Bebas Neue")
    assert count <= 5, f"Bebas Neue should only remain on splash/logo brand marks; found {count} usages"
    # The de-Bebased surfaces must stay Figtree.
    for selector in (".sec-head", ".page-title", '.stTabs [data-baseweb="tab"]'):
        block = _css_block(selector)
        assert "Bebas" not in block, f"{selector} must not use Bebas Neue (retired in the revamp)"


def test_sidebar_is_thin_dark_rail():
    """The sidebar is a thin (<=140px) dark-navy rail."""
    assert "--fp-sidebar-bg" in _SRC, "the sidebar_bg token must be emitted as a custom property"
    m = re.search(r"\.stSidebar \{\{[^}]*?width:\s*(\d+)px", _SRC)
    assert m, "the bare .stSidebar width rule was not found"
    width = int(m.group(1))
    assert width <= 140, f"the sidebar should be a thin rail (<=140px); found width:{width}px"


def test_fp_tokens_emitted_as_custom_properties():
    """The core FP design tokens are emitted as --fp-* custom properties on :root."""
    for var in ("--fp-app-bg", "--fp-surface", "--fp-primary", "--fp-border", "--fp-divider", "--font-body"):
        assert var in _SRC, f"{var} custom property must be emitted in ui_shared.py"


def test_primary_stays_heater_red():
    """The revamp keeps HEATER red as the primary/action color (not FP yellow)."""
    from src.ui_shared import THEME

    assert THEME["primary"] == "#e63946", f"primary must stay HEATER red #e63946, got {THEME['primary']!r}"
