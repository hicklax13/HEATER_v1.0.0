"""Lock the Combustion Index UI redesign (2026-06-08).

Regression guard over the design-system changes in ``src/ui_shared.py`` +
``.streamlit/config.toml`` that replaced the earlier "FantasyPros revamp" look
(red ``#e63946``, Figtree/Bebas) with the Combustion Index look:

- ``THEME["primary"]`` is hot orange ``#ff6d00`` (the headline brand change).
- The ``@import`` loads Archivo + Inter + IBM Plex Mono — Figtree/Bebas retired.
- The display/body/mono font tokens + the ``--fp-*`` design tokens are emitted.
- ``.sec-head`` uses the Archivo display font and is NOT all-caps.
- The sidebar is a deep-navy chrome rail (navy token + a thin desktop-only width
  rule scoped inside ``@media (min-width: 768px)``).
- The main block-container is full-width (``max-width: none``, no 1180px clamp).
- No emoji anywhere in ``inject_custom_css()`` output or ``PAGE_ICONS``. The
  functional negative icons (``x_mark``/``alert``/``reject``) may still carry the
  ember-red ``#e63946`` — that is an allowed functional color, not the brand.

These lock the "feel" of the redesign so a future edit can't silently bring back
the red brand, the shouty Bebas all-caps headers, or a narrow clamped layout.
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


# ── Palette ───────────────────────────────────────────────────────────


def test_primary_is_combustion_orange():
    """The redesign makes hot orange the primary/brand color (was red)."""
    from src.ui_shared import THEME

    assert THEME["primary"] == "#ff6d00", f"primary must be Combustion orange #ff6d00, got {THEME['primary']!r}"


# ── Fonts ─────────────────────────────────────────────────────────────


def test_import_loads_combustion_fonts_not_figtree_bebas():
    """The Google-Fonts @import loads Archivo + Inter + IBM Plex Mono only."""
    m = re.search(r"@import url\('https://fonts\.googleapis\.com[^']*'\)", _SRC)
    assert m, "the Google-Fonts @import was not found in ui_shared.py"
    import_line = m.group()
    assert "Archivo" in import_line, "@import must load Archivo (display font)"
    assert "Inter" in import_line, "@import must load Inter (body font)"
    assert "IBM+Plex+Mono" in import_line, "@import must load IBM Plex Mono"
    # Retired fonts must be gone from the whole module (not just the @import).
    assert "Figtree" not in _SRC, "Figtree was retired in the Combustion redesign"
    assert "Bebas" not in _SRC, "Bebas Neue was retired in the Combustion redesign"


def test_font_tokens_emitted():
    """The display/body/mono font tokens are emitted as CSS custom properties."""
    for token in ("--font-display", "--font-body", "--font-mono"):
        assert token in _SRC, f"{token} custom property must be emitted in ui_shared.py"
    # The tokens map to the Combustion families.
    assert "--font-display: 'Archivo'" in _SRC, "--font-display must be Archivo"
    assert "--font-body: 'Inter'" in _SRC, "--font-body must be Inter"
    assert "--font-mono: 'IBM Plex Mono'" in _SRC, "--font-mono must be IBM Plex Mono"


def test_sec_head_uses_display_font_and_is_not_all_caps():
    """`.sec-head` must be the Archivo display font, title-case (not all-caps)."""
    block = _css_block(".sec-head")
    assert "var(--font-display)" in block, ".sec-head must use the Archivo --font-display token"
    assert "text-transform: none" in block, ".sec-head must set text-transform:none (no all-caps)"
    assert "Bebas" not in block, ".sec-head must not use Bebas Neue"


# ── Design tokens ─────────────────────────────────────────────────────


def test_fp_tokens_emitted_as_custom_properties():
    """The core --fp-* design tokens (incl. the navy chrome token) are emitted."""
    for var in (
        "--fp-app-bg",
        "--fp-surface",
        "--fp-primary",
        "--fp-navy",
        "--fp-border",
        "--fp-divider",
        "--fp-sidebar-bg",
    ):
        assert var in _SRC, f"{var} custom property must be emitted in ui_shared.py"


# ── Sidebar (navy chrome rail) ─────────────────────────────────────────


def test_sidebar_is_navy_rail_with_desktop_scoped_width():
    """The sidebar is a deep-navy rail; its thin width is desktop-scoped."""
    # Navy token present + bar painted with the navy gradient.
    assert "--fp-navy" in _SRC, "the navy chrome token must be emitted"
    sidebar_block = _css_block(".stSidebar")
    assert "var(--fp-navy)" in sidebar_block, ".stSidebar must paint with the navy chrome token"

    # The thin-rail width rule lives inside an @media (min-width: 768px) block so
    # phones keep the full-width slide-over drawer (mobile-nav fix invariant).
    media_idx = _SRC.find("@media (min-width: 768px)")
    width_re = re.compile(r"\.stSidebar \{\{\s*width:\s*(\d+)px", re.DOTALL)
    m = None
    search_from = 0
    while True:
        media_idx = _SRC.find("@media (min-width: 768px)", search_from)
        if media_idx == -1:
            break
        # Look for a width-bearing .stSidebar rule shortly after this media query.
        window = _SRC[media_idx : media_idx + 400]
        wm = width_re.search(window)
        if wm:
            m = wm
            break
        search_from = media_idx + 1
    assert m, "a desktop-scoped (@media min-width:768px) .stSidebar width rule was not found"
    width = int(m.group(1))
    assert width <= 140, f"the sidebar should be a thin rail (<=140px) on desktop; found width:{width}px"


# ── Full-width layout ─────────────────────────────────────────────────


def test_block_container_is_full_width():
    """The main block container is full-width (max-width:none, no 1180px clamp)."""
    assert "max-width: 1180px" not in _SRC, "the old 1180px content clamp must be gone (full-width redesign)"
    # The combined block-container rule sets max-width: none.
    m = re.search(r"\.block-container \{\{[^}]*?max-width:\s*none", _SRC, re.DOTALL)
    grouped = re.search(
        r"\[data-testid=\"stMainBlockContainer\"\][^{]*\{\{\s*max-width:\s*none",
        _SRC,
        re.DOTALL,
    )
    assert m or grouped or "max-width: none" in _SRC, "the block container must set max-width:none (full-width)"


# ── No-emoji sweep ─────────────────────────────────────────────────────

# Emoji ranges: misc symbols & pictographs, emoticons, transport/map, supplemental
# symbols, dingbats, regional indicators, variation selectors, ZWJ.
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\U0000fe00-\U0000fe0f\U0000200d\U00002b00-\U00002bff\U00002190-\U000021ff\U00002300-\U000023ff]"
)


def test_no_emoji_in_css_output():
    """inject_custom_css() output must contain no emoji characters."""
    from src.ui_shared import inject_custom_css

    captured: list[str] = []

    class _StStub:
        def markdown(self, body, *a, **k):
            captured.append(body)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    import src.ui_shared as ui_shared

    real_st = ui_shared.st
    ui_shared.st = _StStub()
    try:
        inject_custom_css()
    finally:
        ui_shared.st = real_st

    blob = "".join(captured)
    assert blob, "inject_custom_css produced no markdown output"
    found = _EMOJI_RE.findall(blob)
    assert not found, f"inject_custom_css output must not contain emoji; found {found!r}"


def test_no_emoji_in_page_icons():
    """PAGE_ICONS are inline SVGs — no emoji characters allowed."""
    from src.ui_shared import PAGE_ICONS

    blob = "".join(str(v) for v in PAGE_ICONS.values())
    found = _EMOJI_RE.findall(blob)
    assert not found, f"PAGE_ICONS must not contain emoji; found {found!r}"
