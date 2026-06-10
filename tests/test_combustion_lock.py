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


# ── Font-lock ("these fonts only") ─────────────────────────────────────


def test_body_font_locked_on_document_root():
    """`html, body` is forced to Inter (--font-body) so the bare <body> element
    no longer computes Streamlit's default Source Sans (Combustion detail pass).

    Headings (Archivo) + figures (mono) keep their own families via later rules,
    so this is a root body-font lock only, not a blanket override.
    """
    m = re.search(
        r"html,\s*body\s*\{\{\s*font-family:\s*var\(--font-body\)\s*!important;",
        _SRC,
    )
    assert m, "ui_shared must emit `html, body {{ font-family: var(--font-body) !important; }}` (root body-font lock)"


def test_widget_containers_font_locked_to_body():
    """Streamlit's built-in widgets are font-locked to Inter (--font-body).

    BaseWeb selects/inputs/tabs, widget labels, metrics, etc. ship their own
    font stacks that escape the base `.stApp` family. The font-lock block forces
    `font-family: var(--font-body)` onto a comma-joined selector list that
    includes representative widget containers. Assert that a representative
    selector (`[data-baseweb="select"]` AND `[data-testid="stWidgetLabel"]`)
    lives in the SAME rule that declares the body font.
    """
    decl = "font-family: var(--font-body) !important;"
    # Walk every occurrence of the body-font declaration and inspect the
    # selector list immediately preceding it (from the prior rule's closing
    # "}}" up to the "{{" that opens this rule).
    found_select = False
    found_label = False
    search_from = 0
    while True:
        decl_idx = _SRC.find(decl, search_from)
        if decl_idx == -1:
            break
        open_idx = _SRC.rfind("{{", 0, decl_idx)
        prev_close = _SRC.rfind("}}", 0, open_idx)
        selector_blob = _SRC[prev_close + 2 : open_idx] if prev_close != -1 else _SRC[:open_idx]
        if '[data-baseweb="select"]' in selector_blob:
            found_select = True
        if '[data-testid="stWidgetLabel"]' in selector_blob:
            found_label = True
        search_from = decl_idx + len(decl)
    assert found_select, (
        '[data-baseweb="select"] must appear in a `font-family: var(--font-body)` '
        "rule (BaseWeb selects must be font-locked to Inter)"
    )
    assert found_label, (
        '[data-testid="stWidgetLabel"] must appear in a `font-family: var(--font-body)` '
        "rule (widget labels must be font-locked to Inter)"
    )


# ── Clickable-link convention (orange + underline, main content only) ───


def test_clickable_helper_is_orange_and_underlined():
    """The `.heater-link` / `.clickable` helper carries the interactive look:
    orange (--fp-primary) + underline, so it reads as a link/window-opener
    wherever it is dropped (Combustion clickable-link convention).
    """
    needle = ".heater-link, .clickable {{"
    idx = _SRC.find(needle)
    assert idx != -1, "the `.heater-link, .clickable` helper rule must exist in ui_shared.py"
    end = _SRC.find("}}", idx)
    assert end != -1, "the `.heater-link, .clickable` helper rule has no closing brace"
    block = _SRC[idx:end]
    assert "color: var(--fp-primary)" in block, ".heater-link/.clickable must be orange (color: var(--fp-primary))"
    assert "text-decoration: underline" in block, (
        ".heater-link/.clickable must be underlined (text-decoration: underline)"
    )


def test_content_links_scoped_to_main_not_sidebar():
    """The content-hyperlink color rule (orange + underline on bare <a>) is
    SCOPED to main content — prefixed with the main block container / .stMain —
    and is NOT applied globally to the sidebar nav (which stays bone-on-navy).

    The content-link rule is the one that pairs the orange color with an
    underline (buttons/hover rules reuse the orange color but never underline a
    bare link). Find the orange-color declaration that is immediately followed
    by `text-decoration: underline` and inspect that rule's selector list.
    """
    idx = -1
    search_from = 0
    while True:
        cand = _SRC.find("color: var(--fp-primary) !important;", search_from)
        if cand == -1:
            break
        # The content-link rule pairs color + underline within the next few lines.
        if "text-decoration: underline" in _SRC[cand : cand + 200]:
            idx = cand
            break
        search_from = cand + 1
    assert idx != -1, "an orange + underline content-link rule (color: var(--fp-primary) + underline) must exist"

    open_idx = _SRC.rfind("{{", 0, idx)
    prev_close = _SRC.rfind("}}", 0, open_idx)
    selector_blob = _SRC[prev_close + 2 : open_idx] if prev_close != -1 else _SRC[:open_idx]
    # Strip CSS comments so the explanatory prose (which mentions ".stSidebar"
    # to document the exclusion) isn't mistaken for an actual selector.
    selector_blob = re.sub(r"/\*.*?\*/", "", selector_blob, flags=re.DOTALL)
    assert ("stMainBlockContainer" in selector_blob) or ("stMain" in selector_blob), (
        "the content-link color rule must be scoped to main content "
        "(stMainBlockContainer / stMain), not applied globally"
    )
    assert ".stSidebar" not in selector_blob, (
        "the content-link orange rule must NOT target .stSidebar (sidebar nav links stay bone-on-navy)"
    )


# ── Combustion Finale — machined micro-detail layer (2026-06-10) ───────


def test_motion_tokens_emitted():
    """The motion token system (--dur-1/--dur-2/--ease-snap) is emitted."""
    for token in ("--dur-1:", "--dur-2:", "--ease-snap:"):
        assert token in _SRC, f"{token} motion token must be emitted in ui_shared.py"


def test_reduced_motion_block_present():
    """A prefers-reduced-motion block kills animation for sensitive users."""
    assert "@media (prefers-reduced-motion: reduce)" in _SRC, (
        "ui_shared.py must carry a prefers-reduced-motion block (launch-grade a11y)"
    )


def test_brand_selection_present():
    """Text selection is branded (navy in main, orange in sidebar)."""
    assert "::selection" in _SRC, "a ::selection rule must exist"


def test_focus_visible_ring_present():
    """Keyboard focus gets a brand-orange :focus-visible ring."""
    assert ":focus-visible" in _SRC, "a :focus-visible ring rule must exist"


def test_canvas_grain_present():
    """The app canvas carries the layered grain + blueprint-grid texture."""
    assert "fractalNoise" in _SRC, "the feTurbulence grain data-URI must be on the canvas"
    app_block = _css_block(".stApp")
    assert "data:image/svg+xml" in app_block, ".stApp background must layer the grain data-URI"


def test_no_entrance_animation_on_persistent_cards():
    """Streamlit remounts DOM on every rerun, so entrance keyframes on
    persistent cards replay on every widget click. The six-card slideUp
    rule must be gone (slideUp itself may remain for true entrances)."""
    m = re.search(r"\.glass,\s*\.hero,\s*\.alt[^}]*animation:\s*slideUp", _SRC)
    assert m is None, "persistent cards must not carry the slideUp entrance animation"


def test_dialog_chrome_present():
    """The player-dossier dialog gets Combustion chrome (wide + machined)."""
    assert 'div[data-testid="stDialog"]' in _SRC, "stDialog chrome rules must exist"
    assert "min(92vw, 1100px)" in _SRC, "the dialog must widen to min(92vw, 1100px)"


def test_portal_chrome_present():
    """BaseWeb portals (tooltip/toast) mount on <body> and must be branded navy."""
    assert '[data-testid="stToast"]' in _SRC, "toast chrome must exist"
    assert "stTooltipContent" in _SRC, "tooltip chrome must exist"


def test_empty_state_helper_exists_and_is_emoji_free():
    """render_empty_state replaces bare st.info in data-empty contexts."""
    from src.ui_shared import build_empty_state_html

    html_out = build_empty_state_html("No data yet", "Bootstrap is still warming up.")
    assert 'class="empty-state"' in html_out
    assert "No data yet" in html_out
    assert not _EMOJI_RE.findall(html_out), "empty-state must be emoji-free (SVG icons only)"


def test_compact_table_html_cols_render_unescaped():
    """Standings rank badges are pre-built trusted HTML — html_cols columns
    must render raw while every other column stays escaped (2026-06-10 fix:
    the category grid was showing literal '<span ...>' text)."""
    import pandas as pd

    from src.ui_shared import build_compact_table_html

    df = pd.DataFrame(
        {
            "Team": ["A-Team <script>"],
            "R": ['<span class="rb">1</span>'],
        }
    )
    out = build_compact_table_html(df, html_cols={"R"})
    assert '<span class="rb">1</span>' in out, "html_cols cell must render unescaped"
    assert "&lt;script&gt;" in out, "non-html_cols cells must stay escaped"
