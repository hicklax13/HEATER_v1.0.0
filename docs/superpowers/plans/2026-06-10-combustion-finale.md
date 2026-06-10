# Combustion Finale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a machined micro-detail layer + cross-page consistency to the deployed Combustion Index design — launch-grade, zero behavior change.

**Architecture:** Phase A (sequential) lands the foundation in `src/ui_shared.py` + `.streamlit/config.toml` behind new lock tests. Phase B (4 parallel subagents, disjoint files) sweeps every page for token hygiene + pattern unification + uplifts the 4 undercooked pages. Phase C verifies in the browser, runs the full suite, and ships.

**Tech Stack:** Streamlit ≥1.47 native advanced theming, pure CSS (f-string in `inject_custom_css()`, braces escaped `{{ }}`), Archivo variable-font `wdth` axis, pytest structural guards. **No new PyPI packages** (researched 2026-06-10: all candidates are iframe-based/unmaintained and would break the font-lock).

**Spec:** `docs/superpowers/specs/2026-06-10-combustion-finale-design.md`

---

## Shared Conventions (read before any task)

1. **CSS lives in `inject_custom_css()`** in `src/ui_shared.py` — it is an f-string: literal CSS braces MUST be doubled `{{ }}`; `{t["key"]}` interpolates THEME.
2. **Palette law:** orange `#ff6d00` family + navy + neutrals only. Ember `#e0492f` ONLY for functional negatives. NEVER red as brand. No emoji anywhere — icons are inline SVG from `PAGE_ICONS`.
3. **Token replacement table (Phase B):** in HTML/CSS strings use `var(--…)`; in Python (Plotly args etc.) use `T["…"]` (import `T` from `src.ui_shared`):

| Found literal | CSS context | Python context |
|---|---|---|
| `#FF9800`, `#f97316` | `var(--fp-primary)` | `T["primary"]` |
| `#9E9E9E`, `#6b7280`, `#666666`, `#666` | `var(--fp-tx-muted)` | `T["tx_muted"]` |
| `#9ca3af` | `var(--fp-tx-subtle)` | `T["tx_subtle"]` |
| `#4CAF50`, `#22c55e` | `{T["green"]}` interp | `T["green"]` |
| `#84cc16` | `{T["green_l"]}` interp | `T["green_l"]` |
| `#ef4444` | `{T["danger"]}` interp | `T["danger"]` |
| `#2c2f36` | `var(--fp-tx)` | `T["tx"]` |
| `#f5f5f5` | `var(--fp-divider)` | `T["divider"]` |
| `#e8f5e9` | `rgba(31,157,107,.08)` | same string |
| `#457b9d` (badge) | — | `T["sky"]` |
| `#9c27b0` (badge) | — | `T["purple"]` |
| `#999` (badge) | — | `T["tx_subtle"]` |

   **Do NOT change:** news-source brand colors in `pages/1_My_Team.py` (`#c41230` ESPN, `#6001d2` Yahoo, `#1a73e8` RotoWire, `#002d72` MLB), `TEAM_BRAND` values, ember `#e0492f` functional negatives, and anything in `src/cheat_sheet.py` (out of scope).
4. **Panel pattern (uplift work):** prefer the existing helpers — `render_panel(build_panel_html(title, body_html, fig_label="FIG.0N · LABEL", accent="top"))`, `render_styled_table(df)`, `build_heatbar_html(pct)`, `build_stat_readout_html(label, value)`, `sec(title)`, `render_eyebrow(text)`. New classes available after Task 3: `.chip` (+`.hot`, `.dot-live`), `.hero-num`, `.hr-fade`/`.hr-heat`, `.empty-state` via `render_empty_state(...)`, `.t-eyebrow/.t-fig/.t-label/.t-caption`, `.instr-panel.heat`.
5. **Page invariants (every page edit must preserve):** `inject_custom_css()` → `require_auth()` → `log_page_view()` order; `require_page_enabled("page:<stem>")`; `render_feedback_widget()` present; `format_stat()` for rate stats (no inline `f"{x:.3f}"` near ERA/WHIP); `if not multi_user_enabled():` guard around `st.set_page_config`; viewer-team resolver on personalized pages. When in doubt run the page-targeted structural tests listed in each task.
6. **Motion law:** animate only `transform`/`opacity` (+ box-shadow at card scale); no entrance keyframes on persistent content (Streamlit remounts on every rerun); max 1–2 infinite animations visible per page.
7. **Commits:** small, prefixed `style(finale):`, run `python -m ruff format <files>` + `python -m ruff check <files>` before each commit (pre-commit hook enforces).
8. **Python:** use `.venv/Scripts/python.exe` for all commands (Windows).

---

## Phase A — Foundation (sequential)

### Task 1: Branch

**Files:** none (git only)

- [ ] **Step 1:** `git checkout -b ui/combustion-finale-2026-06-10` (from master, which must be at/after `15ad94d`)

### Task 2: Failing lock tests for the foundation layer

**Files:**
- Modify: `tests/test_combustion_lock.py` (append section)

- [ ] **Step 1: Append these tests** to `tests/test_combustion_lock.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_combustion_lock.py -q`
Expected: the 9 new tests FAIL (tokens/blocks/helper missing); the 13 existing PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_combustion_lock.py
git commit -m "test(finale): failing locks for machined micro-detail layer"
```

### Task 3: Foundation CSS layer in `src/ui_shared.py`

**Files:**
- Modify: `src/ui_shared.py` (anchors given per step; file is ~5400 lines — read each anchor region before editing)

- [ ] **Step 1: Motion tokens** — in the `:root {{` block, directly after the `--font-mono: 'IBM Plex Mono', monospace;` line (~line 792), add:

```css
        --dur-1: 120ms;
        --dur-2: 180ms;
        --ease-snap: cubic-bezier(.2, .7, .3, 1);
```

- [ ] **Step 2: Grain + blueprint canvas** — in the `.stApp {{` rule (~line 802), replace `background: var(--fp-app-bg) !important;` with:

```css
        background:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.78' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E") repeat,
            linear-gradient(rgba(17,39,68,.022) 1px, transparent 1px) 0 0 / 100% 28px,
            linear-gradient(90deg, rgba(17,39,68,.016) 1px, transparent 1px) 0 0 / 28px 100%,
            var(--fp-app-bg) !important;
```

- [ ] **Step 3: Machined top edge** — replace ALL occurrences (Edit `replace_all`) of
`box-shadow: 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);`
with
`box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);`
(hits `.instr-panel`, `.glass`, `.hero`). Then in `.cmd-bar` replace `box-shadow: 0 1px 3px rgba(24,26,32,.06);` the same way. Then in `.glass:hover` and `.hero:hover` replace `box-shadow: 0 4px 14px rgba(24,26,32,.10);` with `box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 4px 14px rgba(24,26,32,.10);` and replace `transform: none;` with `transform: translateY(-1px);` in those two hover rules ONLY (leave `.alt:hover` as is).

- [ ] **Step 4: Kill entrance-thrash** — delete this rule (~line 1588, keep the `@keyframes slideUp` above it — `.reco-banner-detail` still uses it):

```css
    .glass, .hero, .alt, .metric-card, .player-card, .cat-card {{
        animation: slideUp 0.4s ease-out both;
    }}
```

- [ ] **Step 5: Delete the old white-toast rule** — find the rule containing `.stToast` (white bg / charcoal text from the early block) and remove `.stToast`-related declarations there (the new navy portal chrome in Step 6 replaces it; keep `stPopover` styling).

- [ ] **Step 6: Append the finale block** — insert immediately BEFORE the `@media print` block (or, if easier to anchor, before the closing `</style>` of the main injected stylesheet), this complete section (note doubled braces; `{t["primary"]}`/`{t["tx_muted"]}` interpolate):

```css
    /* ═══ COMBUSTION FINALE — machined micro-detail layer (2026-06-10) ═══ */

    /* Interaction chrome */
    * {{ scrollbar-width: thin; scrollbar-color: #c9c8c3 transparent; }}
    .stSidebar * {{ scrollbar-color: rgba(238,241,246,.28) transparent; }}
    ::selection {{ background: var(--fp-navy); color: #ffffff; }}
    .stSidebar ::selection {{ background: var(--fp-primary); color: #ffffff; }}
    :focus-visible {{ outline: 2px solid var(--fp-primary); outline-offset: 2px; border-radius: 4px; }}
    :focus:not(:focus-visible) {{ outline: none; }}

    /* Numeric law — stat columns never shimmy */
    .stat, .mono, td.num, .heater-table td, .compact-table td, .draft-board td,
    [data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums slashed-zero;
    }}

    /* Reading polish */
    h1, h2, h3, .sec-head {{ text-wrap: balance; }}
    .stMarkdown p {{ text-wrap: pretty; }}
    h1, h2, h3 {{ scroll-margin-top: 84px; }}

    /* Gradient hairline dividers */
    .hr-fade {{ height: 1px; border: 0; margin: 18px 0;
        background: linear-gradient(90deg, transparent, var(--fp-border) 15%, var(--fp-border) 85%, transparent); }}
    .hr-heat {{ height: 1px; border: 0; margin: 18px 0;
        background: linear-gradient(90deg, var(--fp-primary), rgba(255,109,0,.25) 45%, transparent 80%); }}
    [data-testid="stDivider"] hr {{ height: 1px; border: 0;
        background: linear-gradient(90deg, transparent, var(--fp-border) 15%, var(--fp-border) 85%, transparent); }}

    /* Hero numerals — Archivo width axis + orange gradient clip (orange ONLY, never red) */
    .hero-num {{
        font-family: var(--font-display);
        font-weight: 900;
        font-stretch: 112%;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.015em;
        background: linear-gradient(180deg, #ff8a2a, #ff6d00 55%, #e85f00);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }}

    /* Chip system — embossed token look; .dot-live strictly for live indicators */
    .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        font: 600 11px/1.6 var(--font-body); letter-spacing: .05em; text-transform: uppercase;
        background: var(--fp-surface); border: 1px solid var(--fp-border); color: var(--fp-tx-muted);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.6), 0 1px 2px rgba(24,26,32,.05);
        transition: border-color var(--dur-1) var(--ease-snap), transform var(--dur-1) var(--ease-snap); }}
    .chip:hover {{ border-color: var(--fp-primary); transform: translateY(-1px); }}
    .chip.hot {{ color: var(--fp-primary); border-color: rgba(255,109,0,.45); background: rgba(255,109,0,.06); }}
    .chip.cold {{ color: var(--fp-cold); border-color: rgba(95,125,156,.45); background: rgba(95,125,156,.07); }}
    @keyframes dotPulse {{ 50% {{ box-shadow: 0 0 0 4px rgba(255,109,0,0); }} }}
    .chip .dot-live {{ width: 6px; height: 6px; border-radius: 50%; background: var(--fp-primary);
        box-shadow: 0 0 0 0 rgba(255,109,0,.45); animation: dotPulse 2.2s ease-out infinite; }}

    /* Conic heat wash — opt-in second light layer for hero panels */
    .instr-panel.heat::before {{
        background:
            radial-gradient(120% 80% at 100% 0%, rgba(255,109,0,.05), transparent 42%),
            conic-gradient(from 210deg at 108% -8%, rgba(255,154,60,.07), transparent 28%);
    }}

    /* Empty state — instrument-styled no-data panel */
    .empty-state {{
        position: relative;
        border: 1px dashed var(--fp-border);
        border-radius: 14px;
        padding: 34px 26px;
        margin: 8px 0 14px;
        text-align: center;
        background:
            radial-gradient(rgba(24,26,32,.03) 1px, transparent 1.3px) 0 0 / 20px 20px,
            var(--fp-surface);
    }}
    .empty-state .es-icon svg {{ width: 26px; height: 26px; stroke: var(--fp-primary); margin: 0 auto 8px; display: block; }}
    .empty-state .es-title {{ font-family: var(--font-display); font-weight: 800; font-size: 15px; color: var(--fp-tx); margin-bottom: 4px; }}
    .empty-state .es-body {{ font-family: var(--font-body); font-size: 12.5px; color: var(--fp-tx-muted); max-width: 460px; margin: 0 auto; line-height: 1.55; }}

    /* Type-scale utilities — stop minting inline sizes */
    .t-eyebrow {{ font-family: var(--font-body); font-weight: 700; font-size: 10px; letter-spacing: .28em; text-transform: uppercase; color: var(--fp-tx-muted); }}
    .t-fig {{ font-family: var(--font-mono); font-size: 10px; letter-spacing: .12em; color: var(--fp-tx-muted); }}
    .t-label {{ font-family: var(--font-body); font-weight: 600; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--fp-tx-muted); }}
    .t-caption {{ font-family: var(--font-body); font-size: 12px; color: var(--fp-tx-muted); }}

    /* Table micro-interaction — inset accent bar on row hover (no layout shift) */
    .heater-table tbody tr, .compact-table tbody tr, .draft-board tbody tr {{
        transition: background-color var(--dur-1) var(--ease-snap), box-shadow var(--dur-1) var(--ease-snap);
    }}
    .heater-table tbody tr:hover, .compact-table tbody tr:hover {{
        box-shadow: inset 3px 0 0 var(--fp-primary);
    }}

    /* Sticky header depth cue — hairline + soft scroll shadow */
    .heater-table thead th, .compact-table thead th {{
        box-shadow: 0 1px 0 var(--fp-border), 0 4px 10px rgba(24,26,32,.04);
    }}

    /* Animated tab underline — the highlight glides between tabs */
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--fp-primary); height: 2px; border-radius: 2px;
        transition: all 240ms cubic-bezier(.4, 0, .2, 1); }}
    .stTabs [data-baseweb="tab-border"] {{ background-color: var(--fp-divider); height: 1px; }}

    /* Expander chrome */
    [data-testid="stExpander"] summary {{ transition: background-color var(--dur-1) var(--ease-snap), color var(--dur-1) var(--ease-snap); }}
    [data-testid="stExpander"] summary:hover {{ background: #faf9f6; color: var(--fp-primary); }}
    [data-testid="stExpanderToggleIcon"] {{ transition: transform 180ms var(--ease-snap); }}

    /* Dialog chrome — player dossier */
    div[data-testid="stDialog"] div[role="dialog"] {{
        position: relative;
        width: min(92vw, 1100px);
        border-radius: 16px;
        border: 1px solid var(--fp-border);
        box-shadow: 0 24px 80px rgba(16,33,58,.38), inset 0 1px 0 rgba(255,255,255,.9);
    }}
    div[data-testid="stDialog"] div[role="dialog"]::before {{
        content: ""; position: absolute; top: 0; left: 24px; right: 24px; height: 2px;
        background: linear-gradient(90deg, var(--fp-primary), transparent 70%);
        pointer-events: none;
    }}

    /* Portal chrome — tooltips + toasts mount on <body>, outside .stApp */
    div[data-baseweb="tooltip"], [data-testid="stTooltipContent"] {{
        background: var(--fp-navy) !important; color: #eef1f6 !important;
        border-radius: 8px !important;
        font: 500 12.5px/1.45 var(--font-body) !important;
        box-shadow: 0 8px 24px rgba(16,33,58,.3) !important;
    }}
    [data-testid="stToast"] {{
        background: var(--fp-navy) !important; color: #eef1f6 !important;
        border-radius: 10px !important; border-left: 3px solid var(--fp-primary) !important;
        box-shadow: 0 12px 32px rgba(16,33,58,.35) !important;
    }}
    [data-testid="stToast"] * {{ color: #eef1f6 !important; }}

    /* Metric self-tinting via :has() */
    [data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Up"]) {{
        background: rgba(31,157,107,.05); border-radius: 10px; padding: 6px 10px; }}
    [data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Down"]) {{
        background: rgba(224,73,47,.04); border-radius: 10px; padding: 6px 10px; }}

    /* Glide grid — selection ring + hairline refinement */
    :root {{
        --gdg-accent-color: {t["primary"]} !important;
        --gdg-accent-light: rgba(255,109,0,.10) !important;
        --gdg-horizontal-border-color: rgba(24,26,32,.07) !important;
        --gdg-text-medium: {t["tx_muted"]} !important;
        --gdg-cell-horizontal-padding: 10px !important;
        --gdg-rounding-radius: 10px !important;
        --gdg-bg-search-result: rgba(255,174,66,.22) !important;
    }}

    /* Sidebar machining — inner hairline + grouped-nav section headers */
    .stSidebar {{ box-shadow: inset -1px 0 0 rgba(255,255,255,.06); }}
    [data-testid="stNavSectionHeader"] {{
        font-family: var(--font-body) !important; font-size: 9px !important; font-weight: 700 !important;
        letter-spacing: .22em !important; text-transform: uppercase !important;
        color: rgba(238,241,246,.55) !important; margin-top: 10px !important;
    }}

    /* Shimmer — two-tone on-brand + no-motion fallback lives in the block below */
    .shimmer {{ background: linear-gradient(100deg, #efeeea 40%, #f7f6f3 50%, #efeeea 60%) 0 0 / 200% 100%; }}

    /* Reduced motion — keep as the LAST rules in the sheet */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
            scroll-behavior: auto !important;
        }}
        .shimmer {{ animation: none; background: #efeeea; }}
    }}
```

- [ ] **Step 7: Variable-font axis + preconnect** — replace the `@import` line's Archivo segment `family=Archivo:wght@500;600;700;800;900` with `family=Archivo:wdth,wght@62..125,500..900` (Inter + IBM Plex Mono segments unchanged). Then, directly before the `<style>` opening tag inside the same `st.markdown` f-string, add:

```html
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
```

- [ ] **Step 8: `render_empty_state` helper** — add after the `sec(title)` function (near line 634):

```python
def build_empty_state_html(title: str, body: str = "", icon_key: str = "baseball") -> str:
    """Build the instrument-styled empty-data state panel (Combustion Finale).

    Replaces bare st.info() in data-empty contexts: dot-grid dashed panel,
    centered, orange SVG icon — never emoji.
    """
    icon = PAGE_ICONS.get(icon_key, PAGE_ICONS["baseball"])
    body_html = f'<div class="es-body">{body}</div>' if body else ""
    return (
        '<div class="empty-state">'
        f'<div class="es-icon">{icon}</div>'
        f'<div class="es-title">{title}</div>'
        f"{body_html}"
        "</div>"
    )


def render_empty_state(title: str, body: str = "", icon_key: str = "baseball") -> None:
    """st.markdown wrapper for build_empty_state_html."""
    st.markdown(build_empty_state_html(title, body, icon_key), unsafe_allow_html=True)
```

- [ ] **Step 9: Run the lock suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_combustion_lock.py tests/test_mobile_nav.py -q`
Expected: ALL pass (22 in combustion lock).

- [ ] **Step 10: Sanity-import + ruff + commit**

```bash
.venv/Scripts/python.exe -c "from src.ui_shared import inject_custom_css, render_empty_state; print('ok')"
.venv/Scripts/python.exe -m ruff format src/ui_shared.py tests/test_combustion_lock.py && .venv/Scripts/python.exe -m ruff check src/ui_shared.py tests/test_combustion_lock.py
git add src/ui_shared.py tests/test_combustion_lock.py
git commit -m "style(finale): machined micro-detail foundation layer + empty-state helper"
```

### Task 4: Native advanced theming (`.streamlit/config.toml` + requirements pin)

**Files:**
- Modify: `.streamlit/config.toml`
- Modify: `requirements.txt:1`

- [ ] **Step 1:** Replace the `[theme]` section of `.streamlit/config.toml` with (leave `[server]`/`[browser]`/`[client]` untouched — `enableXsrfProtection` is structurally guarded):

```toml
[theme]
base = "light"
primaryColor = "#ff6d00"
backgroundColor = "#f4f3f1"
secondaryBackgroundColor = "#ffffff"
textColor = "#1b1c20"
linkColor = "#ff6d00"
borderColor = "#e3e2de"
baseRadius = "12px"
buttonRadius = "10px"
dataframeBorderColor = "#e3e2de"
dataframeHeaderBackgroundColor = "#ffffff"
headingFontWeights = [900, 800, 800]
chartCategoricalColors = ["#ff6d00", "#ff9a3c", "#ffae42", "#1f9d6b", "#5f7d9c", "#8aa6c0", "#b0b5be", "#cdd1d8"]
font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400..700&display=swap"
headingFont = "Archivo:https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&display=swap"
codeFont = "'IBM Plex Mono':https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap"

[theme.sidebar]
backgroundColor = "#112744"
textColor = "#eef1f6"
borderColor = "#0c1d33"
primaryColor = "#ff6d00"
```

- [ ] **Step 2:** In `requirements.txt` line 1, change `streamlit>=1.42.0` → `streamlit>=1.47.0` and update the trailing comment to `# 1.47 adds advanced theming (fonts-by-URL, chartCategoricalColors, theme.sidebar); 1.42 st.context.cookies (BR-1)`.

- [ ] **Step 3: Verify config parses + theming keys accepted** (local streamlit is 1.57.0):

Run: `.venv/Scripts/python.exe -c "from streamlit import config as c; c.get_config_options(force_reparse=True); print('theme.linkColor =', c.get_option('theme.linkColor')); print('sidebar bg =', c.get_option('theme.sidebar.backgroundColor'))"`
Expected: prints `#ff6d00` and `#112744`, no "not a valid config option" warnings. If a key errors, REMOVE that one key and note it in the commit message.

- [ ] **Step 4: Run guards + commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_streamlit_security_settings.py tests/test_streamlit_min_version.py -q
git add .streamlit/config.toml requirements.txt
git commit -m "style(finale): adopt Streamlit >=1.47 native advanced theming (fonts-by-URL, sidebar block, chart palette)"
```

### Task 5: Off-palette guard (failing until Phase B completes)

**Files:**
- Create: `tests/test_no_offpalette_hex_in_pages.py`

- [ ] **Step 1: Write the guard:**

```python
"""No off-palette hex literals in pages/, app.py, or ui_analytics_badge (Combustion Finale, 2026-06-10).

The Combustion Index palette lives in src/ui_shared.py THEME + the --fp-* tokens.
Pages must not hardcode the Material/Tailwind-era colors that leaked in before the
finale sweep. News-source brand colors in My Team (ESPN #c41230, Yahoo #6001d2,
RotoWire #1a73e8, MLB #002d72) and TEAM_BRAND team colors are intentional and NOT
in the banned set. src/cheat_sheet.py (print artifact) is out of scope.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_SCANNED = sorted((_ROOT / "pages").glob("*.py")) + [
    _ROOT / "app.py",
    _ROOT / "src" / "ui_analytics_badge.py",
]

_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")

_BANNED = {
    "#ff9800", "#f97316",            # material/tailwind orange -> --fp-primary
    "#9e9e9e", "#6b7280", "#666666", "#666",  # material grays -> --fp-tx-muted
    "#9ca3af",                       # tailwind gray-400 -> --fp-tx-subtle
    "#4caf50", "#22c55e",            # material/tailwind green -> THEME green
    "#84cc16",                       # tailwind lime -> THEME green_l
    "#ef4444",                       # tailwind red -> THEME danger
    "#2c2f36",                       # near-charcoal drift -> --fp-tx
    "#f5f5f5",                       # material gray-100 -> --fp-divider
    "#e8f5e9",                       # material green-50 -> rgba green tint
    "#457b9d", "#9c27b0", "#999",    # legacy badge colors -> THEME sky/purple/tx_subtle
}


def test_no_offpalette_hex():
    hits: list[str] = []
    for path in _SCANNED:
        text = path.read_text(encoding="utf-8")
        for m in _HEX_RE.finditer(text):
            if m.group().lower() in _BANNED:
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(_ROOT)}:{line} {m.group()}")
    assert not hits, "Off-palette hex literals found (use THEME/--fp-* tokens):\n" + "\n".join(hits)
```

- [ ] **Step 2: Run to verify it fails** (it documents the current leakage):

Run: `.venv/Scripts/python.exe -m pytest tests/test_no_offpalette_hex_in_pages.py -q`
Expected: FAIL with a list of hits across ~8 files.

- [ ] **Step 3: Commit**

```bash
git add tests/test_no_offpalette_hex_in_pages.py
git commit -m "test(finale): off-palette hex guard (red until page sweeps land)"
```

---

## Phase B — Page sweeps (4 parallel subagents; disjoint files; all depend on Phase A)

> Every group: apply the Shared Conventions token table to YOUR files only; convert data-empty `st.info`/`st.warning` calls to `render_empty_state(...)` (operational/validation warnings stay native); replace `st.divider()` / `st.markdown("---")` / bare `<hr>` with `st.markdown('<div class="hr-fade"></div>', unsafe_allow_html=True)` (or `.hr-heat` directly under a section header when emphasis is wanted); import any new helpers from `src.ui_shared`. After edits, EVERY group runs (substituting its own files):
>
> ```bash
> .venv/Scripts/python.exe -m ruff format <files> && .venv/Scripts/python.exe -m ruff check <files>
> .venv/Scripts/python.exe -m pytest tests/test_pages_have_auth_guard.py tests/test_pages_format_compliance.py tests/test_pages_have_feedback_and_usage.py tests/test_pages_guard_set_page_config.py tests/test_admin_pages_flag_enforced.py tests/test_combustion_lock.py -q
> .venv/Scripts/python.exe -m pytest tests/test_no_offpalette_hex_in_pages.py -q   # confirm YOUR files no longer appear in the failure list
> .venv/Scripts/python.exe -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in [<your files>]]; print('parse ok')"
> ```
> Commit per page: `style(finale): <page> token hygiene + detail`.

### Task 6 (Group A): app.py + My Team + Lineup Optimizer + `.page-title` retirement

**Files:**
- Modify: `app.py:179` (header), `pages/1_My_Team.py`, `pages/2_Line-up_Optimizer.py`, `src/ui_shared.py` (orphaned `.page-title` cleanup ONLY)

- [ ] **Step 1 (app.py header):** Replace line 179's `.page-title` pill markdown with the standard header used by all 13 pages. Read the surrounding function first; the replacement is:

```python
    render_page_header(
        "HEATER",
        eyebrow="FOURZYNBURN LEAGUE · 2026 SEASON",
        fig="FIG.00 · COMMAND HOME",
    )
```

Ensure `render_page_header` is imported from `src.ui_shared` in app.py (add to the existing import list if absent). The splash screen (`.splash-title`) is untouched.

- [ ] **Step 2 (.page-title retirement):** `grep -rn "page-title" app.py pages/ src/` — after Step 1 the only remaining references must be inside `src/ui_shared.py` (CSS defs ~1597–1627, two `@media` variants ~2359/2417, one selector list ~2594, a JS helper ~2765, and the `render_page_header` docstring ~2939). Delete: the `.page-title`/`.page-title-wrap`/`.page-title span` CSS rules, their media-query variants, and the `.page-title` JS-helper block (read it first — if it is generic and serves other classes, remove only the `.page-title` parts). In the ~2594 selector list, remove only the `.page-title,` token. Keep the docstring mention ("Replaces the old navy .page-title pill").

- [ ] **Step 3 (My Team):** Apply the token table — this page's only banned literal class is `#2c2f36` → `var(--fp-tx)`; KEEP the four news-source brand hexes. Then detail: the weekly-matchup hero number (record) gets `class="hero-num"`; any LIVE indicator dot becomes `<span class="chip hot"><span class="dot-live"></span>LIVE</span>` (max one per page).

- [ ] **Step 4 (Lineup Optimizer):** Token table — `#FF9800`→`var(--fp-primary)`, `#9E9E9E`→`var(--fp-tx-muted)`, `#4CAF50`→`{T["green"]}`. Keep the existing `_section_label()` pattern (it's good). Empty states: convert "no lineup yet"-style `st.info` to `render_empty_state`.

- [ ] **Step 5:** Group verification block (see Phase B preamble) + per-page commits.

### Task 7 (Group B): Closer Monitor + Matchup Planner + League Standings + Punt Analyzer

**Files:**
- Modify: `pages/3_Closer_Monitor.py`, `pages/5_Matchup_Planner.py`, `pages/6_League_Standings.py`, `pages/10_Punt_Analyzer.py`

- [ ] **Step 1 (Closer Monitor):** token hygiene only (already gold standard) — no banned literals expected; verify and move on.
- [ ] **Step 2 (Matchup Planner UPLIFT):** Replace `#6b7280` → `var(--fp-tx-muted)`. Then: wrap the category-probability section in `render_panel(build_panel_html("Category Outlook", body, fig_label="FIG.02 · WIN PROBABILITY BY CAT", accent="top"))`; each category row uses `build_heatbar_html(prob_pct, win=prob>=50)`; per-game detail cards get `.instr-panel accent-left` treatment with `.t-eyebrow` date labels and mono figures; the headline win-probability number gets `class="hero-num"` (~34px via existing heading context).
- [ ] **Step 3 (League Standings UPLIFT):** Replace `#6b7280` → `var(--fp-tx-muted)`. Then: "Your Position" rank gets `class="hero-num"`; standings tables route through `render_styled_table` (verify they already do; if raw `st.dataframe`, leave the dataframe but it now inherits Glide chrome); add rank-movement chips (`.chip hot` for ▲-style moves — NO arrows/emoji, use text "UP n"/"DN n" or the `trending_up` SVG from PAGE_ICONS); wrap the three tab bodies' lead sections in instrument panels with `fig_label`s.
- [ ] **Step 4 (Punt Analyzer):** `#2c2f36` → `var(--fp-tx)`. Already uses `render_panel` — add `.hr-heat` under the page's first section header and chips for punt-candidate categories (`.chip cold` for punted, `.chip hot` for protected).
- [ ] **Step 5:** Group verification block + per-page commits.

### Task 8 (Group C): Trade Analyzer + Trade Finder + Free Agents

**Files:**
- Modify: `pages/11_Trade_Analyzer.py`, `pages/12_Trade_Finder.py`, `pages/14_Free_Agents.py`

- [ ] **Step 1 (Trade Analyzer):** `#9ca3af` → `var(--fp-tx-subtle)`, `#6b7280` → `var(--fp-tx-muted)` (×5). The verdict banner stays; add machined detail: `.hr-fade` between builder and results; surplus-SGP headline number gets `class="hero-num"`.
- [ ] **Step 2 (Trade Finder UPLIFT):** `#666666` → `var(--fp-tx-muted)`. Opportunity cards become `.instr-panel accent-left` with: `.t-eyebrow` partner-team label, Archivo-800 player names (existing `.alt .a-name` pattern is fine), `.chip` rows for category deltas (`.chip hot` gains / `.chip cold` losses), mono SGP figures. Use `render_panel`/`build_panel_html` where the card is a plain div today.
- [ ] **Step 3 (Free Agents):** `#2c2f36` → `var(--fp-tx)`. Convert the empty-roster `st.info` to `render_empty_state("No league data yet", no_league_data_message(), icon_key="users")` (import `no_league_data_message` if not present — check its current usage on the page first).
- [ ] **Step 4:** Group verification block + per-page commits.

### Task 9 (Group D): Player Compare + Leaders + Databank + Draft Simulator + analytics badge

**Files:**
- Modify: `pages/16_Player_Compare.py`, `pages/17_Leaders.py`, `pages/19_Player_Databank.py`, `pages/20_Draft_Simulator.py`, `src/ui_analytics_badge.py`

- [ ] **Step 1 (Player Compare):** `#f5f5f5` → `var(--fp-divider)`, `#e8f5e9` → `rgba(31,157,107,.08)`. Radar/Plotly calls must use `get_plotly_layout()`/`get_plotly_polar()` (verify; fix if any literal colorway). Winner-highlight cells keep green-tint via the rgba above.
- [ ] **Step 2 (Leaders):** `#2c2f36` → `var(--fp-tx)`. Add `.hr-heat` under the page header; leaders tables get hover-accent rows automatically (foundation) — verify they use `.heater-table`/`render_styled_table`.
- [ ] **Step 3 (Databank UPLIFT):** wrap the search/profile header in `render_panel(build_panel_html(player_name, body, fig_label="FIG.01 · CAREER LOOKUP", accent="top"))`; season-by-season table routes through `render_compact_table`/`render_styled_table` if currently raw; add `.t-eyebrow` year labels; empty search-state → `render_empty_state("Search the databank", "Type a player name to pull multi-year history.", icon_key="free_agents")`.
- [ ] **Step 4 (Draft Simulator):** `#f97316` → `var(--fp-primary)`, `#ef4444` → `{T["danger"]}`, `#84cc16` → `{T["green_l"]}`, `#22c55e` → `{T["green"]}`. Draft board already styled — verify the `current-pick` heatPulse is the only infinite animation on the page.
- [ ] **Step 5 (analytics badge):** in `src/ui_analytics_badge.py`: `#457b9d` → `T["sky"]`, `#9c27b0` → `T["purple"]`, `#999` → `T["tx_subtle"]` (import `T` if needed).
- [ ] **Step 6:** Group verification block + per-page commits (badge file: also run `.venv/Scripts/python.exe -m pytest tests/ -k "analytics_badge" -q` if tests exist).

---

## Phase C — Integration, verification, ship

### Task 10: Guard green + integration

- [ ] **Step 1:** `.venv/Scripts/python.exe -m pytest tests/test_no_offpalette_hex_in_pages.py tests/test_combustion_lock.py -q` — Expected: ALL PASS.
- [ ] **Step 2:** `git status` clean; review `git diff master --stat` for unexpected files.

### Task 11: Browser verification (run by the orchestrator, not a subagent)

- [ ] **Step 1:** Start the app with preview tools (`streamlit run app.py`, v1 mode — no MULTI_USER env). Wait for bootstrap/splash to finish.
- [ ] **Step 2:** For EVERY page (Home + 13): snapshot + screenshot at desktop width. Verify: grain+grid canvas visible (subtle), machined card edges, no layout breaks, fonts locked (no Source Sans), tab underline glides, hover accent bars on tables, chips/hero-num render, empty states styled, no emoji, no console errors.
- [ ] **Step 3:** Resize to 380px; verify My Team + Lineup Optimizer keep the mobile drawer + readable layout.
- [ ] **Step 4:** Open the player dossier dialog from My Team — verify width ≈1100px, orange top hairline, deep shadow.
- [ ] **Step 5:** Fix anything broken (orchestrator edits directly), re-screenshot, then `git commit` fixes as `style(finale): browser-verify fixups`.

### Task 12: Full suite + docs

- [ ] **Step 1:** `.venv/Scripts/python.exe -m pytest tests/ -n auto --dist loadfile --ignore=tests/test_cheat_sheet.py -q` — Expected: ~5260+ pass, 0 fail.
- [ ] **Step 2:** `python -m ruff check .` clean.
- [ ] **Step 3:** Update `CLAUDE.md`: (a) Streamlit UI section — extend the Combustion bullet with the finale layer (motion tokens, grain canvas, reduced-motion, portal chrome, native config theming) and note `render_empty_state`; (b) structural-invariants table — extend the `test_combustion_lock.py` row text with the finale locks and ADD a row for `test_no_offpalette_hex_in_pages.py`; (c) File Structure — `.streamlit/config.toml` line now says "advanced native theming (Combustion palette, fonts-by-URL, sidebar block)".
- [ ] **Step 4:** Commit docs: `docs(finale): CLAUDE.md design-system + guard updates`.

### Task 13: Ship

- [ ] **Step 1:** `git checkout master && git pull && git merge ui/combustion-finale-2026-06-10` (expect fast-forward or clean merge).
- [ ] **Step 2:** `git push origin master` (pre-push structural suite must pass — includes both new/extended guards). Railway auto-redeploys.
- [ ] **Step 3:** `gh run watch <run-id> --exit-status` → CI green. Delete the branch: `git branch -d ui/combustion-finale-2026-06-10`.
- [ ] **Step 4:** Update memory (`feature_ui_ux_revamp_fantasypros.md` + `project_live_onboarding_state.md`): finale shipped, master SHA, next = invite leaguemates.
