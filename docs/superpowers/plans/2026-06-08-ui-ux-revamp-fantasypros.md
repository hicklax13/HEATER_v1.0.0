# HEATER FP-inspired UI/UX Revamp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Also use superpowers:test-driven-development for the guard tasks and plugin-dev/pr-review-toolkit:code-reviewer + frontend-design for the polish pass.

**Goal:** Restyle the entire HEATER Streamlit app to feel like fantasypros.com — thin dark icon-rail sidebar, lots of whitespace, clean rounded white cards + lighter tables, one clean sans (Figtree, no Bebas all-caps) — while keeping HEATER's soul (red `#e63946` primary, amber `#ff9f1c`, cream/white).

**Architecture:** Front-load a shared design system in `src/ui_shared.py` (CSS custom-property tokens + `inject_custom_css` + the table/card/pill renderers + the Plotly layout) so all 13 pages inherit the look; then a light per-page sweep (heading case, whitespace, remove redundant local CSS). Verification = structural guards stay green + before/after Playwright screenshots per page (CSS is not unit-testable).

**Tech Stack:** Streamlit (multi-page), CSS-in-`st.markdown` injected by `inject_custom_css()`, Google Fonts (Figtree, IBM Plex Mono), Plotly, Playwright (MCP) for screenshots, pytest for the structural guards.

**Spec:** `docs/superpowers/specs/2026-06-08-ui-ux-revamp-fantasypros-design.md`. **Brief:** `docs/design/2026-06-07-ui-ux-revamp-fantasypros-brief.md`.

---

## Verification approach (read first)

- **Guards (unit-testable):** after every CSS/theme change, run `python -m pytest tests/test_standings_theme_keys.py tests/test_mobile_nav.py tests/test_pages_format_compliance.py tests/test_no_merge_conflict_markers.py -q`. They must stay green. If you change/remove a `THEME` key, fix every `T["key"]` subscript (the BR-4 crash class) in the same task.
- **Visual (screenshots):** before starting, capture a baseline; after each page, capture again. Use Playwright (MCP) against a local `streamlit run app.py` (login as `testuser`/`test1234` — a member — or the admin). Save to `docs/design/screenshots/<page>-{before,after}.png`. Do NOT commit large PNGs to git history unless the owner wants them; reference them in the PR/walkthrough.
- **No-emoji rule:** icons are inline SVGs from `PAGE_ICONS` — never add emoji.
- **Streamlit specificity:** Streamlit inline CSS is high-specificity; nearly every rule needs `!important`.
- **Plotly 6:** use `rgba(r,g,b,a)`, never 8-digit hex.
- **MULTI_USER flag-off:** must stay byte-for-byte where guards require; styling is flag-agnostic, but run the flag-off backcompat guards if you touch `app.py`/`nav.py`.
- **Deploy:** single replica; deploy from `master`; **confirm every push/deploy with the owner** (CLI novice). A redeploy resets sessions.

## File structure (what changes + why)

- `src/ui_shared.py` — **primary.** `THEME` tokens, `inject_custom_css()` (the one CSS injector), table renderers (`render_compact_table`, `build_compact_table_html`, `render_roster_table`, `render_styled_table`), `render_position_pills`, `sec()`, `get_plotly_layout()`/`get_plotly_polar()`. One responsibility: app-wide styling. (It's 3,716 lines — do NOT restructure it; edit the styling functions in place.)
- `.streamlit/config.toml` — Streamlit base theme (must match the new palette).
- `pages/*.py` (13) — light per-page sweep: heading case + whitespace; remove redundant local `<style>`/CSS that fights the new system.
- `app.py` / `src/nav.py` — sidebar rail (CSS in `ui_shared`; only touch these if the rail needs markup/classes beyond CSS).
- `tests/test_*` — update guards in lockstep where styling assertions change.

---

## Task 0: Branch + baseline screenshots

**Files:** none (setup).

- [ ] **Step 1:** Create the branch off `master`.

```bash
git checkout master && git pull origin master
git checkout -b ui/fp-revamp-2026-06-08
```

- [ ] **Step 2:** Start the app locally and capture a baseline of 4 representative pages (My Team, Lineup Optimizer, League Standings, Free Agents) at desktop (1280) and mobile (390) via Playwright. Save under `docs/design/screenshots/`.

Run: `streamlit run app.py` (separate shell), then drive Playwright to each page. (Login: `testuser`/`test1234` or admin.)
Expected: 8 baseline PNGs captured. These are the "before".

- [ ] **Step 3:** Commit the setup note (not the PNGs unless owner wants them).

```bash
git commit --allow-empty -m "chore(ui): start FP revamp branch + baseline screenshots captured"
```

---

## Task 1: Design tokens (THEME + CSS custom properties)

**Files:**
- Modify: `src/ui_shared.py` (`THEME` dict ~line 324; `inject_custom_css()` ~line 600)
- Test: `tests/test_standings_theme_keys.py` (existing — must stay green)

- [ ] **Step 1: Write/extend the guard test** that every key referenced as `T["..."]` across pages exists in `THEME`, and that the new keys are present.

```python
# tests/test_theme_tokens_complete.py
from src.ui_shared import THEME

def test_new_fp_tokens_present():
    for k in ("bg", "surface", "sidebar_bg", "sidebar_ink", "border", "divider",
              "tx", "tx_muted", "tx_subtle", "primary", "warn", "ink", "card"):
        assert k in THEME, f"THEME missing '{k}'"
```

- [ ] **Step 2: Run it — expect FAIL** (`surface`/`sidebar_bg`/`tx_muted`/etc. not yet in THEME).

Run: `python -m pytest tests/test_theme_tokens_complete.py -q`
Expected: FAIL (KeyError-style assertion).

- [ ] **Step 3: Add the tokens to `THEME`** (keep existing keys; add the FP neutrals). Keep `primary`/`warn`/`tx`/`ink`/`card`/`good`/`bad`.

```python
THEME = {
    # --- kept (HEATER soul) ---
    "primary": "#e63946",   # red — stays the primary/action color
    "warn": "#ff9f1c",      # amber accent
    "tx": "#1d1d1f",
    "ink": "#ffffff",       # text on accent
    "card": "#ffffff",
    "good": "#2a9d8f", "bad": "#e63946",  # (keep existing values)
    # --- FP-style neutrals (new/adjusted) ---
    "bg": "#f6f7f9",          # cooler app bg so white cards pop (was #f4f5f0)
    "surface": "#ffffff",
    "sidebar_bg": "#10213a",  # dark charcoal-navy rail
    "sidebar_ink": "#e8edf5",
    "border": "#e6e8ec",      # softened (was #d4d5cf)
    "divider": "#eef0f3",
    "tx_muted": "#5b6470",
    "tx_subtle": "#8a929c",
    # ...preserve any other existing keys verbatim...
}
```

- [ ] **Step 4:** In `inject_custom_css()`, emit the tokens as CSS custom properties on `:root` (so renderers reference vars, not literals):

```css
:root {
  --fp-app-bg:#f6f7f9; --fp-surface:#fff; --fp-primary:#e63946; --fp-amber:#ff9f1c;
  --fp-tx:#1d1d1f; --fp-tx-muted:#5b6470; --fp-tx-subtle:#8a929c;
  --fp-border:#e6e8ec; --fp-divider:#eef0f3;
  --fp-sidebar-bg:#10213a; --fp-sidebar-ink:#e8edf5;
  --fp-radius:12px; --fp-radius-sm:8px; --fp-shadow:0 1px 3px rgba(16,33,58,.08);
  --font-body:'Figtree',system-ui,-apple-system,sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
}
.stApp{ background:var(--fp-app-bg) !important; }
```

- [ ] **Step 5: Run guards — expect PASS.**

Run: `python -m pytest tests/test_theme_tokens_complete.py tests/test_standings_theme_keys.py -q`
Expected: PASS. (If any page used a key you renamed, fix its `T["..."]` now.)

- [ ] **Step 6: Commit.**

```bash
git add src/ui_shared.py tests/test_theme_tokens_complete.py
git commit -m "feat(ui): FP design tokens + CSS custom properties (revamp task 1)"
```

---

## Task 2: Typography — retire Bebas all-caps, Figtree everywhere

**Files:** Modify `src/ui_shared.py` (`inject_custom_css()` font `@import` ~line 610 + all `font-family:'Bebas Neue'` rules ~lines 677-804; `.sec-head` in `sec()`).

- [ ] **Step 1:** Keep the Google Fonts import for **Figtree + IBM Plex Mono**; drop Bebas Neue from the import (or keep ONLY for an optional `.heater-logo` class).

```css
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
```

- [ ] **Step 2:** Replace every `font-family:'Bebas Neue',sans-serif; text-transform:uppercase` rule with Figtree, title case, FP scale:

```css
.sec-head{ font-family:var(--font-body)!important; font-weight:700!important;
  text-transform:none!important; font-size:20px!important; letter-spacing:-.01em;
  color:var(--fp-tx)!important; }
h1,.heater-h1{ font-family:var(--font-body)!important; font-weight:700!important; font-size:28px!important; text-transform:none!important; }
h2{ font-weight:600!important; font-size:20px!important; }
h3{ font-weight:600!important; font-size:16px!important; }
/* numeric/stat cells keep the mono face */
.stat,.mono,td.num{ font-family:var(--font-mono)!important; }
```

- [ ] **Step 3 (optional):** Keep Bebas ONLY on the brand mark (`.heater-logo`) for a touch of soul; re-add a minimal `family=Bebas+Neue` import scoped to that one class.

- [ ] **Step 4: Verify** — `grep -n "Bebas Neue" src/ui_shared.py` shows only the logo (or none). Run the format + theme guards green. Screenshot My Team — headers should read as clean Figtree, not all-caps.

- [ ] **Step 5: Commit.**

```bash
git add src/ui_shared.py && git commit -m "feat(ui): retire Bebas all-caps headers; Figtree type scale (revamp task 2)"
```

---

## Task 3: Base components — cards, buttons, tabs, inputs

**Files:** Modify `src/ui_shared.py` (`inject_custom_css()`).

- [ ] **Step 1:** Add FP-style component CSS (append inside the `<style>` block):

```css
/* cards / metrics */
.fp-card,[data-testid="stMetric"],.metric-card{ background:var(--fp-surface)!important;
  border:1px solid var(--fp-border)!important; border-radius:var(--fp-radius)!important;
  box-shadow:var(--fp-shadow)!important; padding:20px!important; }
/* buttons */
.stButton>button{ border-radius:var(--fp-radius-sm)!important; font-weight:600!important;
  border:1px solid var(--fp-border)!important; }
.stButton>button[kind="primary"],.stButton>button[data-testid="baseButton-primary"]{
  background:var(--fp-primary)!important; color:var(--fp-ink,#fff)!important; border-color:var(--fp-primary)!important; }
/* tabs — FP underline strip */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--fp-divider); }
.stTabs [data-baseweb="tab"]{ font-weight:600; color:var(--fp-tx-muted); padding:8px 14px; }
.stTabs [aria-selected="true"]{ color:var(--fp-primary)!important; box-shadow:inset 0 -2px 0 var(--fp-primary); }
/* inputs */
.stTextInput input,.stSelectbox div[data-baseweb="select"]{ border-radius:var(--fp-radius-sm)!important; border-color:var(--fp-border)!important; }
/* spacing — more air */
.block-container{ padding-top:1.5rem!important; max-width:1180px; }
```

- [ ] **Step 2: Verify** — screenshot Lineup Optimizer (6 tabs) + a page with buttons. Tabs = underline-active red; buttons rounded; cards white/rounded. Guards green.

- [ ] **Step 3: Commit.**

```bash
git add src/ui_shared.py && git commit -m "feat(ui): FP cards/buttons/tabs/inputs + roomier spacing (revamp task 3)"
```

---

## Task 4: Table renderers — light FP treatment (DRY)

**Files:** Modify `src/ui_shared.py` — `render_compact_table` / `build_compact_table_html` (~2517/2810), `render_roster_table` (~2763), `render_styled_table` (~577), `_build_default_column_config` (~2866).

- [ ] **Step 1:** In the CSS these renderers emit, replace heavy borders/dark headers with the FP-light look:

```css
.fp-tbl{ width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.fp-tbl thead th{ background:transparent!important; color:var(--fp-tx-subtle)!important;
  font-weight:600!important; text-transform:none!important; border-bottom:1px solid var(--fp-divider)!important;
  position:sticky; top:0; }
.fp-tbl tbody td{ border-bottom:1px solid var(--fp-divider)!important; padding:8px 10px!important; }
.fp-tbl tbody tr:hover td{ background:#fafbfc!important; }
.fp-tbl td.num{ font-family:var(--font-mono)!important; text-align:right; }
```

(Keep the existing class names if the renderers already use them; otherwise map the existing classes to these rules. Preserve sticky-header + health-dot behavior.)

- [ ] **Step 2:** For `st.dataframe`-based tables (glide-data-grid), update the `--gdg-*` CSS vars (already present ~line 622) to the FP palette + Figtree.

- [ ] **Step 3: Verify** — screenshot My Team roster + Free Agents board + Leaders. Tables read lighter/airier, mono numerics, hover. Run `python -m pytest tests/test_pages_format_compliance.py -q` (green — `format_stat` still used).

- [ ] **Step 4: Commit.**

```bash
git add src/ui_shared.py && git commit -m "feat(ui): FP-light table renderers (revamp task 4)"
```

---

## Task 5: Pills/badges + Plotly layout

**Files:** Modify `src/ui_shared.py` — `render_position_pills` (~2418), injury-dot CSS, `get_plotly_layout`/`get_plotly_polar` (~536/548).

- [ ] **Step 1:** Soften pills/badges (rounded, muted FP fills) + injury dots (keep CSS-dot, FP colors).

```css
.pos-pill{ border-radius:999px!important; background:#eef1f5!important; color:var(--fp-tx-muted)!important;
  font-weight:600!important; padding:2px 8px!important; font-size:11px!important; }
```

- [ ] **Step 2:** Update `get_plotly_layout()` to the new palette — transparent paper/plot bg, Figtree font, muted gridlines, `rgba()` only:

```python
layout = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Figtree, sans-serif", color=t["tx"]),
    xaxis=dict(gridcolor="rgba(16,33,58,0.06)"), yaxis=dict(gridcolor="rgba(16,33,58,0.06)"),
    colorway=[t["primary"], t["warn"], "#2a9d8f", "#457b9d", "#8a929c"],
)
```

- [ ] **Step 3: Verify** — screenshot Matchup Planner (charts) + Player Compare (radar). Charts match the palette; no 8-digit-hex errors in console. Guards green.

- [ ] **Step 4: Commit.**

```bash
git add src/ui_shared.py && git commit -m "feat(ui): FP pills/badges + Plotly palette (revamp task 5)"
```

---

## Task 6: Sidebar → thin dark icon rail

**Files:** Modify `src/ui_shared.py` (`inject_custom_css()` — sidebar CSS); verify `src/nav.py`/`app.py` only if markup is needed. Test: `tests/test_mobile_nav.py` (must stay green).

- [ ] **Step 1:** Add sidebar-rail CSS. **Confirm the exact Streamlit selectors in the running app** (DevTools) — `stSidebar`, `stSidebarNav` DOM differs by Streamlit version; these are the current best-known:

```css
section[data-testid="stSidebar"]{ background:var(--fp-sidebar-bg)!important; width:72px!important; min-width:72px!important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul{ padding-top:8px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{ display:flex; flex-direction:column;
  align-items:center; gap:4px; padding:10px 4px!important; color:var(--fp-sidebar-ink)!important;
  font-size:10px!important; text-align:center; border-radius:8px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{ background:rgba(255,255,255,.06)!important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{
  background:rgba(230,57,70,.14)!important; box-shadow:inset 3px 0 0 var(--fp-primary); color:#fff!important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a svg{ width:22px;height:22px; }
```

- [ ] **Step 2:** Ensure each nav item shows its `PAGE_ICONS` SVG + a short label (truncate long page names). If `st.navigation` doesn't render icons, inject them per-item via the page `title=` icons or a small JS shim — prefer the CSS/`PAGE_ICONS` route; document the chosen mechanism.

- [ ] **Step 3:** **Mobile:** the desktop header-hide must remain `@media (min-width:768px)` only (so the drawer toggle survives on phones). Run:

Run: `python -m pytest tests/test_mobile_nav.py -q`
Expected: PASS. Then Playwright at 390px — the rail collapses to the Streamlit drawer + the menu toggle is visible.

- [ ] **Step 4: Verify** — screenshot any page desktop (rail ~72px, icons + labels, active = red) + mobile (drawer works).

- [ ] **Step 5: Commit.**

```bash
git add src/ui_shared.py && git commit -m "feat(ui): thin dark icon-rail sidebar (revamp task 6)"
```

---

## Task 7: Streamlit base theme (`config.toml`)

**Files:** Modify `.streamlit/config.toml`. Test: `tests/test_streamlit_security_settings.py` (keep `enableXsrfProtection=true`).

- [ ] **Step 1:** Match the base theme to the new palette (Streamlit reads these before CSS):

```toml
[theme]
base="light"
primaryColor="#e63946"
backgroundColor="#f6f7f9"
secondaryBackgroundColor="#ffffff"
textColor="#1d1d1f"
font="sans serif"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_streamlit_security_settings.py -q` — PASS (don't drop XSRF/CORS settings).

- [ ] **Step 3: Commit.**

```bash
git add .streamlit/config.toml && git commit -m "feat(ui): Streamlit base theme matches FP palette (revamp task 7)"
```

---

## Tasks 8–20: Per-page sweep (one task per page)

> The design system now drives the look. Each page task is the SAME procedure — apply it per page so each is a self-contained, screenshot-verified commit. Pages (stems): `1_My_Team`, `2_Line-up_Optimizer`, `3_Closer_Monitor`, `5_Matchup_Planner`, `6_League_Standings`, `10_Punt_Analyzer`, `11_Trade_Analyzer`, `12_Trade_Finder`, `14_Free_Agents`, `16_Player_Compare`, `17_Leaders`, `19_Player_Databank`, `20_Draft_Simulator`.

**Per-page procedure (repeat for each):**

- [ ] **Step 1:** `grep -nE "<style>|st.markdown\(.*<|text-transform:uppercase|Bebas" pages/<PAGE>.py` — find local CSS/markup that fights the new system.
- [ ] **Step 2:** Remove/duplicate-free the redundant local CSS (let `inject_custom_css` own it); convert ALL-CAPS section headers to title case via `sec()`/the new `.sec-head`.
- [ ] **Step 3:** Wrap loose content blocks in `.fp-card` where it reads as a card; add whitespace (`st.markdown("<br>")` → prefer container spacing). Don't change data logic.
- [ ] **Step 4:** Run that page's guards + a render smoke:
  `python -m pytest tests/test_pages_format_compliance.py tests/test_pages_have_auth_guard.py -q`
  and a quick AppTest/import check that the page still renders (no exception).
- [ ] **Step 5:** Playwright screenshot the page desktop + mobile; compare to baseline.
- [ ] **Step 6: Commit** `git add pages/<PAGE>.py && git commit -m "feat(ui): FP sweep <PAGE> (revamp task N)"`.

(Front-load the high-traffic pages: My Team, Lineup, Matchup, Trade Analyzer, Free Agents, Standings.)

---

## Task 21: Splash / brand mark (`app.py`)

**Files:** Modify `app.py` (splash/home + logo treatment). Test: `tests/test_app_no_hardcoded_categories.py`, `tests/test_no_direct_sqlite_connect_in_scripts.py` (keep green).

- [ ] **Step 1:** Apply the new tokens to the splash/home; keep (optionally) Bebas on the HEATER wordmark only. Don't touch `main()` control flow / the scheduler/auth gates.
- [ ] **Step 2:** Run `python -m pytest tests/test_app_main_auth_gate.py tests/test_plan4_scheduler_wiring.py -q` — PASS (styling didn't disturb the gates).
- [ ] **Step 3: Commit.**

---

## Task 22: Update structural guards + full suite

**Files:** `tests/test_mobile_nav.py`, `tests/test_standings_theme_keys.py`, `tests/test_pages_format_compliance.py`, any font/CSS-asserting test.

- [ ] **Step 1:** Update any guard whose expected styling string changed (e.g., a test asserting a specific font name or THEME value) — update the expectation, do NOT weaken the guard's intent.
- [ ] **Step 2:** Add a guard locking the new look where it matters (e.g., `test_no_bebas_allcaps_headers` — `.sec-head` must not be `text-transform:uppercase`; sidebar width rule present).
- [ ] **Step 3: Run the full suite + structural gate.**

Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q`
Expected: all green (~5,100+). Then the pre-push structural gate.

- [ ] **Step 4: Commit.**

---

## Task 23: Review, merge, deploy (owner-gated)

- [ ] **Step 1:** Run `pr-review-toolkit:code-reviewer` + `coderabbit:code-review` over the branch diff; address findings.
- [ ] **Step 2:** Assemble a before/after screenshot deck (all 13 pages, desktop + a mobile spot-check); send to the owner via the chat for sign-off.
- [ ] **Step 3:** On owner approval: fast-forward `master`, **confirm the push with the owner**, push → Railway redeploys (resets sessions). 
- [ ] **Step 4:** Live before/after walkthrough (Playwright vs the live URL) per page; confirm no crashes, mobile usable, the look matches the deck. Update `qa-findings.md` / the memory: UI revamp shipped.

---

## Self-review (spec coverage)
- Sidebar icon rail → Task 6 ✓ · Whitespace/layout → Tasks 3,8-20 ✓ · Color tokens → Task 1,7 ✓ ·
  Typography (retire Bebas, Figtree) → Task 2 ✓ · Cards/tables/buttons/tabs → Tasks 3,4 ✓ ·
  Pills/Plotly → Task 5 ✓ · Per-page → Tasks 8-21 ✓ · Guards → Task 22 ✓ · Review+deploy → Task 23 ✓.
- Exact FP font: kept Figtree per the approved spec (one-token swap noted in Task 2 if owner later wants a literal match).
- All steps cite exact files + concrete CSS/commands; verification = guards-green + screenshots (UI is not unit-TDD-able).
