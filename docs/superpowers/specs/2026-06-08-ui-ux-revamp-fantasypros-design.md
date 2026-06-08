# HEATER UI/UX Revamp — fantasypros.com-inspired (design spec)

> **Status: APPROVED DESIGN (2026-06-08).** Build is a **separate, focused session**
> (owner choice). This spec + the brief (`docs/design/2026-06-07-ui-ux-revamp-fantasypros-brief.md`)
> carry over. No implementation in the design session — the next step is `writing-plans`.

## Goal & direction

Restyle the entire HEATER app to feel like **fantasypros.com** — thin icon-rail sidebar,
lots of whitespace, clean rounded cards/tables, a single clean sans — **while keeping
HEATER's soul**: the red/amber/cream palette stays, red remains the primary/action color
(not FP's yellow), and a touch of the bold display feel is retained on the brand mark only.

### Decisions locked in brainstorming (2026-06-08)
1. **Intensity:** FP-inspired, keep HEATER soul (not a 1:1 clone; palette stays HEATER).
2. **Sequencing:** Approve this spec now; implement in a fresh session.
3. **Primary color:** keep **red `#e63946`** as primary/action (FP uses yellow — the red is
   what keeps it unmistakably HEATER).
4. **Typography:** keep **Figtree** for body *and* headings; **retire Bebas Neue all-caps**
   from page headers (the biggest "feel" change — softens the shouty sports vibe toward FP's
   calm); keep **IBM Plex Mono** for stat numbers. Bebas may remain ONLY on the logo/splash.
5. **Exact FP font:** not required (we keep Figtree). If the owner later wants a literal FP
   match, confirm FP's `font-family` via browser DevTools during the build and swap the
   `--font-body` token — a one-token change.

## Current state (what we're changing)

- **`src/ui_shared.py`** (~3,716 lines) — the styling hub:
  - `THEME` dict (line ~324): `bg #f4f5f0`, `card #ffffff`, `primary #e63946`, `warn #ff9f1c`,
    `tx #1d1d1f`, `border #d4d5cf`, `ink #ffffff`, plus `good/bad/muted/surface` etc.
  - Fonts via Google Fonts `@import` (line ~610): Figtree + Bebas Neue + IBM Plex Mono.
  - `inject_custom_css()` (line ~600) — the single CSS injector run on every page.
  - Many custom HTML renderers: `render_compact_table`, `build_compact_table_html`,
    `render_roster_table`, `render_styled_table`, `render_position_pills`,
    `build_category_heatmap_html`, `_headshot_img_html`, `sec()`, `get_plotly_layout`.
  - `PAGE_ICONS` (line ~24) — inline 24×24 SVGs (no emoji; reuse for the icon rail).
- **`src/nav.py` + `app.py`** — role-aware `st.navigation()` builds the sidebar (under MULTI_USER).
- **`.streamlit/config.toml`** — Streamlit theme (Heater palette).
- **`pages/*.py`** (13 in-season pages) — each calls `inject_custom_css()` + renders its own
  tables/cards; many use the `ui_shared` renderers above.

## Design system

### A. Color tokens (extend `THEME`, keep the base palette)
Keep: `primary #e63946`, `warn/amber #ff9f1c`, cream/white surfaces, `tx #1d1d1f`.
Add/clarify FP-style neutrals (proposed — finalize in build):
- `app_bg` (page) `#f6f7f9` (slightly cooler than the current cream so white cards pop FP-style)
- `surface` (cards) `#ffffff`
- `sidebar_bg` (dark rail) `#10213a` (HEATER-charcoal-navy) with `sidebar_ink #e8edf5`,
  active item = `primary` red
- `border` soften to `#e6e8ec`; `divider` `#eef0f3`
- text hierarchy: `tx #1d1d1f`, `tx_muted #5b6470`, `tx_subtle #8a929c`
- Keep `good`/`bad`/`warn` for stat deltas (HEATER green/red/amber).
All as CSS custom properties (`--fp-*`) on `:root` so pages reference tokens, not literals.
**Constraint:** Plotly 6 needs `rgba()` not 8-digit hex (existing gotcha).

### B. Typography
- `--font-body: 'Figtree', system-ui, sans-serif;` for everything (body + headings).
- **Retire `Bebas Neue`** from `.sec-head` and page headers → Figtree 600/700, title case,
  with a clear scale: h1 ~28/600, h2 ~20/600, h3 ~16/600, body ~14/400, caption ~12/500.
- Keep `IBM Plex Mono` for numeric/stat cells (`format_stat` output, tables) — tabular feel.
- Reduce ALL-CAPS usage broadly (FP uses sentence/title case).

### C. Sidebar → thin icon rail
- Restyle `st.navigation`'s sidebar via CSS to a compact **~68px dark rail**: vertically
  stacked items = `PAGE_ICONS` SVG (≈22px) + an ≈10px label under it; active item gets a red
  left-accent + red icon tint; hover lightens. Collapse the wide text nav.
- **Mobile:** must keep the sidebar/menu toggle (guard `test_mobile_nav.py`: header hidden only
  `@media (min-width:768px)`) — the rail collapses to the Streamlit drawer on phones.
- Logo at the top of the rail (small HEATER mark; Bebas allowed here).

### D. Layout & spacing
- White cards on `--app-bg`; generous padding (cards ~20px, sections ~24-32px gaps).
- Constrain main content width for readability; keep the existing full-width fix where needed
  (`inject_custom_css` block-container max-width — verify per page).
- Fewer borders/dividers; rely on whitespace + soft shadows for separation (FP calm).
- Optional right-rail pattern for pages with secondary cards (where it fits; not forced).

### E. Components
- **Cards:** white, `border-radius: 12px`, soft shadow `0 1px 3px rgba(16,33,58,.08)`, roomy padding.
- **Tables** (the `ui_shared` renderers): lighter FP treatment — subtle/borderless header,
  row hover, more row height/air, mono numerics, sticky header retained. Restyle the renderers'
  emitted CSS, NOT each page (DRY).
- **Buttons:** rounded (`8px`), solid red primary / outline secondary; consistent sizing.
- **Tabs:** clean FP tab strip (underline-active, muted inactive); applies to the 6-tab Lineup,
  7-tab Leaders, etc.
- **Badges/pills** (`render_position_pills`, injury dots): softer, rounded, FP-muted.
- **Plotly** (`get_plotly_layout`): match the new palette (transparent bg, Figtree font, muted grid).

## Files to modify (build session)
- **Primary:** `src/ui_shared.py` (THEME tokens, `inject_custom_css`, all table/card/pill
  renderers, `get_plotly_layout`/`get_plotly_polar`, `sec`).
- `.streamlit/config.toml` (theme to match).
- `src/nav.py` / `app.py` (sidebar rail markup/classes if needed beyond CSS).
- `pages/*.py` (13) — per-page whitespace/layout/heading-case cleanup (mostly consume the new
  system; minimal per-page CSS).
- Splash/`app.py` draft page (logo/brand treatment).

## Structural guards to update (in lockstep — don't weaken)
- `test_mobile_nav.py` — keep the desktop-only header hide; verify the icon rail collapses on mobile.
- `test_standings_theme_keys.py` + any `T["key"]` usage — if THEME keys change, update subscripts
  (BR-4 class: a missing key crashes a page).
- `test_pages_format_compliance.py` — keep `format_stat` usage (no inline rate-stat formatting).
- `tests/test_*` referencing fonts/CSS/THEME — update expected values where styling changed.
- No-emoji rule stays (SVGs only).

## Build sequence (for the plan)
1. **Design-system pass** in `ui_shared.py`: tokens → `inject_custom_css` (fonts, base, cards,
   buttons, tabs, sidebar rail) → table/pill renderers → plotly layout. Screenshot a couple of
   pages before/after.
2. **Sidebar rail** wired + verified desktop + mobile (Playwright).
3. **Per-page sweep** across the 13 pages: heading case, whitespace, remove redundant local CSS,
   confirm tables/cards inherit the new look. One commit per logical group; screenshot each page.
4. **Guards** updated + full suite green; structural-invariant gate green.
5. **Review** (frontend-design polish + a visual pass) → deploy from `master` **with owner confirm**
   (single replica; redeploy resets sessions) → live screenshot verification per page.

## Open items (confirm during build)
- FP exact `font-family` (only if the owner wants a literal match — else keep Figtree).
- Final neutral token hex values (A) — tune against real screenshots.
- Whether to keep Bebas on the splash/logo (recommend: yes, small touch of HEATER).

## Success criteria
- Every page reads visibly "FP-clean" (icon rail, whitespace, rounded white cards, calm type)
  while still HEATER (red primary, amber accents, cream/white).
- No page crashes; all structural guards + the full CI suite green; mobile usable (rail collapses).
- Owner approves the look on a live before/after walkthrough.

## Constraints (carried from project rules)
- `MULTI_USER` flag-off stays byte-for-byte where guards require it.
- No emoji in UI (inline SVGs only); Streamlit CSS needs `!important`; `T["ink"]` for text-on-accent.
- Single Railway replica (hard invariant); deploys from `master`; **confirm pushes/deploys with the
  owner** (CLI novice). Big visual change → screenshot before/after per page.
