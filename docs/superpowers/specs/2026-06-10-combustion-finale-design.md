# Combustion Finale — Launch-Grade Micro-Detail & Consistency Pass (Design Spec)

**Date:** 2026-06-10
**Status:** Approved for implementation (owner directive 2026-06-10: "finish the UI/UX redesign overhaul … more visual and background detail, VERY minor but a LOT more detail … enhanced visual effects, UI/UX elements, custom designs … HEATER colorway … consistent designs, text sizes and fonts across all pages … as modern as possible … highest quality frontend design elements and packages … clean and professional, launch-grade")
**Branch:** `ui/combustion-finale-2026-06-10`

## 1. Goal

Take the deployed Combustion Index design from "styled" to "machined": add a layer of very fine visual detail (textures, edges, motion, typography) across the whole app, eliminate every remaining cross-page inconsistency (off-palette colors, divergent header/empty-state patterns), and adopt the highest-quality platform capabilities available (Streamlit ≥1.46 native advanced theming, Archivo variable-font axes). Zero behavior changes. Light mode only. The look stays exactly Combustion — off-white canvas, navy chrome, orange `#ff6d00`, Archivo/Inter/IBM Plex Mono.

## 2. Non-Goals

- No new pages, features, or data flows.
- No layout restructuring of pages that already carry instrument detail (My Team, Closer Monitor are the gold standard — they get the foundation layer for free and only token-hygiene edits).
- No dark mode.
- No flattening of deliberate in-panel micro-typography (8.5–13.5px instrument labels are intentional detail, not drift).
- `src/cheat_sheet.py` (print/PDF artifact, tests skipped on Windows) is out of scope.

## 3. Package & Platform Decisions (researched 2026-06-10)

**Adopt nothing from PyPI.** Every candidate package (streamlit-extras, streamlit-lottie, streamlit-shadcn-ui, streamlit-antd-components, streamlit-card, streamlit-elements, hydralit, streamlit-aggrid, streamlit-echarts, st-theme) renders in an iframe (breaking the font-lock and tokens), is unmaintained in 2026, or both. The app's hand-rolled HTML components are structurally higher quality. `stylable_container` is obsolete — core `st.container(key=…)` → `.st-key-*` is native.

**Adopt instead (the actual "highest-quality packages"):**
1. **Streamlit native advanced theming** (`.streamlit/config.toml`, all ≥1.46 — the pinned `streamlit>=1.40` resolves to ~1.56 in Docker): `linkColor`, `borderColor`, `baseRadius`, `buttonRadius`, `dataframeBorderColor`, `dataframeHeaderBackgroundColor`, `headingFontWeights`, `chartCategoricalColors` (= `THEME["tiers"]`), Google-Fonts-URL `font`/`headingFont`/`codeFont`, and a `[theme.sidebar]` block (navy/bone/orange). This natively themes surfaces CSS can't reach (Glide canvas first-paint font, BaseWeb internals) and reduces `!important` warfare. The existing CSS font-lock stays (structurally tested) but stops being load-bearing.
2. **Archivo variable-font `wdth` axis (62–125)** in the Google-Fonts URLs — unlocks `font-stretch: 112%` scoreboard-DNA hero numerals and `font-stretch: 87%` dense table headers from the same file. No new fonts.
3. **Font preconnect** — `<link rel="preconnect" href="https://fonts.gstatic.com">` to cut cold-load FOUT. (Self-hosting woff2 deferred — deploy-surface change not worth it for 12 users.)

## 4. Workstream 1 — Foundation Layer (`src/ui_shared.py` + `.streamlit/config.toml`)

All additions live in `inject_custom_css()` (CSS braces f-string-escaped `{{ }}`) and config.toml. New reusable classes are opt-in; nothing existing breaks.

### 4.1 Canvas & surface detail
- **Layered grain + blueprint grid on `.stApp`**: 350-byte feTurbulence SVG data-URI noise (opacity baked ≤0.05) + 28px navy hairline grid layers composited under `var(--fp-app-bg)`. The whole app stops being a flat dead background; offset vs the panels' 20px dot grid avoids moiré.
- **Machined top-edge**: add `inset 0 1px 0 rgba(255,255,255,.9)` to `.instr-panel`, `.glass`, `.cmd-bar`, `.hero` box-shadows — the milled-edge premium tell.
- **Conic heat wash variant**: `.instr-panel.heat::before` adds a second conic-gradient layer (opt-in; hero panels only, most panels stay quiet).
- **Sidebar rail detail**: subtle top-to-bottom navy gradient already exists; add a 1px inner-right hairline `rgba(255,255,255,.06)` and nav section-header microlabel styling (`[data-testid="stNavSectionHeader"]`) for the MULTI_USER Admin group.

### 4.2 Motion system
- **Motion tokens**: `--dur-1: 120ms; --dur-2: 180ms; --ease-snap: cubic-bezier(.2,.7,.3,1)` in `:root`; migrate existing ad-hoc transition literals on touched rules.
- **Fix entrance-animation rerun thrash**: remove `slideUp` from persistent cards (`.glass`, `.hero`, `.alt`, `.metric-card`, `.player-card`, `.cat-card`) — Streamlit remounts DOM on every rerun so they currently re-animate on every widget click. Keep entrances only for genuinely-entering surfaces (`.lock-in`, splash title, toasts/dialog).
- **Standardized micro-hover**: cards get `translateY(-1px)` + shadow lift at `--dur-2`; table rows get background tint + `inset 3px 0 0 var(--fp-primary)` accent bar at `--dur-1` (no layout shift).
- **Animated tab underline**: style `[data-baseweb="tab-highlight"]` (orange, 2px, 240ms glide) + `[data-baseweb="tab-border"]` (divider). Consolidate the two conflicting tab rulesets (early glass-pill vs late flat-underline) into the flat-underline one.
- **`prefers-reduced-motion` block** at end of stylesheet (kills all animation for vestibular-sensitive users).

### 4.3 Interaction chrome
- **Custom scrollbars** (standard properties only — Chromium ≥121 disables `::-webkit-scrollbar` when `scrollbar-color` is set): thin, `#c9c8c3` on canvas, translucent bone on navy sidebar.
- **Brand `::selection`**: navy bg/white text in main; orange in sidebar.
- **`:focus-visible` ring system**: 2px orange outline, offset 2, keyboard-only.
- **Tooltip + toast portal chrome**: BaseWeb portals mount on `<body>` outside `.stApp` — brand them navy with orange left rail (toast) and navy pill (tooltip).
- **Expander polish**: hairline border, summary hover tint, chevron 180ms rotate.
- **Dialog chrome (player dossier)**: `div[data-testid="stDialog"] div[role="dialog"]` → `width: min(92vw, 1100px)`, 16px radius, deep shadow + machined inset, orange top hairline via `::before`. Pass `width="large"` at the `st.dialog` call site.

### 4.4 Typography & numerics
- **Numeric law**: `font-variant-numeric: tabular-nums slashed-zero` blanket rule on `.stat`, `.mono`, `td.num`, table cells, `[data-testid="stMetricValue"]`, mono-font elements.
- **`.hero-num` class**: Archivo-900, `font-stretch: 112%`, orange gradient text-clip (`#ff8a2a → #ff6d00 → #e85f00` — orange family only, never red) for the 2–3 biggest numerals per page.
- **Reading polish**: `text-wrap: balance` on headings/`.sec-head`, `text-wrap: pretty` on paragraphs, `scroll-margin-top` on anchors.
- **Type-scale utility classes**: formalize the existing micro-scale as named classes — `.t-eyebrow` (10px/0.28em up), `.t-fig` (mono 10px), `.t-label` (11px up), `.t-caption` (12px muted), so future code stops minting inline sizes. Existing coherent inline sizes inside instrument panels are left as-is.

### 4.5 Component upgrades
- **`.chip` system**: inner white ring (`inset 0 0 0 1px rgba(255,255,255,.6)`) + hover lift; `.dot-live` 6px pulsing dot (the ONLY new infinite animation; max 1–2 visible per page).
- **`.hr-fade` / `.hr-heat` gradient hairline dividers** + restyle `[data-testid="stDivider"] hr` to the fade variant.
- **`.empty-state` class + `render_empty_state(title, body, icon_key)` helper**: instrument-panel-styled empty-data state (dot grid, centered, muted ink, orange icon) to replace bare `st.info()` in data-empty contexts.
- **Sticky table headers**: scroll-hairline shadow on `thead th` in scrollable wrappers.
- **Glide grid extras**: `--gdg-accent-color` (orange selection ring), `--gdg-accent-light`, lighter horizontal borders, `--gdg-rounding-radius`, search-result wash.
- **`:has()` conditional chrome**: `st.metric` containers self-tint by delta direction (green/ember 4–5% washes).

### 4.6 config.toml (additive; `[server]` keys untouched — XSRF is structurally guarded)
Full `[theme]` upgrade per §3.1 + `[theme.sidebar]`. `font = "sans serif"` comment removed (stale since 1.46).

## 5. Workstream 2 — Consistency Sweep (all 13 pages + app.py + ui_analytics_badge)

### 5.1 Token hygiene (mechanical, per-page)
Replace off-palette hex literals with tokens/THEME lookups:
- Material leakage: `#FF9800→var(--fp-primary)`, `#9E9E9E/#666666/#6b7280/#9ca3af→var(--fp-tx-muted)` (or `--fp-tx-subtle` for the lightest), `#4CAF50/#22c55e/#84cc16→{t["green"]}`, `#ef4444→{t["danger"]}`, `#f97316→var(--fp-primary)`, `#f5f5f5/#e8f5e9→` surface/green-tint tokens.
- Near-charcoal drift: `#2c2f36→var(--fp-tx)` in pages (ui_shared's `--gdg-text-header` keeps it).
- `src/ui_analytics_badge.py`: `#457b9d→{t["sky"]}`, `#9c27b0→{t["purple"]}`, `#999→{t["tx_subtle"]}`.
- **Intentional exceptions (do NOT change):** news-source brand colors in My Team (ESPN/Yahoo/RotoWire/MLB), `TEAM_BRAND` colors, ember `#e0492f` on functional negatives.

### 5.2 Pattern unification
- **Home (app.py)**: retire the lone `.page-title` navy-pill header; use `render_page_header()` with eyebrow + fig like every other page. Add the foundation detail (grain canvas arrives free). Splash screen keeps its reveal animation.
- **Empty states**: swap bare `st.info`/`st.warning` in data-empty contexts for `render_empty_state()` (validation/transient warnings stay native).
- **KPI convention**: custom `.metric-card`/stat-readout HTML for computed summaries; `st.metric` only where delta arrows matter (now self-tinting via `:has()`).
- **Dividers**: replace `st.divider()`/`---` and flat border rules with `.hr-fade`/`.hr-heat`.

### 5.3 Undercooked-page uplift (to Closer-Monitor standard)
Four pages scored "undercooked" in the audit and get a real detail pass (instrument panels, eyebrow/fig labels, corner ticks, heatbars, mono figures, hover rows — using ONLY existing + WS1 classes):
- `6_League_Standings` — standings tables → instrument treatment, rank-movement chips, `.hero-num` for "Your Position".
- `5_Matchup_Planner` — category probability readouts → heatbar + chip treatment, per-game cards get panel chrome.
- `12_Trade_Finder` — opportunity cards → `.instr-panel accent-left` + chips + mono figures.
- `19_Player_Databank` — search results → panel chrome, season rows → compact-table treatment.

## 6. Workstream 3 — Guards & Verification

- **Extend `tests/test_combustion_lock.py`**: motion tokens emitted; reduced-motion block present; `::selection` present; `:focus-visible` rule present; grain data-URI present on `.stApp`; no `slideUp` on the six persistent-card classes; dialog width rule present.
- **New structural guard `tests/test_no_offpalette_hex_in_pages.py`**: pages/ + app.py must not contain the known-bad literals (`#9E9E9E`, `#4CAF50`, `#ef4444`, `#84cc16`, `#6b7280`, `#9ca3af`, `#FF9800`, `#f97316`, `#22c55e`, `#666666`, `#2c2f36`) — with an explicit allowlist for the My Team news-source brand colors.
- **Browser verification**: run the app (MULTI_USER off, sample data), screenshot every page at desktop width + 2 pages at 380px mobile, verify: grain visible, no layout breaks, tabs glide, dialog chrome, no emoji, fonts locked.
- **Full suite + ruff** green before merge; pre-push structural suite as final gate.

## 7. Constraints (locked invariants — violating any fails CI)

- `THEME["primary"] == "#ff6d00"`; @import keeps Archivo+Inter+IBM Plex Mono (adding `wdth` axis params is allowed; Figtree/Bebas banned).
- No emoji anywhere (icons = inline SVG).
- Sidebar: navy rail, thin width rule stays inside `@media (min-width: 768px)`; mobile keeps drawer + header/toolbar (mobile-nav fix).
- Full-width block container (`max-width: none`).
- `.heater-link`/`.clickable` orange+underline convention; content-link rule scoped to main, never `.stSidebar`.
- CSS in `inject_custom_css()` f-string: braces escaped `{{ }}`; Plotly gets `rgba()` not 8-digit hex.
- Animate only `transform`/`opacity`/`box-shadow`-at-card-scale; max 1–2 infinite animations visible per page; no `backdrop-filter` on large surfaces; no entrance keyframes on persistent content.
- `enableXsrfProtection = true` stays in config.toml.
- All pages keep `inject_custom_css()` → `require_auth()` → `log_page_view()` ordering and every other structural guard.

## 8. Execution Shape

1. **Phase A (sequential):** WS1 foundation + config.toml + extended lock tests — single subagent, because every later task depends on the new classes/tokens.
2. **Phase B (4 parallel subagents):** page-group sweeps —
   A: `app.py`, `1_My_Team`, `2_Line-up_Optimizer` · B: `3_Closer_Monitor`, `5_Matchup_Planner`, `6_League_Standings`, `10_Punt_Analyzer` · C: `11_Trade_Analyzer`, `12_Trade_Finder`, `14_Free_Agents` · D: `16_Player_Compare`, `17_Leaders`, `19_Player_Databank`, `20_Draft_Simulator`, `src/ui_analytics_badge.py`.
   Groups touch disjoint files; each applies §5.1 hygiene + §5.2 patterns, and B/C/D carry the §5.3 uplift pages.
3. **Phase C (sequential):** new off-palette guard test, browser verification sweep, full suite, fix-ups, merge to master + push (Railway auto-redeploy) per owner directive.
