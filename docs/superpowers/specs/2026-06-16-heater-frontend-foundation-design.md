# HEATER frontend re-platform — Sub-project A: design system & frontend foundation

> **Status:** Draft for review · **Date:** 2026-06-16 · **Branch:** `feat/frontend-replatform-foundation`
> **Owner:** Connor (Team Hickey) · **Author:** design brainstorm session
> **Supersedes for the frontend:** the Streamlit-bound "Combustion Index" presentation layer (`src/ui_shared.py` + `pages/*`). The Python analytics layer is **kept** (see §2).

---

## 1. Why this exists

HEATER's UI lives inside Streamlit. Streamlit is excellent for shipping analytics fast, but it caps how professional and distinctive the presentation can get: you fight its widget DOM, custom interaction needs React components with a build step, and the visual ceiling is "good dashboard," not "premium product." The owner has explicitly chosen to **re-platform the frontend to a modern web stack** to lift that ceiling.

This document specs **only the first sub-project (A)**: the design system + a runnable frontend foundation. It is the part that is genuinely "design," it produces something beautiful and clickable fast, and — because it's built in the real target stack — it becomes the actual frontend skeleton rather than throwaway mockups.

---

## 2. Program context (the whole board)

A full re-platform is a multi-month program, decomposed into four sub-projects. Each gets its own spec → plan → build cycle.

| # | Sub-project | What it is | Status |
|---|---|---|---|
| **A** | **Design system + frontend foundation** | Brand guide, design tokens (light + dark), component kit, app shell + nav, and 3 fully-built pages, on mock data, in Next.js. | **This spec** |
| B | Backend API (FastAPI) | Expose the existing Python engines as typed JSON endpoints; auth; Yahoo data service. | Future |
| C | Page migration (×14) | Rebuild each remaining page in React against the API. | Future |
| D | Cutover & hosting | Run both apps in parallel, migrate the 12 users, retire Streamlit. | Future |

### The load-bearing principle: keep the brain, replace the face

HEATER's value is its Python analytics — `src/engine/` (6-phase trade pipeline), `src/optimizer/` (21-module LP optimizer), `src/valuation.py` (SGP/VORP), Bayesian projections, Monte Carlo sims, and ~6,300 passing tests. **None of that is rewritten.** The re-platform replaces the *presentation layer* only (roughly `pages/` + `ui_shared.py`). Sub-project B will wrap the existing engines in a thin FastAPI layer; this React frontend consumes that API. This is the "thin-client" pattern HEATER's own `CLAUDE.md` names as the deferred re-platform path — now explicitly green-lit.

**Consequence for Sub-project A:** we build the frontend against a **typed mock-data layer** (`lib/data/*`) whose function signatures are designed to match the future API. When the API lands (B), we swap the implementation behind those functions; pages don't change.

---

## 3. Goals & non-goals (Sub-project A)

### Goals
1. A distinctive, professional-grade visual identity for HEATER — anchored on the v2.0.0 logo — that reads modern, high-tech, and trustworthy to a **novice** user.
2. A documented, reusable component library ("HEATER UI kit") with light (default) + dark themes.
3. A runnable Next.js app shell + navigation reflecting the in-season information architecture.
4. Three fully-built, clickable pages — **My Team, Lineup Optimizer, Free Agents** — proving the system across the three page archetypes the rest of the app reuses.
5. Deployed as a cheap static preview the owner can click through on desktop and phone.
6. WCAG 2.1 AA accessibility and real mobile responsiveness baked in from the start.

### Non-goals (explicitly deferred to later sub-projects)
- The FastAPI backend and any real data wiring (Sub-project B).
- The other 11 pages (Sub-project C).
- Auth, multi-user, Yahoo OAuth, the AI chat panel, admin surfaces (later).
- Cutover, user migration, retiring Streamlit (Sub-project D).
- Pixel-faithful reproduction of every analytics output — fixtures approximate real outputs using real Team Hickey numbers.

---

## 4. Design direction (locked)

**Name:** *Combustion · Field* — the next generation of the Combustion Index system, light-first.

**The hybrid:** one spine, two disciplines.

- **Broadcast spine** — the personality. Stadium-night confidence translated to a light canvas: big condensed numerals, hot-orange accents, a matchup "scorebug" hero. Straight from the v2.0.0 logo's energy. *(ESPN / DAZN / sportsbook.)*
- **Editorial discipline** — the comprehension layer. Generous whitespace, clear hierarchy, and plain-language "what does this mean / what do I do?" insights so a novice is never lost. *(The Athletic.)*
- **Quant data layer** — the precision. Monospaced tabular figures, tight aligned grids, color used **only as signal**, for the deep numbers power users want. *(Bloomberg terminal.)*

**Theme:** **light-first** (default), dark mode available. Rationale: the audience is 12 casual leaguemates; light is more approachable and preserves continuity with today's white canvas. The navy chrome rail (today's deep-navy sidebar) carries over unchanged, so the re-platform reads as an *evolution* of Combustion, not a betrayal of it.

**Brand-token continuity:** the core palette is the existing `THEME` (`src/ui_shared.py`): orange `#ff6d00` primary, navy `#112744` chrome, green `#1f9d6b` positive, ember `#e0492f` functional-negative. We extend it (dark surfaces, warm tints) but never break it. The structural rule **orange = action/positive/brand, never decoration** carries forward verbatim.

The three approved mockups (Broadcast / Quant / Editorial comparison, then the My Team hero in dark and light) are the visual reference for this direction.

---

## 5. Brand foundation

### 5.1 Logo (v2.0.0)
- Primary asset: flaming-baseball mark + chrome "HEATER" wordmark with orange flame trail.
- `HEATER_Logo_v2.0.0_Transparent_Background.*` — on light/photographic surfaces.
- `HEATER_Logo_v2.0.0_Blue_Background.*` — on the navy chrome (rail header, splash, login).
- **Rail mark:** a simplified flaming-baseball glyph in an orange rounded square (the wordmark is too wide for the 56px rail). Export an SVG mark for crisp scaling.
- Clear space ≥ the height of the baseball; never recolor the wordmark; never place the transparent (chrome-on-light) lockup on a busy photo without a scrim.

### 5.2 Color tokens

Defined as CSS variables; both themes ship day one. Light is default.

**Light (default)**
| Token | Value | Use |
|---|---|---|
| `--bg` | `#ffffff` | App canvas |
| `--surface` | `#f5f6f8` | Cards, metric tiles |
| `--surface-2` | `#eef0f3` | Insets, table zebra |
| `--border` | `#e3e2de` | Hairline borders |
| `--border-strong` | `#d8d6cf` | Emphasis/hover borders |
| `--text` | `#1b1c20` | Primary text |
| `--text-2` | `#646a78` | Secondary text |
| `--text-3` | `#9aa0ac` | Muted/hints |
| `--chrome` | `#0e2038` | Nav rail / top bar (navy) |
| `--chrome-ink` | `#eef1f6` | Text on chrome |
| `--chrome-muted` | `#8aa0ba` | Inactive rail icons |
| `--primary` | `#ff6d00` | Brand / action / positive accent |
| `--primary-hover` | `#ff7a1a` | Hover |
| `--primary-press` | `#e8480a` | Active/press |
| `--on-primary` | `#ffffff` | Text/icon on orange |
| `--warm-bg` / `--warm-border` / `--warm-ink` | `#fff5ec` / `#ffd9b8` / `#b8480a` | "Lever"/insight tints |
| `--pos` | `#1f9d6b` | Ahead / good |
| `--neg` | `#e0492f` | Behind / bad (functional ember) |
| `--caution` | `#c4870f` | Toss-up (AA-safe gold on white) |
| `--info` | `#5f7d9c` | Neutral/steel |
| `--focus` | `#ff6d00` | Focus ring |

**Dark (optional)**
| Token | Value |
|---|---|
| `--bg` | `#0a1626` |
| `--surface` | `#0f2238` |
| `--surface-2` | `#0d1d31` |
| `--border` | `rgba(255,255,255,.09)` |
| `--text` / `--text-2` / `--text-3` | `#eef3f9` / `#9fb0c2` / `#6f8196` |
| `--chrome` | `#070f1c` |
| `--primary` / `--primary-hover` | `#ff6d00` / `#ff9a3c` |
| `--pos` / `--neg` / `--caution` | `#2bbd86` / `#ff5a3c` / `#ffae42` |

**Signal semantics (both themes):** green = you're ahead / sustainable / buy; ember = behind / risk / sell; gold = toss-up / watch; orange = action or brand emphasis. Color is **never** the only signal — pair with text, sign (+/−), or icon (carry forward today's accessibility rule).

### 5.3 Typography (four roles)

Self-hosted via `next/font` (all OFL / Google Fonts — no licensing cost).

| Role | Family | Weights | Used for |
|---|---|---|---|
| **Display numerals** | Saira Condensed | 600, 700 | Hero stats (46%), big numbers, scorebug team names |
| **Title** | Archivo | 600, 700 | Page titles (`Team Hickey.`), section heads, wordmark |
| **Body** | Inter | 400, 500 | Body copy, insights, UI labels, buttons |
| **Figures** | IBM Plex Mono | 400, 500, 600 | Eyebrows, tabular data, IDs, stat tables |

Scale (rem, 16px base): hero numeral 56–64px · h1 30px · h2 22px · h3 16px · body 16px · UI label 13–14px · eyebrow 11px (uppercase, `letter-spacing .16em`). Two body weights only (400/500). **Tabular figures everywhere numbers align** (`font-variant-numeric: tabular-nums`).

`format_stat` precision is law and ports verbatim: AVG/OBP `.3f`, ERA/WHIP `.2f`, SGP `+.2f`, counting stats integer, Win% as whole-number %.

### 5.4 Space, radius, motion
- Spacing scale (4px base): 4, 8, 12, 16, 20, 24, 32, 48.
- Radius: `sm 8` · `md 10` · `lg 12` · `xl 16`. No rounded corners on single-sided accent borders.
- Elevation: flat with hairline borders (matches today's aesthetic). Soft shadow only on floating overlays (menu, dialog, toast).
- Motion: 120ms (micro) / 200ms (panel) ease-out. Broadcast energy = a one-time count-up on hero numerals + bar fills on first paint, **not** persistent animation. Honor `prefers-reduced-motion` (no count-ups, instant states).

### 5.5 Voice & tone
Editorial and plain-spoken. Every coined metric answers, in the UI, "what is this / what do I do?" via a tooltip (port the existing `JARGON` dict + `jargon_help`). Headlines are sentences, not labels ("You're 13 steals behind — three free agents fix it."). No hype, no jargon without a definition, no emoji (carry forward).

### 5.6 Iconography
Outline icon set: **Lucide** (primary — React-native, tree-shakeable, huge coverage), with **Tabler** as fallback for any sport-specific glyph Lucide lacks. Replaces today's inline-SVG `PAGE_ICONS`. Injury/health states are colored dots (port `build_health_dot_html` semantics). Every icon has an accessible label or `aria-hidden`.

---

## 6. Design tokens → implementation

- Tokens live in `web/lib/tokens.css` as CSS custom properties (light under `:root`, dark under `[data-theme="dark"]`).
- Tailwind reads them via a thin theme extension (`tailwind.config.ts` maps `colors.primary` → `var(--primary)`, etc.) so utilities stay on-brand and a token change cascades everywhere.
- We ship **our** scale (colors, spacing, radius, type roles) — not Tailwind's defaults — so the result can't drift into generic.
- A `tokens.md` reference doc is part of the deliverable.

---

## 7. Component library — the HEATER UI kit

Each component is a bounded unit: clear purpose, typed props, documented states. Many port a current `src/ui_shared.py` helper (noted) so behavior/semantics carry over.

| Component | Purpose | Ports from |
|---|---|---|
| `AppShell` | Rail + top bar + content frame; theme switch | `inject_custom_css` chrome |
| `NavRail` / `MobileNav` | Desktop icon rail; mobile bottom bar/drawer | `src/nav.py` + sidebar |
| `PageHeader` | Eyebrow `SECTION · / FIG.NN — TITLE` → big `Title.` (orange period) → orange rule | `render_page_header` |
| `Card` / `Panel` | Surface container | `.card` surfaces |
| `MetricCard` | Label + big numeral (record, IP pace, moves left) | metric cards |
| `StatTable` | Monospaced, tabular, signal-colored; trusted-HTML cells opt-out of escaping | `build_compact_table_html` (`html_cols`) |
| `CategoryBar` / `WinProbBar` | Win-prob heatbar, per-category edge bars | Matchup heatbar |
| `Chip` / `Badge` / `HealthDot` | Status, tier, injury dots | `build_health_dot_html`, chips |
| `Button` | primary / secondary / ghost | buttons |
| `Tabs` | Section tabs w/ animated underline | `st.tabs` |
| `Select` / `Input` / `Slider` / `Toggle` | Form controls (Radix-backed) | widgets |
| `Dialog` / `Sheet` | Player dossier, confirmations | `st.dialog` chrome |
| `Tooltip` / `GlossaryPopover` | Jargon help, definitions | `jargon_help`, `render_glossary_expander` |
| `EmptyState` | Instrument-styled no-data panel | `render_empty_state` |
| `FreshnessChip` | Honest data-age chip (amber > 24h) | `render_data_freshness_chip`, `humanize_age` |
| `InsightCard` | The editorial "this week's lever" callout + CTA | new pattern |
| `PlayerRow` / `PlayerCard` | Player line item + hover/detail | `player_card` |
| `Sparkline` / `TrendChart` | Inline + full charts (Recharts) | Plotly figures |
| `Skeleton` / `LoadingState` | Async placeholders | spinners |
| `Banner` / `Toast` | Broadcast + maintenance notices | `app_settings` banners |

Charts use **Recharts** themed to the tokens (visx is the upgrade path for bespoke viz). Every chart carries an `aria-label` + text fallback and a non-color secondary cue.

---

## 8. Information architecture & navigation

In-season is the default mode (mirror `src/nav.is_in_season()`), landing on **My Team** — not the Draft Tool. The rail groups:

- **Season:** My Team (default) · Lineup Optimizer · Closer Monitor · Pitcher Streaming · Matchup Planner · League Standings · Punt Analyzer
- **Trades:** Trade Analyzer · Trade Finder
- **Wire:** Free Agents
- **Research:** Player Compare · Leaders · Player Databank
- **Preseason:** Draft Tool · Draft Simulator

Sub-project A ships the rail with all entries; only the 3 foundation pages route to real screens, the rest to a styled "coming soon in the new app" placeholder.

---

## 9. Page specs (the 3 foundation pages)

### 9.1 My Team (archetype: dashboard / landing)
- **PageHeader** (`Team Hickey.`) + record/rank subline + **FreshnessChip**.
- **Matchup scorebug hero** — you vs opponent, big orange win-prob, tie/loss/projected (broadcast).
- **InsightCard** — the week's single biggest lever in plain language + CTA.
- **CategoryTable** — you / opp / edge / win% per category, flip-target row highlighted.
- **MetricCard** row — IP pace, moves left, roster count.
- States: loading skeleton; empty (no matchup) via `EmptyState`; stale-data amber chip.

### 9.2 Lineup Optimizer (archetype: interactive tool)
- **PageHeader** + a controls panel (date, scope, optimize button, rate-mode toggles).
- **Results region**: optimized lineup as `StatTable`/`PlayerRow`s with start/sit + forced-start flags, a projected-category delta strip, and a "why" detail per slot.
- Async pattern: optimize → loading state → results (mirrors the existing async optimize UX). With mock data the "compute" is a short simulated delay returning a fixture lineup.
- States: pre-run prompt, running skeleton, results, error/empty.

### 9.3 Free Agents (archetype: data table / leaderboard)
- **PageHeader** + filter bar (position, level default "MLB only", category-fit, search) + sort.
- **StatTable** of FA candidates: rank, player, positions, marginal value, ownership heat, fit chips; row → `PlayerCard` dossier (`Dialog`).
- Add/drop affordance (confirm `Dialog`), CSV export button.
- States: results, empty filter result via `EmptyState`, loading skeleton.

All three consume `lib/data/*` fixtures keyed on real Team Hickey numbers (§11).

---

## 10. Technical architecture

```
web/
  app/                      # Next.js App Router
    layout.tsx              # AppShell, theme provider, fonts
    page.tsx                # → My Team
    lineup/page.tsx
    free-agents/page.tsx
    (placeholder)/...       # styled "coming soon" for non-foundation routes
  components/               # the HEATER UI kit (§7), one folder per component
  lib/
    tokens.css             # CSS variables (light + dark)
    data/                   # typed mock-data layer (future API swap point)
      myTeam.ts  lineup.ts  freeAgents.ts  types.ts
    format.ts              # port of format_stat precision rules
    fixtures/              # real Team Hickey JSON
  styles/                  # globals
  tailwind.config.ts
  package.json
```

- **Next.js (App Router) + TypeScript**, static-export-able for free preview hosting (Vercel).
- **Tailwind** mapped to our token layer; **Radix** primitives for accessible behavior; **Recharts** for charts; **Lucide/Tabler** icons; **next/font** for the four families.
- **Data-access seam:** pages call typed functions (`getMyTeam()`, `getFreeAgents(filters)`) that today return fixtures and tomorrow call the FastAPI endpoints — the contract is designed now so Sub-project B is a backend-only change.
- Lives in a new `web/` folder; the Streamlit app and Python package are **untouched** and keep running.

---

## 11. Data fixtures (real numbers)

Use the live audit snapshot so the foundation feels real: Team Hickey **3-7-1**, **10th of 12**, 7 GB from 1st; **Week 12** vs "The Good The Vlad The Ugly", win **46% / tie 19% / loss 35%**, projected **6-6**; categories R/HR/RBI/SB/AVG/OBP + W/L/SV/K/ERA/WHIP (inverse L/ERA/WHIP); roster **23 active + 4 IL**; biggest gap **SB −13**. Fixtures are illustrative where exact engine output isn't reproducible without the backend.

---

## 12. Accessibility & responsive

- **WCAG 2.1 AA.** Keyboard-navigable (Radix), visible orange `:focus-visible` ring, screen-reader labels on icons/charts. **Contrast rule (carried forward):** orange `#ff6d00` on white fails AA for small text — orange is for large numerals, icons, accents, and fills only; body text is charcoal/navy.
- **Mobile-first responsive** (leaguemates are on phones): rail collapses to a bottom nav/drawer < 768px; tables scroll horizontally or stack; touch targets ≥ 44px; hero/cards scale. Carries forward the Phase 7 mobile + a11y work rather than re-deriving it.
- Dark mode via `[data-theme]`, persisted per user.

---

## 13. Deliverables & acceptance criteria

Sub-project A is done when:
1. `web/` runs locally (`npm run dev`) and builds a static export.
2. A deployed preview URL works on desktop + phone.
3. Brand guide + `tokens.md` + component-kit docs exist.
4. Light + dark themes both ship and pass AA contrast on text.
5. The UI kit (§7) is implemented and used by the pages.
6. My Team, Lineup Optimizer, Free Agents are fully built, clickable, and populated with the §11 fixtures; non-foundation routes show the styled placeholder.
7. Keyboard nav + screen-reader labels verified on the 3 pages; mobile layout verified at 375px and 768px.

---

## 14. Risks & open questions
- **Chart parity:** Recharts must cover what Plotly did; bespoke viz may need visx later. (Low risk for the 3 foundation pages.)
- **Fixture realism:** without the backend, some outputs are approximations — clearly an interim state, resolved in Sub-project B.
- **Preview hosting:** Vercel free tier assumed; confirm before deploy.
- **API contract:** designed implicitly via `lib/data/types.ts` now; formalized in Sub-project B.
- **Font set:** four families is intentional (distinct roles); revisit if load weight is an issue (all are subsettable via `next/font`).

---

## 15. What happens after this spec
1. Owner reviews this spec.
2. On approval → `writing-plans` produces the task-by-task implementation plan for Sub-project A.
3. Build the foundation in `web/`, screenshotting pages for review as we go.
4. Then Sub-projects B → C → D, each its own spec/plan/build cycle.
