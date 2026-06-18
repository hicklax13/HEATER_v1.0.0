# HEATER Web — 10/10 Quality Uplift (design spec)

**Date:** 2026-06-17
**Track:** CMO / frontend (`web/`)
**Status:** approved direction; pending implementation plan

## Goal

Take the HEATER Next.js app (6 pages: Team, Optimizer, Matchup, Trades, Players,
Research) from a solid ~6.5/10 "clean template" to a flagship-grade 10/10.
Close the gap on three fronts: **life** (motion), **visual sophistication**
(data-viz, depth, signature moment), and **finish** (states, micro-interaction,
rhythm, mobile).

## Direction (locked)

- **Aesthetic:** *Sleek precision + cinematic signature moments.* Calm, fast,
  restrained ~90% of the time (Linear/Vercel/Stripe discipline), with a few
  dramatic, unforgettable beats where they earn it.
- **Strategy:** *Systems-first.* Build a small number of shared advanced systems,
  then apply them across all six pages (parallelized). Highest leverage, keeps
  the app cohesive.
- **Palette/theme:** *Evolve within brand.* Keep the navy/orange Combustion
  identity; add real depth/elevation and richer tones. No wholesale repalette.

## Non-negotiables

- Every change is verified in the **real preview** (real fonts/tokens/motion),
  not isolated mockups.
- `pnpm exec tsc --noEmit` and `pnpm run lint` stay green on every commit.
- Per-page `*-data.ts` contracts are unchanged — viz/motion consume the same
  shapes, so the CEO/backend track can still drop in live data per page.
- `prefers-reduced-motion` is honored by every animation.
- No new monospace; figures stay on the display font (Archivo) per prior decision.
- Streamlit app untouched.

---

## Part 1 — Five shared systems (build once, apply everywhere)

### S1. Motion system
Extend `web/src/lib/motion.ts` with a tokenized set + small primitives:
- `useCountUp(value)` — tabular number count-up, reduced-motion-gated (generalize
  the WinHero pattern).
- `<Reveal>` — staggered mount via framer-motion variants (container + item),
  used to choreograph each page's sections.
- `<CrossFade>` — skeleton → content transition.
- Card/row/chip hover + press springs (shared variants).
- Shared-layout transitions where lists reorder (e.g., optimize apply, filters).

**Apply:** page-mount stagger on all 6; count-ups on hero/score/key-stat numbers;
hover/press on cards, table rows, chips, buttons.

### S2. Data-viz kit — `web/src/components/viz/`
A reusable chart set, Combustion-themed. Use **recharts** (already installed) for
the complex charts; keep hand-rolled SVG for micro-viz. All consume existing data
shapes.
- `WinProbTrend` — win-probability / rank over time (area + marker).
- `CategoryRadar` — you vs opponent across the 12 categories.
- `PercentileBar` / distribution band — where a player/stat sits in the league.
- `TrendSpark` — richer sparkline (gradient fill, end-dot, hi/lo).
- `HeatBar` — category heat (warm = winning, cool = behind).

**Apply:** Team (win-prob trend on the hero, category radar in Category Outlook),
Matchup (radar + heat on the totals), Research (percentile/distribution columns),
Trades (before/after impact viz).

### S3. Depth & material
Add an elevation system to `globals.css` + a `Card`/`Surface` upgrade:
- `--elev-1/2/3`: layered shadows + a subtle top inner-highlight so light surfaces
  read crafted, not flat.
- Optional fine-grain texture token for premium panels (very low opacity).
- Refine the dark heroes: richer gradient stop + a crisp edge light.
- `Card` gains `tone="flat" | "raised" | "inset"`.

### S4. Signature instrument — `web/src/components/viz/HeatGauge.tsx` ⭐
The cinematic centerpiece and the product's face. A live win-probability gauge:
- Arc + hero numeral that **color-shifts cool↔warm** along the `heatColor` ramp by
  value (reads visibly "cooling" at 46).
- Mount: sweep + warm-up animation; idle: subtle thermal shimmer/pulse; glow tuned
  to the value; reduced-motion → static.
- Replaces/upgrades the current WinHero gauge and is reused on Matchup.

### S5. Type & numerals
- Exploit Archivo's `wdth` axis for engineered hero numerals (font-stretch via
  `--fp-*` width tokens); ship a `<HeroNum>` component.
- Tighten the type scale: clear steps for eyebrow / display / title / body /
  figure so hierarchy reads at a glance.

---

## Part 2 — Polish pass (the finish)

- **P1. States:** standardize four states per async data region (loading skeleton
  / empty / error / loaded) via an `AsyncRegion` pattern + `EmptyState` /
  `ErrorState` components. Replace ad-hoc handling on every page.
- **P2. Micro-interaction & focus:** hover/press on all interactive elements;
  consistent `:focus-visible` rings; command-palette refinement (recent, grouping).
- **P3. Spacing & rhythm:** one vertical-rhythm scale + shared table primitives
  (`DataTable` head/cell/row) so density is consistent page-to-page.
- **P4. Player popup:** wire real `mlbId`s into the mock data so headshots load;
  enrich the fallback to feel real; keep FA vs rostered states.
- **P5. Mobile & cohesion:** responsive pass (breakpoints, horizontally
  scrollable wide tables, 44px touch targets); unify the light/dark page chrome so
  Matchup and the dark-hero pages feel like one product.

---

## Architecture

- New: `web/src/components/viz/` (charts + HeatGauge), motion primitives in
  `web/src/lib/motion.ts` (+ a `components/motion/` if components are needed),
  `components/ui/` upgrades (Card tones, AsyncRegion, EmptyState, ErrorState,
  DataTable, HeroNum).
- Tokens: elevation + width-axis + refined tones in `globals.css` / `lib/tokens.ts`.
- Reuse-first: pages compose these; no page owns bespoke chart/motion code.

## Execution order

- **Phase A — foundations:** depth/material tokens, type/numerals, motion
  primitives (cross-cutting; do first).
- **Phase B — signature:** HeatGauge → Team + Matchup (highest wow; ship early to
  set the bar).
- **Phase C — data-viz kit:** build charts, then apply per page (parallelizable).
- **Phase D — polish:** states, micro-interaction, rhythm, popup, mobile/cohesion
  (parallelizable per page/area).

Per-page application in Phases C/D is parallelized via subagents once the shared
systems exist. Each PR-sized chunk is verified in the real preview.

## Success criteria (what 10/10 means here)

1. Every page has choreographed entrance motion + number count-ups.
2. At least one *real* chart per page where it adds genuine insight.
3. The HeatGauge is the recognizable, cinematic face of the product.
4. Consistent elevation, typography, spacing, and four data-states everywhere.
5. Clean on mobile; light/dark pages read as one product.
6. All changes real-preview-verified; tsc + lint green throughout.
