# HEATER — brand & design vision (CMO)

> **Status:** Draft for review · **Date:** 2026-06-16 · **Branch:** `feat/frontend-replatform-foundation`
> **Lane:** CMO (brand, design, frontend, UI/UX, identity, positioning). Parallel to the CEO/Product session's technical migration (engine → FastAPI + Postgres + workers).
> **North Star:** HEATER becomes a real, monetized, public consumer product (see `CLAUDE.md` ★ NORTH STAR + memory `project_north_star_monetized_product`). This document is the brand source of truth for that product.
> **Relationship to other specs:** This supersedes the *brand section* of `2026-06-16-heater-frontend-foundation-design.md`. That spec's stack, architecture, component archetypes, and page list still stand — they re-skin to the identity defined here. Earlier "light-first Combustion hybrid" exploration is retired in favor of the dark "analyst's edge" identity below.

---

> **⚠ LOCKED 2026-06-16 — overrides §3.2 (mark) and §3.3 (dark-first) below.**
> The owner selected the **official HEATER logo**: a flaming-baseball **icon** + a chrome flame **wordmark** (production assets in `brand/`: `heater-icon.png`, `heater-wordmark.png`, transparent + `previews/` on white/navy, and favicon sizes 512/192/180/64/32/16). The abstract **"Hot Zone H" is retired.** The app **theme is LIGHT-first** (the dark-first call below is reversed per owner). The **"heat as data"** heat-ramp (hot = winning / cold = losing) is retained as the *data-viz* language only — the *logo* is the flaming baseball, not the heat grid. Source: owner-chosen Gemini renders, background-removed + sized via `brand_inbox/_process_logos.py`.

## 1. Strategy — the wedge

A competitive scan of 13 fantasy/analytics brands (Sleeper, Underdog, DraftKings, FanDuel, ESPN, Yahoo, PFF, The Athletic, FanGraphs, FantasyPros, Rotowire, Baseball Savant, Fantrax) found a clean, exploitable split:

- **Credible-but-dated:** FanGraphs, FantasyPros, Fantrax, Rotowire — deep data, but they look like *tools*, not products. Bright white, dense tables, system fonts, ad clutter.
- **Fundable-but-shallow:** Sleeper, Underdog, DraftKings — gorgeous, modern, dark, fundable — but casual/gamey, not analytically deep.

**Nobody owns "premium AND deep."** That is HEATER's wedge.

> **Positioning statement:** *HEATER is the analyst's edge for fantasy baseball — pro-grade analytics that finally look and feel like a premium product.*

- **Target customer:** committed fantasy-baseball managers who will pay for an edge (H2H/points/dynasty players, league sweats, second-screen analysts). Not the lapsed-casual market — the people who already open three tabs to set a lineup.
- **Brand personality:** credible · sharp · premium · confident · data-true · hot-blooded. The voice of a great analyst who respects your time and never bluffs.
- **Category cues we adopt (from research):** dark-first UI (the "modern money app" signal), one restrained hero accent, custom type, generous space, a single confident mark. **Category cues we reject:** rainbow palettes, bright-white data dumps, system fonts, ad clutter, cartoon mascots.
- **Color whitespace:** red is ESPN's, green is DraftKings/FanGraphs, blue is FanDuel. **Heat-orange on dark is uncrowded** — and it's already HEATER's DNA.

---

## 2. The big idea — "heat as data"

HEATER's name means a fastball *and* being on fire. We make that literal: **the entire brand is thermal.** One **heat ramp** (cold → hot) is the organizing system for the logo, the data visualization, and the language.

**Core semantic: hot = winning / high value, cold = losing / low value.** This replaces the conventional red/green good-bad pair everywhere in the product.

Three reasons this is the right call, not just a pretty one:
1. **Ownable & differentiated** — no competitor measures in heat; everyone else is red/green/blue.
2. **Intuitive in baseball vernacular** — "he's *on fire*," "the bats are *ice-cold*." The metaphor is already how fans talk.
3. **More accessible** — red/green is exactly the pair ~8% of men (deuteran/protan colorblindness) can't separate; orange↔blue is fully distinguishable. The distinctive choice is also the inclusive one.

---

## 3. Identity

### 3.1 Name
**HEATER.** Kept, unconditionally. Short, ownable, memorable, and a perfect double meaning (fastball + heat/winning). All brand equity stays.

### 3.2 The mark — "Hot Zone H"
A 3×3 strike-zone grid whose lit cells form the letter **H** (left column + right column + center cell; top-middle and bottom-middle are negative space), heat-shaded radially from a white-hot center core out to ember corners.

It reads **three ways at once**:
- a **pitch heat-map** (the analyst/baseball read),
- the **letter H** (the brand initial / app icon),
- a **thermal hotspot** (the "heat as data" core).

This is the synthesis of the three explored directions (Strike Zone 01 + Velocity H 03 + Thermal core 02), locked 2026-06-16.

**System:**
- **Primary lockup:** mark + `HEATER` wordmark (horizontal). A stacked lockup exists for square contexts.
- **App icon / favicon:** the mark in a rounded square; verified legible at **16px** (a test the prior v2.0.0 flaming-baseball logo fails).
- **Variants:** full-color (ember corners → hot legs → white-hot core), one-color (single fill, 7 cells), on-light (near-black on white).
- **Optional motion:** a faint concentric "thermal pulse" + soft glow behind the mark on dark surfaces (splash, login) — the Thermal element, animated. Never on persistent small placements.
- **Clear space:** ≥ one grid-cell on all sides. **Don'ts:** no recolor outside the heat ramp, no drop shadow on the cells, no rotation, no reintroducing literal flames or a realistic baseball.

> The v2.0.0 flaming-baseball logo is **retired** as the product mark (it reads as clip-art and fails the favicon/app-icon test). It may live on as nostalgic/marketing flavor only.

### 3.3 Color system (dark-first)

Tokens as CSS variables; dark is the default/hero, light ships as an option.

**Dark (default)**
| Token | Value | Use |
|---|---|---|
| `--bg` | `#0a0e16` | App canvas (deep night-navy black) |
| `--rail` | `#070a10` | Nav rail / deepest chrome |
| `--surface` | `#0f141d` | Cards, panels |
| `--surface-2` | `#131a24` | Insets, highlighted rows |
| `--border` | `rgba(255,255,255,.08)` | Hairlines |
| `--text` | `#e6edf3` | Primary text |
| `--text-2` | `#9aa6b5` | Secondary |
| `--text-3` | `#6b7688` | Muted / mono captions |
| `--primary` | `#ff6d00` | Brand / action / "hot" |
| `--hot-2` | `#ff8c1a` | Hotter step |
| `--ember` | `#e8551f` | Warm step |
| `--white-hot` | `#ffe6c2` | Core highlight |
| `--cold` | `#5b7da0` | "Cold" signal |
| `--cold-2` | `#3a5a82` | Colder step |
| `--cold-deep` | `#2c466b` | Coldest |
| `--mid` | `#c9923f` | Toss-up (ramp midpoint) |
| `--pos-confirm` | `#4fae8a` | System "live/ok" (non-data, e.g. fresh dot) |

**The heat ramp (signature, 10 stops cold→hot):** `#1e3a5f · #2c4a72 · #3a5a82 · #6f4f63 · #b34a3a · #e8551f · #ff6d00 · #ff8c1a · #ffb24d · #ffe0a8`. Drives win-probability, category edges, hot/cold, and all categorical charts.

**Light (option):** `--bg #ffffff`, `--surface #f5f6f8`, `--text #1b1c20`, `--text-2 #646a78`, `--primary #ff6d00`; the cold end shifts to AA-safe stops on white (`#3a5a82` text-safe). Light is calmer; dark is the brand.

### 3.4 Typography
| Role | Family | Weights | Used for |
|---|---|---|---|
| **Display / wordmark** | Archivo | 600, 700 | `HEATER`, page titles, big stat numerals |
| **Body** | Inter | 400, 500 | Copy, insights, UI labels, buttons |
| **Figures / mono** | JetBrains Mono | 400, 500, 600 | Eyebrows, tabular data, IDs, the heat-ramp captions |

Self-hosted via `next/font` (OFL). This shifts off the earlier broadcast-leaning Saira Condensed toward an **analyst/terminal** voice: Archivo for confident technical display, mono for credible tabular figures (`tabular-nums` everywhere numbers align). `format_stat` precision is law (AVG/OBP `.3f`, ERA/WHIP `.2f`, SGP `+.2f`, Win% integer).

Scale (16px base): hero numeral 54–64px · h1 30 · h2 22 · h3 16 · body 16 · UI label 13 · eyebrow 11 (uppercase, `.16em`).

### 3.5 Voice & tone
Editorial, plain-spoken, and fluent in **heat vernacular** — "ice-cold in steals," "on fire in homers," "your bats are cooling off." Every screen answers *what does this mean / what do I do?* Credibility over hype: no exclamation marks, no jargon without a tooltip, no emoji. Headlines are sentences ("You're 13 steals behind — three free agents fix it."), not labels.

### 3.6 Iconography & motion
- **Icons:** Lucide outline (Tabler fallback for sport-specific glyphs).
- **Motion:** restrained and purposeful. A one-time count-up on hero numerals; bar/heat fills on first paint; a subtle thermal glow/shimmer on dark to make data feel "live." Honor `prefers-reduced-motion`.

---

## 4. Design language, applied

The app is a **premium instrument panel**. The heat ramp is the through-line: win-probability sits on it, category edges are colored by it, hot/cold streaks read on it, every chart uses it.

Signature components (re-skinned from the foundation kit into this identity):
- **Matchup scorebug hero** — you vs opponent, big Archivo win-prob, the value plotted on a cold→hot heat bar.
- **Heat category table** — monospaced, with `edge` and a per-row **heat dot** colored on the ramp (hot = winning the cat). Flip-target row highlighted.
- **"This week's lever"** — the editorial plain-language call to action, in heat vernacular.
- **Freshness chip, PRO tier chip, player dossier, empty states** — all carried from the foundation kit.

Reference: the full-system **My Team** mockup approved 2026-06-16 (dark, Hot Zone H, heat ramp, Archivo/mono).

---

## 5. The product is a brand surface — monetization presentation

(CMO owns *presentation*; CEO owns billing/Stripe.)

From the research, the model that fits a solo-operator-plus-AI product:
- **Freemium, gate the edge.** Free: standings, scores, basic roster. **Pro gates the compute-heavy edge** — the Monte Carlo trade engine, lineup optimizer, playoff odds, FA recommender, and the AI chat.
- **Two tiers, not three** (less to maintain): **Free → Pro**. Optional future **GM/Dynasty** tier for multi-team/advanced sims.
- **Price band (category-aligned):** Pro ≈ **$8.99/mo or $79/yr** (annual ~30–40% off — the FanGraphs/The Athletic pattern). Avoid PFF's $39.99 ceiling unless we go ultra-premium later.
- **Upgrade moments:** surface Pro at the point of value — e.g., a blurred trade-sim verdict with "Unlock the full simulation," the optimizer's "Optimize" button, playoff-odds. Tasteful, never nagging.
- **Brand expression of tiers:** the `PRO` chip uses the hot end of the ramp; Free is neutral. Upgrade = "turn up the heat."

---

## 6. Brand surfaces beyond the app

A monetized product lives or dies at the **front door**, not just the dashboard.
- **Marketing / landing site** — the conversion surface. Dark, cinematic, the Hot Zone H, a live heat-ramp hero, "the analyst's edge" headline, proof (the depth: trade engine, optimizer, sims), pricing, CTA. **A first-class CMO deliverable**, built in the same stack.
- **App icon / PWA**, **social/OG cards**, **email** (receipts, lever-of-the-week digest), **login/splash** (the animated thermal mark).
- **Domain / handles / trademark** — flag for the CEO/legal lane (name "HEATER" availability + mark).

---

## 7. Phasing — the CMO design/brand workstream

Dovetails with the CEO's technical migration; each phase is shippable.

| Phase | CMO deliverable | Depends on |
|---|---|---|
| **D0 — Brand foundation** | This vision + design tokens (dark+light) + the Hot Zone H asset set (SVG, app icon, favicon) + voice guide | — |
| **D1 — Component kit + shell** | The HEATER UI kit (foundation spec §7) skinned to this identity; app shell + nav | D0 |
| **D2 — Foundation pages** | My Team · Lineup Optimizer · Free Agents, fully built on mock data, dark identity | D1; CEO data-contract types |
| **D3 — Landing/marketing site** | Public conversion site + pricing page | D0 |
| **D4 — Page-by-page migration** | Remaining 11 pages, re-skinned, wired to the API | CEO API (Sub-project B) |
| **D5 — Light mode + polish** | Light theme parity, motion pass, a11y audit | D2 |

---

## 8. Deliverables & acceptance (D0–D2, the immediate build)
1. Token files (dark + light) + the heat-ramp scale, committed under `web/lib`.
2. Hot Zone H asset set: SVG (color/mono/on-light), app icon, favicon at 16/32/180/512.
3. The UI kit + app shell, skinned to this identity, documented.
4. My Team, Lineup Optimizer, Free Agents built, clickable, on real Team-Hickey fixtures, dark-first.
5. AA contrast verified on text; the heat semantic verified separable for colorblind simulation.
6. A deployed preview (desktop + phone).

---

## 9. Risks & open questions
- **Red/green → heat usability:** intuitive in baseball terms + colorblind-safe, but worth a quick check with a few real leaguemates that "cold = losing" reads instantly. Mitigation: always pair heat color with the `+/−` edge value and the ramp legend.
- **Light-mode parity:** the heat ramp must stay legible on white (cold end needs AA-safe stops). Resolved in D5.
- **Name/domain/trademark:** "HEATER" availability — hand off to CEO/legal.
- **Scope discipline:** the landing site (D3) is tempting to gold-plate; keep it to one strong page until there's revenue.
- **Two sessions, one working tree:** CMO writes stay in `web/` + new docs; coordinate doc edits with the CEO session.

---

## 10. Next step
Owner reviews this vision → on approval, `writing-plans` turns D0–D2 into a task-by-task build plan for the design/brand workstream, executed in `web/` alongside the CEO migration.
