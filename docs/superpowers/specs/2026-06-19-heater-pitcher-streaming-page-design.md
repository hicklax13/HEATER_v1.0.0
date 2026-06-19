# HEATER Web — Pitcher Streaming page (Stream Finder) + "Analyze Any Starter" — design spec

**Date:** 2026-06-19
**Track:** CMO / frontend (`web/`) — first M1 missing-page build (Sub-project C)
**Status:** approved direction; pending implementation plan
**Roadmap:** M1 page parity (`docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md`)

## Goal

Build the **Pitcher Streaming** page as a React surface in `web/`, and add a new
**"Analyze Any Starter"** capability the owner requested: pick *any* probable
starter in MLB (SP / SP-RP / RP) for a given day and run it through HEATER's
streaming-score algorithm, with a full factor-by-factor readout.

This is the **first of the ~8 missing pages** in M1. It ships **Stream Finder
first** (the highest-value tab); the other three Streamlit tabs are deferred.

## Scope

**In scope (this build):**
- The **Stream Finder** surface: date selector, budget strip, the heat-ranked
  **Stream Board**, top-pick callout, per-pitcher "why" breakdown.
- The new **Analyze Any Starter** feature: a probable-pitcher picker (any MLB
  starter, any near date) → a **Scorecard** with the factor breakdown.
- A signature data-viz (`StreamScorecard`): heat gauge + diverging factor bars.
- Mock-first data, typed so the mock shape **is** the API contract spec for the
  CEO track. Live wiring is deferred (slice-0 pattern, env-flag gated).

**Out of scope (YAGNI — later passes):**
- The **Matchup Microscope**, **Week Planner**, and **Track Record** tabs.
- Building the scoring **algorithm** itself (CEO lane — `src/`/`api/`).
- Adding a **Vegas / betting-odds data source** (phase-2 fast-follow, owner-set).
- Live API wiring (waits on the CEO extending `/api/streaming` + the analyze
  endpoint; this spec defines that contract).

## Lane split (two-track model)

- **Frontend (this track, `web/`):** the page, the picker, the scorecard, the
  viz, the typed mock, and this contract spec.
- **Algorithm + endpoints (CEO track, `src/`/`api/`):** the scoring engine and
  the `/api/streaming` extension + the analyze endpoint. HEATER already has a
  streaming engine (`src/optimizer/stream_analyzer.py`) producing a 0–100 Stream
  Score from **6 weighted components**; the new scorer **enhances** it (owner
  decision: "enhance now, Vegas next"). This spec is the requirement input.

## Context — the parity target & the divergence

- **Streamlit page:** `pages/4_Pitcher_Streaming.py` — 4 tabs. We port the
  **Stream Finder** tab and add the new picker; the other tabs are deferred.
- **Existing engine output (`build_stream_board`):** per candidate it already
  returns score, status, confidence, `num_starts`, `net_sgp`, `opp_wrc_plus`,
  `opp_k_pct`, `park_factor`, `expected_ip/k/er`, `win_probability`,
  `percent_owned`, `risk_flags`, and a `components` dict of the 6 factors.
  **The data exists; the API just doesn't expose most of it yet.**
- **Existing API contract (`/api/streaming`):** `StreamingResponse {date,
  candidates[StreamCandidate{player, team, opponent, score, status, actionable,
  reason}]}` — a *thin slice* of the board, and **Yahoo-dependent (empty
  locally)**. So this page is **mock-first**, wired live after the CEO widens
  the contract. `PlayerRef` now carries `mlb_id`/`team_abbr`/`team_id`, and
  **streaming is in the populate-now group**, so headshots/logos work as soon as
  the wider contract lands.

## Research summary — streaming factors (informs the algorithm spec)

Full taxonomy + sources captured in the research pass (Connor-requested). The
factors that matter, mapped to HEATER's existing 6 components and a phase:

| Factor (research importance) | Helps | Existing component | Phase |
|---|---|---|---|
| Opponent offense — wRC+/wOBA, vs-hand splits **(#1)** | ERA/WHIP | `matchup` | ✅ have |
| Opponent team K%, vs-hand **(top-3)** | K | `matchup` | ✅ have |
| Pitcher skill — K-BB%, xFIP/SIERA → net SGP | all | `sgp` | ✅ have |
| Opposing-lineup exposure (PvB wOBA) | ERA/WHIP | `lineup` | ✅ have |
| Park + HR park factor + weather | ERA/WHIP | `env` | ✅ have |
| Recent form (L14) | all | `form` | ✅ have |
| Win probability (for W) | W | `winprob` | ✅ have (internal) |
| Two-start week + leash/expected IP | K/W volume | board (`num_starts`) | ✅ have |
| **Vegas implied team total / moneyline (#2 signal)** | ERA/WHIP/W | — | ⏭ **phase-2** |
| **CSW% / SwStr%** (stickiest K-skill) | K | proxy only | ⏭ phase-2 |

**Headline finding:** the engine already covers ~7 of the top factors; the one
expert-rated top signal it lacks is **Vegas implied totals / moneyline** (no
betting-odds data source in HEATER). That's the phase-2 fast-follow, not a
phase-1 blocker.

**Nuances to encode (engine + UI copy):** win-streaming is high-variance (lean
on it, don't over-weight); two-start weeks double K/W *and* ratio risk;
openers/bulk are a W trap; BvP small samples are noise; don't double-count
park/weather (already partly in opponent stats + Vegas).

## Frontend design

### Route, nav, layout
- New route `web/src/app/streaming/page.tsx` → `/streaming`. Add a nav item to
  `TopBar` (and mobile hamburger). Eyebrow "DAILY".
- Layout mirrors the `/optimizer` command-center: header → controls → budget
  strip → hero board → top-pick → breakdowns. Combustion design system
  throughout (navy/heat, Archivo/Inter, `Card`, `heatColor`, `HeroNum`,
  `useCountUp`, `PlayerDialog`/`PlayerAvatar`, `EmptyState`, `PageError/Empty`).

### Stream Board (hero)
- Heat-ranked table of the date's probable starters. Columns: rank, pitcher
  (opens `PlayerDialog`, keyed by `mlbId`), team, opp (`vs`/`@`), **Stream Score
  0–100** (heatColor ramp + `HeroNum`), status/confidence, GS (two-start badge),
  Net SGP, opp wRC+, opp K%, park, xIP/xK/xER, W%, own%, **risk chips**.
- Locked/final rows stay visible but greyed + non-actionable (parity with the
  Streamlit "show locked" behavior). Sorted by score; toggle "show locked".
- Row hover + clickable player cells everywhere (house convention).

### Budget strip
- 3 ops cards (the `/optimizer` ops pattern): **Adds left** `X / 10`, **Weekly
  IP pace** `X / 54`, **Cats in play** (losing/tied categories). Verdict tint
  via `heatColor`/status.

### Top-pick callout
- The #1 actionable stream, surfaced as a hero card (the Players "Top Pickup"
  pattern): pitcher, matchup, score dial, one-line why.

### Analyze Any Starter (the new feature)
- A **combobox/search** (cmdk) listing **every probable starter in MLB** for the
  selected date — filterable by position group (SP / SP-RP / RP) — *not* limited
  to the board, FAs, or the user's roster. Each option shows likelihood
  (`confirmed` / `likely` / `projected`) since probables firm up near game day.
- On select → a **Scorecard panel**: the big `StreamScorecard` viz (score + factor
  bars), the expected line (IP/K/ER/W%), risk flags, and per-factor detail rows
  ("Opponent offense — vs SF: 88 wRC+, 26% K"). This is the "run any pitcher
  through the algorithm" surface.
- The board's per-row **"why"** reuses the `StreamScorecard` viz driven by each
  candidate's `components` (always present on every board row). The labeled
  per-factor `detail` rows (`factors[]`, with strings like "vs SF: 88 wRC+") are
  an **Analyze-mode enrichment** shown below the viz — one viz, two entry points;
  the picker adds the detail layer the board rows don't carry.

### Signature viz — `StreamScorecard`
- Hand-rolled SVG (no chart lib, per house style), two parts:
  1. **Score dial** — the 0–100 Stream Score as a heat gauge (`HeatGauge`-style,
     `heatColor` fill).
  2. **Factor bars** — horizontal **diverging** bars for the 6 components
     (`matchup/env/form/lineup/sgp/winprob`), each −1…+1, heat-colored by sign &
     magnitude, labeled.
- Lives in `web/src/components/viz/StreamScorecard.tsx`; driven by `components`
  (the 6 values present on every candidate), so it renders identically on the
  board "why" and the Analyze panel. `PitcherScorecard` (data type) is distinct
  from `StreamScorecard` (the viz component) — no name collision.

### States
- `usePageData` four-state machine (`loading | error | empty | loaded`) with the
  `?state=` dev override. `PageError`/`PageEmpty` for page-level; in-board empty
  ("no probables posted yet — typically 1–5 days out") via `EmptyState`.

## Data contract (mock shape = the API contract spec)

New module `web/src/lib/streaming-data.ts`. Types (camelCase frontend; the
adapter bridges the snake_case API, as with leaders):

```ts
interface StreamPlayer {            // PlayerRef-aligned
  name: string; pos: string;
  teamAbbr: string; teamId: number; mlbId: number;
}
interface StreamComponents {        // the 6 engine factors, each −1..+1
  matchup: number; env: number; form: number;
  lineup: number; sgp: number; winprob: number;
}
type StreamStatus = "open" | "probable" | "locked" | "final";
type StreamConfidence = "high" | "med" | "low";

interface StreamCandidate {
  rank: number;
  player: StreamPlayer;
  opponent: string; isHome: boolean;
  score: number;                    // 0–100
  status: StreamStatus; confidence: StreamConfidence; actionable: boolean;
  numStarts: number;                // 1 or 2
  netSgp: number;
  oppWrcPlus: number; oppKpct: number; park: number;
  xIp: number; xK: number; xEr: number;
  winPct: number; ownPct: number;
  riskFlags: string[];              // HIGH_WHIP, SHORT_LEASH, ELITE_OFFENSE,
                                    // HITTER_PARK, WIND_OUT, LOW_CONFIDENCE
  components: StreamComponents;
  expectedLine: string;             // "6.0 IP · 7 K · 2 ER"
  why: string;
}
interface FactorDetail {
  key: keyof StreamComponents; label: string;
  value: number; weight: number; detail: string;
}
interface PitcherScorecard extends StreamCandidate { factors: FactorDetail[]; }

interface ProbableStarter {         // the Analyze-Any-Starter picker list
  player: StreamPlayer; team: string; opponent: string; isHome: boolean;
  posGroup: "SP" | "SP/RP" | "RP";
  startLikelihood: "confirmed" | "likely" | "projected";
}
interface BudgetStrip {
  addsLeft: number; addsTotal: number;   // 10
  ipPace: number; ipTarget: number;       // 54
  catsInPlay: string[];
}
interface StreamingData {
  date: string;
  budget: BudgetStrip;
  topPick: StreamCandidate | null;
  board: StreamCandidate[];
  probables: ProbableStarter[];
}
```

`fetchStreaming(date)` returns mock `StreamingData` by default; live path gated
by `NEXT_PUBLIC_HEATER_LIVE`. The Analyze action returns a `PitcherScorecard`;
on mock it's synthesized deterministically per pitcher.

## API contract handoff (for the CEO track)

What the CEO must extend so this page can wire live (gaps vs the current
`/api/streaming`):

1. **Widen `StreamCandidate`** in `api/contracts/streaming.py` to expose the full
   board row the engine already computes: `rank, is_home, confidence,
   num_starts, net_sgp, opp_wrc_plus, opp_k_pct, park, expected_ip/k/er,
   win_pct, own_pct, risk_flags[], components{6}, expected_line`. (No engine
   change — `build_stream_board` returns these.)
2. **Add to `StreamingResponse`:** `top_pick`, `budget {adds_left, adds_total,
   ip_pace, ip_target, cats_in_play[]}`, and `probables[]` (the full pickable
   list for the date).
3. **New endpoint — Analyze Any Starter:** `POST /api/streaming/analyze
   {pitcher_id, date}` → a scorecard (the widened candidate + a `factors[]`
   breakdown). Phase-1 = score via the existing 6-component engine for *any*
   probable pitcher (rostered or not). Phase-2 = add Vegas implied-total /
   moneyline + CSW% factors (new data source).
4. **PlayerRef** already carries `mlb_id`/`team_abbr`/`team_id`; streaming
   populates them — headshots/logos work on wire.

## Scoring algorithm spec (requirement for the CEO)

- **Phase 1 (now):** the existing `stream_analyzer` 0–100 score, generalized to
  score **any probable pitcher** for a date (the engine currently scores FAs /
  the board; the analyze endpoint asks it to score an arbitrary `pitcher_id`).
  Factors = the 6 components above (all data on hand). Return per-factor
  contributions so the UI breakdown is engine-truth, not re-derived.
- **Phase 2 (fast-follow):** add the research's top missing signals — **Vegas
  implied team total + moneyline** (needs a sportsbook-odds data source;
  feasibility/cost is a CEO/infra call) and **CSW%/SwStr%** — as new weighted
  components. Keeps the same contract; new keys appear in `components`/`factors`.
- **Guardrails from research (encode as weighting/UX, not new features):**
  win-prob is high-variance (cap its weight), two-start doubles ratio risk
  (surface it, don't silently reward), openers/bulk flagged, BvP de-weighted to
  stabilization, no park/weather double-count.

## Wiring (slice-0 pattern)

- `fetchStreaming` / Analyze action: mock by default; `NEXT_PUBLIC_HEATER_LIVE=1`
  → `apiGet`/`apiPost` through the Next proxy → adapter (snake→camel) → mock
  fallback on error/empty. Default behavior (flag off) is 100% mock.
- New preview launch config reuses `heater-web-live`.

## Verification

- `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build` green (flag-off path).
- Preview screenshots: the board (loaded), the **Analyze Any Starter** scorecard,
  and the empty + error states (`?state=`). Confirm `/streaming` nav works,
  mobile layout holds, combustion lock unaffected.

## Files touched

- **New:** `web/src/app/streaming/page.tsx`, `web/src/lib/streaming-data.ts`,
  `web/src/components/viz/StreamScorecard.tsx`, plus any small board/scorecard
  subcomponents under `web/src/components/streaming/`.
- **Edited:** `web/src/components/chrome/TopBar.tsx` (nav item); `.claude/`
  (preview config if needed).
- **Untouched:** `api/`, `src/`, all other pages and `web/src/lib/*-data.ts`.

## Index of governing docs

- Migration roadmap: `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md`.
- Contract gaps (M0 input): `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`.
- Slice-0 wiring pattern: `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md`.
- Streamlit parity target: `pages/4_Pitcher_Streaming.py`; engine:
  `src/optimizer/stream_analyzer.py`.
