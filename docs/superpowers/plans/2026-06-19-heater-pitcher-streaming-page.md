# Pitcher Streaming Page (Stream Finder + Analyze Any Starter) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React `/streaming` page (Stream Finder) plus the "Analyze Any Starter" feature — pick any probable MLB starter for a date and see a factor-by-factor stream scorecard — in `web/`, mock-first.

**Architecture:** A new App-Router page (`web/src/app/streaming/page.tsx`) driven by the `usePageData` four-state machine over a typed mock (`web/src/lib/streaming-data.ts`). Presentational pieces live in `web/src/components/streaming/`; the signature viz (`StreamScorecard`) reuses the existing `HeatGauge` dial + a new diverging `FactorBars` SVG. Player cells reuse the canonical `PlayerRef`/`PlayerDialog`/`PlayerAvatar`. The mock shape **is** the API contract spec for the CEO track — live wiring is deferred (separate slice once `/api/streaming` is widened).

**Tech Stack:** Next.js 16 (App Router, RSC) + React 19 + TypeScript 5.9 + Tailwind v4 + framer-motion. Package manager pnpm. **No unit-test runner in `web/`** — the gate is `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build`, all run **from `web/`**, plus Claude_Preview screenshots for visual tasks and `node web/scripts/audit-mock-ids.mjs` (run from repo root) for any hand-typed `mlbId`. Per the spec `docs/superpowers/specs/2026-06-19-heater-pitcher-streaming-page-design.md`.

**Lane:** `web/` only. Do NOT touch `api/` or `src/`. Branch `feat/web-pitcher-streaming` already exists (the spec is committed there).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `web/src/lib/streaming-data.ts` (create) | Types (the contract) + mock `STREAMING` data + `fetchStreaming()` + `analyzePitcher()` synthesize. |
| `web/src/components/viz/FactorBars.tsx` (create) | New SVG: 6 diverging factor bars (−1…+1, heat-colored). |
| `web/src/components/viz/StreamScorecard.tsx` (create) | Signature viz: composes `HeatGauge` (score) + `FactorBars`. |
| `web/src/components/streaming/BudgetStrip.tsx` (create) | 3 ops cards: adds left / IP pace / cats in play. |
| `web/src/components/streaming/TopPickCallout.tsx` (create) | The #1 actionable stream hero card. |
| `web/src/components/streaming/StreamBoard.tsx` (create) | Heat-ranked board table + per-row "why" (StreamScorecard). |
| `web/src/components/streaming/AnalyzeStarter.tsx` (create) | Probable-pitcher picker + scorecard panel (the new feature). |
| `web/src/app/streaming/page.tsx` (create) | The page: header, date selector, four-state machine, composition. |
| `web/src/components/chrome/TopBar.tsx` (modify) | Add the `/streaming` nav item. |

Note on reuse: the mock's `player` field is the existing `PlayerRef` from `@/lib/types` (`{name,pos,teamAbbr,teamId,mlbId}`), so every player cell plugs into `PlayerDialog`/`PlayerLink`/`PlayerAvatar` with no new types.

---

## Task 1: Data layer — types, mock, fetch, analyze

**Files:**
- Create: `web/src/lib/streaming-data.ts`

- [ ] **Step 1: Create the module with the full contract types + mock + helpers**

```ts
import type { PlayerRef } from "./types";

/**
 * Mock Pitcher Streaming data. The shape IS the API contract spec for the CEO
 * track (widen /api/streaming + add the analyze endpoint). Live wiring deferred.
 * Spec: docs/superpowers/specs/2026-06-19-heater-pitcher-streaming-page-design.md
 */

/** The 6 engine score components, each normalized −1..+1 (matches stream_analyzer). */
export interface StreamComponents {
  matchup: number; // opponent offense (wRC+/K%/splits)
  env: number; // park + weather
  form: number; // pitcher recent form
  lineup: number; // opposing-lineup exposure
  sgp: number; // pitcher skill (xFIP/SIERA -> net SGP)
  winprob: number; // win probability
}

export type StreamStatus = "open" | "probable" | "locked" | "final";
export type StreamConfidence = "high" | "med" | "low";
export type PosGroup = "SP" | "SP/RP" | "RP";

export interface StreamCandidate {
  rank: number;
  player: PlayerRef;
  opponent: string; // opponent team abbr
  isHome: boolean;
  score: number; // 0–100
  status: StreamStatus;
  confidence: StreamConfidence;
  actionable: boolean;
  numStarts: number; // 1 or 2 (two-start week)
  netSgp: number;
  oppWrcPlus: number;
  oppKpct: number;
  park: number; // park factor (1.00 = neutral)
  xIp: number;
  xK: number;
  xEr: number;
  winPct: number; // 0–100
  ownPct: number; // 0–100
  riskFlags: string[]; // HIGH_WHIP, SHORT_LEASH, ELITE_OFFENSE, HITTER_PARK, WIND_OUT, LOW_CONFIDENCE
  components: StreamComponents;
  expectedLine: string; // "6.0 IP · 7 K · 2 ER"
  why: string;
}

export interface FactorDetail {
  key: keyof StreamComponents;
  label: string;
  value: number; // −1..+1
  weight: number; // 0..1
  detail: string; // "vs SF: 88 wRC+ (8th-weakest), 26% K"
}

export interface PitcherScorecard extends StreamCandidate {
  factors: FactorDetail[];
}

export interface ProbableStarter {
  player: PlayerRef;
  team: string;
  opponent: string;
  isHome: boolean;
  posGroup: PosGroup;
  startLikelihood: "confirmed" | "likely" | "projected";
}

export interface BudgetStrip {
  addsLeft: number;
  addsTotal: number; // 10
  ipPace: number;
  ipTarget: number; // 54
  catsInPlay: string[];
}

export interface StreamingData {
  date: string;
  budget: BudgetStrip;
  topPick: StreamCandidate | null;
  board: StreamCandidate[];
  probables: ProbableStarter[];
}

const FACTOR_LABEL: Record<keyof StreamComponents, string> = {
  matchup: "Opponent offense",
  env: "Park & weather",
  form: "Recent form",
  lineup: "Opposing lineup",
  sgp: "Pitcher skill",
  winprob: "Win probability",
};
const FACTOR_WEIGHT: Record<keyof StreamComponents, number> = {
  matchup: 0.28,
  env: 0.14,
  form: 0.12,
  lineup: 0.16,
  sgp: 0.22,
  winprob: 0.08,
};

const pr = (
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
): PlayerRef => ({ name, pos, teamAbbr, teamId, mlbId });

// Board = the date's scored probable starters, ranked by Stream Score.
// NOTE: every mlbId/teamId here MUST pass `node web/scripts/audit-mock-ids.mjs`
// (Step 2). The four marked ✓ are reused from existing audited mocks.
const cand = (
  rank: number,
  player: PlayerRef,
  opponent: string,
  isHome: boolean,
  score: number,
  c: StreamComponents,
  partial: Partial<StreamCandidate>,
): StreamCandidate => ({
  rank,
  player,
  opponent,
  isHome,
  score,
  status: "probable",
  confidence: score >= 70 ? "high" : score >= 50 ? "med" : "low",
  actionable: true,
  numStarts: 1,
  netSgp: 0,
  oppWrcPlus: 100,
  oppKpct: 22,
  park: 1.0,
  xIp: 5.5,
  xK: 6,
  xEr: 2.5,
  winPct: 50,
  ownPct: 0,
  riskFlags: [],
  components: c,
  expectedLine: "",
  why: "",
  ...partial,
});

export const STREAMING: StreamingData = {
  date: "Thu Jun 19",
  budget: { addsLeft: 6, addsTotal: 10, ipPace: 31, ipTarget: 54, catsInPlay: ["K", "ERA", "WHIP"] },
  topPick: null, // set below to board[0]
  board: [
    cand(1, pr("Tarik Skubal", "SP", "DET", 116, 669373), "CWS", true, 86, // ✓
      { matchup: 0.78, env: 0.32, form: 0.55, lineup: 0.6, sgp: 0.82, winprob: 0.4 },
      { numStarts: 1, netSgp: 1.42, oppWrcPlus: 84, oppKpct: 26.5, park: 0.97, xIp: 6.4, xK: 8.1, xEr: 1.9, winPct: 62, ownPct: 99,
        riskFlags: [], expectedLine: "6.4 IP · 8 K · 1.9 ER", why: "Elite arm vs a bottom-5 offense in a pitcher park." }),
    cand(2, pr("Paul Skenes", "SP", "PIT", 134, 694973), "MIA", false, 83, // ✓
      { matchup: 0.7, env: 0.25, form: 0.6, lineup: 0.55, sgp: 0.85, winprob: 0.1 },
      { netSgp: 1.18, oppWrcPlus: 89, oppKpct: 24.1, park: 0.99, xIp: 6.2, xK: 7.6, xEr: 2.0, winPct: 47, ownPct: 98,
        riskFlags: [], expectedLine: "6.2 IP · 8 K · 2.0 ER", why: "Ace ratios; weak-hitting opponent, but low win odds (poor run support)." }),
    cand(3, pr("Garrett Crochet", "SP", "BOS", 111, 676979), "TB", true, 74,
      { matchup: 0.5, env: 0.1, form: 0.45, lineup: 0.4, sgp: 0.75, winprob: 0.45 },
      { netSgp: 0.88, oppWrcPlus: 96, oppKpct: 23.0, park: 1.04, xIp: 6.0, xK: 8.4, xEr: 2.6, winPct: 58,
        ownPct: 94, riskFlags: ["HITTER_PARK"], expectedLine: "6.0 IP · 8 K · 2.6 ER", why: "Top-tier strikeouts; Fenway adds some ratio risk." }),
    cand(4, pr("Cole Ragans", "SP", "KC", 118, 666142), "DET", false, 68,
      { matchup: 0.35, env: 0.2, form: 0.3, lineup: 0.3, sgp: 0.62, winprob: 0.4 },
      { netSgp: 0.64, oppWrcPlus: 101, oppKpct: 21.5, park: 0.98, xIp: 5.9, xK: 7.2, xEr: 2.8, winPct: 55,
        ownPct: 91, expectedLine: "5.9 IP · 7 K · 2.8 ER", why: "Strong K upside; average matchup." }),
    cand(5, pr("Freddy Peralta", "SP", "MIL", 158, 642547), "CHC", true, 63,
      { matchup: 0.2, env: -0.1, form: 0.35, lineup: 0.15, sgp: 0.55, winprob: 0.5 },
      { netSgp: 0.41, oppWrcPlus: 108, oppKpct: 20.8, park: 1.03, xIp: 5.6, xK: 7.0, xEr: 3.1, winPct: 60,
        ownPct: 88, riskFlags: ["ELITE_OFFENSE"], expectedLine: "5.6 IP · 7 K · 3.1 ER", why: "Good win odds; Cubs offense is a real threat." }),
    cand(6, pr("Hunter Greene", "SP", "CIN", 113, 668881), "COL", false, 41,
      { matchup: -0.3, env: -0.7, form: 0.1, lineup: -0.25, sgp: 0.5, winprob: 0.2 },
      { netSgp: -0.22, oppWrcPlus: 118, oppKpct: 19.0, park: 1.28, xIp: 5.2, xK: 6.4, xEr: 4.2, winPct: 44,
        riskFlags: ["HITTER_PARK", "ELITE_OFFENSE"], ownPct: 85, expectedLine: "5.2 IP · 6 K · 4.2 ER", why: "Coors. Strikeouts, but a blow-up risk to ratios." }),
  ],
  probables: [
    // Board pitchers are pickable too, plus rostered/elite arms not on the FA board:
    { player: pr("Tarik Skubal", "SP", "DET", 116, 669373), team: "DET", opponent: "CWS", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Paul Skenes", "SP", "PIT", 134, 694973), team: "PIT", opponent: "MIA", isHome: false, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Garrett Crochet", "SP", "BOS", 111, 676979), team: "BOS", opponent: "TB", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Cole Ragans", "SP", "KC", 118, 666142), team: "KC", opponent: "DET", isHome: false, posGroup: "SP", startLikelihood: "likely" },
    { player: pr("Freddy Peralta", "SP", "MIL", 158, 642547), team: "MIL", opponent: "CHC", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Hunter Greene", "SP", "CIN", 113, 668881), team: "CIN", opponent: "COL", isHome: false, posGroup: "SP", startLikelihood: "likely" },
    { player: pr("Logan Gilbert", "SP", "SEA", 136, 669302), team: "SEA", opponent: "HOU", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("George Kirby", "SP", "SEA", 136, 669923), team: "SEA", opponent: "HOU", isHome: true, posGroup: "SP", startLikelihood: "projected" },
    { player: pr("Zack Wheeler", "SP", "PHI", 143, 554430), team: "PHI", opponent: "NYM", isHome: false, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Spencer Strider", "SP", "ATL", 144, 675911), team: "ATL", opponent: "WSH", isHome: true, posGroup: "SP", startLikelihood: "likely" },
  ],
};
STREAMING.topPick = STREAMING.board[0];

/** Build the per-factor detail rows from a candidate's components. */
export function factorsFor(c: StreamCandidate): FactorDetail[] {
  const keys = Object.keys(c.components) as (keyof StreamComponents)[];
  const detailText: Record<keyof StreamComponents, string> = {
    matchup: `vs ${c.opponent}: ${c.oppWrcPlus} wRC+, ${c.oppKpct.toFixed(0)}% K`,
    env: `Park ${c.park.toFixed(2)}${c.riskFlags.includes("HITTER_PARK") ? " (hitter park)" : ""}`,
    form: "Recent starts vs season baseline",
    lineup: "Regressed opposing-lineup wOBA exposure",
    sgp: `${c.netSgp >= 0 ? "+" : ""}${c.netSgp.toFixed(2)} net SGP (xFIP/SIERA)`,
    winprob: `${c.winPct}% win`,
  };
  return keys.map((k) => ({
    key: k,
    label: FACTOR_LABEL[k],
    value: c.components[k],
    weight: FACTOR_WEIGHT[k],
    detail: detailText[k],
  }));
}

/** Mock-first fetch. Live path (NEXT_PUBLIC_HEATER_LIVE) is a deferred slice. */
export function fetchStreaming(delayMs = 600): Promise<StreamingData> {
  return new Promise((resolve) => setTimeout(() => resolve(STREAMING), delayMs));
}

/** Analyze any probable pitcher → a full scorecard. On mock: derive from the
 *  board if present, else synthesize a neutral-ish card deterministically from
 *  the pitcher's mlbId so repeat picks are stable. (Live: POST /api/streaming/analyze.) */
export function analyzePitcher(p: ProbableStarter): PitcherScorecard {
  const onBoard = STREAMING.board.find((b) => b.player.mlbId === p.player.mlbId);
  if (onBoard) return { ...onBoard, factors: factorsFor(onBoard) };
  // Deterministic synth: hash the mlbId into a 35..80 score + plausible components.
  const h = (p.player.mlbId % 46) + 35; // 35..80
  const f = (seed: number) => Math.round(((((p.player.mlbId >> seed) % 17) - 8) / 8) * 100) / 100; // −1..+1
  const base: StreamCandidate = {
    rank: 0,
    player: p.player,
    opponent: p.opponent,
    isHome: p.isHome,
    score: h,
    status: "probable",
    confidence: h >= 70 ? "high" : h >= 50 ? "med" : "low",
    actionable: true,
    numStarts: 1,
    netSgp: Math.round((h - 55) / 25 * 100) / 100,
    oppWrcPlus: 92 + (p.player.mlbId % 26),
    oppKpct: 19 + (p.player.mlbId % 9),
    park: 0.95 + (p.player.mlbId % 30) / 100,
    xIp: 5.2 + (p.player.mlbId % 12) / 10,
    xK: 5 + (p.player.mlbId % 5),
    xEr: 2.2 + (p.player.mlbId % 18) / 10,
    winPct: 40 + (p.player.mlbId % 30),
    ownPct: p.player.mlbId % 80,
    riskFlags: h < 50 ? ["LOW_CONFIDENCE"] : [],
    components: { matchup: f(1), env: f(2), form: f(3), lineup: f(4), sgp: f(5), winprob: f(6) },
    expectedLine: "",
    why: "Synthesized scorecard (live scorer pending).",
  };
  base.expectedLine = `${base.xIp.toFixed(1)} IP · ${base.xK.toFixed(0)} K · ${base.xEr.toFixed(1)} ER`;
  return { ...base, factors: factorsFor(base) };
}
```

- [ ] **Step 2: Audit the hand-typed player ids, then type-check + lint**

Run (from repo root): `node web/scripts/audit-mock-ids.mjs`
Expected: PASS for every player. If any `mlbId`/`teamId` is flagged, fix it to the value the script reports and re-run until clean.

Run (from `web/`): `pnpm exec tsc --noEmit` → Expected: exit 0.
Run (from `web/`): `pnpm run lint` → Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add web/src/lib/streaming-data.ts
git commit -m "feat(web): streaming page data layer — types, mock, analyze (M1)"
```

---

## Task 2: Signature viz — FactorBars + StreamScorecard

**Files:**
- Create: `web/src/components/viz/FactorBars.tsx`
- Create: `web/src/components/viz/StreamScorecard.tsx`

- [ ] **Step 1: Create `FactorBars.tsx` (6 diverging bars, −1…+1)**

Pattern: mirrors the center-out tug-of-war in `web/src/components/viz/CategoryBattle.tsx` (a bar grows from a center line; positive = heat to the right, negative = steel to the left), but driven by the 6 `StreamComponents` values instead of category win/loss.

```tsx
"use client";

import { motion, useReducedMotion } from "framer-motion";
import { EASE_SNAP } from "@/lib/motion";
import { COLORS } from "@/lib/tokens";
import type { StreamComponents } from "@/lib/streaming-data";

const LABELS: Record<keyof StreamComponents, string> = {
  matchup: "Opp offense",
  env: "Park / wx",
  form: "Form",
  lineup: "Lineup",
  sgp: "Skill",
  winprob: "Win odds",
};
const ORDER: (keyof StreamComponents)[] = ["matchup", "sgp", "lineup", "env", "form", "winprob"];

/** Diverging bars for the 6 stream-score components, each −1..+1.
 *  Positive (helps the stream) grows right in heat; negative grows left in steel. */
export function FactorBars({ components }: { components: StreamComponents }) {
  const reduce = useReducedMotion();
  return (
    <div className="space-y-1.5">
      {ORDER.map((k, i) => {
        const v = Math.max(-1, Math.min(1, components[k]));
        const pos = v >= 0;
        const half = (Math.abs(v) * 50).toFixed(1);
        return (
          <div key={k} className="grid grid-cols-[4.5rem_1fr_2.25rem] items-center gap-2 text-[11px]">
            <span className="truncate font-semibold text-ink-2">{LABELS[k]}</span>
            <div className="relative h-2.5 rounded-full bg-surface-2">
              <span
                className="absolute left-1/2 top-1/2 z-10 h-3.5 w-px -translate-x-1/2 -translate-y-1/2 bg-ink-3/40"
                aria-hidden
              />
              <motion.span
                className="absolute top-0 h-full"
                style={{
                  background: pos ? COLORS.heat : COLORS.steel,
                  ...(pos ? { left: "50%", borderRadius: "0 9999px 9999px 0" } : { right: "50%", borderRadius: "9999px 0 0 9999px" }),
                }}
                initial={reduce ? false : { width: "0%" }}
                animate={{ width: `${half}%` }}
                transition={{ duration: 0.45, ease: EASE_SNAP, delay: 0.08 + i * 0.03 }}
              />
            </div>
            <span className="tnum text-right font-semibold text-ink-3">
              {pos ? "+" : ""}
              {v.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Create `StreamScorecard.tsx` (HeatGauge dial + FactorBars)**

```tsx
"use client";

import { HeatGauge } from "@/components/viz/HeatGauge";
import { FactorBars } from "@/components/viz/FactorBars";
import type { StreamComponents } from "@/lib/streaming-data";

/** The Pitcher Streaming signature instrument: a 0–100 Stream Score dial
 *  (reusing HeatGauge) over the 6 diverging factor bars. Used on the Analyze
 *  panel and the board "why". */
export function StreamScorecard({
  score,
  components,
  size = 168,
}: {
  score: number;
  components: StreamComponents;
  size?: number;
}) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="rounded-2xl bg-navy px-4 pb-3 pt-1">
        <HeatGauge value={score} label="Stream Score" size={size} />
      </div>
      <div className="w-full">
        <FactorBars components={components} />
      </div>
    </div>
  );
}
```

Note: `HeatGauge`'s numeral/label are white-on-navy (built for the navy hero), hence the `bg-navy` wrapper.

- [ ] **Step 3: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit` → Expected: exit 0.
Run (from `web/`): `pnpm run lint` → Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/viz/FactorBars.tsx web/src/components/viz/StreamScorecard.tsx
git commit -m "feat(web): StreamScorecard signature viz (score dial + factor bars)"
```

---

## Task 3: BudgetStrip + TopPickCallout

**Files:**
- Create: `web/src/components/streaming/BudgetStrip.tsx`
- Create: `web/src/components/streaming/TopPickCallout.tsx`

- [ ] **Step 1: Create `BudgetStrip.tsx` (3 ops cards)**

```tsx
import { Repeat, Timer, Crosshair } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import type { BudgetStrip as Budget } from "@/lib/streaming-data";

export function BudgetStrip({ budget }: { budget: Budget }) {
  const ipPct = Math.min(100, Math.round((budget.ipPace / budget.ipTarget) * 100));
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <Card className="p-4">
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-ink-3">
          <Repeat className="size-3.5 text-heat" aria-hidden /> Adds left
        </div>
        <div className="mt-1 text-navy">
          <HeroNum width={74} className="text-3xl">{budget.addsLeft}</HeroNum>
          <span className="ml-1 text-sm font-semibold text-ink-3">/ {budget.addsTotal}</span>
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-ink-3">
          <Timer className="size-3.5 text-heat" aria-hidden /> Weekly IP pace
        </div>
        <div className="mt-1 text-navy">
          <HeroNum width={74} className="text-3xl">{budget.ipPace}</HeroNum>
          <span className="ml-1 text-sm font-semibold text-ink-3">/ {budget.ipTarget} IP</span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
          <span className="block h-full rounded-full bg-heat" style={{ width: `${ipPct}%` }} />
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-ink-3">
          <Crosshair className="size-3.5 text-heat" aria-hidden /> Cats in play
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {budget.catsInPlay.length === 0 ? (
            <span className="text-[13px] text-ink-2">No matchup data</span>
          ) : (
            budget.catsInPlay.map((c) => (
              <span key={c} className="rounded-md bg-heat/10 px-2 py-0.5 text-[12px] font-bold text-heat">
                {c}
              </span>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
```

- [ ] **Step 2: Create `TopPickCallout.tsx` (#1 stream hero)**

```tsx
import { Flame } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { heatColor } from "@/lib/tokens";
import type { StreamCandidate } from "@/lib/streaming-data";

export function TopPickCallout({ pick }: { pick: StreamCandidate }) {
  const col = heatColor(pick.score);
  return (
    <Card className="overflow-hidden p-0">
      <div className="flex flex-wrap items-center gap-4 bg-gradient-to-r from-navy to-[#15294a] p-5 text-white">
        <PlayerAvatar mlbId={pick.player.mlbId} teamId={pick.player.teamId} name={pick.player.name} size={56} ring="ring-white/20" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-flame">
            <Flame className="size-3.5" aria-hidden /> Top stream today
          </div>
          <PlayerDialog player={pick.player}>
            <button className="block truncate text-left font-display text-xl font-extrabold text-white hover:text-flame">
              {pick.player.name}
            </button>
          </PlayerDialog>
          <div className="text-[13px] text-white/70">
            {pick.player.teamAbbr} {pick.isHome ? "vs" : "@"} {pick.opponent} · {pick.expectedLine}
          </div>
          <p className="mt-1 text-[13px] text-white/85">{pick.why}</p>
        </div>
        <div className="text-right leading-none" style={{ color: col }}>
          <HeroNum width={70} className="text-4xl">{pick.score}</HeroNum>
          <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-white/60">Score</div>
        </div>
      </div>
    </Card>
  );
}
```

- [ ] **Step 3: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit` → Expected: exit 0.
Run (from `web/`): `pnpm run lint` → Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/streaming/BudgetStrip.tsx web/src/components/streaming/TopPickCallout.tsx
git commit -m "feat(web): streaming budget strip + top-pick callout"
```

---

## Task 4: StreamBoard (heat-ranked table + per-row why)

**Files:**
- Create: `web/src/components/streaming/StreamBoard.tsx`

- [ ] **Step 1: Create `StreamBoard.tsx`**

Behavior: a table of board candidates. Score cell colored via `heatColor` + `HeroNum`. Player name via `PlayerLink`. Risk flags as small chips. Locked/final rows (`actionable === false`) get `opacity-50`. Each row expands (Radix-free `<details>`/`useState` toggle) to show the `StreamScorecard` viz for that candidate. Columns: Rank, Pitcher, Opp, Score, Conf, GS, Net SGP, Opp wRC+, Opp K%, Park, xK, W%, Own%, Risk.

```tsx
"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerLink } from "@/components/player/PlayerLink";
import { StreamScorecard } from "@/components/viz/StreamScorecard";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import type { StreamCandidate } from "@/lib/streaming-data";

function TH({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th scope="col" className={cn("whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy", left ? "text-left" : "text-right")}>
      {children}
    </th>
  );
}

export function StreamBoard({ board }: { board: StreamCandidate[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px]">
          <thead>
            <tr className="border-b border-line">
              <TH left>#</TH><TH left>Pitcher</TH><TH left>Opp</TH><TH>Score</TH><TH>Conf</TH>
              <TH>GS</TH><TH>Net SGP</TH><TH>wRC+</TH><TH>K%</TH><TH>Park</TH><TH>xK</TH><TH>W%</TH><TH>Own%</TH><TH left>Risk</TH>
            </tr>
          </thead>
          <tbody className="tnum text-[13px]">
            {board.map((c) => {
              const expanded = open === c.rank;
              return (
                <Row key={c.rank} c={c} expanded={expanded} onToggle={() => setOpen(expanded ? null : c.rank)} />
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Row({ c, expanded, onToggle }: { c: StreamCandidate; expanded: boolean; onToggle: () => void }) {
  const col = heatColor(c.score);
  return (
    <>
      <tr className={cn("border-b border-line/60 transition-colors hover:bg-surface", !c.actionable && "opacity-50")}>
        <td className="px-2.5 py-2 text-left font-bold text-ink-3">{c.rank}</td>
        <td className="px-2.5 py-2 text-left">
          <PlayerLink player={c.player} />
          <span className="ml-1 text-[11px] text-ink-3">{c.player.teamAbbr}</span>
        </td>
        <td className="px-2.5 py-2 text-left text-ink-2">{c.isHome ? "vs " : "@ "}{c.opponent}</td>
        <td className="px-2.5 py-2 text-right">
          <HeroNum width={72} className="text-lg" style={{ color: col }}>{c.score}</HeroNum>
        </td>
        <td className="px-2.5 py-2 text-right uppercase text-ink-3">{c.confidence}</td>
        <td className="px-2.5 py-2 text-right">{c.numStarts === 2 ? <span className="rounded bg-flame/15 px-1.5 font-bold text-flame">2-start</span> : "1"}</td>
        <td className={cn("px-2.5 py-2 text-right font-semibold", c.netSgp >= 0 ? "text-ok" : "text-ember")}>{c.netSgp >= 0 ? "+" : ""}{c.netSgp.toFixed(2)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.oppWrcPlus}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.oppKpct.toFixed(1)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.park.toFixed(2)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.xK.toFixed(1)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.winPct}</td>
        <td className="px-2.5 py-2 text-right text-ink-3">{c.ownPct}</td>
        <td className="px-2.5 py-2 text-left">
          <div className="flex items-center gap-1.5">
            {c.riskFlags.map((f) => (
              <span key={f} className="rounded bg-ember/10 px-1.5 py-0.5 text-[10px] font-bold text-ember">{f}</span>
            ))}
            <button onClick={onToggle} aria-label="Toggle why" className="ml-auto rounded p-1 text-ink-3 hover:text-heat">
              <ChevronDown className={cn("size-4 transition-transform", expanded && "rotate-180")} aria-hidden />
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-line/60 bg-surface">
          <td colSpan={14} className="px-4 py-4">
            <div className="flex flex-wrap items-start gap-6">
              <StreamScorecard score={c.score} components={c.components} size={140} />
              <p className="max-w-md flex-1 text-[13px] text-ink-2">{c.why} <span className="block mt-1 text-ink-3">Expected: {c.expectedLine}.</span></p>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
```

- [ ] **Step 2: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit` → Expected: exit 0.
Run (from `web/`): `pnpm run lint` → Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/streaming/StreamBoard.tsx
git commit -m "feat(web): heat-ranked StreamBoard table with per-row why"
```

---

## Task 5: AnalyzeStarter (picker + scorecard)

**Files:**
- Create: `web/src/components/streaming/AnalyzeStarter.tsx`

- [ ] **Step 1: Create `AnalyzeStarter.tsx`**

Behavior: a position-group filter (`SP` / `SP/RP` / `RP` / `All`) + a native `<select>` of the filtered probables; on change, call `analyzePitcher` and render the `StreamScorecard` + the `factors[]` detail rows + the expected line + risk chips. Default selection = the first probable.

```tsx
"use client";

import { useMemo, useState } from "react";
import { Microscope } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { StreamScorecard } from "@/components/viz/StreamScorecard";
import { analyzePitcher, type ProbableStarter, type PosGroup } from "@/lib/streaming-data";

const GROUPS: (PosGroup | "All")[] = ["All", "SP", "SP/RP", "RP"];

export function AnalyzeStarter({ probables }: { probables: ProbableStarter[] }) {
  const [group, setGroup] = useState<PosGroup | "All">("All");
  const list = useMemo(
    () => (group === "All" ? probables : probables.filter((p) => p.posGroup === group)),
    [group, probables],
  );
  const [mlbId, setMlbId] = useState<number>(probables[0]?.player.mlbId ?? 0);
  const selected = list.find((p) => p.player.mlbId === mlbId) ?? list[0];
  const card = selected ? analyzePitcher(selected) : null;

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Microscope className="size-4 text-heat" aria-hidden /> Analyze any starter
      </div>
      <p className="mb-3 text-[12px] text-ink-3">
        Score any probable starter in MLB for this date — rostered or not — through the streaming algorithm.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex gap-1">
          {GROUPS.map((g) => (
            <button
              key={g}
              onClick={() => setGroup(g)}
              className={`rounded-lg px-2.5 py-1 text-[12px] font-bold ${group === g ? "bg-navy text-white" : "bg-surface text-ink-2 hover:bg-surface-2"}`}
            >
              {g}
            </button>
          ))}
        </div>
        <select
          value={selected?.player.mlbId ?? 0}
          onChange={(e) => setMlbId(Number(e.target.value))}
          className="min-h-9 flex-1 rounded-lg border border-line bg-canvas px-3 text-sm font-semibold text-navy"
        >
          {list.map((p) => (
            <option key={p.player.mlbId} value={p.player.mlbId}>
              {p.player.name} ({p.team}) {p.isHome ? "vs" : "@"} {p.opponent} · {p.startLikelihood}
            </option>
          ))}
        </select>
      </div>

      {card && selected && (
        <div className="mt-5 flex flex-wrap items-start gap-6">
          <StreamScorecard score={card.score} components={card.components} size={168} />
          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex items-center gap-3">
              <PlayerAvatar mlbId={selected.player.mlbId} teamId={selected.player.teamId} name={selected.player.name} size={44} />
              <div>
                <div className="font-display text-lg font-bold text-navy">{selected.player.name}</div>
                <div className="text-[12px] text-ink-2">{selected.player.pos} · {card.expectedLine}</div>
              </div>
            </div>
            {card.riskFlags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {card.riskFlags.map((f) => (
                  <span key={f} className="rounded bg-ember/10 px-1.5 py-0.5 text-[11px] font-bold text-ember">{f}</span>
                ))}
              </div>
            )}
            <ul className="space-y-1.5">
              {card.factors.map((fct) => (
                <li key={fct.key} className="flex items-baseline justify-between gap-3 border-b border-line/50 pb-1 text-[12px]">
                  <span className="font-semibold text-navy">{fct.label}</span>
                  <span className="flex-1 truncate px-2 text-ink-3">{fct.detail}</span>
                  <span className={`tnum font-bold ${fct.value >= 0 ? "text-ok" : "text-ember"}`}>{fct.value >= 0 ? "+" : ""}{fct.value.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}
```

- [ ] **Step 2: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit` → Expected: exit 0.
Run (from `web/`): `pnpm run lint` → Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/streaming/AnalyzeStarter.tsx
git commit -m "feat(web): Analyze Any Starter picker + scorecard panel"
```

---

## Task 6: The page (compose + four-state machine)

**Files:**
- Create: `web/src/app/streaming/page.tsx`

- [ ] **Step 1: Create `page.tsx`** (mirrors the structure of `web/src/app/optimizer/page.tsx`)

```tsx
"use client";

import { motion } from "framer-motion";
import { CalendarDays } from "lucide-react";
import { fetchStreaming, type StreamingData } from "@/lib/streaming-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { BudgetStrip } from "@/components/streaming/BudgetStrip";
import { TopPickCallout } from "@/components/streaming/TopPickCallout";
import { StreamBoard } from "@/components/streaming/StreamBoard";
import { AnalyzeStarter } from "@/components/streaming/AnalyzeStarter";

export default function StreamingPage() {
  const { state, retry } = usePageData(fetchStreaming);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty icon={CalendarDays} title="No streamable starts" body="No probable starters are posted for this date yet — probables typically appear 1–5 days out." />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={12} />}
    </>
  );
}

function Loaded({ data }: { data: StreamingData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Daily · {data.date}</div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Pitcher Streaming</h1>
        <p className="mt-1 text-[13px] text-ink-2">Heat-ranked probable starters, scored on matchup, park, form, skill, and win odds.</p>
      </motion.div>
      <motion.div variants={staggerItem}><BudgetStrip budget={data.budget} /></motion.div>
      {data.topPick && <motion.div variants={staggerItem}><TopPickCallout pick={data.topPick} /></motion.div>}
      <motion.div variants={staggerItem}><StreamBoard board={data.board} /></motion.div>
      <motion.div variants={staggerItem}><AnalyzeStarter probables={data.probables} /></motion.div>
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-72" />
      <div className="grid gap-3 sm:grid-cols-3">
        <Skeleton className="h-24 rounded-2xl" /><Skeleton className="h-24 rounded-2xl" /><Skeleton className="h-24 rounded-2xl" />
      </div>
      <Skeleton className="h-80 w-full rounded-2xl" />
    </div>
  );
}
```

- [ ] **Step 2: Build + verify in the browser**

Run (from `web/`): `pnpm build` → Expected: exit 0, route `/streaming` listed.
Then preview (Claude_Preview): `preview_start` `heater-web`; navigate `/streaming`; screenshot the loaded board + the Analyze panel. Navigate `/streaming?state=error` and `/streaming?state=empty`; screenshot each. Confirm the score dial, factor bars, risk chips, and player dialogs render; resize to mobile and confirm the table scrolls.

- [ ] **Step 3: Commit**

```bash
git add web/src/app/streaming/page.tsx
git commit -m "feat(web): /streaming page — Stream Finder + Analyze Any Starter"
```

---

## Task 7: Navigation

**Files:**
- Modify: `web/src/components/chrome/TopBar.tsx:24-31` (the `NAV` array)

- [ ] **Step 1: Add the nav item** (insert after Optimizer)

```tsx
const NAV = [
  { label: "Team", href: "/" },
  { label: "Optimizer", href: "/optimizer" },
  { label: "Streaming", href: "/streaming" },
  { label: "Matchup", href: "/matchup" },
  { label: "Trades", href: "/trades" },
  { label: "Players", href: "/players" },
  { label: "Research", href: "/research" },
];
```

- [ ] **Step 2: Build + verify nav**

Run (from `web/`): `pnpm build` → Expected: exit 0.
Preview: confirm "Streaming" appears in the desktop nav and the mobile hamburger, links to `/streaming`, and shows the active underline when on the page. Screenshot desktop + mobile nav.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/chrome/TopBar.tsx
git commit -m "feat(web): add Streaming to the top nav"
```

---

## Task 8: Final integration verification

- [ ] **Step 1: Full gate**

Run (from `web/`): `pnpm exec tsc --noEmit` → exit 0.
Run (from `web/`): `pnpm run lint` → exit 0.
Run (from `web/`): `pnpm build` → exit 0.
Run (from repo root): `node web/scripts/audit-mock-ids.mjs` → PASS.

- [ ] **Step 2: Visual sweep (Claude_Preview, `heater-web`)**

Screenshot and confirm: `/streaming` loaded (board + budget + top-pick + analyze), `?state=error` (CloudOff card), `?state=empty`, a player dialog opening from a board row, the Analyze panel switching pitchers + position-group filter, and mobile (375px) layout. Confirm no console errors (`preview_console_logs` level error).

- [ ] **Step 3: Merge spec + plan + build to master and push**

```bash
git checkout master
git merge --ff-only feat/web-pitcher-streaming
git branch -d feat/web-pitcher-streaming
git pull --no-rebase --no-edit origin master
git push origin master   # pre-push hook runs ~393 structural-invariant tests
```

---

## Self-Review (completed during planning)

- **Spec coverage:** Stream Board ✓(T4), budget strip ✓(T3), top-pick ✓(T3), Analyze Any Starter ✓(T5), signature viz ✓(T2), four states ✓(T6), nav ✓(T7), mock=contract ✓(T1), deferred tabs/live-wire = out of scope (not built). API handoff + algorithm spec live in the spec doc (CEO lane) — no task here, by design.
- **Placeholder scan:** none — all code shown; the one tool-gated item (`audit-mock-ids`) is a real correctness gate, not a placeholder.
- **Type consistency:** `PlayerRef` (`@/lib/types`) reused everywhere; `StreamComponents`/`StreamCandidate`/`PitcherScorecard`/`ProbableStarter`/`BudgetStrip`/`StreamingData` defined in T1 and consumed unchanged in T2–T6; `heatColor(0..100)`, `HeatGauge {value,label,size}`, `usePageData` shape all match the real APIs read from source.
- **No-test-runner note:** the gate is tsc+lint+build+preview+audit per task, per the standing `web/` rule — there is no Jest/Vitest in `web/`, so no `*.test.ts` steps are written (adding a runner is out of scope).
