# HEATER Web 10/10 Uplift — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the HEATER `web/` app from ~6.5/10 to a flagship 10/10 via shared advanced systems (motion, data-viz, depth, a signature gauge, typography) applied across all six pages, plus a polish pass.

**Architecture:** Systems-first. Build reusable primitives in `web/src/lib` + `web/src/components/{ui,viz,motion}`, then compose them into pages. Per-page `*-data.ts` contracts are unchanged. Evolve the navy/orange Combustion palette with real elevation; sleek by default, cinematic on the signature gauge.

**Tech Stack:** Next.js 16 (App Router), React 19, TypeScript, Tailwind v4 (`@theme` tokens), framer-motion, recharts (already installed), lucide-react.

**Verification (no web test runner configured):** every task ends with `pnpm -C web exec tsc --noEmit` (PASS), `pnpm -C web run lint` (clean), and a real **preview screenshot** confirming the change. That trio is the "test" for this plan.

---

## File map

- `web/src/app/globals.css` — elevation tokens, width-axis numeral utility, refined surfaces.
- `web/src/lib/tokens.ts` — elevation/echo helpers if needed.
- `web/src/lib/motion.ts` — `useCountUp`, reveal variants, spring presets (extend existing).
- `web/src/components/ui/Card.tsx` — `tone` variants (flat/raised/inset).
- `web/src/components/ui/HeroNum.tsx` — engineered width-axis numeral (new).
- `web/src/components/ui/Reveal.tsx` — staggered-reveal wrapper (new).
- `web/src/components/viz/HeatGauge.tsx` — signature win-prob gauge (new).
- `web/src/components/viz/*` — chart kit (Phase C).
- `web/src/components/ui/{AsyncRegion,EmptyState,ErrorState,DataTable}.tsx` — polish (Phase D).

---

## PHASE A — Foundations

### Task A1: Elevation + material tokens, Card tones

**Files:**
- Modify: `web/src/app/globals.css` (`@theme` block + a small `@layer utilities`)
- Modify: `web/src/components/ui/Card.tsx`

- [ ] **Step 1: Add elevation tokens + inner-highlight utility to globals.css**

In the `@theme` block, add:

```css
  /* elevation (light surfaces) */
  --shadow-elev-1: 0 1px 2px rgba(16,32,55,0.05), 0 1px 1px rgba(16,32,55,0.04);
  --shadow-elev-2: 0 1px 2px rgba(16,32,55,0.05), 0 10px 30px rgba(16,32,55,0.07);
  --shadow-elev-3: 0 2px 4px rgba(16,32,55,0.06), 0 24px 60px rgba(16,32,55,0.13);
```

After the `@utility tnum { … }` block, add:

```css
/* crisp top inner-highlight for raised surfaces */
@utility edge-hi {
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}
```

- [ ] **Step 2: Give Card tone variants**

Replace the `Card` body so it accepts `tone`:

```tsx
export function Card({
  tone = "raised",
  className,
  children,
}: {
  tone?: "flat" | "raised" | "inset";
  className?: string;
  children: React.ReactNode;
}) {
  const tones = {
    flat: "bg-canvas border border-line",
    raised: "bg-canvas border border-line shadow-[var(--shadow-elev-2)] edge-hi",
    inset: "bg-surface border border-line",
  } as const;
  return (
    <div className={cn("relative overflow-hidden rounded-2xl", tones[tone], className)}>
      {/* machined top edge */}
      <span aria-hidden className="pointer-events-none absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-white/70 to-transparent" />
      {children}
    </div>
  );
}
```

(Keep the existing import of `cn`. If Card currently hardcodes a shadow class, this replaces it — `raised` is the default so existing call sites are unchanged visually, just richer.)

- [ ] **Step 3: Verify** — `pnpm -C web exec tsc --noEmit` PASS; `pnpm -C web run lint` clean; preview `/` shows cards with slightly deeper, layered elevation + a hairline top edge.

- [ ] **Step 4: Commit**

```bash
git -C web/.. add web/src/app/globals.css web/src/components/ui/Card.tsx
git commit -m "feat(web): elevation tokens + Card tone variants (depth system)"
```

### Task A2: Width-axis hero numerals (`HeroNum`)

**Files:**
- Create: `web/src/components/ui/HeroNum.tsx`

- [ ] **Step 1: Create HeroNum**

```tsx
/** Engineered hero numeral — exploits Archivo's width (wdth) axis + tabular figures. */
export function HeroNum({
  children,
  width = 78,
  className,
  style,
}: {
  children: React.ReactNode;
  width?: number; // Archivo wdth axis 62..125
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <span
      className={className}
      style={{
        fontFamily: "var(--font-display), system-ui, sans-serif",
        fontVariationSettings: `"wdth" ${width}`,
        fontVariantNumeric: "tabular-nums",
        fontWeight: 800,
        letterSpacing: "-0.01em",
        ...style,
      }}
    >
      {children}
    </span>
  );
}
```

- [ ] **Step 2: Verify** — tsc PASS, lint clean. (Applied in B1.)

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(web): HeroNum — Archivo width-axis engineered numeral"
```

### Task A3: Motion primitives (`useCountUp`, Reveal)

**Files:**
- Modify: `web/src/lib/motion.ts`
- Create: `web/src/components/ui/Reveal.tsx`

- [ ] **Step 1: Extend `lib/motion.ts`**

Add (keep existing `SPRING`/`EASE_SNAP` exports):

```ts
import { animate, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

/** Count a number up to `value` on mount; static under reduced-motion. */
export function useCountUp(value: number, duration = 0.9): number {
  const reduce = useReducedMotion();
  const [n, setN] = useState(0);
  useEffect(() => {
    if (reduce) return;
    const controls = animate(0, value, {
      duration,
      ease: EASE_SNAP,
      onUpdate: (v) => setN(Math.round(v)),
    });
    return () => controls.stop();
  }, [value, duration, reduce]);
  return reduce ? value : n;
}

/** Staggered reveal variants (container + item). */
export const revealContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
};
export const revealItem = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.32, ease: EASE_SNAP } },
};
```

> Note: `useCountUp` returns a *derived* value under reduced-motion (no synchronous `setState` in the effect) — this is required to pass the `react-hooks/set-state-in-effect` lint rule that bit us before.

- [ ] **Step 2: Create `Reveal.tsx`**

```tsx
"use client";
import { motion } from "framer-motion";
import { revealContainer, revealItem } from "@/lib/motion";

/** Wrap a list of sections; children reveal in a stagger on mount. */
export function Reveal({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div variants={revealContainer} initial="hidden" animate="show" className={className}>
      {children}
    </motion.div>
  );
}
export function RevealItem({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div variants={revealItem} className={className}>
      {children}
    </motion.div>
  );
}
```

- [ ] **Step 3: Apply to the Team page** — in `web/src/app/page.tsx`, wrap the loaded `Loaded` content sections in `<Reveal>` / `<RevealItem>` (replace the existing local `container`/`item` variants with the shared ones).

- [ ] **Step 4: Verify** — tsc PASS, lint clean; preview `/` shows sections staggering in on load; reduced-motion off = instant.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(web): motion primitives — useCountUp + Reveal stagger"
```

---

## PHASE B — Signature HeatGauge ⭐

### Task B1: Build `HeatGauge`

**Files:**
- Create: `web/src/components/viz/HeatGauge.tsx`

- [ ] **Step 1: Create HeatGauge** (generalize the WinHero arc; add cool↔warm color, count-up, glow, mount sweep, reduced-motion)

```tsx
"use client";
import { useId } from "react";
import { HeroNum } from "@/components/ui/HeroNum";
import { useCountUp } from "@/lib/motion";
import { heatColor } from "@/lib/tokens";

const START = 130;
const SWEEP = 280;
function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}
function arc(cx: number, cy: number, r: number, a0: number, a1: number): string {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  const large = a1 - a0 <= 180 ? 0 : 1;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/** Signature win-probability instrument: the arc + numeral color-shift cool→warm with the value. */
export function HeatGauge({
  value,
  label = "Win Probability",
  size = 200,
}: {
  value: number; // 0..100
  label?: string;
  size?: number;
}) {
  const disp = useCountUp(value);
  const col = heatColor(disp);
  const id = useId();
  const cx = 100;
  const cy = 100;
  const r = 80;
  const valAngle = START + (disp / 100) * SWEEP;
  const h = size * 0.875;
  return (
    <div className="relative" style={{ width: size, height: h }}>
      <svg width={size} height={h} viewBox="0 0 200 175" aria-hidden focusable="false">
        <defs>
          <linearGradient id={`g-${id}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={col} stopOpacity="0.65" />
            <stop offset="1" stopColor={col} />
          </linearGradient>
        </defs>
        <path d={arc(cx, cy, r, START, START + SWEEP)} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="12" strokeLinecap="round" />
        <path
          d={arc(cx, cy, r, START, Math.max(START + 0.01, valAngle))}
          fill="none"
          stroke={`url(#g-${id})`}
          strokeWidth="12"
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 12px ${col}99)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-1.5">
        <div className="leading-none" style={{ color: col, textShadow: `0 0 28px ${col}66` }}>
          <HeroNum width={70} style={{ fontSize: size * 0.32 }}>
            {disp}
          </HeroNum>
          <span className="align-top" style={{ fontSize: size * 0.14, fontWeight: 800 }}>
            %
          </span>
        </div>
        <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/75">{label}</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify** — tsc PASS, lint clean.

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(web): signature HeatGauge (cool→warm win-prob instrument)"
```

### Task B2: Use HeatGauge in the Team hero

**Files:**
- Modify: `web/src/components/myteam/WinHero.tsx`

- [ ] **Step 1:** Replace the inline gauge SVG + numeral block in `WinHero` with `<HeatGauge value={matchup.winPct} />`. Remove the now-dead `polar`/`arc`/`START`/`SWEEP`/`useCountUp`-local code and `heatColor` import if unused. Keep the delta line + SplitBar.
- [ ] **Step 2: Verify** — tsc PASS, lint clean; preview `/` shows the gauge reading "46" in a cool blue, count-up on load, glow.
- [ ] **Step 3: Commit** — `git commit -am "feat(web): Team hero uses the shared HeatGauge"`

### Task B3: Use HeatGauge in the Matchup score header

**Files:**
- Modify: `web/src/app/matchup/page.tsx`

- [ ] **Step 1:** In `ScoreHeader`, replace the plain `data.you.score vs data.opp.score` numerals with a compact `<HeatGauge value={data.winPct} size={150} label="Win Probability" />` centered between the two team heads (keep the team heads + day tabs). Add `winPct` to `MatchupData` (already present? if not, add `winPct: 46`).
- [ ] **Step 2: Verify** — tsc PASS, lint clean; preview `/matchup` shows the same cinematic gauge as Team, unifying the two heroes.
- [ ] **Step 3: Commit** — `git commit -am "feat(web): Matchup hero uses the shared HeatGauge"`

---

## PHASE C — Data-viz kit (outline; parallelizable per chart, then per page)

Each chart: `web/src/components/viz/<Name>.tsx`, Combustion-themed, consumes existing data shapes; verify tsc+lint+preview; commit.

- [ ] **C1 `TrendSpark`** — richer sparkline (gradient area, end dot, hi/lo). Replace `Sparkline` usages.
- [ ] **C2 `WinProbTrend`** — recharts area of `trajectory`/win-prob over weeks → Team hero + Season Trajectory.
- [ ] **C3 `CategoryRadar`** — recharts radar, you vs opp, 12 cats → Team Category Outlook + Matchup totals.
- [ ] **C4 `PercentileBar` / `HeatBar`** — distribution band + category heat → Research (percentile column) + Matchup (heat on totals) + Players (value context).
- [ ] **C5 Apply per page** (parallel subagents): Team, Matchup, Research, Trades each get the relevant chart(s).

## PHASE D — Polish pass (outline; parallelizable per area/page)

- [ ] **D1 `AsyncRegion` + `EmptyState` + `ErrorState`** — standardize loading/empty/error/loaded; adopt on all 6 pages.
- [ ] **D2 Micro-interaction & focus** — shared hover/press spring on cards/rows/chips; consistent `:focus-visible` rings; command-palette polish (recent + grouping).
- [ ] **D3 Spacing/rhythm + `DataTable`** — shared table head/cell/row primitives; one vertical-rhythm scale; adopt across table pages (Optimizer, Matchup, Players, Research, Trades).
- [ ] **D4 Player popup real data** — wire real `mlbId`s into mock data so headshots load; enrich fallback; keep FA vs rostered.
- [ ] **D5 Mobile + cohesion** — responsive breakpoints, scrollable wide tables, 44px targets; unify light/dark page chrome.

---

## Self-review

- **Spec coverage:** S1 motion → A3; S2 viz → C; S3 depth → A1; S4 HeatGauge → B; S5 type → A2/HeroNum; polish P1–P5 → D1–D5. All covered.
- **Type consistency:** `HeatGauge({value,label,size})`, `useCountUp(value,duration)`, `Reveal`/`RevealItem`, `Card tone`, `HeroNum width` — names consistent across tasks.
- **Verification:** tsc + lint + preview on every task (no web test runner; stated upfront).
