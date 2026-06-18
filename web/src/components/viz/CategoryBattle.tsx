"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Swords } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EASE_SNAP } from "@/lib/motion";
import type { CatCol } from "@/lib/matchup-data";
import { cn } from "@/lib/utils";

/** Parse a stat string ("15", ".284", "0.86") into a number, or null. */
function parseNum(s: string): number | null {
  const v = parseFloat(s.replace(/[^0-9.]/g, ""));
  return Number.isFinite(v) ? v : null;
}

/**
 * Direction-agnostic dominance of a category, 0..1.
 * Uses the ratio of the larger value to the smaller, so it works for both
 * counting cats (R, K) and inverse cats (ERA, WHIP) without sign reasoning —
 * the winner is supplied separately via the `win` field.
 * Floored at 0.2 so even a razor-thin win shows a visible nub; a shutout = full.
 */
function dominance(youStr: string, oppStr: string): number {
  const a = parseNum(youStr);
  const b = parseNum(oppStr);
  if (a === null || b === null) return 0.4;
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  if (hi === 0) return 0.15;
  if (lo === 0) return 1;
  const ratio = hi / lo;
  return Math.max(0.2, Math.min(1, (ratio - 1) / 2 + 0.2));
}

/**
 * Category Battle — the matchup's signature instrument.
 * A momentum strip (categories won vs lost) over a per-category tug-of-war:
 * each scored cat draws a bar out from center toward the winner, length ∝
 * dominance, heat (you) vs steel (opponent). Bars animate center-out on mount.
 */
export function CategoryBattle({
  cats,
  youScore,
  oppScore,
}: {
  cats: CatCol[];
  youScore: number;
  oppScore: number;
}) {
  const reduce = useReducedMotion();
  const scored = cats.filter((c) => c.win === "you" || c.win === "opp");
  const won = scored.filter((c) => c.win === "you").length;
  const lost = scored.filter((c) => c.win === "opp").length;
  const total = won + lost || 1;

  return (
    <Card className="p-5">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 font-display text-base font-bold text-navy">
          <Swords className="size-4 text-heat" aria-hidden />
          Category Battle
        </h2>
        <div className="flex items-center gap-3 text-[12px] font-semibold">
          <span className="flex items-center gap-1.5 text-navy">
            <span className="size-2 rounded-full bg-heat" aria-hidden />
            You <span className="tnum">{youScore}</span>
          </span>
          <span className="flex items-center gap-1.5 text-ink-2">
            <span className="size-2 rounded-full bg-steel" aria-hidden />
            Opp <span className="tnum">{oppScore}</span>
          </span>
        </div>
      </header>

      {/* Momentum strip — won vs lost proportion */}
      <div
        className="mb-5 flex h-2.5 overflow-hidden rounded-full bg-surface-2"
        role="img"
        aria-label={`You lead categories ${won} to ${lost}.`}
      >
        <motion.span
          className="block bg-heat"
          initial={reduce ? false : { width: "0%" }}
          animate={{ width: `${(won / total) * 100}%` }}
          transition={{ duration: 0.6, ease: EASE_SNAP }}
        />
        <motion.span
          className="block bg-steel"
          initial={reduce ? false : { width: "0%" }}
          animate={{ width: `${(lost / total) * 100}%` }}
          transition={{ duration: 0.6, ease: EASE_SNAP }}
        />
      </div>

      {/* Per-category tug-of-war */}
      <div className="space-y-1.5">
        {scored.map((c, i) => {
          const youWin = c.win === "you";
          const half = (dominance(c.you, c.opp) * 50).toFixed(1);
          return (
            <div
              key={c.key}
              className="grid grid-cols-[2.25rem_3rem_1fr_3rem] items-center gap-2 text-[12px]"
            >
              <span className="font-display font-bold text-navy">{c.key}</span>
              <span
                className={cn(
                  "tnum text-right",
                  youWin ? "font-bold text-heat" : "text-ink-3",
                )}
              >
                {c.you}
              </span>

              <div className="relative h-2.5 rounded-full bg-surface-2">
                <span
                  className="absolute left-1/2 top-1/2 z-10 h-3.5 w-px -translate-x-1/2 -translate-y-1/2 bg-ink-3/40"
                  aria-hidden
                />
                <motion.span
                  className={cn(
                    "absolute top-0 h-full",
                    youWin ? "right-1/2 rounded-l-full bg-heat" : "left-1/2 rounded-r-full bg-steel",
                  )}
                  initial={reduce ? false : { width: "0%" }}
                  animate={{ width: `${half}%` }}
                  transition={{ duration: 0.5, ease: EASE_SNAP, delay: 0.1 + i * 0.03 }}
                />
              </div>

              <span
                className={cn(
                  "tnum text-left",
                  !youWin ? "font-bold text-steel" : "text-ink-3",
                )}
              >
                {c.opp}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
