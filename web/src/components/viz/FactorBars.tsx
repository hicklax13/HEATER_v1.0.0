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
