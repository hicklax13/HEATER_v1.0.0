"use client";

import { Fragment, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowUp, ArrowDown, Minus, ChevronDown, Grid2x2 } from "lucide-react";
import type { CategoryRow } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { CategoryRadar } from "@/components/viz/CategoryRadar";
import { Sparkline } from "@/components/ui/Sparkline";
import { COLORS } from "@/lib/tokens";
import { cn } from "@/lib/utils";

const CONTRIB: Record<string, { name: string; val: string }[]> = {
  HR: [{ name: "Aaron Judge", val: "6" }, { name: "Kyle Schwarber", val: "4" }, { name: "Pete Alonso", val: "3" }],
  K: [{ name: "Tarik Skubal", val: "38" }, { name: "Garrett Crochet", val: "31" }, { name: "Hunter Greene", val: "27" }],
  OBP: [{ name: "Aaron Judge", val: ".430" }, { name: "Juan Soto", val: ".388" }, { name: "Bobby Witt Jr.", val: ".352" }],
  ERA: [{ name: "Tarik Skubal", val: "1.50" }, { name: "Bullpen", val: "4.92" }, { name: "Spot starts", val: "6.10" }],
  SV: [{ name: "Primary closer", val: "6" }, { name: "Setup man", val: "3" }],
  SB: [{ name: "Bobby Witt Jr.", val: "4" }, { name: "Everyone else", val: "5" }],
};
const PROJECTED: Record<string, string> = {
  HR: "31", K: "540", OBP: ".337", ERA: "4.05", SV: "16", SB: "15",
};

function winMeta(pct: number) {
  if (pct >= 55) return { cls: "text-ok", bar: COLORS.ok };
  if (pct >= 45) return { cls: "text-steel", bar: COLORS.steel };
  return { cls: "text-ember", bar: COLORS.ember };
}

export function CategoryOutlook({
  rows,
  pulseLever,
}: {
  rows: CategoryRow[];
  pulseLever?: boolean;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const reduce = useReducedMotion();

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 p-4 pb-3">
        <h2 className="flex items-center gap-2 font-display text-base font-bold text-navy">
          <Grid2x2 className="size-4 text-heat" aria-hidden />
          Category Outlook
        </h2>
        <p className="text-[11px] text-ink-3">
          Arrows show whether you&apos;re <span className="font-medium text-ok">ahead ↑</span> or{" "}
          <span className="font-medium text-ember">behind ↓</span> — already adjusted for stats where
          lower is better (ERA).
        </p>
      </div>

      <div className="flex flex-col items-center gap-1 border-b border-line px-4 pb-4 pt-1">
        <CategoryRadar data={rows.map((r) => ({ cat: r.key, you: r.winPct }))} />
        <p className="text-[11px] text-ink-3">Win&nbsp;% by category — dashed ring is 50% break-even</p>
      </div>

      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-line text-[11px] uppercase tracking-wide text-ink-3">
            <th scope="col" className="py-2 pl-4 font-medium">Category</th>
            <th scope="col" className="px-2 py-2 text-right font-medium">You</th>
            <th scope="col" className="px-2 py-2 text-right font-medium">Opp</th>
            <th scope="col" className="px-2 py-2 text-right font-medium">Edge</th>
            <th scope="col" className="px-2 py-2 text-center font-medium">10-wk</th>
            <th scope="col" className="px-2 py-2 text-right font-medium">Win %</th>
            <th scope="col" className="w-9 pr-3" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isOpen = open === r.key;
            const w = winMeta(r.winPct);
            const Glyph = r.edgeDir === "good" ? ArrowUp : r.edgeDir === "bad" ? ArrowDown : Minus;
            const glyphCls =
              r.edgeDir === "good" ? "text-ok" : r.edgeDir === "bad" ? "text-ember" : "text-steel";
            const standing =
              r.edgeDir === "good" ? "ahead" : r.edgeDir === "bad" ? "behind" : "even";
            const highlight = !!r.isLever && pulseLever;
            return (
              <Fragment key={r.key}>
                <tr
                  className={cn(
                    "border-b border-line/70 transition-colors",
                    r.isLever && "bg-heat/[0.04]",
                    highlight && "ring-1 ring-inset ring-heat motion-safe:animate-pulse",
                  )}
                >
                  <th scope="row" className="py-2.5 pl-4">
                    <span className="flex items-center gap-2">
                      <Glyph className={cn("size-4", glyphCls)} aria-label={standing} />
                      <span className="font-display text-sm font-bold text-navy">{r.key}</span>
                      {r.isLever && (
                        <span className="rounded bg-heat/10 px-1.5 py-0.5 text-[10px] font-medium text-heat">
                          lever
                        </span>
                      )}
                    </span>
                  </th>
                  <td className="tnum px-2 py-2.5 text-right text-ink">{r.you}</td>
                  <td className="tnum px-2 py-2.5 text-right text-ink-2">{r.opp}</td>
                  <td
                    className={cn(
                      "tnum px-2 py-2.5 text-right font-semibold",
                      r.edgeDir === "good" ? "text-ok" : r.edgeDir === "bad" ? "text-ember" : "text-steel",
                    )}
                  >
                    {r.edge}
                  </td>
                  <td className="px-2 py-2.5">
                    <div className="flex justify-center">
                      {r.spark && r.spark.length > 0 ? (
                        <Sparkline
                          data={r.spark}
                          color={r.edgeDir === "good" ? COLORS.ok : r.edgeDir === "bad" ? COLORS.ember : COLORS.steel}
                          width={56}
                          height={16}
                        />
                      ) : (
                        <span className="text-[11px] text-ink-3">—</span>
                      )}
                    </div>
                  </td>
                  <td className="px-2 py-2.5">
                    <div className="flex flex-col items-end gap-1">
                      <span className={cn("tnum text-sm font-semibold", w.cls)}>{r.winPct}</span>
                      <span className="h-1 w-12 overflow-hidden rounded-full bg-surface-2">
                        <span
                          className="block h-full rounded-full"
                          style={{ width: `${r.winPct}%`, background: w.bar }}
                        />
                      </span>
                    </div>
                  </td>
                  <td className="pr-3">
                    <button
                      onClick={() => setOpen(isOpen ? null : r.key)}
                      aria-expanded={isOpen}
                      aria-controls={`cat-${r.key}`}
                      aria-label={`${r.key} details`}
                      className="flex size-8 items-center justify-center rounded-lg text-ink-3 hover:bg-surface hover:text-ink"
                    >
                      <ChevronDown
                        className={cn(
                          "size-4 transition-transform duration-[var(--dur-1)] motion-reduce:transition-none",
                          isOpen && "rotate-180",
                        )}
                        aria-hidden
                      />
                    </button>
                  </td>
                </tr>
                <tr>
                  <td colSpan={7} className="p-0">
                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.div
                          id={`cat-${r.key}`}
                          initial={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
                          animate={reduce ? { opacity: 1 } : { height: "auto", opacity: 1 }}
                          exit={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
                          transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
                          className="overflow-hidden"
                        >
                          <div className="grid gap-4 bg-surface/60 px-4 py-3 sm:grid-cols-[1fr_auto]">
                            <div>
                              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-3">
                                Top contributors
                              </div>
                              <ul className="space-y-1">
                                {(CONTRIB[r.key] ?? []).map((c) => (
                                  <li key={c.name} className="flex justify-between text-sm">
                                    <span className="text-ink">{c.name}</span>
                                    <span className="tnum text-ink-2">{c.val}</span>
                                  </li>
                                ))}
                              </ul>
                              {r.isLever && (
                                <p className="mt-2 text-[12px] text-heat">
                                  This is your lever category — see the 3 suggested pickups above.
                                </p>
                              )}
                            </div>
                            <div className="sm:text-right">
                              <div className="text-[11px] font-medium uppercase tracking-wide text-ink-3">
                                Projected final
                              </div>
                              <div className="tnum font-display text-2xl font-bold text-navy">
                                {PROJECTED[r.key] ?? "—"}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </td>
                </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}
