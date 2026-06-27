"use client";

import { Wand2, ArrowUpCircle, ArrowDownCircle } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

const CONF_TONE: Record<string, string> = {
  Clear: "bg-ok/12 text-ok",
  Lean: "bg-heat/12 text-heat",
  "Toss-up": "bg-surface-2 text-ink-2",
};

/** Open-slot summary as "OF×2 · Util×1 · …" (only positions with > 0 open). */
function openSlotSummary(openSlots: Record<string, number>): string {
  const parts = Object.entries(openSlots)
    .filter(([, n]) => n > 0)
    .map(([slot, n]) => `${slot}×${n}`);
  return parts.join(" · ");
}

export function VerdictPanel({
  openSlots,
  confidenceLabel,
  reasoning,
  startNames,
  sitNames,
  onApply,
  applying,
  canApply,
}: {
  openSlots: Record<string, number>;
  confidenceLabel: string;
  reasoning: string;
  startNames: string[];
  sitNames: string[];
  onApply: () => void;
  applying: boolean;
  canApply: boolean;
}) {
  const slots = openSlotSummary(openSlots);
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
          <Wand2 className="size-4 text-heat" aria-hidden />
          Verdict
        </div>
        <span
          className={cn(
            "rounded-md px-2 py-0.5 text-[11px] font-bold",
            CONF_TONE[confidenceLabel] ?? "bg-surface-2 text-ink-2",
          )}
        >
          {confidenceLabel}
        </span>
      </div>

      {slots && (
        <div className="mb-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wide text-ink-3">Open slots</div>
          <div className="tnum text-[13px] font-semibold text-navy">{slots}</div>
        </div>
      )}

      {startNames.length > 0 && (
        <div className="mb-2.5">
          <div className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide text-ok">
            <ArrowUpCircle className="size-3.5" aria-hidden />
            Start
          </div>
          <ul className="space-y-0.5">
            {startNames.map((n) => (
              <li key={n} className="text-[13px] font-semibold text-navy">
                {n}
              </li>
            ))}
          </ul>
        </div>
      )}

      {sitNames.length > 0 && (
        <div className="mb-3">
          <div className="mb-1 flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide text-ember">
            <ArrowDownCircle className="size-3.5" aria-hidden />
            Sit
          </div>
          <ul className="space-y-0.5">
            {sitNames.map((n) => (
              <li key={n} className="text-[13px] font-medium text-ink-2">
                {n}
              </li>
            ))}
          </ul>
        </div>
      )}

      {reasoning && <p className="mb-3 border-t border-line pt-3 text-[12px] leading-snug text-ink-2">{reasoning}</p>}

      <button
        onClick={onApply}
        disabled={!canApply || applying}
        className="inline-flex w-full min-h-10 items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] enabled:hover:scale-[1.02] enabled:active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transform-none"
      >
        <Wand2 className="size-4" aria-hidden />
        {applying ? "Applying…" : "Apply to open slots"}
      </button>
    </Card>
  );
}
