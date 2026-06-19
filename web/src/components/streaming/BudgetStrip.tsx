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
