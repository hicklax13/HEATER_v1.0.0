"use client";

import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Wand2, TrendingUp, TrendingDown, Minus, Clock, ArrowRight, Target } from "lucide-react";
import {
  fetchOptimizer,
  type OptimizerData,
  type CatImpact,
  type FaPickup,
  type OptimizerScope,
} from "@/lib/optimizer-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { LineupTable } from "@/components/optimizer/LineupTable";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty, PageLocked, PageNotLinked } from "@/components/ui/PageStates";
import { cn } from "@/lib/utils";

const SCOPES: { id: OptimizerScope; label: string }[] = [
  { id: "today", label: "Today" },
  { id: "rest_of_week", label: "Rest of Week" },
  { id: "rest_of_season", label: "Rest of Season" },
];

export default function OptimizerPage() {
  const [scope, setScope] = useState<OptimizerScope>("today");
  // Stable per-scope fetcher: identity changes with scope → usePageData refetches.
  const fetcher = useCallback(() => fetchOptimizer(scope), [scope]);
  const { state, retry } = usePageData(fetcher);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        <div className="mb-5">
          <ScopeSelector scope={scope} onChange={setScope} />
        </div>
        {state.status === "loading" && <LoadingView />}
        {state.status === "locked" && <PageLocked feature="The Optimizer" />}
        {state.status === "unlinked" && <PageNotLinked />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty icon={Wand2} title="No lineup to optimize" body="We couldn't find your roster for this window." />
        )}
        {state.status === "loaded" && <Loaded data={state.data} scope={scope} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}

function ScopeSelector({ scope, onChange }: { scope: OptimizerScope; onChange: (s: OptimizerScope) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface p-1" role="tablist" aria-label="Optimize horizon">
      {SCOPES.map((s) => (
        <button
          key={s.id}
          role="tab"
          aria-selected={scope === s.id}
          onClick={() => onChange(s.id)}
          className={cn(
            "min-h-9 rounded-lg px-3.5 text-[12px] font-bold transition-colors",
            scope === s.id ? "bg-navy text-white" : "text-ink-2 hover:text-navy",
          )}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

function Loaded({ data, scope }: { data: OptimizerData; scope: OptimizerScope }) {
  const all = [...data.starters, ...data.bench];
  const scopeLabel = SCOPES.find((s) => s.id === scope)?.label ?? "Today";
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header date={data.date} optimal={data.optimal} scopeLabel={scopeLabel} />
      </motion.div>
      <motion.div variants={staggerItem} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <Card className="p-5">
          <SectionHead title="Your Roster" sub={scopeLabel} />
          <LineupTable slots={all} />
        </Card>
        <aside className="space-y-4">
          {data.faSuggestions.length > 0 && <PickupsCard pickups={data.faSuggestions} />}
          {data.daily && <DailyPlanCard daily={data.daily} />}
          {data.ipPace && <PaceCard ipPace={data.ipPace} />}
          {data.impact.length > 0 && <ImpactCard impact={data.impact} />}
        </aside>
      </motion.div>
    </motion.div>
  );
}

function Header({ date, optimal, scopeLabel }: { date: string; optimal: boolean; scopeLabel: string }) {
  const subline = optimal
    ? "Your lineup is already optimal for this window."
    : "Slots flagged with → can improve your projection.";
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Lineup · {scopeLabel} · {date}
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Optimizer</h1>
        <p className="mt-1 text-[13px] text-ink-2">{subline}</p>
      </div>
    </div>
  );
}

function SectionHead({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h2 className="font-display text-base font-bold text-navy">{title}</h2>
      <span className="text-[11px] text-ink-3">{sub}</span>
    </div>
  );
}

function PickupsCard({ pickups }: { pickups: FaPickup[] }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <ArrowRight className="size-4 text-heat" aria-hidden />
        Recommended Pickups
      </div>
      <ul className="space-y-3">
        {pickups.map((p, i) => (
          <li key={i} className="rounded-lg border border-line bg-surface p-2.5">
            <div className="flex items-center justify-between gap-2">
              <PlayerDialog player={p.add}>
                <button type="button" className="flex min-w-0 items-center gap-2 text-left">
                  <PlayerAvatar mlbId={p.add.mlbId} teamId={p.add.teamId} name={p.add.name} size={24} />
                  <span className="min-w-0">
                    <span className="block text-[9px] font-bold uppercase tracking-wide text-ok">Add</span>
                    <span className="block truncate text-[13px] font-semibold text-navy">{p.add.name}</span>
                  </span>
                </button>
              </PlayerDialog>
              {p.netSgpDelta > 0 && (
                <span className="tnum shrink-0 rounded-md bg-heat/12 px-2 py-0.5 text-[11px] font-bold text-heat">
                  +{p.netSgpDelta.toFixed(2)} SGP
                </span>
              )}
            </div>
            <div className="mt-1.5 flex items-center gap-2 pl-1 text-[11px] text-ink-2">
              <span className="font-bold uppercase tracking-wide text-ember">Drop</span>
              <span className="truncate font-semibold text-navy">{p.drop.name}</span>
            </div>
            {p.categoryImpact.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {p.categoryImpact.map((c) => (
                  <span key={c.label} className="tnum rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-ink-2">
                    {c.label} {c.value}
                  </span>
                ))}
              </div>
            )}
            {p.reasoning && <p className="mt-1.5 text-[11px] leading-snug text-ink-3">{p.reasoning}</p>}
          </li>
        ))}
      </ul>
    </Card>
  );
}

const RATE_MODE_TONE: Record<string, string> = {
  protect: "bg-ok/12 text-ok",
  compete: "bg-heat/12 text-heat",
  abandon: "bg-surface-2 text-ink-3",
};

/** Daily-mode matchup context: posture (W/T/L), focus categories (losing, ordered
 *  by urgency), rate-stat stance (ERA/WHIP protect|compete|abandon), and the
 *  engine's plain-language recommendations. Rendered only when the daily meta exists. */
function DailyPlanCard({ daily }: { daily: NonNullable<OptimizerData["daily"]> }) {
  const focus = [...daily.losing].sort((a, b) => (daily.urgency[b] ?? 0) - (daily.urgency[a] ?? 0));
  const rateCats = Object.keys(daily.rateModes);
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Target className="size-4 text-heat" aria-hidden />
        Daily Plan
      </div>
      <div className="mb-3 grid grid-cols-3 gap-1.5 text-center">
        <StatePill n={daily.winning.length} label="Winning" tone="ok" />
        <StatePill n={daily.tied.length} label="Tied" tone="steel" />
        <StatePill n={daily.losing.length} label="Losing" tone="ember" />
      </div>
      {focus.length > 0 && (
        <div className="mb-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wide text-ink-3">Focus</div>
          <div className="flex flex-wrap gap-1.5">
            {focus.map((c) => (
              <span key={c} className="rounded-md bg-heat/12 px-2 py-0.5 text-[11px] font-bold text-heat">
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
      {rateCats.length > 0 && (
        <div className="mb-3">
          <div className="mb-1 text-[10px] font-bold uppercase tracking-wide text-ink-3">Rate stance</div>
          <div className="flex flex-wrap gap-1.5">
            {rateCats.map((c) => {
              const mode = daily.rateModes[c];
              return (
                <span
                  key={c}
                  className={cn(
                    "rounded-md px-2 py-0.5 text-[11px] font-bold capitalize",
                    RATE_MODE_TONE[mode] ?? "bg-surface-2 text-ink-3",
                  )}
                >
                  {c} · {mode}
                </span>
              );
            })}
          </div>
        </div>
      )}
      {daily.recommendations.length > 0 && (
        <ul className="space-y-1.5 border-t border-line pt-3">
          {daily.recommendations.map((r, i) => (
            <li key={i} className="flex gap-1.5 text-[12px] leading-snug text-ink-2">
              <span className="mt-[5px] size-1.5 shrink-0 rounded-full bg-heat" aria-hidden />
              {r}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function StatePill({ n, label, tone }: { n: number; label: string; tone: "ok" | "steel" | "ember" }) {
  const toneCls = tone === "ok" ? "text-ok" : tone === "ember" ? "text-ember" : "text-steel";
  return (
    <div className="rounded-lg bg-surface py-1.5">
      <div className={cn("tnum font-display text-lg font-bold leading-none", toneCls)}>{n}</div>
      <div className="mt-0.5 text-[9px] font-bold uppercase tracking-wide text-ink-3">{label}</div>
    </div>
  );
}

function PaceCard({ ipPace }: { ipPace: { value: number; total: number } }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Clock className="size-4 text-heat" aria-hidden />
        This Week
      </div>
      <Meter
        label="IP pace"
        value={ipPace.value}
        total={ipPace.total}
        unit="IP"
        caption={`On pace toward the ${ipPace.total} IP minimum.`}
      />
    </Card>
  );
}

function Meter({
  label,
  value,
  total,
  unit,
  caption,
  tone = "heat",
}: {
  label: string;
  value: number;
  total: number;
  unit?: string;
  caption?: string;
  tone?: "heat" | "cool";
}) {
  const pct = Math.min(100, Math.round((value / total) * 100));
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] font-medium text-ink-2">{label}</span>
        <span className="tnum text-[13px] font-bold text-navy">
          {value}
          <span className="font-normal text-ink-3"> / {total}{unit ? ` ${unit}` : ""}</span>
          <span className="tnum ml-1.5 text-[11px] font-semibold text-ink-3">{pct}%</span>
        </span>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <span
          className={cn("block h-full rounded-full", tone === "cool" ? "bg-steel" : "bg-heat")}
          style={{ width: `${pct}%` }}
        />
      </div>
      {caption && <div className="mt-1 text-[11px] text-ink-3">{caption}</div>}
    </div>
  );
}

function ImpactCard({ impact }: { impact: CatImpact[] }) {
  return (
    <Card className="p-4">
      <div className="mb-1 text-[12px] font-bold uppercase tracking-wider text-navy">
        Today&apos;s Projected Output
      </div>
      <p className="mb-3 text-[11px] text-ink-3">Expected category totals if you start the optimal lineup.</p>
      <div className="grid grid-cols-3 gap-2">
        {impact.map((c) => (
          <ImpactCell key={c.key} c={c} />
        ))}
      </div>
    </Card>
  );
}

function ImpactCell({ c }: { c: CatImpact }) {
  const Icon = c.trend === "up" ? TrendingUp : c.trend === "down" ? TrendingDown : Minus;
  const col = c.trend === "up" ? "text-ok" : c.trend === "down" ? "text-ember" : "text-ink-3";
  return (
    <div className="rounded-lg bg-surface p-2.5 text-center transition-colors hover:bg-surface-2">
      <HeroNum width={72} className="block text-[21px] leading-none text-navy">
        {c.proj}
      </HeroNum>
      <div
        className={cn(
          "mt-1.5 flex items-center justify-center gap-0.5 text-[10px] font-bold uppercase tracking-wide",
          col,
        )}
      >
        <Icon className="size-3" aria-hidden />
        {c.key}
      </div>
    </div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-72" />
      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <Skeleton className="h-[28rem] w-full rounded-2xl" />
        <Skeleton className="h-[28rem] w-full rounded-2xl" />
      </div>
    </div>
  );
}
