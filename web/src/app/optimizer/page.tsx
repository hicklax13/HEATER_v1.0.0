"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Wand2,
  TrendingUp,
  TrendingDown,
  Minus,
  Check,
  ArrowDown,
  Clock,
  Repeat,
} from "lucide-react";
import { fetchOptimizer, type OptimizerData, type CatImpact } from "@/lib/optimizer-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { LineupTable } from "@/components/optimizer/LineupTable";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { cn } from "@/lib/utils";

/** When optimized, move each swap's `in` player into the `out` player's slot
 *  (now starting) and drop the `out` player to the bench — so clicking Optimize
 *  visibly rebuilds the lineup, not just a banner. */
function applySwaps(
  starters: OptimizerData["starters"],
  bench: OptimizerData["bench"],
  swaps: OptimizerData["swaps"],
  optimized: boolean,
): { starters: OptimizerData["starters"]; bench: OptimizerData["bench"] } {
  if (!optimized || swaps.length === 0) return { starters, bench };
  const s = [...starters];
  const b = [...bench];
  for (const sw of swaps) {
    const outIdx = s.findIndex((x) => x.player.name === sw.out);
    const inIdx = b.findIndex((x) => x.player.name === sw.in);
    if (outIdx === -1 || inIdx === -1) continue;
    const outSlot = s[outIdx];
    const inSlot = b[inIdx];
    s[outIdx] = { ...inSlot, slot: outSlot.slot, status: "start", note: undefined };
    b[inIdx] = { ...outSlot, slot: "BN", status: "bench", note: "Optimized to bench" };
  }
  return { starters: s, bench: b };
}

export default function OptimizerPage() {
  const { state, retry } = usePageData(fetchOptimizer);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Wand2}
            title="No lineup to optimize"
            body="We couldn't find your roster for today."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}

function Loaded({ data }: { data: OptimizerData }) {
  const [optimized, setOptimized] = useState(false);
  const view = applySwaps(data.starters, data.bench, data.swaps, optimized);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header date={data.date} optimized={optimized} onOptimize={() => setOptimized(true)} />
      </motion.div>
      {optimized && (
        <motion.div variants={staggerItem}>
          <SuccessBanner swaps={data.swaps} />
        </motion.div>
      )}
      <motion.div variants={staggerItem} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-6">
          <Card className="p-5">
            <SectionHead title="Starting Lineup" sub="Today" />
            <LineupTable slots={view.starters} />
          </Card>
          <Card className="p-5">
            <SectionHead title="Bench" sub="Available To Swap In" />
            <LineupTable slots={view.bench} />
          </Card>
        </div>
        <aside className="space-y-4">
          <SwapCard swaps={data.swaps} starters={data.starters} bench={data.bench} optimized={optimized} />
          <PaceCard ipPace={data.ipPace} movesLeft={data.movesLeft} />
          <ImpactCard impact={data.impact} />
        </aside>
      </motion.div>
    </motion.div>
  );
}

function Header({
  date,
  optimized,
  onOptimize,
}: {
  date: string;
  optimized: boolean;
  onOptimize: () => void;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Lineup · {date}
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Optimizer</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          {optimized
            ? "Your lineup is optimal for today."
            : "1 change can improve today's projection."}
        </p>
      </div>
      <button
        onClick={onOptimize}
        className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 py-2.5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
      >
        <Wand2 className="size-4" aria-hidden />
        Optimize Lineup
      </button>
    </div>
  );
}

function SuccessBanner({ swaps }: { swaps: OptimizerData["swaps"] }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-ok/30 bg-ok/10 px-4 py-3 text-[13px] font-semibold text-ok">
      <Check className="size-4 shrink-0" aria-hidden />
      Lineup optimized — applied {swaps.length} change{swaps.length === 1 ? "" : "s"}
      {swaps[0] ? ` (${swaps[0].gain})` : ""}.
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

type Slot = OptimizerData["starters"][number];

function SwapCard({
  swaps,
  starters,
  bench,
  optimized,
}: {
  swaps: OptimizerData["swaps"];
  starters: OptimizerData["starters"];
  bench: OptimizerData["bench"];
  optimized: boolean;
}) {
  const byName = new Map<string, Slot>([...starters, ...bench].map((s) => [s.player.name, s]));
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Repeat className="size-4 text-heat" aria-hidden />
        Recommended Change{swaps.length === 1 ? "" : "s"}
      </div>
      {swaps.length === 0 ? (
        <p className="text-[13px] text-ink-2">No changes — your lineup is optimal.</p>
      ) : (
        <ul className="space-y-3">
          {swaps.map((s, i) => (
            <li key={i}>
              <SwapRow tone="out" label="Sit" slot={byName.get(s.out)} fallbackName={s.out} />
              <div className="my-1 flex items-center gap-2 pl-3.5">
                <span className="flex size-5 items-center justify-center rounded-full bg-heat/12 text-heat">
                  <ArrowDown className="size-3.5" aria-hidden />
                </span>
                <span className="tnum rounded-md bg-heat/12 px-2 py-0.5 text-[11px] font-bold text-heat">
                  {s.gain}
                </span>
              </div>
              <SwapRow tone="in" label="Start" slot={byName.get(s.in)} fallbackName={s.in} />
            </li>
          ))}
        </ul>
      )}
      {optimized && swaps.length > 0 && (
        <div className="mt-3 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wide text-ok">
          <Check className="size-3.5" aria-hidden />
          Applied
        </div>
      )}
    </Card>
  );
}

function SwapRow({
  tone,
  label,
  slot,
  fallbackName,
}: {
  tone: "out" | "in";
  label: string;
  slot?: Slot;
  fallbackName: string;
}) {
  const cls = cn(
    "flex w-full items-center gap-2.5 rounded-lg border border-line border-l-[3px] bg-surface px-2.5 py-2 text-left",
    tone === "out" ? "border-l-ember/60" : "border-l-ok/60",
  );
  const labelCls = tone === "out" ? "text-ember" : "text-ok";
  const content = (
    <>
      {slot && (
        <PlayerAvatar
          mlbId={slot.player.mlbId}
          teamId={slot.player.teamId}
          name={slot.player.name}
          size={26}
        />
      )}
      <span className="min-w-0 flex-1">
        <span className={cn("block text-[9px] font-bold uppercase tracking-wide", labelCls)}>{label}</span>
        <span className="block truncate text-[13px] font-semibold text-navy">
          {slot?.player.name ?? fallbackName}
        </span>
      </span>
      {slot && (
        <span className="shrink-0 text-right">
          <span className="tnum block font-display text-base font-bold text-navy">{slot.value}</span>
          <span className="block text-[10px] text-ink-3">{slot.matchup}</span>
        </span>
      )}
    </>
  );
  if (!slot) return <div className={cls}>{content}</div>;
  return (
    <PlayerDialog player={slot.player}>
      <button
        className={cn(
          cls,
          "transition-colors hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50",
        )}
      >
        {content}
      </button>
    </PlayerDialog>
  );
}

function PaceCard({
  ipPace,
  movesLeft,
}: {
  ipPace: { value: number; total: number };
  movesLeft: { value: number; total: number };
}) {
  const movesUsed = movesLeft.total - movesLeft.value;
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
        caption={`On pace to clear the ${ipPace.total} IP minimum.`}
      />
      <div className="mt-3.5">
        <Meter
          label="Moves left"
          value={movesLeft.value}
          total={movesLeft.total}
          tone="cool"
          caption={`${movesUsed} used · comfortable burn rate.`}
        />
      </div>
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
