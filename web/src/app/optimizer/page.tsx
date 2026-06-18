"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Wand2,
  TrendingUp,
  TrendingDown,
  Minus,
  Check,
  ArrowRight,
  Clock,
  Repeat,
} from "lucide-react";
import { fetchOptimizer, type OptimizerData, type CatImpact } from "@/lib/optimizer-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { LineupTable } from "@/components/optimizer/LineupTable";
import { cn } from "@/lib/utils";

export default function OptimizerPage() {
  const [data, setData] = useState<OptimizerData | null>(null);
  const [optimized, setOptimized] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchOptimizer().then((d) => {
      if (alive) setData(d);
    });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {!data ? (
          <LoadingView />
        ) : (
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
                  <LineupTable slots={data.starters} />
                </Card>
                <Card className="p-5">
                  <SectionHead title="Bench" sub="Available To Swap In" />
                  <LineupTable slots={data.bench} />
                </Card>
              </div>
              <aside className="space-y-4">
                <SwapCard swaps={data.swaps} optimized={optimized} />
                <PaceCard ipPace={data.ipPace} movesLeft={data.movesLeft} />
                <ImpactCard impact={data.impact} />
              </aside>
            </motion.div>
          </motion.div>
        )}
      </main>
      <Footer freshnessMinutes={9} />
    </>
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
        className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-[#ff7a2e] to-heat px-5 py-2.5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
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

function SwapCard({ swaps, optimized }: { swaps: OptimizerData["swaps"]; optimized: boolean }) {
  return (
    <Card className="p-4">
      <div className="mb-2 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Repeat className="size-4 text-heat" aria-hidden />
        Recommended Change{swaps.length === 1 ? "" : "s"}
      </div>
      {swaps.length === 0 ? (
        <p className="text-[13px] text-ink-2">No changes — your lineup is optimal.</p>
      ) : (
        <ul className="space-y-2">
          {swaps.map((s, i) => (
            <li key={i} className="rounded-lg bg-surface p-2.5 text-[12.5px]">
              <div className="flex flex-wrap items-center gap-1.5 font-semibold">
                <span className="text-ember">Sit {s.out}</span>
                <ArrowRight className="size-3.5 text-ink-3" aria-hidden />
                <span className="text-ok">Start {s.in}</span>
              </div>
              <div className="tnum mt-0.5 text-[11px] text-ink-2">{s.gain}</div>
            </li>
          ))}
        </ul>
      )}
      {optimized && swaps.length > 0 && (
        <div className="mt-2 text-[11px] font-bold uppercase tracking-wide text-ok">Applied</div>
      )}
    </Card>
  );
}

function PaceCard({
  ipPace,
  movesLeft,
}: {
  ipPace: { value: number; total: number };
  movesLeft: { value: number; total: number };
}) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Clock className="size-4 text-heat" aria-hidden />
        This Week
      </div>
      <Meter label="IP pace" value={ipPace.value} total={ipPace.total} unit="IP" />
      <div className="mt-3">
        <Meter label="Moves left" value={movesLeft.value} total={movesLeft.total} />
      </div>
    </Card>
  );
}

function Meter({
  label,
  value,
  total,
  unit,
}: {
  label: string;
  value: number;
  total: number;
  unit?: string;
}) {
  const pct = Math.min(100, Math.round((value / total) * 100));
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] font-medium text-ink-2">{label}</span>
        <span className="tnum text-[13px] font-bold text-navy">
          {value}
          <span className="font-normal text-ink-3"> / {total}{unit ? ` ${unit}` : ""}</span>
        </span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <span className="block h-full rounded-full bg-heat" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ImpactCard({ impact }: { impact: CatImpact[] }) {
  return (
    <Card className="p-4">
      <div className="mb-3 text-[12px] font-bold uppercase tracking-wider text-navy">
        Today&apos;s Projected Output
      </div>
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
    <div className="rounded-lg bg-surface p-2 text-center">
      <div className="tnum font-display text-lg font-bold text-navy">{c.proj}</div>
      <div
        className={cn(
          "flex items-center justify-center gap-0.5 text-[10px] font-bold uppercase tracking-wide",
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
