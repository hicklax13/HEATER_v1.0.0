"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Check, Bell, RefreshCw, Inbox, type LucideIcon } from "lucide-react";
import { fetchMyTeam } from "@/lib/data";
import type { MyTeamData } from "@/lib/types";
import { EASE_SNAP } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { TopBar } from "@/components/chrome/TopBar";
import { Footer } from "@/components/chrome/Footer";
import { WinHero } from "@/components/myteam/WinHero";
import { Movers } from "@/components/myteam/Movers";
import { LeverCard } from "@/components/myteam/LeverCard";
import { CategoryOutlook } from "@/components/myteam/CategoryOutlook";
import { OpsCards } from "@/components/myteam/OpsCards";
import { SeasonTrajectory } from "@/components/myteam/SeasonTrajectory";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

type State =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
  | { status: "loaded"; data: MyTeamData };

const container = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } };
const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.22, ease: EASE_SNAP } },
};

const CHIP_ICON: Record<string, LucideIcon> = { activity: Activity, check: Check, bell: Bell };
const CHIP_TONE: Record<string, string> = {
  ok: "text-ok",
  bad: "text-ember",
  warn: "text-steel",
  info: "text-ink-2",
};

export default function MyTeamPage() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [leverHover, setLeverHover] = useState(false);

  const load = useCallback(() => {
    setState({ status: "loading" });
    fetchMyTeam()
      .then((d) => setState(d ? { status: "loaded", data: d } : { status: "empty" }))
      .catch(() => setState({ status: "error" }));
  }, []);

  useEffect(() => {
    let alive = true;
    fetchMyTeam()
      .then((d) => {
        if (alive) setState(d ? { status: "loaded", data: d } : { status: "empty" });
      })
      .catch(() => {
        if (alive) setState({ status: "error" });
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="flex min-h-full flex-col">
      <TopBar />
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <ErrorView onRetry={load} />}
        {state.status === "empty" && <EmptyView />}
        {state.status === "loaded" && (
          <Loaded data={state.data} leverHover={leverHover} setLeverHover={setLeverHover} />
        )}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={state.data.freshnessMinutes} />}
    </div>
  );
}

function Loaded({
  data,
  leverHover,
  setLeverHover,
}: {
  data: MyTeamData;
  leverHover: boolean;
  setLeverHover: (v: boolean) => void;
}) {
  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-6">
      <motion.header variants={item} className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="tnum text-[12px] font-medium uppercase tracking-[0.16em] text-navy">
            {data.eyebrow}
          </div>
          <h1 className="mt-1 font-display text-3xl font-extrabold tracking-tight text-navy">
            {data.teamName}
          </h1>
          <div className="tnum mt-1 text-[13px] text-ink-2">{data.subline}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          {data.statusChips.map((c) => {
            const Icon = CHIP_ICON[c.icon] ?? Activity;
            return (
              <span
                key={c.label}
                className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-line bg-canvas px-3 text-[12px] font-medium text-ink-2 shadow-[0_1px_2px_rgba(16,32,55,0.04)]"
              >
                <Icon className={cn("size-3.5", CHIP_TONE[c.tone])} aria-hidden />
                {c.label}
              </span>
            );
          })}
        </div>
      </motion.header>

      <motion.div variants={item}>
        <WinHero matchup={data.matchup} />
      </motion.div>

      <motion.div variants={item}>
        <Movers movers={data.movers} scope={data.moversScope} />
      </motion.div>

      <motion.div variants={item}>
        <LeverCard
          headline={data.lever.headline}
          behindBy={data.lever.behindBy}
          pickups={data.lever.pickups}
          onHoverChange={setLeverHover}
        />
      </motion.div>

      <motion.div variants={item} className="grid gap-6 lg:grid-cols-[1fr_320px] lg:items-start">
        <CategoryOutlook rows={data.categories} pulseLever={leverHover} />
        <OpsCards cards={data.ops} />
      </motion.div>

      <motion.div variants={item}>
        <SeasonTrajectory points={data.trajectory} playoffCut={data.playoffCutRank} />
      </motion.div>
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading your team">
      <div className="space-y-2">
        <Skeleton className="h-3 w-48" />
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-3 w-56" />
      </div>
      <Skeleton className="h-48 w-full rounded-2xl" />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-52 rounded-2xl" />
        ))}
      </div>
      <Skeleton className="h-28 w-full rounded-2xl" />
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <Skeleton className="h-80 rounded-2xl" />
        <div className="space-y-4">
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
        </div>
      </div>
    </div>
  );
}

function ErrorView({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto mt-10 max-w-md p-8 text-center">
      <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-ember/10">
        <RefreshCw className="size-5 text-ember" aria-hidden />
      </div>
      <h2 className="mt-4 font-display text-lg font-bold text-navy">We couldn&apos;t load your team</h2>
      <p className="mt-1 text-sm text-ink-2">
        The data service didn&apos;t respond. Your roster is safe — try again.
      </p>
      <button
        onClick={onRetry}
        className="mt-5 inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-[#ff7a2e] to-heat px-5 text-sm font-semibold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
      >
        <RefreshCw className="size-4" aria-hidden />
        Retry
      </button>
    </Card>
  );
}

function EmptyView() {
  return (
    <Card className="mx-auto mt-10 max-w-md p-8 text-center">
      <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-surface-2">
        <Inbox className="size-5 text-ink-3" aria-hidden />
      </div>
      <h2 className="mt-4 font-display text-lg font-bold text-navy">No team data yet</h2>
      <p className="mt-1 text-sm text-ink-2">
        Connect your Yahoo league and we&apos;ll build your dashboard automatically.
      </p>
    </Card>
  );
}
