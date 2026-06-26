"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Check, Bell, Inbox, Trophy, type LucideIcon } from "lucide-react";
import { fetchMyTeam, fetchYourPlayoffOdds } from "@/lib/data";
import type { MyTeamData } from "@/lib/types";
import { EASE_SNAP } from "@/lib/motion";
import { heatColor } from "@/lib/tokens";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty, PageNotLinked } from "@/components/ui/PageStates";
import { cn } from "@/lib/utils";
import { Footer } from "@/components/chrome/Footer";
import { WinHero } from "@/components/myteam/WinHero";
import { Movers } from "@/components/myteam/Movers";
import { LeverCard } from "@/components/myteam/LeverCard";
import { CategoryOutlook } from "@/components/myteam/CategoryOutlook";
import { OpsCards } from "@/components/myteam/OpsCards";
import { SeasonTrajectory } from "@/components/myteam/SeasonTrajectory";
import { WinProbTrend } from "@/components/viz/WinProbTrend";
import { Skeleton } from "@/components/ui/Skeleton";

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
  const { state, retry } = usePageData(fetchMyTeam);
  const [leverHover, setLeverHover] = useState(false);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "unlinked" && <PageNotLinked />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Inbox}
            title="No team data yet"
            body="Connect your Yahoo league and we'll build your dashboard automatically."
          />
        )}
        {state.status === "loaded" && (
          <Loaded data={state.data} leverHover={leverHover} setLeverHover={setLeverHover} />
        )}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={state.data.freshnessMinutes} />}
    </>
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
          <PlayoffOddsChip fallback={data.playoffOdds} />
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

      {/* Lever is null when you're winning every category (API) → hide the card. */}
      {data.lever && (
        <motion.div variants={item}>
          <LeverCard
            categoryKey={data.lever.categoryKey}
            headline={data.lever.headline}
            behindBy={data.lever.behindBy}
            pickups={data.lever.pickups}
            onHoverChange={setLeverHover}
          />
        </motion.div>
      )}

      <motion.div variants={item} className="grid gap-6 lg:grid-cols-[1fr_320px] lg:items-start">
        <CategoryOutlook rows={data.categories} pulseLever={leverHover} />
        <OpsCards cards={data.ops} />
      </motion.div>

      {/* Win-Prob Trend + Season Trajectory are per-week HISTORY (deferred — need a
          snapshot table). Empty in live mode → hidden; they reappear when backed. */}
      {data.winProbTrend.length > 0 && (
        <motion.div variants={item}>
          <WinProbTrend data={data.winProbTrend} />
        </motion.div>
      )}

      {data.trajectory.length > 0 && (
        <motion.div variants={item}>
          <SeasonTrajectory points={data.trajectory} playoffCut={data.playoffCutRank} />
        </motion.div>
      )}
    </motion.div>
  );
}

/** Forward playoff-odds chip — self-fetches so the ~2.4s sim doesn't block the
 *  dashboard. Hidden until odds resolve (live) or a mock fallback is supplied. */
function PlayoffOddsChip({ fallback }: { fallback?: number }) {
  const [odds, setOdds] = useState<number | undefined>(fallback);
  useEffect(() => {
    let alive = true;
    fetchYourPlayoffOdds().then((v) => {
      if (alive && v !== undefined) setOdds(v);
    });
    return () => {
      alive = false;
    };
  }, []);
  if (odds === undefined) return null;
  return (
    <span className="inline-flex min-h-9 items-center gap-1.5 rounded-lg border border-heat/30 bg-heat/[0.06] px-3 text-[12px] font-semibold text-navy shadow-[0_1px_2px_rgba(16,32,55,0.04)]">
      <Trophy className="size-3.5 text-heat" aria-hidden />
      <span className="tnum font-bold" style={{ color: heatColor(odds) }}>
        {odds}%
      </span>
      Playoff Odds
    </span>
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

