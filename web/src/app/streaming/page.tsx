"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CalendarDays } from "lucide-react";
import { fetchStreaming, formatStreamDate, type StreamingData } from "@/lib/streaming-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { BudgetStrip } from "@/components/streaming/BudgetStrip";
import { TopPickCallout } from "@/components/streaming/TopPickCallout";
import { StreamBoard } from "@/components/streaming/StreamBoard";
import { AnalyzeStarter } from "@/components/streaming/AnalyzeStarter";
import { DateStrip, next7Days } from "@/components/streaming/DateStrip";

export default function StreamingPage() {
  const days = useMemo(() => next7Days(), []);
  const [selected, setSelected] = useState(days[0]);
  // Stable per-date fetcher: identity changes only when `selected` changes,
  // so usePageData's effect re-runs exactly on date selection (not every render).
  const fetcher = useMemo(() => () => fetchStreaming(selected), [selected]);
  const { state, retry } = usePageData(fetcher);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        <div className="mb-5 space-y-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
              Daily · {formatStreamDate(selected)}
            </div>
            <h1 className="font-display text-3xl font-extrabold text-navy">Pitcher Streaming</h1>
            <p className="mt-1 text-[13px] text-ink-2">
              Heat-ranked probable starters, scored on matchup, park, form, skill, and win odds — weighted by this
              week&apos;s category needs.
            </p>
          </div>
          <DateStrip days={days} selected={selected} onSelect={setSelected} />
        </div>
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={CalendarDays}
            title="No streamable starts"
            body="No probable starters are posted for this date yet — probables typically appear 1–5 days out."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={12} />}
    </>
  );
}

function Loaded({ data }: { data: StreamingData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <BudgetStrip budget={data.budget} />
      </motion.div>
      {data.topPick && (
        <motion.div variants={staggerItem}>
          <TopPickCallout pick={data.topPick} />
        </motion.div>
      )}
      <motion.div variants={staggerItem}>
        <StreamBoard board={data.board} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <AnalyzeStarter probables={data.probables} date={data.date} />
      </motion.div>
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-72" />
      <div className="grid gap-3 sm:grid-cols-3">
        <Skeleton className="h-24 rounded-2xl" />
        <Skeleton className="h-24 rounded-2xl" />
        <Skeleton className="h-24 rounded-2xl" />
      </div>
      <Skeleton className="h-80 w-full rounded-2xl" />
    </div>
  );
}
