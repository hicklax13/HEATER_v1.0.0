"use client";

import { motion } from "framer-motion";
import { Trophy } from "lucide-react";
import { fetchStandings, type StandingsData } from "@/lib/standings-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { StandingsTable } from "@/components/standings/StandingsTable";
import { PlayoffOddsPanel } from "@/components/standings/PlayoffOddsPanel";

export default function StandingsPage() {
  const { state, retry } = usePageData(fetchStandings);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty icon={Trophy} title="No standings yet" body="League standings load from Yahoo once the season data syncs." />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={30} />}
    </>
  );
}

function Loaded({ data }: { data: StandingsData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">League · FourzynBurn</div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Standings</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Records, per-category league ranks, and Monte-Carlo playoff odds.
        </p>
      </motion.div>
      <motion.div variants={staggerItem}>
        <StandingsTable data={data} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <PlayoffOddsPanel data={data} />
      </motion.div>
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-72" />
      <Skeleton className="h-96 w-full rounded-2xl" />
      <Skeleton className="h-64 w-full rounded-2xl" />
    </div>
  );
}
