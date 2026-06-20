"use client";

import { motion } from "framer-motion";
import { ShieldAlert } from "lucide-react";
import { fetchClosers, type ClosersData } from "@/lib/closers-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { CloserCard } from "@/components/closers/CloserCard";

export default function ClosersPage() {
  const { state, retry } = usePageData(fetchClosers);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={ShieldAlert}
            title="No closer data"
            body="Bullpen depth charts aren't loaded yet — they refresh from the data bootstrap."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={20} />}
    </>
  );
}

function Loaded({ data }: { data: ClosersData }) {
  const entries = [...data.entries].sort((a, b) => b.security - a.security);
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Bullpen · Save Depth Chart
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Closer Monitor</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Job security across the league — sorted most secure first. Click a name for the full card.
        </p>
      </motion.div>
      <motion.div
        variants={staggerItem}
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      >
        {entries.map((e) => (
          <CloserCard key={e.team} entry={e} />
        ))}
      </motion.div>
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-72" />
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-44 rounded-2xl" />
        ))}
      </div>
    </div>
  );
}
