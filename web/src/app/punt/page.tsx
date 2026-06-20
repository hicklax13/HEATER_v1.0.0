"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Target, ArrowRightLeft, ArrowRight } from "lucide-react";
import { fetchPunt, type PuntData } from "@/lib/punt-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { PuntVerdictTable } from "@/components/punt/PuntVerdictTable";

export default function PuntPage() {
  const { state, retry } = usePageData(fetchPunt);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty icon={Target} title="No punt analysis yet" body="Punt detection needs your league standings — syncs from Yahoo." />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={30} />}
    </>
  );
}

function Loaded({ data }: { data: PuntData }) {
  const compete = data.cats.filter((c) => c.verdict === "compete").length;
  const punts = data.candidates;
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Strategy · Category Punt</div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Punt Analyzer</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Which categories to concede — and where to reinvest. Auto-detected from your standings.
        </p>
      </motion.div>

      <motion.div variants={staggerItem}>
        <Card className="overflow-hidden p-0">
          <div className="bg-gradient-to-r from-navy to-[#15294a] p-5 text-white">
            <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-flame">
              <Target className="size-3.5" aria-hidden /> Punt strategy
            </div>
            {punts.length > 0 ? (
              <>
                <div className="mt-1 font-display text-xl font-extrabold">Built to punt {punts.join(" + ")}</div>
                <p className="mt-1 max-w-[68ch] text-[13px] text-white/80">
                  You&apos;re bottom-of-league in {punts.join(" / ")}
                  {" "}and can&apos;t realistically gain there. Concede them and pour resources into the {compete}
                  {" "}categories you can win.
                </p>
              </>
            ) : (
              <>
                <div className="mt-1 font-display text-xl font-extrabold">No clear punt — compete across the board</div>
                <p className="mt-1 text-[13px] text-white/80">No category is a lost cause; stay balanced.</p>
              </>
            )}
          </div>
        </Card>
      </motion.div>

      <motion.div variants={staggerItem}>
        <PuntVerdictTable data={data} />
      </motion.div>

      {punts.length > 0 && (
        <motion.div variants={staggerItem}>
          <Card className="p-5">
            <div className="mb-1 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
              <ArrowRightLeft className="size-4 text-heat" aria-hidden /> Reallocate
            </div>
            <p className="mb-4 max-w-[70ch] text-[13px] text-ink-2">
              Punting {punts.join(" + ")} frees roster spots. Drop the {punts.join("/")}-only specialists (saves-only
              relievers, high-ERA innings) and reinvest in your active categories.
            </p>
            <Link
              href="/players"
              className="inline-flex min-h-10 items-center gap-1.5 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-4 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
            >
              Browse free agents
              <ArrowRight className="size-4" aria-hidden />
            </Link>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-64" />
      <Skeleton className="h-28 w-full rounded-2xl" />
      <Skeleton className="h-80 w-full rounded-2xl" />
    </div>
  );
}
