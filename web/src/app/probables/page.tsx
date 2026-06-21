"use client";

import { CalendarDays } from "lucide-react";
import { fetchProbables } from "@/lib/probables-data";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { ProbableGrid } from "@/components/probables/ProbableGrid";

export default function ProbablesPage() {
  const { state, retry } = usePageData(fetchProbables);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        <div className="mb-5">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
            Schedule · Next 7 Days
          </div>
          <h1 className="font-display text-3xl font-extrabold text-navy">Probable Pitchers</h1>
          <p className="mt-1 text-[13px] text-ink-2">
            Every team&apos;s probable starter for the week — scored by matchup ease, flagged for two-start weeks, and
            tagged by league availability.
          </p>
        </div>
        {state.status === "loading" && <Skeleton className="h-96 w-full rounded-2xl" />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={CalendarDays}
            title="No probables posted"
            body="Probable starters typically appear 1–5 days out — check back soon."
          />
        )}
        {state.status === "loaded" && <ProbableGrid data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={60} />}
    </>
  );
}
