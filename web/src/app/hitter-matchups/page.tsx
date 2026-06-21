"use client";

import { TrendingUp } from "lucide-react";
import { fetchHitterMatchups } from "@/lib/hitter-matchups-data";
import { usePageData } from "@/lib/use-page-data";
import { Footer } from "@/components/chrome/Footer";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
import { HitterMatchupGrid } from "@/components/probables/HitterMatchupGrid";

export default function HitterMatchupsPage() {
  const { state, retry } = usePageData(fetchHitterMatchups);
  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        <div className="mb-5">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
            Schedule · Next 7 Days
          </div>
          <h1 className="font-display text-3xl font-extrabold text-navy">Hitter Matchups</h1>
          <p className="mt-1 text-[13px] text-ink-2">
            Every team&apos;s batting schedule for the week — scored by matchup ease against the opposing starter,
            ranked by weekly difficulty, and tagged by league availability.
          </p>
        </div>
        {state.status === "loading" && <Skeleton className="h-96 w-full rounded-2xl" />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={TrendingUp}
            title="No matchup data"
            body="Hitter matchup data will appear once probable pitchers are posted — check back soon."
          />
        )}
        {state.status === "loaded" && <HitterMatchupGrid data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={60} />}
    </>
  );
}
