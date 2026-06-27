"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Scale, Users, Wand2 } from "lucide-react";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { PageNotLinked } from "@/components/ui/PageStates";
import { LineupTable } from "@/components/optimizer/LineupTable";
import { ScopeSelector } from "@/components/startsit/ScopeSelector";
import { PlayerMultiSelect } from "@/components/startsit/PlayerMultiSelect";
import { CompareCard } from "@/components/startsit/CompareCard";
import { VerdictPanel } from "@/components/startsit/VerdictPanel";
import { type PlayerPick } from "@/lib/player-search";
import { isTeamNotLinked } from "@/lib/api/errors";
import {
  compareStartSit,
  optimizeStartSit,
  type Scope,
  type StartSitCompareData,
  type StartSitOptimizeData,
} from "@/lib/start-sit-data";

type ErrKind = "unlinked" | "error" | null;

export default function StartSitPage() {
  const [scope, setScope] = useState<Scope>("today");
  const [selected, setSelected] = useState<PlayerPick[]>([]);
  const [compare, setCompare] = useState<StartSitCompareData | null>(null);
  const [optimized, setOptimized] = useState<StartSitOptimizeData | null>(null);
  const [comparing, setComparing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<ErrKind>(null);

  const ids = selected.map((p) => p.id);
  const ready = selected.length >= 2;

  // Selection edits invalidate any prior comparison/lineup.
  const onSelect = (next: PlayerPick[]) => {
    setSelected(next);
    setCompare(null);
    setOptimized(null);
    setError(null);
  };

  const runCompare = (forScope: Scope, forIds: number[]) => {
    if (forIds.length < 2) return;
    setComparing(true);
    setError(null);
    setOptimized(null);
    compareStartSit(forScope, forIds)
      .then(setCompare)
      .catch((e) => setError(isTeamNotLinked(e) ? "unlinked" : "error"))
      .finally(() => setComparing(false));
  };

  const onScope = (s: Scope) => {
    setScope(s);
    // Re-fire only when a comparison already exists (the user has committed a set).
    if (compare) runCompare(s, ids);
    else setOptimized(null);
  };

  const onApply = () => {
    if (!ready) return;
    setApplying(true);
    setError(null);
    optimizeStartSit(scope, ids)
      .then(setOptimized)
      .catch((e) => setError(isTeamNotLinked(e) ? "unlinked" : "error"))
      .finally(() => setApplying(false));
  };

  return (
    <>
      <main className="w-full flex-1 space-y-6 px-5 py-6">
        <Header />

        <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-5">
          <motion.div variants={staggerItem}>
            <Card className="space-y-4 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Horizon</div>
                <ScopeSelector value={scope} onChange={onScope} />
              </div>
              <PlayerMultiSelect selected={selected} onChange={onSelect} />
              <div className="flex items-center gap-3">
                <button
                  onClick={() => runCompare(scope, ids)}
                  disabled={!ready || comparing}
                  className="inline-flex min-h-10 items-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] enabled:hover:scale-[1.02] enabled:active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transform-none"
                >
                  <Scale className="size-4" aria-hidden />
                  {comparing ? "Comparing…" : "Compare"}
                </button>
                {!ready && <span className="text-[12px] text-ink-3">Pick at least 2 players to compare.</span>}
              </div>
            </Card>
          </motion.div>

          {error === "unlinked" ? (
            <motion.div variants={staggerItem}>
              <PageNotLinked />
            </motion.div>
          ) : error === "error" ? (
            <motion.div variants={staggerItem}>
              <Card className="mx-auto max-w-md">
                <EmptyState
                  icon={Users}
                  tone="error"
                  title="We couldn't compare those players"
                  body="The data service didn't respond. Please try again."
                />
              </Card>
            </motion.div>
          ) : compare ? (
            <motion.div variants={staggerItem}>
              <Results data={compare} selected={selected} onApply={onApply} applying={applying} optimized={optimized} />
            </motion.div>
          ) : (
            !comparing && (
              <motion.div variants={staggerItem}>
                <Card className="mx-auto max-w-md">
                  <EmptyState
                    icon={Scale}
                    title="No comparison yet"
                    body="Pick 2–6 players above, then Compare to see a ranked start/sit verdict."
                  />
                </Card>
              </motion.div>
            )
          )}
        </motion.div>
      </main>
      <Footer freshnessMinutes={9} />
    </>
  );
}

function Header() {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Daily Decisions</div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Start/Sit</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Compare any 2–6 players for a window, then apply the best fits to your open slots.
        </p>
      </div>
    </div>
  );
}

/** Normalize a name for matching candidates ↔ selected picks. */
function normName(s: string): string {
  return s.trim().toLowerCase();
}

function Results({
  data,
  selected,
  onApply,
  applying,
  optimized,
}: {
  data: StartSitCompareData;
  selected: PlayerPick[];
  onApply: () => void;
  applying: boolean;
  optimized: StartSitOptimizeData | null;
}) {
  // Resolve each candidate to its HEATER id (by mlbId when both nonzero, else by
  // name) so we can mark START vs SIT against the verdict's id lists; and map the
  // verdict ids back to names for the panel.
  const byId = new Map(selected.map((p) => [p.id, p.name] as const));
  const startSet = new Set(data.verdict.startIds);
  const candId = (cName: string, cMlb: number): number | undefined => {
    const hit =
      selected.find((p) => cMlb > 0 && p.mlbId === cMlb) ?? selected.find((p) => normName(p.name) === normName(cName));
    return hit?.id;
  };
  const startNames = data.verdict.startIds.map((id) => byId.get(id) ?? `#${id}`);
  const sitNames = data.verdict.sitIds.map((id) => byId.get(id) ?? `#${id}`);

  const lineup = optimized ? [...optimized.starters, ...optimized.bench] : [];

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-3">
          {data.candidates.length === 0 ? (
            <Card className="mx-auto max-w-md">
              <EmptyState
                icon={Scale}
                title="No ranking available"
                body="We couldn't score these players for this window. Try a different horizon or selection."
              />
            </Card>
          ) : (
            data.candidates.map((c) => {
              const id = candId(c.player.name, c.player.mlbId);
              const started = id === undefined ? undefined : startSet.has(id);
              return <CompareCard key={`${c.rank}-${c.player.name}`} c={c} started={started} />;
            })
          )}
        </div>
        <aside>
          <VerdictPanel
            openSlots={data.openSlots}
            confidenceLabel={data.confidenceLabel}
            reasoning={data.verdict.reasoning}
            startNames={startNames}
            sitNames={sitNames}
            onApply={onApply}
            applying={applying}
            canApply={selected.length >= 2}
          />
        </aside>
      </div>

      {optimized && (
        <Card className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 font-display text-base font-bold text-navy">
              <Wand2 className="size-4 text-heat" aria-hidden />
              Filled Lineup
            </div>
            <span className="text-[11px] text-ink-3">{optimized.summary}</span>
          </div>
          {lineup.length > 0 ? (
            <LineupTable slots={lineup} />
          ) : (
            <EmptyState
              icon={Wand2}
              title="No lineup returned"
              body="No open slots were filled for this window. This populates with live Yahoo data."
            />
          )}
        </Card>
      )}
    </div>
  );
}
