"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, ArrowUp, ArrowDown, TrendingUp } from "lucide-react";
import { fetchTrades, type TradesData, type TradeRec, type TradePlayer, type CatImpact } from "@/lib/trades-data";
import { staggerContainer, staggerItem, useCountUp } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { ComparePanel } from "@/components/trades/ComparePanel";
import { cn } from "@/lib/utils";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";

type Tab = "finder" | "compare";

const YOU = "Team Hickey";

export default function TradesPage() {
  const [tab, setTab] = useState<Tab>("finder");
  const { state, retry } = usePageData(fetchTrades);

  return (
    <>
      <main className="w-full flex-1 space-y-6 px-5 py-6">
        <PageHead tab={tab} onTab={setTab} />

        {tab === "finder" && (
          <>
            {state.status === "loading" && <LoadingView />}
            {state.status === "error" && <PageError onRetry={retry} />}
            {state.status === "empty" && (
              <PageEmpty
                icon={TrendingUp}
                title="No trade ideas yet"
                body="We need a bit more league data to surface targets."
              />
            )}
            {state.status === "loaded" && <FinderView data={state.data} />}
          </>
        )}

        {tab === "compare" && <ComparePanel />}
      </main>
      <Footer freshnessMinutes={9} />
    </>
  );
}

const TABS: { key: Tab; label: string }[] = [
  { key: "finder", label: "Finder" },
  { key: "compare", label: "Compare" },
];

function PageHead({ tab, onTab }: { tab: Tab; onTab: (t: Tab) => void }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Trade Workbench · Week 13
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Trades</h1>
      </div>
      <div role="tablist" aria-label="Trades views" className="inline-flex rounded-xl border border-line bg-surface p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => onTab(t.key)}
            className={cn(
              "rounded-lg px-4 py-1.5 text-[13px] font-bold transition-colors duration-[var(--dur-1)]",
              tab === t.key ? "bg-navy text-white" : "text-ink-2 hover:text-navy",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function FinderView({ data }: { data: TradesData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-4">
      <motion.p variants={staggerItem} className="text-[13px] text-ink-2">
        {data.needs.length > 0 ? (
          <>
            {data.recs.length} deals that target your needs:{" "}
            {data.needs.map((n, i) => (
              <span key={n}>
                <span className="font-bold text-heat">{n}</span>
                {i < data.needs.length - 1 ? ", " : ""}
              </span>
            ))}
            .
          </>
        ) : (
          <>
            {data.recs.length} {data.recs.length === 1 ? "trade idea" : "trade ideas"} surfaced from your
            roster.
          </>
        )}
      </motion.p>
      {data.recs.map((rec) => (
        <motion.div key={rec.id} variants={staggerItem}>
          <TradeCard rec={rec} />
        </motion.div>
      ))}
    </motion.div>
  );
}

function gradeColor(grade: string): string {
  const g = grade[0];
  if (g === "A") return "bg-ok/12 text-ok";
  if (g === "B") return "bg-heat/12 text-heat";
  return "bg-surface-2 text-ink-2";
}

function TradeCard({ rec }: { rec: TradeRec }) {
  const playoff = useCountUp(rec.playoffDelta ?? 0);
  return (
    <Card className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Trade With</div>
          <div className="font-display text-base font-bold text-navy">
            {rec.partner}
            {rec.partnerRecord && (
              <span className="tnum ml-2 text-[12px] font-medium text-ink-3">{rec.partnerRecord}</span>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <div className="text-right">
            <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Verdict</div>
            <div className="text-[13px] font-bold text-navy">{rec.verdict}</div>
          </div>
          {rec.grade ? (
            <span
              className={cn(
                "flex size-14 items-center justify-center rounded-2xl font-display text-[30px] font-extrabold leading-none",
                gradeColor(rec.grade),
              )}
            >
              {rec.grade}
            </span>
          ) : rec.netSgp !== undefined ? (
            <span
              className={cn(
                "flex size-14 flex-col items-center justify-center rounded-2xl font-display leading-none",
                rec.netSgp >= 0 ? "bg-ok/12 text-ok" : "bg-ember/12 text-ember",
              )}
              title="Net SGP gain from this trade"
            >
              <span className="tnum text-[19px] font-extrabold">
                {rec.netSgp >= 0 ? "+" : ""}
                {rec.netSgp.toFixed(1)}
              </span>
              <span className="text-[8px] font-bold uppercase tracking-wide opacity-70">SGP</span>
            </span>
          ) : null}
        </div>
      </div>

      <div className="grid items-center gap-3 md:grid-cols-[1fr_auto_1fr]">
        <Side label="You Give" players={rec.give} rosteredBy={YOU} tone="give" />
        <div className="flex justify-center">
          <span className="flex size-9 items-center justify-center rounded-full bg-surface text-ink-3">
            <ArrowRight className="size-5" aria-hidden />
          </span>
        </div>
        <Side label="You Get" players={rec.get} rosteredBy={rec.partner} tone="get" />
      </div>

      {rec.impact && rec.impact.length > 0 && <ImpactLedger impact={rec.impact} />}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <p className="max-w-[60ch] text-[12.5px] text-ink-2">{rec.rationale}</p>
        <div className="flex items-center gap-3">
          {rec.playoffDelta !== undefined && (
            <span className="tnum inline-flex items-center gap-1 text-[12px] font-semibold text-ok">
              <TrendingUp className="size-3.5" aria-hidden />
              +{playoff}% playoff odds
            </span>
          )}
          <button className="inline-flex min-h-9 items-center gap-1 rounded-lg bg-gradient-to-b from-heat-bright to-heat px-4 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none">
            Analyze
          </button>
        </div>
      </div>
    </Card>
  );
}

function Side({
  label,
  players,
  rosteredBy,
  tone,
}: {
  label: string;
  players: TradePlayer[];
  rosteredBy: string;
  tone: "give" | "get";
}) {
  return (
    <div>
      <div
        className={cn(
          "mb-1.5 text-[10px] font-bold uppercase tracking-wide",
          tone === "give" ? "text-ember" : "text-ok",
        )}
      >
        {label}
      </div>
      <div className="space-y-1.5">
        {players.map((p, i) => (
          <PlayerDialog key={`${p.name}-${i}`} player={{ ...p, rosteredBy }}>
            <button className="flex w-full items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-left transition-colors duration-[var(--dur-1)] hover:border-heat/40 hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50">
              <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
              <span className="min-w-0">
                <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
                <span className="tnum block text-[10.5px] text-ink-3">{p.keyStat ?? p.posLabel}</span>
              </span>
            </button>
          </PlayerDialog>
        ))}
      </div>
    </div>
  );
}

/** Gains-vs-gives-up tug bar + the per-category delta chips. */
function ImpactLedger({ impact }: { impact: CatImpact[] }) {
  const gains = impact.filter((i) => i.dir === "up").length;
  const losses = impact.filter((i) => i.dir === "down").length;
  const total = impact.length || 1;
  return (
    <div className="mt-4">
      <div className="mb-1.5 flex items-center justify-between text-[10px] font-bold uppercase tracking-wide">
        <span className="text-ok">
          {gains} {gains === 1 ? "gain" : "gains"}
        </span>
        <span className="text-ember">gives up {losses}</span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-surface-2">
        <span className="block h-full bg-ok" style={{ width: `${(gains / total) * 100}%` }} />
        <span className="block h-full bg-ember" style={{ width: `${(losses / total) * 100}%` }} />
      </div>
      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {impact.map((im) => (
          <ImpactChip key={im.cat} im={im} />
        ))}
      </div>
    </div>
  );
}

function ImpactChip({ im }: { im: CatImpact }) {
  const up = im.dir === "up";
  const Icon = up ? ArrowUp : ArrowDown;
  return (
    <span
      className={cn(
        "tnum inline-flex items-center gap-0.5 rounded-md px-2 py-0.5 text-[11px] font-bold",
        up ? "bg-ok/12 text-ok" : "bg-ember/12 text-ember",
      )}
    >
      <Icon className="size-3" aria-hidden />
      {im.cat} {im.delta}
    </span>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-56" />
      <Skeleton className="h-52 w-full rounded-2xl" />
      <Skeleton className="h-52 w-full rounded-2xl" />
      <Skeleton className="h-52 w-full rounded-2xl" />
    </div>
  );
}
