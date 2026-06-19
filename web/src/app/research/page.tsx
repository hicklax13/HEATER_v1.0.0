"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Flame, Snowflake, TrendingUp, TrendingDown, ArrowUp, ArrowDown, Minus, Search, SearchX } from "lucide-react";
import { fetchResearch, type ResearchData, type LeaderRow, type Lens } from "@/lib/research-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { EmptyState } from "@/components/ui/EmptyState";
import { cn } from "@/lib/utils";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";

const LENSES: { key: Lens; label: string }[] = [
  { key: "overall", label: "Leaders" },
  { key: "hot", label: "Hot" },
  { key: "cold", label: "Cold" },
  { key: "breakout", label: "Breakouts" },
  { key: "sell", label: "Sell-High" },
];

const TAG: Record<LeaderRow["tag"], { label: string; cls: string }> = {
  hot: { label: "Hot", cls: "bg-heat/12 text-heat" },
  cold: { label: "Cold", cls: "bg-steel/15 text-steel" },
  breakout: { label: "Breakout", cls: "bg-ok/12 text-ok" },
  sell: { label: "Sell-High", cls: "bg-warn/15 text-warn" },
};

export default function ResearchPage() {
  const { state, retry } = usePageData(fetchResearch);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Flame}
            title="No leaders to show"
            body="Leaderboard data isn't available yet."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}

function Loaded({ data }: { data: ResearchData }) {
  const [lens, setLens] = useState<Lens>("overall");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.leaders.filter((p) => {
      if (lens !== "overall" && p.tag !== lens) return false;
      if (q && !p.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, lens, query]);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header />
      </motion.div>
      <motion.div variants={staggerItem}>
        <Card className="p-5">
          <Toolbar lens={lens} setLens={setLens} query={query} setQuery={setQuery} shown={rows.length} />
          <LeaderTable rows={rows} />
        </Card>
      </motion.div>
    </motion.div>
  );
}

function Header() {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
        Research · Week 13
      </div>
      <h1 className="font-display text-3xl font-extrabold text-navy">Research</h1>
      <p className="mt-1 text-[13px] text-ink-2">League-wide leaders, risers, and regression risks.</p>
    </div>
  );
}

function lensIcon(key: Lens) {
  if (key === "hot") return <Flame className="size-3.5" aria-hidden />;
  if (key === "cold") return <Snowflake className="size-3.5" aria-hidden />;
  if (key === "breakout") return <TrendingUp className="size-3.5" aria-hidden />;
  if (key === "sell") return <TrendingDown className="size-3.5" aria-hidden />;
  return null;
}

function Toolbar({
  lens,
  setLens,
  query,
  setQuery,
  shown,
}: {
  lens: Lens;
  setLens: (l: Lens) => void;
  query: string;
  setQuery: (q: string) => void;
  shown: number;
}) {
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <div className="flex flex-wrap gap-1.5">
        {LENSES.map((l) => (
          <button
            key={l.key}
            onClick={() => setLens(l.key)}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-[12px] font-bold transition-colors",
              lens === l.key ? "bg-navy text-white" : "bg-surface text-ink-2 hover:bg-surface-2",
            )}
          >
            {lensIcon(l.key)}
            {l.label}
          </button>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span className="tnum text-[11px] text-ink-3">{shown} players</span>
        <label className="flex items-center gap-2 rounded-lg border border-line bg-canvas px-3 py-1.5">
          <Search className="size-4 text-ink-3" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search players"
            className="w-36 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-3"
          />
        </label>
      </div>
    </div>
  );
}

function LeaderTable({ rows }: { rows: LeaderRow[] }) {
  if (rows.length === 0) {
    return (
      <EmptyState
        icon={SearchX}
        title="No players match"
        body="Try a different lens or search a different name."
      />
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[680px]">
        <thead>
          <tr className="border-b border-line">
            <Th>#</Th>
            <Th left>Player</Th>
            <Th left>Key Stats</Th>
            <Th left>Why</Th>
            <Th>Trend</Th>
            <Th left>Value</Th>
          </tr>
        </thead>
        <tbody className="text-[13px]">
          {rows.map((p) => {
            const tag = TAG[p.tag];
            const up = p.trend === "up";
            const TrendIcon = up ? ArrowUp : p.trend === "down" ? ArrowDown : Minus;
            const trendCls = up ? "text-ok" : p.trend === "down" ? "text-ember" : "text-ink-3";
            return (
              <tr
                key={p.rank}
                className="border-b border-line/60 transition-colors duration-[var(--dur-1)] hover:bg-surface/60"
              >
                <td className="tnum px-2.5 py-2.5 text-center font-bold text-ink-3">{p.rank}</td>
                <td className="p-0">
                  <PlayerDialog player={p}>
                    <button className="flex w-full items-center gap-2 rounded px-2.5 py-2.5 text-left transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50">
                      <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
                      <span className="min-w-0">
                        <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
                        <span className="tnum block text-[10.5px] text-ink-3">
                          {p.pos} · {p.teamAbbr}
                          <span className={cn("ml-1 rounded px-1 text-[9px] font-bold", tag.cls)}>{tag.label}</span>
                        </span>
                      </span>
                    </button>
                  </PlayerDialog>
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="tnum flex gap-3 text-ink-2">
                    {p.stats.map((s, i) => (
                      <span key={i} className="font-semibold text-ink">
                        {s}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-2.5 py-2.5 text-[12px] text-ink-2">{p.note}</td>
                <td className="px-2.5 py-2.5 text-center">
                  <span className={cn("inline-flex items-center justify-center", trendCls)}>
                    <TrendIcon className="size-4" aria-hidden />
                  </span>
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-surface-2">
                      <span className="block h-full rounded-full bg-heat" style={{ width: `${p.value}%` }} />
                    </div>
                    <span className="tnum w-7 text-right text-[12px] font-bold text-navy">{p.value}</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy",
        left ? "text-left" : "text-center",
      )}
    >
      {children}
    </th>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-56" />
      <Skeleton className="h-96 w-full rounded-2xl" />
    </div>
  );
}
