"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Search, SearchX, Plus, Flame, ArrowUp, ArrowDown } from "lucide-react";
import { fetchPlayers, type PlayersData, type FreeAgent } from "@/lib/players-data";
import { staggerContainer, staggerItem, useCountUp } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { EmptyState } from "@/components/ui/EmptyState";
import { HeroNum } from "@/components/ui/HeroNum";
import { COLORS, heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";

type Filter = "all" | "hitters" | "pitchers" | "need";

export default function PlayersPage() {
  const { state, retry } = usePageData(fetchPlayers);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Search}
            title="No free agents available"
            body="The free-agent pool is empty right now."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}

function Loaded({ data }: { data: PlayersData }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.freeAgents.filter((p) => {
      if (filter === "hitters" && !p.hitter) return false;
      if (filter === "pitchers" && p.hitter) return false;
      if (filter === "need" && p.fit !== data.topNeed) return false;
      if (q && !p.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, filter, query]);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header />
      </motion.div>
      {data.freeAgents[0] && (
        <motion.div variants={staggerItem}>
          <TopPickup fa={data.freeAgents[0]} need={data.topNeed} />
        </motion.div>
      )}
      <motion.div variants={staggerItem}>
        <Card className="p-5">
          <Toolbar
            filter={filter}
            setFilter={setFilter}
            query={query}
            setQuery={setQuery}
            need={data.topNeed}
            shown={rows.length}
            total={data.freeAgents.length}
          />
          <FATable rows={rows} need={data.topNeed} />
        </Card>
      </motion.div>
    </motion.div>
  );
}

function Header() {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
        Free Agents · 12-Team League
      </div>
      <h1 className="font-display text-3xl font-extrabold text-navy">Players</h1>
      <p className="mt-1 text-[13px] text-ink-2">Available players ranked by value to your roster.</p>
    </div>
  );
}

function TopPickup({ fa, need }: { fa: FreeAgent; need: string }) {
  const value = useCountUp(fa.value);
  const ramp = heatColor(fa.value);
  const helpsNeed = fa.fit === need;
  return (
    <Card tone="raised" className="relative overflow-hidden">
      {/* atmosphere: faint heat glow, top-right */}
      <span
        aria-hidden
        className="pointer-events-none absolute -right-20 -top-24 size-56 rounded-full opacity-[0.08] blur-3xl"
        style={{ background: ramp }}
      />
      <div className="relative flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:gap-5">
        <PlayerDialog player={{ ...fa, rosteredBy: "Free Agent" }}>
          <button className="group flex min-w-0 flex-1 items-center gap-3.5 text-left focus-visible:outline-none">
            <span className="relative shrink-0">
              <PlayerAvatar mlbId={fa.mlbId} teamId={fa.teamId} name={fa.name} size={60} />
              <span className="absolute -bottom-1 -right-1 flex size-6 items-center justify-center rounded-full bg-gradient-to-b from-[#ff7a2e] to-heat font-display text-[12px] font-extrabold text-white shadow-[0_2px_8px_rgba(255,92,16,0.45)] ring-2 ring-canvas">
                1
              </span>
            </span>
            <span className="min-w-0">
              <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.14em] text-heat">
                <Flame className="size-3.5" aria-hidden />
                Top Pickup
              </span>
              <span className="mt-1 block truncate font-display text-[22px] font-extrabold leading-tight tracking-tight text-navy decoration-heat/40 underline-offset-4 group-hover:underline">
                {fa.name}
              </span>
              <span className="tnum mt-0.5 block text-[12.5px] text-ink-2">
                {fa.pos} · {fa.teamAbbr}
                {fa.tag && <span className="ml-1 font-semibold text-heat">· {fa.tag}</span>}
              </span>
            </span>
          </button>
        </PlayerDialog>

        <div className="flex shrink-0 items-center gap-5 sm:border-l sm:border-line sm:pl-5">
          <div className="text-right">
            <HeroNum width={92} className="block text-[46px] leading-[0.85]" style={{ color: ramp }}>
              {value}
            </HeroNum>
            <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-ink-3">Value</div>
            <div className="ml-auto mt-2 h-1.5 w-24 overflow-hidden rounded-full bg-surface-2">
              <span
                className="block h-full rounded-full"
                style={{ width: `${fa.value}%`, background: ramp }}
              />
            </div>
          </div>
          <button className="inline-flex min-h-11 items-center gap-1.5 rounded-xl bg-gradient-to-b from-[#ff7a2e] to-heat px-5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.03] active:scale-95 motion-reduce:transform-none">
            <Plus className="size-4" aria-hidden />
            Add
          </button>
        </div>
      </div>

      <div className="relative flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-line px-5 py-3">
        <p className="text-[12.5px] text-ink-2">
          {helpsNeed ? (
            <>
              Top available <span className="font-bold text-navy">{need}</span> source — only{" "}
              <span className="tnum font-semibold text-navy">{fa.ownPct}%</span> rostered.
            </>
          ) : (
            <>
              Highest-value add on the board —{" "}
              <span className="tnum font-semibold text-navy">{fa.ownPct}%</span> rostered.
            </>
          )}
        </p>
        <div className="tnum flex items-center gap-4">
          {fa.stats.map((s) => (
            <span key={s.label} className="text-[12.5px]">
              <span className="font-bold text-navy">{s.value}</span>{" "}
              <span className="text-[10.5px] text-ink-3">{s.label}</span>
            </span>
          ))}
        </div>
      </div>
    </Card>
  );
}

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "hitters", label: "Hitters" },
  { key: "pitchers", label: "Pitchers" },
];

function Toolbar({
  filter,
  setFilter,
  query,
  setQuery,
  need,
  shown,
  total,
}: {
  filter: Filter;
  setFilter: (f: Filter) => void;
  query: string;
  setQuery: (q: string) => void;
  need: string;
  shown: number;
  total: number;
}) {
  const chips = [...FILTERS, { key: "need" as Filter, label: `Helps ${need}` }];
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <div className="flex flex-wrap gap-1.5">
        {chips.map((c) => (
          <button
            key={c.key}
            onClick={() => setFilter(c.key)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-[12px] font-bold transition-colors",
              filter === c.key ? "bg-navy text-white" : "bg-surface text-ink-2 hover:bg-surface-2",
            )}
          >
            {c.label}
          </button>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span className="tnum text-[11px] text-ink-3">
          {shown} of {total}
        </span>
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

function ownColor(pct: number): string {
  if (pct < 25) return COLORS.ok; // widely available
  if (pct < 50) return COLORS.warn;
  return COLORS.heat; // heavily owned
}

function FATable({ rows, need }: { rows: FreeAgent[]; need: string }) {
  if (rows.length === 0) {
    return (
      <EmptyState
        icon={SearchX}
        title="No players match"
        body="Try clearing a filter or searching a different name."
      />
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px]">
        <thead>
          <tr className="border-b border-line">
            <Th>#</Th>
            <Th left>Player</Th>
            <Th left>Key Stats</Th>
            <Th>Fit</Th>
            <Th>% Rostered</Th>
            <Th left>Value</Th>
            <Th>Add</Th>
          </tr>
        </thead>
        <tbody className="text-[13px]">
          {rows.map((p) => {
            const oc = ownColor(p.ownPct);
            const up = p.ownDelta > 0;
            const DeltaIcon = up ? ArrowUp : ArrowDown;
            return (
              <tr key={p.mlbId} className="border-b border-line/60 transition-colors duration-[var(--dur-1)] hover:bg-surface/60">
                <td className="tnum px-2.5 py-2.5 text-center font-bold text-ink-3">{p.rank}</td>
                <td className="p-0">
                  <PlayerDialog player={{ ...p, rosteredBy: "Free Agent" }}>
                    <button className="flex w-full items-center gap-2 rounded px-2.5 py-2.5 text-left transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50">
                      <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
                      <span className="min-w-0">
                        <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
                        <span className="tnum block text-[10.5px] text-ink-3">
                          {p.pos} · {p.teamAbbr}
                          {p.tag && <span className="ml-1 font-semibold text-heat">· {p.tag}</span>}
                        </span>
                      </span>
                    </button>
                  </PlayerDialog>
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="tnum flex gap-3 text-ink-2">
                    {p.stats.map((s) => (
                      <span key={s.label}>
                        <span className="font-semibold text-ink">{s.value}</span>{" "}
                        <span className="text-[10.5px] text-ink-3">{s.label}</span>
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-2.5 py-2.5 text-center">
                  <span
                    className={cn(
                      "inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold",
                      p.fit === need ? "bg-heat/12 text-heat" : "bg-surface-2 text-ink-2",
                    )}
                  >
                    {p.fit}
                  </span>
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center justify-end gap-1.5">
                    <span className="tnum font-semibold" style={{ color: oc }}>
                      {p.ownPct}%
                    </span>
                    <span className={cn("tnum inline-flex items-center text-[10.5px]", up ? "text-ok" : "text-ink-3")}>
                      <DeltaIcon className="size-3" aria-hidden />
                      {Math.abs(p.ownDelta)}
                    </span>
                  </div>
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-surface-2">
                      <span className="block h-full rounded-full bg-heat" style={{ width: `${p.value}%` }} />
                    </div>
                    <span className="tnum w-7 text-right text-[12px] font-bold text-navy">{p.value}</span>
                  </div>
                </td>
                <td className="px-2.5 py-2.5 text-right">
                  <button
                    aria-label={`Add ${p.name}`}
                    className="inline-flex size-8 items-center justify-center rounded-lg bg-navy text-white transition-colors hover:bg-navy-700"
                  >
                    <Plus className="size-4" aria-hidden />
                  </button>
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
      <Skeleton className="h-20 w-full rounded-2xl" />
      <Skeleton className="h-96 w-full rounded-2xl" />
    </div>
  );
}
