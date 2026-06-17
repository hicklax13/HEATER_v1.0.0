"use client";

import { useEffect, useMemo, useState } from "react";
import { Search, Plus, Flame, ArrowUp, ArrowDown } from "lucide-react";
import { fetchPlayers, type PlayersData, type FreeAgent } from "@/lib/players-data";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerLink } from "@/components/player/PlayerLink";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { COLORS } from "@/lib/tokens";
import { cn } from "@/lib/utils";

type Filter = "all" | "hitters" | "pitchers" | "need";

export default function PlayersPage() {
  const [data, setData] = useState<PlayersData | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    let alive = true;
    fetchPlayers().then((d) => {
      if (alive) setData(d);
    });
    return () => {
      alive = false;
    };
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
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
    <>
      <main className="w-full flex-1 px-5 py-6">
        {!data ? (
          <LoadingView />
        ) : (
          <div className="space-y-6">
            <Header />
            <NeedCallout
              need={data.topNeed}
              count={data.freeAgents.filter((p) => p.fit === data.topNeed).length}
              onShow={() => setFilter("need")}
            />
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
          </div>
        )}
      </main>
      <Footer freshnessMinutes={9} />
    </>
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

function NeedCallout({ need, count, onShow }: { need: string; count: number; onShow: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-line border-l-4 border-l-heat bg-surface p-4">
      <Flame className="size-5 shrink-0 text-heat" aria-hidden />
      <p className="flex-1 text-[14px] text-ink">
        Your biggest gap is <span className="font-bold text-navy">{need}</span> — {count} available free
        agents help most.
      </p>
      <button
        onClick={onShow}
        className="inline-flex min-h-9 items-center gap-1 rounded-lg bg-gradient-to-b from-[#ff7a2e] to-heat px-4 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
      >
        Show {need} Help
      </button>
    </div>
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
    return <p className="py-8 text-center text-[13px] text-ink-3">No players match these filters.</p>;
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
              <tr key={p.mlbId} className="border-b border-line/60">
                <td className="tnum px-2.5 py-2.5 text-center font-bold text-ink-3">{p.rank}</td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center gap-2">
                    <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
                    <div className="min-w-0">
                      <PlayerLink player={{ ...p, rosteredBy: "Free Agent" }} className="text-[13px]" />
                      <div className="tnum text-[10.5px] text-ink-3">
                        {p.pos} · {p.teamAbbr}
                        {p.tag && <span className="ml-1 font-semibold text-heat">· {p.tag}</span>}
                      </div>
                    </div>
                  </div>
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
