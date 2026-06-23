"use client";

import { useEffect, useState } from "react";
import { Search, X, Library, AlertTriangle } from "lucide-react";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { searchPlayers, type PlayerPick } from "@/lib/player-search";
import {
  fetchDatabank,
  databankColumns,
  formatDatabankValue,
  type DatabankData,
} from "@/lib/databank-data";

export default function DatabankPage() {
  const [selected, setSelected] = useState<PlayerPick | null>(null);
  const [data, setData] = useState<DatabankData | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    if (!selected) return;
    let alive = true;
    // setState only in async callbacks (react-hooks/set-state-in-effect).
    Promise.resolve()
      .then(() => {
        if (!alive) return;
        setFetchError(null);
        setLoading(true);
        return fetchDatabank(selected);
      })
      .then((d) => {
        if (alive) setData(d ?? null);
      })
      .catch((e: unknown) => {
        if (alive) {
          setData(null);
          setFetchError(e instanceof Error ? e.message : "Failed to load player history.");
        }
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [selected, retryCount]);

  return (
    <>
      <main className="w-full flex-1 space-y-6 px-5 py-6">
        <header>
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
            Research · Databank
          </div>
          <h1 className="font-display text-3xl font-extrabold text-navy">Player Databank</h1>
          <p className="mt-1 text-[13px] text-ink-2">Look up any player&apos;s year-by-year history.</p>
        </header>

        <SearchPicker
          selected={selected}
          onPick={(p) => {
            setSelected(p);
            setRetryCount(0);
          }}
          onClear={() => {
            setSelected(null);
            setData(null);
            setFetchError(null);
            setRetryCount(0);
          }}
        />

        {loading ? (
          <Skeleton className="h-64 w-full rounded-2xl" />
        ) : selected && data ? (
          <SeasonsTable data={data} />
        ) : selected && fetchError ? (
          <Card className="p-10 text-center">
            <AlertTriangle className="mx-auto mb-2 size-7 text-heat" aria-hidden />
            <p className="text-[13px] font-semibold text-navy">Could not load player history</p>
            <p className="mt-1 text-[12px] text-ink-2">{fetchError}</p>
            <button
              onClick={() => setRetryCount((n) => n + 1)}
              className="mt-4 inline-flex min-h-9 items-center rounded-lg border border-line px-4 text-[13px] font-semibold text-navy transition-colors hover:bg-surface"
            >
              Retry
            </button>
          </Card>
        ) : !selected ? (
          <Card className="p-10 text-center">
            <Library className="mx-auto mb-2 size-7 text-ink-3" aria-hidden />
            <p className="text-[13px] text-ink-2">Search for a player above to see their multi-year stats.</p>
          </Card>
        ) : (
          <Card className="p-8 text-center">
            <p className="text-[13px] text-ink-2">No history found for this player.</p>
          </Card>
        )}
      </main>
      {selected && data && <Footer freshnessMinutes={30} />}
    </>
  );
}

function SearchPicker({
  selected,
  onPick,
  onClear,
}: {
  selected: PlayerPick | null;
  onPick: (p: PlayerPick) => void;
  onClear: () => void;
}) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<PlayerPick[]>([]);

  useEffect(() => {
    let alive = true;
    const id = setTimeout(() => {
      searchPlayers(q).then((r) => alive && setResults(r));
    }, 250);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [q]);

  if (selected) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-line bg-surface px-4 py-3">
        <PlayerAvatar mlbId={selected.mlbId} teamId={selected.teamId} name={selected.name} size={40} />
        <div className="min-w-0 flex-1">
          <div className="font-display text-base font-bold text-navy">{selected.name}</div>
          <div className="tnum text-[12px] text-ink-3">
            {selected.teamAbbr} · {selected.pos}
          </div>
        </div>
        <button
          onClick={onClear}
          className="inline-flex items-center gap-1 rounded-lg border border-line px-3 py-1.5 text-[12px] font-semibold text-ink-2 transition-colors hover:bg-surface-2 hover:text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-heat/50"
        >
          <X className="size-3.5" aria-hidden /> Change
        </button>
      </div>
    );
  }

  const shown = results.slice(0, 12);
  return (
    <div className="relative max-w-xl">
      <div className="flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2.5">
        <Search className="size-4 text-ink-3" aria-hidden />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search any player…"
          className="w-full bg-transparent text-[13px] text-navy outline-none placeholder:text-ink-3"
          aria-label="Search any player"
        />
      </div>
      {shown.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-lg border border-line bg-canvas shadow-[0_8px_24px_rgba(11,24,48,0.18)]">
          {shown.map((p) => (
            <li key={p.id}>
              <button
                onClick={() => {
                  onPick(p);
                  setQ("");
                  setResults([]);
                }}
                className="flex w-full items-center gap-2.5 border-b border-line/60 px-3 py-2 text-left transition-colors last:border-0 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
              >
                <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
                <span className="min-w-0">
                  <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
                  <span className="tnum block text-[10.5px] text-ink-3">
                    {p.teamAbbr} · {p.pos}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SeasonsTable({ data }: { data: DatabankData }) {
  const cols = databankColumns(data.seasons);
  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center gap-3 border-b border-line bg-surface px-4 py-3">
        <PlayerAvatar mlbId={data.player.mlbId} teamId={data.player.teamId} name={data.player.name} size={36} />
        <div>
          <div className="font-display text-base font-bold text-navy">{data.player.name}</div>
          <div className="tnum text-[12px] text-ink-3">
            {data.player.teamAbbr} · {data.player.pos}
          </div>
        </div>
      </div>
      {data.seasons.length === 0 ? (
        <p className="px-4 py-8 text-center text-[13px] text-ink-3">No season data available.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-[13px]">
            <thead>
              <tr className="border-b border-line text-[10.5px] font-bold uppercase tracking-wide text-ink-3">
                <th className="px-3 py-2 text-left">Year</th>
                {cols.map((c) => (
                  <th key={c} className="px-2 py-2 text-right">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tnum">
              {data.seasons.map((s) => (
                <tr key={s.year} className="border-b border-line/60 transition-colors last:border-0 hover:bg-surface/60">
                  <td className="px-3 py-2 text-left font-bold text-navy">{s.year}</td>
                  {cols.map((c) => (
                    <td key={c} className="px-2 py-2 text-right text-ink-2">
                      {formatDatabankValue(c, s.stats[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
