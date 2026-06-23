"use client";

import { useEffect, useState } from "react";
import { Search, X, Scale } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { searchPlayers, type PlayerPick } from "@/lib/player-search";
import {
  fetchCompare,
  bestIndexForCat,
  formatCatValue,
  participatesIn,
  type CompareData,
} from "@/lib/compare-data";
import { cn } from "@/lib/utils";

/**
 * Player Compare — search for any two players and see a head-to-head category
 * table. The picker uses /api/players/search (any player, rostered or FA);
 * /api/compare is pool-backed (real locally).
 */
export function ComparePanel() {
  const [selected, setSelected] = useState<PlayerPick[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlayerPick[]>([]);
  const [compare, setCompare] = useState<CompareData | null>(null);
  const [comparing, setComparing] = useState(false);

  // Debounced player search.
  useEffect(() => {
    let alive = true;
    const id = setTimeout(() => {
      searchPlayers(query).then((r) => alive && setResults(r));
    }, 250);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [query]);

  useEffect(() => {
    if (selected.length < 2) return; // render gate hides any stale comparison
    let alive = true;
    // setState only inside async callbacks (react-hooks/set-state-in-effect).
    Promise.resolve()
      .then(() => {
        if (!alive) return;
        setComparing(true);
        return fetchCompare(selected);
      })
      .then((d) => {
        if (alive && d) setCompare(d);
      })
      .catch(() => {
        if (alive) setCompare(null);
      })
      .finally(() => {
        if (alive) setComparing(false);
      });
    return () => {
      alive = false;
    };
  }, [selected]);

  const chosen = new Set(selected.map((p) => p.id));
  const filtered = results.filter((p) => !chosen.has(p.id)).slice(0, 12);

  const addPlayer = (p: PlayerPick) => setSelected((s) => (s.length >= 2 ? s : [...s, p]));
  const removePlayer = (id: number) => setSelected((s) => s.filter((p) => p.id !== id));

  return (
    <div className="space-y-5">
      {/* Selected slots */}
      <div className="grid grid-cols-2 gap-3">
        {[0, 1].map((i) => (
          <Slot key={i} player={selected[i]} index={i} onRemove={removePlayer} />
        ))}
      </div>

      {/* Picker — shown until two are chosen */}
      {selected.length < 2 && (
        <Card className="overflow-hidden p-0">
          <div className="flex items-center gap-2 border-b border-line bg-surface px-3 py-2.5">
            <Search className="size-4 text-ink-3" aria-hidden />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search any player to compare…"
              className="w-full bg-transparent text-[13px] text-navy outline-none placeholder:text-ink-3"
              aria-label="Search any player to compare"
            />
          </div>
          <ul className="max-h-80 overflow-y-auto">
            {filtered.length === 0 ? (
              <li className="px-4 py-6 text-center text-[13px] text-ink-3">
                {query.trim().length < 2 ? "Type a name to search players." : "No matching players."}
              </li>
            ) : (
              filtered.map((p) => (
                <li key={p.id}>
                  <button
                    onClick={() => addPlayer(p)}
                    className="flex w-full items-center gap-2.5 border-b border-line/60 px-3 py-2 text-left transition-colors duration-[var(--dur-1)] last:border-0 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
                  >
                    <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={30} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
                      <span className="tnum block text-[10.5px] text-ink-3">
                        {p.teamAbbr} · {p.pos}
                      </span>
                    </span>
                  </button>
                </li>
              ))
            )}
          </ul>
        </Card>
      )}

      {/* Comparison */}
      {selected.length === 2 &&
        (comparing ? (
          <Skeleton className="h-80 w-full rounded-2xl" />
        ) : compare && compare.categories.length > 0 ? (
          <CompareTable compare={compare} />
        ) : (
          <Card className="p-8 text-center">
            <Scale className="mx-auto mb-2 size-6 text-ink-3" aria-hidden />
            <p className="text-[13px] text-ink-2">No overlapping categories to compare for these two.</p>
          </Card>
        ))}
    </div>
  );
}

function Slot({
  player,
  index,
  onRemove,
}: {
  player: PlayerPick | undefined;
  index: number;
  onRemove: (id: number) => void;
}) {
  if (!player) {
    return (
      <div className="flex min-h-[60px] items-center justify-center rounded-xl border border-dashed border-line bg-surface/50 text-[12px] font-semibold text-ink-3">
        Pick player {index + 1}
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2.5 rounded-xl border border-line bg-surface px-3 py-2.5">
      <PlayerAvatar mlbId={player.mlbId} teamId={player.teamId} name={player.name} size={34} />
      <span className="min-w-0 flex-1">
        <PlayerDialog player={{ ...player }}>
          <button className="block truncate text-left text-[13.5px] font-bold text-navy hover:text-heat">
            {player.name}
          </button>
        </PlayerDialog>
        <span className="tnum block text-[10.5px] text-ink-3">
          {player.teamAbbr} · {player.pos}
        </span>
      </span>
      <button
        onClick={() => onRemove(player.id)}
        aria-label={`Remove ${player.name}`}
        className="flex size-7 shrink-0 items-center justify-center rounded-full text-ink-3 transition-colors hover:bg-surface-2 hover:text-ember focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-heat/50"
      >
        <X className="size-4" aria-hidden />
      </button>
    </div>
  );
}

function CompareTable({ compare }: { compare: CompareData }) {
  const [a, b] = compare.players;
  // Per-category: a value only counts if the player participates on that side
  // (a pure hitter has no real ERA, so don't award them a phantom 0.00 win). A
  // winner is crowned only when ≥2 players actually participate; otherwise the
  // non-participant shows "—" and neither side is highlighted.
  const rows = compare.categories.map((cat) => {
    const va = a && participatesIn(cat, a.stats) ? a.stats[cat] : undefined;
    const vb = b && participatesIn(cat, b.stats) ? b.stats[cat] : undefined;
    const enough = [va, vb].filter((v) => v !== undefined && Number.isFinite(v)).length >= 2;
    const best = enough ? bestIndexForCat(cat, [va, vb]) : -1;
    return { cat, va, vb, best };
  });
  const aWins = rows.filter((r) => r.best === 0).length;
  const bWins = rows.filter((r) => r.best === 1).length;

  return (
    <Card className="overflow-hidden p-0">
      {/* Header: the two players + the category tally */}
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 border-b border-line bg-surface px-4 py-3">
        <ComparePlayerHead player={a} align="left" wins={aWins} leads={aWins > bWins} />
        <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">vs</span>
        <ComparePlayerHead player={b} align="right" wins={bWins} leads={bWins > aWins} />
      </div>

      <table className="w-full text-[13px]">
        <thead className="sr-only">
          <tr>
            <th scope="col">{a?.name ?? "Player 1"}</th>
            <th scope="col">Category</th>
            <th scope="col">{b?.name ?? "Player 2"}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ cat, va, vb, best }) => (
            <tr key={cat} className="border-b border-line/60 last:border-0">
              <td
                className={cn(
                  "tnum px-4 py-2.5 text-right",
                  best === 0 ? "bg-heat/10 font-bold text-navy" : "text-ink-2",
                )}
                aria-label={`${a?.name ?? "Player 1"} ${cat}: ${va === undefined ? "—" : formatCatValue(cat, va)}${best === 0 ? " (better)" : ""}`}
              >
                {best === 0 && <span className="mr-1 text-[9px] text-heat" aria-hidden>▲</span>}
                {va === undefined ? "—" : formatCatValue(cat, va)}
              </td>
              <th scope="row" className="px-2 py-2.5 text-center text-[11px] font-bold uppercase tracking-wide text-ink-3">
                {cat}
              </th>
              <td
                className={cn(
                  "tnum px-4 py-2.5 text-left",
                  best === 1 ? "bg-heat/10 font-bold text-navy" : "text-ink-2",
                )}
                aria-label={`${b?.name ?? "Player 2"} ${cat}: ${vb === undefined ? "—" : formatCatValue(cat, vb)}${best === 1 ? " (better)" : ""}`}
              >
                {vb === undefined ? "—" : formatCatValue(cat, vb)}
                {best === 1 && <span className="ml-1 text-[9px] text-heat" aria-hidden>▲</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function ComparePlayerHead({
  player,
  align,
  wins,
  leads,
}: {
  player: CompareData["players"][number] | undefined;
  align: "left" | "right";
  wins: number;
  leads: boolean;
}) {
  if (!player) return <div />;
  const right = align === "right";
  return (
    <div className={cn("flex items-center gap-2.5", right && "flex-row-reverse text-right")}>
      <PlayerAvatar mlbId={player.mlbId} teamId={player.teamId} name={player.name} size={36} />
      <div className="min-w-0">
        <PlayerDialog player={{ ...player }}>
          <button className="block truncate font-display text-sm font-bold text-navy hover:text-heat">
            {player.name}
          </button>
        </PlayerDialog>
        <span
          className={cn(
            "tnum text-[11px] font-semibold",
            leads ? "text-heat" : "text-ink-3",
          )}
        >
          {wins} {wins === 1 ? "cat" : "cats"}
        </span>
      </div>
    </div>
  );
}
