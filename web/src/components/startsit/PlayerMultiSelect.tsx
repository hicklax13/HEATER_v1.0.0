"use client";

import { useEffect, useState } from "react";
import { Search, X } from "lucide-react";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { searchPlayers, type PlayerPick } from "@/lib/player-search";
import { cn } from "@/lib/utils";

const MAX_PLAYERS = 6;

/**
 * Multi-select up to 6 players (ANY position, roster OR free agent). Search box
 * (debounced searchPlayers → /api/players/search) + selected chips. Each result
 * click adds (when < 6 and not already selected); chips remove with ✕.
 */
export function PlayerMultiSelect({
  selected,
  onChange,
}: {
  selected: PlayerPick[];
  onChange: (next: PlayerPick[]) => void;
}) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<PlayerPick[]>([]);

  // Debounce the query; guard against stale results with an `alive` flag.
  useEffect(() => {
    let alive = true;
    const id = setTimeout(() => {
      searchPlayers(q).then((r) => alive && setResults(r));
    }, 200);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [q]);

  const chosen = new Set(selected.map((p) => p.id));
  const full = selected.length >= MAX_PLAYERS;
  const shown = results.filter((p) => !chosen.has(p.id)).slice(0, 12);

  const add = (p: PlayerPick) => {
    if (full || chosen.has(p.id)) return;
    onChange([...selected, p]);
    setQ("");
    setResults([]);
  };
  const remove = (id: number) => onChange(selected.filter((p) => p.id !== id));

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">
          Players to compare
        </div>
        <div className="tnum text-[11px] font-semibold text-ink-3">
          {selected.length}/{MAX_PLAYERS}
        </div>
      </div>

      {/* selected chips */}
      <div className="mb-2 flex flex-wrap gap-1.5">
        {selected.length === 0 && (
          <div className="rounded-lg border border-dashed border-line bg-surface/50 px-3 py-2 text-[12px] text-ink-3">
            Pick at least 2 players (roster or free agents).
          </div>
        )}
        {selected.map((p) => (
          <div
            key={p.id}
            className="flex items-center gap-2 rounded-full border border-line bg-surface py-1 pl-1 pr-1.5"
          >
            <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={24} />
            <span className="min-w-0">
              <span className="block truncate text-[12.5px] font-semibold text-navy">{p.name}</span>
              <span className="tnum block text-[10px] text-ink-3">
                {p.teamAbbr} · {p.pos}
              </span>
            </span>
            <button
              onClick={() => remove(p.id)}
              aria-label={`Remove ${p.name}`}
              className="flex size-5 shrink-0 items-center justify-center rounded-full text-ink-3 transition-colors hover:bg-surface-2 hover:text-ember focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-heat/50"
            >
              <X className="size-3" aria-hidden />
            </button>
          </div>
        ))}
      </div>

      {/* search box + results */}
      <div className="relative max-w-md">
        <div
          className={cn(
            "flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5",
            full && "opacity-50",
          )}
        >
          <Search className="size-3.5 text-ink-3" aria-hidden />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            disabled={full}
            placeholder={full ? "Maximum 6 players selected" : "Search players…"}
            className="w-full bg-transparent text-[12.5px] text-navy outline-none placeholder:text-ink-3 disabled:cursor-not-allowed"
            aria-label="Search players to compare"
          />
        </div>
        {!full && shown.length > 0 && (
          <ul className="absolute z-20 mt-1 max-h-60 w-full overflow-y-auto rounded-lg border border-line bg-canvas shadow-[0_8px_24px_rgba(11,24,48,0.18)]">
            {shown.map((p) => (
              <li key={p.id}>
                <button
                  onClick={() => add(p)}
                  className="flex w-full items-center gap-2 border-b border-line/60 px-2.5 py-1.5 text-left transition-colors last:border-0 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
                >
                  <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={24} />
                  <span className="min-w-0">
                    <span className="block truncate text-[12.5px] font-semibold text-navy">{p.name}</span>
                    <span className="tnum block text-[10px] text-ink-3">
                      {p.teamAbbr} · {p.pos}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
