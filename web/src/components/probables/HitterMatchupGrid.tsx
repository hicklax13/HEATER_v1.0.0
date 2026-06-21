"use client";

import { useMemo, useState } from "react";
import type { HitterCell, HitterMatchupsData } from "@/lib/hitter-matchups-data";
import { heatColor } from "@/lib/tokens";
import { PlayerLink } from "@/components/player/PlayerLink";
import { cn } from "@/lib/utils";

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function formatDay(iso: string): { wd: string; md: string } {
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return { wd: "", md: iso };
  const dt = new Date(Date.UTC(y, m - 1, d));
  return { wd: WEEKDAYS[dt.getUTCDay()] ?? "", md: `${m}/${d}` };
}

type FilterKey = "easy" | "tough" | "home" | "away" | "yours";

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "easy", label: "Easy" },
  { key: "tough", label: "Tough" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
  { key: "yours", label: "Yours" },
];

function rowMatches(row: HitterMatchupsData["teams"][number], active: Set<FilterKey>): boolean {
  if (active.has("yours") && row.availability !== "yours") return false;
  return true;
}

function cellMatches(c: HitterCell, active: Set<FilterKey>): boolean {
  if (active.size === 0) return true;
  const diffOn = active.has("easy") || active.has("tough");
  if (diffOn && !((active.has("easy") && c.band === "easy") || (active.has("tough") && c.band === "tough")))
    return false;
  if (active.has("home") && !c.isHome) return false;
  if (active.has("away") && c.isHome) return false;
  return true;
}

export function HitterMatchupGrid({ data }: { data: HitterMatchupsData }) {
  const [active, setActive] = useState<Set<FilterKey>>(new Set());
  const days = useMemo(() => data.days.map(formatDay), [data.days]);

  const toggle = (k: FilterKey) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  return (
    <div className="space-y-4">
      {/* filter chips */}
      <div className="flex flex-wrap items-center gap-1.5">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => toggle(f.key)}
            aria-pressed={active.has(f.key)}
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-semibold transition-colors",
              active.has(f.key)
                ? "border-heat bg-heat text-white"
                : "border-line bg-canvas text-ink-2 hover:bg-surface",
            )}
          >
            {f.label}
          </button>
        ))}
        {active.size > 0 && (
          <button
            type="button"
            onClick={() => setActive(new Set())}
            className="ml-1 text-xs font-semibold text-ink-3 underline-offset-2 hover:text-heat hover:underline"
          >
            Clear
          </button>
        )}
      </div>

      <div className="overflow-x-auto rounded-2xl border border-line bg-canvas">
        <table className="w-full min-w-[860px] border-collapse">
          <thead>
            <tr className="border-b border-line bg-surface">
              <th className="sticky left-0 z-10 bg-surface px-3 py-2 text-left text-[11px] font-bold uppercase tracking-wide text-ink-3">
                Team
              </th>
              {days.map((d, i) => (
                <th key={i} className="px-2 py-2 text-center text-[11px] font-bold text-navy">
                  <div className="uppercase tracking-wide text-ink-3">{d.wd}</div>
                  <div className="tnum">{d.md}</div>
                </th>
              ))}
              <th className="px-3 py-2 text-right text-[11px] font-bold uppercase tracking-wide text-ink-3">
                Totals
              </th>
              <th className="px-3 py-2 text-right text-[11px] font-bold uppercase tracking-wide text-ink-3">
                Rank
              </th>
            </tr>
          </thead>
          <tbody>
            {data.teams.map((row) => {
              const dimRow = !rowMatches(row, active);
              return (
                <tr key={row.team} className={cn("border-b border-line/60", dimRow && "opacity-30")}>
                  <th
                    scope="row"
                    className="sticky left-0 z-10 bg-canvas px-3 py-2 text-left font-display text-sm font-extrabold text-navy"
                  >
                    {row.team}
                    {row.availability === "yours" && (
                      <span className="ml-1.5 rounded bg-heat/15 px-1 py-px text-[9px] font-bold tracking-wide text-heat">
                        YOURS
                      </span>
                    )}
                  </th>
                  {row.cells.map((c, i) => (
                    <td key={i} className="p-1 align-top">
                      {c ? (
                        <Cell c={c} dim={!cellMatches(c, active)} />
                      ) : (
                        <div className="py-3 text-center text-ink-3">—</div>
                      )}
                    </td>
                  ))}
                  <td className="px-3 py-2 text-right">
                    <div className="tnum text-[11px] text-ink-2">
                      <span className="font-semibold">G{row.totals.games}</span>
                      <span className="mx-0.5 text-ink-3">·</span>
                      <span className="text-navy">R{row.totals.vsRhp}</span>
                      <span className="mx-0.5 text-ink-3">·</span>
                      <span className="text-ok">L{row.totals.vsLhp}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className="tnum inline-flex items-center justify-center rounded-full bg-surface px-2 py-px text-[11px] font-bold text-navy">
                      #{row.matchupsRank}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-ink-3">
        Cell color = matchup ease for your bats (hotter = better matchup for your bats). R / L = opposing starter&apos;s
        handedness. Rank #1 = easiest weekly hitting schedule.
      </p>
    </div>
  );
}

function Cell({ c, dim }: { c: HitterCell; dim: boolean }) {
  const tint = heatColor(c.difficulty);
  return (
    <div
      className={cn(
        "min-w-[96px] rounded-lg border border-line px-2 py-1.5 transition-opacity",
        dim && "opacity-30",
      )}
      style={{ background: `color-mix(in srgb, ${tint} 14%, white)` }}
    >
      <div className="flex items-center justify-between text-[10px] font-semibold text-ink-2">
        <span>
          {c.isHome ? "vs" : "@"} {c.opponent}
        </span>
        {c.oppSpThrows && (
          <span
            className={cn(
              "rounded px-1 font-bold",
              c.oppSpThrows === "L"
                ? "bg-ok/15 text-ok"
                : "bg-surface-2 text-ink-3",
            )}
          >
            {c.oppSpThrows}
          </span>
        )}
      </div>
      {c.oppSp ? (
        <PlayerLink player={c.oppSp} className="mt-0.5 block truncate text-[12px] leading-tight" />
      ) : (
        <span className="mt-0.5 block truncate text-[12px] text-ink-3">TBD</span>
      )}
      <div className="mt-1 flex items-center justify-end">
        <span className="tnum text-[10px] font-semibold text-ink-3">{Math.round(c.difficulty)}</span>
      </div>
    </div>
  );
}
