"use client";

import { useMemo, useState } from "react";
import { Flame } from "lucide-react";
import type { ProbCell, ProbablesData } from "@/lib/probables-data";
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

type FilterKey = "easy" | "tough" | "two" | "home" | "away" | "yours" | "taken" | "available";

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "easy", label: "Easy" },
  { key: "tough", label: "Tough" },
  { key: "two", label: "Two-start" },
  { key: "home", label: "Home" },
  { key: "away", label: "Away" },
  { key: "yours", label: "Yours" },
  { key: "taken", label: "Taken" },
  { key: "available", label: "Available" },
];

function matches(c: ProbCell, active: Set<FilterKey>): boolean {
  if (active.size === 0) return true;
  const diffOn = active.has("easy") || active.has("tough");
  const avOn = active.has("yours") || active.has("taken") || active.has("available");
  if (diffOn && !((active.has("easy") && c.band === "easy") || (active.has("tough") && c.band === "tough")))
    return false;
  if (active.has("two") && !c.twoStart) return false;
  if (active.has("home") && !c.isHome) return false;
  if (active.has("away") && c.isHome) return false;
  if (avOn && !active.has(c.availability)) return false;
  return true;
}

const AVAIL = {
  yours: { label: "YOURS", cls: "bg-heat/15 text-heat" },
  taken: { label: "TAKEN", cls: "bg-surface-2 text-ink-3" },
  available: { label: "FA", cls: "bg-ok/15 text-ok" },
} as const;

export function ProbableGrid({ data }: { data: ProbablesData }) {
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
            </tr>
          </thead>
          <tbody>
            {data.teams.map((row) => (
              <tr key={row.team} className="border-b border-line/60">
                <th
                  scope="row"
                  className="sticky left-0 z-10 bg-canvas px-3 py-2 text-left font-display text-sm font-extrabold text-navy"
                >
                  {row.team}
                </th>
                {row.cells.map((c, i) => (
                  <td key={i} className="p-1 align-top">
                    {c ? <Cell c={c} dim={!matches(c, active)} /> : <div className="py-3 text-center text-ink-3">—</div>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-ink-3">
        Cell color = matchup ease (hotter = easier to stream against). <span className="font-semibold text-heat">2×</span>{" "}
        = two-start week · YOURS / TAKEN / FA = league availability.
      </p>
    </div>
  );
}

function Cell({ c, dim }: { c: ProbCell; dim: boolean }) {
  const av = AVAIL[c.availability];
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
        {c.twoStart && (
          <span className="inline-flex items-center gap-0.5 rounded bg-heat px-1 font-bold text-white">
            <Flame className="size-2.5" aria-hidden />
            2×
          </span>
        )}
      </div>
      {c.pitcher ? (
        <PlayerLink player={c.pitcher} className="mt-0.5 block truncate text-[12px] leading-tight" />
      ) : (
        <span className="mt-0.5 block truncate text-[12px] text-ink-3">TBD</span>
      )}
      <div className="mt-1 flex items-center justify-between">
        <span className={cn("rounded px-1 py-px text-[9px] font-bold tracking-wide", av.cls)}>{av.label}</span>
        <span className="tnum text-[10px] font-semibold text-ink-3">{Math.round(c.difficulty)}</span>
      </div>
    </div>
  );
}
