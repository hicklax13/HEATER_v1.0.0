"use client";

import type { LineupSlot, SlotStatus } from "@/lib/optimizer-data";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { cn } from "@/lib/utils";

// Yahoo display order. A player is placed by its CURRENT Yahoo slot (currentSlot);
// when that is unknown (no live Yahoo) we fall back to the recommended slot.
const BATTER_ORDER = ["C", "1B", "2B", "3B", "SS", "OF", "Util", "BN", "IL"];
const PITCHER_ORDER = ["SP", "RP", "P", "BN", "IL"];
const PITCHER_SLOTS = new Set(["SP", "RP", "P"]);
const IL_SLOTS = new Set(["IL", "IL10", "IL15", "IL60", "NA", "DTD"]);

const STATUS: Record<SlotStatus, { label: string; cls: string }> = {
  start: { label: "Start", cls: "bg-ok/12 text-ok" },
  sit: { label: "Bench", cls: "bg-steel/12 text-steel" },
  bench: { label: "Bench", cls: "bg-steel/12 text-steel" },
  off: { label: "IL/Off", cls: "bg-surface-2 text-ink-3" },
};

function normSlot(raw: string): string {
  const s = (raw || "").toUpperCase().trim();
  if (IL_SLOTS.has(s)) return "IL";
  return s || "BN";
}

function slotKey(slot: LineupSlot): string {
  // group by the player's CURRENT Yahoo slot; fall back to the recommended slot
  return normSlot(slot.currentSlot || slot.slot);
}

function isPitcher(slot: LineupSlot): boolean {
  const key = slotKey(slot);
  if (PITCHER_SLOTS.has(key)) return true;
  // BN/IL pitchers: classify by the player's positions
  const pos = (slot.player.pos || "").toUpperCase();
  return /\b(SP|RP|P)\b/.test(pos);
}

function orderIndex(order: string[], key: string): number {
  const i = order.indexOf(key);
  return i === -1 ? order.length : i;
}

function groupAndSort(slots: LineupSlot[], order: string[]): LineupSlot[] {
  return [...slots].sort((a, b) => {
    const ia = orderIndex(order, slotKey(a));
    const ib = orderIndex(order, slotKey(b));
    if (ia !== ib) return ia - ib;
    return (b.value ?? 0) - (a.value ?? 0); // within a slot, best value first
  });
}

/** Render the full roster (starters ∪ bench) grouped by Yahoo slot, batters/pitchers split. */
export function LineupTable({ slots }: { slots: LineupSlot[] }) {
  const batters = groupAndSort(slots.filter((s) => !isPitcher(s)), BATTER_ORDER);
  const pitchers = groupAndSort(slots.filter(isPitcher), PITCHER_ORDER);
  return (
    <div className="space-y-5">
      {batters.length > 0 && <SlotSection title="Batters" rows={batters} />}
      {pitchers.length > 0 && <SlotSection title="Pitchers" rows={pitchers} />}
    </div>
  );
}

function SlotSection({ title, rows }: { title: string; rows: LineupSlot[] }) {
  return (
    <div>
      <div className="mb-2 text-[11px] font-bold uppercase tracking-wide text-ink-3">{title}</div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px]">
          <thead>
            <tr className="border-b border-line">
              <Th left>Slot</Th>
              <Th left>Player</Th>
              <Th left>Eligibility</Th>
              <Th>Value</Th>
              <Th left>Matchup</Th>
              <Th>Decision</Th>
            </tr>
          </thead>
          <tbody className="text-[13px]">
            {rows.map((s, i) => (
              <Row key={i} s={s} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function rowTone(s: LineupSlot): string {
  if (s.status === "off") return "bg-surface-2/40"; // IL / off → gray
  if (s.forcedStart) return "bg-heat/8"; // forced start → orange tint
  if (s.status === "start") return "bg-ok/6"; // start → green tint
  return "bg-steel/6"; // bench → blue/steel tint
}

function Row({ s }: { s: LineupSlot }) {
  const st = s.forcedStart ? { label: "Start", cls: "bg-heat/12 text-heat" } : STATUS[s.status];
  const swapHint = s.currentSlot && s.status === "start" && normSlot(s.currentSlot) !== normSlot(s.slot);
  return (
    <tr
      className={cn(
        "border-b border-line/60 transition-colors duration-[var(--dur-1)] hover:bg-surface/60",
        rowTone(s),
      )}
    >
      <td className="tnum px-2.5 py-2.5 font-bold text-navy">{slotKey(s)}</td>
      <td className="p-0">
        <PlayerDialog player={s.player}>
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded px-2.5 py-2.5 text-left transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
          >
            <PlayerAvatar mlbId={s.player.mlbId} teamId={s.player.teamId} name={s.player.name} size={26} />
            <span className="min-w-0">
              <span className="block text-[13px] font-semibold text-navy">{s.player.name}</span>
              <span className="tnum block text-[10.5px] text-ink-3">{s.player.teamAbbr ?? ""}</span>
            </span>
          </button>
        </PlayerDialog>
      </td>
      <td className="tnum px-2.5 py-2.5 text-ink-2">{s.player.pos}</td>
      <td className="px-2.5 py-2.5">
        <ValueBar value={s.value} />
      </td>
      <td className="tnum px-2.5 py-2.5 text-ink-2">{s.matchup || "—"}</td>
      <td className="px-2.5 py-2.5 text-right">
        <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold", st.cls)}>{st.label}</span>
        {swapHint && (
          <div className="mt-0.5 text-[10px] font-semibold text-heat">→ {normSlot(s.slot)}</div>
        )}
      </td>
    </tr>
  );
}

function Th({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy",
        left ? "text-left" : "text-right",
      )}
    >
      {children}
    </th>
  );
}

function ValueBar({ value }: { value: number }) {
  return (
    <div className="flex items-center justify-end gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
        <span className="block h-full rounded-full bg-heat" style={{ width: `${value}%` }} />
      </div>
      <span className="tnum w-6 text-right text-[12px] font-semibold text-navy">{value}</span>
    </div>
  );
}
