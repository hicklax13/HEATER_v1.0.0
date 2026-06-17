"use client";

import type { LineupSlot, SlotStatus } from "@/lib/optimizer-data";
import { PlayerLink } from "@/components/player/PlayerLink";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { cn } from "@/lib/utils";

const STATUS: Record<SlotStatus, { label: string; cls: string }> = {
  start: { label: "Start", cls: "bg-ok/12 text-ok" },
  sit: { label: "Sit", cls: "bg-ember/12 text-ember" },
  bench: { label: "Bench", cls: "bg-surface-2 text-ink-2" },
  off: { label: "Off", cls: "bg-surface-2 text-ink-3" },
};

export function LineupTable({ slots }: { slots: LineupSlot[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[680px]">
        <thead>
          <tr className="border-b border-line">
            <Th left>Slot</Th>
            <Th left>Player</Th>
            <Th left>Matchup</Th>
            <Th left>Today&apos;s Proj</Th>
            <Th>Value</Th>
            <Th>Status</Th>
          </tr>
        </thead>
        <tbody className="text-[13px]">
          {slots.map((s, i) => {
            const st = STATUS[s.status];
            return (
              <tr key={i} className={cn("border-b border-line/60", s.status === "off" && "opacity-60")}>
                <td className="tnum px-2.5 py-2.5 font-bold text-navy">{s.slot}</td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center gap-2">
                    <PlayerAvatar mlbId={s.player.mlbId} teamId={s.player.teamId} name={s.player.name} size={26} />
                    <div className="min-w-0">
                      <PlayerLink player={s.player} className="text-[13px]" />
                      <div className="tnum text-[10.5px] text-ink-3">
                        {s.player.pos} · {s.player.teamAbbr}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="tnum px-2.5 py-2.5 text-ink-2">{s.matchup}</td>
                <td className="px-2.5 py-2.5">
                  <span className="tnum text-ink">{s.proj}</span>
                  {s.note && <div className="text-[10.5px] font-medium text-ember">{s.note}</div>}
                </td>
                <td className="px-2.5 py-2.5">
                  <ValueBar value={s.value} />
                </td>
                <td className="px-2.5 py-2.5 text-right">
                  <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold", st.cls)}>
                    {st.label}
                  </span>
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
