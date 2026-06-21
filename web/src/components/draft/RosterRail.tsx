"use client";

import { RotateCcw } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { cn } from "@/lib/utils";
import type { DraftPlayer, DraftPick } from "@/lib/draft-data";

/** Sidebar: the user's drafted roster + a recent-picks feed (user highlighted)
 *  + a reset control. */
export function RosterRail({
  myRoster,
  pickLog,
  userTeamIndex,
  onReset,
}: {
  myRoster: DraftPlayer[];
  pickLog: DraftPick[];
  userTeamIndex: number;
  onReset: () => void;
}) {
  const recent = [...pickLog].slice(-12).reverse();
  return (
    <aside className="space-y-4">
      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-[12px] font-bold uppercase tracking-wider text-navy">My Roster</span>
          <span className="tnum text-[11px] font-semibold text-ink-3">{myRoster.length}</span>
        </div>
        {myRoster.length === 0 ? (
          <p className="text-[12px] text-ink-3">No picks yet — draft your first player.</p>
        ) : (
          <ul className="space-y-2">
            {myRoster.map((p, i) => (
              <li key={`${p.id}-${i}`} className="flex items-center gap-2">
                <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={26} />
                <PlayerLink player={p} className="min-w-0 flex-1 truncate text-[13px]" />
                <span className="shrink-0 text-[10px] font-bold uppercase text-ink-3">{p.pos}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card className="p-4">
        <div className="mb-3 text-[12px] font-bold uppercase tracking-wider text-navy">Recent Picks</div>
        {recent.length === 0 ? (
          <p className="text-[12px] text-ink-3">Waiting on the first pick…</p>
        ) : (
          <ul className="space-y-1">
            {recent.map((pk) => {
              const mine = pk.teamIndex === userTeamIndex;
              return (
                <li
                  key={pk.pick}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2 py-1 text-[12px]",
                    mine && "bg-heat/10",
                  )}
                >
                  <span className="tnum w-8 shrink-0 text-[10px] font-bold text-ink-3">#{pk.pick + 1}</span>
                  <span className="min-w-0 flex-1 truncate font-semibold text-navy">{pk.playerName}</span>
                  <span className="shrink-0 text-[10px] font-bold text-ink-3">
                    {mine ? "You" : `T${pk.teamIndex + 1}`}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      <button
        onClick={onReset}
        className="inline-flex min-h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-line px-4 py-2 text-[13px] font-semibold text-ink-2 transition-colors hover:bg-surface"
      >
        <RotateCcw className="size-3.5" aria-hidden /> Reset Draft
      </button>
    </aside>
  );
}
