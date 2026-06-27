"use client";

import { Card } from "@/components/ui/Card";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import type { StartSitCandidate } from "@/lib/start-sit-data";
import { cn } from "@/lib/utils";

/** One ranked Start/Sit candidate: heat bar + player + eligibility + projected
 *  line + per-category impact + matchup + reason. `started` (from the verdict)
 *  drives the START/SIT ribbon; not-playable cards grey out. */
export function CompareCard({ c, started }: { c: StartSitCandidate; started?: boolean }) {
  const score = Math.round(c.startScore);
  return (
    <Card className={cn("p-4", !c.playable && "opacity-60 grayscale")}>
      <div className="flex items-start gap-3">
        {/* rank chip */}
        <span className="tnum mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-surface-2 font-display text-[13px] font-bold text-navy">
          {c.rank}
        </span>

        <div className="min-w-0 flex-1">
          {/* player + ribbon */}
          <div className="flex items-center justify-between gap-2">
            <PlayerDialog player={c.player}>
              <button type="button" className="flex min-w-0 items-center gap-2 text-left">
                <PlayerAvatar mlbId={c.player.mlbId} teamId={c.player.teamId} name={c.player.name} size={32} />
                <span className="min-w-0">
                  <span className="block truncate text-[14px] font-semibold text-navy underline-offset-2 hover:text-heat hover:underline">
                    {c.player.name}
                  </span>
                  <span className="tnum block text-[10.5px] text-ink-3">
                    {c.player.teamAbbr || "—"} · {c.player.pos}
                  </span>
                </span>
              </button>
            </PlayerDialog>
            {started === undefined ? null : (
              <span
                className={cn(
                  "shrink-0 rounded-md px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide",
                  started ? "bg-ok/12 text-ok" : "bg-steel/12 text-steel",
                )}
              >
                {started ? "Start" : "Sit"}
              </span>
            )}
          </div>

          {/* heat bar */}
          <div className="mt-3 flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Start score</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
              <span className="block h-full rounded-full bg-heat" style={{ width: `${score}%` }} />
            </div>
            <span className="tnum w-7 text-right text-[13px] font-bold text-navy">{score}</span>
          </div>

          {/* eligibility chips */}
          {c.eligibleSlots.length > 0 && (
            <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Fills</span>
              {c.eligibleSlots.map((s) => (
                <span key={s} className="tnum rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-ink-2">
                  {s}
                </span>
              ))}
            </div>
          )}

          {/* projected line */}
          {c.projected.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-x-3 gap-y-1">
              {c.projected.map((s) => (
                <span key={s.label} className="text-[12px] text-ink-2">
                  <span className="tnum font-bold text-navy">{s.value}</span>{" "}
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-ink-3">{s.label}</span>
                </span>
              ))}
            </div>
          )}

          {/* category impact chips */}
          {c.categoryImpact.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {c.categoryImpact.map((s) => {
                const negative = s.value.trim().startsWith("-") || s.value.trim().startsWith("−");
                return (
                  <span
                    key={s.label}
                    className={cn(
                      "tnum rounded-md px-1.5 py-0.5 text-[10px] font-bold",
                      negative ? "bg-ember/12 text-ember" : "bg-ok/12 text-ok",
                    )}
                  >
                    {s.label} {s.value}
                  </span>
                );
              })}
            </div>
          )}

          {/* matchup + reason */}
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-2.5 text-[11.5px] text-ink-2">
            {c.matchup && <span className="tnum font-semibold text-navy">{c.matchup}</span>}
            {!c.playable && (
              <span className="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-ink-3">
                Not playable
              </span>
            )}
            {c.reason && <span className="text-ink-3">{c.reason}</span>}
          </div>
        </div>
      </div>
    </Card>
  );
}
