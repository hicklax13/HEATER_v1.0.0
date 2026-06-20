"use client";

import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import type { StandingsData } from "@/lib/standings-data";

/** Monte-Carlo playoff picture: per-team playoff% bar + championship%, top-N cut
 *  line. Odds aren't in /api/standings yet → graceful note on live data. */
export function PlayoffOddsPanel({ data }: { data: StandingsData }) {
  const teams = [...data.teams].sort((a, b) => b.playoffOdds - a.playoffOdds);
  const anyOdds = teams.some((t) => t.playoffOdds > 0);

  return (
    <Card className="p-5">
      <div className="mb-1 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Trophy className="size-4 text-heat" aria-hidden /> Playoff odds
      </div>
      <p className="mb-4 text-[12px] text-ink-3">
        Monte-Carlo playoff probability. Top {data.playoffSpots} make the playoffs.
      </p>
      {!anyOdds ? (
        <div className="text-[13px] text-ink-2">
          Playoff odds need the season-simulation endpoint (a backend add) — not available on live data yet.
        </div>
      ) : (
        <div className="space-y-2">
          {teams.map((t, i) => (
            <div
              key={t.teamName}
              className={cn(
                "flex items-center gap-3",
                i + 1 === data.playoffSpots && "border-b border-dashed border-heat/40 pb-3",
              )}
            >
              <span className={cn("w-32 shrink-0 truncate text-[13px] font-semibold", t.isUser ? "text-heat" : "text-navy")}>
                {t.teamName}
                {t.isUser && " (you)"}
              </span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                <span
                  className="block h-full rounded-full"
                  style={{ width: `${t.playoffOdds}%`, background: heatColor(t.playoffOdds) }}
                />
              </div>
              <span className="tnum w-10 shrink-0 text-right text-[12px] font-bold" style={{ color: heatColor(t.playoffOdds) }}>
                {t.playoffOdds}%
              </span>
              {/* champ% deferred — per-team championship odds aren't engine-derivable yet */}
              {t.champOdds > 0 && (
                <span className="tnum w-20 shrink-0 text-right text-[11px] text-ink-3" title="Championship odds">
                  {t.champOdds}% champ
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
