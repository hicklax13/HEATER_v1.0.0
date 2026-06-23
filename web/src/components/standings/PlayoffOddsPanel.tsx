"use client";

import Link from "next/link";
import { Trophy, Lock, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import type { StandingsData } from "@/lib/standings-data";

/** Monte-Carlo playoff picture: per-team playoff% bar + championship%, top-N cut
 *  line. Odds come from /api/playoff-odds (best-effort, ~2.4s sim, merged by
 *  applyPlayoffOdds); the empty-state note shows only when none are available. */
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
      {data.playoffOddsLocked ? (
        <div className="flex flex-col items-center gap-2 py-4 text-center">
          <span className="flex size-9 items-center justify-center rounded-full bg-heat/12 text-heat">
            <Lock className="size-4" aria-hidden />
          </span>
          <p className="text-[13px] font-semibold text-navy">Playoff odds are a Pro feature</p>
          <Link
            href="/pricing"
            className="inline-flex min-h-9 items-center gap-1.5 rounded-lg bg-gradient-to-b from-heat-bright to-heat px-4 text-[13px] font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
          >
            <Sparkles className="size-3.5" aria-hidden /> See Pro plans
          </Link>
        </div>
      ) : !anyOdds ? (
        <div className="text-[13px] text-ink-2">
          Playoff odds aren&apos;t available right now — they appear once the season simulation has run.
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
