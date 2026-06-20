"use client";

import { Fragment } from "react";
import { Card } from "@/components/ui/Card";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import { CATEGORIES, type StandingsData } from "@/lib/standings-data";

/** Best rank (1) → hot, worst (n) → cold. */
function rankColor(rank: number, n: number): string {
  if (!rank) return "var(--color-surface-2, #eef0f3)";
  const pct = 100 - ((rank - 1) / Math.max(1, n - 1)) * 100;
  return heatColor(pct);
}

function TH({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" | "center" }) {
  return (
    <th
      className={cn(
        "whitespace-nowrap px-2 py-2 text-[10px] font-bold uppercase tracking-wide text-navy",
        align === "right" && "text-right",
        align === "center" && "text-center text-ink-3",
      )}
    >
      {children}
    </th>
  );
}

/** Standings table with a per-category league-rank heatmap + playoff cut line. */
export function StandingsTable({ data }: { data: StandingsData }) {
  const n = data.teams.length;
  const anyOdds = data.teams.some((t) => t.playoffOdds > 0);
  const colSpan = 4 + CATEGORIES.length + (anyOdds ? 1 : 0);

  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] text-[13px]">
          <thead>
            <tr className="border-b border-line">
              <TH>#</TH>
              <TH>Team</TH>
              <TH align="right">Rec</TH>
              <TH align="right">Pts</TH>
              {CATEGORIES.map((c) => (
                <TH key={c} align="center">
                  {c}
                </TH>
              ))}
              {anyOdds && <TH align="right">Playoff</TH>}
            </tr>
          </thead>
          <tbody className="tnum">
            {data.teams.map((t) => (
              <Fragment key={t.teamName}>
                <tr className={cn("border-b border-line/60 transition-colors hover:bg-surface", t.isUser && "bg-heat/5")}>
                  <td className="px-2 py-2 text-left font-bold text-ink-3">{t.rank}</td>
                  <td className={cn("px-2 py-2 text-left font-semibold", t.isUser ? "text-heat" : "text-navy")}>
                    {t.teamName}
                    {t.isUser && (
                      <span className="ml-1.5 rounded bg-heat px-1 py-0.5 text-[9px] font-bold text-white">YOU</span>
                    )}
                  </td>
                  <td className="px-2 py-2 text-right text-ink-2">
                    {t.wins}-{t.losses}
                    {t.ties ? `-${t.ties}` : ""}
                  </td>
                  <td className="px-2 py-2 text-right font-semibold text-navy">{t.points}</td>
                  {CATEGORIES.map((c) => {
                    const r = t.categoryRanks[c];
                    return (
                      <td key={c} className="px-0.5 py-1 text-center">
                        <span
                          className="inline-flex h-5 w-6 items-center justify-center rounded text-[10px] font-bold text-white"
                          style={{ background: rankColor(r, n) }}
                          title={`${c}: ${r ? `#${r}` : "—"}`}
                        >
                          {r || "—"}
                        </span>
                      </td>
                    );
                  })}
                  {anyOdds && (
                    <td className="px-2 py-2 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <div className="h-1.5 w-12 overflow-hidden rounded-full bg-surface-2">
                          <span className="block h-full rounded-full bg-heat" style={{ width: `${t.playoffOdds}%` }} />
                        </div>
                        <span className="text-[11px] font-semibold text-ink-2">{t.playoffOdds}%</span>
                      </div>
                    </td>
                  )}
                </tr>
                {t.rank === data.playoffSpots && (
                  <tr aria-hidden>
                    <td colSpan={colSpan} className="p-0">
                      <div className="border-b-2 border-dashed border-heat/50 bg-heat/5 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-heat">
                        Playoff cut line
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
