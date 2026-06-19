"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerLink } from "@/components/player/PlayerLink";
import { StreamScorecard } from "@/components/viz/StreamScorecard";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import type { StreamCandidate } from "@/lib/streaming-data";

function TH({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th scope="col" className={cn("whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy", left ? "text-left" : "text-right")}>
      {children}
    </th>
  );
}

export function StreamBoard({ board }: { board: StreamCandidate[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px]">
          <thead>
            <tr className="border-b border-line">
              <TH left>#</TH><TH left>Pitcher</TH><TH left>Opp</TH><TH>Score</TH><TH>Conf</TH>
              <TH>GS</TH><TH>Net SGP</TH><TH>wRC+</TH><TH>K%</TH><TH>Park</TH><TH>xK</TH><TH>W%</TH><TH>Own%</TH><TH left>Risk</TH>
            </tr>
          </thead>
          <tbody className="tnum text-[13px]">
            {board.map((c) => {
              const expanded = open === c.rank;
              return (
                <Row key={c.rank} c={c} expanded={expanded} onToggle={() => setOpen(expanded ? null : c.rank)} />
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Row({ c, expanded, onToggle }: { c: StreamCandidate; expanded: boolean; onToggle: () => void }) {
  const col = heatColor(c.score);
  return (
    <>
      <tr className={cn("border-b border-line/60 transition-colors hover:bg-surface", !c.actionable && "opacity-50")}>
        <td className="px-2.5 py-2 text-left font-bold text-ink-3">{c.rank}</td>
        <td className="px-2.5 py-2 text-left">
          <PlayerLink player={c.player} />
          <span className="ml-1 text-[11px] text-ink-3">{c.player.teamAbbr}</span>
        </td>
        <td className="px-2.5 py-2 text-left text-ink-2">{c.isHome ? "vs " : "@ "}{c.opponent}</td>
        <td className="px-2.5 py-2 text-right">
          <HeroNum width={72} className="text-lg" style={{ color: col }}>{c.score}</HeroNum>
        </td>
        <td className="px-2.5 py-2 text-right uppercase text-ink-3">{c.confidence}</td>
        <td className="px-2.5 py-2 text-right">{c.numStarts === 2 ? <span className="rounded bg-flame/15 px-1.5 font-bold text-flame">2-start</span> : "1"}</td>
        <td className={cn("px-2.5 py-2 text-right font-semibold", c.netSgp >= 0 ? "text-ok" : "text-ember")}>{c.netSgp >= 0 ? "+" : ""}{c.netSgp.toFixed(2)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.oppWrcPlus}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.oppKpct.toFixed(1)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.park.toFixed(2)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.xK.toFixed(1)}</td>
        <td className="px-2.5 py-2 text-right text-ink">{c.winPct}</td>
        <td className="px-2.5 py-2 text-right text-ink-3">{c.ownPct}</td>
        <td className="px-2.5 py-2 text-left">
          <div className="flex items-center gap-1.5">
            {c.riskFlags.map((f) => (
              <span key={f} className="rounded bg-ember/10 px-1.5 py-0.5 text-[10px] font-bold text-ember">{f}</span>
            ))}
            <button onClick={onToggle} aria-label="Toggle why" className="ml-auto rounded p-1 text-ink-3 hover:text-heat">
              <ChevronDown className={cn("size-4 transition-transform", expanded && "rotate-180")} aria-hidden />
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-line/60 bg-surface">
          <td colSpan={14} className="px-4 py-4">
            <div className="flex flex-wrap items-start gap-6">
              <StreamScorecard score={c.score} components={c.components} size={140} />
              <p className="max-w-md flex-1 text-[13px] text-ink-2">{c.why} <span className="block mt-1 text-ink-3">Expected: {c.expectedLine}.</span></p>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
