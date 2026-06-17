"use client";

import { LineChart } from "lucide-react";
import type { TrajectoryPoint } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { COLORS } from "@/lib/tokens";

// Fantasy week 1 Monday → "M/D" label. 2026 MLB season opens late March.
const SEASON_START = new Date(2026, 2, 23);
function weekDate(wk: number): string {
  const d = new Date(SEASON_START);
  d.setDate(d.getDate() + (wk - 1) * 7);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

export function SeasonTrajectory({
  points,
  playoffCut,
}: {
  points: TrajectoryPoint[];
  playoffCut: number;
}) {
  const W = 760;
  const H = 240;
  const padX = 44;
  const padTop = 26;
  const padBottom = 46; // room for the date axis
  const weeks = points.length;
  const maxRank = 12;
  const plotBottom = H - padBottom;
  const x = (wk: number) => padX + ((wk - 1) / (weeks - 1)) * (W - 2 * padX);
  const y = (rank: number) => padTop + ((rank - 1) / (maxRank - 1)) * (plotBottom - padTop);

  const line = points.map((p) => `${x(p.week).toFixed(1)},${y(p.rank).toFixed(1)}`).join(" ");
  const area = `${x(1).toFixed(1)},${plotBottom.toFixed(1)} ${line} ${x(weeks).toFixed(1)},${plotBottom.toFixed(1)}`;
  const last = points[points.length - 1];
  const gridRanks = [1, 4, 8, 12];

  return (
    <Card className="p-5">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 font-display text-base font-bold text-navy">
          <LineChart className="size-4 text-heat" aria-hidden />
          Season Trajectory
        </h2>
        <span className="text-[11px] text-ink-3">League rank · weeks 1–{weeks}</span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`League rank by week. Currently ${last.rank}th of ${maxRank}. Playoff cut is the top ${playoffCut}.`}
      >
        {/* playoff-cut band (ranks 1..cut) */}
        <rect
          x={padX}
          y={y(1) - 8}
          width={W - 2 * padX}
          height={y(playoffCut) - y(1) + 8}
          fill={COLORS.ok}
          opacity={0.08}
        />
        <text x={W - padX} y={y(1) - 11} textAnchor="end" className="tnum" fontSize="8.5" fill={COLORS.ok}>
          Playoff cut · top {playoffCut}
        </text>

        {/* hairline grid + rank labels */}
        {gridRanks.map((rk) => (
          <g key={rk}>
            <line x1={padX} y1={y(rk)} x2={W - padX} y2={y(rk)} stroke={COLORS.line} strokeWidth="1" />
            <text x={padX - 8} y={y(rk) + 4} textAnchor="end" className="tnum" fontSize="11" fill={COLORS.ink3}>
              {rk === 1 ? "1st" : `${rk}th`}
            </text>
          </g>
        ))}

        {/* date axis (one tick per week) */}
        {points.map((p) => (
          <text
            key={`d${p.week}`}
            x={x(p.week)}
            y={plotBottom + 18}
            textAnchor="middle"
            className="tnum"
            fontSize="9"
            fill={COLORS.ink3}
          >
            {weekDate(p.week)}
          </text>
        ))}

        {/* area + line (navy with orange end-marker, per brief) */}
        <polyline points={area} fill={COLORS.navy} opacity={0.05} stroke="none" />
        <polyline
          points={line}
          fill="none"
          stroke={COLORS.navy}
          strokeWidth="2.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <circle cx={x(last.week)} cy={y(last.rank)} r="5" fill={COLORS.heat} />
        <circle cx={x(last.week)} cy={y(last.rank)} r="9" fill="none" stroke={COLORS.heat} strokeWidth="1.5" opacity={0.4} />
        <text
          x={x(last.week)}
          y={y(last.rank) - 14}
          textAnchor="end"
          fontSize="8.5"
          fontWeight="600"
          fill={COLORS.heat}
        >
          You are here · {last.rank}th
        </text>
      </svg>
    </Card>
  );
}
