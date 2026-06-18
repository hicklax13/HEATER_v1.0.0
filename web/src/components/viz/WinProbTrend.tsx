import { LineChart } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { COLORS, heatColor } from "@/lib/tokens";

const SEASON_START = new Date(2026, 2, 23);
function weekDate(wk: number): string {
  const d = new Date(SEASON_START);
  d.setDate(d.getDate() + (wk - 1) * 7);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/**
 * Win-probability over the season (hand-rolled SVG area, Combustion-themed).
 * Line + fill take the current heat color so it reads in sync with the HeatGauge.
 */
export function WinProbTrend({ data }: { data: number[] }) {
  if (!data || data.length < 2) return null;
  const W = 760;
  const H = 210;
  const padX = 40;
  const padTop = 22;
  const padBottom = 40;
  const n = data.length;
  const plotB = H - padBottom;
  const x = (i: number) => padX + (i / (n - 1)) * (W - 2 * padX);
  const y = (v: number) => padTop + (1 - Math.max(0, Math.min(100, v)) / 100) * (plotB - padTop);

  const line = data.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${x(0).toFixed(1)},${plotB.toFixed(1)} ${line} ${x(n - 1).toFixed(1)},${plotB.toFixed(1)}`;
  const last = data[n - 1];
  const col = heatColor(last);
  const lx = x(n - 1);
  const ly = y(last);

  return (
    <Card className="p-5">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 font-display text-base font-bold text-navy">
          <LineChart className="size-4 text-heat" aria-hidden />
          Win Probability Trend
        </h2>
        <span className="text-[11px] text-ink-3">Weekly matchup win % · weeks 1–{n}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Win probability by week, currently ${last} percent.`}>
        <defs>
          <linearGradient id="wpt-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={col} stopOpacity={0.28} />
            <stop offset="100%" stopColor={col} stopOpacity={0.02} />
          </linearGradient>
        </defs>

        {/* y gridlines + labels (0/50/100) */}
        {[0, 50, 100].map((v) => (
          <g key={v}>
            <line
              x1={padX}
              y1={y(v)}
              x2={W - padX}
              y2={y(v)}
              stroke={COLORS.line}
              strokeWidth={1}
              strokeDasharray={v === 50 ? "3 3" : undefined}
            />
            <text x={padX - 8} y={y(v) + 3} textAnchor="end" className="tnum" fontSize="10" fill={COLORS.ink3}>
              {v}
            </text>
          </g>
        ))}

        <polyline points={area} fill="url(#wpt-fill)" stroke="none" />
        <polyline points={line} fill="none" stroke={col} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={lx} cy={ly} r="9" fill="none" stroke={col} strokeWidth="1.5" opacity={0.4} />
        <circle cx={lx} cy={ly} r="5" fill={col} />
        <text x={lx} y={ly - 14} textAnchor="end" className="tnum" fontSize="12" fontWeight="700" fill={col}>
          {last}%
        </text>

        {/* x-axis week dates */}
        {data.map((_, i) => (
          <text
            key={i}
            x={x(i)}
            y={plotB + 16}
            textAnchor="middle"
            className="tnum"
            fontSize="9"
            fill={COLORS.ink3}
          >
            {weekDate(i + 1)}
          </text>
        ))}
      </svg>
    </Card>
  );
}
