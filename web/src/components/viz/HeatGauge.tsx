"use client";

import { useId } from "react";
import { HeroNum } from "@/components/ui/HeroNum";
import { useCountUp } from "@/lib/motion";
import { heatColor } from "@/lib/tokens";

const START = 130;
const SWEEP = 280;

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}
function arc(cx: number, cy: number, r: number, a0: number, a1: number): string {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  const large = a1 - a0 <= 180 ? 0 : 1;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/**
 * Signature win-probability instrument. The arc + numeral color-shift cool→warm
 * along the heat ramp as the value changes, and the number counts up on mount.
 * The product's recognizable face — reused on Team + Matchup.
 */
export function HeatGauge({
  value,
  label = "Win Probability",
  size = 200,
  unit = "%",
}: {
  value: number; // 0..100
  label?: string;
  size?: number;
  unit?: string;
}) {
  const disp = useCountUp(value);
  const col = heatColor(disp);
  const id = useId().replace(/:/g, "");
  const valAngle = START + (disp / 100) * SWEEP;
  const h = Math.round(size * 0.875);

  return (
    <div className="relative" style={{ width: size, height: h }}>
      <svg width={size} height={h} viewBox="0 0 200 175" aria-hidden focusable="false">
        <defs>
          <linearGradient id={`hg-${id}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={col} stopOpacity="0.6" />
            <stop offset="1" stopColor={col} />
          </linearGradient>
        </defs>
        <path
          d={arc(100, 100, 80, START, START + SWEEP)}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d={arc(100, 100, 80, START, Math.max(START + 0.01, valAngle))}
          fill="none"
          stroke={`url(#hg-${id})`}
          strokeWidth="12"
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 12px ${col}99)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-1.5">
        <div className="leading-none" style={{ color: col, textShadow: `0 0 28px ${col}66` }}>
          <HeroNum width={70} style={{ fontSize: Math.round(size * 0.32) }}>
            {disp}
          </HeroNum>
          <span style={{ fontSize: Math.round(size * 0.14), fontWeight: 800 }}>{unit}</span>
        </div>
        <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/75">{label}</div>
      </div>
    </div>
  );
}
