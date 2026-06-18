"use client";

import { useEffect, useState } from "react";
import { animate, useMotionValue, useReducedMotion } from "framer-motion";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { Matchup } from "@/lib/types";
import { COLORS, heatColor } from "@/lib/tokens";
import { EASE_SNAP } from "@/lib/motion";
import { HexMesh } from "@/components/ui/HexMesh";

/* eslint-disable @next/next/no-img-element -- local SVG team crests */

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

export function WinHero({ matchup }: { matchup: Matchup }) {
  const reduce = useReducedMotion();
  const final = matchup.winPct;
  const mv = useMotionValue(0);
  const [animated, setAnimated] = useState(0);
  const [par, setPar] = useState({ x: 0, y: 0 });
  const disp = reduce ? final : animated;

  useEffect(() => {
    if (reduce) return;
    const controls = animate(mv, final, {
      duration: 1,
      ease: EASE_SNAP,
      onUpdate: (v) => setAnimated(Math.round(v)),
    });
    return () => controls.stop();
  }, [final, reduce, mv]);

  const col = heatColor(disp);
  const valAngle = START + (disp / 100) * SWEEP;
  const down = matchup.deltaVsLastWeek < 0;

  const onMove = (e: React.MouseEvent<HTMLElement>) => {
    if (reduce) return;
    const r = e.currentTarget.getBoundingClientRect();
    setPar({ x: ((e.clientX - r.left) / r.width - 0.5) * 8, y: ((e.clientY - r.top) / r.height - 0.5) * 8 });
  };

  return (
    <section
      onMouseMove={onMove}
      onMouseLeave={() => setPar({ x: 0, y: 0 })}
      aria-label={`Week matchup. Win probability ${final} percent, ${down ? "down" : "up"} ${Math.abs(matchup.deltaVsLastWeek)} percent versus last week.`}
      className="relative overflow-hidden rounded-2xl border border-white/10 text-chrome shadow-[0_24px_60px_rgba(9,20,42,0.35)]"
      style={{ background: `radial-gradient(130% 150% at 78% -10%, ${COLORS.navy700}, ${COLORS.navyDeep} 65%)` }}
    >
      <span
        aria-hidden
        className="pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent"
      />
      <HexMesh par={par} />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 bottom-0 h-2/3"
        style={{ background: `radial-gradient(60% 90% at 50% 130%, ${COLORS.heat}22, transparent 70%)` }}
      />

      <div className="relative grid items-center gap-4 p-6 md:grid-cols-[1fr_auto_1fr]">
        <Identity
          label="You"
          name={matchup.youName}
          record={matchup.youRecord}
          logo={matchup.youLogo ?? "/brand/team-logo-placeholder.svg"}
        />

        <div className="flex flex-col items-center">
          <div className="relative h-[175px] w-[200px]">
            <svg width="200" height="175" viewBox="0 0 200 175" aria-hidden focusable="false">
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
                stroke={col}
                strokeWidth="12"
                strokeLinecap="round"
                style={{ filter: `drop-shadow(0 0 10px ${col}88)` }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center pt-1.5">
              <div
                className="font-display font-extrabold leading-none"
                style={{ fontSize: 64, color: col, textShadow: `0 0 26px ${col}66` }}
              >
                <span className="tnum">{disp}</span>
                <span style={{ fontSize: 28 }}>%</span>
              </div>
              <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/75">
                Win Probability
              </div>
            </div>
          </div>

          <SplitBar
            win={matchup.winPct}
            tie={matchup.tiePct}
            loss={matchup.lossPct}
            proj={matchup.projLine}
            delta={matchup.deltaVsLastWeek}
          />
        </div>

        <Identity
          label="Opponent"
          name={matchup.oppName}
          record={matchup.oppRecord}
          logo={matchup.oppLogo ?? "/brand/team-logo-opponent.svg"}
          right
        />
      </div>
    </section>
  );
}

function Identity({
  label,
  name,
  record,
  logo,
  right,
}: {
  label: string;
  name: string;
  record: string;
  logo: string;
  right?: boolean;
}) {
  return (
    <div className={`flex flex-col ${right ? "items-end text-right" : "items-start"}`}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-white/65">{label}</div>
      {/* fantasy team logo (placeholder → Yahoo team_logos url when wired) */}
      <img
        src={logo}
        alt=""
        aria-hidden
        className="mt-2 size-14 rounded-full bg-white/5 object-cover shadow-[0_4px_14px_rgba(0,0,0,0.35)] ring-2 ring-white/20"
      />
      <div className="mt-2 font-display text-xl font-bold leading-tight text-chrome md:text-2xl">
        {name}
      </div>
      <div className="tnum mt-1 text-[14px] text-white/70">{record}</div>
    </div>
  );
}

function SplitBar({
  win,
  tie,
  loss,
  proj,
  delta,
}: {
  win: number;
  tie: number;
  loss: number;
  proj: string;
  delta: number;
}) {
  const down = delta < 0;
  const DeltaIcon = down ? ArrowDown : ArrowUp;
  return (
    <div className="group mt-4 w-full max-w-[260px]">
      <div
        className="flex h-2 overflow-hidden rounded-full"
        role="img"
        aria-label={`Win ${win} percent, tie ${tie} percent, loss ${loss} percent`}
      >
        <span className="block bg-heat" style={{ width: `${win}%` }} />
        <span className="block bg-steel" style={{ width: `${tie}%` }} />
        <span className="block bg-ember" style={{ width: `${loss}%` }} />
      </div>
      <div className="tnum mt-1.5 flex justify-between text-[11px] text-white/45 opacity-0 transition-opacity duration-[var(--dur-2)] group-hover:opacity-100">
        <span className="text-heat">Win {win}%</span>
        <span>Tie {tie}%</span>
        <span className="text-ember/90">Loss {loss}%</span>
      </div>
      <div
        className={`mt-1.5 flex items-center justify-center gap-1 font-display text-[12px] font-semibold ${down ? "text-ember" : "text-ok"}`}
      >
        <DeltaIcon className="size-3" aria-hidden />
        {Math.abs(delta)}% vs last week
      </div>
      <div className="tnum mt-1 text-center text-[11px] text-white/50">{proj}</div>
    </div>
  );
}
