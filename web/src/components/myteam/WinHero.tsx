"use client";

import { useState } from "react";
import { useReducedMotion } from "framer-motion";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { Matchup } from "@/lib/types";
import { COLORS } from "@/lib/tokens";
import { HexMesh } from "@/components/ui/HexMesh";
import { HeatGauge } from "@/components/viz/HeatGauge";

/* eslint-disable @next/next/no-img-element -- local SVG team crests */

export function WinHero({ matchup }: { matchup: Matchup }) {
  const reduce = useReducedMotion();
  const final = matchup.winPct;
  const [par, setPar] = useState({ x: 0, y: 0 });
  const hasDelta = typeof matchup.deltaVsLastWeek === "number";
  const down = (matchup.deltaVsLastWeek ?? 0) < 0;

  const onMove = (e: React.MouseEvent<HTMLElement>) => {
    if (reduce) return;
    const r = e.currentTarget.getBoundingClientRect();
    setPar({ x: ((e.clientX - r.left) / r.width - 0.5) * 8, y: ((e.clientY - r.top) / r.height - 0.5) * 8 });
  };

  return (
    <section
      onMouseMove={onMove}
      onMouseLeave={() => setPar({ x: 0, y: 0 })}
      aria-label={`Week matchup. Win probability ${final} percent${hasDelta ? `, ${down ? "down" : "up"} ${Math.abs(matchup.deltaVsLastWeek as number)} percent versus last week` : ""}.`}
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
          <HeatGauge value={final} />

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
  record?: string;
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
      {record && <div className="tnum mt-1 text-[14px] text-white/70">{record}</div>}
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
  proj?: string;
  delta?: number;
}) {
  const hasDelta = typeof delta === "number";
  const down = (delta ?? 0) < 0;
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
      {hasDelta && (
        <div
          className={`mt-1.5 flex items-center justify-center gap-1 font-display text-[12px] font-semibold ${down ? "text-ember" : "text-ok"}`}
        >
          <DeltaIcon className="size-3" aria-hidden />
          {Math.abs(delta as number)}% vs last week
        </div>
      )}
      {proj && <div className="tnum mt-1 text-center text-[11px] text-white/50">{proj}</div>}
    </div>
  );
}
