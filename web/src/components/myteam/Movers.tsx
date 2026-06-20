"use client";

import { motion, useReducedMotion } from "framer-motion";
import { ArrowUp, ArrowDown } from "lucide-react";
import type { Mover, Scope } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { Sparkline } from "@/components/ui/Sparkline";
import { COLORS, MLB } from "@/lib/tokens";
import { teamBrand } from "@/lib/teams";
import { SPRING } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { PlayerDialog } from "@/components/player/PlayerDialog";

const SCOPE_LABEL: Record<Scope, string> = {
  mine: "Your Roster",
  league: "League-Wide",
  mixed: "Your Roster + League",
};

export function Movers({ movers, scope }: { movers: Mover[]; scope: Scope }) {
  return (
    <section aria-labelledby="movers-h">
      <div className="mb-3 flex items-center gap-2">
        <h2 id="movers-h" className="font-display text-sm font-bold uppercase tracking-wide text-navy">
          This Week&apos;s Movers
        </h2>
        <span className="rounded-md bg-surface px-2 py-0.5 text-[11px] font-semibold text-ink-2">
          {SCOPE_LABEL[scope]}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {movers.map((m, i) => (
          <MoverCard key={`${m.name}-${i}`} m={m} />
        ))}
      </div>
    </section>
  );
}

/* eslint-disable @next/next/no-img-element -- remote MLB team-logo watermark */
function TeamHaze({ teamId }: { teamId: number }) {
  const tb = teamBrand(teamId);
  return (
    <>
      {/* dark team-color gradient (lighter left → primary → deep right, like the reference cards) */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background: `linear-gradient(120deg, color-mix(in srgb, ${tb.primary} 80%, white) 0%, ${tb.primary} 50%, color-mix(in srgb, ${tb.primary} 68%, black) 128%)`,
        }}
      />
      {/* large tonal team-logo watermark on the right, bleeding off-edge (matches the reference cards) */}
      <img
        src={MLB.teamLogo(teamId)}
        alt=""
        aria-hidden
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
        style={{ filter: "brightness(0) invert(1) drop-shadow(0 1px 1px rgba(0,0,0,0.25))" }}
        className="pointer-events-none absolute right-0 top-1/2 size-96 -translate-y-1/2 translate-x-[20%] opacity-[0.22]"
      />
    </>
  );
}

function MoverCard({ m }: { m: Mover }) {
  const reduce = useReducedMotion();
  const up = m.trend === "up";
  const TrendIcon = up ? ArrowUp : ArrowDown;

  return (
    <PlayerDialog player={m}>
      <motion.button
        whileHover={reduce ? undefined : { y: -4 }}
        transition={SPRING}
        className="block w-full text-left"
        aria-label={`Open ${m.name} player card`}
      >
        <Card className="relative overflow-hidden p-6 text-center">
          <TeamHaze teamId={m.teamId} />
          <div className="relative mx-auto max-w-[180px] rounded-xl bg-white p-3 shadow-[0_2px_10px_rgba(8,16,32,0.18)]">
            <div className="flex justify-center">
              <PlayerAvatar mlbId={m.mlbId} teamId={m.teamId} name={m.name} size={44} />
            </div>
            <div className="mt-2 font-display text-[15px] font-bold text-navy">{m.name}</div>
            <div className="tnum text-[11px] font-semibold uppercase tracking-wide text-ink-3">
              {m.pos} · {m.teamAbbr}
            </div>
            <div className="mt-2 flex items-center justify-center gap-3">
              <Slot label={m.stat1.label} value={m.stat1.value} />
              <span className="h-7 w-px bg-line" />
              <Slot label={m.stat2.label} value={m.stat2.value} />
            </div>
            <div className="mt-1.5 text-[11px] font-medium text-ink-2">{m.context}</div>
            <div className="mt-2 flex items-center justify-center gap-2">
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold",
                  up ? "bg-ok/10 text-ok" : "bg-ember/10 text-ember",
                )}
              >
                <TrendIcon className="size-3" aria-hidden />
                {m.tag}
              </span>
              {m.spark && m.spark.length > 0 && (
                <Sparkline data={m.spark} color={up ? COLORS.ok : COLORS.ember} width={48} height={16} />
              )}
            </div>
            <div className="mt-2 flex items-center justify-center gap-2 border-t border-line pt-2 text-[10.5px] text-ink-3">
              {m.rosteredByYou && (
                <span className="rounded bg-heat/10 px-1.5 py-0.5 font-semibold text-heat">Yours</span>
              )}
              {m.ownPct !== undefined && <span className="tnum font-medium">{m.ownPct}% Rostered</span>}
            </div>
          </div>
        </Card>
      </motion.button>
    </PlayerDialog>
  );
}

function Slot({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="tnum font-display text-lg font-bold text-navy">{value}</div>
      <div className="tnum text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
    </div>
  );
}
