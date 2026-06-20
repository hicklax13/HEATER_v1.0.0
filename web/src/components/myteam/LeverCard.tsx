"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { Zap, ArrowRight } from "lucide-react";
import type { Pickup } from "@/lib/types";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";

/**
 * The single primary CTA of the view (the one strong orange). Heat left-rail
 * ties it to the SB row in the Category Outlook; hovering pulses that row.
 */
export function LeverCard({
  behindBy,
  pickups,
  onHoverChange,
}: {
  headline: string;
  behindBy: number;
  pickups: Pickup[];
  onHoverChange?: (v: boolean) => void;
}) {
  return (
    <section
      aria-labelledby="lever-h"
      onMouseEnter={() => onHoverChange?.(true)}
      onMouseLeave={() => onHoverChange?.(false)}
      className="relative overflow-hidden rounded-2xl border border-line border-l-4 border-l-heat bg-surface p-5 shadow-[0_1px_2px_rgba(16,32,55,0.05),0_10px_30px_rgba(16,32,55,0.05)]"
    >
      <div className="flex items-center gap-2">
        <span
          id="lever-h"
          className="inline-flex items-center gap-1.5 text-[12px] font-medium uppercase tracking-wider text-navy"
        >
          <Zap className="size-4 text-heat" aria-hidden />
          This week&apos;s lever
        </span>
        <span className="rounded bg-canvas px-2 py-0.5 text-[11px] font-medium text-ink-2 ring-1 ring-line">
          Stolen bases
        </span>
      </div>

      <p className="mt-3 text-[18px] leading-relaxed text-ink">
        You&apos;re{" "}
        <span className="font-semibold text-ember">{behindBy} stolen bases behind</span> your
        opponent — your single biggest gap. Three free agents can close most of it before Sunday.
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <button className="group inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-[#ff7a2e] to-heat px-5 py-2.5 text-sm font-semibold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none">
          See The 3 Pickups
          <ArrowRight
            className="size-4 transition-transform duration-[var(--dur-1)] group-hover:translate-x-1 motion-reduce:transform-none"
            aria-hidden
          />
        </button>

        <div className="flex items-center">
          {pickups.map((p, i) => (
            <Tooltip.Root key={`${p.name}-${i}`}>
              <Tooltip.Trigger asChild>
                <button
                  className="-ml-2 rounded-full first:ml-0"
                  style={{ zIndex: pickups.length - i }}
                  aria-label={`${p.name}, ${p.projStat.label} ${p.projStat.value}`}
                >
                  <PlayerAvatar
                    mlbId={p.mlbId}
                    teamId={p.teamId}
                    name={p.name}
                    size={40}
                    ring="ring-white"
                  />
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  sideOffset={8}
                  className="z-50 rounded-lg bg-navy px-3 py-2 text-xs text-chrome shadow-[0_12px_30px_rgba(0,0,0,0.4)]"
                >
                  <div className="font-medium">{p.name}</div>
                  <div className="tnum text-white/70">
                    {p.projStat.label}: {p.projStat.value}
                  </div>
                  <Tooltip.Arrow className="fill-navy" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          ))}
        </div>
      </div>
    </section>
  );
}
