"use client";

import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { heatColor, MLB } from "@/lib/tokens";
import { teamBrand } from "@/lib/teams";
import type { CloserEntry } from "@/lib/closers-data";

/* eslint-disable @next/next/no-img-element -- remote MLB CDN team logo; next/image
   would require remotePatterns config and per-image sizing we don't need here. */

const CONF_TONE: Record<string, string> = {
  Locked: "bg-ok/12 text-ok",
  High: "bg-heat/12 text-heat",
  Committee: "bg-flame/15 text-flame",
  Shaky: "bg-ember/12 text-ember",
};

/** One team's closer: team-accent card with a job-security heat bar, the
 *  closer's headshot/name, confidence, and (when present) stats + handcuffs. */
export function CloserCard({ entry }: { entry: CloserEntry }) {
  const { closer, team, confidence, role, security, handcuffs, stats } = entry;
  const accent = closer ? teamBrand(closer.teamId).primary : "#0a1f3a";
  const col = heatColor(security);

  return (
    <Card className="relative overflow-hidden p-0">
      <span className="absolute inset-y-0 left-0 w-1" style={{ background: accent }} aria-hidden />
      {closer && (
        <img
          src={MLB.teamLogo(closer.teamId)}
          alt=""
          aria-hidden
          onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
          className="pointer-events-none absolute -right-3 -top-2 size-20 opacity-[0.06]"
        />
      )}
      <div className="relative p-3.5 pl-4">
        <div className="flex items-center gap-1.5">
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-extrabold tracking-wider text-white"
            style={{ background: accent }}
          >
            {team}
          </span>
          <span
            className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-bold ${CONF_TONE[confidence] ?? "bg-surface-2 text-ink-2"}`}
          >
            {confidence}
          </span>
        </div>

        <div className="mt-2 flex items-center gap-2.5">
          {closer && <PlayerAvatar mlbId={closer.mlbId} teamId={closer.teamId} name={closer.name} size={40} />}
          <div className="min-w-0">
            {closer ? (
              <PlayerLink player={closer} className="block truncate text-[15px]" />
            ) : (
              <span className="text-[15px] font-semibold text-ink-3">No closer</span>
            )}
            <div className="text-[11px] text-ink-3">{role}</div>
          </div>
        </div>

        <div className="mt-3 flex items-center gap-2">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
            <span className="block h-full rounded-full" style={{ width: `${security}%`, background: col }} />
          </div>
          <span className="tnum text-[12px] font-bold" style={{ color: col }}>
            {security}%
          </span>
        </div>
        <div className="mt-0.5 text-[9px] font-bold uppercase tracking-wider text-ink-3">Job security</div>

        {stats.length > 0 && (
          <div className="mt-3 flex border-t border-line pt-2.5">
            {stats.map((s) => (
              <div key={s.label} className="flex-1 text-center">
                <div className="text-[8.5px] font-bold uppercase tracking-wide text-ink-3">{s.label}</div>
                <HeroNum width={72} className="text-[15px] text-navy">
                  {s.value}
                </HeroNum>
              </div>
            ))}
          </div>
        )}

        {handcuffs.length > 0 && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5 border-t border-line pt-2">
            <span className="text-[9px] font-bold uppercase tracking-wider text-ink-3">Next up</span>
            {handcuffs.map((h, i) => (
              <PlayerLink key={`${h.name}-${i}`} player={h} className="text-[11px] font-semibold" />
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
