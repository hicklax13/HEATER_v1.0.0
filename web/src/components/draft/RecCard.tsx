"use client";

import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import type { DraftRec } from "@/lib/draft-data";

const TAG_TONE: Record<string, string> = {
  buy: "bg-ok/12 text-ok",
  fair: "bg-flame/15 text-flame",
  avoid: "bg-ember/12 text-ember",
};

/** Featured recommendation card: headshot, name, BUY/FAIR/AVOID tag, score heat
 *  bar, the engine's reason, and a Draft button. */
export function RecCard({ rec, onDraft, disabled }: { rec: DraftRec; onDraft: () => void; disabled: boolean }) {
  const col = heatColor(rec.score);
  const tag = (rec.tag ?? "fair").toLowerCase();
  return (
    <Card className="flex flex-col p-4">
      <div className="flex items-start gap-3">
        <PlayerAvatar mlbId={rec.player.mlbId} teamId={rec.player.teamId} name={rec.player.name} size={44} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <PlayerLink player={rec.player} className="truncate text-[15px]" />
            <span
              className={cn(
                "ml-auto shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide",
                TAG_TONE[tag] ?? "bg-surface-2 text-ink-2",
              )}
            >
              {tag}
            </span>
          </div>
          <div className="text-[11px] text-ink-3">
            {rec.player.pos} · {rec.player.teamAbbr}
          </div>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
          <span
            className="block h-full rounded-full"
            style={{ width: `${Math.min(100, rec.score)}%`, background: col }}
          />
        </div>
        <span className="tnum text-[12px] font-bold" style={{ color: col }}>
          {rec.score}
        </span>
      </div>
      <p className="mt-2 flex-1 text-[12px] leading-snug text-ink-2">{rec.reason}</p>
      <button
        onClick={onDraft}
        disabled={disabled}
        className="mt-3 inline-flex min-h-9 w-full items-center justify-center rounded-lg bg-navy px-4 py-2 text-[13px] font-bold text-white transition-[transform,background-color] hover:bg-[#14305a] active:scale-95 disabled:opacity-50 motion-reduce:transform-none"
      >
        Draft
      </button>
    </Card>
  );
}
