import { Flame } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { heatColor } from "@/lib/tokens";
import type { StreamCandidate } from "@/lib/streaming-data";

export function TopPickCallout({ pick }: { pick: StreamCandidate }) {
  const col = heatColor(pick.score);
  return (
    <Card className="overflow-hidden p-0">
      <div className="flex flex-wrap items-center gap-4 bg-gradient-to-r from-navy to-[#15294a] p-5 text-white">
        <PlayerAvatar mlbId={pick.player.mlbId} teamId={pick.player.teamId} name={pick.player.name} size={56} ring="ring-white/20" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-flame">
            <Flame className="size-3.5" aria-hidden /> Top stream today
          </div>
          <PlayerDialog player={pick.player}>
            <button className="block truncate text-left font-display text-xl font-extrabold text-white hover:text-flame">
              {pick.player.name}
            </button>
          </PlayerDialog>
          <div className="text-[13px] text-white/70">
            {pick.player.teamAbbr} {pick.isHome ? "vs" : "@"} {pick.opponent} · {pick.expectedLine}
          </div>
          <p className="mt-1 text-[13px] text-white/85">{pick.why}</p>
        </div>
        <div className="text-right leading-none" style={{ color: col }}>
          <HeroNum width={70} className="text-4xl">{pick.score}</HeroNum>
          <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-white/60">Score</div>
        </div>
      </div>
    </Card>
  );
}
