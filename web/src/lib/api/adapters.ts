import type { ApiLeadersResponse } from "@/lib/api/types";
import type { LeaderRow, ResearchData } from "@/lib/research-data";

const PITCHER_POS = /\b(P|SP|RP)\b/;

function isHitter(positions: string): boolean {
  return positions ? !PITCHER_POS.test(positions) : true;
}

/** Format the single category figure the API returns (rate cats keep decimals). */
function fmt(value: number, category: string): string {
  const rate = ["AVG", "OBP"].includes(category)
    ? value.toFixed(3)
    : ["ERA", "WHIP"].includes(category)
      ? value.toFixed(2)
      : String(Math.round(value));
  return `${rate} ${category}`;
}

/** Map the minimal API leaders response → the richer frontend ResearchData,
 *  filling unavailable fields with honest placeholders (no headshot, one stat,
 *  forced tag). Surfaces the contract gap visibly. */
export function apiLeadersToResearch(api: ApiLeadersResponse): ResearchData {
  const leaders: LeaderRow[] = api.rows.map((r) => ({
    name: r.player.name,
    pos: r.player.positions,
    // PlayerRef now carries team/mlb fields (snake_case JSON -> camelCase here).
    // /api/leaders emits null for these until M0.2 populates it; the ?? fallbacks
    // keep PlayerAvatar's initials placeholder until real ids arrive, then
    // headshots + team logos light up with no further frontend change.
    teamAbbr: r.player.team_abbr ?? "",
    teamId: r.player.team_id ?? 0,
    mlbId: r.player.mlb_id ?? 0,
    rank: r.rank,
    hitter: isHitter(r.player.positions),
    stats: [fmt(r.value, api.category)], // gap: API gives one figure, not 3
    value: Math.max(20, 100 - (r.rank - 1) * 3), // gap: no 0-100 score -> derive from rank
    trend: "flat", // gap: API has no trend
    tag: "hot", // gap: API has no tag; forced placeholder so the row renders
    note: `Live · ${api.category} leader`,
  }));
  return { leaders };
}
