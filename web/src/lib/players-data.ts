import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { apiPoolToPlayers } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiFreeAgentPoolResponse } from "@/lib/api/types";

/**
 * Mock Free Agents data — available players ranked by marginal value.
 * With NEXT_PUBLIC_HEATER_LIVE=1, fetchPlayers wires to GET /api/free-agents/pool
 * and falls back to this mock on any error or empty response.
 */
export interface FreeAgent extends PlayerRef {
  id: number; // HEATER player_id — needed to call /api/compare (the picker source)
  rank: number; // FA value rank
  ownPct: number; // % rostered across the league
  ownDelta: number; // ownership trend, last day
  value: number; // marginal value to your team (0–100)
  hitter: boolean;
  stats: { label: string; value: string }[]; // 3 key stats
  fit: string; // category this player most helps (R, HR, SB, SV, K, …)
  tag?: string; // "Rising", "Streamer", "Closer role"…
}

export interface PlayersData {
  topNeed: string; // your biggest category gap, e.g. "SB"
  freeAgents: FreeAgent[];
}

const fa = (
  rank: number,
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
  hitter: boolean,
  value: number,
  ownPct: number,
  ownDelta: number,
  fit: string,
  stats: [string, string][],
  tag?: string,
): FreeAgent => ({
  id: mlbId, // mock token (live data supplies the real HEATER id via the adapter)
  rank,
  name,
  pos,
  teamAbbr,
  teamId,
  mlbId,
  hitter,
  value,
  ownPct,
  ownDelta,
  fit,
  tag,
  stats: stats.map(([label, value]) => ({ label, value })),
});

export const PLAYERS: PlayersData = {
  topNeed: "SB",
  freeAgents: [
    fa(1, "Jose Caballero", "2B,SS,3B,OF", "NYY", 147, 676609, true, 86, 24, 5, "SB", [["AVG", ".238"], ["SB", "22"], ["R", "38"]], "Rising"),
    fa(2, "Victor Scott II", "OF", "STL", 138, 687363, true, 81, 17, 8, "SB", [["AVG", ".245"], ["SB", "26"], ["R", "41"]], "Rising"),
    fa(3, "Porter Hodge", "RP", "CHC", 112, 687863, false, 75, 21, 11, "SV", [["ERA", "2.10"], ["SV", "7"], ["K", "38"]], "Closer Role"),
    fa(4, "Brenton Doyle", "OF", "COL", 115, 686668, true, 72, 33, 2, "SB", [["AVG", ".258"], ["HR", "12"], ["SB", "18"]]),
    fa(5, "Jake McCarthy", "OF", "COL", 115, 664983, true, 68, 29, -1, "SB", [["AVG", ".272"], ["SB", "19"], ["R", "44"]]),
    fa(6, "Reese Olson", "SP", "DET", 116, 681857, false, 64, 26, 4, "K", [["ERA", "3.25"], ["K", "78"], ["W", "5"]], "Streamer"),
    fa(7, "Jordan Romano", "RP", "PHI", 143, 605447, false, 61, 38, 3, "SV", [["ERA", "3.40"], ["SV", "9"], ["K", "30"]]),
    fa(8, "Spencer Steer", "1B,OF", "CIN", 113, 668715, true, 57, 44, 0, "RBI", [["AVG", ".244"], ["HR", "14"], ["RBI", "51"]]),
    fa(9, "Jonathan India", "2B,3B,OF", "KC", 118, 663697, true, 53, 36, -2, "R", [["AVG", ".251"], ["R", "49"], ["OBP", ".340"]]),
    fa(10, "Mitch Keller", "SP", "PIT", 134, 656605, false, 49, 41, 1, "K", [["ERA", "3.85"], ["K", "92"], ["W", "6"]]),
  ],
};

/** Live: GET /api/free-agents/pool → adapt; live errors propagate (HIGH-3) so
 *  usePageData reaches error/locked/unlinked. Mock (off-live, or live with an
 *  empty pool): the in-memory PLAYERS after a simulated delay. */
export async function fetchPlayers(delayMs = 600): Promise<PlayersData | null> {
  return liveOrMock(
    async () => {
      // Single-league context: the user's team is fixed until multi-tenancy (M4).
      const api = await apiGet<ApiFreeAgentPoolResponse>("/free-agents/pool", {
        team_name: getViewerTeam(),
        limit: 50,
      });
      return api.free_agents.length > 0 ? apiPoolToPlayers(api) : null;
    },
    () => new Promise<PlayersData>((resolve) => setTimeout(() => resolve(PLAYERS), delayMs)),
  );
}
