import { apiGet } from "@/lib/api/client";
import { apiStandingsToData } from "@/lib/api/adapters";
import type { ApiStandingsResponse } from "@/lib/api/types";

/**
 * Standings data. The standings table wires to GET /api/standings (Yahoo-
 * dependent → mock fallback). Playoff / championship odds are NOT in the API
 * yet (mock-only) — a contract ask for the CEO: a playoff-sim endpoint.
 */

/** The 12 FourzynBurn categories, in heatmap display order (rank 1 = best,
 *  inverse cats already accounted for in the rank). */
export const CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "SV", "K", "L", "ERA", "WHIP"] as const;

export interface TeamStanding {
  rank: number;
  teamName: string;
  wins: number;
  losses: number;
  ties: number;
  points: number;
  categoryRanks: Record<string, number>; // cat → league rank (1 = best)
  playoffOdds: number; // 0–100 (mock-only; API gap)
  champOdds: number; // 0–100 (mock-only; API gap)
  isUser: boolean;
}

export interface StandingsData {
  teams: TeamStanding[];
  playoffSpots: number; // top-N make the playoffs (cut line)
}

const team = (
  rank: number,
  teamName: string,
  wins: number,
  losses: number,
  ties: number,
  points: number,
  ranks: number[], // 12 category ranks, in CATEGORIES order
  playoffOdds: number,
  champOdds: number,
  isUser = false,
): TeamStanding => ({
  rank,
  teamName,
  wins,
  losses,
  ties,
  points,
  categoryRanks: Object.fromEntries(CATEGORIES.map((c, i) => [c, ranks[i]])),
  playoffOdds,
  champOdds,
  isUser,
});

// Mock: 12-team FourzynBurn snapshot. Team Hickey (4th) is a hitting-strong,
// pitching-weak archetype — visible in the category heatmap.
export const STANDINGS: StandingsData = {
  playoffSpots: 4,
  teams: [
    team(1, "Bronx Bombers", 11, 1, 0, 71, [2, 1, 1, 6, 3, 2, 3, 5, 2, 4, 1, 1], 99, 30),
    team(2, "Dinger City", 9, 3, 0, 64, [1, 2, 3, 8, 5, 4, 1, 2, 1, 2, 4, 3], 96, 22),
    team(3, "The Wheelhouse", 8, 4, 0, 59, [4, 3, 2, 3, 4, 3, 5, 7, 4, 3, 5, 6], 89, 15),
    team(4, "Team Hickey", 7, 4, 1, 56, [3, 5, 6, 1, 2, 1, 9, 10, 7, 9, 10, 9], 70, 10, true),
    team(5, "Sluggers Anon", 7, 5, 0, 54, [5, 4, 4, 9, 7, 6, 2, 3, 3, 5, 2, 2], 63, 8),
    team(6, "Rally Caps", 6, 6, 0, 48, [6, 7, 5, 4, 6, 7, 6, 4, 8, 6, 6, 5], 45, 4),
    team(7, "Foul Territory", 6, 6, 0, 46, [8, 6, 8, 2, 9, 8, 4, 1, 5, 7, 3, 4], 40, 3),
    team(8, "Bushleaguers", 5, 7, 0, 41, [7, 8, 7, 5, 8, 9, 7, 6, 6, 8, 7, 8], 22, 1.5),
    team(9, "Curveball Kings", 5, 7, 0, 39, [9, 9, 9, 7, 10, 10, 8, 8, 9, 1, 8, 7], 16, 1),
    team(10, "Whiff Merchants", 4, 8, 0, 32, [10, 10, 10, 10, 11, 11, 10, 9, 10, 10, 9, 10], 6, 0.3),
    team(11, "Pinch Hitters", 3, 9, 0, 26, [11, 11, 11, 12, 1, 5, 11, 11, 11, 11, 12, 11], 2, 0.1),
    team(12, "Cellar Dwellers", 2, 10, 0, 18, [12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 11, 12], 1, 0),
  ],
};

export async function fetchStandings(delayMs = 600): Promise<StandingsData> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiStandingsResponse>("/standings");
      if ((api.teams?.length ?? 0) > 0) return apiStandingsToData(api);
    } catch {
      // fall through to mock
    }
  }
  return new Promise((resolve) => setTimeout(() => resolve(STANDINGS), delayMs));
}
