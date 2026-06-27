import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiPost } from "@/lib/api/client";
import { apiOptimizeToData } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiLineupOptimizeResponse } from "@/lib/api/types";

/**
 * Optimizer data — today's recommended lineup for Team Hickey. With
 * NEXT_PUBLIC_HEATER_LIVE=1, fetchOptimizer POSTs /api/lineup/optimize in DAILY
 * mode (start/sit by 0-100 heat value). Daily mode needs live Yahoo, so it
 * returns empty in local dev (→ the page's empty-state) and populates on Railway;
 * off-live falls back to the mock below. The mock shape is the contract.
 */
export type SlotStatus = "start" | "sit" | "bench" | "off";

export interface LineupSlot {
  slot: string; // C, 1B, …, SP, RP, P, Util, BN
  player: PlayerRef & { ownPct?: number };
  matchup: string; // "vs SF", "@NYM", "OFF"
  proj?: string; // short projected line (mock only — the daily API has no per-stat line)
  value: number; // 0–100 daily value
  status: SlotStatus;
  note?: string;
  currentSlot?: string; // player's current Yahoo slot (live — lets us diff swaps)
  forcedStart?: boolean; // started despite a poor matchup (roster-forced)
}

export interface CatImpact {
  key: string;
  proj: string;
  trend: "up" | "down" | "flat";
}

export type OptimizerScope = "today" | "rest_of_week" | "rest_of_season";

export interface FaPickup {
  add: PlayerRef;
  drop: PlayerRef;
  netSgpDelta: number;
  categoryImpact: { label: string; value: string }[];
  reasoning: string;
  urgencyCategories: string[];
}

export interface OptimizerSwap {
  out: string;
  in: string;
  gain?: string; // mock only — the daily API ranks by heat value, not an SGP delta
}

/** Day-level matchup context from the daily optimizer's `daily{}` meta.
 *  Populated only in daily mode with live Yahoo data; the panel hides when empty. */
export interface DailyContext {
  winning: string[]; // category keys you're currently winning
  losing: string[]; // …losing (these are the focus cats)
  tied: string[];
  rateModes: Record<string, string>; // ERA/WHIP → "protect" | "compete" | "abandon"
  urgency: Record<string, number>; // category → 0..1 urgency weight (orders the focus cats)
  recommendations: string[]; // plain-language daily advice
}

export interface OptimizerData {
  date: string;
  optimal: boolean; // is the current lineup already optimal?
  starters: LineupSlot[];
  bench: LineupSlot[];
  ipPace?: { value: number; total: number }; // daily mode only
  movesLeft?: { value: number; total: number }; // not in the optimize contract
  swaps: OptimizerSwap[];
  impact: CatImpact[];
  daily?: DailyContext; // daily-mode matchup context (urgency/rate-modes/game-state/recs)
  faSuggestions: FaPickup[]; // available "drop X for Y" pickups (composed from recommend_fa_moves)
}

const p = (
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
): PlayerRef => ({ name, pos, teamAbbr, teamId, mlbId });

export const OPTIMIZER: OptimizerData = {
  date: "Tue Jun 17",
  optimal: false,
  starters: [
    { slot: "C", player: p("Will Smith", "C", "LAD", 119, 669257), matchup: "vs COL", proj: "0.9 H · 0.4 R", value: 61, status: "start" },
    { slot: "1B", player: p("Matt Olson", "1B", "ATL", 144, 621566), matchup: "@MIA", proj: "1.0 H · 0.5 RBI", value: 68, status: "start" },
    { slot: "2B", player: p("Ozzie Albies", "2B", "ATL", 144, 645277), matchup: "@MIA", proj: "0.9 H · 0.3 SB", value: 55, status: "start" },
    { slot: "3B", player: p("José Ramírez", "3B", "CLE", 114, 608070), matchup: "vs MIN", proj: "1.1 H · 0.5 RBI", value: 74, status: "start" },
    { slot: "SS", player: p("Bobby Witt Jr.", "SS", "KC", 118, 677951), matchup: "vs DET", proj: "1.2 H · 0.4 SB", value: 81, status: "start" },
    { slot: "OF", player: p("Aaron Judge", "RF", "NYY", 147, 592450), matchup: "@BAL", proj: "1.0 H · 0.6 HR", value: 88, status: "start" },
    { slot: "OF", player: p("Kyle Tucker", "RF", "LAD", 119, 663656), matchup: "@STL", proj: "1.0 H · 0.4 RBI", value: 64, status: "start" },
    { slot: "OF", player: p("Juan Soto", "RF", "NYM", 121, 665742), matchup: "vs PHI", proj: "0.7 H · 0.2 HR", value: 41, status: "sit", note: "2-for-21 + LHP starter — sit for Merrill (RHP)" },
    { slot: "Util", player: p("Yordan Alvarez", "DH", "HOU", 117, 670541), matchup: "vs SEA", proj: "1.0 H · 0.5 HR", value: 72, status: "start" },
    { slot: "Util", player: p("Marcell Ozuna", "DH", "PIT", 134, 542303), matchup: "@MIA", proj: "0.9 H · 0.4 RBI", value: 57, status: "start" },
    { slot: "SP", player: p("Tarik Skubal", "SP", "DET", 116, 669373), matchup: "@KC", proj: "6.0 IP · 7 K", value: 90, status: "start", note: "Two-start week" },
    { slot: "SP", player: p("Zack Wheeler", "SP", "PHI", 143, 554430), matchup: "@NYM", proj: "6.1 IP · 7 K", value: 85, status: "start" },
    { slot: "RP", player: p("Emmanuel Clase", "RP", "CLE", 114, 661403), matchup: "vs MIN", proj: "1.0 IP · SV", value: 66, status: "start" },
    { slot: "RP", player: p("Devin Williams", "RP", "NYM", 121, 642207), matchup: "@BAL", proj: "1.0 IP · K", value: 60, status: "start" },
  ],
  bench: [
    { slot: "BN", player: p("Jackson Merrill", "OF", "SD", 135, 701538), matchup: "vs PHI", proj: "1.0 H · 0.4 RBI", value: 63, status: "bench", note: "RHP matchup — start over Soto" },
    { slot: "BN", player: p("Spencer Strider", "SP", "ATL", 144, 675911), matchup: "OFF", proj: "—", value: 0, status: "off", note: "Not pitching today" },
    { slot: "BN", player: p("Jordan Westburg", "3B", "BAL", 110, 676059), matchup: "vs NYY", proj: "0.8 H · 0.3 R", value: 44, status: "bench" },
  ],
  ipPace: { value: 38, total: 54 },
  movesLeft: { value: 7, total: 10 },
  swaps: [
    { out: "Juan Soto", in: "Jackson Merrill", gain: "+0.22 SGP today" },
  ],
  impact: [
    { key: "R", proj: "6.4", trend: "up" },
    { key: "HR", proj: "2.1", trend: "up" },
    { key: "RBI", proj: "5.8", trend: "flat" },
    { key: "SB", proj: "0.9", trend: "down" },
    { key: "K", proj: "29", trend: "up" },
    { key: "SV", proj: "1.4", trend: "flat" },
  ],
  daily: {
    winning: ["HR", "R", "OBP", "K"],
    losing: ["SB", "SV", "AVG"],
    tied: ["RBI"],
    rateModes: { ERA: "compete", WHIP: "protect" },
    urgency: { SB: 0.82, SV: 0.74, AVG: 0.5, HR: 0.18 },
    recommendations: [
      "Start Merrill over Soto vs the RHP — SB upside in a category you're losing.",
      "Hold Clase for the save; WHIP is a protect category this week.",
    ],
  },
  faSuggestions: [],
};

/** Live: POST /api/lineup/optimize at the chosen scope → adapt. scope="today" runs the
 *  daily DCV start/sit path on the backend; rest_of_week/rest_of_season run the standard LP.
 *  Live errors propagate (HIGH-3) so usePageData reaches error/locked(402)/unlinked(409); an
 *  empty lineup → null → page `empty`. Off-live → the in-memory OPTIMIZER mock. */
export async function fetchOptimizer(scope: OptimizerScope = "today"): Promise<OptimizerData | null> {
  return liveOrMock(
    async () => {
      const api = await apiPost<ApiLineupOptimizeResponse>("/lineup/optimize", {
        team_name: getViewerTeam(),
        scope,
      });
      const data = apiOptimizeToData(api);
      return data.starters.length > 0 || data.faSuggestions.length > 0 ? data : null;
    },
    () => new Promise<OptimizerData>((resolve) => setTimeout(() => resolve(OPTIMIZER), 400)),
  );
}
