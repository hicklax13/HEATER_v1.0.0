import type { PlayerRef } from "./types";
import { apiPost } from "@/lib/api/client";
import { apiDraftRecommendToData, apiDraftSimulateToData } from "@/lib/api/adapters";
import type {
  ApiDraftRecommendResponse,
  ApiDraftSimulateResponse,
  ApiDraftConfig,
  ApiDraftPick,
} from "@/lib/api/types";
import { mockRecommend, mockSimulate } from "./draft-mock";

/**
 * Draft Simulator data layer. The API is STATELESS: the client owns `config`
 * + `pickLog` and the server replays them every call. With NEXT_PUBLIC_HEATER_LIVE=1
 * the fetchers POST to /api/draft/{recommend,simulate-picks}; offline (or on any
 * error) they fall back to the client-side mock draft (draft-mock.ts) so the page
 * is fully playable without the Python backend. See
 * docs/superpowers/specs/2026-06-20-heater-draft-simulator-design.md
 */

export interface DraftConfig {
  numTeams: number;
  numRounds: number;
  userTeamIndex: number; // 0-based seat (UI input is 1-based → subtract 1)
}

export interface DraftPick {
  pick: number; // 0-indexed overall pick number at the moment of the pick
  teamIndex: number; // 0-based; authoritative (echo what the server returned)
  playerId: number;
  playerName: string;
  positions: string; // comma-separated, e.g. "SS,3B"
}

export interface DraftClock {
  currentPick: number;
  round: number; // 1-indexed
  pickInRound: number; // 1-indexed
  pickingTeamIndex: number; // 0-indexed team on the clock
  isUserTurn: boolean;
}

/** A draftable player — a PlayerRef (so it drops into PlayerLink/PlayerDialog)
 *  plus the HEATER `id` needed for the pick_log and an optional mock ADP. */
export interface DraftPlayer extends PlayerRef {
  id: number;
  adp?: number;
}

export interface DraftRec {
  player: DraftPlayer;
  rank: number;
  score: number; // ~0-100 composite
  projectedSgp: number;
  confidence: string | null; // HIGH | MEDIUM | LOW
  tag: string | null; // buy | fair | avoid
  reason: string;
}

export interface SimResult {
  clock: DraftClock;
  picks: DraftPick[]; // only the NEW picks made this call
}

export interface RecResult {
  clock: DraftClock;
  recs: DraftRec[];
  summary: string;
}

const live = () => process.env.NEXT_PUBLIC_HEATER_LIVE === "1";

/** Snake-draft order — pure + local (no round-trip). Even rounds run forward,
 *  odd rounds reverse. `pick` is the 0-indexed overall pick number. */
export function pickingTeam(pick: number, numTeams: number): number {
  const round = Math.floor(pick / numTeams);
  const pos = pick % numTeams;
  return round % 2 === 0 ? pos : numTeams - 1 - pos;
}

export const totalPicks = (c: DraftConfig): number => c.numTeams * c.numRounds;

// --- request serializers (camel → snake) ---
function toApiConfig(c: DraftConfig): ApiDraftConfig {
  return {
    num_teams: c.numTeams,
    num_rounds: c.numRounds,
    user_team_index: c.userTeamIndex,
    roster_config: null,
  };
}
function toApiPick(p: DraftPick): ApiDraftPick {
  return {
    pick: p.pick,
    team_index: p.teamIndex,
    player_id: p.playerId,
    player_name: p.playerName,
    positions: p.positions,
  };
}

/** Advance the AI opponents to the user's next turn (or end of draft). Live →
 *  POST /api/draft/simulate-picks; 402 → useDraft locked phase, and any OTHER
 *  live error propagates (HIGH-3) so useDraft surfaces it instead of fabricating a
 *  mock draft. Off-live → the client-side mock. */
export async function draftSimulate(config: DraftConfig, pickLog: DraftPick[]): Promise<SimResult> {
  if (live()) {
    const api = await apiPost<ApiDraftSimulateResponse>("/draft/simulate-picks", {
      config: toApiConfig(config),
      pick_log: pickLog.map(toApiPick),
      seed: null,
    });
    return apiDraftSimulateToData(api);
  }
  return mockSimulate(config, pickLog);
}

/** Recommendations for the user's current pick (call only when it's their turn).
 *  Live → POST /api/draft/recommend; 402 → useDraft locked phase, and any OTHER
 *  live error propagates (HIGH-3) so useDraft surfaces it instead of fabricating
 *  mock recs for the user's real draft. Off-live → the client-side mock. */
export async function draftRecommend(
  config: DraftConfig,
  pickLog: DraftPick[],
  topN = 8,
): Promise<RecResult> {
  if (live()) {
    const api = await apiPost<ApiDraftRecommendResponse>("/draft/recommend", {
      config: toApiConfig(config),
      pick_log: pickLog.map(toApiPick),
      top_n: topN,
      n_simulations: 300,
    });
    return apiDraftRecommendToData(api);
  }
  return mockRecommend(config, pickLog, topN);
}
