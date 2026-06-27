import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiPost } from "@/lib/api/client";
import { isLive } from "@/lib/api/live";
import type {
  ApiStartSitCompareResponse,
  ApiStartSitCandidate,
  ApiStartSitOptimizeResponse,
} from "@/lib/api/types";
import { apiOptSlotToData } from "@/lib/api/adapters";
import type { LineupSlot } from "@/lib/optimizer-data";

/**
 * Start/Sit data. Scope + selected player ids drive each request, so these
 * fetchers take args and are called imperatively (NOT via usePageData). Live →
 * POST /api/start-sit/{compare,optimize}; off-live → a small deterministic mock so
 * the showcase renders. The shapes ARE the contract.
 */
export type Scope = "today" | "rest_of_week" | "rest_of_season";

export const SCOPE_LABELS: Record<Scope, string> = {
  today: "Today",
  rest_of_week: "Rest of Week",
  rest_of_season: "Rest of Season",
};

export interface StatItem {
  label: string;
  value: string;
}

export interface StartSitCandidate {
  player: PlayerRef;
  startScore: number; // 0-100 heat
  rank: number;
  eligibleSlots: string[];
  projected: StatItem[];
  categoryImpact: StatItem[];
  matchup: string;
  reason: string;
  playable: boolean;
}

export interface StartSitVerdict {
  startIds: number[];
  sitIds: number[];
  reasoning: string;
}

export interface StartSitCompareData {
  scope: Scope;
  candidates: StartSitCandidate[];
  verdict: StartSitVerdict;
  openSlots: Record<string, number>;
  confidence: number;
  confidenceLabel: string;
}

export interface StartSitOptimizeData {
  scope: Scope;
  starters: LineupSlot[];
  bench: LineupSlot[];
  summary: string;
}

function toPlayerRef(p: {
  name: string;
  positions: string;
  mlb_id?: number | null;
  team_abbr?: string | null;
  team_id?: number | null;
}): PlayerRef {
  return { name: p.name, pos: p.positions, teamAbbr: p.team_abbr ?? "", teamId: p.team_id ?? 0, mlbId: p.mlb_id ?? 0 };
}

function adaptCandidate(c: ApiStartSitCandidate): StartSitCandidate {
  return {
    player: toPlayerRef(c.player),
    startScore: c.start_score ?? 0,
    rank: c.rank ?? 0,
    eligibleSlots: c.eligible_slots ?? [],
    projected: c.projected ?? [],
    categoryImpact: c.category_impact ?? [],
    matchup: c.matchup ?? "",
    reason: c.reason ?? "",
    playable: c.playable ?? true,
  };
}

export function adaptCompare(api: ApiStartSitCompareResponse): StartSitCompareData {
  return {
    scope: (api.scope ?? "today") as Scope,
    candidates: (api.candidates ?? []).map(adaptCandidate),
    verdict: {
      startIds: api.verdict?.start_ids ?? [],
      sitIds: api.verdict?.sit_ids ?? [],
      reasoning: api.verdict?.reasoning ?? "",
    },
    openSlots: api.open_slots ?? {},
    confidence: api.confidence ?? 0,
    confidenceLabel: api.confidence_label ?? "Toss-up",
  };
}

export function adaptOptimize(api: ApiStartSitOptimizeResponse): StartSitOptimizeData {
  return {
    scope: (api.scope ?? "today") as Scope,
    starters: (api.slots ?? []).map((s) => apiOptSlotToData(s, "starter")),
    bench: (api.bench ?? []).map((s) => apiOptSlotToData(s, "bench")),
    summary: api.summary ?? "",
  };
}

/** Compare 2-6 selected players at a scope → ranked verdict. Live errors propagate. */
export async function compareStartSit(scope: Scope, playerIds: number[]): Promise<StartSitCompareData> {
  if (!isLive()) return mockCompare(scope, playerIds);
  const api = await apiPost<ApiStartSitCompareResponse>("/start-sit/compare", {
    team_name: getViewerTeam(),
    scope,
    player_ids: playerIds,
  });
  return adaptCompare(api);
}

/** Apply the selected candidates to the user's open slots (authoritative LP). */
export async function optimizeStartSit(scope: Scope, playerIds: number[]): Promise<StartSitOptimizeData> {
  if (!isLive()) return mockOptimize(scope, playerIds);
  const api = await apiPost<ApiStartSitOptimizeResponse>("/start-sit/optimize", {
    team_name: getViewerTeam(),
    scope,
    player_ids: playerIds,
  });
  return adaptOptimize(api);
}

// --- off-live mocks (showcase) -------------------------------------------------
function mockCompare(scope: Scope, ids: number[]): StartSitCompareData {
  const candidates: StartSitCandidate[] = ids.slice(0, 6).map((id, i) => ({
    player: { name: `Player ${id}`, pos: "OF", teamAbbr: "NYY", teamId: 147, mlbId: 0 },
    startScore: Math.max(20, 100 - i * 18),
    rank: i + 1,
    eligibleSlots: ["OF", "Util"],
    projected: [
      { label: "HR", value: String(3 - i) },
      { label: "R", value: String(6 - i) },
    ],
    categoryImpact: [
      { label: "HR", value: `+${(1.2 - i * 0.2).toFixed(2)}` },
      { label: "SB", value: `+${(0.4 + i * 0.1).toFixed(2)}` },
    ],
    matchup: i % 2 ? "@ COL" : "vs SF",
    reason: i === 0 ? "Favorable park + platoon edge" : "Average matchup",
    playable: true,
  }));
  const startIds = candidates.slice(0, Math.min(2, candidates.length)).map((c) => c.player.mlbId || c.rank);
  return {
    scope,
    candidates,
    verdict: { startIds, sitIds: candidates.slice(2).map((c) => c.rank), reasoning: "Start the top 2 that fit your open slots." },
    openSlots: { OF: 2, Util: 1 },
    confidence: 0.34,
    confidenceLabel: "Clear",
  };
}

function mockOptimize(scope: Scope, ids: number[]): StartSitOptimizeData {
  return {
    scope,
    starters: ids.slice(0, 2).map((id) => ({
      slot: "OF",
      player: { name: `Player ${id}`, pos: "OF", teamAbbr: "NYY", teamId: 147, mlbId: 0 },
      matchup: "vs SF",
      value: 70,
      status: "start" as const,
    })),
    bench: [],
    summary: `${Math.min(2, ids.length)} starters set.`,
  };
}
