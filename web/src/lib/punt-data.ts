import { apiGet } from "@/lib/api/client";
import { apiPuntToData } from "@/lib/api/adapters";
import type { ApiPuntResponse } from "@/lib/api/types";

/**
 * Punt-strategy data: a per-category compete/punt verdict + the reallocation
 * action. Wires to GET /api/punt (Yahoo-dependent → mock fallback; empty
 * locally until standings sync).
 */

export type PuntVerdict = "compete" | "tossup" | "punt";

export interface PuntCat {
  cat: string;
  rank: number; // current league rank (1 = best)
  gainable: boolean; // can you realistically still gain here?
  verdict: PuntVerdict;
  recommendation: string;
}

export interface PuntData {
  teamName: string;
  nTeams: number;
  candidates: string[]; // auto-detected punt categories
  cats: PuntCat[];
}

/** Derive a compete/tossup/punt verdict from rank + gainability + the engine's
 *  detected punt candidates. */
export function verdictFor(
  cat: string,
  rank: number,
  gainable: boolean,
  candidates: string[],
  nTeams: number,
): PuntVerdict {
  if (candidates.includes(cat)) return "punt";
  if (gainable || rank <= Math.ceil(nTeams / 2)) return "compete";
  return "tossup";
}

const MOCK_CANDIDATES = ["SV", "ERA"];

const c = (cat: string, rank: number, gainable: boolean, recommendation: string): PuntCat => ({
  cat,
  rank,
  gainable,
  recommendation,
  verdict: verdictFor(cat, rank, gainable, MOCK_CANDIDATES, 12),
});

// Mock: Team Hickey — hitting-strong, pitching-weak (consistent with the
// Standings mock). The engine "detects" SV + ERA as punts (rank 10, not gainable).
export const PUNT: PuntData = {
  teamName: "Team Hickey",
  nTeams: 12,
  candidates: MOCK_CANDIDATES,
  cats: [
    c("SB", 1, true, "League-best — your anchor category."),
    c("OBP", 1, true, "League-best on-base. Build around it."),
    c("AVG", 2, true, "Top-tier contact; safe to compete."),
    c("R", 3, true, "Strong — runs follow the on-base edge."),
    c("HR", 5, true, "Solid; gainable with one power add."),
    c("RBI", 6, true, "Middle of the pack; lineup-driven."),
    c("K", 7, true, "Gainable — streaming starters adds strikeouts."),
    c("W", 9, false, "Thin rotation; hard to climb."),
    c("WHIP", 9, false, "Weak ratios; only improves with arms."),
    c("L", 9, false, "Tied to the same thin rotation."),
    c("SV", 10, false, "No closers — concede it and reallocate."),
    c("ERA", 10, false, "Ratio sink — concede it and reallocate."),
  ],
};

export async function fetchPunt(delayMs = 600): Promise<PuntData> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiPuntResponse>("/punt", { team_name: "Team Hickey" });
      if ((api.categories?.length ?? 0) > 0) return apiPuntToData(api);
    } catch {
      // fall through to mock
    }
  }
  return new Promise((resolve) => setTimeout(() => resolve(PUNT), delayMs));
}
