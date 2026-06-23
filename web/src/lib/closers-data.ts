import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { apiClosersToData } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiClosersResponse } from "@/lib/api/types";
import { securityFor } from "./closer-security";

/**
 * Mock Closer Monitor data — closer + handcuffs + job security across MLB.
 * With NEXT_PUBLIC_HEATER_LIVE=1, fetchClosers wires to GET /api/closers and
 * falls back to this mock on any error or empty response.
 *
 * Contract note for the CEO track: /api/closers is DB-backed (real closer
 * PlayerRefs + headshots) but currently MINIMAL — it returns only
 * closer/handcuffs/confidence/role/team, with confidence uniformly "Shaky",
 * no handcuffs, and no stats. To reach the high-value "Saves Finder" it should
 * populate real confidence levels + handcuffs + a numeric job-security % +
 * proj SV/ERA/WHIP. See docs/superpowers/specs/2026-06-19-heater-closer-monitor-archived-saves-finder.md
 */

export interface CloserStat {
  label: string;
  value: string;
}

export interface CloserEntry {
  team: string; // team abbr
  closer: PlayerRef | null;
  role: string; // "Closer" / "Committee"
  confidence: string; // "Locked" / "High" / "Committee" / "Shaky"
  security: number; // 0–100 job security (drives the heat bar)
  handcuffs: PlayerRef[]; // next-in-line (rendered only when present)
  stats: CloserStat[]; // PROJ SV / ERA / WHIP (rendered only when present)
}

export interface ClosersData {
  entries: CloserEntry[];
}

const ref = (name: string, teamAbbr: string, teamId: number, mlbId: number): PlayerRef => ({
  name,
  pos: "RP",
  teamAbbr,
  teamId,
  mlbId,
});

const cl = (
  team: string,
  closer: PlayerRef,
  confidence: string,
  stats: [string, string][],
  handcuffs: PlayerRef[] = [],
): CloserEntry => ({
  team,
  closer,
  role: confidence === "Committee" ? "Committee" : "Closer",
  confidence,
  security: securityFor(confidence),
  handcuffs,
  stats: stats.map(([label, value]) => ({ label, value })),
});

// NOTE: every mlbId/teamId MUST pass `node scripts/audit-mock-ids.mjs`.
export const CLOSERS: ClosersData = {
  entries: [
    cl("HOU", ref("Josh Hader", "HOU", 117, 623352), "Locked", [["PROJ SV", "38"], ["ERA", "2.15"], ["WHIP", "0.92"]]),
    cl("NYM", ref("Edwin Díaz", "NYM", 121, 621242), "Locked", [["PROJ SV", "36"], ["ERA", "2.30"], ["WHIP", "0.98"]]),
    cl("ATH", ref("Mason Miller", "ATH", 133, 695243), "Locked", [["PROJ SV", "34"], ["ERA", "2.05"], ["WHIP", "0.95"]]),
    cl("MIN", ref("Jhoan Duran", "MIN", 142, 661395), "High", [["PROJ SV", "32"], ["ERA", "2.55"], ["WHIP", "1.08"]]),
    cl("STL", ref("Ryan Helsley", "STL", 138, 664854), "High", [["PROJ SV", "33"], ["ERA", "2.60"], ["WHIP", "1.04"]]),
    cl("CLE", ref("Cade Smith", "CLE", 114, 671922), "High", [["PROJ SV", "28"], ["ERA", "2.70"], ["WHIP", "1.02"]]),
    cl("ATL", ref("Raisel Iglesias", "ATL", 144, 628452), "High", [["PROJ SV", "30"], ["ERA", "3.05"], ["WHIP", "1.10"]]),
    cl("TB", ref("Pete Fairbanks", "TB", 139, 664126), "Committee", [["PROJ SV", "22"], ["ERA", "3.40"], ["WHIP", "1.20"]]),
    cl("BOS", ref("Aroldis Chapman", "BOS", 111, 547973), "Committee", [["PROJ SV", "20"], ["ERA", "3.30"], ["WHIP", "1.22"]]),
    cl("ARI", ref("Paul Sewald", "ARI", 109, 623149), "Shaky", [["PROJ SV", "18"], ["ERA", "3.85"], ["WHIP", "1.28"]]),
  ],
};

/** Live: GET /api/closers → adapt; live errors propagate (HIGH-3) so usePageData
 *  reaches error/locked/unlinked. Mock (off-live, or live with no entries): the
 *  in-memory CLOSERS after a simulated delay. */
export async function fetchClosers(delayMs = 600): Promise<ClosersData | null> {
  return liveOrMock(
    async () => {
      const api = await apiGet<ApiClosersResponse>("/closers");
      return api.entries.length > 0 ? apiClosersToData(api) : null;
    },
    () => new Promise<ClosersData>((resolve) => setTimeout(() => resolve(CLOSERS), delayMs)),
  );
}
