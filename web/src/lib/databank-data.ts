import { apiGet } from "@/lib/api/client";
import { apiDatabankToData } from "@/lib/api/adapters";
import type { ApiDatabankResponse } from "@/lib/api/types";
import type { PlayerRef } from "./types";
import type { PlayerPick } from "./player-search";

/**
 * Player Databank — a "look up anyone" multi-year history. The picker is the
 * shared player search (any player, real ids); the history comes from
 * GET /api/databank?player_id=N (DB-backed → real data locally). Off-live falls
 * back to a demo history for the picked player so the showcase still works.
 */
export interface DatabankSeason {
  year: number;
  stats: Record<string, number>; // category → value
}

export interface DatabankData {
  player: PlayerRef;
  seasons: DatabankSeason[]; // newest first
}

// Off-live demo: a generic 3-year template (the picked player's identity is kept).
const DEMO_SEASONS: DatabankSeason[] = [
  { year: 2026, stats: { R: 74, HR: 22, RBI: 61, SB: 8, AVG: 0.291, OBP: 0.372 } },
  { year: 2025, stats: { R: 96, HR: 31, RBI: 88, SB: 11, AVG: 0.279, OBP: 0.358 } },
  { year: 2024, stats: { R: 88, HR: 27, RBI: 79, SB: 9, AVG: 0.271, OBP: 0.349 } },
];

/** Look up a player's multi-year history. Live → GET /api/databank?player_id=N
 *  (real locally) — errors propagate so the caller can distinguish an outage
 *  from a genuinely empty history. Off-live → a demo history carrying the picked
 *  player's identity (never throws). */
export async function fetchDatabank(player: PlayerPick): Promise<DatabankData | null> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    const api = await apiGet<ApiDatabankResponse>("/databank", { player_id: player.id });
    return apiDatabankToData(api);
  }
  return {
    player: { name: player.name, pos: player.pos, teamAbbr: player.teamAbbr, teamId: player.teamId, mlbId: player.mlbId },
    seasons: DEMO_SEASONS,
  };
}

// Canonical column order; any extra keys are appended alphabetically.
const COL_ORDER = [
  "R", "HR", "RBI", "SB", "AVG", "OBP", "SLG", "OPS", "AB", "H", "BB",
  "W", "L", "SV", "K", "IP", "ERA", "WHIP", "FIP", "K9",
];

/** The union of stat keys across all seasons, in canonical order. */
export function databankColumns(seasons: DatabankSeason[]): string[] {
  const keys = new Set<string>();
  for (const s of seasons) for (const k of Object.keys(s.stats)) keys.add(k);
  const ordered = COL_ORDER.filter((k) => keys.has(k));
  const extras = [...keys].filter((k) => !COL_ORDER.includes(k)).sort();
  return [...ordered, ...extras];
}

const RATE_KEYS = new Set(["AVG", "OBP", "SLG", "OPS", "BABIP"]);
const TWO_DP_KEYS = new Set(["ERA", "WHIP", "FIP", "SIERA", "XFIP", "K9", "IP"]);

/** Format a databank stat value by key. */
export function formatDatabankValue(key: string, v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return "—";
  const k = key.toUpperCase();
  if (RATE_KEYS.has(k)) return v.toFixed(3).replace(/^(-?)0\./, "$1.");
  if (TWO_DP_KEYS.has(k)) return v.toFixed(2);
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}
