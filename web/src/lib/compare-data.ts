import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { apiCompareToData } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiCompareResponse } from "@/lib/api/types";
import type { PlayerPick } from "@/lib/player-search";

/**
 * Player Compare data — head-to-head category comparison. With
 * NEXT_PUBLIC_HEATER_LIVE=1, fetchCompare wires to GET /api/compare?ids=… (pool-
 * backed → real locally) and falls back to a demo comparison. The picker searches
 * ANY player (player-search) — its HEATER `id` is what /api/compare consumes.
 */

/** Lower-is-better categories (FourzynBurn inverse stats). */
export const INVERSE_CATS = new Set(["ERA", "WHIP", "L"]);

const HITTING_CATS = new Set(["R", "HR", "RBI", "SB", "AVG", "OBP"]);

/** Whether a player produces on this category's side. A pure hitter's pitching
 *  stats come back as 0 (and vice-versa) — comparing them would award a hitter a
 *  phantom 0.00 ERA "win". So derive participation from the stat line: hitting if
 *  any hitting production, pitching if any pitching production (TWPs do both). L
 *  is excluded from the pitching test since a real pitcher can post 0 losses. */
export function participatesIn(cat: string, stats: Record<string, number>): boolean {
  const hits =
    (stats.AVG ?? 0) > 0 ||
    (stats.OBP ?? 0) > 0 ||
    (stats.R ?? 0) > 0 ||
    (stats.HR ?? 0) > 0 ||
    (stats.RBI ?? 0) > 0 ||
    (stats.SB ?? 0) > 0;
  const pitches =
    (stats.ERA ?? 0) > 0 ||
    (stats.WHIP ?? 0) > 0 ||
    (stats.W ?? 0) > 0 ||
    (stats.SV ?? 0) > 0 ||
    (stats.K ?? 0) > 0;
  return HITTING_CATS.has(cat.toUpperCase()) ? hits : pitches;
}

export interface ComparePlayerData extends PlayerRef {
  stats: Record<string, number>; // category → projected value
}

export interface CompareData {
  categories: string[];
  players: ComparePlayerData[];
}

/** Index of the player who wins a category (inverse-aware); -1 on tie / no data. */
export function bestIndexForCat(cat: string, values: (number | undefined)[]): number {
  const inverse = INVERSE_CATS.has(cat.toUpperCase());
  let best = -1;
  let bestVal: number | null = null;
  let tie = false;
  values.forEach((v, i) => {
    if (v === undefined || !Number.isFinite(v)) return;
    if (bestVal === null) {
      bestVal = v;
      best = i;
      return;
    }
    if (v === bestVal) tie = true;
    else if (inverse ? v < bestVal : v > bestVal) {
      bestVal = v;
      best = i;
      tie = false;
    }
  });
  return tie ? -1 : best;
}

/** Display a category value: AVG/OBP at 3dp (no leading zero), ERA/WHIP at 2dp,
 *  counting cats as ints; "—" when missing. */
export function formatCatValue(cat: string, v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return "—";
  const c = cat.toUpperCase();
  if (c === "AVG" || c === "OBP") return v.toFixed(3).replace(/^(-?)0\./, "$1.");
  if (c === "ERA" || c === "WHIP") return v.toFixed(2);
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

// Off-live demo so the showcase works: a sample hitter line (player 0 leads
// most cats). Live /api/compare returns the real, full category set.
const DEMO_CATS = ["R", "HR", "RBI", "SB", "AVG", "OBP"];
const DEMO_LINE: Record<string, [number, number]> = {
  R: [92, 74],
  HR: [33, 21],
  RBI: [98, 63],
  SB: [14, 7],
  AVG: [0.301, 0.268],
  OBP: [0.388, 0.339],
};
function mockCompare(players: PlayerPick[]): CompareData {
  return {
    categories: DEMO_CATS,
    players: players.map((p, i) => ({
      name: p.name,
      pos: p.pos,
      teamAbbr: p.teamAbbr,
      teamId: p.teamId,
      mlbId: p.mlbId,
      stats: Object.fromEntries(DEMO_CATS.map((c) => [c, DEMO_LINE[c][Math.min(i, 1)]])),
    })),
  };
}

/** Fetch a head-to-head comparison for the picked players (need ≥2). Live → GET
 *  /api/compare?ids=…; live errors propagate (HIGH-3) so the caller surfaces them
 *  instead of fabricating a demo line for the user's real picks. Mock (off-live,
 *  or live with an empty response): a demo comparison. */
export async function fetchCompare(players: PlayerPick[]): Promise<CompareData> {
  if (players.length < 2) return { categories: [], players: [] };
  return liveOrMock(
    async () => {
      const api = await apiGet<ApiCompareResponse>("/compare", { ids: players.map((p) => p.id).join(",") });
      return (api.players?.length ?? 0) > 0 ? apiCompareToData(api) : mockCompare(players);
    },
    () => mockCompare(players),
  );
}
