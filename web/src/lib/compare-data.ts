import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { apiCompareToData } from "@/lib/api/adapters";
import type { ApiCompareResponse } from "@/lib/api/types";
import { PLAYERS } from "@/lib/players-data";

/**
 * Player Compare data — head-to-head category comparison. With
 * NEXT_PUBLIC_HEATER_LIVE=1, fetchCompare wires to GET /api/compare?ids=… (pool-
 * backed → real locally) and falls back to a mock synthesized from the FA pool.
 * The picker sources players from the FA pool (players-data) — its HEATER `id`
 * is what /api/compare consumes.
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

const NON_NUMERIC = /[^0-9.\-]/g;

/** Mock: synthesize a comparison from the selected FA-pool players by parsing
 *  their display-string stats into numbers. Sparse by nature (each FA carries
 *  ~3 stats); live /api/compare returns the full category set. */
function mockCompare(ids: number[]): CompareData {
  const picked = ids
    .map((id) => PLAYERS.freeAgents.find((f) => f.id === id))
    .filter((f): f is NonNullable<typeof f> => !!f);
  const categories: string[] = [];
  for (const p of picked) {
    for (const s of p.stats) if (!categories.includes(s.label)) categories.push(s.label);
  }
  const players: ComparePlayerData[] = picked.map((p) => {
    const stats: Record<string, number> = {};
    for (const s of p.stats) {
      const n = parseFloat(s.value.replace(NON_NUMERIC, ""));
      if (Number.isFinite(n)) stats[s.label] = n;
    }
    return { name: p.name, pos: p.pos, teamAbbr: p.teamAbbr, teamId: p.teamId, mlbId: p.mlbId, stats };
  });
  return { categories, players };
}

/** Fetch a head-to-head comparison for the given HEATER player ids (need ≥2).
 *  Live → /api/compare?ids=…; falls back to the FA-pool-synthesized mock on
 *  error / empty. */
export async function fetchCompare(ids: number[]): Promise<CompareData> {
  if (ids.length < 2) return { categories: [], players: [] };
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiCompareResponse>("/compare", { ids: ids.join(",") });
      if ((api.players?.length ?? 0) > 0) return apiCompareToData(api);
    } catch {
      // fall through to mock
    }
  }
  return mockCompare(ids);
}
