import { apiGet } from "@/lib/api/client";
import type { ApiPlayerSearchResponse } from "@/lib/api/types";
import type { PlayerRef } from "./types";

/**
 * Pool-wide player search (powers the trade-builder pickers; reusable for
 * compare-any-player + databank). Live → GET /api/players/search (any player,
 * rostered or FA, with HEATER id). Off-live → a small notable-player list so the
 * showcase still works. Returns PlayerPicks (frontend PlayerRef + the HEATER id).
 */
export interface PlayerPick extends PlayerRef {
  id: number; // HEATER player_id — fed to /api/trade/evaluate, /api/compare, etc.
}

// Notable players for off-live (mock) search, so the builder works in the showcase.
const MOCK_POOL: PlayerPick[] = [
  { id: 1, name: "Aaron Judge", pos: "RF", teamAbbr: "NYY", teamId: 147, mlbId: 592450 },
  { id: 2, name: "Shohei Ohtani", pos: "DH", teamAbbr: "LAD", teamId: 119, mlbId: 660271 },
  { id: 3, name: "Bobby Witt Jr.", pos: "SS", teamAbbr: "KC", teamId: 118, mlbId: 677951 },
  { id: 4, name: "Tarik Skubal", pos: "SP", teamAbbr: "DET", teamId: 116, mlbId: 669373 },
  { id: 5, name: "Juan Soto", pos: "RF", teamAbbr: "NYM", teamId: 121, mlbId: 665742 },
  { id: 6, name: "Elly De La Cruz", pos: "SS", teamAbbr: "CIN", teamId: 113, mlbId: 682829 },
  { id: 7, name: "Corbin Carroll", pos: "OF", teamAbbr: "ARI", teamId: 109, mlbId: 682998 },
  { id: 8, name: "Trea Turner", pos: "SS", teamAbbr: "PHI", teamId: 143, mlbId: 607208 },
  { id: 9, name: "Emmanuel Clase", pos: "RP", teamAbbr: "CLE", teamId: 114, mlbId: 661403 },
  { id: 10, name: "Marcell Ozuna", pos: "Util", teamAbbr: "PIT", teamId: 134, mlbId: 542303 },
  { id: 11, name: "Logan Gilbert", pos: "SP", teamAbbr: "SEA", teamId: 136, mlbId: 669302 },
  { id: 12, name: "José Ramírez", pos: "3B", teamAbbr: "CLE", teamId: 114, mlbId: 608070 },
];

/** Search players by name. Needs ≥2 chars. Live → /api/players/search; off-live →
 *  the notable-player mock list. Never throws (errors → []). */
export async function searchPlayers(q: string): Promise<PlayerPick[]> {
  const query = q.trim();
  if (query.length < 2) return [];
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiPlayerSearchResponse>("/players/search", { q: query, limit: 25 });
      return (api.results ?? []).map((p) => ({
        id: p.id,
        name: p.name,
        pos: p.positions,
        teamAbbr: p.team_abbr ?? "",
        teamId: p.team_id ?? 0,
        mlbId: p.mlb_id ?? 0,
      }));
    } catch {
      return [];
    }
  }
  const ql = query.toLowerCase();
  return MOCK_POOL.filter((p) => p.name.toLowerCase().includes(ql));
}
