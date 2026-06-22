import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { liveOrMock } from "@/lib/api/live";

/**
 * Hitter Matchup 7-day grid data. Mock by default; with NEXT_PUBLIC_HEATER_LIVE=1,
 * fetchHitterMatchups wires to /api/schedule/hitter-matchups and falls back to this
 * mock on any error or empty response. (Single-league: team_name "Team Hickey" until M4.)
 */

export type HitterAvailability = "yours" | "other";
export type Band = "easy" | "medium" | "tough";

export interface HitterCell {
  oppSp: PlayerRef | null;
  oppSpThrows: "L" | "R" | "";
  opponent: string; // the MLB team the batting team plays
  isHome: boolean; // true = the BATTING team is at home
  difficulty: number; // 0-100 (HIGHER = better matchup for the hitters)
  band: Band;
  status: string;
  confidence: string;
}

export interface HitterRow {
  team: string; // batting team abbr
  cells: (HitterCell | null)[]; // aligned to days[]; null = off-day
  totals: { games: number; vsRhp: number; vsLhp: number };
  matchupsRank: number; // 1 = easiest weekly hitting schedule
  availability: HitterAvailability;
}

export interface HitterMatchupsData {
  days: string[]; // YYYY-MM-DD column dates
  teams: HitterRow[];
}

// ---- API shapes (snake_case; minimal local mirror of the contract) ----
interface ApiOppSp {
  name: string;
  positions: string;
  mlb_id?: number | null;
  team_abbr?: string | null;
  team_id?: number | null;
}
interface ApiHitterCell {
  opp_sp?: ApiOppSp | null;
  opp_sp_throws?: string;
  opponent?: string;
  is_home?: boolean;
  difficulty?: number;
  band?: string;
  status?: string;
  confidence?: string;
}
interface ApiHitterGrid {
  days?: string[];
  teams?: {
    team: string;
    cells: (ApiHitterCell | null)[];
    totals?: { games?: number; vs_rhp?: number; vs_lhp?: number };
    matchups_rank?: number;
    availability?: string;
  }[];
}

function toRef(p: ApiOppSp): PlayerRef {
  return {
    name: p.name,
    pos: p.positions,
    teamAbbr: p.team_abbr ?? "",
    teamId: p.team_id ?? 0,
    mlbId: p.mlb_id ?? 0,
  };
}

function apiToData(api: ApiHitterGrid): HitterMatchupsData {
  return {
    days: api.days ?? [],
    teams: (api.teams ?? []).map((t) => ({
      team: t.team,
      cells: (t.cells ?? []).map((c) =>
        c
          ? {
              oppSp: c.opp_sp ? toRef(c.opp_sp) : null,
              oppSpThrows: (["L", "R"].includes(c.opp_sp_throws ?? "") ? c.opp_sp_throws : "") as "L" | "R" | "",
              opponent: c.opponent ?? "",
              isHome: c.is_home ?? false,
              difficulty: c.difficulty ?? 0,
              band: (["easy", "medium", "tough"].includes(c.band ?? "") ? c.band : "medium") as Band,
              status: c.status ?? "",
              confidence: c.confidence ?? "",
            }
          : null,
      ),
      totals: {
        games: t.totals?.games ?? 0,
        vsRhp: t.totals?.vs_rhp ?? 0,
        vsLhp: t.totals?.vs_lhp ?? 0,
      },
      matchupsRank: t.matchups_rank ?? 0,
      availability: (["yours", "other"].includes(t.availability ?? "") ? t.availability : "other") as HitterAvailability,
    })),
  };
}

// ---- mock ----
const DAYS = ["2026-06-21", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-27"];

const spRef = (
  name: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
): PlayerRef => ({ name, pos: "SP", teamAbbr, teamId, mlbId });

const hcell = (
  sp: PlayerRef | null,
  throws: "L" | "R" | "",
  opp: string,
  home: boolean,
  diff: number,
): HitterCell => ({
  oppSp: sp,
  oppSpThrows: throws,
  opponent: opp,
  isHome: home,
  difficulty: diff,
  band: diff >= 60 ? "easy" : diff <= 40 ? "tough" : "medium",
  status: "",
  confidence: "HIGH",
});

const N: null = null;

export const HITTER_MATCHUPS_MOCK: HitterMatchupsData = {
  days: DAYS,
  teams: [
    {
      team: "NYY",
      cells: [
        hcell(spRef("Tarik Skubal", "DET", 116, 669373), "L", "DET", false, 82),
        hcell(spRef("Aaron Civale", "TB", 139, 656302), "R", "TB", true, 71),
        N,
        hcell(spRef("Ranger Suarez", "PHI", 143, 622250), "L", "PHI", false, 55),
        N,
        hcell(spRef("Dean Kremer", "BAL", 110, 656756), "R", "BAL", true, 74),
        N,
      ],
      totals: { games: 5, vsRhp: 3, vsLhp: 2 },
      matchupsRank: 2,
      availability: "yours",
    },
    {
      team: "LAD",
      cells: [
        hcell(spRef("Logan Gilbert", "SEA", 136, 669302), "R", "SEA", true, 68),
        N,
        hcell(spRef("Sandy Alcantara", "MIA", 146, 645261), "R", "MIA", false, 77),
        N,
        hcell(spRef("Garrett Crochet", "BOS", 111, 676979), "L", "BOS", true, 44),
        N,
        hcell(spRef("Hunter Greene", "CIN", 113, 668881), "R", "CIN", false, 80),
      ],
      totals: { games: 4, vsRhp: 3, vsLhp: 1 },
      matchupsRank: 4,
      availability: "other",
    },
    {
      team: "ATL",
      cells: [
        N,
        hcell(spRef("Gerrit Cole", "NYY", 147, 543037), "R", "NYY", false, 36),
        hcell(spRef("Framber Valdez", "HOU", 117, 664285), "L", "HOU", true, 72),
        N,
        hcell(spRef("Paul Skenes", "PIT", 134, 694973), "R", "PIT", false, 38),
        hcell(spRef("Brayan Bello", "BOS", 111, 678394), "R", "BOS", true, 69),
        N,
      ],
      totals: { games: 4, vsRhp: 3, vsLhp: 1 },
      matchupsRank: 6,
      availability: "other",
    },
    {
      team: "HOU",
      cells: [
        hcell(spRef("Spencer Strider", "ATL", 144, 675911), "R", "ATL", false, 58),
        N,
        hcell(spRef("Cole Ragans", "KC", 118, 666142), "L", "KC", true, 83),
        hcell(spRef("Seth Lugo", "KC", 118, 607625), "R", "KC", true, 78),
        N,
        hcell(spRef("Yoshinobu Yamamoto", "LAD", 119, 808967), "R", "LAD", false, 31),
        N,
      ],
      totals: { games: 4, vsRhp: 3, vsLhp: 1 },
      matchupsRank: 3,
      availability: "other",
    },
  ],
};

/** Live: GET /api/schedule/hitter-matchups → adapt; live errors propagate (HIGH-3)
 *  so usePageData reaches error/locked/unlinked. Mock (off-live, or live with an
 *  empty grid): the in-memory HITTER_MATCHUPS_MOCK after a simulated delay. */
export async function fetchHitterMatchups(delayMs = 500): Promise<HitterMatchupsData> {
  return liveOrMock(
    async () => {
      const api = await apiGet<ApiHitterGrid>("/schedule/hitter-matchups", { days: 7, team_name: getViewerTeam() });
      return (api.teams?.length ?? 0) > 0 ? apiToData(api) : HITTER_MATCHUPS_MOCK;
    },
    () => new Promise<HitterMatchupsData>((resolve) => setTimeout(() => resolve(HITTER_MATCHUPS_MOCK), delayMs)),
  );
}
