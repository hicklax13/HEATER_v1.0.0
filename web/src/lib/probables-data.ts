import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";

/**
 * Probable Pitcher 7-day grid data. Mock by default; with NEXT_PUBLIC_HEATER_LIVE=1,
 * fetchProbables wires to /api/schedule/probables and falls back to this mock on any
 * error or empty response. (Single-league: team_name "Team Hickey" until M4.)
 */

export type Availability = "yours" | "taken" | "available";
export type Band = "easy" | "medium" | "tough";

export interface ProbCell {
  pitcher: PlayerRef | null;
  opponent: string; // opposing team abbr
  isHome: boolean;
  difficulty: number; // 0-100 (HIGHER = easier matchup to stream against)
  band: Band;
  twoStart: boolean;
  availability: Availability;
  rosteredBy: string | null;
  status: string;
  confidence: string;
}

export interface ProbRow {
  team: string;
  cells: (ProbCell | null)[]; // aligned to days[]; null = off-day
}

export interface ProbablesData {
  days: string[]; // YYYY-MM-DD column dates
  teams: ProbRow[];
}

// ---- API shapes (snake_case; minimal local mirror of the contract) ----
interface ApiCell {
  pitcher: { name: string; positions: string; mlb_id?: number | null; team_abbr?: string | null; team_id?: number | null } | null;
  opponent?: string;
  is_home?: boolean;
  difficulty?: number;
  band?: string;
  two_start?: boolean;
  availability?: string;
  rostered_by?: string | null;
  status?: string;
  confidence?: string;
}
interface ApiProbableGrid {
  days?: string[];
  teams?: { team: string; cells: (ApiCell | null)[] }[];
}

function toRef(p: NonNullable<ApiCell["pitcher"]>): PlayerRef {
  return { name: p.name, pos: p.positions, teamAbbr: p.team_abbr ?? "", teamId: p.team_id ?? 0, mlbId: p.mlb_id ?? 0 };
}

function apiToData(api: ApiProbableGrid): ProbablesData {
  return {
    days: api.days ?? [],
    teams: (api.teams ?? []).map((t) => ({
      team: t.team,
      cells: (t.cells ?? []).map((c) =>
        c
          ? {
              pitcher: c.pitcher ? toRef(c.pitcher) : null,
              opponent: c.opponent ?? "",
              isHome: c.is_home ?? false,
              difficulty: c.difficulty ?? 0,
              band: (["easy", "medium", "tough"].includes(c.band ?? "") ? c.band : "medium") as Band,
              twoStart: c.two_start ?? false,
              availability: (["yours", "taken", "available"].includes(c.availability ?? "")
                ? c.availability
                : "available") as Availability,
              rosteredBy: c.rostered_by ?? null,
              status: c.status ?? "",
              confidence: c.confidence ?? "",
            }
          : null,
      ),
    })),
  };
}

// ---- mock ----
const DAYS = ["2026-06-21", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-27"];

const ref = (name: string, teamAbbr: string, teamId: number, mlbId: number): PlayerRef => ({
  name,
  pos: "SP",
  teamAbbr,
  teamId,
  mlbId,
});

const cell = (p: PlayerRef, opp: string, home: boolean, diff: number, av: Availability, two = false): ProbCell => ({
  pitcher: p,
  opponent: opp,
  isHome: home,
  difficulty: diff,
  band: diff >= 60 ? "easy" : diff <= 40 ? "tough" : "medium",
  twoStart: two,
  availability: av,
  rosteredBy: av === "yours" ? "Team Hickey" : av === "taken" ? "Rivals" : null,
  status: "",
  confidence: "HIGH",
});

const N: null = null;

export const PROBABLES_MOCK: ProbablesData = {
  days: DAYS,
  teams: [
    {
      team: "DET",
      cells: [
        cell(ref("Tarik Skubal", "DET", 116, 669373), "CWS", true, 86, "yours", true),
        N,
        N,
        cell(ref("Tarik Skubal", "DET", 116, 669373), "KC", false, 78, "yours", true),
        N,
        cell(ref("Jack Flaherty", "DET", 116, 656427), "BAL", true, 54, "taken"),
        N,
      ],
    },
    {
      team: "PIT",
      cells: [
        cell(ref("Paul Skenes", "PIT", 134, 694973), "MIA", false, 83, "taken"),
        N,
        N,
        cell(ref("Jared Jones", "PIT", 134, 700066), "CHC", true, 49, "available"),
        N,
        N,
        cell(ref("Paul Skenes", "PIT", 134, 694973), "STL", true, 71, "taken"),
      ],
    },
    {
      team: "BOS",
      cells: [
        N,
        cell(ref("Garrett Crochet", "BOS", 111, 676979), "TB", true, 74, "yours", true),
        N,
        N,
        cell(ref("Garrett Crochet", "BOS", 111, 676979), "NYY", false, 58, "yours", true),
        N,
        cell(ref("Brayan Bello", "BOS", 111, 678394), "TOR", true, 46, "available"),
      ],
    },
    {
      team: "KC",
      cells: [
        cell(ref("Cole Ragans", "KC", 118, 666142), "DET", false, 62, "taken"),
        N,
        N,
        cell(ref("Seth Lugo", "KC", 118, 607625), "DET", false, 51, "available"),
        N,
        N,
        N,
      ],
    },
    {
      team: "CIN",
      cells: [
        cell(ref("Hunter Greene", "CIN", 113, 668881), "COL", false, 38, "taken"),
        N,
        N,
        N,
        cell(ref("Nick Lodolo", "CIN", 113, 666157), "ARI", true, 55, "available"),
        N,
        cell(ref("Hunter Greene", "CIN", 113, 668881), "LAD", true, 33, "taken"),
      ],
    },
    {
      team: "SEA",
      cells: [
        cell(ref("Logan Gilbert", "SEA", 136, 669302), "HOU", true, 69, "available"),
        N,
        cell(ref("George Kirby", "SEA", 136, 669923), "HOU", true, 64, "available"),
        N,
        N,
        cell(ref("Bryan Woo", "SEA", 136, 693433), "LAA", false, 72, "yours"),
        N,
      ],
    },
    {
      team: "ATL",
      cells: [
        N,
        cell(ref("Spencer Strider", "ATL", 144, 675911), "WSH", true, 80, "yours", true),
        N,
        N,
        cell(ref("Spencer Strider", "ATL", 144, 675911), "PHI", false, 47, "yours", true),
        N,
        cell(ref("Chris Sale", "ATL", 144, 519242), "NYM", true, 67, "taken"),
      ],
    },
    {
      team: "LAD",
      cells: [
        cell(ref("Yoshinobu Yamamoto", "LAD", 119, 808967), "SF", true, 60, "taken"),
        N,
        N,
        cell(ref("Tyler Glasnow", "LAD", 119, 607192), "SD", false, 44, "taken"),
        N,
        N,
        cell(ref("Yoshinobu Yamamoto", "LAD", 119, 808967), "CIN", false, 57, "taken"),
      ],
    },
  ],
};

/** Mock by default; live (NEXT_PUBLIC_HEATER_LIVE=1) fetches the grid, falling
 *  back to the mock on any error or empty response. */
export async function fetchProbables(delayMs = 500): Promise<ProbablesData> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiProbableGrid>("/schedule/probables", { days: 7, team_name: "Team Hickey" });
      if ((api.teams?.length ?? 0) > 0) return apiToData(api);
    } catch {
      // fall through to mock
    }
  }
  return new Promise((resolve) => setTimeout(() => resolve(PROBABLES_MOCK), delayMs));
}
