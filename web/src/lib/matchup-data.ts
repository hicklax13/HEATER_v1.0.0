/**
 * Mock Matchup data — this week's H2H category battle vs the opponent.
 * Swap this module for the API client in Sub-project B; the shape is the contract.
 */
export type CatStatus = "win" | "loss" | "tossup";

export interface CatMatchup {
  key: string;
  group: "Hitting" | "Pitching";
  you: string;
  opp: string;
  winPct: number; // your win probability for this category (0–100)
  inverse?: boolean; // lower is better (L, ERA, WHIP)
}

export interface MatchupData {
  week: number;
  daysLeft: number;
  winPct: number; // overall matchup win probability
  you: { name: string; record: string; logo: string };
  opp: { name: string; record: string; logo: string };
  categories: CatMatchup[];
}

export function catStatus(winPct: number): CatStatus {
  if (winPct >= 55) return "win";
  if (winPct <= 44) return "loss";
  return "tossup";
}

export const MATCHUP: MatchupData = {
  week: 12,
  daysLeft: 4,
  winPct: 46,
  you: { name: "Team Hickey", record: "3–7–1", logo: "/brand/team-logo-placeholder.svg" },
  opp: { name: "The Good The Vlad The Ugly", record: "7–4–0", logo: "/brand/team-logo-opponent.svg" },
  categories: [
    { key: "R", group: "Hitting", you: "512", opp: "495", winPct: 58 },
    { key: "HR", group: "Hitting", you: "142", opp: "126", winPct: 72 },
    { key: "RBI", group: "Hitting", you: "505", opp: "498", winPct: 53 },
    { key: "SB", group: "Hitting", you: "48", opp: "71", winPct: 16 },
    { key: "AVG", group: "Hitting", you: ".262", opp: ".256", winPct: 59 },
    { key: "OBP", group: "Hitting", you: ".336", opp: ".333", winPct: 51 },
    { key: "W", group: "Pitching", you: "44", opp: "43", winPct: 52 },
    { key: "L", group: "Pitching", you: "39", opp: "35", winPct: 38, inverse: true },
    { key: "SV", group: "Pitching", you: "28", opp: "41", winPct: 27 },
    { key: "K", group: "Pitching", you: "615", opp: "558", winPct: 66 },
    { key: "ERA", group: "Pitching", you: "4.02", opp: "3.68", winPct: 36, inverse: true },
    { key: "WHIP", group: "Pitching", you: "1.25", opp: "1.18", winPct: 41, inverse: true },
  ],
};

export function fetchMatchup(delayMs = 600): Promise<MatchupData> {
  return new Promise((resolve) => setTimeout(() => resolve(MATCHUP), delayMs));
}
