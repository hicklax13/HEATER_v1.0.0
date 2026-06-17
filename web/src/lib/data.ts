import type { MyTeamData } from "./types";

/**
 * Mock My Team data — real Team Hickey numbers + real MLB ids for headshots.
 * Swap this module for the API client in Sub-project B; the shape is the contract.
 */
export const MY_TEAM: MyTeamData = {
  eyebrow: "Season · Week 12 · Team",
  teamName: "Team Hickey",
  subline: "3–7–1 · 10th of 12 · 7 GB from 1st",
  freshnessMinutes: 14,
  statusChips: [
    { label: "2 On IL", tone: "bad", icon: "activity" },
    { label: "Lineup Set", tone: "ok", icon: "check" },
    { label: "3 News", tone: "info", icon: "bell" },
  ],
  matchup: {
    youName: "Team Hickey",
    youRecord: "3–7–1",
    youLogo: "/brand/team-logo-placeholder.svg",
    oppName: "The Good The Vlad The Ugly",
    oppRecord: "7–4–0",
    oppLogo: "/brand/team-logo-opponent.svg",
    winPct: 46,
    tiePct: 19,
    lossPct: 35,
    projLine: "proj 6–6",
    deltaVsLastWeek: -3,
  },
  moversScope: "mine",
  movers: [
    {
      name: "Aaron Judge", pos: "RF", teamAbbr: "NYY", teamId: 147, mlbId: 592450,
      stat1: { label: "HR", value: "3" }, stat2: { label: "RBI", value: "7" },
      context: "4 Multi-Hit Games", trend: "up", tag: "On Fire",
      spark: [1, 2, 1, 3, 2, 4, 5], ownPct: 100, rosteredByYou: true,
    },
    {
      name: "Bobby Witt Jr.", pos: "SS", teamAbbr: "KC", teamId: 118, mlbId: 677951,
      stat1: { label: "SB", value: "4" }, stat2: { label: "AVG", value: ".360" },
      context: "Hitting In All 6", trend: "up", tag: "Heating",
      spark: [2, 2, 3, 2, 4, 4, 5], ownPct: 100, rosteredByYou: true,
    },
    {
      name: "Tarik Skubal", pos: "SP", teamAbbr: "DET", teamId: 116, mlbId: 669373,
      stat1: { label: "K", value: "13" }, stat2: { label: "ERA", value: "1.50" },
      context: "1 W · 12.0 IP", trend: "up", tag: "Ace Week",
      spark: [3, 4, 3, 5, 4, 5, 6], ownPct: 100, rosteredByYou: true,
    },
    {
      name: "Juan Soto", pos: "RF", teamAbbr: "NYM", teamId: 121, mlbId: 665742,
      stat1: { label: "AVG", value: ".095" }, stat2: { label: "HR", value: "0" },
      context: "2-For-21 Stretch", trend: "down", tag: "Ice Cold",
      spark: [5, 4, 3, 2, 2, 1, 1], ownPct: 96, rosteredByYou: true,
    },
  ],
  lever: {
    categoryKey: "SB",
    headline: "You're 13 stolen bases behind your opponent — your single biggest gap.",
    behindBy: 13,
    pickups: [
      { name: "Corbin Carroll", pos: "OF", teamAbbr: "ARI", teamId: 109, mlbId: 682998, projStat: { label: "proj SB", value: "6" } },
      { name: "Elly De La Cruz", pos: "SS", teamAbbr: "CIN", teamId: 113, mlbId: 682829, projStat: { label: "proj SB", value: "5" } },
      { name: "Trea Turner", pos: "SS", teamAbbr: "PHI", teamId: 143, mlbId: 607208, projStat: { label: "proj SB", value: "4" } },
    ],
  },
  categories: [
    { key: "HR", you: "18", opp: "14", edge: "+4", edgeDir: "good", winPct: 71, higherBetter: true, spark: [10, 11, 12, 13, 14, 16, 18] },
    { key: "K", you: "312", opp: "281", edge: "+31", edgeDir: "good", winPct: 64, higherBetter: true, spark: [40, 90, 140, 190, 240, 280, 312] },
    { key: "OBP", you: ".338", opp: ".334", edge: "+.004", edgeDir: "even", winPct: 53, higherBetter: true, spark: [330, 336, 333, 339, 335, 337, 338] },
    { key: "ERA", you: "4.10", opp: "3.62", edge: "−0.48", edgeDir: "bad", winPct: 38, higherBetter: false, spark: [360, 372, 380, 372, 395, 402, 410] },
    { key: "SV", you: "9", opp: "15", edge: "−6", edgeDir: "bad", winPct: 29, higherBetter: true, spark: [2, 3, 4, 5, 6, 8, 9] },
    { key: "SB", you: "9", opp: "22", edge: "−13", edgeDir: "bad", winPct: 12, higherBetter: true, spark: [1, 2, 3, 4, 6, 8, 9], isLever: true },
  ],
  ops: [
    { key: "ip", label: "IP pace", value: 38, total: 54, unit: "IP", verdict: "On pace to hit the 54 IP minimum.", status: "ok" },
    { key: "moves", label: "Moves left", value: 7, total: 10, unit: "", verdict: "3 used — comfortable burn rate.", status: "ok" },
    { key: "roster", label: "Roster", value: 23, total: 27, unit: "", verdict: "2 new injuries this week — check IL.", status: "warn" },
  ],
  trajectory: [
    { week: 1, rank: 7 }, { week: 2, rank: 6 }, { week: 3, rank: 5 }, { week: 4, rank: 4 },
    { week: 5, rank: 6 }, { week: 6, rank: 7 }, { week: 7, rank: 6 }, { week: 8, rank: 8 },
    { week: 9, rank: 9 }, { week: 10, rank: 8 }, { week: 11, rank: 10 }, { week: 12, rank: 10 },
  ],
  playoffCutRank: 4,
};

/** Simulated async fetch so the UI can exercise loading → loaded. */
export function fetchMyTeam(delayMs = 700): Promise<MyTeamData> {
  return new Promise((resolve) => setTimeout(() => resolve(MY_TEAM), delayMs));
}
