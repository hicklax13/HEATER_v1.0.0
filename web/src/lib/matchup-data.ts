/**
 * Mock Matchup data — the live H2H matchup view (Yahoo-style).
 * Swap this module for the API client in Sub-project B; the shape is the contract.
 */
export type GameState = "final" | "live" | "sched" | "none";

export interface MatchPlayer {
  name: string;
  teamAbbr: string;
  teamId: number;
  mlbId: number;
  pos: string; // eligible positions, e.g. "C" or "2B,3B,SS"
  status: string; // game status line
  state: GameState;
  stats: string[]; // 7 values aligned to the hitter/pitcher columns
  badge?: "IL" | "DTD";
}

export interface RosterRow {
  slot: string;
  you: MatchPlayer | null;
  opp: MatchPlayer | null;
}

export interface TeamSide {
  name: string;
  manager: string;
  record: string; // "4-7-1 | 8th"
  score: number; // H2H category score
}

export interface CatCol {
  key: string;
  you: string;
  opp: string;
  win?: "you" | "opp"; // omitted for non-scored context columns (H/AB, IP) or ties
}

export interface LeagueTeam {
  name: string;
  manager: string;
  record: string;
  score: number;
}

export interface MatchupData {
  week: number;
  dateTabs: string[];
  you: TeamSide;
  opp: TeamSide;
  cats: CatCol[];
  hitterColumns: string[];
  pitcherColumns: string[];
  hitters: RosterRow[];
  pitchers: RosterRow[];
  hitterTotals: { you: string[]; opp: string[] };
  pitcherTotals: { you: string[]; opp: string[] };
  league: { a: LeagueTeam; b: LeagueTeam }[];
}

function mp(
  name: string,
  teamAbbr: string,
  teamId: number,
  pos: string,
  status: string,
  state: GameState,
  statline: string,
  badge?: "IL" | "DTD",
): MatchPlayer {
  return { name, teamAbbr, teamId, mlbId: 0, pos, status, state, stats: statline.split(" "), badge };
}

const HIT = ["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"];
const PIT = ["IP", "W", "L", "SV", "K", "ERA", "WHIP"];
const HSCHED = "0/0 0 0 0 0 - -";
const HNONE = "- - - - - - -";
const PNONE = "- - - - - - -";
const PSCHED = "0.0 0 0 0 0 - -";

export const MATCHUP: MatchupData = {
  week: 13,
  dateTabs: ["Live", "Totals", "Mon 6/15", "Tue 6/16", "Wed 6/17", "Thu 6/18", "Fri 6/19", "Sat 6/20", "Sun 6/21"],
  you: { name: "Team Hickey", manager: "Connor", record: "4-7-1 | 8th", score: 7 },
  opp: { name: "Baty Babies", manager: "dandre", record: "6-4-2 | 4th", score: 5 },
  cats: [
    { key: "H/AB", you: "25/88", opp: "18/82" },
    { key: "R", you: "15", opp: "8", win: "you" },
    { key: "HR", you: "6", opp: "3", win: "you" },
    { key: "RBI", you: "14", opp: "9", win: "you" },
    { key: "SB", you: "1", opp: "3", win: "opp" },
    { key: "AVG", you: ".284", opp: ".220", win: "you" },
    { key: "OBP", you: ".326", opp: ".333", win: "opp" },
    { key: "IP", you: "21.0", opp: "38.0" },
    { key: "W", you: "2", opp: "3", win: "opp" },
    { key: "L", you: "0", opp: "2", win: "you" },
    { key: "SV", you: "0", opp: "3", win: "opp" },
    { key: "K", you: "17", opp: "34", win: "opp" },
    { key: "ERA", you: "0.86", opp: "3.79", win: "you" },
    { key: "WHIP", you: "0.95", opp: "1.18", win: "you" },
  ],
  hitterColumns: HIT,
  pitcherColumns: PIT,
  hitters: [
    { slot: "C",
      you: mp("D. Dingler", "DET", 116, "C", "Final (L) 2-4 @ HOU", "final", "1/4 0 0 0 0 .250 .250"),
      opp: mp("I. Herrera", "STL", 138, "C", "Final (L) 1-6 vs SD", "final", "1/3 0 0 0 1 .333 .500") },
    { slot: "1B",
      you: mp("M. Olson", "ATL", 144, "1B", "Top 3rd, 0-5 vs SF · Fielding", "live", "0/1 0 0 0 0 .000 .000"),
      opp: mp("R. Devers", "SF", 137, "1B", "Top 3rd, 5-0 @ ATL · At Bat", "live", "1/1 0 0 1 0 1.000 1.000") },
    { slot: "2B",
      you: mp("N. Gonzales", "PIT", 134, "2B,3B,SS", "9:40 PM @ ATH", "sched", HSCHED),
      opp: mp("B. Lowe", "TB", 139, "2B", "9:40 PM @ ATH", "sched", HSCHED) },
    { slot: "3B",
      you: mp("W. Castro", "COL", 115, "1B,2B,3B,SS,OF", "8:05 PM @ CHC", "sched", HNONE),
      opp: mp("A. Bregman", "CHC", 112, "3B", "8:05 PM vs COL", "sched", HSCHED) },
    { slot: "SS",
      you: mp("D. Swanson", "CHC", 112, "SS", "8:05 PM vs COL", "sched", HSCHED),
      opp: mp("J. Wetherholt", "STL", 138, "2B,3B,SS", "Final (L) 1-6 vs SD", "final", "1/4 0 0 0 0 .250 .250") },
    { slot: "OF",
      you: mp("J. Marsee", "MIA", 146, "OF", "Final (W) 12-4 vs PHI", "final", "1/5 2 1 1 0 .200 .200"),
      opp: mp("C. Carroll", "ARI", 109, "OF", "Final (W) 8-1 vs LAA", "final", "1/5 1 1 4 0 .200 .200") },
    { slot: "OF",
      you: mp("Y. Alvarez", "HOU", 117, "OF", "Final (W) 4-2 vs DET", "final", "1/4 1 1 1 0 .250 .250"),
      opp: mp("J. Chourio", "MIL", 158, "OF", "Top 2nd, 3-0 vs CLE · Fielding", "live", "0/1 0 0 0 0 .000 .000") },
    { slot: "OF",
      you: mp("M. Trout", "LAA", 108, "OF", "Final (L) 1-8 @ ATH", "final", "1/3 0 0 0 0 .333 .500"),
      opp: mp("S. Suzuki", "CHC", 112, "OF", "8:05 PM vs COL", "sched", HSCHED) },
    { slot: "Util",
      you: mp("J. Merrill", "SD", 135, "OF", "Final (W) 6-1 @ STL", "final", "3/5 1 1 2 1 .600 .600"),
      opp: mp("W. Abreu", "BOS", 111, "OF", "Bot 4th, 0-2 vs TOR · Due 6th", "live", "1/2 0 0 0 0 .500 .500") },
    { slot: "Util",
      you: mp("B. Reynolds", "PIT", 134, "OF", "9:40 PM @ ATH", "sched", HSCHED),
      opp: mp("S. Ohtani", "LAD", 119, "Util", "Final (W) 5-4 vs TB", "final", "0/1 0 0 0 0 .000 .000") },
    { slot: "BN",
      you: mp("C. Raleigh", "SEA", 136, "C", "9:40 PM vs BAL", "sched", HSCHED),
      opp: mp("M. Garcia", "KC", 118, "2B,3B,SS,OF", "Final (W) 6-2 @ WSH", "final", HNONE, "DTD") },
    { slot: "BN",
      you: mp("M. Muncy", "LAD", 119, "3B", "Final (W) 5-4 vs TB", "final", "0/1 0 0 0 0 .000 .000"),
      opp: null },
    { slot: "IL",
      you: mp("A. Martínez", "CLE", 114, "2B,OF", "Top 2nd, 0-3 @ MIL", "live", HNONE, "IL"),
      opp: mp("C. Seager", "TEX", 140, "SS", "No Game", "none", HNONE, "IL") },
  ],
  hitterTotals: {
    you: ["7/22", "4", "2", "4", "1", ".318", ".348"],
    opp: ["5/17", "1", "1", "5", "1", ".294", ".333"],
  },
  pitchers: [
    { slot: "SP",
      you: mp("E. Rodriguez", "ARI", 109, "SP", "Final (W) 8-1 vs LAA", "final", "7.0 1 0 0 5 1.29 1.29"),
      opp: mp("G. Cole", "NYY", 147, "SP", "Top 3rd, 4-3 vs CWS", "live", PNONE) },
    { slot: "SP",
      you: mp("K. Bradish", "BAL", 110, "SP", "9:40 PM @ SEA", "sched", PSCHED),
      opp: mp("W. Warren", "NYY", 147, "SP", "Top 3rd, 4-3 vs CWS", "live", PNONE) },
    { slot: "RP",
      you: mp("R. Suarez", "ATL", 144, "RP", "Top 3rd, 0-5 vs SF", "live", PNONE),
      opp: mp("A. Muñoz", "SEA", 136, "RP", "9:40 PM vs BAL", "sched", PNONE) },
    { slot: "RP",
      you: mp("R. Iglesias", "ATL", 144, "RP", "Final (L) 2-7 vs SF", "final", PNONE),
      opp: mp("J. Hader", "HOU", 117, "RP", "Final (W) 4-2 vs DET", "final", "1.0 0 0 1 3 9.00 1.00") },
    { slot: "P",
      you: mp("J. Hoffman", "TOR", 141, "RP", "Bot 2-0 @ BOS", "live", PNONE),
      opp: mp("P. Tolle", "BOS", 111, "SP,RP", "Bot 4th, 0-2 vs TOR", "live", PNONE) },
    { slot: "P",
      you: mp("K. Harrison", "MIL", 158, "SP,RP", "Top 3-0 vs CLE", "live", PNONE),
      opp: mp("S. Alcantara", "MIA", 146, "SP", "Final (W) 12-4 @ PHI", "final", "6.0 1 0 0 6 3.00 1.50") },
    { slot: "P",
      you: mp("P. Lambert", "HOU", 117, "SP", "Final (W) 4-2 vs DET", "final", "7.0 1 0 0 5 1.29 0.29"),
      opp: mp("T. Bibee", "CLE", 114, "SP", "Top 2nd, 0-3 @ MIL", "live", PNONE) },
    { slot: "P",
      you: null,
      opp: mp("J. Assad", "CHC", 112, "SP,RP", "Top 1st, 0-0 vs COL · Pitching", "live", PSCHED) },
    { slot: "BN",
      you: mp("F. Valdez", "DET", 116, "SP", "Final (L) 2-4 @ HOU", "final", PNONE),
      opp: mp("T. Melton", "DET", 116, "SP,RP", "Final (L) 2-4 @ HOU", "final", PNONE) },
    { slot: "BN",
      you: mp("T. Bradley", "MIN", 142, "SP", "No Game", "none", PNONE),
      opp: mp("A. Abbott", "CIN", 113, "SP", "Final (L) 1-9 vs NYM", "final", PNONE) },
    { slot: "BN",
      you: mp("L. Roupp", "SF", 137, "SP", "Top 3rd, 5-0 @ ATL", "live", PNONE),
      opp: mp("N. Eovaldi", "TEX", 140, "SP", "No Game", "none", PNONE) },
    { slot: "BN",
      you: mp("C. Sale", "ATL", 144, "SP", "Top 3rd, 0-5 vs SF", "live", PNONE),
      opp: mp("E. Cabrera", "CHC", 112, "SP", "Top 1st, 0-0 vs COL", "live", PNONE, "DTD") },
    { slot: "IL",
      you: mp("B. Ober", "MIN", 142, "SP", "No Game", "none", PNONE, "IL"),
      opp: mp("S. Strider", "ATL", 144, "SP", "Top 3rd, 0-5 vs SF", "live", PNONE, "IL") },
    { slot: "IL",
      you: mp("G. Crochet", "BOS", 111, "SP", "Bot 4th, 2-0 vs TOR", "live", PNONE, "IL"),
      opp: mp("E. Pagán", "CIN", 113, "RP", "Final (L) 1-9 vs NYM", "final", PNONE, "IL") },
    { slot: "IL",
      you: mp("S. Bieber", "TOR", 141, "SP", "Bot 4th, 2-0 @ BOS", "live", PNONE, "IL"),
      opp: mp("E. Díaz", "LAD", 119, "RP", "Final (W) 5-4 vs TB", "final", PNONE, "IL") },
  ],
  pitcherTotals: {
    you: ["14.0", "2", "0", "0", "10", "1.29", "0.79"],
    opp: ["7.0", "1", "0", "1", "9", "3.86", "1.43"],
  },
  league: [
    { a: { name: "Team Hickey", manager: "Connor", record: "4-7-1 | 8th", score: 7 }, b: { name: "Baty Babies", manager: "dandre", record: "6-4-2 | 4th", score: 5 } },
    { a: { name: "BUBBA CROSBY", manager: "elias", record: "5-5-2 | 6th", score: 10 }, b: { name: "The Good The Vlad The Ugly", manager: "Ricky", record: "4-7-1 | 10th", score: 2 } },
    { a: { name: "Jonny Jockstrap", manager: "Jon", record: "4-7-1 | 9th", score: 7 }, b: { name: "Cyrus The Greats", manager: "Montezuma", record: "2-8-2 | 12th", score: 3 } },
    { a: { name: "My Precious", manager: "Nicholas", record: "8-3-1 | 2nd", score: 4 }, b: { name: "Over the Rembow", manager: "Alex", record: "11-1-0 | 1st", score: 5 } },
    { a: { name: "HUMAN INTELLIGENCE", manager: "LOV3", record: "7-4-1 | 3rd", score: 5 }, b: { name: "Go yanks", manager: "Ben", record: "6-4-3 | 5th", score: 4 } },
    { a: { name: "On a Twosday", manager: "Matt", record: "6-6-0 | 7th", score: 8 }, b: { name: "Going...Going...Gonorrhea", manager: "Sam", record: "2-8-2 | 11th", score: 3 } },
  ],
};

export function fetchMatchup(delayMs = 600): Promise<MatchupData> {
  return new Promise((resolve) => setTimeout(() => resolve(MATCHUP), delayMs));
}
