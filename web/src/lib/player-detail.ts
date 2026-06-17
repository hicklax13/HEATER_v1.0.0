import type { PlayerRef } from "./types";
import { teamBrand } from "./teams";

/* ---------- types ---------- */
export interface GameRow {
  date: string;
  opp: string;
  result: string; // "W 3-1" / "L 1-8" / "Sat 4:10 PM" (upcoming)
  upcoming: boolean;
  line: string[]; // cells aligned to columns()
}
export interface StatRow {
  cat: string;
  season: string;
  l30: string;
  l14: string;
  l7: string;
  avg: string;
  std: string;
}
export interface PriorRow {
  cat: string;
  y2025: string;
  y2024: string;
}
export interface ProjRow {
  cat: string;
  today: string;
  n7: string;
  n14: string;
  n30: string;
  ros: string;
  avg: string;
  std: string;
}
export interface HistoryEvent {
  kind: "drafted" | "traded" | "added" | "dropped";
  date: string;
  text: string;
  member: string;
}
export interface PlayerDetail {
  mlbId: number;
  teamId: number;
  name: string;
  pos: string;
  bats: string;
  jersey: string;
  teamName: string;
  isPitcher: boolean;
  ownPct: number;
  ownDelta: number;
  rosteredBy: string;
  headline: { label: string; value: string }[];
  ranks: { label: string; value: string }[];
  gameColumns: string[];
  gameLog: GameRow[];
  stats: StatRow[];
  prior: { y2025Rank: number; y2024Rank: number; rows: PriorRow[] };
  projections: ProjRow[];
  history: HistoryEvent[];
}

/* ---------- bases ---------- */
type CatMap = Record<string, number>;
interface Base {
  name: string;
  pos: string;
  teamId: number;
  bats: string;
  jersey: string;
  gp: number;
  ip?: number;
  rankOverall: number;
  rankPos: number;
  pitcher?: boolean;
  ownPct?: number;
  ownDelta?: number;
  season: CatMap;
  y2025: CatMap;
  y2024: CatMap;
  y2025Rank: number;
  y2024Rank: number;
  history: HistoryEvent[];
}

const HIT = ["R", "HR", "RBI", "SB", "AVG", "OBP"];
const PIT = ["W", "L", "SV", "K", "ERA", "WHIP"];
const RATE = new Set(["AVG", "OBP", "ERA", "WHIP"]);

const BASE: Record<number, Base> = {
  592450: {
    name: "Aaron Judge", pos: "RF", teamId: 147, bats: "Bats Right", jersey: "#99",
    gp: 70, rankOverall: 2, rankPos: 1,
    season: { R: 58, HR: 24, RBI: 60, SB: 6, AVG: 0.322, OBP: 0.44 },
    y2025: { R: 122, HR: 53, RBI: 144, SB: 9, AVG: 0.322, OBP: 0.458 },
    y2024: { R: 122, HR: 58, RBI: 144, SB: 10, AVG: 0.322, OBP: 0.458 },
    y2025Rank: 1, y2024Rank: 1,
    history: [{ kind: "drafted", date: "3/7, 6:02 PM", text: "Drafted 1st overall (1st round)", member: "Team Hickey" }],
  },
  677951: {
    name: "Bobby Witt Jr.", pos: "SS", teamId: 118, bats: "Bats Right", jersey: "#7",
    gp: 71, rankOverall: 4, rankPos: 1,
    season: { R: 55, HR: 18, RBI: 52, SB: 22, AVG: 0.305, OBP: 0.36 },
    y2025: { R: 110, HR: 32, RBI: 98, SB: 31, AVG: 0.332, OBP: 0.389 },
    y2024: { R: 125, HR: 32, RBI: 109, SB: 31, AVG: 0.332, OBP: 0.389 },
    y2025Rank: 3, y2024Rank: 2,
    history: [{ kind: "drafted", date: "3/7, 6:05 PM", text: "Drafted 5th overall (1st round)", member: "Team Hickey" }],
  },
  669373: {
    name: "Tarik Skubal", pos: "SP", teamId: 116, bats: "Throws Left", jersey: "#29",
    gp: 15, ip: 95, rankOverall: 6, rankPos: 1, pitcher: true,
    season: { W: 9, L: 3, SV: 0, K: 130, ERA: 2.1, WHIP: 0.95 },
    y2025: { W: 18, L: 4, SV: 0, K: 241, ERA: 2.39, WHIP: 0.92 },
    y2024: { W: 18, L: 4, SV: 0, K: 228, ERA: 2.39, WHIP: 0.92 },
    y2025Rank: 1, y2024Rank: 1,
    history: [{ kind: "drafted", date: "3/7, 6:14 PM", text: "Drafted 18th overall (2nd round)", member: "Team Hickey" }],
  },
  665742: {
    name: "Juan Soto", pos: "RF", teamId: 121, bats: "Bats Left", jersey: "#22",
    gp: 70, rankOverall: 9, rankPos: 5, ownPct: 96, ownDelta: -2,
    season: { R: 50, HR: 19, RBI: 45, SB: 5, AVG: 0.265, OBP: 0.39 },
    y2025: { R: 128, HR: 41, RBI: 105, SB: 12, AVG: 0.288, OBP: 0.419 },
    y2024: { R: 128, HR: 41, RBI: 109, SB: 7, AVG: 0.288, OBP: 0.419 },
    y2025Rank: 5, y2024Rank: 3,
    history: [
      { kind: "drafted", date: "3/7, 6:08 PM", text: "Drafted 8th overall (1st round)", member: "Over the Rembow" },
      { kind: "traded", date: "5/2, 1:43 PM", text: "Traded to Team Hickey for Gunnar Henderson", member: "Team Hickey" },
    ],
  },
};

/* ---------- formatting ---------- */
function fmt(cat: string, v: number): string {
  if (cat === "AVG" || cat === "OBP") return v.toFixed(3).replace(/^0/, "");
  if (cat === "ERA" || cat === "WHIP") return v.toFixed(2);
  return String(Math.round(v));
}

function hitterLog(): { columns: string[]; rows: GameRow[] } {
  const columns = ["Date", "Opp", "Result", "H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"];
  const played: [string, string, string, string, number, number, number, number, string, string][] = [
    ["Jun 14", "@NYM", "L 1-8", "1/4", 0, 0, 0, 0, ".250", ".250"],
    ["Jun 13", "@NYM", "W 3-1", "1/4", 1, 0, 0, 0, ".250", ".250"],
    ["Jun 12", "@NYM", "L 5-7", "1/5", 1, 1, 1, 0, ".200", ".200"],
    ["Jun 10", "@CWS", "L 1-2", "1/3", 0, 0, 0, 0, ".333", ".500"],
    ["Jun 9", "@CWS", "L 5-6", "2/4", 2, 2, 3, 0, ".500", ".600"],
    ["Jun 7", "PIT", "W 3-2", "0/4", 0, 0, 0, 0, ".000", ".000"],
    ["Jun 6", "PIT", "W 6-3", "1/4", 1, 0, 0, 1, ".250", ".250"],
    ["Jun 5", "PIT", "W 6-3", "2/5", 1, 0, 0, 0, ".400", ".400"],
    ["Jun 4", "TOR", "L 2-7", "0/3", 0, 0, 1, 0, ".000", ".000"],
    ["Jun 3", "TOR", "W 4-1", "2/4", 1, 1, 2, 0, ".500", ".600"],
  ];
  const upcoming: [string, string, string][] = [
    ["Jun 20", "MIL", "Sat 4:10 PM"],
    ["Jun 19", "MIL", "Fri 7:15 PM"],
    ["Jun 18", "SF", "Thu 7:15 PM"],
  ];
  const rows: GameRow[] = [
    ...upcoming.map((u) => ({ date: u[0], opp: u[1], result: u[2], upcoming: true, line: ["—", "—", "—", "—", "—", "—"] })),
    ...played.map((p) => ({
      date: p[0], opp: p[1], result: p[2], upcoming: false,
      line: [p[3], String(p[4]), String(p[5]), String(p[6]), String(p[7]), p[8], p[9]].slice(0, 7),
    })),
  ];
  return { columns, rows };
}

function pitcherLog(): { columns: string[]; rows: GameRow[] } {
  const columns = ["Date", "Opp", "Result", "IP", "H", "ER", "K", "BB", "Dec", "ERA", "WHIP"];
  const played: [string, string, string, string, number, number, number, number, string, string, string][] = [
    ["Jun 13", "@NYM", "W 3-1", "7.0", 4, 1, 9, 1, "W", "1.29", "0.71"],
    ["Jun 8", "@CWS", "W 5-1", "6.2", 5, 1, 8, 2, "W", "1.35", "1.05"],
    ["Jun 3", "TOR", "L 2-3", "6.0", 6, 3, 7, 1, "L", "4.50", "1.17"],
    ["May 28", "BAL", "W 4-2", "7.0", 3, 1, 11, 0, "W", "1.29", "0.43"],
    ["May 23", "@SEA", "ND 5-4", "6.0", 5, 2, 6, 3, "ND", "3.00", "1.33"],
    ["May 18", "HOU", "W 6-2", "7.1", 4, 2, 10, 1, "W", "2.45", "0.68"],
  ];
  const upcoming: [string, string, string][] = [
    ["Jun 18", "SF", "Wed 7:15 PM"],
    ["Jun 23", "@MIL", "Mon 8:10 PM"],
  ];
  const rows: GameRow[] = [
    ...upcoming.map((u) => ({ date: u[0], opp: u[1], result: u[2], upcoming: true, line: ["—", "—", "—", "—", "—", "—", "—", "—"] })),
    ...played.map((p) => ({
      date: p[0], opp: p[1], result: p[2], upcoming: false,
      line: [p[3], String(p[4]), String(p[5]), String(p[6]), String(p[7]), p[8], p[9], p[10]],
    })),
  ];
  return { columns, rows };
}

/* ---------- synth ---------- */
function statRows(b: Base, cats: string[]): StatRow[] {
  return cats.map((c) => {
    const t = b.season[c];
    if (RATE.has(c)) {
      const std = c === "AVG" || c === "OBP" ? "±.045" : "±0.40";
      return { cat: c, season: fmt(c, t), l30: fmt(c, t + 0.012), l14: fmt(c, t - 0.02), l7: fmt(c, t + 0.03), avg: fmt(c, t), std };
    }
    const perG = t / b.gp;
    return {
      cat: c, season: String(t),
      l30: String(Math.round(t * 0.3)),
      l14: String(Math.round(t * 0.15)),
      l7: String(Math.round(t * 0.075)),
      avg: perG.toFixed(2),
      std: `±${Math.max(0.3, Math.sqrt(perG)).toFixed(2)}`,
    };
  });
}

function projRows(b: Base, cats: string[]): ProjRow[] {
  const REMAIN = 92;
  return cats.map((c) => {
    const t = b.season[c];
    if (RATE.has(c)) {
      return { cat: c, today: fmt(c, t), n7: fmt(c, t), n14: fmt(c, t), n30: fmt(c, t), ros: fmt(c, t), avg: fmt(c, t), std: c === "AVG" || c === "OBP" ? "±.040" : "±0.35" };
    }
    const perG = t / b.gp;
    return {
      cat: c,
      today: perG.toFixed(1),
      n7: String(Math.round(perG * 7)),
      n14: String(Math.round(perG * 14)),
      n30: String(Math.round(perG * 30)),
      ros: String(Math.round(perG * REMAIN)),
      avg: perG.toFixed(2),
      std: `±${Math.max(0.3, Math.sqrt(perG)).toFixed(2)}`,
    };
  });
}

function fallback(m: PlayerRef): Base {
  return {
    name: m.name, pos: m.pos, teamId: m.teamId, bats: "—", jersey: "", gp: 70,
    rankOverall: 120, rankPos: 40, pitcher: /SP|RP|^P$/.test(m.pos),
    season: { R: 30, HR: 8, RBI: 30, SB: 4, AVG: 0.255, OBP: 0.32 },
    y2025: { R: 60, HR: 16, RBI: 60, SB: 8, AVG: 0.255, OBP: 0.32 },
    y2024: { R: 60, HR: 16, RBI: 60, SB: 8, AVG: 0.255, OBP: 0.32 },
    y2025Rank: 120, y2024Rank: 120,
    history: [{ kind: "added", date: "—", text: "On your roster", member: "Team Hickey" }],
  };
}

export function getPlayerDetail(
  m: PlayerRef & { ownPct?: number; ownDelta?: number; rosteredBy?: string },
): PlayerDetail {
  const b = BASE[m.mlbId] ?? fallback(m);
  const isPitcher = !!b.pitcher;
  const cats = isPitcher ? PIT : HIT;
  const tb = teamBrand(b.teamId);
  const { columns, rows } = isPitcher ? pitcherLog() : hitterLog();

  const headline: { label: string; value: string }[] = [
    { label: "Rank", value: String(b.rankOverall) },
    { label: isPitcher ? "GS" : "GP", value: String(b.gp) },
    ...cats.map((c) => ({ label: c, value: fmt(c, b.season[c]) })),
  ];

  const ranks = [
    { label: "Season Rank", value: `#${b.rankOverall}` },
    { label: "Pos Rank", value: `#${b.rankPos}` },
    { label: "L7 Rank", value: `#${Math.max(1, b.rankOverall - 1)}` },
    { label: "L14 Rank", value: `#${b.rankOverall}` },
    { label: "L30 Rank", value: `#${b.rankOverall + 3}` },
  ];

  return {
    mlbId: m.mlbId, teamId: b.teamId, name: b.name, pos: b.pos, bats: b.bats,
    jersey: b.jersey, teamName: tb.name, isPitcher,
    ownPct: b.ownPct ?? m.ownPct ?? 100,
    ownDelta: b.ownDelta ?? m.ownDelta ?? 0,
    rosteredBy: m.rosteredBy ?? "Team Hickey",
    headline, ranks, gameColumns: columns, gameLog: rows,
    stats: statRows(b, cats),
    prior: {
      y2025Rank: b.y2025Rank,
      y2024Rank: b.y2024Rank,
      rows: cats.map((c) => ({ cat: c, y2025: fmt(c, b.y2025[c]), y2024: fmt(c, b.y2024[c]) })),
    },
    projections: projRows(b, cats),
    history: b.history,
  };
}
