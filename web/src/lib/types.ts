export type Trend = "up" | "down" | "flat";
export type Scope = "mine" | "league" | "mixed";

export interface PlayerRef {
  name: string;
  pos: string;
  teamAbbr: string;
  teamId: number;
  mlbId: number;
}

export interface Mover extends PlayerRef {
  /** Fixed slot grammar: two primary stats + one context line (hitters & pitchers). */
  stat1: { label: string; value: string };
  stat2: { label: string; value: string };
  context: string;
  trend: Trend;
  tag: string; // "on fire", "ice cold", etc.
  spark?: number[]; // 7-day micro trend — optional (not in the live contract)
  ownPct?: number; // optional (not in the live contract)
  rosteredByYou: boolean;
}

export interface CategoryRow {
  key: string; // HR, K, OBP, ERA, SV, SB
  you: string;
  opp: string;
  edge: string; // signed display, e.g. "+4", "-0.48"
  /** Is the edge good or bad FOR YOU (polarity-resolved). */
  edgeDir: "good" | "bad" | "even";
  winPct: number; // 0..100
  higherBetter: boolean; // false for ERA/WHIP/L
  spark?: number[]; // 10-week trend of YOUR value — optional (not in the live contract)
  isLever?: boolean;
}

export interface Pickup extends PlayerRef {
  projStat: { label: string; value: string };
}

export interface Matchup {
  youName: string;
  youRecord: string;
  youLogo?: string;
  oppName: string;
  oppRecord?: string; // optional: not in the live contract
  oppLogo?: string;
  winPct: number;
  tiePct: number;
  lossPct: number;
  projLine?: string; // "proj 6-6" — optional (not in the live contract)
  deltaVsLastWeek?: number; // +4 — optional (winprob history, deferred)
}

export interface OpsCard {
  key: "ip" | "moves" | "roster";
  label: string;
  value: number;
  total: number;
  unit?: string;
  verdict: string; // plain-words end-state
  status: "ok" | "warn" | "bad";
}

export interface TrajectoryPoint {
  week: number;
  rank: number;
}

export interface MyTeamData {
  eyebrow: string;
  teamName: string;
  subline: string;
  freshnessMinutes: number;
  statusChips: { label: string; tone: "ok" | "warn" | "bad" | "info"; icon: string }[];
  matchup: Matchup;
  movers: Mover[];
  moversScope: Scope;
  lever?: {
    categoryKey: string;
    headline: string;
    behindBy: number;
    pickups: Pickup[];
  };
  categories: CategoryRow[];
  ops: OpsCard[];
  trajectory: TrajectoryPoint[];
  winProbTrend: number[];
  playoffCutRank: number;
}

export type DataState<T> =
  | { status: "loading" }
  | { status: "empty" }
  | { status: "error"; message: string }
  | { status: "loaded"; data: T };
