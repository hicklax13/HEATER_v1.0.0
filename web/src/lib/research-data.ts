import type { PlayerRef } from "./types";
import { apiGet } from "@/lib/api/client";
import { apiOverallToResearch } from "@/lib/api/adapters";
import type { ApiLeadersOverallResponse } from "@/lib/api/types";

/**
 * Research data — a cross-category value leaderboard with 5 lenses
 * (overall / hot / cold / breakout / sell). Each lens is its own board from
 * /api/leaders/overall?lens=X. Mock by default; live behind NEXT_PUBLIC_HEATER_LIVE.
 */
export type Lens = "overall" | "hot" | "cold" | "breakout" | "sell";

export const LENSES: Lens[] = ["overall", "hot", "cold", "breakout", "sell"];

export interface LeaderRow extends PlayerRef {
  rank: number;
  hitter: boolean;
  stats: string[]; // 3 key stats
  value: number; // 0–100 overall value
  trend: "up" | "down" | "flat";
  tag: string; // "" on the overall board; "hot"/"cold"/"breakout"/"sell" on the lens boards
  note: string; // short context line
}

export interface ResearchData {
  byLens: Record<Lens, LeaderRow[]>;
}

const lr = (
  rank: number,
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
  hitter: boolean,
  value: number,
  trend: "up" | "down" | "flat",
  tag: string,
  note: string,
  stats: string[],
): LeaderRow => ({ rank, name, pos, teamAbbr, teamId, mlbId, hitter, value, trend, tag, note, stats });

const MOCK_ROWS: LeaderRow[] = [
  lr(1, "Aaron Judge", "RF", "NYY", 147, 592450, true, 98, "up", "hot", "Leads the league in HR", ["24 HR", "58 R", ".322 AVG"]),
  lr(2, "Bobby Witt Jr.", "SS", "KC", 118, 677951, true, 94, "up", "hot", "Elite 5-category shortstop", ["22 SB", ".305 AVG", "52 RBI"]),
  lr(3, "Tarik Skubal", "SP", "DET", 116, 669373, false, 93, "flat", "hot", "Best ratios among starters", ["130 K", "2.10 ERA", "0.95 WHIP"]),
  lr(4, "Paul Skenes", "SP", "PIT", 134, 694973, false, 91, "up", "hot", "Surging — ace upside", ["118 K", "1.96 ERA", "0.92 WHIP"]),
  lr(5, "Elly De La Cruz", "SS", "CIN", 113, 682829, true, 90, "up", "hot", "Leads the league in SB", ["28 SB", "18 HR", "55 R"]),
  lr(6, "Corbin Carroll", "OF", "ARI", 109, 682998, true, 86, "up", "breakout", "Power + speed leap", ["21 HR", "19 SB", ".265 AVG"]),
  lr(7, "Pete Crow-Armstrong", "OF", "CHC", 112, 691718, true, 84, "up", "breakout", "Toolsy breakout season", ["16 HR", "24 SB", ".270 AVG"]),
  lr(8, "Jackson Merrill", "OF", "SD", 135, 701538, true, 80, "up", "breakout", "Sophomore jump", ["15 HR", "10 SB", ".292 AVG"]),
  lr(9, "Marcell Ozuna", "Util", "PIT", 134, 542303, true, 74, "down", "sell", "HR/FB% due to regress", ["20 HR", "58 RBI", ".265 AVG"]),
  lr(10, "Juan Soto", "RF", "NYM", 121, 665742, true, 72, "down", "cold", "Worst slump of his career", ["19 HR", ".265 AVG", "2-for-21"]),
  lr(11, "Mookie Betts", "SS", "LAD", 119, 605141, true, 70, "down", "cold", "Quietly cold for 3 weeks", ["12 HR", ".248 AVG", "8-game skid"]),
  lr(12, "Spencer Steer", "1B", "CIN", 113, 668715, true, 60, "down", "sell", "BABIP-fueled, unsustainable", ["14 HR", "51 RBI", ".244 AVG"]),
];

/** Group a flat row list into the 5 lens boards (overall = all; each lens = its tag). */
function groupByLens(rows: LeaderRow[]): Record<Lens, LeaderRow[]> {
  return {
    overall: rows,
    hot: rows.filter((r) => r.tag === "hot"),
    cold: rows.filter((r) => r.tag === "cold"),
    breakout: rows.filter((r) => r.tag === "breakout"),
    sell: rows.filter((r) => r.tag === "sell"),
  };
}

export const RESEARCH: ResearchData = { byLens: groupByLens(MOCK_ROWS) };

const isLive = () => process.env.NEXT_PUBLIC_HEATER_LIVE === "1";

/** Fetch a single lens board. Live: /api/leaders/overall?lens=… with a 4s
 *  timeout (the backend breakout lens currently hangs >12s) — a timeout/error
 *  yields an empty board, never a throw. Mock: the in-memory board. */
export function fetchLens(lens: Lens): Promise<LeaderRow[]> {
  if (!isLive()) return Promise.resolve(RESEARCH.byLens[lens]);
  const req = apiGet<ApiLeadersOverallResponse>("/leaders/overall", { lens, limit: 30 })
    .then(apiOverallToResearch)
    .catch(() => [] as LeaderRow[]);
  const timeout = new Promise<LeaderRow[]>((resolve) => setTimeout(() => resolve([]), 4000));
  return Promise.race([req, timeout]);
}

/** Initial load. Live: fetch ONLY the overall board (fast — the default tab);
 *  the other lenses load on demand (see fetchLens) so one slow lens can't block
 *  the page and parallel requests can't starve each other on a single-worker
 *  API. Empty live board → null → honest empty state. Mock (off-live): in-memory
 *  RESEARCH after a simulated delay. */
export async function fetchResearch(delayMs = 600): Promise<ResearchData | null> {
  if (isLive()) {
    const overall = await fetchLens("overall");
    if (overall.length > 0) {
      return { byLens: { overall, hot: [], cold: [], breakout: [], sell: [] } };
    }
    return null;
  }
  return new Promise((resolve) => setTimeout(() => resolve(RESEARCH), delayMs));
}
