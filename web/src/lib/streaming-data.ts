import type { PlayerRef } from "./types";
import { apiGet, apiPost } from "@/lib/api/client";
import { apiStreamingToData, apiScorecard } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiStreamingResponse, ApiStreamAnalyzeResponse } from "@/lib/api/types";

/**
 * Pitcher Streaming data. Mock by default; with NEXT_PUBLIC_HEATER_LIVE=1,
 * fetchStreaming / analyzePitcher wire to /api/streaming (+ /analyze) and fall
 * back to this mock on any error or empty response.
 */

/** The 6 engine score components, each normalized -1..+1 (matches stream_analyzer). */
export interface StreamComponents {
  matchup: number; // opponent offense (wRC+/K%/splits)
  env: number; // park + weather
  form: number; // pitcher recent form
  lineup: number; // opposing-lineup exposure
  sgp: number; // pitcher skill (xFIP/SIERA -> net SGP)
  winprob: number; // win probability
}

export type StreamStatus = "open" | "probable" | "locked" | "final";
export type StreamConfidence = "high" | "med" | "low";
export type PosGroup = "SP" | "SP/RP" | "RP";

export interface StreamCandidate {
  rank: number;
  player: PlayerRef;
  opponent: string; // opponent team abbr
  isHome: boolean;
  score: number; // 0-100
  status: StreamStatus;
  confidence: StreamConfidence;
  actionable: boolean;
  numStarts: number; // 1 or 2 (two-start week)
  netSgp: number;
  oppWrcPlus: number;
  oppKpct: number;
  park: number; // park factor (1.00 = neutral)
  xIp: number;
  xK: number;
  xEr: number;
  winPct: number; // 0-100
  ownPct: number; // 0-100
  riskFlags: string[]; // HIGH_WHIP, SHORT_LEASH, ELITE_OFFENSE, HITTER_PARK, WIND_OUT, LOW_CONFIDENCE
  components: StreamComponents;
  expectedLine: string; // "6.0 IP · 7 K · 2 ER"
  why: string;
}

export interface FactorDetail {
  key: keyof StreamComponents;
  label: string;
  value: number; // -1..+1
  weight: number; // 0..1
  detail: string; // "vs SF: 88 wRC+ (8th-weakest), 26% K"
}

export interface PitcherScorecard extends StreamCandidate {
  factors: FactorDetail[];
}

export interface ProbableStarter {
  player: PlayerRef;
  pitcherId?: number; // HEATER player_id (live only) — for the analyze POST
  team: string;
  opponent: string;
  isHome: boolean;
  posGroup: PosGroup;
  startLikelihood: "confirmed" | "likely" | "projected";
}

export interface BudgetStrip {
  addsLeft: number;
  addsTotal: number; // 10
  ipPace: number;
  ipTarget: number; // 54
  catsInPlay: string[];
}

export interface StreamingData {
  date: string; // canonical YYYY-MM-DD (also the /streaming/analyze POST body); format for DISPLAY only
  budget: BudgetStrip;
  topPick: StreamCandidate | null;
  board: StreamCandidate[];
  probables: ProbableStarter[];
  urgency: Record<string, number>; // this-week per-category need (0-1); {} when no live matchup
}

/** Format the canonical `YYYY-MM-DD` streaming date for human display
 *  ("Wed Jun 23"). Parsed as a LOCAL date (no UTC offset) so it never shows the
 *  prior day; non-ISO (e.g. the mock's "Thu Jun 19") / empty values pass through
 *  unchanged. Display-only — never mutate the stored `date`, which the live
 *  analyze request sends as-is to the backend. */
export function formatStreamDate(raw: string | undefined): string {
  if (!raw) return "";
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(raw);
  if (!m) return raw;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

const FACTOR_LABEL: Record<keyof StreamComponents, string> = {
  matchup: "Opponent offense",
  env: "Park & weather",
  form: "Recent form",
  lineup: "Opposing lineup",
  sgp: "Pitcher skill",
  winprob: "Win probability",
};
const FACTOR_WEIGHT: Record<keyof StreamComponents, number> = {
  matchup: 0.28,
  env: 0.14,
  form: 0.12,
  lineup: 0.16,
  sgp: 0.22,
  winprob: 0.08,
};

const pr = (
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
): PlayerRef => ({ name, pos, teamAbbr, teamId, mlbId });

const cand = (
  rank: number,
  player: PlayerRef,
  opponent: string,
  isHome: boolean,
  score: number,
  c: StreamComponents,
  partial: Partial<StreamCandidate>,
): StreamCandidate => ({
  rank,
  player,
  opponent,
  isHome,
  score,
  status: "probable",
  confidence: score >= 70 ? "high" : score >= 50 ? "med" : "low",
  actionable: true,
  numStarts: 1,
  netSgp: 0,
  oppWrcPlus: 100,
  oppKpct: 22,
  park: 1.0,
  xIp: 5.5,
  xK: 6,
  xEr: 2.5,
  winPct: 50,
  ownPct: 0,
  riskFlags: [],
  components: c,
  expectedLine: "",
  why: "",
  ...partial,
});

export const STREAMING: StreamingData = {
  date: "Thu Jun 19",
  budget: { addsLeft: 6, addsTotal: 10, ipPace: 31, ipTarget: 54, catsInPlay: ["K", "ERA", "WHIP"] },
  topPick: null,
  board: [
    cand(1, pr("Tarik Skubal", "SP", "DET", 116, 669373), "CWS", true, 86,
      { matchup: 0.78, env: 0.32, form: 0.55, lineup: 0.6, sgp: 0.82, winprob: 0.4 },
      { numStarts: 1, netSgp: 1.42, oppWrcPlus: 84, oppKpct: 26.5, park: 0.97, xIp: 6.4, xK: 8.1, xEr: 1.9, winPct: 62, ownPct: 99,
        riskFlags: [], expectedLine: "6.4 IP · 8 K · 1.9 ER", why: "Elite arm vs a bottom-5 offense in a pitcher park." }),
    cand(2, pr("Paul Skenes", "SP", "PIT", 134, 694973), "MIA", false, 83,
      { matchup: 0.7, env: 0.25, form: 0.6, lineup: 0.55, sgp: 0.85, winprob: 0.1 },
      { netSgp: 1.18, oppWrcPlus: 89, oppKpct: 24.1, park: 0.99, xIp: 6.2, xK: 7.6, xEr: 2.0, winPct: 47, ownPct: 98,
        riskFlags: [], expectedLine: "6.2 IP · 8 K · 2.0 ER", why: "Ace ratios; weak-hitting opponent, but low win odds (poor run support)." }),
    cand(3, pr("Garrett Crochet", "SP", "BOS", 111, 676979), "TB", true, 74,
      { matchup: 0.5, env: 0.1, form: 0.45, lineup: 0.4, sgp: 0.75, winprob: 0.45 },
      { numStarts: 2, netSgp: 0.88, oppWrcPlus: 96, oppKpct: 23.0, park: 1.04, xIp: 6.0, xK: 8.4, xEr: 2.6, winPct: 58,
        ownPct: 94, riskFlags: ["HITTER_PARK"], expectedLine: "6.0 IP · 8 K · 2.6 ER", why: "Top-tier strikeouts; Fenway adds some ratio risk." }),
    cand(4, pr("Cole Ragans", "SP", "KC", 118, 666142), "DET", false, 68,
      { matchup: 0.35, env: 0.2, form: 0.3, lineup: 0.3, sgp: 0.62, winprob: 0.4 },
      { netSgp: 0.64, oppWrcPlus: 101, oppKpct: 21.5, park: 0.98, xIp: 5.9, xK: 7.2, xEr: 2.8, winPct: 55,
        ownPct: 91, expectedLine: "5.9 IP · 7 K · 2.8 ER", why: "Strong K upside; average matchup." }),
    cand(5, pr("Freddy Peralta", "SP", "MIL", 158, 642547), "CHC", true, 63,
      { matchup: 0.2, env: -0.1, form: 0.35, lineup: 0.15, sgp: 0.55, winprob: 0.5 },
      { netSgp: 0.41, oppWrcPlus: 108, oppKpct: 20.8, park: 1.03, xIp: 5.6, xK: 7.0, xEr: 3.1, winPct: 60,
        ownPct: 88, riskFlags: ["ELITE_OFFENSE"], expectedLine: "5.6 IP · 7 K · 3.1 ER", why: "Good win odds; Cubs offense is a real threat." }),
    cand(6, pr("Hunter Greene", "SP", "CIN", 113, 668881), "COL", false, 41,
      { matchup: -0.3, env: -0.7, form: 0.1, lineup: -0.25, sgp: 0.5, winprob: 0.2 },
      { status: "locked", actionable: false, netSgp: -0.22, oppWrcPlus: 118, oppKpct: 19.0, park: 1.28, xIp: 5.2, xK: 6.4, xEr: 4.2, winPct: 44,
        riskFlags: ["HITTER_PARK", "ELITE_OFFENSE"], ownPct: 85, expectedLine: "5.2 IP · 6 K · 4.2 ER", why: "Coors. Strikeouts, but a blow-up risk to ratios." }),
  ],
  probables: [
    { player: pr("Tarik Skubal", "SP", "DET", 116, 669373), team: "DET", opponent: "CWS", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Paul Skenes", "SP", "PIT", 134, 694973), team: "PIT", opponent: "MIA", isHome: false, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Garrett Crochet", "SP", "BOS", 111, 676979), team: "BOS", opponent: "TB", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Cole Ragans", "SP", "KC", 118, 666142), team: "KC", opponent: "DET", isHome: false, posGroup: "SP", startLikelihood: "likely" },
    { player: pr("Freddy Peralta", "SP", "MIL", 158, 642547), team: "MIL", opponent: "CHC", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Hunter Greene", "SP", "CIN", 113, 668881), team: "CIN", opponent: "COL", isHome: false, posGroup: "SP", startLikelihood: "likely" },
    { player: pr("Logan Gilbert", "SP", "SEA", 136, 669302), team: "SEA", opponent: "HOU", isHome: true, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("George Kirby", "SP", "SEA", 136, 669923), team: "SEA", opponent: "HOU", isHome: true, posGroup: "SP", startLikelihood: "projected" },
    { player: pr("Zack Wheeler", "SP", "PHI", 143, 554430), team: "PHI", opponent: "NYM", isHome: false, posGroup: "SP", startLikelihood: "confirmed" },
    { player: pr("Spencer Strider", "SP", "ATL", 144, 675911), team: "ATL", opponent: "WSH", isHome: true, posGroup: "SP", startLikelihood: "likely" },
  ],
  urgency: {},
};
STREAMING.topPick = STREAMING.board[0];

/** Build the per-factor detail rows from a candidate's components. */
export function factorsFor(c: StreamCandidate): FactorDetail[] {
  const keys = Object.keys(c.components) as (keyof StreamComponents)[];
  const detailText: Record<keyof StreamComponents, string> = {
    matchup: `vs ${c.opponent}: ${c.oppWrcPlus} wRC+, ${c.oppKpct.toFixed(0)}% K`,
    env: `Park ${c.park.toFixed(2)}${c.riskFlags.includes("HITTER_PARK") ? " (hitter park)" : ""}`,
    form: "Recent starts vs season baseline",
    lineup: "Regressed opposing-lineup wOBA exposure",
    sgp: `${c.netSgp >= 0 ? "+" : ""}${c.netSgp.toFixed(2)} net SGP (xFIP/SIERA)`,
    winprob: `${c.winPct}% win`,
  };
  return keys.map((k) => ({
    key: k,
    label: FACTOR_LABEL[k],
    value: c.components[k],
    weight: FACTOR_WEIGHT[k],
    detail: detailText[k],
  }));
}

/** Live: GET /api/streaming → adapt; live errors propagate (HIGH-3) so usePageData
 *  reaches error/locked/unlinked. Mock (off-live, or live with an empty board):
 *  the in-memory STREAMING after a simulated delay. */
export async function fetchStreaming(date?: string, delayMs = 600): Promise<StreamingData | null> {
  return liveOrMock(
    async () => {
      const api = await apiGet<ApiStreamingResponse>("/streaming", date ? { date } : undefined);
      return (api.candidates?.length ?? 0) > 0 ? apiStreamingToData(api) : null;
    },
    () => new Promise<StreamingData>((resolve) => setTimeout(() => resolve(STREAMING), delayMs)),
  );
}

/** Synthesize a scorecard from the mock board / a deterministic pitcher hash. */
function mockScorecard(p: ProbableStarter): PitcherScorecard {
  const onBoard = STREAMING.board.find((b) => b.player.mlbId === p.player.mlbId);
  if (onBoard) return { ...onBoard, factors: factorsFor(onBoard) };
  const h = (p.player.mlbId % 46) + 35; // 35..80
  const f = (seed: number) => Math.round(((((p.player.mlbId >> seed) % 17) - 8) / 8) * 100) / 100; // -1..+1
  const base: StreamCandidate = {
    rank: 0,
    player: p.player,
    opponent: p.opponent,
    isHome: p.isHome,
    score: h,
    status: "probable",
    confidence: h >= 70 ? "high" : h >= 50 ? "med" : "low",
    actionable: true,
    numStarts: 1,
    netSgp: Math.round((h - 55) / 25 * 100) / 100,
    oppWrcPlus: 92 + (p.player.mlbId % 26),
    oppKpct: 19 + (p.player.mlbId % 9),
    park: 0.95 + (p.player.mlbId % 30) / 100,
    xIp: 5.2 + (p.player.mlbId % 12) / 10,
    xK: 5 + (p.player.mlbId % 5),
    xEr: 2.2 + (p.player.mlbId % 18) / 10,
    winPct: 40 + (p.player.mlbId % 30),
    ownPct: p.player.mlbId % 80,
    riskFlags: h < 50 ? ["LOW_CONFIDENCE"] : [],
    components: { matchup: f(1), env: f(2), form: f(3), lineup: f(4), sgp: f(5), winprob: f(6) },
    expectedLine: "",
    why: "Synthesized scorecard (live scorer pending).",
  };
  base.expectedLine = `${base.xIp.toFixed(1)} IP · ${base.xK.toFixed(0)} K · ${base.xEr.toFixed(1)} ER`;
  return { ...base, factors: factorsFor(base) };
}

/** Analyze any probable pitcher → a full scorecard. Live: POST
 *  /api/streaming/analyze (returns null if the pitcher isn't a probable that
 *  date); live errors propagate (HIGH-3) so the Analyze panel shows a failure
 *  instead of a fabricated "Synthesized scorecard". When live but the pitcher has
 *  no HEATER id / no date, OR off-live: synthesize deterministically. */
export async function analyzePitcher(p: ProbableStarter, date?: string): Promise<PitcherScorecard | null> {
  return liveOrMock(
    async () => {
      if (p.pitcherId == null || !date) return mockScorecard(p);
      const api = await apiPost<ApiStreamAnalyzeResponse>("/streaming/analyze", { pitcher_id: p.pitcherId, date });
      return apiScorecard(api); // null when found:false
    },
    () => mockScorecard(p),
  );
}
