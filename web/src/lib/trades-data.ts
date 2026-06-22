import { getViewerTeam } from "@/lib/viewer-team";
import type { PlayerRef } from "./types";
import { apiGet, apiPost } from "@/lib/api/client";
import { apiTradeFinderToData, apiTradeEvaluateToData } from "@/lib/api/adapters";
import { liveOrMock } from "@/lib/api/live";
import type { ApiTradeFinderResponse, ApiTradeEvaluationResponse } from "@/lib/api/types";
import type { PlayerPick } from "./player-search";

/**
 * Trades data — the Finder tab's auto-suggested trades. Mock by default; live
 * behind NEXT_PUBLIC_HEATER_LIVE via /api/trade-finder. The contract is lean
 * (partner + giving/receiving + net_sgp + rationale), so grade/impact/playoffDelta/
 * partnerRecord are mock-only and hidden on live data; verdict is derived from
 * net_sgp. Suggestions are roster-relative → empty locally, real on Railway.
 */
export interface TradePlayer extends PlayerRef {
  posLabel: string; // "RF · NYM"
  keyStat?: string; // headline line, e.g. ".265 AVG · 19 HR" — optional (mock only)
}

export interface CatImpact {
  cat: string;
  delta: string; // signed, display-ready ("+9", "-3", "+.004")
  dir: "up" | "down";
}

export interface TradeRec {
  id: number;
  partner: string;
  partnerRecord?: string; // optional — not in the trade-finder contract
  grade?: string; // "A-", "B+"… — optional (mock); live shows netSgp instead
  netSgp?: number; // net SGP gain from the trade (live trade-finder signal)
  verdict: string; // "Fair", "You win", … (derived from netSgp on live data)
  give: TradePlayer[];
  get: TradePlayer[];
  impact?: CatImpact[]; // optional — no per-category breakdown in the contract
  rationale: string;
  playoffDelta?: number; // change in playoff odds (pp) — optional (mock only)
}

export interface TradesData {
  needs: string[];
  recs: TradeRec[];
}

const tp = (
  name: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
  posLabel: string,
  keyStat: string,
): TradePlayer => ({ name, teamAbbr, teamId, mlbId, pos: posLabel.split(" · ")[0], posLabel, keyStat });

export const TRADES: TradesData = {
  needs: ["SB", "SV"],
  recs: [
    {
      id: 1,
      partner: "Over the Rembow",
      partnerRecord: "11-1-0 · 1st",
      grade: "A-",
      verdict: "Fair",
      give: [tp("Juan Soto", "NYM", 121, 665742, "RF · NYM", ".265 AVG · 19 HR")],
      get: [tp("Elly De La Cruz", "CIN", 113, 682829, "SS · CIN", "28 SB · 18 HR")],
      impact: [
        { cat: "SB", delta: "+9", dir: "up" },
        { cat: "R", delta: "+4", dir: "up" },
        { cat: "HR", delta: "-3", dir: "down" },
      ],
      rationale: "Soto's in a 2-for-21 skid — Elly attacks your single biggest gap (SB) and barely dents your power.",
      playoffDelta: 6,
    },
    {
      id: 2,
      partner: "BUBBA CROSBY",
      partnerRecord: "5-5-2 · 6th",
      grade: "B+",
      verdict: "You win slightly",
      give: [
        tp("Marcell Ozuna", "PIT", 134, 542303, "Util · PIT", "20 HR · 58 RBI"),
        tp("Logan Gilbert", "SEA", 136, 669302, "SP · SEA", "118 K · 3.10 ERA"),
      ],
      get: [
        tp("Emmanuel Clase", "CLE", 114, 661403, "RP · CLE", "19 SV · 1.90 ERA"),
        tp("Jonathan India", "KC", 118, 663697, "2B · KC", "49 R · .340 OBP"),
      ],
      impact: [
        { cat: "SV", delta: "+12", dir: "up" },
        { cat: "SB", delta: "+3", dir: "up" },
        { cat: "K", delta: "-22", dir: "down" },
      ],
      rationale: "Converts surplus strikeout depth into the saves you're currently punting.",
      playoffDelta: 4,
    },
    {
      id: 3,
      partner: "My Precious",
      partnerRecord: "8-3-1 · 2nd",
      grade: "B",
      verdict: "Fair",
      give: [tp("Matt Olson", "ATL", 144, 621566, "1B · ATL", "20 HR · 51 RBI")],
      get: [tp("Trea Turner", "PHI", 143, 607208, "SS · PHI", "24 SB · .290 AVG")],
      impact: [
        { cat: "SB", delta: "+7", dir: "up" },
        { cat: "AVG", delta: "+.004", dir: "up" },
        { cat: "HR", delta: "-8", dir: "down" },
        { cat: "RBI", delta: "-5", dir: "down" },
      ],
      rationale: "Swaps raw power you're already winning for the speed + average you need.",
      playoffDelta: 3,
    },
    {
      id: 4,
      partner: "Go yanks",
      partnerRecord: "6-4-3 · 5th",
      grade: "B-",
      verdict: "You win",
      give: [tp("Tarik Skubal", "DET", 116, 669373, "SP · DET", "130 K · 2.10 ERA")],
      get: [
        tp("José Ramírez", "CLE", 114, 608070, "3B · CLE", "18 HR · 22 SB"),
        tp("Jordan Romano", "PHI", 143, 605447, "RP · PHI", "9 SV"),
      ],
      impact: [
        { cat: "SB", delta: "+5", dir: "up" },
        { cat: "SV", delta: "+9", dir: "up" },
        { cat: "K", delta: "-45", dir: "down" },
        { cat: "W", delta: "-3", dir: "down" },
      ],
      rationale: "Sell-high on Skubal's elite ratios for multi-category help — risky on strikeouts.",
      playoffDelta: 2,
    },
  ],
};

/** Live: GET /api/trade-finder?team_name=… → adapt. Live errors propagate
 *  (HIGH-3) so usePageData reaches error/locked (402)/unlinked (409) — NEVER the
 *  fabricated mock. Empty suggestions (roster-relative → empty locally, real on
 *  Railway) still resolve to the showcase mock so the page demos. Mock (off-live):
 *  the in-memory TRADES after a simulated delay. */
export async function fetchTrades(delayMs = 600): Promise<TradesData> {
  const mock = () => new Promise<TradesData>((resolve) => setTimeout(() => resolve(TRADES), delayMs));
  return liveOrMock(async () => {
    const api = await apiGet<ApiTradeFinderResponse>("/trade-finder", {
      team_name: getViewerTeam(),
      limit: 10,
    });
    // Real suggestions → adapt; empty (no error) → fall back to the demo mock.
    return (api.suggestions?.length ?? 0) > 0 ? apiTradeFinderToData(api) : mock();
  }, mock);
}

/* ── Build-a-trade evaluator ───────────────────────────────────────────── */

export interface CatDelta {
  cat: string;
  delta: number; // signed SGP delta for this category from the trade
}

export interface TradeEval {
  grade: string;
  verdict: string;
  surplusSgp: number;
  confidencePct: number;
  giving: PlayerRef[];
  receiving: PlayerRef[];
  categoryImpacts: CatDelta[];
  deltaPlayoffProb?: number;
  deltaChampProb?: number;
  summary: string;
  warnings: string[];
}

/** Off-live demo evaluation so the builder works in the showcase. */
function mockEvaluate(giving: PlayerPick[], receiving: PlayerPick[]): TradeEval {
  return {
    grade: "B+",
    verdict: "Fair",
    surplusSgp: 1.2,
    confidencePct: 68,
    giving,
    receiving,
    categoryImpacts: [
      { cat: "SB", delta: 2.1 },
      { cat: "R", delta: 0.8 },
      { cat: "HR", delta: -1.4 },
    ],
    deltaPlayoffProb: 3,
    summary: "A roughly balanced swap with a modest net gain. (Demo — connect live data for a real grade.)",
    warnings: [],
  };
}

/** Evaluate a proposed trade. Live: POST /api/trade/evaluate (receiving side is
 *  pool-backed → real locally; the giving side needs your roster → full on
 *  Railway). Live errors PROPAGATE (HIGH-3) so BuildPanel can show locked (402) /
 *  not-linked (409) / error instead of the fabricated demo grade. Off-live: a demo
 *  evaluation. Returns null only when a side is empty (nothing to evaluate yet). */
export async function evaluateTrade(
  giving: PlayerPick[],
  receiving: PlayerPick[],
): Promise<TradeEval | null> {
  if (giving.length === 0 || receiving.length === 0) return null;
  return liveOrMock(
    () =>
      apiPost<ApiTradeEvaluationResponse>("/trade/evaluate", {
        team_name: getViewerTeam(),
        giving_ids: giving.map((p) => p.id),
        receiving_ids: receiving.map((p) => p.id),
        enable_mc: false,
      }).then(apiTradeEvaluateToData),
    () => mockEvaluate(giving, receiving),
  );
}
