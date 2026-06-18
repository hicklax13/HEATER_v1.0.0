import type { PlayerRef } from "./types";

/**
 * Mock Trades data — Trade Finder recommendations targeting your needs.
 * Swap this module for the API client in Sub-project B; the shape is the contract.
 */
export interface TradePlayer extends PlayerRef {
  posLabel: string; // "RF · NYM"
  keyStat: string; // headline line, e.g. ".265 AVG · 19 HR"
}

export interface CatImpact {
  cat: string;
  delta: string; // signed, display-ready ("+9", "-3", "+.004")
  dir: "up" | "down";
}

export interface TradeRec {
  id: number;
  partner: string;
  partnerRecord: string;
  grade: string; // "A-", "B+"…
  verdict: string; // "Fair", "You win", "Slight overpay"
  give: TradePlayer[];
  get: TradePlayer[];
  impact: CatImpact[];
  rationale: string;
  playoffDelta: number; // change in playoff odds (pp)
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
        tp("Marcell Ozuna", "ATL", 144, 542303, "Util · ATL", "20 HR · 58 RBI"),
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
        tp("Jordan Romano", "PHI", 143, 656266, "RP · PHI", "9 SV"),
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

export function fetchTrades(delayMs = 600): Promise<TradesData> {
  return new Promise((resolve) => setTimeout(() => resolve(TRADES), delayMs));
}
