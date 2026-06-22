"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type DraftConfig,
  type DraftPick,
  type DraftClock,
  type DraftRec,
  type DraftPlayer,
  draftSimulate,
  draftRecommend,
  totalPicks,
} from "./draft-data";
import { isPaywall } from "@/lib/api/errors";

export type DraftPhase = "setup" | "drafting" | "complete" | "locked";

interface DraftState {
  phase: DraftPhase;
  config: DraftConfig | null;
  pickLog: DraftPick[];
  myRoster: DraftPlayer[]; // the user's own picks, kept as full DraftPlayers (for headshots)
  clock: DraftClock | null;
  recs: DraftRec[];
  summary: string;
  busy: boolean;
}

const INITIAL: DraftState = {
  phase: "setup",
  config: null,
  pickLog: [],
  myRoster: [],
  clock: null,
  recs: [],
  summary: "",
  busy: false,
};

/**
 * Draft state machine. Owns config + pickLog (the stateless contract) and the
 * two-call turn loop: simulate-picks advances the AI to the user's turn, then
 * recommend fetches the board. A `busy` guard + a state ref prevent re-entrant
 * picks and stale-closure reads during the async round-trips.
 */
export function useDraft() {
  const [state, setState] = useState<DraftState>(INITIAL);
  const ref = useRef(state);
  useEffect(() => {
    ref.current = state;
  });

  // Advance the AI to the user's turn, then (if it's their turn) fetch the board.
  async function advance(config: DraftConfig, pickLog: DraftPick[]) {
    const sim = await draftSimulate(config, pickLog);
    const log = [...pickLog, ...sim.picks];
    let clock = sim.clock;
    let recs: DraftRec[] = [];
    let summary = "";
    const done = clock.currentPick >= totalPicks(config);
    if (!done && clock.isUserTurn) {
      const r = await draftRecommend(config, log);
      recs = r.recs;
      summary = r.summary;
      clock = r.clock;
    }
    return { phase: (done ? "complete" : "drafting") as DraftPhase, log, clock, recs, summary };
  }

  const start = useCallback(async (config: DraftConfig) => {
    if (ref.current.busy) return;
    ref.current = { ...ref.current, busy: true }; // sync guard before the await
    setState({ ...INITIAL, phase: "drafting", config, busy: true });
    try {
      const r = await advance(config, []);
      setState({
        phase: r.phase,
        config,
        pickLog: r.log,
        myRoster: [],
        clock: r.clock,
        recs: r.recs,
        summary: r.summary,
        busy: false,
      });
    } catch (e) {
      // Live: 402 → locked; any other API/network error surfaces back to setup
      // (no fabricated mock draft). Off-live, advance() resolves from the mock.
      setState({ ...INITIAL, phase: isPaywall(e) ? "locked" : "setup", config });
    }
  }, []);

  const pick = useCallback(async (player: DraftPlayer) => {
    const cur = ref.current;
    if (!cur.config || cur.busy || cur.phase !== "drafting" || !cur.clock?.isUserTurn) return;
    const config = cur.config;
    const userPick: DraftPick = {
      pick: cur.pickLog.length,
      teamIndex: config.userTeamIndex,
      playerId: player.id,
      playerName: player.name,
      positions: player.pos,
    };
    const baseLog = [...cur.pickLog, userPick];
    const myRoster = [...cur.myRoster, player];
    ref.current = { ...cur, busy: true }; // sync guard so a double-click can't re-enter before the re-render
    setState((s) => ({ ...s, pickLog: baseLog, myRoster, recs: [], busy: true })); // optimistic
    try {
      const r = await advance(config, baseLog);
      setState((s) => ({
        ...s,
        phase: r.phase,
        pickLog: r.log,
        myRoster,
        clock: r.clock,
        recs: r.recs,
        summary: r.summary,
        busy: false,
      }));
    } catch (e) {
      setState((s) => ({ ...s, busy: false, phase: isPaywall(e) ? "locked" : s.phase }));
    }
  }, []);

  const reset = useCallback(() => setState(INITIAL), []);

  return { ...state, start, pick, reset };
}
