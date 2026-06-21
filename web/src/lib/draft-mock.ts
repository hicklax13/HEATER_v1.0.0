import type {
  DraftConfig,
  DraftPick,
  DraftClock,
  DraftPlayer,
  DraftRec,
  SimResult,
  RecResult,
} from "./draft-data";

/**
 * Client-side mock draft — lets /draft run fully offline (owner-approved 2026-06-20).
 * Mirrors the server's stateless replay: mockSimulate/mockRecommend take the same
 * (config, pickLog) the API does. This module imports ONLY types from draft-data
 * (erased at compile time), so there's no runtime import cycle.
 *
 * Every 6-digit mlbId below is validated by scripts/audit-mock-ids.mjs. Generic
 * "Prospect N" fillers carry mlbId 0 (PlayerLink fallback avatar; audit-exempt)
 * to give the pool depth for a full 12×23 draft.
 */

const rp = (
  name: string,
  pos: string,
  teamAbbr: string,
  teamId: number,
  mlbId: number,
  adp: number,
): DraftPlayer => ({ id: mlbId, name, pos, teamAbbr, teamId, mlbId, adp });

// ~40 real players (best ADP first) — enough for 3+ real rounds in a 12-team draft.
const MOCK_POOL: DraftPlayer[] = [
  rp("Aaron Judge", "RF", "NYY", 147, 592450, 1),
  rp("Bobby Witt Jr.", "SS", "KC", 118, 677951, 2),
  rp("Tarik Skubal", "SP", "DET", 116, 669373, 3),
  rp("Juan Soto", "RF", "NYM", 121, 665742, 4),
  rp("José Ramírez", "3B", "CLE", 114, 608070, 5),
  rp("Yordan Alvarez", "DH", "HOU", 117, 670541, 6),
  rp("Elly De La Cruz", "SS", "CIN", 113, 682829, 7),
  rp("Mookie Betts", "SS", "LAD", 119, 605141, 8),
  rp("Paul Skenes", "SP", "PIT", 134, 694973, 9),
  rp("Corbin Carroll", "OF", "ARI", 109, 682998, 10),
  rp("Kyle Tucker", "RF", "CHC", 112, 663656, 11),
  rp("Pete Crow-Armstrong", "OF", "CHC", 112, 691718, 12),
  rp("Trea Turner", "SS", "PHI", 143, 607208, 13),
  rp("Zack Wheeler", "SP", "PHI", 143, 554430, 14),
  rp("Garrett Crochet", "SP", "BOS", 111, 676979, 15),
  rp("Matt Olson", "1B", "ATL", 144, 621566, 16),
  rp("Ozzie Albies", "2B", "ATL", 144, 645277, 17),
  rp("Jackson Merrill", "OF", "SD", 135, 701538, 18),
  rp("Freddy Peralta", "SP", "MIL", 158, 642547, 19),
  rp("Cole Ragans", "SP", "KC", 118, 666142, 20),
  rp("Hunter Greene", "SP", "CIN", 113, 668881, 21),
  rp("George Kirby", "SP", "SEA", 136, 669923, 22),
  rp("Logan Gilbert", "SP", "SEA", 136, 669302, 23),
  rp("Will Smith", "C", "LAD", 119, 669257, 24),
  rp("Marcell Ozuna", "DH", "ATL", 144, 542303, 25),
  rp("Jordan Westburg", "3B", "BAL", 110, 676059, 26),
  rp("Mitch Keller", "SP", "PIT", 134, 656605, 27),
  rp("Emmanuel Clase", "RP", "CLE", 114, 661403, 28),
  rp("Josh Hader", "RP", "HOU", 117, 623352, 29),
  rp("Edwin Díaz", "RP", "NYM", 121, 621242, 30),
  rp("Mason Miller", "RP", "ATH", 133, 695243, 31),
  rp("Jhoan Duran", "RP", "MIN", 142, 661395, 32),
  rp("Ryan Helsley", "RP", "STL", 138, 664854, 33),
  rp("Spencer Strider", "SP", "ATL", 144, 675911, 34),
  rp("Devin Williams", "RP", "NYY", 147, 642207, 35),
  rp("Cade Smith", "RP", "CLE", 114, 671922, 36),
  rp("Raisel Iglesias", "RP", "ATL", 144, 628452, 37),
  rp("Pete Fairbanks", "RP", "TB", 139, 664126, 38),
  rp("Aroldis Chapman", "RP", "BOS", 111, 547973, 39),
  rp("Paul Sewald", "RP", "ARI", 109, 623149, 40),
];

const FILLER_POS = ["SP", "RP", "OF", "2B", "SS", "3B", "1B", "C"];
function fillerPlayer(seq: number): DraftPlayer {
  return {
    id: 900000 + seq,
    name: `Prospect ${seq}`,
    pos: FILLER_POS[seq % FILLER_POS.length],
    teamAbbr: "FA",
    teamId: 0,
    mlbId: 0,
    adp: 300 + seq,
  };
}

/** Best-available players (lowest ADP first), padded with generic fillers so a
 *  full draft can complete past the ~40 real names. The filler budget scales with
 *  how many players are already drafted (NOT `limit`), so even a 20×30 = 600-pick
 *  draft always finds `limit` undrafted fillers instead of stalling. */
function available(draftedIds: Set<number>, limit: number): DraftPlayer[] {
  const out = MOCK_POOL.filter((p) => !draftedIds.has(p.id));
  const budget = draftedIds.size + limit + 100; // always enough headroom
  let seq = 1;
  while (out.length < limit) {
    const f = fillerPlayer(seq++);
    if (!draftedIds.has(f.id)) out.push(f);
    if (seq > budget) break; // backstop (cannot be hit in normal use)
  }
  return out.slice(0, limit);
}

function teamForPick(pick: number, numTeams: number): number {
  const round = Math.floor(pick / numTeams);
  const pos = pick % numTeams;
  return round % 2 === 0 ? pos : numTeams - 1 - pos;
}

function buildClock(config: DraftConfig, current: number): DraftClock {
  const total = config.numTeams * config.numRounds;
  if (current >= total) {
    return {
      currentPick: total,
      round: config.numRounds,
      pickInRound: config.numTeams,
      pickingTeamIndex: 0,
      isUserTurn: false,
    };
  }
  const round = Math.floor(current / config.numTeams);
  const pos = current % config.numTeams;
  const team = teamForPick(current, config.numTeams);
  return {
    currentPick: current,
    round: round + 1,
    pickInRound: pos + 1,
    pickingTeamIndex: team,
    isUserTurn: team === config.userTeamIndex,
  };
}

/** Advance AI opponents (top-5 by ADP, lightly randomized) to the user's next
 *  turn or the end of the draft. Returns only the NEW picks + the fresh clock. */
export function mockSimulate(config: DraftConfig, pickLog: DraftPick[]): SimResult {
  const total = config.numTeams * config.numRounds;
  const drafted = new Set(pickLog.map((p) => p.playerId));
  const picks: DraftPick[] = [];
  let current = pickLog.length;

  while (current < total && teamForPick(current, config.numTeams) !== config.userTeamIndex) {
    const pool = available(drafted, 6);
    const topK = pool.slice(0, Math.min(4, pool.length));
    const choice = topK[Math.floor(Math.random() * topK.length)] ?? pool[0];
    if (!choice) break;
    picks.push({
      pick: current,
      teamIndex: teamForPick(current, config.numTeams),
      playerId: choice.id,
      playerName: choice.name,
      positions: choice.pos,
    });
    drafted.add(choice.id);
    current++;
  }
  return { clock: buildClock(config, current), picks };
}

const TAGS = ["buy", "fair", "avoid"] as const;
function reasonFor(p: DraftPlayer, i: number): string {
  if (i === 0) return `Best player available — top of the board at ${p.pos}.`;
  if ((p.adp ?? 99) <= 24) return `Elite ${p.pos} value still on the board.`;
  if (p.mlbId === 0) return `Depth flier — upside ${p.pos} to round out the bench.`;
  return `Solid ${p.pos} to keep the roster balanced.`;
}

/** Top-N available with a derived score / tag / reason — mirrors the API rec shape. */
export function mockRecommend(config: DraftConfig, pickLog: DraftPick[], topN: number): RecResult {
  const drafted = new Set(pickLog.map((p) => p.playerId));
  const pool = available(drafted, topN);
  const recs: DraftRec[] = pool.map((player, i) => ({
    player,
    rank: i + 1,
    score: Math.max(20, Math.round(96 - i * 7 - (player.adp ?? 0) * 0.02)),
    projectedSgp: Math.round((40 - i * 2.5) * 10) / 10,
    confidence: i < 2 ? "HIGH" : i < 4 ? "MEDIUM" : "LOW",
    tag: i === 0 ? TAGS[0] : i < topN - 1 ? TAGS[1] : TAGS[2],
    reason: reasonFor(player, i),
  }));
  const round = Math.floor(pickLog.length / config.numTeams) + 1;
  return { clock: buildClock(config, pickLog.length), recs, summary: `Round ${round} — your pick.` };
}
