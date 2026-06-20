import type {
  ApiLeadersOverallResponse,
  ApiFreeAgentPoolResponse,
  ApiStreamingResponse,
  ApiStreamAnalyzeResponse,
  ApiStandingsResponse,
  ApiPuntResponse,
  ApiMatchupResponse,
} from "@/lib/api/types";
import type { StandingsData, TeamStanding } from "@/lib/standings-data";
import { verdictFor, type PuntData, type PuntCat } from "@/lib/punt-data";
import type {
  MatchupData,
  MatchPlayer,
  TeamSide,
  RosterRow,
  GameState,
} from "@/lib/matchup-data";
import type { PlayerRef } from "@/lib/types";
import type { LeaderRow } from "@/lib/research-data";
import type { FreeAgent, PlayersData } from "@/lib/players-data";
import type {
  StreamingData,
  StreamCandidate,
  PitcherScorecard,
  ProbableStarter,
  StreamStatus,
  StreamConfidence,
  StreamComponents,
} from "@/lib/streaming-data";

/** Flatten an API PlayerRef (snake_case + nullable) → the frontend PlayerRef. */
function toPlayerRef(p: {
  name: string;
  positions: string;
  mlb_id?: number | null;
  team_abbr?: string | null;
  team_id?: number | null;
}): PlayerRef {
  return { name: p.name, pos: p.positions, teamAbbr: p.team_abbr ?? "", teamId: p.team_id ?? 0, mlbId: p.mlb_id ?? 0 };
}

const TRENDS = ["up", "down", "flat"];

/** Map /api/leaders/overall rows → frontend LeaderRow[]. A clean 1:1 (real
 *  0–100 value, 3 key stats, trend, tag, note + enriched PlayerRef) — replaces
 *  the old lossy per-category proxy. The overall lens has empty tag/flat trend;
 *  the dedicated lenses (hot/cold/breakout/sell) carry their tag. */
export function apiOverallToResearch(api: ApiLeadersOverallResponse): LeaderRow[] {
  return (api.rows ?? []).map((r) => ({
    ...toPlayerRef(r.player),
    rank: r.rank,
    hitter: r.hitter ?? true,
    stats: r.stats ?? [],
    value: Math.round(r.value ?? 0),
    trend: (TRENDS.includes(r.trend ?? "") ? r.trend : "flat") as LeaderRow["trend"],
    tag: r.tag ?? "",
    note: r.note ?? "",
  }));
}

/** Map the FA-pool response → the frontend PlayersData. The contract is a near
 *  1:1 match (rank/value/own/stats/fit/tag + enriched PlayerRef); this just
 *  renames snake_case → camelCase and flattens the PlayerRef (null-safe). */
export function apiPoolToPlayers(api: ApiFreeAgentPoolResponse): PlayersData {
  const freeAgents: FreeAgent[] = api.free_agents.map((it) => ({
    name: it.player.name,
    pos: it.player.positions,
    teamAbbr: it.player.team_abbr ?? "",
    teamId: it.player.team_id ?? 0,
    mlbId: it.player.mlb_id ?? 0,
    rank: it.rank,
    ownPct: it.own_pct,
    ownDelta: it.own_delta,
    value: it.value,
    hitter: it.hitter,
    stats: it.stats, // StatItem {label,value} === frontend stat shape
    fit: it.fit,
    tag: it.tag ?? undefined,
  }));
  return { topNeed: api.top_need, freeAgents };
}

type ApiStreamCandidate = NonNullable<ApiStreamingResponse["candidates"]>[number];

// Engine emits RAW UPPERCASE status/confidence; map to the frontend's lowercase.
const STREAM_STATUS: Record<string, StreamStatus> = {
  PROBABLE: "probable",
  LOCKED: "locked",
  FINAL: "final",
  OPEN: "open",
};
const STREAM_CONF: Record<string, StreamConfidence> = { HIGH: "high", MEDIUM: "med", LOW: "low" };

/** Map one API stream candidate (or analyze scorecard, which extends it) → the
 *  frontend StreamCandidate. Caveats handled: status/confidence casing,
 *  win_pct 0–1 → 0–100, reason → why, NaN-safe fallbacks. */
function toStreamCandidate(c: ApiStreamCandidate): StreamCandidate {
  return {
    rank: c.rank,
    player: toPlayerRef(c.player),
    opponent: c.opponent,
    isHome: c.is_home,
    score: Math.round(c.score),
    status: STREAM_STATUS[c.status] ?? "probable",
    confidence: STREAM_CONF[c.confidence] ?? "med",
    actionable: c.actionable,
    numStarts: c.num_starts,
    netSgp: c.net_sgp,
    oppWrcPlus: Math.round(c.opp_wrc_plus),
    oppKpct: c.opp_k_pct,
    park: c.park,
    xIp: c.expected_ip,
    xK: c.expected_k,
    xEr: c.expected_er,
    winPct: Math.round(c.win_pct * 100), // API win_pct is 0–1
    ownPct: Math.round(c.own_pct),
    riskFlags: c.risk_flags ?? [],
    components: (c.components ?? { matchup: 0, env: 0, form: 0, lineup: 0, sgp: 0, winprob: 0 }) as StreamComponents,
    expectedLine: c.expected_line,
    why: c.reason ?? "",
  };
}

/** Map the /api/streaming response → the frontend StreamingData. */
export function apiStreamingToData(api: ApiStreamingResponse): StreamingData {
  const b = api.budget;
  return {
    date: api.date ?? "",
    budget: {
      addsLeft: b?.adds_left ?? 0,
      addsTotal: b?.adds_total ?? 10,
      ipPace: Math.round(b?.ip_pace ?? 0), // 0 for now (deferred backend plumbing)
      ipTarget: Math.round(b?.ip_target ?? 54), // float (53.8…) → 54
      catsInPlay: b?.cats_in_play ?? [],
    },
    topPick: api.top_pick ? toStreamCandidate(api.top_pick) : null,
    board: (api.candidates ?? []).map(toStreamCandidate),
    probables: (api.probables ?? []).map((p) => ({
      player: toPlayerRef(p.player),
      pitcherId: p.player.id, // HEATER player_id — needed for the analyze POST
      team: p.team,
      opponent: p.opponent,
      isHome: p.is_home,
      posGroup: p.pos_group as ProbableStarter["posGroup"],
      startLikelihood: p.start_likelihood as ProbableStarter["startLikelihood"],
    })),
  };
}

/** Map the analyze response → a frontend PitcherScorecard (or null if the
 *  pitcher isn't a probable that date). */
export function apiScorecard(api: ApiStreamAnalyzeResponse): PitcherScorecard | null {
  if (!api.found || !api.scorecard) return null;
  const c = toStreamCandidate(api.scorecard);
  return {
    ...c,
    factors: (api.scorecard.factors ?? []).map((f) => ({
      key: f.key as keyof StreamComponents,
      label: f.label,
      value: f.value,
      weight: f.weight,
      detail: f.detail,
    })),
  };
}

/** Map /api/standings → frontend StandingsData. Playoff/championship odds aren't
 *  in the API yet → 0 (the page hides the odds view when all-zero). */
export function apiStandingsToData(api: ApiStandingsResponse): StandingsData {
  const teams: TeamStanding[] = (api.teams ?? []).map((t) => ({
    rank: t.rank ?? 0,
    teamName: t.team_name ?? "",
    wins: t.wins ?? 0,
    losses: t.losses ?? 0,
    ties: t.ties ?? 0,
    points: t.points ?? 0,
    categoryRanks: t.category_ranks ?? {},
    playoffOdds: 0, // gap: no playoff sim in /api/standings yet
    champOdds: 0,
    isUser: (t.team_name ?? "") === "Team Hickey", // single-league until M4
  }));
  return { teams, playoffSpots: 4 };
}

/** Map /api/punt → frontend PuntData. Derives the compete/tossup/punt verdict
 *  from the engine's rank + gainability + punt_candidates. */
export function apiPuntToData(api: ApiPuntResponse): PuntData {
  const candidates = api.punt_candidates ?? [];
  const nTeams = 12; // FourzynBurn
  const cats: PuntCat[] = (api.categories ?? []).map((c) => {
    const rank = c.current_rank ?? 0;
    const gainable = c.gainable ?? false;
    return {
      cat: c.cat,
      rank,
      gainable,
      verdict: verdictFor(c.cat, rank, gainable, candidates, nTeams),
      recommendation: c.recommendation ?? "",
    };
  });
  return { teamName: api.team_name ?? "", nTeams, candidates, cats };
}

type ApiRosterRow = NonNullable<ApiMatchupResponse["hitters"]>[number];
type ApiMatchPlayer = NonNullable<ApiRosterRow["you"]>;
type ApiTeamSide = NonNullable<ApiMatchupResponse["you"]>;
type ApiSideTotals = NonNullable<ApiMatchupResponse["hitter_totals"]>;

const GAME_STATES: GameState[] = ["final", "live", "sched", "none"];

/** Format a category float → the display string the mock used: AVG/OBP at 3dp
 *  with no leading zero (".284"), ERA/WHIP at 2dp ("0.86"), counting cats as ints. */
function fmtCatVal(cat: string, v: number): string {
  const c = cat.toUpperCase();
  if (c === "AVG" || c === "OBP") return v.toFixed(3).replace(/^(-?)0\./, "$1.");
  if (c === "ERA" || c === "WHIP") return v.toFixed(2);
  return String(Math.round(v));
}

function toTeamSide(t: ApiTeamSide | undefined): TeamSide {
  return { name: t?.name ?? "", manager: t?.manager ?? "", record: t?.record ?? "", score: t?.score ?? 0 };
}

/** API MatchPlayer → frontend. `pos` comes from `player.positions` (the real
 *  eligible positions); the API's own `MatchPlayer.pos` is just the slot, which
 *  the frontend already gets from `RosterRow.slot`. `stats` are PROJECTED season
 *  lines and `state` is the day's schedule state — NOT live box scores (a planned
 *  future CEO slice). */
function toMatchPlayer(mp: ApiMatchPlayer | null | undefined): MatchPlayer | null {
  if (!mp) return null;
  const r = toPlayerRef(mp.player);
  const state = (GAME_STATES.includes(mp.state as GameState) ? mp.state : "none") as GameState;
  return {
    name: r.name,
    teamAbbr: r.teamAbbr,
    teamId: r.teamId,
    mlbId: r.mlbId,
    pos: r.pos,
    status: mp.status ?? "",
    state,
    stats: mp.stats ?? [],
    badge: mp.badge === "IL" || mp.badge === "DTD" ? mp.badge : undefined,
  };
}

function toRosterRow(r: ApiRosterRow): RosterRow {
  return { slot: r.slot ?? "", you: toMatchPlayer(r.you), opp: toMatchPlayer(r.opp) };
}

function toTotals(t: ApiSideTotals | undefined): { you: string[]; opp: string[] } {
  return { you: t?.you ?? [], opp: t?.opp ?? [] };
}

/** Map /api/matchup → frontend MatchupData. The league scoreboard (other 6
 *  matchups) isn't in the contract yet (Matchup-C) → `league: []`; the page hides
 *  that section in live mode, so it reappears automatically when the slice lands. */
export function apiMatchupToData(api: ApiMatchupResponse): MatchupData {
  return {
    week: api.week ?? 0,
    dateTabs: api.date_tabs ?? [],
    you: toTeamSide(api.you),
    opp: toTeamSide(api.opp),
    cats: (api.categories ?? []).map((c) => ({
      key: c.cat,
      you: fmtCatVal(c.cat, c.you),
      opp: fmtCatVal(c.cat, c.opp),
      win: c.win === "you" || c.win === "opp" ? c.win : undefined,
    })),
    hitterColumns: api.hitter_columns ?? [],
    pitcherColumns: api.pitcher_columns ?? [],
    hitters: (api.hitters ?? []).map(toRosterRow),
    pitchers: (api.pitchers ?? []).map(toRosterRow),
    hitterTotals: toTotals(api.hitter_totals),
    pitcherTotals: toTotals(api.pitcher_totals),
    league: [],
  };
}
