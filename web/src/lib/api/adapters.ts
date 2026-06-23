import type {
  ApiLeadersOverallResponse,
  ApiFreeAgentPoolResponse,
  ApiStreamingResponse,
  ApiStreamAnalyzeResponse,
  ApiStandingsResponse,
  ApiPuntResponse,
  ApiMatchupResponse,
  ApiCompareResponse,
  ApiMyTeamResponse,
  ApiPlayoffOddsResponse,
  ApiTradeFinderResponse,
  ApiTradeEvaluationResponse,
  ApiDatabankResponse,
  ApiLineupOptimizeResponse,
  ApiLineupSlot,
  ApiClosersResponse,
  ApiDraftRecommendResponse,
  ApiDraftSimulateResponse,
  ApiDraftClock,
  ApiDraftPick,
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
import type { PlayerRef, MyTeamData, Matchup, Mover, CategoryRow, OpsCard, Scope } from "@/lib/types";
import type { LeaderRow } from "@/lib/research-data";
import type { FreeAgent, PlayersData } from "@/lib/players-data";
import type { CompareData } from "@/lib/compare-data";
import type { TradesData, TradePlayer, TradeEval } from "@/lib/trades-data";
import type { DatabankData } from "@/lib/databank-data";
import type { OptimizerData, LineupSlot as OptSlot, SlotStatus } from "@/lib/optimizer-data";
import type { ClosersData } from "@/lib/closers-data";
import { securityFor } from "@/lib/closer-security";
import { getViewerTeam } from "@/lib/viewer-team";
import type {
  DraftClock as VMDraftClock,
  DraftPick as VMDraftPick,
  DraftPlayer,
  RecResult,
  SimResult,
} from "@/lib/draft-data";
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
    id: it.player.id, // HEATER player_id — the compare picker feeds this to /api/compare
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

/** The API's streaming `date` is a raw `YYYY-MM-DD` (server "today"); the page
 *  renders it verbatim ("Daily · {date}"). Format it to the human "Wed Jun 23"
 *  style the mock uses. Non-ISO / empty values pass through unchanged.
 *  Parsed as a LOCAL date (no UTC offset) so it never shows the prior day. */
function formatStreamDate(raw: string | undefined): string {
  if (!raw) return "";
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(raw);
  if (!m) return raw;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

/** Map the /api/streaming response → the frontend StreamingData. */
export function apiStreamingToData(api: ApiStreamingResponse): StreamingData {
  const b = api.budget;
  return {
    date: formatStreamDate(api.date),
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
  // Live team names carry Yahoo emoji prefixes ("🏆 Team Hickey"); normalize before
  // the viewer compare or the "YOU" row never highlights. The viewer's team is the
  // API's identity-resolved team (cached by fetchMyTeam), NOT a hardcode — so each
  // logged-in user's own row highlights, not Team Hickey's.
  const youNorm = normTeamName(getViewerTeam());
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
    isUser: normTeamName(t.team_name ?? "") === youNorm, // single-league until M4
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

/** Normalize a team name for matching: strip emoji/punctuation (Yahoo names can be
 *  prefixed, e.g. "🔥 Team Hickey") and lowercase. The header's you.name is the
 *  clean passed team_name, but league[] carries the raw Yahoo display name — so an
 *  exact compare would miss the viewer's own pairing. */
function normTeamName(s: string): string {
  return s.replace(/[^a-z0-9]+/gi, " ").trim().toLowerCase();
}

/** 1 if the viewer's team is on either side of a league pairing, else 0 — used to
 *  sort the viewer's own pairing to the front (the LeagueMatchups component
 *  trophies/highlights entry 0, matching the mock). */
function leagueHasYou(m: { a: { name: string }; b: { name: string } }, you: string): number {
  const yn = normTeamName(you);
  return yn !== "" && (normTeamName(m.a.name) === yn || normTeamName(m.b.name) === yn) ? 1 : 0;
}

/** Map /api/matchup → frontend MatchupData. `league` is the full week scoreboard
 *  (all 6 pairings, INCLUDING the viewer's own); each side reuses TeamSide, which
 *  is structurally identical to the frontend's LeagueTeam. The viewer's pairing is
 *  sorted first to match the mock + the component's entry-0 trophy. Per-team weekly
 *  `score` is 0 locally and populates on Railway (same env-caveat as the rosters);
 *  pairings + records render live regardless. */
export function apiMatchupToData(api: ApiMatchupResponse): MatchupData {
  const youName = (api.you?.name ?? "").trim();
  const league = (api.league ?? [])
    .map((m) => ({ a: toTeamSide(m.a), b: toTeamSide(m.b) }))
    .sort((m1, m2) => leagueHasYou(m2, youName) - leagueHasYou(m1, youName));
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
    league,
  };
}

/** Map /api/compare → frontend CompareData. categories[] + per-player stat maps
 *  (category → projected value); PlayerRef enriched for headshots/logos. The
 *  frontend computes the winner-per-category (the API returns raw values). */
export function apiCompareToData(api: ApiCompareResponse): CompareData {
  return {
    categories: api.categories ?? [],
    players: (api.players ?? []).map((p) => ({
      ...toPlayerRef(p.player),
      stats: p.stats ?? {},
    })),
  };
}

/* ── My Team (flagship dashboard) ───────────────────────────────────────── */

type ApiMover = NonNullable<ApiMyTeamResponse["movers"]>[number];
type ApiCategoryLine = ApiMyTeamResponse["categories"][number];
type ApiOpsCard = NonNullable<ApiMyTeamResponse["ops"]>[number];
type ApiStatusChip = NonNullable<ApiMyTeamResponse["status_chips"]>[number];

const OPS_KEY: Record<string, OpsCard["key"]> = {
  ip_pace: "ip",
  moves_left: "moves",
  roster_health: "roster",
};
const OPS_STATUS: Record<string, OpsCard["status"]> = { ok: "ok", warn: "warn", danger: "bad" };
const MOVER_TREND = new Set(["up", "down"]);
const EDGE_EPS = 1e-9;

/** Header status chip {label, value, status} → display label + tone + icon. */
function toStatusChip(c: ApiStatusChip): { label: string; tone: "ok" | "warn" | "bad" | "info"; icon: string } {
  const key = (c.label ?? "").toLowerCase();
  if (key.includes("il")) return { label: `${c.value} On IL`, tone: c.value > 0 ? "bad" : "ok", icon: "activity" };
  if (key.includes("news")) return { label: `${c.value} News`, tone: "info", icon: "bell" };
  const tone = c.status === "warn" ? "warn" : c.status === "ok" ? "ok" : "info";
  return { label: `${c.value} ${c.label}`.trim(), tone, icon: "check" };
}

function toTeamMatchup(api: ApiMyTeamResponse): Matchup {
  const m = api.matchup;
  return {
    youName: api.team_name ?? "",
    youRecord: api.record ?? "",
    oppName: m?.opponent ?? "",
    winPct: Math.round((m?.win_prob ?? 0) * 100),
    tiePct: Math.round((m?.tie_prob ?? 0) * 100),
    lossPct: Math.round((m?.loss_prob ?? 0) * 100),
    // oppRecord / projLine / deltaVsLastWeek / logos omitted — not in the contract
    // (projLine + delta are history; WinHero defaults the logos to placeholders).
  };
}

function toMover(m: ApiMover): Mover {
  const r = toPlayerRef(m.player);
  const stats = m.stats ?? [];
  return {
    name: r.name,
    pos: r.pos,
    teamAbbr: r.teamAbbr,
    teamId: r.teamId,
    mlbId: r.mlbId,
    stat1: stats[0] ?? { label: "", value: "" },
    stat2: stats[1] ?? { label: "", value: "" },
    context: m.context ?? "",
    trend: (MOVER_TREND.has(m.trend) ? m.trend : "flat") as Mover["trend"],
    tag: m.tag ?? "",
    rosteredByYou: m.rostered_by_you ?? true,
    // spark + ownPct omitted — not in the contract (MoverCard hides them).
  };
}

/** API CategoryLine → frontend CategoryRow. The display edge + edgeDir are
 *  computed from you/opp/inverse (polarity-resolved: positive = good for you), so
 *  inverse cats (ERA/WHIP/L) read correctly regardless of the API `edge` sign. */
function toCategoryRow(c: ApiCategoryLine, leverKey: string): CategoryRow {
  const inverse = c.inverse ?? false;
  const you = c.you ?? 0;
  const opp = c.opp ?? 0;
  const polarity = inverse ? opp - you : you - opp; // > 0 ⇒ you're ahead
  const edgeDir = polarity > EDGE_EPS ? "good" : polarity < -EDGE_EPS ? "bad" : "even";
  const sign = polarity < -EDGE_EPS ? "−" : "+";
  return {
    key: c.cat,
    you: fmtCatVal(c.cat, you),
    opp: fmtCatVal(c.cat, opp),
    edge: `${sign}${fmtCatVal(c.cat, Math.abs(polarity))}`,
    edgeDir,
    winPct: Math.round((c.win_prob ?? 0) * 100),
    higherBetter: !inverse,
    isLever: !!leverKey && c.cat === leverKey,
    // spark omitted — not in the contract (CategoryOutlook hides the 10-wk cell).
  };
}

function toOpsCard(o: ApiOpsCard): OpsCard {
  return {
    key: OPS_KEY[o.key] ?? "roster",
    label: o.label ?? "",
    value: o.value ?? 0,
    total: o.total ?? 0,
    verdict: o.verdict ?? "",
    status: OPS_STATUS[o.status] ?? "ok",
  };
}

/** Map /api/me/team → frontend MyTeamData. The contract covers the substance
 *  (header, matchup probs, movers, lever, categories, ops); the mock's flourishes
 *  (category/mover sparklines, ownPct, projLine/delta) aren't in the contract →
 *  omitted, and the components degrade. trajectory + win_prob_trend are deferred
 *  (per-week snapshot table) → empty arrays so the page hides those charts. */
export function apiMyTeamToData(api: ApiMyTeamResponse): MyTeamData {
  const leverKey = api.lever?.category_key ?? "";
  return {
    eyebrow: api.eyebrow ?? "",
    teamName: api.team_name ?? "",
    subline: api.subline ?? "",
    freshnessMinutes: api.freshness_minutes ?? 0,
    statusChips: (api.status_chips ?? []).map(toStatusChip),
    matchup: toTeamMatchup(api),
    moversScope: (api.movers_scope as Scope) || "mine",
    movers: (api.movers ?? []).map(toMover),
    lever: api.lever
      ? {
          categoryKey: api.lever.category_key ?? "",
          headline: api.lever.headline ?? "",
          behindBy: api.lever.behind_by ?? 0,
          pickups: (api.lever.pickups ?? []).map((p) => ({ ...toPlayerRef(p.player), projStat: p.proj_stat })),
        }
      : undefined,
    categories: (api.categories ?? []).map((c) => toCategoryRow(c, leverKey)),
    ops: (api.ops ?? []).map(toOpsCard),
    trajectory: [], // deferred (per-week history) → page hides the chart
    winProbTrend: [], // deferred (per-week history) → page hides the chart
    playoffCutRank: api.playoff_cut_rank ?? 4,
  };
}

/** Join /api/playoff-odds league[] into standings rows by team name (emoji-
 *  normalized) → populate playoffOdds. champOdds stays 0 (not in this endpoint —
 *  the panel/table hide it); returns a new StandingsData. */
export function applyPlayoffOdds(data: StandingsData, api: ApiPlayoffOddsResponse): StandingsData {
  const byName = new Map((api.league ?? []).map((t) => [normTeamName(t.team), t.playoff_odds]));
  return {
    ...data,
    teams: data.teams.map((t) => {
      const odds = byName.get(normTeamName(t.teamName));
      return odds === undefined ? t : { ...t, playoffOdds: Math.round(odds) };
    }),
  };
}

/* ── Trade Finder ───────────────────────────────────────────────────────── */

type ApiTradeSuggestion = NonNullable<ApiTradeFinderResponse["suggestions"]>[number];

/** A net-SGP gain → a plain verdict (the contract has no grade/verdict). */
function verdictFromSgp(net: number): string {
  if (net >= 1.5) return "You win";
  if (net >= 0.3) return "Favorable";
  if (net > -0.3) return "Fair";
  return "You overpay";
}

function toTradePlayer(p: Parameters<typeof toPlayerRef>[0]): TradePlayer {
  const r = toPlayerRef(p);
  // keyStat omitted — the suggestion carries no per-player stat line.
  return { ...r, posLabel: r.teamAbbr ? `${r.pos} · ${r.teamAbbr}` : r.pos };
}

/** Map /api/trade-finder → frontend TradesData. Lean contract (partner + giving/
 *  receiving + net_sgp + rationale): verdict is derived from net_sgp;
 *  grade/impact/playoffDelta/partnerRecord/needs aren't in it → omitted (the card
 *  hides them). */
export function apiTradeFinderToData(api: ApiTradeFinderResponse): TradesData {
  const recs = (api.suggestions ?? []).map((s: ApiTradeSuggestion, i) => ({
    id: i,
    partner: s.partner_team ?? "",
    netSgp: s.net_sgp,
    verdict: verdictFromSgp(s.net_sgp ?? 0),
    give: (s.giving ?? []).map(toTradePlayer),
    get: (s.receiving ?? []).map(toTradePlayer),
    rationale: s.rationale ?? "",
  }));
  return { needs: [], recs };
}

/** Map /api/trade/evaluate → frontend TradeEval (the Phase-1 SGP grade is the
 *  authority). Snake→camel; null playoff/champ deltas → undefined; giving/
 *  receiving echoed + PlayerRef-enriched. */
export function apiTradeEvaluateToData(api: ApiTradeEvaluationResponse): TradeEval {
  return {
    grade: api.grade ?? "",
    verdict: api.verdict ?? "",
    surplusSgp: api.surplus_sgp ?? 0,
    confidencePct: api.confidence_pct ?? 0,
    giving: (api.giving ?? []).map(toPlayerRef),
    receiving: (api.receiving ?? []).map(toPlayerRef),
    categoryImpacts: (api.category_impacts ?? []).map((c) => ({ cat: c.cat, delta: c.delta ?? 0 })),
    deltaPlayoffProb: api.delta_playoff_prob ?? undefined,
    deltaChampProb: api.delta_champ_prob ?? undefined,
    summary: api.summary ?? "",
    warnings: api.warnings ?? [],
  };
}

/** Map /api/databank → frontend DatabankData (player + newest-first seasons). */
export function apiDatabankToData(api: ApiDatabankResponse): DatabankData {
  return {
    player: toPlayerRef(api.player),
    seasons: (api.seasons ?? []).map((s) => ({ year: s.year, stats: s.stats ?? {} })),
  };
}

const REASON_NOTE: Record<string, string> = {
  LOCKED: "Game already started",
  IL: "On IL",
  OFF_DAY: "Not playing today",
};

/** Map one API LineupSlot → a frontend OptimizerData slot. Daily mode carries the
 *  0-100 heat value + matchup; there's no per-stat proj line (mock-only). */
function toOptSlot(s: ApiLineupSlot, kind: "starter" | "bench"): OptSlot {
  const status: SlotStatus =
    kind === "bench"
      ? s.reason === "OFF_DAY"
        ? "off"
        : "bench"
      : s.action === "SIT"
        ? "sit"
        : "start";
  const note = s.forced_start
    ? "Forced start — poor matchup"
    : s.reason
      ? (REASON_NOTE[s.reason] ?? s.reason)
      : undefined;
  return {
    slot: s.slot,
    player: toPlayerRef(s.player),
    matchup: s.matchup ?? "",
    value: Math.round(s.value ?? 0),
    status,
    note,
    currentSlot: s.current_slot || undefined,
    forcedStart: s.forced_start || undefined,
  };
}

/** Map /api/lineup/optimize (daily mode) → frontend OptimizerData. */
export function apiOptimizeToData(api: ApiLineupOptimizeResponse): OptimizerData {
  const slots = api.slots ?? [];
  const starters = slots.map((s) => toOptSlot(s, "starter"));
  const bench = (api.bench ?? []).map((s) => toOptSlot(s, "bench"));
  // Each daily swap = a benched player to START in `slot`; the OUT is the SIT
  // starter currently holding that slot (current_slot === the target slot).
  const swaps = (api.daily?.swaps ?? [])
    .map((sw) => ({
      out: slots.find((s) => s.action === "SIT" && s.current_slot === sw.slot)?.player.name ?? "",
      in: sw.player.name,
      gain: undefined as string | undefined,
    }))
    .filter((s) => s.in);
  const ip = api.daily?.ip_pace;
  const d = api.daily;
  const daily = d
    ? {
        winning: d.winning ?? [],
        losing: d.losing ?? [],
        tied: d.tied ?? [],
        rateModes: d.rate_modes ?? {},
        urgency: d.urgency ?? {},
        recommendations: d.recommendations ?? [],
      }
    : undefined;
  // Attach only when the engine returned something to show (no live Yahoo → all empty → hide the panel).
  const hasDaily =
    !!daily &&
    (daily.winning.length > 0 ||
      daily.losing.length > 0 ||
      daily.tied.length > 0 ||
      Object.keys(daily.rateModes).length > 0 ||
      daily.recommendations.length > 0);
  return {
    date: api.date || "Today",
    optimal: api.optimal ?? false,
    starters,
    bench,
    ipPace: ip ? { value: Math.round(ip.projected ?? 0), total: Math.round(ip.target ?? 0) || 54 } : undefined,
    movesLeft: undefined, // not part of the optimize contract
    swaps,
    impact: (api.impact ?? []).map((c) => ({
      key: c.key,
      proj: c.proj,
      trend: (c.trend as "up" | "down" | "flat") ?? "flat",
    })),
    daily: hasDaily ? daily : undefined,
  };
}

/** Map /api/closers → frontend ClosersData. The API is minimal (closer ref +
 *  confidence label + role + handcuffs); we derive the 0–100 job-security from
 *  the confidence label and leave `stats` empty until the backend exposes proj
 *  SV/ERA/WHIP (the "Saves Finder" enhancement — see the archive spec). */
export function apiClosersToData(api: ApiClosersResponse): ClosersData {
  return {
    entries: (api.entries ?? []).map((e) => {
      const confidence = e.confidence || "Shaky";
      return {
        team: e.team,
        closer: e.closer ? toPlayerRef(e.closer) : null,
        role: e.role || (confidence === "Committee" ? "Committee" : "Closer"),
        confidence,
        security: securityFor(confidence),
        handcuffs: (e.handcuffs ?? []).map(toPlayerRef),
        stats: [], // API has no proj SV/ERA/WHIP yet (CEO track to populate)
      };
    }),
  };
}

// --- Draft Simulator (snake_case API → camelCase view-models) ---
function toVMDraftClock(c: ApiDraftClock): VMDraftClock {
  return {
    currentPick: c.current_pick,
    round: c.round,
    pickInRound: c.pick_in_round,
    pickingTeamIndex: c.picking_team_index,
    isUserTurn: c.is_user_turn,
  };
}

function toVMDraftPick(p: ApiDraftPick): VMDraftPick {
  return {
    pick: p.pick,
    teamIndex: p.team_index,
    playerId: p.player_id,
    playerName: p.player_name,
    positions: p.positions,
  };
}

function toDraftPlayer(p: {
  id: number;
  name: string;
  positions: string;
  mlb_id?: number | null;
  team_abbr?: string | null;
  team_id?: number | null;
}): DraftPlayer {
  return { ...toPlayerRef(p), id: p.id };
}

/** Map POST /api/draft/simulate-picks → the new AI picks + fresh clock. */
export function apiDraftSimulateToData(api: ApiDraftSimulateResponse): SimResult {
  return { clock: toVMDraftClock(api.clock), picks: (api.picks ?? []).map(toVMDraftPick) };
}

/** Map POST /api/draft/recommend → recommendations + clock + summary. */
export function apiDraftRecommendToData(api: ApiDraftRecommendResponse): RecResult {
  return {
    clock: toVMDraftClock(api.clock),
    summary: api.summary ?? "",
    recs: (api.recommendations ?? []).map((r) => ({
      player: toDraftPlayer(r.player),
      rank: r.rank,
      // The engine returns raw floats (e.g. 97.3529…); round for display (matches
      // the mock's clean integers) — the heat bar still uses the value directly.
      score: Math.round(r.score),
      projectedSgp: Math.round(r.projected_sgp * 10) / 10,
      confidence: r.confidence ?? null,
      tag: r.tag ?? null,
      reason: r.reason ?? "",
    })),
  };
}
