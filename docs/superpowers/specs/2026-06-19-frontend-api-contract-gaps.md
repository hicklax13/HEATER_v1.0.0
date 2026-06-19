# HEATER — Frontend ↔ API Contract Gaps

**Date:** 2026-06-19
**Track:** CMO / frontend (`web/`) — input for the CEO / backend track (Sub-project B)
**Status:** analysis; not a commitment to change `api/` here

## Purpose

The frontend mocks (`web/src/lib/*-data.ts`) were hand-built richer than the
B-slice API contracts (`api/contracts/*.py`); they diverged. This doc enumerates,
per page, what the frontend needs vs what the API returns, so the backend track
can extend the contracts to reach data parity (the precondition for wiring each
page to live data, per the slice-0 wiring proof).

---

## Cross-cutting: `PlayerRef` is missing `mlb_id` and team

**This is the single highest-priority extension** — it unblocks headshots and team
logos on every page.

`api/contracts/common.py` defines `PlayerRef` as:

```python
class PlayerRef(BaseModel):
    id: int                     # HEATER internal player_id
    name: str
    positions: str
    yahoo_player_key: str | None
```

The frontend (`web/src/lib/types.ts`) defines its own `PlayerRef` as:

```typescript
interface PlayerRef {
  name: string;
  pos: string;         // single eligible-position string
  teamAbbr: string;   // e.g. "NYY"  → MLB team logo
  teamId: number;     // MLB Stats API team id → team logo URL
  mlbId: number;      // MLB Stats API player id → headshot URL
}
```

Key gaps:

| API field | Frontend need | Notes |
|-----------|---------------|-------|
| `id` (HEATER `player_id`) | Not used directly by the UI | Cannot be used for headshots or team logos |
| — | `mlb_id` | Required for `PlayerAvatar` headshot (`img.mlb.com/headshots/.../{mlbId}.jpg`) |
| — | `teamAbbr` | Required for team logo display (e.g. `NYY`) |
| — | `teamId` | MLB Stats API numeric team id; used for team logo URLs |
| `positions` (string) | `pos` (string) | Name differs; semantically equivalent — minor rename |
| `yahoo_player_key` | Not needed by the UI | Keep for write-back; UI does not display it |

**Recommendation:** extend `PlayerRef` to:

```python
class PlayerRef(BaseModel):
    id: int
    mlb_id: int | None = None
    name: str
    positions: str
    team_abbr: str | None = None
    team_id: int | None = None
    yahoo_player_key: str | None = None
```

`mlb_id` is present on `players.mlb_id` in the DB; `team_abbr`/`team_id` are
derivable from `players.team` (abbr) and the MLB Stats API team map.

---

## Per-page gaps

### 1. Team page — `/api/me/team`

API contract: `api/contracts/my_team.py` → `MyTeamResponse`
Frontend mock: `web/src/lib/data.ts` + `web/src/lib/types.ts` → `MyTeamData`

The API returns: `team_name`, `record`, `rank`, `matchup{opponent, week, win_prob, tie_prob, loss_prob}`, `categories[{cat, you, opp, edge, win_prob, inverse}]`.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `teamName` | Yes (`team_name`) | Minor rename only |
| `record` | Yes | Present |
| `rank` | Yes | Present |
| `matchup.win_prob` / `tie_prob` / `loss_prob` | Yes | Present |
| `matchup.opponent` | Partial (`opponent` string only) | Missing `oppRecord`, `oppLogo`, `youLogo`, `youRecord`, `projLine`, `deltaVsLastWeek` |
| `categories[].cat/you/opp/edge/win_prob/inverse` | Yes | Core fields present |
| `categories[].edgeDir` | No | Derived: "good"/"bad"/"even" based on `edge` sign and `inverse` flag — can be computed client-side, but `edge` is a `float` not a display string |
| `categories[].spark` | Missing | 10-week trend array for the category sparkline chart; not returned |
| `categories[].higherBetter` | Partial (`inverse` bool) | Semantically equivalent to `not inverse`; rename/alias needed |
| `categories[].isLever` | Missing | Boolean flag: which category is the single biggest gap ("lever"); not computed by API |
| `eyebrow` | Missing | Display string: "Season · Week N · Team" |
| `subline` | Missing | Display string: "3–7–1 · 10th of 12 · 7 GB from 1st" |
| `freshnessMinutes` | Missing | Data freshness age in minutes (from refresh_log) |
| `statusChips` | Missing | Array of status badges: IL count, lineup status, news count |
| `movers` | Missing | Hot/cold player movers (up to 4): each needs name, pos, teamAbbr, teamId, **mlbId**, stat1, stat2, context, trend, tag, spark, ownPct, rosteredByYou |
| `moversScope` | Missing | "mine" / "league" / "mixed" |
| `lever` | Missing | Single-category weakness object: categoryKey, headline, behindBy, pickups (each a PlayerRef + projStat) |
| `ops` | Missing | 3 operational cards: IP pace, moves left, roster health — each with value/total/verdict/status |
| `trajectory` | Missing | Week-by-week rank history array (for the rank trajectory chart) |
| `winProbTrend` | Missing | Week-by-week win probability history array |
| `playoffCutRank` | Missing | The rank that earns a playoff spot (4 in FourzynBurn) |

**Biggest gap:** The API returns a minimal core (record + matchup probs + category lines). The frontend's Team page is a full dashboard — movers, ops cards, lever, trajectory chart, win-prob trend, and playoff context are all missing from the contract.

---

### 2. Optimizer page — `/api/lineup/optimize`

API contract: `api/contracts/lineup.py` → `LineupOptimizeResponse`
Frontend mock: `web/src/lib/optimizer-data.ts` → `OptimizerData`

The API returns: `team_name`, `date`, `slots[{slot, player(PlayerRef), action, projected, forced_start, reason}]`, `summary`.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `date` | Yes | Present |
| `slots[].slot` | Yes | Present |
| `slots[].player` (name, positions) | Yes (via `PlayerRef`) | Missing `mlb_id`, `teamAbbr`, `teamId` — see cross-cutting gap |
| `slots[].status` (`start`/`sit`/`bench`/`off`) | Partial (`action`: "START"/"SIT") | Frontend has 4 states; API has 2 — "bench" (on roster but not in play today) and "off" (no game) are not distinguished |
| `slots[].value` (0–100 daily value score) | Missing | API returns `projected: float` (raw SGP or DCV number), not a 0–100 display score |
| `slots[].matchup` | Missing | Display string: "vs SF", "@NYM", "OFF" — the API doesn't return opponent context per slot |
| `slots[].proj` | Partial (`projected: float`) | Frontend wants a human-readable projected line ("6.0 IP · 7 K"); API returns a raw float |
| `slots[].note` | Partial (`reason: str | None`) | Semantically equivalent — minor rename |
| `optimal` | Missing | Boolean: is the current lineup already optimal? (drives the "lineup is already optimized" empty state) |
| `bench` slots | Partial (`slots` includes all) | Frontend separates `starters` and `bench` arrays; API has a single flat `slots` list — workable but the frontend adapter must filter |
| `ipPace` | Missing | `{value, total}` for weekly IP pace vs 54 IP target |
| `movesLeft` | Missing | `{value, total}` for remaining weekly transactions |
| `swaps` | Missing | Array of recommended swaps: `{out, in, gain}` — the key action items |
| `impact` | Missing | Per-category projected daily totals: `[{key, proj, trend}]` (e.g. "R: 6.4, trend up") |

**Biggest gap:** The API is a minimal start-vs-sit decider. The frontend is a full daily command center — it needs matchup context per slot, a 0–100 value score, the IP/moves operational counters, a swap recommendation list, and per-category projected impact. Also, bench vs starter distinction and the `optimal` flag need to be added.

---

### 3. Matchup page — `/api/matchup`

API contract: `api/contracts/matchup.py` → `MatchupResponse`
Frontend mock: `web/src/lib/matchup-data.ts` → `MatchupData`

The API returns: `team_name`, `opponent`, `week`, `projected_cat_wins`, `win_prob`, `categories[{cat, you, opp, win_prob, inverse}]`.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `week` | Yes | Present |
| `you.name` / `opp.name` | Partial (`team_name`, `opponent` strings) | Present as strings; missing `manager`, `record`, `score` (H2H cat wins this week) |
| `cats[].you` / `cats[].opp` | Yes (as floats) | Present; frontend displays as strings — minor formatting |
| `cats[].win` ("you"/"opp"/undefined) | Missing | Which side is currently winning each category; API has `win_prob` (probability) not a discrete winner |
| `win_prob` (overall matchup) | Yes | Present |
| `projected_cat_wins` | Yes | Present |
| `dateTabs` | Missing | Array of date tab labels for the week ("Live", "Totals", "Mon 6/15", …) |
| `hitterColumns` / `pitcherColumns` | Missing | Column header arrays for the matchup table ("H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP") |
| `hitters` (roster rows) | Missing | Full per-slot roster comparison: slot, you-player, opp-player — each `MatchPlayer` has name, teamAbbr, teamId, **mlbId**, pos, status string, game state, stat line array, IL/DTD badge |
| `pitchers` (roster rows) | Missing | Same as hitters but for pitching slots |
| `hitterTotals` / `pitcherTotals` | Missing | Aggregate stat columns for each side |
| `league` | Missing | All other current-week matchups (for the league-scoreboard sidebar): array of `{a, b}` each a `LeagueTeam{name, manager, record, score}` |

**Biggest gap:** The API returns only aggregate win probabilities and category-level floats. The frontend renders the full Yahoo-style matchup page — a roster-vs-roster comparison table with per-player game status, live stats, and the full league scoreboard. The API needs a completely new `hitters`/`pitchers`/`league` structure, plus per-player `mlb_id`/`teamId` for headshots and logos.

---

### 4. Trades page — `/api/trade-finder`

API contract: `api/contracts/trade_finder.py` → `TradeFinderResponse`
Frontend mock: `web/src/lib/trades-data.ts` → `TradesData`

The API returns: `team_name`, `suggestions[{partner_team, giving[PlayerRef], receiving[PlayerRef], net_sgp, rationale}]`.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `suggestions[].partner_team` | Yes | Present (maps to `partner`) |
| `suggestions[].giving` / `receiving` (PlayerRef) | Yes | Present; missing `mlb_id`, `teamAbbr`, `teamId` (cross-cutting gap); also missing `posLabel` and `keyStat` per player |
| `suggestions[].rationale` | Yes | Present |
| `suggestions[].net_sgp` | Partial | Present; frontend uses `grade` ("A-", "B+") and `verdict` ("Fair", "You win"), not raw SGP — these derived fields are missing |
| `grade` | Missing | Letter grade ("A-", "B+", etc.) derived from net_sgp; not in API |
| `verdict` | Missing | Short fairness verdict ("Fair", "You win slightly", "You win") |
| `partnerRecord` | Missing | Opponent team's record string ("11-1-0 · 1st") |
| `impact[{cat, delta, dir}]` | Missing | Per-category impact of the trade: category key, signed display delta ("+9"), direction |
| `playoffDelta` | Missing | Change in playoff odds in percentage points |
| `needs` | Missing | Top-level array of your current category needs (e.g. `["SB", "SV"]`) — used for the "Your needs" header chips |
| Per-player `posLabel` | Missing | Display string: "RF · NYM" — position + team in one label |
| Per-player `keyStat` | Missing | Headline stat line for the player: ".265 AVG · 19 HR" |

**Biggest gap:** The API returns raw `net_sgp` but the frontend's trade cards need the derived `grade`/`verdict` display values, plus per-category `impact` breakdown, `playoffDelta`, partner record, and per-player display fields (`posLabel`, `keyStat`). The top-level `needs` array (your category gaps) is also missing.

---

### 5. Players page — `/api/free-agents`

API contract: `api/contracts/free_agents.py` → `FreeAgentsResponse`
Frontend mock: `web/src/lib/players-data.ts` → `PlayersData`

**Fundamental semantic mismatch:** The API endpoint returns add/drop **recommendations** (`FreeAgentRec{add, drop, marginal_value, categories_helped, ownership_pct, rationale}`) — it answers "what moves should I make?" The frontend renders a **ranked FA pool** — a full browseable list of available players ranked by marginal value, suitable for filtering and sorting.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `freeAgents[].rank` | Missing | Ordinal rank in the FA value list |
| `freeAgents[].value` (0–100) | Partial (`marginal_value: float`) | Raw float vs 0–100 display score — different scale |
| `freeAgents[].ownPct` | Yes (`ownership_pct`) | Present; minor rename |
| `freeAgents[].ownDelta` | Missing | Ownership trend (change since yesterday) — not in API |
| `freeAgents[].hitter` | Missing | Boolean: hitter vs pitcher — not in API |
| `freeAgents[].stats` | Missing | 3 key stats per player (`[{label, value}]`) — not in API |
| `freeAgents[].fit` | Partial (`categories_helped: list[str]`) | `categories_helped` is an array; `fit` is the single most-impactful category |
| `freeAgents[].tag` | Missing | "Rising", "Streamer", "Closer Role" — not in API |
| `freeAgents[].mlbId` / `teamAbbr` / `teamId` | Missing | Cross-cutting PlayerRef gap — no headshot or team logo |
| `topNeed` | Missing | Top-level string: your biggest category gap (e.g. "SB") |
| Player pool scope | Structural | The `/api/free-agents` rec endpoint pairs adds with drops; the frontend needs a standalone **ranked pool endpoint** (e.g. `/api/free-agents/pool`) that returns all available FAs sorted by value regardless of drop pairing |

**Biggest gap:** The endpoint answers a different question than the page needs. A new endpoint (or a mode parameter) returning the full ranked FA pool is required alongside the existing recommendations endpoint.

---

### 6. Research page — `/api/leaders`

API contract: `api/contracts/leaders.py` → `LeadersResponse`
Frontend mock: `web/src/lib/research-data.ts` → `ResearchData`

The API returns: `category: str`, `rows[{rank, player(PlayerRef), value: float}]`.

| Frontend field | In API? | Gap / recommended extension |
|---------------|---------|----------------------------|
| `rows[].rank` | Yes | Present |
| `rows[].player` (name, positions) | Yes (via `PlayerRef`) | Missing `mlb_id`, `teamAbbr`, `teamId` (cross-cutting gap) |
| `rows[].value` (0–100 overall score) | Partial (`value: float`) | API `value` is the raw category stat (e.g. "24 HR" as 24.0); frontend expects a 0–100 overall player value score |
| `rows[].stats` (3-element array) | Missing | Frontend shows 3 key stats per player ("24 HR", "58 R", ".322 AVG"); API returns only the single category value |
| `rows[].trend` | Missing | "up" / "down" / "flat" — not in API |
| `rows[].tag` | Missing | Lens label ("hot", "cold", "breakout", "sell") — not in API |
| `rows[].note` | Missing | Short context line ("Leads the league in HR") — not in API |
| `rows[].hitter` | Missing | Boolean: hitter vs pitcher — not in API |
| `category` (per-category endpoint) | Partial | API is per-category (`?category=HR`); frontend's Research page shows an **overall** value leaderboard across all categories, plus lens tabs (hot/cold/breakout/sell) — a cross-category ranking, not a single-stat leaderboard |
| Lens tabs (overall / hot / cold / breakout / sell) | Missing | The frontend has 5 lenses; the API has only a per-category raw-value list — no lens filtering |

**Biggest gap:** The endpoint is per-category and returns a raw stat value. The frontend needs a cross-category overall leaderboard with 5 lens filters (overall/hot/cold/breakout/sell), each row enriched with 3 key stats, a 0–100 value score, trend, tag, and note.

---

## Prioritization

**Tier 1 — Universal fix (do first):**
Extend `api/contracts/common.py` `PlayerRef` with `mlb_id: int | None`, `team_abbr: str | None`, `team_id: int | None`. This single change unblocks headshots (`PlayerAvatar`) and team logos on every page — Team, Optimizer, Matchup, Trades, Players, and Research all derive from `PlayerRef`.

**Tier 2 — Semantic fixes (one per page, highest user-visible impact):**

- **Players:** The `/api/free-agents` endpoint answers the wrong question for the FA pool page. Add a ranked-pool endpoint (e.g. `/api/free-agents/pool`) or a mode param that returns all FAs sorted by marginal value without pairing to a drop.
- **Research:** The `/api/leaders` endpoint is per-category raw stat. Add a `/api/leaders/overall` endpoint (or a `lens` query param) returning cross-category value rows with trend/tag/note/hitter and 3 key stats.
- **Matchup:** The API needs `hitters`/`pitchers` roster comparison arrays — the full per-slot matchup table with per-player game status and live stats. This is the largest volume of new contract surface.

**Tier 3 — Dashboard enrichment (Team page sub-objects):**
Add `movers`, `lever`, `ops`, `trajectory`, `winProbTrend`, `playoffCutRank`, `statusChips`, `eyebrow`, `subline`, `freshnessMinutes` to `MyTeamResponse`. These are the dashboard sub-components below the matchup hero; the page functions at a minimal level without them but is significantly degraded.

**Tier 4 — Optimizer enrichment:**
Add `value` (0–100), `matchup` display string, `swaps`, `impact`, `ipPace`, `movesLeft`, and `optimal` to `LineupOptimizeResponse`. The start/sit decision is usable without these but the full command-center experience requires them.

**Tier 5 — Trades enrichment:**
Add `grade`, `verdict`, `partnerRecord`, `impact[]`, `playoffDelta`, per-player `posLabel`/`keyStat`, and top-level `needs[]` to `TradeFinderResponse`. The raw `net_sgp` value is interpretable but the display experience depends on these derived fields.
