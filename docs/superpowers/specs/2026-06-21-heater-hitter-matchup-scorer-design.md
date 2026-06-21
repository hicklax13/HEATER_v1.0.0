# HEATER Hitter Matchup Scorer + Grid — Design Spec (Probable Pitcher P2)

**Date:** 2026-06-21
**Lane:** CEO — full-stack on this feature (`src/` + `api/` + `web/`).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-probable-pitcher-schedule-design.md` (Phase P2).
**Status:** design — approved by owner 2026-06-21 ("do what is most mathematically correct / most accurate").

## Scope

Phase P2 of the Probable Pitcher feature: the **Hitter Matchup grid** — the inverse lens of
the shipped Probable Pitcher grid (P1, `028613b`). TEAM rows × next-7-days columns; each cell
shows the opposing probable **starter** that team's bats face that day (name, L/R hand, home/away)
with a **matchup difficulty** (easy/tough for the hitters), a per-team **totals strip**
(games, # vs RHP, # vs LHP), and a **matchups-rank**. Same league-connection model + filters as P1.

## The accuracy decision (why this design)

The owner asked for the *most accurate* approach, not the most granular. Two candidates were weighed:

- **Per-hitter lineup rollup** (score each rostered hitter vs the SP, average): rejected. For a
  7-day-ahead grid the daily lineup is **unknown**, so it must guess the lineup (injects error);
  individual platoon splits are small-sample **noise**; and it would be a new **uncalibrated** model.
- **Calibrated symmetric inverse** (chosen): reuse the already-calibrated pitcher matchup engine,
  viewed from the batter's side. Inherits the existing calibration, uses stable team-level wRC+
  (park/league-adjusted, large sample), and is **perfectly consistent** with the Probable grid.

### Key realization — the board already computes it

`build_stream_board` scores every probable SP via `score_stream_candidate`, which calls
`compute_pitcher_matchup_score(SP_stats, opp_context)` where `opp_context =
get_opponent_offense_context(<the SP's opponent>, SP.throws, team_strength)`
(`src/optimizer/stream_analyzer.py:83`). That opponent **is the batting team**. So the board's
per-SP `matchup_score` (raw 0–10, `src/two_start.py:265`) already encodes "this SP vs the offense
he is facing." The hitter matchup is its **calibrated inverse**:

```
hitter_difficulty(Team X vs SP Y) = clamp((10 − pitcher_matchup_score(SP Y vs Team X)) × 10, 0, 100)
```

Higher = better for hitters (= tougher for the SP), 0–100, **same orientation + band thresholds as
the Probable grid** so the existing `_band` (easy ≥60 / tough ≤40), `heatColor`, and the React grid
component all transfer. By construction the two grids are complementary (no separate tuning, no drift).

## The one new engine piece: `compute_hitter_matchup_score`

Lives beside the pitcher scorer — `src/optimizer/stream_analyzer.py` (or a thin sibling
`src/optimizer/hitter_matchup.py` if it grows). Pure, NaN-safe, self-contained, unit-testable.

```python
def compute_hitter_matchup_score(
    opp_sp_stats: dict[str, float],      # opposing SP, from the pool: k_bb_pct, xfip, csw_pct
    team_offense: dict[str, float],      # batting team vs the SP's hand: wrc_plus, k_pct (PERCENT)
    park_factor: float = 1.0,            # venue (the batting team's park when they're home)
    hitters_home: bool = True,
) -> dict:
    """Calibrated inverse of the pitcher matchup. Returns
    {difficulty: 0-100, band, matchup_10, pitcher_score}. Higher difficulty = better for hitters."""
```

Internally:
1. `opp = {"wrc_plus": team_offense["wrc_plus"], "k_pct": team_offense["k_pct"] / 100.0}`
   — **the percent→fraction boundary** (`get_opponent_offense_context` emits percent; the pitcher
   scorer expects a fraction). Pinned by a dedicated test.
2. `pitcher_score = compute_pitcher_matchup_score(opp_sp_stats, opponent_team_stats=opp,
   park_factor=park_factor, is_home=not hitters_home)` — reuses the calibrated model unchanged.
3. `difficulty = clamp((10 - pitcher_score) * 10, 0, 100)`; `band = _band(difficulty)`.

Because the service feeds it the **same** SP stats / offense / park the board used, the recompute
reproduces the board's `matchup_score` exactly → the two grids stay consistent while the function
remains independently testable.

**Engine shape facts pinned (verified in source, the M0 lesson):**
- `compute_pitcher_matchup_score` (`src/two_start.py:265`): blends `k_bb_pct`/`xfip`/`csw_pct`
  (defaults 0.10 / 4.00 / league CSW), opponent `wrc_plus`/`k_pct` (fraction), reciprocal-inverted
  park capped [0.5, 2.0], home 1.03 / away 0.97; returns 0–10.
- `get_opponent_offense_context` (`src/optimizer/stream_analyzer.py:83`): returns
  `{wrc_plus, k_pct (percent), bb_pct (percent), iso?, l14_wrc_plus?, split_source}`; prefers the
  vs-hand split (`wrc_plus_vs_lhp/rhp`), falls back to overall (`split_source` tells which).
- Opposing-SP estimators (`xfip`/`k_bb_pct`/`csw_pct`) come from `ctx.player_pool` by `mlb_id` —
  NOT from `fetch_opposing_pitchers` (whose `xfip` is always None off the season endpoint).

**Deferred (optional, owner-approved as optional): L14 recent-form blend.** `l14_wrc_plus` is already
surfaced by `get_opponent_offense_context`. Blending it into `team_offense.wrc_plus` would improve
point-in-time accuracy, but to keep the two grids consistent it must be applied to **both** the
pitcher and hitter matchup together — a follow-up, not v1.

## API — `GET /api/schedule/hitter-matchups?days=7`

Mirrors P1 exactly: same `ScheduleService` seam (the one `src/`-importing layer) → contract →
thin AST-guarded router → DI in `api/deps.py` → mount in `api/main.py` → regen `openapi.json`.
Read-only, free/ungated (consistent with `/api/streaming` + `/api/schedule/probables`). Never-raise
→ empty grid. `team_name` query param for the viewer's "yours" row tag (mirrors P1; the
ViewerContext migration is the Platform/M4 lane and will migrate both schedule routes together).

Contracts (`api/contracts/schedule.py`, reusing `PlayerRef`):

```python
class HitterMatchupCell(BaseModel):
    opp_sp: PlayerRef | None = None
    opp_sp_throws: str = ""        # "L" | "R" | ""
    opponent: str = ""             # the opposing SP's team abbr
    is_home: bool = False          # the BATTING team's home/away
    difficulty: float = 0.0        # 0-100, higher = better for hitters
    band: str = "medium"           # easy | medium | tough
    status: str = ""
    confidence: str = ""

class HitterTeamTotals(BaseModel):
    games: int = 0
    vs_rhp: int = 0
    vs_lhp: int = 0

class HitterMatchupTeamRow(BaseModel):
    team: str
    cells: list[HitterMatchupCell | None] = []   # aligned to days
    totals: HitterTeamTotals = HitterTeamTotals()
    matchups_rank: int = 0       # 1 = easiest weekly hitting schedule
    availability: str = "other"  # "yours" if the viewer rosters >=1 hitter from this MLB team, else "other"

class HitterMatchupGridResponse(BaseModel):
    days: list[str] = []
    teams: list[HitterMatchupTeamRow] = []
```

### Service assembly (`ScheduleService.hitter_matchups`)

1. `ctx = build_optimizer_context(..., user_team_name=team_name, level_filter="MLB only")`.
2. For each of the 7 dates, `board = build_stream_board(ctx, date, include_rostered=True)` (same
   call P1 uses). Each board SP row carries: `player_id, mlb_id, player_name, team, throws,
   opponent, is_home, park_factor, opp_wrc_plus, opp_k_pct, matchup_score, status, confidence`.
3. For each SP row → the **batting team = row.opponent**. Build that team's cell for the day:
   - `team_offense = {wrc_plus: row.opp_wrc_plus, k_pct: row.opp_k_pct}` (already vs the SP's hand).
   - `opp_sp_stats` from `ctx.player_pool` by `row.mlb_id` (`k_bb_pct`/`xfip`/`csw_pct`; NaN-safe defaults).
   - `park_factor = row.park_factor`; `hitters_home = not row.is_home`.
   - `compute_hitter_matchup_score(...)` → difficulty + band.
   - `opp_sp = make_player_ref(...)` from the SP row; `opp_sp_throws = row.throws`.
4. Pure grid assembly (mirrors P1 `_assemble_grid`): batting-team rows × day cells, None on off-days.
5. Per-team **totals** (count populated cells / by `opp_sp_throws`) and **matchups_rank** (rank by
   mean cell difficulty, descending; rank 1 = easiest schedule). **`availability="yours"`** for an
   MLB-team row when the viewer rosters ≥1 hitter from that team — derived once from
   `ctx.user_roster_ids` → `ctx.player_pool` rows (`is_hitter==1`) → the set of their `team` abbrs.
   Empty locally (no rosters) → all "other"; populated on Railway (same graceful degradation as P1).

**Filters (frontend):** Home / Away / Easy / Tough, plus a "Yours" toggle on the row tag. Two-start
does not apply (a pitcher concept) and is omitted.

All pure cell/score/assembly logic factored into module functions for **DB-free** unit tests
(synthetic board rows + synthetic pool stats); the `build_optimizer_context` + `build_stream_board`
orchestration is verified live (same split as P1).

## Frontend (`web/`)

A `/hitter-matchups` view (or a tab toggle on `/probables`) reusing the P1 grid component:
TEAM rows × day columns, `heatColor(difficulty)`, each cell = opposing SP (`PlayerLink` →
`PlayerDialog`) + L/R chip + home/away, the totals strip + rank per team, the same filter chips
(Home/Away/Easy/Tough) and the YOURS row tag. Mock-first; the mock shape = the API contract.
Coordinate the tab seam with the CMO's streaming page (the shared-surface rule). Verified via preview.

## Testing

- **`compute_hitter_matchup_score`** (new, against REAL shapes — the M0 lesson):
  - Inverse monotonicity: a tough SP (high k_bb_pct, low xFIP) → **low** hitter difficulty; a weak SP
    → high. Strong offense (high wRC+) → higher difficulty than a weak offense vs the same SP.
  - Park: hitter-favorable (pf>1) → higher; pitcher-park (pf<1) → lower.
  - Home/away: batting team home > away, same inputs.
  - **The percent→fraction boundary**: a `team_offense.k_pct=22.0` (percent) must produce the same
    result as the pitcher scorer fed `k_pct=0.22` — a dedicated regression test (the documented gotcha).
  - **Consistency**: feeding the scorer the same inputs the board used reproduces `10 - row.matchup_score`.
  - NaN/missing SP stats → league-average defaults, never raise / never NaN.
- **Service** (DB-free): synthetic board rows + a fake pool → assert the SP→opponent remap, totals,
  rank ordering, off-day None, and never-raise → empty grid.
- **HTTP route**: fake-service dependency-override (`tests/api/test_api_schedule.py` extended).
- **Structural**: router stays logic-free (`test_no_logic_in_routers`); `openapi.json` snapshot regen.
- **Frontend**: `pnpm build` gate + preview render.

## Risks / notes

- **CMO tab seam** — the streaming/probables page is shared; coordinate the tab integration by
  `git pull --no-rebase` before edits, reconcile after.
- **Distribution/band calibration** — `(10 − matchup) × 10` is linear; if real difficulties cluster
  (poor easy/tough spread), a sigmoid spread (matching the stream_score mapping) is a tunable
  follow-up. v1 reuses the P1 band thresholds for consistency.
- **Cost** — reuses the same 7× `build_stream_board` the Probable grid already pays; cache per
  (date-range) with a short TTL if needed (shared with P1's board cache).

## Index

- New: `compute_hitter_matchup_score` (`src/optimizer/stream_analyzer.py`), `ScheduleService.hitter_matchups`,
  `HitterMatchup*` contracts, `/api/schedule/hitter-matchups`, the `web/` hitter grid view.
- Reuses: `compute_pitcher_matchup_score` (`src/two_start.py:265`), `get_opponent_offense_context`
  + `build_stream_board` (`src/optimizer/stream_analyzer.py`), the P1 `ScheduleService` + grid component.
