# Hitter Matchup Grid (Probable Pitcher P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 7-day Hitter Matchup grid — TEAM rows × day columns showing each team's bats vs the opposing probable starter, scored as the calibrated inverse of the pitcher matchup so it stays perfectly complementary to the shipped Probable Pitcher grid (P1).

**Architecture:** One new pure engine function `compute_hitter_matchup_score` reuses the calibrated `compute_pitcher_matchup_score` (same input extraction as `score_stream_candidate`, then `(10 − pitcher_matchup) × 10`). A new `ScheduleService.hitter_matchups` reuses the same `build_stream_board ×7` the Probable grid uses, remapping each starter's row onto its **opponent's** (the batting team's) cell, plus per-team totals + a matchups-rank + a "yours" row tag. A new `GET /api/schedule/hitter-matchups` mirrors the P1 router pattern. The frontend adds an inverse grid view reusing the P1 grid component.

**Tech Stack:** Python 3.12/3.14, FastAPI 0.137.1 (pinned), pydantic v2, pytest; Next.js 16 + React 19 + TypeScript + Tailwind v4 (pnpm).

**Spec:** `docs/superpowers/specs/2026-06-21-heater-hitter-matchup-scorer-design.md`.

**Conventions (all verified in source):**
- Test runner: `.venv/Scripts/python.exe -m pytest`.
- Ruff before every commit: `python -m ruff format <files>` then `python -m ruff check <files>` (pre-commit hook enforces).
- `_get_num(obj, key, default=None)` (`src/optimizer/stream_analyzer.py:145`) extracts a finite float from a dict OR pandas Series, else `default`. Reuse it; do NOT add a new float coercer.
- `_TEAM_NEUTRAL` (module-level in `stream_analyzer.py`) carries neutral `wrc_plus`/`k_pct` (k_pct in PERCENT).
- `compute_pitcher_matchup_score` is already imported at `stream_analyzer.py:47`.
- The API service `_g(row, key, default)` (`api/services/schedule_service.py:32`) reads dict OR Series; `_f` coerces finite floats; `_band` bands 0-100 (easy ≥60 / tough ≤40); `make_player_ref` builds a NaN-safe `PlayerRef`. Reuse all of these.

---

### Task 1: Engine — `compute_hitter_matchup_score`

**Files:**
- Modify: `src/optimizer/stream_analyzer.py` (add the function near `get_opponent_offense_context`, after line ~135)
- Test: `tests/test_hitter_matchup_score.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hitter_matchup_score.py`:

```python
"""Real-shape tests for compute_hitter_matchup_score — the calibrated inverse of the
pitcher matchup scorer. Calls the REAL compute_pitcher_matchup_score (no mocks)."""

from src.optimizer.stream_analyzer import compute_hitter_matchup_score
from src.two_start import compute_pitcher_matchup_score

# k_bb_pct, xfip, csw_pct: an ace vs a batting-practice arm.
_TOUGH_SP = {"k_bb_pct": 0.25, "xfip": 2.80, "csw_pct": 0.33}
_WEAK_SP = {"k_bb_pct": 0.03, "xfip": 5.20, "csw_pct": 0.24}
# team_offense: wrc_plus + k_pct in PERCENT (matches get_opponent_offense_context output).
_STRONG_OFF = {"wrc_plus": 120.0, "k_pct": 18.0}
_WEAK_OFF = {"wrc_plus": 82.0, "k_pct": 27.0}


def _inverse(sp, wrc, k_pct_frac, pf, sp_is_home):
    p = compute_pitcher_matchup_score(
        sp, opponent_team_stats={"wrc_plus": wrc, "k_pct": k_pct_frac},
        park_factor=pf, is_home=sp_is_home,
    )
    return round(max(0.0, min(100.0, (10.0 - p) * 10.0)), 1)


def test_weak_pitcher_is_a_better_hitting_matchup_than_an_ace():
    tough = compute_hitter_matchup_score(_TOUGH_SP, _STRONG_OFF, park_factor=1.0, hitters_home=True)
    weak = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=1.0, hitters_home=True)
    assert 0.0 <= tough <= 100.0 and 0.0 <= weak <= 100.0
    assert weak > tough


def test_strong_offense_scores_higher_than_weak_vs_same_pitcher():
    strong = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=True)
    weak = compute_hitter_matchup_score(_WEAK_SP, _WEAK_OFF, hitters_home=True)
    assert strong > weak


def test_hitter_friendly_park_raises_difficulty():
    coors = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=1.30, hitters_home=True)
    pitcher_park = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=0.92, hitters_home=True)
    assert coors > pitcher_park


def test_home_bats_beat_away_bats_same_inputs():
    home = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=True)
    away = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=False)
    assert home > away


def test_percent_to_fraction_boundary():
    # team_offense.k_pct is PERCENT (22.0); the pitcher scorer expects a FRACTION (0.22).
    got = compute_hitter_matchup_score(_WEAK_SP, {"wrc_plus": 100.0, "k_pct": 22.0},
                                       park_factor=1.0, hitters_home=True)
    # hitters_home=True -> the SP is AWAY -> is_home=False
    assert got == _inverse(_WEAK_SP, 100.0, 0.22, 1.0, False)


def test_exact_complement_of_board_matchup():
    sp = {"k_bb_pct": 0.15, "xfip": 3.6, "csw_pct": 0.30}
    off = {"wrc_plus": 105.0, "k_pct": 21.0}
    got = compute_hitter_matchup_score(sp, off, park_factor=1.05, hitters_home=True)
    assert got == _inverse(sp, 105.0, 0.21, 1.05, False)


def test_nan_and_missing_inputs_never_raise_and_stay_in_range():
    nan = float("nan")
    a = compute_hitter_matchup_score({"k_bb_pct": nan, "xfip": nan, "csw_pct": nan},
                                     {"wrc_plus": nan, "k_pct": nan})
    b = compute_hitter_matchup_score({}, {})
    assert isinstance(a, float) and 0.0 <= a <= 100.0
    assert isinstance(b, float) and 0.0 <= b <= 100.0
    # all-missing -> engine league defaults -> the two are identical
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hitter_matchup_score.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_hitter_matchup_score'`.

- [ ] **Step 3: Implement the function**

In `src/optimizer/stream_analyzer.py`, immediately after `get_opponent_offense_context` (ends ~line 135), add:

```python
def compute_hitter_matchup_score(
    opp_sp_stats: Any,
    team_offense: Any,
    park_factor: float = 1.0,
    hitters_home: bool = True,
) -> float:
    """Calibrated inverse of the pitcher matchup, for the batting team's side.

    Mirrors ``score_stream_candidate``'s matchup block (same input extraction) so the
    result is the EXACT complement of the Probable grid's ``matchup_score``: a starter
    who is a great stream against this offense is, by construction, a tough matchup for
    these bats.

    Args:
        opp_sp_stats: The opposing starter's pool row (dict/Series): reads
            ``k_bb_pct``/``xfip``/``csw_pct``/``era``/``whip`` (missing -> engine defaults).
        team_offense: The batting team's offense vs the SP's hand: ``wrc_plus`` and
            ``k_pct`` (in PERCENT, as ``get_opponent_offense_context`` emits).
        park_factor: Venue park factor (the batting team's park when they are home).
        hitters_home: True if the batting team is at home.

    Returns:
        0-100, higher = better hitting matchup. NaN-safe, never raises.
    """
    pitcher_stats = {
        key: val
        for key in ("k_bb_pct", "xfip", "csw_pct", "era", "whip")
        if (val := _get_num(opp_sp_stats, key)) is not None
    }
    opp_stats = {
        "wrc_plus": _get_num(team_offense, "wrc_plus", _TEAM_NEUTRAL["wrc_plus"]),
        "k_pct": _get_num(team_offense, "k_pct", _TEAM_NEUTRAL["k_pct"]) / 100.0,
    }
    pitcher_score = compute_pitcher_matchup_score(
        pitcher_stats,
        opponent_team_stats=opp_stats,
        park_factor=park_factor,
        is_home=not hitters_home,
    )
    return round(max(0.0, min(100.0, (10.0 - pitcher_score) * 10.0)), 1)
```

(`Any`, `_get_num`, `_TEAM_NEUTRAL`, and `compute_pitcher_matchup_score` are already imported/defined in this module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hitter_matchup_score.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
python -m ruff format src/optimizer/stream_analyzer.py tests/test_hitter_matchup_score.py
python -m ruff check src/optimizer/stream_analyzer.py tests/test_hitter_matchup_score.py
git add src/optimizer/stream_analyzer.py tests/test_hitter_matchup_score.py
git commit -m "feat(engine): compute_hitter_matchup_score — calibrated inverse of the pitcher matchup (P2)"
```

---

### Task 2: Contracts + Service — `ScheduleService.hitter_matchups`

**Files:**
- Modify: `api/contracts/schedule.py` (append the 4 hitter models)
- Modify: `api/services/schedule_service.py` (add pure helpers + the `hitter_matchups` method)
- Test: `tests/api/test_schedule_service.py` (append DB-free tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_schedule_service.py`:

```python
from api.services.schedule_service import (
    _assemble_hitter_grid,
    _to_hitter_cell,
)


def _sp_row(pid, team, opponent, **kw):
    """A synthetic build_stream_board row (the STARTER's row)."""
    base = {
        "player_id": pid,
        "player_name": f"SP{pid}",
        "mlb_id": 600000 + pid,
        "team": team,            # the starter's team
        "opponent": opponent,    # the batting team this starter faces
        "throws": "R",
        "is_home": True,         # the STARTER is home -> batting team is away
        "park_factor": 1.0,
        "opp_wrc_plus": 100.0,
        "opp_k_pct": 22.0,
        "status": "PROBABLE",
        "confidence": "HIGH",
    }
    base.update(kw)
    return base


def test_to_hitter_cell_remaps_to_batting_team_and_inverts_home():
    # SP on LAD (home) vs SF -> the cell belongs to SF, and SF is AWAY.
    batting_team, cell = _to_hitter_cell(_sp_row(1, "LAD", "SF"), {"xfip": 5.2})
    assert batting_team == "SF"
    assert cell.is_home is False                     # batting team away (SP was home)
    assert cell.opponent == "LAD"                    # the team SF plays = the SP's team
    assert cell.opp_sp is not None and cell.opp_sp.mlb_id == 600001
    assert cell.opp_sp_throws == "R"
    assert 0.0 <= cell.difficulty <= 100.0
    assert cell.band in {"easy", "medium", "tough"}


def test_to_hitter_cell_weak_pitcher_scores_higher_than_ace():
    _, easy = _to_hitter_cell(_sp_row(1, "LAD", "SF"), {"k_bb_pct": 0.03, "xfip": 5.4, "csw_pct": 0.24})
    _, tough = _to_hitter_cell(_sp_row(2, "LAD", "SF"), {"k_bb_pct": 0.27, "xfip": 2.7, "csw_pct": 0.34})
    assert easy.difficulty > tough.difficulty


def test_assemble_hitter_grid_offday_totals_rank_and_yours():
    dates = ["2026-06-21", "2026-06-22"]
    # Day 0: SP(NYY) faces BOS (R), SP(LAD) faces SF (L-hander).
    # Day 1: SP(NYY) faces BOS again (R). SF has an off day on day 1.
    boards = [
        [_sp_row(1, "NYY", "BOS", throws="R"), _sp_row(2, "LAD", "SF", throws="L")],
        [_sp_row(1, "NYY", "BOS", throws="R")],
    ]
    pool = {600001: {"k_bb_pct": 0.05, "xfip": 5.0}, 600002: {"k_bb_pct": 0.28, "xfip": 2.6}}
    grid = _assemble_hitter_grid(boards, dates, pool, user_hitter_teams={"BOS"})
    assert grid.days == dates
    rows = {r.team: r for r in grid.teams}
    assert set(rows) == {"BOS", "SF"}
    # BOS bats both days (vs a weak RHP) -> 2 games, both vs RHP, none LHP.
    assert rows["BOS"].totals.games == 2
    assert rows["BOS"].totals.vs_rhp == 2 and rows["BOS"].totals.vs_lhp == 0
    # SF bats day 0 only -> off-day cell is None.
    assert rows["SF"].cells[1] is None and rows["SF"].totals.games == 1
    assert rows["SF"].totals.vs_lhp == 1
    # BOS (weak RHP both days) is an easier schedule than SF (ace LHP) -> rank 1.
    assert rows["BOS"].matchups_rank == 1 and rows["SF"].matchups_rank == 2
    # Yours tag: viewer rosters >=1 BOS hitter.
    assert rows["BOS"].availability == "yours" and rows["SF"].availability == "other"


def test_assemble_hitter_grid_skips_blank_batting_team():
    grid = _assemble_hitter_grid([[_sp_row(1, "LAD", "")]], ["2026-06-21"], {}, set())
    assert grid.teams == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_schedule_service.py -q`
Expected: FAIL — `ImportError: cannot import name '_assemble_hitter_grid'`.

- [ ] **Step 3: Add the contracts**

Append to `api/contracts/schedule.py`:

```python
class HitterMatchupCell(BaseModel):
    opp_sp: PlayerRef | None = None
    opp_sp_throws: str = ""  # "L" | "R" | ""
    opponent: str = ""  # the MLB team the batting team plays (the SP's team)
    is_home: bool = False  # the BATTING team's home/away
    difficulty: float = 0.0  # 0-100, higher = better for the hitters
    band: str = "medium"  # easy | medium | tough
    status: str = ""
    confidence: str = ""


class HitterTeamTotals(BaseModel):
    games: int = 0
    vs_rhp: int = 0
    vs_lhp: int = 0


class HitterMatchupTeamRow(BaseModel):
    team: str  # batting team abbreviation
    cells: list[HitterMatchupCell | None] = []  # aligned to HitterMatchupGridResponse.days
    totals: HitterTeamTotals = HitterTeamTotals()
    matchups_rank: int = 0  # 1 = easiest weekly hitting schedule
    availability: str = "other"  # "yours" if the viewer rosters >=1 hitter from this MLB team


class HitterMatchupGridResponse(BaseModel):
    days: list[str] = []
    teams: list[HitterMatchupTeamRow] = []
```

- [ ] **Step 4: Add the pure helpers + the service method**

In `api/services/schedule_service.py`, update the contract import to include the new models:

```python
from api.contracts.schedule import (
    HitterMatchupCell,
    HitterMatchupGridResponse,
    HitterMatchupTeamRow,
    HitterTeamTotals,
    ProbableCell,
    ProbableGridResponse,
    ProbableTeamRow,
)
```

Add these pure helpers (place after `_assemble_grid`, ~line 108):

```python
def _to_hitter_cell(sp_row, sp_pool_stats) -> tuple[str, HitterMatchupCell]:
    """Map a starter's board row onto its OPPONENT's (the batting team's) cell.

    Returns (batting_team, cell). The batting team faces this starter; the cell carries
    the inverse-of-the-pitcher-matchup difficulty (higher = better for the bats)."""
    from src.optimizer.stream_analyzer import compute_hitter_matchup_score

    batting_team = str(_g(sp_row, "opponent", "") or "")
    hitters_home = not bool(_g(sp_row, "is_home", False))
    team_offense = {"wrc_plus": _g(sp_row, "opp_wrc_plus"), "k_pct": _g(sp_row, "opp_k_pct")}
    difficulty = compute_hitter_matchup_score(
        sp_pool_stats or {},
        team_offense,
        park_factor=_f(_g(sp_row, "park_factor"), 1.0),
        hitters_home=hitters_home,
    )
    try:
        pid = int(_g(sp_row, "player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    cell = HitterMatchupCell(
        opp_sp=make_player_ref(
            id=pid,
            name=str(_g(sp_row, "player_name", "") or ""),
            positions="SP",
            mlb_id=_g(sp_row, "mlb_id"),
            team_abbr=_g(sp_row, "team"),
        ),
        opp_sp_throws=str(_g(sp_row, "throws", "") or ""),
        opponent=str(_g(sp_row, "team", "") or ""),
        is_home=hitters_home,
        difficulty=difficulty,
        band=_band(difficulty),
        status=str(_g(sp_row, "status", "") or ""),
        confidence=str(_g(sp_row, "confidence", "") or ""),
    )
    return batting_team, cell


def _hitter_team_row(team, cells, user_hitter_teams) -> HitterMatchupTeamRow:
    populated = [c for c in cells if c is not None]
    vs_rhp = sum(1 for c in populated if (c.opp_sp_throws or "").upper() == "R")
    vs_lhp = sum(1 for c in populated if (c.opp_sp_throws or "").upper() == "L")
    return HitterMatchupTeamRow(
        team=team,
        cells=cells,
        totals=HitterTeamTotals(games=len(populated), vs_rhp=vs_rhp, vs_lhp=vs_lhp),
        availability="yours" if team in (user_hitter_teams or set()) else "other",
    )


def _assemble_hitter_grid(
    boards_by_day: list,
    date_list: list[str],
    pool_stats: dict,
    user_hitter_teams: set,
) -> HitterMatchupGridResponse:
    """Pure assembly: each starter row -> its opponent's batting cell; then totals + rank."""
    cell_map: dict[tuple[str, int], HitterMatchupCell] = {}
    teams: set[str] = set()
    for day_index, rows in enumerate(boards_by_day):
        for row in rows or []:
            sp_stats = (pool_stats or {}).get(_g(row, "mlb_id")) or {}
            batting_team, cell = _to_hitter_cell(row, sp_stats)
            if not batting_team:
                continue
            teams.add(batting_team)
            cell_map[(batting_team, day_index)] = cell
    team_rows = [
        _hitter_team_row(
            team,
            [cell_map.get((team, i)) for i in range(len(date_list))],
            user_hitter_teams,
        )
        for team in sorted(teams)
    ]
    # matchups_rank: 1 = easiest schedule (highest mean cell difficulty).
    def _mean_diff(r: HitterMatchupTeamRow) -> float:
        vals = [c.difficulty for c in r.cells if c is not None]
        return sum(vals) / len(vals) if vals else 0.0

    for rank, r in enumerate(sorted(team_rows, key=_mean_diff, reverse=True), start=1):
        r.matchups_rank = rank
    return HitterMatchupGridResponse(days=date_list, teams=team_rows)
```

Add the two ctx-reading helpers + the public method to the `ScheduleService` class (after `probables`/`_roster_map`):

```python
    def hitter_matchups(self, days: int = 7, team_name: str | None = None) -> HitterMatchupGridResponse:
        """7-day hitter matchup grid: batting-team rows x day columns, each cell the
        opposing starter + calibrated-inverse difficulty. Reuses build_stream_board x7
        (same board as the Probable grid). team_name enables the 'yours' row tag.
        Never raises -> empty grid."""
        days = max(1, min(int(days or 7), _MAX_DAYS))
        date_list = _date_range(days)
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=LeagueConfig(),
                user_team_name=team_name or None,
                level_filter="MLB only",
            )
            pool_stats = self._pool_pitcher_stats(ctx)
            user_hitter_teams = self._user_hitter_teams(ctx)

            boards_by_day = []
            for date in date_list:
                try:
                    board = build_stream_board(ctx, date, include_rostered=True)
                    rows = [] if board is None or board.empty else [r for _, r in board.iterrows()]
                except Exception:
                    rows = []
                boards_by_day.append(rows)

            return _assemble_hitter_grid(boards_by_day, date_list, pool_stats, user_hitter_teams)
        except Exception:
            return HitterMatchupGridResponse(days=date_list, teams=[])

    @staticmethod
    def _pool_pitcher_stats(ctx) -> dict:
        """{mlb_id: {k_bb_pct, xfip, csw_pct, era, whip}} from ctx.player_pool (never-raise -> {})."""
        out: dict = {}
        try:
            pool = getattr(ctx, "player_pool", None)
            if pool is None or pool.empty or "mlb_id" not in pool.columns:
                return out
            keys = [c for c in ("k_bb_pct", "xfip", "csw_pct", "era", "whip") if c in pool.columns]
            for _, r in pool.iterrows():
                mid = r.get("mlb_id")
                try:
                    if mid is None or (isinstance(mid, float) and math.isnan(mid)):
                        continue
                    out[int(mid)] = {k: r.get(k) for k in keys}
                except (TypeError, ValueError):
                    continue
            return out
        except Exception:
            return out

    @staticmethod
    def _user_hitter_teams(ctx) -> set:
        """MLB team abbrs the viewer rosters >=1 hitter from (never-raise -> set())."""
        try:
            uids = {int(p) for p in getattr(ctx, "user_roster_ids", []) or []}
            if not uids:
                return set()
            pool = getattr(ctx, "player_pool", None)
            if pool is None or pool.empty or "player_id" not in pool.columns or "team" not in pool.columns:
                return set()
            sub = pool[pool["player_id"].isin(uids)]
            if "is_hitter" in sub.columns:
                sub = sub[sub["is_hitter"] == 1]
            return {str(t) for t in sub["team"].dropna().unique() if str(t)}
        except Exception:
            return set()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_schedule_service.py -q`
Expected: PASS (all prior P1 tests + the 4 new ones).

- [ ] **Step 6: Format, lint, commit**

```bash
python -m ruff format api/contracts/schedule.py api/services/schedule_service.py tests/api/test_schedule_service.py
python -m ruff check api/contracts/schedule.py api/services/schedule_service.py tests/api/test_schedule_service.py
git add api/contracts/schedule.py api/services/schedule_service.py tests/api/test_schedule_service.py
git commit -m "feat(api): hitter-matchup grid contracts + ScheduleService.hitter_matchups (P2)"
```

---

### Task 3: Router + HTTP test + OpenAPI

**Files:**
- Modify: `api/routers/schedule.py` (add the GET route)
- Test: `tests/api/test_api_schedule.py` (append a fake-service override test)
- Regenerate: `api/openapi.json` + `web/src/lib/api/generated.ts`

- [ ] **Step 1: Write the failing HTTP test**

Append to `tests/api/test_api_schedule.py` (mirror the existing P1 fake-service test in that file — read it first for the exact app/override fixture style, then add):

```python
def test_hitter_matchups_route_returns_grid(client_with_fake_schedule):
    """The route delegates to ScheduleService.hitter_matchups and serializes the grid."""
    client, fake = client_with_fake_schedule
    resp = client.get("/api/schedule/hitter-matchups?days=3&team_name=Team%20Hickey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["days"] == fake.hitter_days
    assert body["teams"][0]["team"] == "BOS"
    assert body["teams"][0]["totals"]["games"] == 1
    assert fake.hitter_calls == [(3, "Team Hickey")]
```

Extend the fake service used by that file's fixture so it implements `hitter_matchups` (add to the existing fake class — match its style):

```python
    def hitter_matchups(self, days=7, team_name=None):
        from api.contracts.schedule import (
            HitterMatchupCell, HitterMatchupGridResponse, HitterMatchupTeamRow, HitterTeamTotals,
        )
        self.hitter_calls.append((days, team_name))
        self.hitter_days = ["2026-06-21", "2026-06-22", "2026-06-23"][:days]
        cell = HitterMatchupCell(opp_sp_throws="R", difficulty=70.0, band="easy")
        return HitterMatchupGridResponse(
            days=self.hitter_days,
            teams=[HitterMatchupTeamRow(team="BOS", cells=[cell, None, None],
                                        totals=HitterTeamTotals(games=1, vs_rhp=1))],
        )
```

(Add `self.hitter_calls = []` to the fake's `__init__`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_schedule.py -q`
Expected: FAIL — 404 (route not mounted yet).

- [ ] **Step 3: Add the route**

Append to `api/routers/schedule.py` (and add `HitterMatchupGridResponse` to the contract import):

```python
@router.get("/hitter-matchups", response_model=HitterMatchupGridResponse)
def hitter_matchups(
    days: int = Query(7, ge=1, le=14),
    team_name: str | None = Query(None, description="Viewer's team — enables the 'yours' tag"),
    svc: ScheduleService = Depends(get_schedule_service),
) -> HitterMatchupGridResponse:
    return svc.hitter_matchups(days=days, team_name=team_name)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_schedule.py -q`
Expected: PASS.

- [ ] **Step 5: Regenerate OpenAPI + the frontend client, verify the snapshot test**

```bash
python scripts/export_openapi.py
.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q
pnpm -C web run gen:api
```

Expected: `export_openapi.py` rewrites `api/openapi.json` (adds the `/api/schedule/hitter-matchups` path); the snapshot test PASSES (it compares against the freshly-exported file). If the snapshot test fails with ONLY unrelated FastAPI error-schema diffs, confirm `fastapi==0.137.1` is the installed version (`pip show fastapi`); a stale local FastAPI produces a false mismatch.

- [ ] **Step 6: Run the no-logic-in-routers guard + commit**

```bash
.venv/Scripts/python.exe -m pytest tests/api/test_no_logic_in_routers.py -q
python -m ruff format api/routers/schedule.py tests/api/test_api_schedule.py
python -m ruff check api/routers/schedule.py tests/api/test_api_schedule.py
git add api/routers/schedule.py tests/api/test_api_schedule.py api/openapi.json web/src/lib/api/generated.ts
git commit -m "feat(api): GET /api/schedule/hitter-matchups route + openapi (P2)"
```

---

### Task 4: Frontend — Hitter Matchup grid view

**Files:**
- Create: `web/src/lib/hitter-matchups-data.ts` (types + mock + live fetch + adapter)
- Create: `web/src/components/probables/HitterMatchupGrid.tsx` (adapted from `ProbableGrid.tsx`)
- Create: `web/src/app/hitter-matchups/page.tsx` (usePageData four-state)
- Modify: `web/src/components/chrome/TopBar.tsx` (add the nav entry — CMO-shared chrome; `git pull --no-rebase` first)
- Regenerated already in Task 3: `web/src/lib/api/generated.ts`

> **Read first:** `web/src/lib/probables-data.ts`, `web/src/components/probables/ProbableGrid.tsx`, `web/src/app/probables/page.tsx`. This task mirrors those three files; the deltas are spelled out below.

- [ ] **Step 1: Create the data module**

Create `web/src/lib/hitter-matchups-data.ts` — mirror `probables-data.ts` exactly, with these field changes per cell: `oppSp` (PlayerRef-like, replaces `pitcher`), `oppSpThrows: "L"|"R"|""`, `opponent`, `isHome`, `difficulty`, `band`, plus per-row `totals {games, vsRhp, vsLhp}`, `matchupsRank`, `availability: "yours"|"other"`. The adapter maps the snake_case API (`opp_sp`, `opp_sp_throws`, `matchups_rank`, `totals.vs_rhp`/`vs_lhp`) → camelCase, reusing the same `toRef`, allow-list validation for `band`, NaN-safe `?? ` guards, and the `(api.teams?.length ?? 0) > 0` live-gate-else-mock pattern from `probables-data.ts`. Endpoint: `/api/schedule/hitter-matchups?days=7&team_name=...` via the same Next proxy + generated client. Provide a 4-team × 7-day mock with a mix of L/R starters and a couple off-day `null` cells.

- [ ] **Step 2: Create the grid component**

Create `web/src/components/probables/HitterMatchupGrid.tsx` — adapted from `ProbableGrid.tsx`. Deltas:
- Cell renders the **opposing starter** (`PlayerLink` on `oppSp`) + an L/R chip (`oppSpThrows`) + `isHome ? "vs " : "@ "` + `opponent`, colored by `heatColor(difficulty)` with the `Math.round(difficulty)` badge (identical color logic to P1).
- Drop the two-start treatment (not applicable).
- Add a trailing **totals column** per row: `G {games} · R {vsRhp} · L {vsLhp}` and a `#${matchupsRank}` rank chip.
- Filters: keep Home / Away / Easy / Tough; replace the availability filter with a single **Yours** toggle that dims rows whose `availability !== "yours"` (reuse the dim-not-hide pattern).
- Legend reads "hotter = better matchup for your bats."

- [ ] **Step 3: Create the page**

Create `web/src/app/hitter-matchups/page.tsx` — copy `web/src/app/probables/page.tsx` verbatim, swapping: the import to `fetchHitterMatchups` from `hitter-matchups-data`, the component to `HitterMatchupGrid`, and the page title/eyebrow to "Hitter Matchups". Keep the `usePageData(fetchHitterMatchups)` four-state machine and the stable module-level fetcher reference (no inline arrow).

- [ ] **Step 4: Add the nav entry (CMO-shared chrome)**

```bash
git pull --no-rebase origin master
```

Then add a `{ href: "/hitter-matchups", label: "Hitter Matchups" }` entry to `web/src/components/chrome/TopBar.tsx` next to the existing `/probables` entry (match the exact object shape already used there).

- [ ] **Step 5: Verify types, lint, build**

```bash
pnpm -C web exec tsc --noEmit
pnpm -C web run lint
pnpm -C web run build
```

Expected: tsc clean, lint clean, `build` succeeds and prerenders `/hitter-matchups`.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/hitter-matchups-data.ts web/src/components/probables/HitterMatchupGrid.tsx web/src/app/hitter-matchups/page.tsx web/src/components/chrome/TopBar.tsx
git commit -m "feat(web): Hitter Matchup 7-day grid view (P2 frontend)"
```

---

## Self-Review

**1. Spec coverage:**
- `compute_hitter_matchup_score` (calibrated inverse, pool SP stats, team wRC+ splits via the board's `opp_wrc_plus`/`opp_k_pct`, percent→fraction boundary) → Task 1. ✓
- `GET /api/schedule/hitter-matchups` + `HitterMatchup*` contracts + the `ScheduleService` seam → Tasks 2-3. ✓
- SP→opponent remap, totals strip (games/vs_rhp/vs_lhp), matchups-rank, "yours" row tag (≥1 hitter from that MLB team) → Task 2. ✓
- Inverse grid view + filters + reuse of the P1 component → Task 4. ✓
- Real-shape tests for the scorer; DB-free service tests; fake-service HTTP test; openapi snapshot; router-logic guard → Tasks 1-3. ✓
- L14 form blend → intentionally deferred per spec (not a task). ✓

**2. Placeholder scan:** No TBD/TODO. Frontend Task 4 references existing P1 files as templates (repo files, not other plan tasks) with explicit deltas — acceptable per the "follow existing patterns" rule; the new logic (totals/rank/throws-chip/yours-toggle) is spelled out.

**3. Type consistency:** `compute_hitter_matchup_score(opp_sp_stats, team_offense, park_factor, hitters_home) -> float` is used identically in the Task 1 tests and the Task 2 `_to_hitter_cell`. Contract field names (`opp_sp`, `opp_sp_throws`, `matchups_rank`, `totals.vs_rhp/vs_lhp`, `availability`) match across the contract definition (Task 2), the fake service + HTTP test (Task 3), and the frontend adapter (Task 4). `_to_hitter_cell` returns `(batting_team, cell)` consistently in its definition, the service loop, and the Task 2 tests.
