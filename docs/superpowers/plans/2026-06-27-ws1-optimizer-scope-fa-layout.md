# WS1 — Optimizer: Scope + FA-aware + Yahoo-slot Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live React Optimizer (a) optimize over three horizons (**Today / Rest of Week / Rest of Season**) via the engine-native `scope` values, matchup-aware on every scope; (b) recommend available FA pickups ("drop X for available Y") alongside the LP lineup; and (c) render the roster grouped by Yahoo slot (batters/pitchers split, decision-colored), mirroring the Streamlit page. Backend is `api/services/lineup_service.py` + `api/contracts/lineup.py` (the ONE place importing `src/`); frontend is the optimizer page + LineupTable + adapters.

**Architecture:** The proven API slice pattern — contract (`api/contracts/`) → service seam (`api/services/lineup_service.py`, the only `src/`-importer) → thin logic-free router (`api/routers/lineup.py`, already wired) → DI provider (`api/deps.py::get_lineup_service`, already wired) → fake-service DI test (`tests/api/test_api_*.py` via `app.dependency_overrides`) → mount (`api/main.py`, already mounted) → regenerate `api/openapi.json` (`python scripts/export_openapi.py`, snapshot-guarded by `tests/api/test_openapi_contract.py`). The router `POST /api/lineup/optimize` already forwards `req.scope` + `req.mode` to `service.optimize(team, date, scope, mode)`; `LineupService.optimize` already branches `mode=="daily"` → `_optimize_daily` else `_optimize_standard`. This WS makes `scope` (not `mode`) the driver: the service maps `scope=="today"` → the daily-DCV path, `rest_of_week`/`rest_of_season` → the standard LP at that scope. `build_optimizer_context(scope, yds, …)` natively supports all three horizons and already calls `yds.get_matchup()` internally for FA-side urgency; the standard LP additionally gets the live matchup passed to `pipeline.optimize(matchup=…)`. `recommend_fa_moves(ctx)` composes over the same `ctx` (the canonical Free Agents pattern).

**Tech Stack:** Python 3.12 (CI) / 3.14 (local), FastAPI `0.137.1` + httpx `0.28.1` (PINNED — do not bump; the openapi snapshot guard depends on them), Pydantic v2, pandas. Frontend: Next.js 16 + React 19 + TypeScript 5.9 + Tailwind v4 + framer-motion, pnpm. Tests: `python -m pytest tests/api/test_xxx.py -v` (DB-free — local/worktree DB may be EMPTY, so API tests use fake services via `app.dependency_overrides`; engine-seam unit tests monkeypatch at the BOUND name in `api.services.lineup_service`'s lazy imports, i.e. `src.yahoo_data_service.get_yahoo_data_service` etc.). Frontend gate (run from `web/`): `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build`. Windows PowerShell shell.

**Key facts verified against the real code:**
- `recommend_fa_moves(ctx, max_moves=3)` returns `list[dict]`; each dict keys (from `_evaluate_swaps`, `fa_recommender.py:1059-1075`): `add_id`, `add_name`, `add_positions`, `drop_id`, `drop_name`, `drop_positions`, `net_sgp_delta` (float), `category_impact` (a `{cat: float}` **dict**, NOT a list), `reasoning` (str), `urgency_categories` (list[str]), `news_warning`, `ownership_trend`, `sustainability`. It early-returns `[]` when `ctx.adds_remaining_this_week <= 0` or `not ctx.user_roster_ids or ctx.player_pool.empty`.
- `build_optimizer_context(scope, yds, config=None, weeks_remaining=16, …, user_team_name=None, roster=None, level_filter="MLB only")` — `scope ∈ {"today","rest_of_week","rest_of_season"}`. It internally calls `yds.get_matchup()` (shared_data_layer.py:355) and computes `ctx.urgency_weights`/`ctx.category_weights` from it.
- The standard LP path (`_optimize_standard`) currently does NOT pass the matchup to `pipeline.optimize()`; the daily path does (`_optimize_daily` line 124-131).
- `_to_slots` (lineup_service.py:485-537) builds standard starters/bench but sets NO `current_slot` (only daily `_daily_slots` does, via `_current_slots`).
- `LineupOptimizeRequest` already has `scope: str = "rest_of_season"` and `mode: str = "standard"`; the router forwards both.
- Existing DI test template: `tests/api/test_lineup.py::test_post_lineup_optimize_returns_contract` uses `app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()` + `TestClient`. The `require_viewer_context` + `require_pro` deps pass through in the unconfigured test env (no 401/402).
- Engine-seam unit-test monkeypatch targets (from `test_lineup.py::_patch_optimizer_seam`): `src.yahoo_data_service.get_yahoo_data_service`, `src.optimizer.pipeline.LineupOptimizerPipeline`, `src.game_day.get_target_game_date`, `src.database.load_player_pool`. For FA composition add `src.optimizer.shared_data_layer.build_optimizer_context` and `src.optimizer.fa_recommender.recommend_fa_moves`.
- Shared contracts (REUSE, do not redefine): `PlayerRef` + `StatItem` in `api/contracts/common.py`; `CategoryImpact` in `api/contracts/trade.py`. Build refs via `make_player_ref` / `player_ref_from_pool` in `api/services/player_ref.py`.
- openapi regen: `python scripts/export_openapi.py` writes `api/openapi.json` (indent=2, sort_keys, trailing newline).
- Frontend types regen from openapi: the repo generates `web/src/lib/api/generated.ts` from `api/openapi.json` (the `components["schemas"]` source for `web/src/lib/api/types.ts`). Find the codegen command in `web/package.json` scripts (Task 13).

---

## Task 1: Add `FaSuggestion` contract + `LineupOptimizeResponse.fa_suggestions` + `LineupSlot.value_breakdown`

**Files:**
- Modify: `api/contracts/lineup.py` (imports at line 7; `LineupSlot` lines 17-27; `LineupOptimizeResponse` lines 69-80)
- Test: `tests/api/test_lineup.py` (append at end, after line 430)

- [ ] 1. Write a failing contract test. Append to `tests/api/test_lineup.py`:
```python
# ── WS1: FaSuggestion + fa_suggestions + value_breakdown contract ──


def test_fa_suggestion_contract_shape():
    from api.contracts.common import StatItem
    from api.contracts.lineup import FaSuggestion

    fs = FaSuggestion(
        add=PlayerRef(id=10, name="Add Guy", positions="OF"),
        drop=PlayerRef(id=20, name="Drop Guy", positions="2B"),
        net_sgp_delta=1.42,
        category_impact=[StatItem(label="HR", value="+0.30"), StatItem(label="SB", value="+0.10")],
        reasoning="Upgrades HR + SB you're losing.",
        urgency_categories=["HR", "SB"],
    )
    d = fs.model_dump()
    assert d["add"]["id"] == 10 and d["drop"]["id"] == 20
    assert d["net_sgp_delta"] == 1.42
    assert d["category_impact"][0] == {"label": "HR", "value": "+0.30"}
    assert d["urgency_categories"] == ["HR", "SB"]


def test_lineup_response_fa_suggestions_defaults_empty():
    resp = LineupOptimizeResponse(team_name="T", date="2027-04-05", slots=[])
    assert resp.model_dump()["fa_suggestions"] == []


def test_lineup_slot_value_breakdown_defaults_empty():
    slot = LineupSlot(slot="OF", player=PlayerRef(id=1, name="X", positions="OF"), action="START")
    assert slot.model_dump()["value_breakdown"] == []
```
- [ ] 2. Run it — expect FAIL (ImportError: cannot import name `FaSuggestion`):
```
python -m pytest tests/api/test_lineup.py::test_fa_suggestion_contract_shape tests/api/test_lineup.py::test_lineup_response_fa_suggestions_defaults_empty tests/api/test_lineup.py::test_lineup_slot_value_breakdown_defaults_empty -v
```
- [ ] 3. Implement. In `api/contracts/lineup.py`, change the import line (line 7) from `from api.contracts.common import PlayerRef` to:
```python
from api.contracts.common import PlayerRef, StatItem
```
Then add `value_breakdown` to `LineupSlot` (insert after the `current_slot` field, line 27):
```python
    value_breakdown: list[StatItem] = Field(
        default_factory=list
    )  # per-category DCV/value contributions (WS5 inline explainer; additive, daily/standard)
```
Add a new `FaSuggestion` model immediately before `LineupOptimizeResponse` (before line 69):
```python
class FaSuggestion(BaseModel):
    """A 'drop X for available Y' free-agent upgrade, alongside the LP lineup.

    Composed from recommend_fa_moves(ctx) over the optimize context. The LP still
    optimizes the CURRENT roster; these are the available-pickup layer.
    """

    add: PlayerRef
    drop: PlayerRef
    net_sgp_delta: float = 0.0  # net SGP gain of the swap (add value − drop cost)
    category_impact: list[StatItem] = Field(default_factory=list)  # per-cat SGP deltas, formatted
    reasoning: str = ""
    urgency_categories: list[str] = Field(default_factory=list)  # losing/tied cats this swap helps
```
Add the field to `LineupOptimizeResponse` (after the `daily` field, line 80):
```python
    fa_suggestions: list[FaSuggestion] = Field(
        default_factory=list
    )  # available "drop X for Y" pickups (composed from recommend_fa_moves)
```
- [ ] 4. Run the tests — expect PASS:
```
python -m pytest tests/api/test_lineup.py::test_fa_suggestion_contract_shape tests/api/test_lineup.py::test_lineup_response_fa_suggestions_defaults_empty tests/api/test_lineup.py::test_lineup_slot_value_breakdown_defaults_empty -v
```
- [ ] 5. Commit:
```
git add api/contracts/lineup.py tests/api/test_lineup.py && git commit -m "feat(lineup-contract): FaSuggestion + fa_suggestions + LineupSlot.value_breakdown (WS1)"
```

---

## Task 2: Populate `current_slot` in the standard `_to_slots` path

**Files:**
- Modify: `api/services/lineup_service.py` (`_to_slots`, lines 484-537; reuse the existing `_current_slots` static helper at lines 437-451)
- Test: `tests/api/test_lineup.py` (append after Task 1's tests)

- [ ] 1. Write a failing test. Append to `tests/api/test_lineup.py`:
```python
# ── WS1: _to_slots populates current_slot for ALL scopes (standard, not just daily) ──


def test_to_slots_populates_current_slot_from_roster():
    lineup = _lineup([{"slot": "OF", "player_name": "Judge", "player_id": 1}])
    roster = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["Judge", "Benchy"],
            "positions": ["OF", "2B"],
            "selected_position": ["BN", "BN"],  # Judge currently benched; LP wants him in OF
        }
    )
    starters, bench = LineupService._to_slots(lineup, pool=None, roster=roster)
    assert starters[0].player.id == 1 and starters[0].current_slot == "BN"  # diffable swap (BN → OF)
    assert bench[0].player.id == 2 and bench[0].current_slot == "BN"
```
- [ ] 2. Run it — expect FAIL (`current_slot == ""`, not `"BN"`, because `_to_slots` never sets it for starters):
```
python -m pytest tests/api/test_lineup.py::test_to_slots_populates_current_slot_from_roster -v
```
- [ ] 3. Implement in `api/services/lineup_service.py::_to_slots`. After `if not isinstance(lineup, dict): return [], []` (line 491), build the current-slot map once:
```python
        current = LineupService._current_slots(roster)
```
In the starters loop, add `current_slot=current.get(pid, "")` to the `LineupSlot(...)` constructor (the one starting at line 502), e.g. after the `status="start",` line:
```python
                    current_slot=current.get(pid, ""),
```
In the bench loop, the bench `LineupSlot(...)` (line 526) already computes `slot` from `selected_position`; add `current_slot` set to that same current slot:
```python
                        current_slot=current.get(pid, ""),
```
(`_current_slots` already guards a None/empty roster → empty dict, so `current.get(pid, "")` is safe when roster is None.)
- [ ] 4. Run the test — expect PASS. Also re-run the existing `_to_slots` tests to confirm no regression:
```
python -m pytest tests/api/test_lineup.py::test_to_slots_populates_current_slot_from_roster tests/api/test_lineup.py::test_to_slots_starters_from_assignments_and_bench_from_roster -v
```
- [ ] 5. Commit:
```
git add api/services/lineup_service.py tests/api/test_lineup.py && git commit -m "feat(lineup): populate current_slot in standard _to_slots (WS1 Yahoo-slot layout)"
```

---

## Task 3: Route `scope` end-to-end + pass live matchup into the standard LP

**Files:**
- Modify: `api/services/lineup_service.py` (`optimize`, lines 39-44; `_optimize_standard`, lines 47-93)
- Test: `tests/api/test_lineup.py` (append after Task 2's test)

- [ ] 1. Write a failing test. Append to `tests/api/test_lineup.py`:
```python
# ── WS1: scope routing + standard-path live-matchup pass-through ──


def test_optimize_scope_today_routes_to_daily_path(monkeypatch):
    captured = {}

    class _FakePipeline:
        def __init__(self, roster, **kw):
            captured["mode"] = kw.get("mode")

        def optimize(self, **kw):
            captured["matchup"] = kw.get("matchup")
            return {"daily_dcv": captured.get("dcv"), "daily_lineup": {"starters": [], "bench": []}}

    calls = _patch_optimizer_seam(monkeypatch, _FakePipeline)
    resp = LineupService().optimize(team_name="Team Hickey", scope="today")
    assert captured["mode"] == "daily"  # scope=today → daily-DCV path
    assert resp.mode == "daily"


def test_optimize_scope_week_passes_matchup_to_standard_lp(monkeypatch):
    captured = {}

    class _FakePipeline:
        def __init__(self, roster, **kw):
            captured["mode"] = kw.get("mode")

        def optimize(self, **kw):
            captured["matchup"] = kw.get("matchup")
            ids = list(captured_roster["player_id"])
            return {
                "lineup": {
                    "assignments": [{"slot": "OF", "player_name": "x", "player_id": int(i)} for i in ids],
                    "projected_stats": {"hr": 55},
                    "status": "Optimal",
                }
            }

    calls = _patch_optimizer_seam(monkeypatch, _FakePipeline)
    # the seam's _OptYDS.get_matchup returns None; make it return a sentinel so we can assert pass-through
    class _MatchupYDS(_OptYDS):
        def get_matchup(self):
            return {"categories": []}

    captured_roster_holder = {}

    def _yds():
        y = _MatchupYDS(calls)
        return y

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _yds)
    global captured_roster
    captured_roster = _MatchupYDS(calls).get_team_roster("Team Hickey")
    resp = LineupService().optimize(team_name="Team Hickey", scope="rest_of_week")
    assert captured["mode"] == "standard"  # week/season → standard LP
    assert captured["matchup"] == {"categories": []}  # live matchup threaded into the standard LP
    assert resp.mode == "standard"
```
- [ ] 2. Run it — expect FAIL (`scope="today"` currently routes by `mode`, which defaults to `"standard"`, so `captured["mode"]` is `"standard"` not `"daily"`; and `_optimize_standard` never passes `matchup`, so `captured["matchup"]` is None):
```
python -m pytest "tests/api/test_lineup.py::test_optimize_scope_today_routes_to_daily_path" "tests/api/test_lineup.py::test_optimize_scope_week_passes_matchup_to_standard_lp" -v
```
- [ ] 3. Implement in `api/services/lineup_service.py`. Replace the `optimize` method body (lines 39-44) so `scope` drives the path (`mode` becomes a back-compat override):
```python
    def optimize(
        self, team_name: str, date=None, scope: str = "rest_of_season", mode: str = "standard"
    ) -> LineupOptimizeResponse:
        # scope is the driver: today → daily DCV; rest_of_week / rest_of_season → standard LP.
        # mode is kept as a back-compat override (an explicit mode="daily" still forces daily).
        sc = str(scope or "rest_of_season").lower()
        if sc == "today" or str(mode or "standard").lower() == "daily":
            return self._optimize_daily(team_name, date)
        return self._optimize_standard(team_name, date, sc)
```
In `_optimize_standard`, fetch the live matchup (graceful) and pass it to `pipeline.optimize(...)`. After the `roster = yds.get_team_roster(team_name)` line (line 69), insert:
```python
            matchup = None
            try:
                matchup = yds.get_matchup()
            except Exception:
                matchup = None
```
Change the `result = pipeline.optimize() if hasattr(pipeline, "optimize") else None` line (line 73) to:
```python
            result = pipeline.optimize(matchup=matchup) if hasattr(pipeline, "optimize") else None
```
(The pipeline accepts `**kwargs` and reads `kwargs.get("matchup")` for urgency; passing it on the standard path biases week/season optimize. No-matchup → None → engine falls back to neutral weights, never raises.)
- [ ] 4. Run the tests — expect PASS. Re-run the standard-roster regression test too:
```
python -m pytest "tests/api/test_lineup.py::test_optimize_scope_today_routes_to_daily_path" "tests/api/test_lineup.py::test_optimize_scope_week_passes_matchup_to_standard_lp" "tests/api/test_lineup.py::test_optimize_standard_uses_enriched_team_roster_not_all_teams" -v
```
- [ ] 5. Commit:
```
git add api/services/lineup_service.py tests/api/test_lineup.py && git commit -m "feat(lineup): scope drives daily-vs-standard path + standard LP is matchup-aware (WS1)"
```

---

## Task 4: Add `_fa_suggestions` builder + map FA dicts → `FaSuggestion`

**Files:**
- Modify: `api/services/lineup_service.py` (imports lines 11-19; add a `_fa_suggestions` static helper near the other shared helpers, after `_load_pool` at line 394)
- Test: `tests/api/test_lineup.py` (append)

- [ ] 1. Write a failing test (pure mapper — no engine call). Append to `tests/api/test_lineup.py`:
```python
# ── WS1: FA-move dict → FaSuggestion mapping (DB-free; the engine's real dict keys) ──


def test_fa_moves_to_suggestions_maps_engine_dict_keys():
    moves = [
        {
            "add_id": 10,
            "add_name": "Add Guy",
            "add_positions": "OF",
            "drop_id": 20,
            "drop_name": "Drop Guy",
            "drop_positions": "2B",
            "net_sgp_delta": 1.4231,
            "category_impact": {"HR": 0.3012, "SB": 0.1001, "ERA": -0.02},  # a {cat: float} DICT
            "reasoning": "Upgrades HR + SB.",
            "urgency_categories": ["HR", "SB"],
        }
    ]
    out = LineupService._fa_suggestions(moves, pool=None)
    assert len(out) == 1
    fs = out[0]
    assert fs.add.id == 10 and fs.add.name == "Add Guy"
    assert fs.drop.id == 20 and fs.drop.name == "Drop Guy"
    assert fs.net_sgp_delta == 1.4231
    assert fs.urgency_categories == ["HR", "SB"]
    labels = {si.label: si.value for si in fs.category_impact}
    assert labels["HR"] == "+0.30" and labels["SB"] == "+0.10" and labels["ERA"] == "-0.02"


def test_fa_moves_to_suggestions_handles_empty_and_garbage():
    assert LineupService._fa_suggestions(None, None) == []
    assert LineupService._fa_suggestions([], None) == []
    # a malformed move (missing ids) is skipped, never raises
    assert LineupService._fa_suggestions([{"reasoning": "x"}], None) == []
```
- [ ] 2. Run it — expect FAIL (AttributeError: `_fa_suggestions`):
```
python -m pytest "tests/api/test_lineup.py::test_fa_moves_to_suggestions_maps_engine_dict_keys" "tests/api/test_lineup.py::test_fa_moves_to_suggestions_handles_empty_and_garbage" -v
```
- [ ] 3. Implement. First extend the contract import in `api/services/lineup_service.py` (lines 11-19) to include `FaSuggestion`:
```python
from api.contracts.lineup import (
    CatImpact,
    DailyMeta,
    FaSuggestion,
    IpPace,
    LineupOptimizeResponse,
    LineupSlot,
    Swap,
)
```
Also import the shared `StatItem`:
```python
from api.contracts.common import StatItem
```
(place it next to the existing `from api.services.player_ref import player_ref_from_pool` import, line 19.) Then add the static helper after `_load_pool` (after line 394):
```python
    @staticmethod
    def _fa_suggestions(moves, pool) -> list[FaSuggestion]:
        """Map recommend_fa_moves() dicts → FaSuggestion contracts.

        The engine emits category_impact as a {cat: float} dict; we render each as a
        StatItem(label=cat, value=+/-N.NN SGP). A move missing add_id/drop_id is skipped
        (never raises). PlayerRefs are pool-enriched (mlb_id/team for headshots/logos)."""
        out: list[FaSuggestion] = []
        if not isinstance(moves, list):
            return out
        for m in moves:
            if not isinstance(m, dict):
                continue
            try:
                add_id = int(m.get("add_id", 0) or 0)
                drop_id = int(m.get("drop_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if not add_id or not drop_id:
                continue
            cat_items: list[StatItem] = []
            for cat, val in dict(m.get("category_impact") or {}).items():
                fv = LineupService._f(val, float("nan"))
                if not math.isfinite(fv):
                    continue
                cat_items.append(StatItem(label=str(cat), value=f"{fv:+.2f}"))
            out.append(
                FaSuggestion(
                    add=player_ref_from_pool(
                        add_id, pool, name=m.get("add_name"), positions=m.get("add_positions")
                    ),
                    drop=player_ref_from_pool(
                        drop_id, pool, name=m.get("drop_name"), positions=m.get("drop_positions")
                    ),
                    net_sgp_delta=LineupService._f(m.get("net_sgp_delta")),
                    category_impact=cat_items,
                    reasoning=str(m.get("reasoning") or "").strip(),
                    urgency_categories=[str(c) for c in (m.get("urgency_categories") or [])],
                )
            )
        return out
```
- [ ] 4. Run the tests — expect PASS:
```
python -m pytest "tests/api/test_lineup.py::test_fa_moves_to_suggestions_maps_engine_dict_keys" "tests/api/test_lineup.py::test_fa_moves_to_suggestions_handles_empty_and_garbage" -v
```
- [ ] 5. Commit:
```
git add api/services/lineup_service.py tests/api/test_lineup.py && git commit -m "feat(lineup): _fa_suggestions mapper (engine dict → FaSuggestion contract) (WS1)"
```

---

## Task 5: Compose `recommend_fa_moves(ctx)` into both optimize paths

**Files:**
- Modify: `api/services/lineup_service.py` (`_optimize_standard` return, lines 84-93; `_optimize_daily` return, lines 143-152; add a `_compose_fa` static helper)
- Test: `tests/api/test_lineup.py` (append)

- [ ] 1. Write a failing engine-seam test (monkeypatch `build_optimizer_context` + `recommend_fa_moves` at their BOUND lazy-import names). Append to `tests/api/test_lineup.py`:
```python
# ── WS1: FA composition — _compose_fa builds ctx at scope and maps recommend_fa_moves ──


def test_compose_fa_calls_context_at_scope_and_maps_moves(monkeypatch):
    seen = {}

    def _fake_ctx(scope, yds, **kw):
        seen["scope"] = scope
        seen["user_team_name"] = kw.get("user_team_name")
        return object()  # opaque ctx; recommend_fa_moves is faked below

    def _fake_recs(ctx, max_moves=3):
        seen["max_moves"] = max_moves
        return [
            {
                "add_id": 10,
                "add_name": "Add Guy",
                "add_positions": "OF",
                "drop_id": 20,
                "drop_name": "Drop Guy",
                "drop_positions": "2B",
                "net_sgp_delta": 1.0,
                "category_impact": {"HR": 0.3},
                "reasoning": "ok",
                "urgency_categories": ["HR"],
            }
        ]

    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _fake_ctx)
    monkeypatch.setattr("src.optimizer.fa_recommender.recommend_fa_moves", _fake_recs)
    out = LineupService()._compose_fa("Team Hickey", "rest_of_week", yds=object(), pool=None)
    assert seen["scope"] == "rest_of_week" and seen["user_team_name"] == "Team Hickey"
    assert len(out) == 1 and out[0].add.id == 10


def test_compose_fa_never_raises_on_engine_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no live data")

    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _boom)
    assert LineupService()._compose_fa("T", "today", yds=object(), pool=None) == []
```
- [ ] 2. Run it — expect FAIL (AttributeError: `_compose_fa`):
```
python -m pytest "tests/api/test_lineup.py::test_compose_fa_calls_context_at_scope_and_maps_moves" "tests/api/test_lineup.py::test_compose_fa_never_raises_on_engine_failure" -v
```
- [ ] 3. Implement. Add the `_compose_fa` helper in `api/services/lineup_service.py` (after `_fa_suggestions`):
```python
    def _compose_fa(self, team_name: str, scope: str, yds, pool) -> list[FaSuggestion]:
        """Build a matchup-aware optimizer context at `scope` and map recommend_fa_moves
        → FaSuggestions (the canonical Free Agents composition). build_optimizer_context
        calls yds.get_matchup() internally for FA-side urgency. Never raises (missing live
        data / no roster → []); recommend_fa_moves itself returns [] when adds_remaining<=0
        or the roster/pool is empty."""
        try:
            from src.optimizer.fa_recommender import recommend_fa_moves
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.valuation import LeagueConfig

            ctx = build_optimizer_context(
                scope,
                yds,
                config=LeagueConfig(),
                user_team_name=team_name,
            )
            moves = recommend_fa_moves(ctx)
            return self._fa_suggestions(moves, pool)
        except Exception as exc:
            logger.warning("LineupService._compose_fa failed: %s", exc)
            return []
```
Now wire it into both return paths. In `_optimize_standard`, after `optimal = self._optimal(...)` (line 79) and before the `if slots:` block, compose FA suggestions (scope is the method's `scope` arg):
```python
            fa_suggestions = self._compose_fa(team_name, scope, yds, pool)
```
Add `fa_suggestions=fa_suggestions,` to the `LineupOptimizeResponse(...)` return (after `mode="standard",`, line 92). Initialize `fa_suggestions: list[FaSuggestion] = []` at the top of `_optimize_standard` (near the other locals, after line 53) so the except path still returns a valid (empty) list:
```python
        fa_suggestions: list[FaSuggestion] = []
```
In `_optimize_daily`, do the same: initialize `fa_suggestions: list[FaSuggestion] = []` near line 102, compose after `optimal = self._optimal(...)` (line 138) using scope `"today"`:
```python
            fa_suggestions = self._compose_fa(team_name, "today", yds, pool)
```
and add `fa_suggestions=fa_suggestions,` to the daily `LineupOptimizeResponse(...)` return (after `daily=daily,`, line 151).
- [ ] 4. Run the tests — expect PASS. Re-run the full lineup test file to confirm no regression:
```
python -m pytest tests/api/test_lineup.py -v
```
- [ ] 5. Commit:
```
git add api/services/lineup_service.py tests/api/test_lineup.py && git commit -m "feat(lineup): compose recommend_fa_moves into both optimize paths → fa_suggestions (WS1)"
```

---

## Task 6: End-to-end DI test for `fa_suggestions` over the router

**Files:**
- Test: `tests/api/test_api_lineup_fa.py` (Create — basename MUST start with `test_api_`)

- [ ] 1. Write the failing DI test (fake service via `app.dependency_overrides`, asserts the new field round-trips through the route). Create `tests/api/test_api_lineup_fa.py`:
```python
"""WS1 DI test: fa_suggestions round-trips through POST /api/lineup/optimize."""

from starlette.testclient import TestClient

from api.contracts.common import PlayerRef, StatItem
from api.contracts.lineup import FaSuggestion, LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app


class _FakeLineupService:
    def optimize(self, team_name, date=None, scope="rest_of_season", mode="standard"):
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="Starter", positions="OF"),
                    action="START",
                    status="start",
                    current_slot="BN",
                )
            ],
            summary="1 starter",
            mode="standard" if scope != "today" else "daily",
            fa_suggestions=[
                FaSuggestion(
                    add=PlayerRef(id=10, name="Add Guy", positions="OF"),
                    drop=PlayerRef(id=20, name="Drop Guy", positions="2B"),
                    net_sgp_delta=1.4,
                    category_impact=[StatItem(label="HR", value="+0.30")],
                    reasoning="Upgrades HR.",
                    urgency_categories=["HR"],
                )
            ],
        )


def _client():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    return TestClient(app)


def test_optimize_returns_fa_suggestions():
    body = _client().post(
        "/api/lineup/optimize", json={"team_name": "Team Hickey", "scope": "rest_of_week"}
    ).json()
    assert body["mode"] == "standard"
    fs = body["fa_suggestions"]
    assert len(fs) == 1
    assert fs[0]["add"]["id"] == 10 and fs[0]["drop"]["id"] == 20
    assert fs[0]["category_impact"][0] == {"label": "HR", "value": "+0.30"}
    assert fs[0]["urgency_categories"] == ["HR"]
    # current_slot present on the lineup slot (layout grouping key)
    assert body["slots"][0]["current_slot"] == "BN"


def test_optimize_scope_today_echoes_daily_mode():
    body = _client().post("/api/lineup/optimize", json={"team_name": "Team Hickey", "scope": "today"}).json()
    assert body["mode"] == "daily"
```
- [ ] 2. Run it — expect FAIL only if a contract field is missing; since Tasks 1-5 added them, it should PASS once those are in. Run to confirm:
```
python -m pytest tests/api/test_api_lineup_fa.py -v
```
- [ ] 3. (If it fails for a reason other than missing fields, fix the cause in the service/contract. No new impl expected here — this is the integration guard.)
- [ ] 4. Confirm the router-no-logic + pro-gating structural guards still pass (the router is unchanged but re-run to be safe):
```
python -m pytest tests/api/test_no_logic_in_routers.py tests/api/test_api_pro_gating.py -v
```
- [ ] 5. Commit:
```
git add tests/api/test_api_lineup_fa.py && git commit -m "test(api): fa_suggestions + scope-today DI round-trip over /api/lineup/optimize (WS1)"
```

---

## Task 7: Regenerate `api/openapi.json`

**Files:**
- Modify: `api/openapi.json` (generated)

- [ ] 1. Confirm the snapshot guard currently FAILS (the contract changed in Tasks 1-5 but openapi is stale):
```
python -m pytest tests/api/test_openapi_contract.py -v
```
Expect FAIL: "api/openapi.json is stale".
- [ ] 2. Regenerate:
```
python scripts/export_openapi.py
```
Expect output: `wrote .../api/openapi.json`.
- [ ] 3. Verify the snapshot guard now PASSES, and that `FaSuggestion` is in the schema:
```
python -m pytest tests/api/test_openapi_contract.py -v
```
- [ ] 4. Run the whole `tests/api/` suite to confirm green across the slice:
```
python -m pytest tests/api/ -q
```
- [ ] 5. Commit:
```
git add api/openapi.json && git commit -m "chore(openapi): regenerate for FaSuggestion + fa_suggestions + value_breakdown (WS1)"
```

---

## Task 8: Regenerate frontend TS types from openapi

**Files:**
- Modify: `web/src/lib/api/generated.ts` (generated from `api/openapi.json`)

- [ ] 1. Find the codegen script. From `web/`, inspect `package.json` scripts for the openapi→TS generator (e.g. a `gen:api` / `openapi-typescript` script):
```
python -c "import json; s=json.load(open('web/package.json'))['scripts']; print('\n'.join(f'{k}: {v}' for k,v in s.items()))"
```
- [ ] 2. Run that codegen script from `web/` (substitute the real script name found in step 1; it points at `../api/openapi.json` → `src/lib/api/generated.ts`). Example shape:
```
cd web; pnpm run gen:api
```
- [ ] 3. Verify the generated type now carries `fa_suggestions` and the `FaSuggestion` schema:
```
python -c "import pathlib,re; t=pathlib.Path('web/src/lib/api/generated.ts').read_text(encoding='utf-8'); print('FaSuggestion' in t, 'fa_suggestions' in t, 'value_breakdown' in t)"
```
Expect `True True True`.
- [ ] 4. Type-check passes (no consumers changed yet, so this should be clean). From `web/`:
```
cd web; pnpm exec tsc --noEmit
```
- [ ] 5. Commit:
```
git add web/src/lib/api/generated.ts && git commit -m "chore(web): regenerate API types for fa_suggestions/FaSuggestion (WS1)"
```

---

## Task 9: Frontend data layer — scope + FA suggestions + grouped layout types

**Files:**
- Modify: `web/src/lib/optimizer-data.ts` (types lines 15-62; `fetchOptimizer` lines 126-138)
- Modify: `web/src/lib/api/adapters.ts` (`apiOptimizeToData` lines 587-635)

- [ ] 1. Add the new frontend types + scope-aware fetcher to `web/src/lib/optimizer-data.ts`. After the `CatImpact` interface (line 33), add:
```typescript
export type OptimizerScope = "today" | "rest_of_week" | "rest_of_season";

export interface FaPickup {
  add: PlayerRef;
  drop: PlayerRef;
  netSgpDelta: number;
  categoryImpact: { label: string; value: string }[];
  reasoning: string;
  urgencyCategories: string[];
}
```
Add `faSuggestions: FaPickup[]` to the `OptimizerData` interface (after `daily?` at line 61):
```typescript
  faSuggestions: FaPickup[]; // available "drop X for Y" pickups (composed from recommend_fa_moves)
```
Replace the `fetchOptimizer` function (lines 126-138) with a scope-parameterized version that maps the UI scope to the `scope` field (Today/Week/Season all hit the same endpoint; the backend picks daily-vs-standard):
```typescript
/** Live: POST /api/lineup/optimize at the chosen scope → adapt. scope="today" runs the
 *  daily DCV start/sit path on the backend; rest_of_week/rest_of_season run the standard LP.
 *  Live errors propagate (HIGH-3) so usePageData reaches error/locked(402)/unlinked(409); an
 *  empty lineup → null → page `empty`. Off-live → the in-memory OPTIMIZER mock. */
export async function fetchOptimizer(scope: OptimizerScope = "today"): Promise<OptimizerData | null> {
  return liveOrMock(
    async () => {
      const api = await apiPost<ApiLineupOptimizeResponse>("/lineup/optimize", {
        team_name: getViewerTeam(),
        scope,
      });
      const data = apiOptimizeToData(api);
      return data.starters.length > 0 || data.faSuggestions.length > 0 ? data : null;
    },
    () => new Promise<OptimizerData>((resolve) => setTimeout(() => resolve(OPTIMIZER), 400)),
  );
}
```
Add `faSuggestions: []` to the `OPTIMIZER` mock object (after the `daily:` block, before the closing `}` at line 120):
```typescript
  faSuggestions: [],
```
- [ ] 2. Map `fa_suggestions` in the adapter. In `web/src/lib/api/adapters.ts`, inside `apiOptimizeToData` (lines 587-635), add before the `return {` (line 620):
```typescript
  const faSuggestions = (api.fa_suggestions ?? []).map((f) => ({
    add: toPlayerRef(f.add),
    drop: toPlayerRef(f.drop),
    netSgpDelta: f.net_sgp_delta ?? 0,
    categoryImpact: (f.category_impact ?? []).map((s) => ({ label: s.label, value: s.value })),
    reasoning: f.reasoning ?? "",
    urgencyCategories: f.urgency_categories ?? [],
  }));
```
Add `faSuggestions,` to the returned object (after `daily: hasDaily ? daily : undefined,`, line 633).
- [ ] 3. Type-check from `web/` — expect PASS (the generated `ApiLineupOptimizeResponse` now has `fa_suggestions`):
```
cd web; pnpm exec tsc --noEmit
```
- [ ] 4. Lint from `web/`:
```
cd web; pnpm run lint
```
- [ ] 5. Commit:
```
git add web/src/lib/optimizer-data.ts web/src/lib/api/adapters.ts && git commit -m "feat(web): optimizer scope param + faSuggestions in data layer + adapter (WS1)"
```

---

## Task 10: Yahoo-slot grouped layout in `LineupTable`

**Files:**
- Modify: `web/src/components/optimizer/LineupTable.tsx` (whole file, currently lines 1-103)

- [ ] 1. Rewrite `LineupTable.tsx` to render the union of starters+bench grouped by `currentSlot` in Yahoo order, split into Batters / Pitchers sections, decision-colored rows, columns Slot · Player · Eligibility · Value · Matchup · Decision. Replace the file contents:
```tsx
"use client";

import type { LineupSlot, SlotStatus } from "@/lib/optimizer-data";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { cn } from "@/lib/utils";

// Yahoo display order. A player is placed by its CURRENT Yahoo slot (currentSlot);
// when that is unknown (no live Yahoo) we fall back to the recommended slot.
const BATTER_ORDER = ["C", "1B", "2B", "3B", "SS", "OF", "Util", "BN", "IL"];
const PITCHER_ORDER = ["SP", "RP", "P", "BN", "IL"];
const PITCHER_SLOTS = new Set(["SP", "RP", "P"]);
const IL_SLOTS = new Set(["IL", "IL10", "IL15", "IL60", "NA", "DTD"]);

const STATUS: Record<SlotStatus, { label: string; cls: string }> = {
  start: { label: "Start", cls: "bg-ok/12 text-ok" },
  sit: { label: "Bench", cls: "bg-steel/12 text-steel" },
  bench: { label: "Bench", cls: "bg-steel/12 text-steel" },
  off: { label: "IL/Off", cls: "bg-surface-2 text-ink-3" },
};

function normSlot(raw: string): string {
  const s = (raw || "").toUpperCase().trim();
  if (IL_SLOTS.has(s)) return "IL";
  return s || "BN";
}

function slotKey(slot: LineupSlot): string {
  // group by the player's CURRENT Yahoo slot; fall back to the recommended slot
  return normSlot(slot.currentSlot || slot.slot);
}

function isPitcher(slot: LineupSlot): boolean {
  const key = slotKey(slot);
  if (PITCHER_SLOTS.has(key)) return true;
  // BN/IL pitchers: classify by the player's positions
  const pos = (slot.player.pos || "").toUpperCase();
  return /\b(SP|RP|P)\b/.test(pos);
}

function orderIndex(order: string[], key: string): number {
  const i = order.indexOf(key);
  return i === -1 ? order.length : i;
}

function groupAndSort(slots: LineupSlot[], order: string[]): LineupSlot[] {
  return [...slots].sort((a, b) => {
    const ia = orderIndex(order, slotKey(a));
    const ib = orderIndex(order, slotKey(b));
    if (ia !== ib) return ia - ib;
    return (b.value ?? 0) - (a.value ?? 0); // within a slot, best value first
  });
}

/** Render the full roster (starters ∪ bench) grouped by Yahoo slot, batters/pitchers split. */
export function LineupTable({ slots }: { slots: LineupSlot[] }) {
  const batters = groupAndSort(slots.filter((s) => !isPitcher(s)), BATTER_ORDER);
  const pitchers = groupAndSort(slots.filter(isPitcher), PITCHER_ORDER);
  return (
    <div className="space-y-5">
      {batters.length > 0 && <SlotSection title="Batters" rows={batters} />}
      {pitchers.length > 0 && <SlotSection title="Pitchers" rows={pitchers} />}
    </div>
  );
}

function SlotSection({ title, rows }: { title: string; rows: LineupSlot[] }) {
  return (
    <div>
      <div className="mb-2 text-[11px] font-bold uppercase tracking-wide text-ink-3">{title}</div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px]">
          <thead>
            <tr className="border-b border-line">
              <Th left>Slot</Th>
              <Th left>Player</Th>
              <Th left>Eligibility</Th>
              <Th>Value</Th>
              <Th left>Matchup</Th>
              <Th>Decision</Th>
            </tr>
          </thead>
          <tbody className="text-[13px]">
            {rows.map((s, i) => (
              <Row key={i} s={s} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function rowTone(s: LineupSlot): string {
  if (s.status === "off") return "bg-surface-2/40"; // IL / off → gray
  if (s.forcedStart) return "bg-heat/8"; // forced start → orange tint
  if (s.status === "start") return "bg-ok/6"; // start → green tint
  return "bg-steel/6"; // bench → blue/steel tint
}

function Row({ s }: { s: LineupSlot }) {
  const st = s.forcedStart ? { label: "Start", cls: "bg-heat/12 text-heat" } : STATUS[s.status];
  const swapHint = s.currentSlot && s.status === "start" && normSlot(s.currentSlot) !== normSlot(s.slot);
  return (
    <tr
      className={cn(
        "border-b border-line/60 transition-colors duration-[var(--dur-1)] hover:bg-surface/60",
        rowTone(s),
      )}
    >
      <td className="tnum px-2.5 py-2.5 font-bold text-navy">{slotKey(s)}</td>
      <td className="p-0">
        <PlayerDialog player={s.player}>
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded px-2.5 py-2.5 text-left transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
          >
            <PlayerAvatar mlbId={s.player.mlbId} teamId={s.player.teamId} name={s.player.name} size={26} />
            <span className="min-w-0">
              <span className="block text-[13px] font-semibold text-navy">{s.player.name}</span>
              <span className="tnum block text-[10.5px] text-ink-3">{s.player.teamAbbr ?? ""}</span>
            </span>
          </button>
        </PlayerDialog>
      </td>
      <td className="tnum px-2.5 py-2.5 text-ink-2">{s.player.pos}</td>
      <td className="px-2.5 py-2.5">
        <ValueBar value={s.value} />
      </td>
      <td className="tnum px-2.5 py-2.5 text-ink-2">{s.matchup || "—"}</td>
      <td className="px-2.5 py-2.5 text-right">
        <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold", st.cls)}>{st.label}</span>
        {swapHint && (
          <div className="mt-0.5 text-[10px] font-semibold text-heat">→ {normSlot(s.slot)}</div>
        )}
      </td>
    </tr>
  );
}

function Th({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy",
        left ? "text-left" : "text-right",
      )}
    >
      {children}
    </th>
  );
}

function ValueBar({ value }: { value: number }) {
  return (
    <div className="flex items-center justify-end gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
        <span className="block h-full rounded-full bg-heat" style={{ width: `${value}%` }} />
      </div>
      <span className="tnum w-6 text-right text-[12px] font-semibold text-navy">{value}</span>
    </div>
  );
}
```
- [ ] 2. Type-check from `web/`:
```
cd web; pnpm exec tsc --noEmit
```
Expect a type error in `web/src/app/optimizer/page.tsx` (it currently renders TWO `<LineupTable>` — one for starters, one for bench — and references `data.swaps`/`applySwaps`; the new grouped table takes the union). That error is fixed in Task 11. If `tsc` only flags page.tsx, proceed.
- [ ] 3. Lint the new component file in isolation (the page is updated next):
```
cd web; pnpm exec eslint src/components/optimizer/LineupTable.tsx
```
- [ ] 4. (No commit yet — the page consumes this; commit together in Task 11 to keep the build green.)
- [ ] 5. Proceed to Task 11.

---

## Task 11: Optimizer page — scope selector + single grouped table + pickups panel

**Files:**
- Modify: `web/src/app/optimizer/page.tsx` (whole file, currently lines 1-484)

- [ ] 1. Update the page to (a) add a Today/Rest of Week/Rest of Season selector that re-fetches via a `useCallback`-stable fetcher keyed on scope, (b) render ONE grouped `<LineupTable>` over starters∪bench, (c) render a "Recommended pickups" panel from `data.faSuggestions`. Make these edits to `web/src/app/optimizer/page.tsx`:

  (a) Replace the imports block (lines 1-27) to drop the unused swap icons and add the pieces used below:
```tsx
"use client";

import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Wand2, TrendingUp, TrendingDown, Minus, Clock, ArrowRight } from "lucide-react";
import {
  fetchOptimizer,
  type OptimizerData,
  type CatImpact,
  type FaPickup,
  type OptimizerScope,
} from "@/lib/optimizer-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { HeroNum } from "@/components/ui/HeroNum";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { LineupTable } from "@/components/optimizer/LineupTable";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty, PageLocked, PageNotLinked } from "@/components/ui/PageStates";
import { cn } from "@/lib/utils";

const SCOPES: { id: OptimizerScope; label: string }[] = [
  { id: "today", label: "Today" },
  { id: "rest_of_week", label: "Rest of Week" },
  { id: "rest_of_season", label: "Rest of Season" },
];
```

  (b) Delete the `applySwaps` helper (lines 29-51) entirely — the grouped table no longer simulates swaps.

  (c) Replace the `OptimizerPage` component (lines 53-75) to own the scope state and pass a stable scoped fetcher:
```tsx
export default function OptimizerPage() {
  const [scope, setScope] = useState<OptimizerScope>("today");
  // Stable per-scope fetcher: identity changes with scope → usePageData refetches.
  const fetcher = useCallback(() => fetchOptimizer(scope), [scope]);
  const { state, retry } = usePageData(fetcher);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        <div className="mb-5">
          <ScopeSelector scope={scope} onChange={setScope} />
        </div>
        {state.status === "loading" && <LoadingView />}
        {state.status === "locked" && <PageLocked feature="The Optimizer" />}
        {state.status === "unlinked" && <PageNotLinked />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty icon={Wand2} title="No lineup to optimize" body="We couldn't find your roster for this window." />
        )}
        {state.status === "loaded" && <Loaded data={state.data} scope={scope} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}

function ScopeSelector({ scope, onChange }: { scope: OptimizerScope; onChange: (s: OptimizerScope) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface p-1" role="tablist" aria-label="Optimize horizon">
      {SCOPES.map((s) => (
        <button
          key={s.id}
          role="tab"
          aria-selected={scope === s.id}
          onClick={() => onChange(s.id)}
          className={cn(
            "min-h-9 rounded-lg px-3.5 text-[12px] font-bold transition-colors",
            scope === s.id ? "bg-navy text-white" : "text-ink-2 hover:text-navy",
          )}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}
```

  (d) Replace the `Loaded` component (lines 77-116) to render ONE grouped table over starters∪bench plus the pickups panel:
```tsx
function Loaded({ data, scope }: { data: OptimizerData; scope: OptimizerScope }) {
  const all = [...data.starters, ...data.bench];
  const scopeLabel = SCOPES.find((s) => s.id === scope)?.label ?? "Today";
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header date={data.date} optimal={data.optimal} scopeLabel={scopeLabel} />
      </motion.div>
      <motion.div variants={staggerItem} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <Card className="p-5">
          <SectionHead title="Your Roster" sub={scopeLabel} />
          <LineupTable slots={all} />
        </Card>
        <aside className="space-y-4">
          {data.faSuggestions.length > 0 && <PickupsCard pickups={data.faSuggestions} />}
          {data.daily && <DailyPlanCard daily={data.daily} />}
          {data.ipPace && <PaceCard ipPace={data.ipPace} />}
          {data.impact.length > 0 && <ImpactCard impact={data.impact} />}
        </aside>
      </motion.div>
    </motion.div>
  );
}
```

  (e) Replace `Header` (lines 118-154) with a swap-button-free version (the grouped table is read-only; Yahoo writes are unavailable):
```tsx
function Header({ date, optimal, scopeLabel }: { date: string; optimal: boolean; scopeLabel: string }) {
  const subline = optimal
    ? "Your lineup is already optimal for this window."
    : "Slots flagged with → can improve your projection.";
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Lineup · {scopeLabel} · {date}
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Optimizer</h1>
        <p className="mt-1 text-[13px] text-ink-2">{subline}</p>
      </div>
    </div>
  );
}
```

  (f) Delete the now-unused `SuccessBanner` (lines 156-164), `SwapCard` (lines 177-225), `SwapRow` (lines 307-360), and the `Slot` type alias (line 175). Add a `PickupsCard` (place near `DailyPlanCard`):
```tsx
function PickupsCard({ pickups }: { pickups: FaPickup[] }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <ArrowRight className="size-4 text-heat" aria-hidden />
        Recommended Pickups
      </div>
      <ul className="space-y-3">
        {pickups.map((p, i) => (
          <li key={i} className="rounded-lg border border-line bg-surface p-2.5">
            <div className="flex items-center justify-between gap-2">
              <PlayerDialog player={p.add}>
                <button className="flex min-w-0 items-center gap-2 text-left">
                  <PlayerAvatar mlbId={p.add.mlbId} teamId={p.add.teamId} name={p.add.name} size={24} />
                  <span className="min-w-0">
                    <span className="block text-[9px] font-bold uppercase tracking-wide text-ok">Add</span>
                    <span className="block truncate text-[13px] font-semibold text-navy">{p.add.name}</span>
                  </span>
                </button>
              </PlayerDialog>
              {p.netSgpDelta > 0 && (
                <span className="tnum shrink-0 rounded-md bg-heat/12 px-2 py-0.5 text-[11px] font-bold text-heat">
                  +{p.netSgpDelta.toFixed(2)} SGP
                </span>
              )}
            </div>
            <div className="mt-1.5 flex items-center gap-2 pl-1 text-[11px] text-ink-2">
              <span className="font-bold uppercase tracking-wide text-ember">Drop</span>
              <span className="truncate font-semibold text-navy">{p.drop.name}</span>
            </div>
            {p.categoryImpact.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {p.categoryImpact.map((c) => (
                  <span key={c.label} className="tnum rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-ink-2">
                    {c.label} {c.value}
                  </span>
                ))}
              </div>
            )}
            {p.reasoning && <p className="mt-1.5 text-[11px] leading-snug text-ink-3">{p.reasoning}</p>}
          </li>
        ))}
      </ul>
    </Card>
  );
}
```

  (g) Update `PaceCard` (lines 362-397) to drop the `movesLeft` prop (not in the optimize contract; the old code already passed `undefined`). Change its signature + body to take only `ipPace`:
```tsx
function PaceCard({ ipPace }: { ipPace: { value: number; total: number } }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Clock className="size-4 text-heat" aria-hidden />
        This Week
      </div>
      <Meter
        label="IP pace"
        value={ipPace.value}
        total={ipPace.total}
        unit="IP"
        caption={`On pace toward the ${ipPace.total} IP minimum.`}
      />
    </Card>
  );
}
```
Leave `DailyPlanCard`, `StatePill`, `Meter`, `ImpactCard`, `ImpactCell`, `LoadingView`, `RATE_MODE_TONE`, `SectionHead` as-is (remove any now-unused imports they referenced — e.g. `Check`, `ArrowDown`, `Repeat`, `Target` are no longer used; the import block in (a) already omits them).
- [ ] 2. Type-check from `web/` — expect PASS:
```
cd web; pnpm exec tsc --noEmit
```
- [ ] 3. Lint from `web/` — expect PASS (no unused vars/imports):
```
cd web; pnpm run lint
```
- [ ] 4. Production build (the real gate — `web/` has no CI):
```
cd web; pnpm build
```
Expect a successful build.
- [ ] 5. Commit:
```
git add web/src/components/optimizer/LineupTable.tsx web/src/app/optimizer/page.tsx && git commit -m "feat(web): optimizer scope selector + Yahoo-slot grouped layout + pickups panel (WS1)"
```

---

## Task 12: Full-suite verification + finish

**Files:** none (verification only)

- [ ] 1. Run the full API test suite to confirm the backend slice is green end-to-end:
```
python -m pytest tests/api/ -q
```
- [ ] 2. Run the lineup engine-seam + contract tests once more focused:
```
python -m pytest tests/api/test_lineup.py tests/api/test_api_lineup_fa.py tests/api/test_openapi_contract.py -v
```
- [ ] 3. Frontend gate trio from `web/`:
```
cd web; pnpm exec tsc --noEmit; pnpm run lint; pnpm build
```
- [ ] 4. Confirm the openapi snapshot + structural router guards are green (regression safety):
```
python -m pytest tests/api/test_openapi_contract.py tests/api/test_no_logic_in_routers.py tests/api/test_api_pro_gating.py -v
```
- [ ] 5. Final commit if any stray changes remain (otherwise skip):
```
git add -A && git commit -m "chore(ws1): final verification pass — optimizer scope + FA-aware + Yahoo-slot layout" || echo "nothing to commit"
```

---

## Notes for the integrator

- **Shared files** with other workstreams: `api/main.py` (route mounts — unchanged here, the lineup route is already mounted), `api/openapi.json`, `web/src/lib/api/generated.ts`, `web/src/lib/api/adapters.ts`. Per the parallel-reconcile lesson, resolve `openapi.json` and `generated.ts` conflicts by **regenerating** (`python scripts/export_openapi.py` then the web codegen), never by hand-merging. `adapters.ts` edits here are confined to the `apiOptimizeToData` function + its FA-mapping block — disjoint from WS2/WS4 adapter additions.
- **Live-only degradation (documented, not a bug):** `current_slot`, `matchup`, and FA `category_impact`/`net_sgp_delta` only fully populate with live Yahoo data on Railway. In the empty local/worktree DB, `_compose_fa` returns `[]` (no roster → `recommend_fa_moves` short-circuits) and the standard slots carry empty `current_slot`/`matchup` — the page renders gracefully (grouped by recommended slot, "—" matchups, no pickups panel). Verify full population against the live API per the live-verification local-stack memory.
- **`mode` is now service-internal back-compat.** The frontend sends only `scope`; an explicit `mode="daily"` still forces the daily path so existing callers (and the `test_post_lineup_optimize_daily_returns_daily_meta` test) keep working.
- **No `src/` engine changes** in this WS — `lineup_service.py` is the only file importing `src/`, and it composes the UNCHANGED `build_optimizer_context` / `recommend_fa_moves` / pipeline. WS3 builds on this stabilized service afterward (do not edit `lineup_service.py` in parallel).
