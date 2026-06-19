# M0 Slice 2 — PlayerRef enrichment for the remaining 5 sites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the optional `PlayerRef.mlb_id` / `team_abbr` / `team_id` fields (added in M0 slice 1) in the 5 service sites that were deferred: compare, draft, leaders (data in hand) and lineup, fa (need a pool lookup). After this, all player-bearing endpoints carry headshot + team-logo data.

**Architecture:** Two shapes. **At-site** (compare, draft, leaders) already have `mlb_id`+`team` in scope (a pool row, the draft engine's result row which preserves pool columns, or — for leaders — after adding `p.mlb_id, p.team` to its SELECT) → call the existing `make_player_ref(...)`. **Pool-lookup** (lineup, fa) only have `player_id` → add ONE shared helper `player_ref_from_pool(player_id, pool, ...)` to `api/services/player_ref.py` and pass the pool in. The new `pool` parameters are OPTIONAL (default `None`) so existing call sites/tests are unaffected; `None`/empty pool → no enrichment (name/positions preserved). fa also fixes a pre-existing bug (it reads the wrong move-dict id key, so its `PlayerRef.id` is always 0).

**Tech Stack:** FastAPI + Pydantic contracts; pandas; pytest. `src/` is NOT modified. Routers are NOT touched. The `PlayerRef` schema does NOT change (slice 1 already added the fields), so `api/openapi.json` is NOT regenerated.

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = the worktree root.

---

## File Structure

- `api/services/player_ref.py` — **modify**: add `player_ref_from_pool(...)`.
- `api/services/compare_service.py` — **modify**: enrich the inline ref (`compare()` ~line 54).
- `api/services/draft_service.py` — **modify**: enrich `_to_recs` (~line 142; result rows carry mlb_id+team).
- `api/services/leaders_service.py` — **modify**: add `p.mlb_id, p.team` to the SELECT; extract + enrich a `_to_leader_row` helper.
- `api/services/lineup_service.py` — **modify**: load the pool in `optimize()`, thread it through `_to_slots(result, pool=None)` via the lookup helper.
- `api/services/fa_service.py` — **modify**: fix the id key (`add_id`/`drop_id`), thread `ctx.player_pool` through `_to_rec(move, pool=None)` via the lookup helper.
- `tests/api/test_api_player_ref_pool_lookup.py` — **create**: unit tests for the new helper.
- `tests/api/test_api_player_ref_enrichment.py` — **append**: per-site enrichment tests for the 5 sites.

---

### Task 1: Add the `player_ref_from_pool` helper

**Files:**
- Modify: `api/services/player_ref.py`
- Test: `tests/api/test_api_player_ref_pool_lookup.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_api_player_ref_pool_lookup.py`:

```python
"""Unit tests for player_ref_from_pool (M0 slice 2)."""

from __future__ import annotations

import pandas as pd

from api.services.player_ref import player_ref_from_pool


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": 1, "name": "Aaron Judge", "positions": "OF", "mlb_id": 592450, "team": "NYY"},
            {"player_id": 2, "name": "NoIds", "positions": "SP", "mlb_id": None, "team": None},
        ]
    )


def test_found_enriches_and_prefers_provided_name_positions():
    ref = player_ref_from_pool(1, _pool(), name="Judge (engine)", positions="RF")
    assert ref.id == 1
    assert ref.mlb_id == 592450
    assert ref.team_abbr == "NYY"
    assert ref.team_id == 147
    assert ref.name == "Judge (engine)"  # provided name preferred over pool's
    assert ref.positions == "RF"  # provided positions preferred


def test_found_falls_back_to_pool_name_positions_when_not_provided():
    ref = player_ref_from_pool(1, _pool())
    assert ref.name == "Aaron Judge"
    assert ref.positions == "OF"
    assert ref.mlb_id == 592450


def test_found_row_with_null_ids_degrades_cleanly():
    ref = player_ref_from_pool(2, _pool(), name="NoIds", positions="SP")
    assert ref.mlb_id is None
    assert ref.team_abbr is None
    assert ref.team_id is None


def test_not_found_uses_provided_then_placeholder():
    ref = player_ref_from_pool(999, _pool(), name="Ghost", positions="OF")
    assert ref.id == 999
    assert ref.name == "Ghost"
    assert ref.mlb_id is None
    ref2 = player_ref_from_pool(999, _pool())
    assert ref2.name == "Player 999"  # placeholder when nothing provided


def test_none_or_empty_pool_does_not_raise():
    assert player_ref_from_pool(1, None, name="X", positions="OF").mlb_id is None
    assert player_ref_from_pool(1, pd.DataFrame(), name="X", positions="OF").team_abbr is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_pool_lookup.py -q`
Expected: FAIL — `ImportError: cannot import name 'player_ref_from_pool'`.

- [ ] **Step 3: Implement the helper**

Append to `api/services/player_ref.py` (after `make_player_ref`):

```python
def player_ref_from_pool(
    player_id,
    pool,
    *,
    name=None,
    positions=None,
    yahoo_player_key: str | None = None,
) -> PlayerRef:
    """Build an enriched PlayerRef by looking up player_id in the player pool.

    Pulls mlb_id + team (→ team_id) from the matching pool row. PREFERS the
    caller-provided name/positions (the engine's view) and only falls back to
    the pool's when they're absent, then to a "Player {id}" placeholder. A
    missing/None/empty pool or a not-found id degrades to a valid un-enriched
    ref (never raises).
    """
    try:
        pid = int(player_id) if player_id is not None else 0
    except (TypeError, ValueError):
        pid = 0
    mlb_id = None
    team_abbr = None
    final_name = name
    final_pos = positions
    try:
        import pandas as pd

        if isinstance(pool, pd.DataFrame) and not pool.empty:
            match = pool[pool["player_id"] == pid]
            if not match.empty:
                r = match.iloc[0]
                mlb_id = r.get("mlb_id")
                team_abbr = r.get("team")
                if not final_name:
                    name_col = "player_name" if "player_name" in pool.columns else "name"
                    final_name = r.get(name_col)
                if not final_pos:
                    final_pos = r.get("positions")
    except Exception:
        pass
    return make_player_ref(
        id=pid,
        name=str(final_name) if final_name else f"Player {pid}",
        positions=str(final_pos) if final_pos else "",
        mlb_id=mlb_id,
        team_abbr=team_abbr,
        yahoo_player_key=yahoo_player_key,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_pool_lookup.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add api/services/player_ref.py tests/api/test_api_player_ref_pool_lookup.py
git commit -m "feat(api): add player_ref_from_pool lookup helper (mlb_id/team by player_id)"
```

---

### Task 2: At-site enrichment — compare + draft

Both have mlb_id+team in scope: compare reads pool row `r`; draft's engine `results` rows preserve the pool's columns.

**Files:**
- Modify: `api/services/compare_service.py` (the `PlayerRef(...)` at ~line 54)
- Modify: `api/services/draft_service.py` (the `PlayerRef(...)` in `_to_recs` at ~line 142)
- Test: append to `tests/api/test_api_player_ref_enrichment.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_api_player_ref_enrichment.py`:

```python
def test_draft_to_recs_enriches_from_engine_results():
    import pandas as pd

    from api.services.draft_service import DraftService

    results = pd.DataFrame(
        [
            {
                "player_id": 11,
                "player_name": "Corbin Carroll",
                "positions": "OF",
                "mlb_id": 682998,
                "team": "ARI",
                "overall_rank": 1,
                "composite_value": 95.0,
                "mc_mean_sgp": 3.2,
                "confidence_level": "high",
                "buy_fair_avoid": "buy",
            }
        ]
    )
    recs = DraftService._to_recs(results)
    assert recs[0].player.mlb_id == 682998
    assert recs[0].player.team_abbr == "ARI"
    assert recs[0].player.team_id == 109


def test_compare_enriches_from_pool(monkeypatch):
    import pandas as pd

    import src.database as db
    from api.services.compare_service import CompareService

    fake_pool = pd.DataFrame(
        [{"player_id": 3, "name": "Mookie Betts", "positions": "OF", "mlb_id": 605141, "team": "LAD",
          "r": 90, "hr": 20, "rbi": 70, "sb": 10, "avg": 0.300, "obp": 0.380,
          "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0}]
    )
    monkeypatch.setattr(db, "load_player_pool", lambda: fake_pool)
    resp = CompareService().compare([3])
    assert resp.players[0].player.mlb_id == 605141
    assert resp.players[0].player.team_abbr == "LAD"
    assert resp.players[0].player.team_id == 119
```

NOTE: `compare()` imports `load_player_pool` lazily via `from src.database import load_player_pool`, which resolves the attribute on the `src.database` module at call time — so patching `src.database.load_player_pool` works. Confirm the `DraftService` class name + that `_to_recs` is a staticmethod (it is). The `team_id` expectations: ARI=109, LAD=119 (from `_STATSAPI_TEAM_IDS`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: FAIL — draft/compare refs have `mlb_id is None`.

- [ ] **Step 3: Edit compare_service.py**

Add the import `from api.services.player_ref import make_player_ref` (with the other imports), and replace the `ref = PlayerRef(...)` block (the one inside the `for pid` loop, after `r = row.iloc[0]`) with:

```python
                ref = make_player_ref(
                    id=pid,
                    name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                    positions=str(r.get("positions", "") or ""),
                    mlb_id=r.get("mlb_id"),
                    team_abbr=r.get("team"),
                )
```

(Leave the unused `from api.contracts.common import PlayerRef` import only if it's still referenced elsewhere in the file; if it becomes unused, remove it to satisfy ruff.)

- [ ] **Step 4: Edit draft_service.py**

Add `from api.services.player_ref import make_player_ref` (with the other imports), and replace the `player=PlayerRef(...)` inside `_to_recs` with:

```python
                    player=make_player_ref(
                        id=pid,
                        name=str(g("player_name") or g("name") or ""),
                        positions=str(g("positions") or ""),
                        mlb_id=g("mlb_id"),
                        team_abbr=g("team"),
                    ),
```

(Remove the now-unused `PlayerRef` import if nothing else in the file uses it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: PASS (the two new tests + all prior).

- [ ] **Step 6: Commit**

```bash
git add api/services/compare_service.py api/services/draft_service.py tests/api/test_api_player_ref_enrichment.py
git commit -m "feat(api): enrich compare + draft PlayerRefs (data already in scope)"
```

---

### Task 3: leaders — SQL add + `_to_leader_row` extraction + enrich

**Files:**
- Modify: `api/services/leaders_service.py` (SELECT ~lines 41-50; the `ldf.iterrows()` mapping ~lines 78-93)
- Test: append to `tests/api/test_api_player_ref_enrichment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_api_player_ref_enrichment.py`:

```python
def test_leaders_to_leader_row_enriches():
    from api.services.leaders_service import LeadersService

    row = {"player_id": 21, "name": "Bobby Witt Jr.", "positions": "SS", "mlb_id": 677951, "team": "KC", "hr": 24}
    leader_row = LeadersService._to_leader_row(1, row, "hr")
    assert leader_row.rank == 1
    assert leader_row.value == 24.0
    assert leader_row.player.mlb_id == 677951
    assert leader_row.player.team_abbr == "KC"
    assert leader_row.player.team_id == 118
```

(KC=118. Confirm the `LeadersService` class name; `_to_leader_row` will be a new `@staticmethod`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py::test_leaders_to_leader_row_enriches -q`
Expected: FAIL — `AttributeError: ... has no attribute '_to_leader_row'`.

- [ ] **Step 3: Add the SQL columns**

In `leaders_service.py`, in the `SELECT` block, add `p.mlb_id,` and `p.team,` immediately after the `p.positions,` line:

```python
                        s.player_id,
                        p.name,
                        p.positions,
                        p.mlb_id,
                        p.team,
                        p.is_hitter,
```

- [ ] **Step 4: Extract + enrich `_to_leader_row`**

Add `from api.services.player_ref import make_player_ref` to the imports. Add this staticmethod to `LeadersService`:

```python
    @staticmethod
    def _to_leader_row(rank: int, row, stat_col: str) -> LeaderRow:
        g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        raw_val = g(stat_col)
        try:
            value = float(raw_val) if raw_val is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        return LeaderRow(
            rank=rank,
            player=make_player_ref(
                id=int(g("player_id", 0) or 0),
                name=str(g("name", "") or ""),
                positions=str(g("positions", "") or ""),
                mlb_id=g("mlb_id"),
                team_abbr=g("team"),
            ),
            value=value,
        )
```

Then replace the inline `for rank, (_, row) in enumerate(ldf.iterrows(), start=1): ... rows.append(LeaderRow(...))` body so it delegates:

```python
            for rank, (_, row) in enumerate(ldf.iterrows(), start=1):
                rows.append(self._to_leader_row(rank, row, stat_col))
```

(Keep the surrounding `stat_col = _STAT_COL_MAP.get(...)` line. Remove the now-unused inline `player_id`/`name`/`positions`/`raw_val`/`value` locals and the direct `PlayerRef` import if unused.)

- [ ] **Step 5: Run the test + the leaders contract test**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py::test_leaders_to_leader_row_enriches tests/api/test_api_leaders.py -q`
Expected: PASS (new test + the existing leaders endpoint test still green).

- [ ] **Step 6: Commit**

```bash
git add api/services/leaders_service.py tests/api/test_api_player_ref_enrichment.py
git commit -m "feat(api): enrich leaders PlayerRefs (add mlb_id/team to SELECT + _to_leader_row)"
```

---

### Task 4: Pool-lookup enrichment — lineup + fa (fa also fixes the id-key bug)

**Files:**
- Modify: `api/services/lineup_service.py` (`optimize()` + `_to_slots`)
- Modify: `api/services/fa_service.py` (`get_free_agents()` + `_to_rec`)
- Test: append to `tests/api/test_api_player_ref_enrichment.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_api_player_ref_enrichment.py`:

```python
def test_lineup_to_slots_enriches_via_pool():
    import pandas as pd

    from api.services.lineup_service import LineupService

    pool = pd.DataFrame([{"player_id": 31, "name": "X", "positions": "OF", "mlb_id": 660271, "team": "LAA"}])
    result = {"lineup": [{"slot": "OF", "player_id": 31, "player_name": "Shohei Ohtani", "positions": "OF", "action": "START"}]}
    slots = LineupService._to_slots(result, pool)
    assert slots[0].player.mlb_id == 660271
    assert slots[0].player.team_abbr == "LAA"
    assert slots[0].player.team_id == 108
    assert slots[0].player.name == "Shohei Ohtani"  # engine name preferred


def test_lineup_to_slots_without_pool_still_works():
    from api.services.lineup_service import LineupService

    result = {"lineup": [{"slot": "OF", "player_id": 31, "player_name": "X", "positions": "OF", "action": "START"}]}
    slots = LineupService._to_slots(result)  # pool defaults to None
    assert slots[0].player.mlb_id is None
    assert slots[0].player.id == 31


def test_fa_to_rec_enriches_and_fixes_id_key():
    import pandas as pd

    from api.services.fa_service import FaService

    pool = pd.DataFrame(
        [
            {"player_id": 41, "name": "Add Guy", "positions": "2B", "mlb_id": 700000, "team": "NYM"},
            {"player_id": 42, "name": "Drop Guy", "positions": "SP", "mlb_id": 800000, "team": "SF"},
        ]
    )
    move = {
        "add_id": 41, "add_name": "Add Guy", "add_positions": "2B",
        "drop_id": 42, "drop_name": "Drop Guy", "drop_positions": "SP",
        "net_sgp_delta": 1.5,
    }
    rec = FaService._to_rec(move, pool)
    assert rec.add.id == 41  # was always 0 before the id-key fix
    assert rec.add.mlb_id == 700000
    assert rec.add.team_abbr == "NYM"
    assert rec.add.team_id == 121
    assert rec.drop is not None
    assert rec.drop.id == 42
    assert rec.drop.mlb_id == 800000
    assert rec.drop.team_id == 137
```

NOTE: confirm the service class names (`LineupService`, `FaService`) and the `FreeAgentRec` field names (`add` / `drop`) by reading the files; adjust references if they differ, keep assertions identical. NYM=121, SF=137, LAA=108.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: FAIL — lineup/fa refs not enriched; `_to_slots`/`_to_rec` don't accept a pool arg yet; fa `rec.add.id` is 0.

- [ ] **Step 3: Edit lineup_service.py**

Add `from api.services.player_ref import player_ref_from_pool` to imports. Change `_to_slots` to accept an optional pool and use the helper:

```python
    @staticmethod
    def _to_slots(result, pool=None) -> list[LineupSlot]:
        # `result` shape is the integration seam; map defensively. Return [] if absent.
        rows = []
        if result is None:
            return rows
        lineup = getattr(result, "lineup", None) or (result.get("lineup") if isinstance(result, dict) else None) or []
        for r in lineup:
            g = r.get if isinstance(r, dict) else lambda k, d=None: getattr(r, k, d)
            rows.append(
                LineupSlot(
                    slot=str(g("slot", "") or ""),
                    player=player_ref_from_pool(
                        g("player_id", 0),
                        pool,
                        name=g("player_name", ""),
                        positions=g("positions", ""),
                    ),
                    action="START" if g("action", "START") in ("START", "start", True) else "SIT",
                    projected=float(g("projected", 0.0) or 0.0),
                    forced_start=bool(g("forced_start", False)),
                    reason=g("reason"),
                )
            )
        return rows
```

In `optimize()`, load the pool defensively and pass it to `_to_slots`. Replace `slots = self._to_slots(result)` with:

```python
            pool = None
            try:
                from src.database import load_player_pool

                pool = load_player_pool()
            except Exception:
                pool = None
            slots = self._to_slots(result, pool)
```

(Remove the now-unused `from api.contracts.common import PlayerRef` import if nothing else uses it.)

- [ ] **Step 4: Edit fa_service.py**

Add `from api.services.player_ref import player_ref_from_pool` to imports. Change `_to_rec` to accept an optional pool, FIX the id keys (`add_id`/`drop_id`), and enrich. Replace the `add_ref`/`drop_ref` construction:

```python
    @staticmethod
    def _to_rec(move, pool=None) -> FreeAgentRec:
        g = move.get if isinstance(move, dict) else lambda k, d=None: getattr(move, k, d)
        add_ref = player_ref_from_pool(
            g("add_id", 0),
            pool,
            name=g("add_name", g("name", "")),
            positions=g("add_positions", ""),
        )
        drop_name = g("drop_name", None)
        drop_ref = None
        if drop_name:
            drop_ref = player_ref_from_pool(
                g("drop_id", 0),
                pool,
                name=drop_name,
                positions=g("drop_positions", ""),
            )
```

(Keep the rest of `_to_rec` — the `FreeAgentRec(...)` build using `add_ref`/`drop_ref` and the other move fields — unchanged. Remove the now-unused `PlayerRef` import if unused.)

In `get_free_agents()`, pass the context pool to `_to_rec`. Replace the loop body `recs.append(self._to_rec(move))` with:

```python
            for move in recommend_fa_moves(ctx, max_moves=limit) or []:
                recs.append(self._to_rec(move, getattr(ctx, "player_pool", None)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: PASS (the three new tests + all prior).

- [ ] **Step 6: Commit**

```bash
git add api/services/lineup_service.py api/services/fa_service.py tests/api/test_api_player_ref_enrichment.py
git commit -m "feat(api): enrich lineup + fa PlayerRefs via pool lookup; fix fa id-key bug (add_id/drop_id)"
```

---

### Task 5: Full api suite + openapi-unchanged check + ruff

**Files:** none (verification only).

- [ ] **Step 1: Confirm the OpenAPI snapshot is UNCHANGED**

M0.2 changes no contract — `PlayerRef` already had the fields. So `api/openapi.json` must NOT need regeneration.
Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS with NO regeneration. (If it FAILS, something changed a contract unexpectedly — stop and report; do NOT blindly regenerate.)

- [ ] **Step 2: Run the FULL api suite + fix any fallout**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS.

If a PRE-EXISTING fa test fails because it built a fake move with the OLD `add_player_id`/`drop_player_id` keys or asserted `id == 0`: update it to use `add_id`/`drop_id` and the correct id (the engine's real keys — this is the bug fix). If a test calls `_to_slots(result)` / `_to_rec(move)` with the old single-arg form, it should STILL PASS (pool defaults to None → no enrichment); if such a test asserted enrichment fields, that's not expected. Do NOT weaken assertions; only correct keys/expected ids that the bug fix legitimately changes. If any test fails for another reason, stop and report.

- [ ] **Step 3: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: both clean.

- [ ] **Step 4: Commit (only if Step 2 required test edits / Step 3 reformatted)**

```bash
git add -A
git commit -m "test(api): update fa tests for corrected id keys + ruff"
```

(If nothing changed in Steps 2-3, skip this commit.)

---

## Self-Review (completed by plan author)

**Spec coverage** (the 5 deferred sites from slice 1's scope note): compare ✔ (Task 2), draft ✔ (Task 2), leaders ✔ (Task 3), lineup ✔ (Task 4), fa ✔ (Task 4). The shared lookup helper ✔ (Task 1).

**Backward compatibility:** new `pool` params on `_to_slots`/`_to_rec` default to `None` → existing single-arg callers/tests unaffected (pool=None → no enrichment, identical output). The `PlayerRef` schema is unchanged → `api/openapi.json` not regenerated (Task 5 Step 1 guards this).

**Bug fix:** fa's `add_player_id`/`drop_player_id` reads (which never matched the engine's `add_id`/`drop_id` keys → id always 0) are corrected to `add_id`/`drop_id`. This is required for the pool lookup and fixes a real latent defect; called out in the commit + tests.

**Placeholder scan:** every code step shows full code; no TBD/"handle edge cases". ✔

**Type consistency:** `player_ref_from_pool(player_id, pool, *, name=None, positions=None, yahoo_player_key=None)` used identically in Tasks 3-4 as defined in Task 1; `make_player_ref(...)` (from slice 1) used in Task 2. Team-id expectations match `_STATSAPI_TEAM_IDS` (ARI=109, LAD=119, KC=118, NYM=121, SF=137, LAA=108). ✔
