# M0 — Widen the Streaming contract (Stream Finder board) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen `StreamCandidate` to expose the full board row the engine already computes, and add `top_pick` + `budget` to `StreamingResponse` — so the CMO's Pitcher Streaming page (Stream Finder board) can wire to live data. This is the CMO spec's "API contract handoff (for the CEO track)" items #1, #2 (board half), and the `top_pick`/`budget`.

**Architecture:** `build_stream_board` (`src/optimizer/stream_analyzer.py`) ALREADY returns every needed column — no engine change. The widening is entirely in the API layer: add `StreamComponents` + `BudgetStrip` models, widen `StreamCandidate`, extend `StreamingResponse`, and widen the service's `_to_candidate` mapping + compute `top_pick`/`budget`. All new fields have defaults (backward-compatible). NaN-safe float coercion (the FA-pool lesson) is applied to every engine float so a missing-data NaN never reaches the JSON response.

**Scope boundary:** This slice = the board (`StreamCandidate` widen + `top_pick` + `budget`). The `probables[]` list + `POST /api/streaming/analyze` ("Analyze Any Starter") are the NEXT slice. `ip_pace` in the budget is DEFERRED to `0.0` here (computing weekly IP pace needs roster-pitcher IP plumbing; documented — the board + adds + ip_target are the priority); the rest of the budget is real.

**Convention:** snake_case; `status`/`confidence` emit the engine's RAW values (e.g. "PROBABLE", "MEDIUM") — the frontend adapter maps them to its lowercase enums (as it does for every contract). Player identity reuses `make_player_ref` (the board row carries `mlb_id`).

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = worktree root.

**Tech Stack:** FastAPI + Pydantic v2; pandas; pytest. Routers logic-free (AST-guarded). `fastapi==0.137.1`/`httpx==0.28.1` pinned (openapi snapshot version-sensitive).

---

## File Structure

- `api/contracts/streaming.py` — **modify**: add `StreamComponents`, `BudgetStrip`; widen `StreamCandidate`; extend `StreamingResponse`.
- `api/services/streaming_service.py` — **modify**: add `_f` NaN-safe float; widen `_to_candidate(row, rank)`; add `_pick_top` + `_build_budget`; wire into `get_streaming`.
- `api/openapi.json` — **regenerate** (new models + widened fields).
- `tests/api/test_api_streaming_widen.py` — **create**: `_to_candidate`/`_pick_top`/`_build_budget` unit tests + the fake-service contract test.

---

### Task 1: Contracts + widened `_to_candidate` mapping

**Files:**
- Modify: `api/contracts/streaming.py`
- Modify: `api/services/streaming_service.py`
- Test: `tests/api/test_api_streaming_widen.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/api/test_api_streaming_widen.py`:

```python
"""Streaming contract widen: mapping + budget/top_pick unit tests + contract test."""

from __future__ import annotations

from api.services.streaming_service import StreamingService, _build_budget, _pick_top


def _board_row(**over):
    row = {
        "player_id": 660271, "mlb_id": 660271, "player_name": "Tarik Skubal", "team": "DET",
        "opponent": "CWS", "is_home": True, "stream_score": 82.5, "status": "PROBABLE",
        "confidence": "HIGH", "num_starts": 2, "actionable": True, "net_sgp": 1.8,
        "opp_wrc_plus": 88.0, "opp_k_pct": 24.5, "park_factor": 0.96,
        "expected_ip": 6.0, "expected_k": 7.0, "expected_er": 2.0,
        "win_probability": 0.62, "percent_owned": 99.0,
        "risk_flags": ["LOW_CONFIDENCE"],
        "components": {"matchup": 0.4, "env": 0.1, "form": 0.2, "lineup": -0.1, "sgp": 0.5, "winprob": 0.3},
    }
    row.update(over)
    return row


def test_to_candidate_maps_all_widened_fields():
    c = StreamingService._to_candidate(_board_row(), rank=1)
    assert c.rank == 1
    assert c.player.mlb_id == 660271
    assert c.player.team_id == 116  # DET
    assert c.is_home is True
    assert c.score == 82.5
    assert c.status == "PROBABLE"  # raw engine value (frontend adapter lowercases)
    assert c.confidence == "HIGH"
    assert c.num_starts == 2
    assert c.net_sgp == 1.8
    assert c.opp_wrc_plus == 88.0
    assert c.opp_k_pct == 24.5
    assert c.park == 0.96
    assert c.expected_ip == 6.0
    assert c.win_pct == 0.62
    assert c.own_pct == 99.0
    assert c.risk_flags == ["LOW_CONFIDENCE"]
    assert c.components.matchup == 0.4
    assert c.components.winprob == 0.3
    assert c.expected_line == "6.0 IP · 7 K · 2 ER"


def test_to_candidate_is_nan_safe():
    c = StreamingService._to_candidate(
        _board_row(opp_wrc_plus=float("nan"), percent_owned=float("nan"),
                   components={"matchup": float("nan")}),
        rank=3,
    )
    assert c.opp_wrc_plus == 0.0  # NaN → 0.0, never reaches JSON as NaN
    assert c.own_pct == 0.0
    assert c.components.matchup == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_widen.py -q`
Expected: FAIL — `ImportError: cannot import name '_build_budget'` / `_to_candidate` doesn't accept `rank` / fields missing.

- [ ] **Step 3: Widen the contracts**

Replace `api/contracts/streaming.py` entirely with:

```python
"""Contract models for the Pitcher Streaming page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class StreamComponents(BaseModel):
    """The 6 stream-score factors, each in [-1, +1]."""

    matchup: float = 0.0
    env: float = 0.0
    form: float = 0.0
    lineup: float = 0.0
    sgp: float = 0.0
    winprob: float = 0.0


class StreamCandidate(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    is_home: bool = False
    score: float = 0.0
    status: str = ""  # raw engine value: "PROBABLE"/"LOCKED"/"FINAL"
    confidence: str = ""  # raw engine value: "HIGH"/"MEDIUM"/"LOW"
    actionable: bool = True
    num_starts: int = 1
    net_sgp: float = 0.0
    opp_wrc_plus: float = 0.0
    opp_k_pct: float = 0.0
    park: float = 1.0
    expected_ip: float = 0.0
    expected_k: float = 0.0
    expected_er: float = 0.0
    win_pct: float = 0.0
    own_pct: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    components: StreamComponents = Field(default_factory=StreamComponents)
    expected_line: str = ""
    rank: int = 0
    reason: str = ""


class BudgetStrip(BaseModel):
    adds_left: int = 0
    adds_total: int = 10
    ip_pace: float = 0.0  # DEFERRED — weekly IP pace plumbing is a follow-up
    ip_target: float = 54.0
    cats_in_play: list[str] = Field(default_factory=list)


class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate] = Field(default_factory=list)
    top_pick: StreamCandidate | None = None
    budget: BudgetStrip = Field(default_factory=BudgetStrip)
```

- [ ] **Step 4: Widen the service mapping**

In `api/services/streaming_service.py`: add `import math` and update the imports to include the new contract names. Add the NaN-safe float helper + rewrite `_to_candidate` to accept `rank` and map all widened fields:

```python
import math

from api.contracts.streaming import BudgetStrip, StreamCandidate, StreamComponents, StreamingResponse
from api.services.player_ref import make_player_ref


def _f(value, default: float = 0.0) -> float:
    """NaN-safe float coercion (None/NaN/junk → default) — keeps NaN out of JSON."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(fval) else fval
```

Replace `_to_candidate` with:

```python
    @staticmethod
    def _to_candidate(row, rank: int = 0) -> StreamCandidate:
        g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
        try:
            pid_int = int(g("player_id", 0) or 0)
        except (TypeError, ValueError):
            pid_int = 0
        comp = g("components", {}) or {}
        ip, k, er = _f(g("expected_ip")), _f(g("expected_k")), _f(g("expected_er"))
        flags = g("risk_flags", []) or []
        return StreamCandidate(
            player=make_player_ref(
                id=pid_int, name=str(g("player_name", "") or ""), positions="SP",
                mlb_id=g("mlb_id"), team_abbr=g("team"),
            ),
            team=str(g("team", "") or ""),
            opponent=str(g("opponent", "") or ""),
            is_home=bool(g("is_home", False)),
            score=_f(g("stream_score")),
            status=str(g("status", "") or ""),
            confidence=str(g("confidence", "") or ""),
            actionable=bool(g("actionable", True)),
            num_starts=int(g("num_starts", 1) or 1),
            net_sgp=_f(g("net_sgp")),
            opp_wrc_plus=_f(g("opp_wrc_plus")),
            opp_k_pct=_f(g("opp_k_pct")),
            park=_f(g("park_factor"), 1.0),
            expected_ip=ip,
            expected_k=k,
            expected_er=er,
            win_pct=_f(g("win_probability")),
            own_pct=_f(g("percent_owned")),
            risk_flags=[str(x) for x in flags],
            components=StreamComponents(
                matchup=_f(comp.get("matchup")), env=_f(comp.get("env")),
                form=_f(comp.get("form")), lineup=_f(comp.get("lineup")),
                sgp=_f(comp.get("sgp")), winprob=_f(comp.get("winprob")),
            ),
            expected_line=f"{ip:.1f} IP · {k:.0f} K · {er:.0f} ER",
            rank=rank,
            reason="",
        )
```

NOTE: confirm the engine board row keys match (`is_home`, `confidence`, `num_starts`, `net_sgp`, `opp_wrc_plus`, `opp_k_pct`, `park_factor`, `expected_ip/k/er`, `win_probability`, `percent_owned`, `risk_flags`, `components` with the 6 sub-keys, `mlb_id`) by reading `build_stream_board` in `src/optimizer/stream_analyzer.py`. If any key differs, use the actual name (keep assertions identical).

- [ ] **Step 5: Run the unit tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_widen.py::test_to_candidate_maps_all_widened_fields tests/api/test_api_streaming_widen.py::test_to_candidate_is_nan_safe -q`
Expected: PASS (2). (`_pick_top`/`_build_budget` tests come in Task 2.)

- [ ] **Step 6: Commit**

```bash
git add api/contracts/streaming.py api/services/streaming_service.py tests/api/test_api_streaming_widen.py
git commit -m "feat(api): widen StreamCandidate to the full board row (NaN-safe) + StreamComponents/BudgetStrip"
```

---

### Task 2: `top_pick` + `budget` + wire into the response + openapi

**Files:**
- Modify: `api/services/streaming_service.py` (add `_pick_top`, `_build_budget`; wire `get_streaming`)
- Test: append to `tests/api/test_api_streaming_widen.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_api_streaming_widen.py`:

```python
def test_pick_top_returns_first_actionable():
    a = StreamingService._to_candidate(_board_row(actionable=False, stream_score=99.0), rank=1)
    b = StreamingService._to_candidate(_board_row(actionable=True, stream_score=80.0), rank=2)
    assert _pick_top([a, b]) is b  # skips the locked higher-scorer
    assert _pick_top([]) is None


class _FakeCtx:
    adds_remaining_this_week = 7
    category_gaps = {"K": -2.0, "ERA": 1.5, "SV": -0.5, "HR": -3.0}  # HR is a hitting cat


def test_build_budget_from_ctx():
    b = _build_budget(_FakeCtx())
    assert b.adds_total == 10
    assert b.adds_left == 7
    assert b.ip_target == 54.0
    assert b.ip_pace == 0.0  # deferred
    # only contested (gap<=0) PITCHING cats; HR (hitting) excluded, ERA (gap>0) excluded
    assert set(b.cats_in_play) == {"K", "SV"}


def test_streaming_endpoint_includes_top_pick_and_budget():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import BudgetStrip, StreamCandidate, StreamingResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _FakeStreaming:
        def get_streaming(self, date=None, limit=25):
            cand = StreamCandidate(player=PlayerRef(id=1, mlb_id=660271, name="X", positions="SP",
                                                    team_abbr="DET", team_id=116),
                                   score=82.5, rank=1, expected_line="6.0 IP · 7 K · 2 ER")
            return StreamingResponse(date="2026-06-19", candidates=[cand], top_pick=cand,
                                     budget=BudgetStrip(adds_left=7, cats_in_play=["K"]))

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _FakeStreaming()
    try:
        body = TestClient(app).get("/api/streaming").json()
        assert body["top_pick"]["expected_line"] == "6.0 IP · 7 K · 2 ER"
        assert body["budget"]["adds_left"] == 7
        assert body["budget"]["ip_target"] == 54.0
        assert body["candidates"][0]["components"]["matchup"] == 0.0
    finally:
        app.dependency_overrides.clear()
```

NOTE: confirm the streaming DI provider name (`get_streaming_service`) in `api/deps.py` and that `api/main.py` exposes `create_app()` (it does — other api tests use it). Adjust references if names differ; keep assertions identical.

- [ ] **Step 2: Run them to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_widen.py -q`
Expected: FAIL — `_pick_top`/`_build_budget` not importable; endpoint lacks top_pick/budget.

- [ ] **Step 3: Add `_pick_top` + `_build_budget` + wire `get_streaming`**

In `api/services/streaming_service.py`, add the two module-level helpers:

```python
def _pick_top(candidates: list[StreamCandidate]) -> StreamCandidate | None:
    """The #1 actionable stream (the board is score-sorted, so the first actionable wins)."""
    for c in candidates:
        if c.actionable:
            return c
    return None


def _build_budget(ctx) -> BudgetStrip:
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT

    adds_total = int(WEEKLY_TRANSACTION_LIMIT)
    try:
        adds_left = int(getattr(ctx, "adds_remaining_this_week", adds_total))
    except (TypeError, ValueError):
        adds_left = adds_total
    try:
        from src.ip_tracker import WEEKLY_TARGET

        ip_target = float(WEEKLY_TARGET)
    except Exception:
        ip_target = 54.0
    gaps = getattr(ctx, "category_gaps", {}) or {}
    pitching = {"W", "L", "SV", "K", "ERA", "WHIP"}
    cats_in_play = [c for c, gap in gaps.items() if c in pitching and _f(gap, 1.0) <= 0]
    return BudgetStrip(
        adds_left=adds_left, adds_total=adds_total,
        ip_pace=0.0, ip_target=ip_target, cats_in_play=cats_in_play,
    )
```

Then update `get_streaming` to compute `rank`, `top_pick`, and `budget` (replace the board loop + the final return):

```python
        candidates: list[StreamCandidate] = []
        top_pick: StreamCandidate | None = None
        budget = BudgetStrip()
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            ctx = build_optimizer_context(
                scope="rest_of_season", yds=yds, config=LeagueConfig(), level_filter="MLB only",
            )
            board = build_stream_board(ctx, target_date)
            if board is not None and not board.empty:
                for rank, (_, row) in enumerate(board.head(limit).iterrows(), start=1):
                    candidates.append(self._to_candidate(row, rank))
            top_pick = _pick_top(candidates)
            budget = _build_budget(ctx)
        except Exception:
            candidates = []  # cold env / no data → empty list

        return StreamingResponse(
            date=target_date, candidates=candidates, top_pick=top_pick, budget=budget,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_widen.py -q`
Expected: PASS (all 5).

- [ ] **Step 5: Regenerate the OpenAPI snapshot**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe scripts/export_openapi.py`
Then: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS. Confirm `api/openapi.json` now has the widened `StreamCandidate` + `StreamComponents`/`BudgetStrip` schemas + `StreamingResponse.top_pick`/`budget`.

- [ ] **Step 6: Commit**

```bash
git add api/services/streaming_service.py api/openapi.json tests/api/test_api_streaming_widen.py
git commit -m "feat(api): add top_pick + budget to StreamingResponse (Stream Finder board)"
```

---

### Task 3: Full api suite + lint

- [ ] **Step 1: Full api suite**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS. If a PRE-EXISTING streaming test asserted the OLD `StreamCandidate` via full-dict equality, update its expected dict to include the new fields (defaults). Do NOT weaken assertions. The `test_no_logic_in_routers` guard must stay green (no router change in this slice).

- [ ] **Step 2: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: clean.

- [ ] **Step 3: Commit (only if Step 1/2 changed files)**

```bash
git add -A && git commit -m "test(api): update streaming tests for widened contract + ruff"
```

---

## Self-Review (completed by plan author)

**Spec coverage** (CMO handoff #1/#2 board half): widened `StreamCandidate` with all engine-computed fields ✔; `expected_line` composed ✔; `components` (6 factors) ✔; `top_pick` ✔; `budget` (adds + ip_target + cats_in_play real; `ip_pace` deferred to 0.0, documented) ✔. `probables[]` + `/analyze` are the explicit NEXT slice.

**Backward compatibility:** every new `StreamCandidate`/`StreamingResponse` field has a default → existing construction + tests still validate. `status`/`confidence` keep emitting raw engine values (no rename/normalize → no existing-test breakage; the frontend adapter maps).

**NaN safety:** `_f()` coerces every engine float (None/NaN/junk → default) so a missing-data NaN never reaches JSON — proactively applying the FA-pool slice's lesson. A NaN-path test is included.

**Placeholder scan:** every code step shows full code; the two "verify the engine keys / DI name" notes are confirmations, not gaps. ✔

**Type consistency:** `_to_candidate(row, rank)`, `_pick_top(candidates)`, `_build_budget(ctx)`, `_f(value, default)` used identically across tasks. Contract field names (`is_home`, `opp_wrc_plus`, `win_pct`, `own_pct`, `expected_line`, `cats_in_play`, etc.) match between models, service, and tests. DET team_id=116 matches `_STATSAPI_TEAM_IDS`. ✔
