# M0 — Streaming `probables[]` + `POST /api/streaming/analyze` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the streaming backend for the CMO's Pitcher Streaming page: add `probables[]` (the picker list of every probable starter for the date) to `StreamingResponse`, and a new `POST /api/streaming/analyze {pitcher_id, date}` endpoint (the "Analyze Any Starter" feature) returning a `PitcherScorecard` (the widened candidate + a 6-factor breakdown). This is the CMO spec's handoff items #2 (`probables`) and #3 (analyze endpoint) — the 19th `/api/*` endpoint.

**Architecture:** Reuse the engine board. `build_stream_board(ctx, date, include_rostered=True)` is the complete "FAs + your roster SPs" probable universe. `get_streaming` keeps its existing `candidates` board build (`include_rostered=False`) UNCHANGED and adds a SECOND build (`include_rostered=True`) → `probables`. `analyze_pitcher` builds that same board and finds the requested `pitcher_id` row (reusing all per-row engine scoring), maps it to `_to_candidate`, and adds `factors[]` (the 6 components + registry weights + composed detail strings). Graceful: pitcher-not-scheduled → `found=False` (no raise). `src/` untouched; routers logic-free.

**Phase-1 scope boundary (documented):** "any probable" = FAs + the user's own roster SPs (the board's universe). Other teams' rostered probables are excluded (not streamable; the board always drops them) — analyzing those is a Phase-2 that needs the direct `score_stream_candidate` assembly. `start_likelihood` is derived from the date-proximity `confidence` tier (HIGH→"confirmed", MEDIUM→"likely", LOW→"projected") — a proxy, since the engine carries no statsapi confirmation flag. `form`/`lineup` factor details are generic (the row carries no raw inputs for them).

**Convention:** snake_case; reuse `make_player_ref` + the slice-1 `_f` NaN-safe float. `PitcherScorecard` inherits the (already-widened) `StreamCandidate`.

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = worktree root.

**Tech Stack:** FastAPI + Pydantic v2; pandas; pytest. `fastapi==0.137.1`/`httpx==0.28.1` pinned.

---

## File Structure

- `api/contracts/streaming.py` — **modify**: add `FactorDetail`, `ProbableStarter`, `PitcherScorecard`, `StreamAnalyzeRequest`, `StreamAnalyzeResponse`; add `probables` to `StreamingResponse`.
- `api/services/streaming_service.py` — **modify**: add `_likelihood_from`, `_to_probable`, `_factors`; add `probables` (2nd board build) in `get_streaming`; add `analyze_pitcher`.
- `api/routers/streaming.py` — **modify**: add the `POST /streaming/analyze` route.
- `api/openapi.json` — **regenerate**.
- `tests/api/test_api_streaming_analyze.py` — **create**: helper unit tests + the GET-probables + POST-analyze contract tests.

---

### Task 1: Contracts + helpers (`_likelihood_from`, `_to_probable`, `_factors`)

**Files:**
- Modify: `api/contracts/streaming.py`
- Modify: `api/services/streaming_service.py`
- Test: `tests/api/test_api_streaming_analyze.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/api/test_api_streaming_analyze.py`:

```python
"""Streaming analyze + probables: helper unit tests + contract tests."""

from __future__ import annotations

from api.services.streaming_service import _factors, _likelihood_from, _to_probable


def _row(**over):
    row = {
        "player_id": 660271, "mlb_id": 660271, "player_name": "Tarik Skubal", "team": "DET",
        "opponent": "CWS", "is_home": True, "confidence": "HIGH",
        "opp_wrc_plus": 88.0, "opp_k_pct": 24.5, "park_factor": 0.96,
        "net_sgp": 1.8, "win_probability": 0.62,
        "components": {"matchup": 0.4, "env": 0.1, "form": 0.2, "lineup": -0.1, "sgp": 0.5, "winprob": 0.3},
    }
    row.update(over)
    return row


def test_likelihood_from_confidence():
    assert _likelihood_from("HIGH") == "confirmed"
    assert _likelihood_from("MEDIUM") == "likely"
    assert _likelihood_from("LOW") == "projected"
    assert _likelihood_from("") == "projected"  # unknown → projected


def test_to_probable_maps_row():
    p = _to_probable(_row())
    assert p.player.mlb_id == 660271
    assert p.player.team_id == 116  # DET
    assert p.team == "DET"
    assert p.opponent == "CWS"
    assert p.is_home is True
    assert p.pos_group == "SP"
    assert p.start_likelihood == "confirmed"


def test_factors_six_components_weights_and_details():
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

    factors = _factors(_row())
    assert [f.key for f in factors] == ["matchup", "sgp", "form", "lineup", "env", "winprob"]
    by_key = {f.key: f for f in factors}
    assert by_key["matchup"].value == 0.4
    assert by_key["matchup"].detail == "vs CWS: 88 wRC+, 25% K"
    assert by_key["sgp"].detail == "+1.80 SGP"
    assert by_key["env"].detail == "Park factor 0.96"
    assert by_key["winprob"].detail == "62% team win prob"
    # weights come from the registry (live read)
    assert by_key["matchup"].weight == float(_CR["stream_score_w_matchup"].value)


def test_factors_is_nan_safe():
    factors = _factors(_row(opp_wrc_plus=float("nan"), components={"matchup": float("nan")}))
    by_key = {f.key: f for f in factors}
    assert by_key["matchup"].value == 0.0  # NaN component → 0.0
    assert "wRC+" in by_key["matchup"].detail  # composes without crashing (0 wRC+)
```

NOTE: `opp_k_pct=24.5` → `"25% K"` (`:.0f` rounds 24.5→24 or 25? Python banker's rounding makes `f"{24.5:.0f}"` = "24"). VERIFY the actual format output and set the expected string to match (`"24% K"` vs `"25% K"`) — adjust the assertion to reality, do not change the format spec.

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_analyze.py -q`
Expected: FAIL — `ImportError: cannot import name '_factors'` etc.

- [ ] **Step 3: Add the contracts**

Append to `api/contracts/streaming.py` (after the existing classes; `StreamCandidate` is already widened from the prior slice):

```python
class FactorDetail(BaseModel):
    key: str  # matchup/sgp/form/lineup/env/winprob
    label: str
    value: float = 0.0  # the component, [-1, +1]
    weight: float = 0.0  # the stream_score weight from CONSTANTS_REGISTRY
    detail: str = ""


class ProbableStarter(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    is_home: bool = False
    pos_group: str = "SP"  # SP/SP-RP/RP — engine probables are SPs
    start_likelihood: str = ""  # confirmed/likely/projected (from confidence proximity)


class PitcherScorecard(StreamCandidate):
    factors: list[FactorDetail] = Field(default_factory=list)


class StreamAnalyzeRequest(BaseModel):
    pitcher_id: int
    date: str = ""  # YYYY-MM-DD; "" → today


class StreamAnalyzeResponse(BaseModel):
    found: bool = False
    scorecard: PitcherScorecard | None = None
```

Then add `probables` to `StreamingResponse` (insert the field):

```python
class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate] = Field(default_factory=list)
    top_pick: StreamCandidate | None = None
    budget: BudgetStrip = Field(default_factory=BudgetStrip)
    probables: list[ProbableStarter] = Field(default_factory=list)
```

- [ ] **Step 4: Add the service helpers**

In `api/services/streaming_service.py`: extend the contract import to include `FactorDetail, PitcherScorecard, ProbableStarter, StreamAnalyzeRequest, StreamAnalyzeResponse`. Add the three module-level helpers (alongside `_f`/`_pick_top`/`_build_budget`):

```python
def _likelihood_from(confidence) -> str:
    """Map the date-proximity confidence tier → a start-likelihood label (proxy)."""
    return {"HIGH": "confirmed", "MEDIUM": "likely", "LOW": "projected"}.get(
        str(confidence or "").upper(), "projected"
    )


def _to_probable(row) -> ProbableStarter:
    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    try:
        pid = int(g("player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    return ProbableStarter(
        player=make_player_ref(
            id=pid, name=str(g("player_name", "") or ""), positions="SP",
            mlb_id=g("mlb_id"), team_abbr=g("team"),
        ),
        team=str(g("team", "") or ""),
        opponent=str(g("opponent", "") or ""),
        is_home=bool(g("is_home", False)),
        pos_group="SP",
        start_likelihood=_likelihood_from(g("confidence")),
    )


def _factors(row) -> list[FactorDetail]:
    """The 6 stream-score factors with registry weights + composed detail strings."""
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    comp = g("components", {}) or {}
    opp = str(g("opponent", "") or "")
    wrc, kpct, park = _f(g("opp_wrc_plus")), _f(g("opp_k_pct")), _f(g("park_factor"), 1.0)
    nsgp, wp = _f(g("net_sgp")), _f(g("win_probability"))

    def _w(key: str) -> float:
        try:
            return float(_CR[f"stream_score_w_{key}"].value)
        except Exception:
            return 0.0

    specs = [
        ("matchup", "Matchup", f"vs {opp}: {wrc:.0f} wRC+, {kpct:.0f}% K"),
        ("sgp", "Streaming value", f"{nsgp:+.2f} SGP"),
        ("form", "Recent form", "L14 form vs baseline"),
        ("lineup", "Lineup", "Opposing lineup exposure"),
        ("env", "Environment", f"Park factor {park:.2f}"),
        ("winprob", "Win probability", f"{wp * 100:.0f}% team win prob"),
    ]
    return [
        FactorDetail(key=k, label=lbl, value=_f(comp.get(k)), weight=_w(k), detail=d)
        for k, lbl, d in specs
    ]
```

- [ ] **Step 5: Run the unit tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_analyze.py -q`
Expected: PASS (4 helper tests). (If `_factors`'s `"{kpct:.0f}% K"` rounds differently than the assertion, fix the TEST's expected string to the real output.)

- [ ] **Step 6: Commit**

```bash
git add api/contracts/streaming.py api/services/streaming_service.py tests/api/test_api_streaming_analyze.py
git commit -m "feat(api): streaming probables/analyze contracts + _to_probable/_factors helpers"
```

---

### Task 2: `get_streaming` probables + `analyze_pitcher` + route + openapi

**Files:**
- Modify: `api/services/streaming_service.py` (`get_streaming` adds probables; new `analyze_pitcher`)
- Modify: `api/routers/streaming.py` (POST route)
- Test: append to `tests/api/test_api_streaming_analyze.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing contract tests**

Append to `tests/api/test_api_streaming_analyze.py`:

```python
def test_streaming_get_includes_probables():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import ProbableStarter, StreamingResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _Fake:
        def get_streaming(self, date=None, limit=25):
            return StreamingResponse(
                date="2026-06-19",
                probables=[ProbableStarter(
                    player=PlayerRef(id=1, mlb_id=660271, name="X", positions="SP",
                                     team_abbr="DET", team_id=116),
                    team="DET", opponent="CWS", is_home=True, start_likelihood="confirmed")],
            )

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/streaming").json()
        assert body["probables"][0]["start_likelihood"] == "confirmed"
        assert body["probables"][0]["player"]["mlb_id"] == 660271
    finally:
        app.dependency_overrides.clear()


def test_analyze_endpoint_returns_scorecard():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import FactorDetail, PitcherScorecard, StreamAnalyzeResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _Fake:
        def analyze_pitcher(self, req):
            return StreamAnalyzeResponse(
                found=True,
                scorecard=PitcherScorecard(
                    player=PlayerRef(id=1, mlb_id=660271, name="Skubal", positions="SP",
                                     team_abbr="DET", team_id=116),
                    score=82.5, rank=1, expected_line="6.0 IP · 7 K · 2 ER",
                    factors=[FactorDetail(key="matchup", label="Matchup", value=0.4,
                                          weight=0.35, detail="vs CWS: 88 wRC+, 24% K")],
                ),
            )

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _Fake()
    try:
        resp = TestClient(app).post("/api/streaming/analyze", json={"pitcher_id": 660271, "date": "2026-06-19"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["scorecard"]["score"] == 82.5
        assert body["scorecard"]["factors"][0]["key"] == "matchup"
        assert body["scorecard"]["factors"][0]["weight"] == 0.35
    finally:
        app.dependency_overrides.clear()
```

NOTE: confirm `get_streaming_service` (DI) + `create_app` names (used by other streaming tests). Adjust if different; keep assertions identical.

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_analyze.py -q`
Expected: FAIL — the POST route 404s; GET response has no `probables` from the real service yet (the fake test fails because the route/contract aren't wired... the GET fake test should actually pass already since `probables` is on the contract — if so, only the POST test fails. Either way, proceed).

- [ ] **Step 3: Add `probables` to `get_streaming` + the `analyze_pitcher` method**

In `api/services/streaming_service.py`, inside `get_streaming`'s `try` block, AFTER the existing `candidates`/`top_pick`/`budget` computation and BEFORE the `return`, add a second board build for probables (leave the existing candidates build UNCHANGED):

```python
            probables: list[ProbableStarter] = []
            try:
                full_board = build_stream_board(ctx, target_date, include_rostered=True)
                if full_board is not None and not full_board.empty:
                    probables = [_to_probable(row) for _, row in full_board.iterrows()]
            except Exception:
                probables = []
```

Initialize `probables: list[ProbableStarter] = []` near the top with `candidates`/`budget`, and pass it to the return:

```python
        return StreamingResponse(
            date=target_date, candidates=candidates, top_pick=top_pick,
            budget=budget, probables=probables,
        )
```

Add the `analyze_pitcher` method to `StreamingService`:

```python
    def analyze_pitcher(self, req: StreamAnalyzeRequest) -> StreamAnalyzeResponse:
        from src.game_day import get_target_game_date

        try:
            date = req.date or str(get_target_game_date())
        except Exception:
            from datetime import UTC, datetime

            date = req.date or datetime.now(UTC).strftime("%Y-%m-%d")
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            ctx = build_optimizer_context(
                scope="rest_of_season", yds=get_yahoo_data_service(),
                config=LeagueConfig(), level_filter="MLB only",
            )
            board = build_stream_board(ctx, date, include_rostered=True)
            if board is not None and not board.empty:
                match = board[board["player_id"] == req.pitcher_id]
                if not match.empty:
                    row = match.iloc[0]
                    rank = int(match.index[0]) + 1  # board is reset_index'd, so index == 0-based rank
                    cand = self._to_candidate(row, rank)
                    scorecard = PitcherScorecard(**cand.model_dump(), factors=_factors(row))
                    return StreamAnalyzeResponse(found=True, scorecard=scorecard)
        except Exception:
            pass
        return StreamAnalyzeResponse(found=False, scorecard=None)
```

- [ ] **Step 4: Add the POST route**

In `api/routers/streaming.py`: extend the imports and add the route (the router already has `prefix="/api"` and is mounted — NO `api/main.py` change):

```python
from api.contracts.streaming import StreamAnalyzeRequest, StreamAnalyzeResponse, StreamingResponse
```

```python
@router.post("/streaming/analyze", response_model=StreamAnalyzeResponse)
def analyze_streaming(
    req: StreamAnalyzeRequest, service=Depends(get_streaming_service)
) -> StreamAnalyzeResponse:
    return service.analyze_pitcher(req)
```

- [ ] **Step 5: Run the contract tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_streaming_analyze.py -q`
Expected: PASS (all 6).

- [ ] **Step 6: Regenerate OpenAPI**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe scripts/export_openapi.py`
Then: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS. Confirm `api/openapi.json` has `POST /api/streaming/analyze` + `PitcherScorecard`/`FactorDetail`/`ProbableStarter`/`StreamAnalyzeRequest`/`StreamAnalyzeResponse` schemas + `StreamingResponse.probables`.

- [ ] **Step 7: Commit**

```bash
git add api/services/streaming_service.py api/routers/streaming.py api/openapi.json tests/api/test_api_streaming_analyze.py
git commit -m "feat(api): POST /api/streaming/analyze + probables[] on StreamingResponse (Analyze Any Starter)"
```

---

### Task 3: Full api suite + lint

- [ ] **Step 1: Full api suite**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS (prior 121 + the new tests). `test_no_logic_in_routers` stays green (the new POST route only calls the service).

- [ ] **Step 2: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: clean.

- [ ] **Step 3: Commit (only if Step 1/2 changed files)**

```bash
git add -A && git commit -m "test(api): ruff + any test updates for streaming analyze"
```

---

## Self-Review (completed by plan author)

**Spec coverage** (CMO handoff #2 `probables` + #3 analyze): `probables[]` on `StreamingResponse` ✔ (2nd board build, existing candidates UNCHANGED); `POST /api/streaming/analyze` → `PitcherScorecard` (widened candidate + `factors[]`) ✔; `factors[]` = 6 components + registry weights + composed details ✔; `ProbableStarter` with pos_group + start_likelihood ✔. Documented deferrals: other-team-rostered probables out of scope; form/lineup details generic; start_likelihood is a confidence proxy.

**Reuse / safety:** reuses `build_stream_board` (no engine change), `_to_candidate`/`_f`/`make_player_ref` (prior slice). `PitcherScorecard` inherits the widened `StreamCandidate`. Existing `candidates` board build untouched (a separate 2nd build for probables → zero risk to the prior slice). `analyze_pitcher` never raises (graceful `found=False`). Router logic-free.

**NaN safety:** `_factors` routes every numeric through `_f`; the NaN-path test is included.

**Placeholder scan:** every code step shows full code; the two "verify" notes (the `:.0f` rounding of opp_k_pct; the DI/create_app names) are confirmations. ✔

**Type consistency:** `_likelihood_from`/`_to_probable`/`_factors` + `analyze_pitcher(req)` + the new models used identically across tasks; `stream_score_w_*` registry keys match the engine; DET=116 matches `_STATSAPI_TEAM_IDS`. `PitcherScorecard(**cand.model_dump(), factors=...)` is valid because it inherits every `StreamCandidate` field. ✔
