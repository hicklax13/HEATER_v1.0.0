# M0 — `GET /api/leaders/overall?lens=` (cross-category 5-lens leaderboard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GET /api/leaders/overall?lens={overall|hot|cold|breakout|sell}` — a cross-category overall-value leaderboard with 5 lens filters, for the frontend Research page (which is already live on `/api/leaders`). The existing per-category `/api/leaders?category=HR` is KEPT unchanged. This is the 20th `/api/*` endpoint (gap spec §6 Research).

**Architecture:** Data-driven, not 5 code paths. The service loads the player pool + a season-stats frame (the SAME SQL the existing `leaders_service` uses), dispatches by `lens` to ONE engine function each (all already exist in `src/`), normalizes that lens's raw score to a uniform `_value` ∈ [0,100], then maps EVERY row the same way: `_to_overall_row(rank, row, pool, lens)` → `PlayerRef` via `player_ref_from_pool` + 3 key stats + `is_hitter` (all from a pool lookup), with per-lens `trend`/`tag`/`note` from a `_LENS_META` table. NaN-safe (the FA-pool/streaming lesson). `src/` untouched; routers logic-free.

**Engine functions (verified signatures):**
- overall → `src.leaders.compute_category_value_leaders(season_stats_df, min_pa=50, min_ip=20.0, top_n=N)` → cols incl. `player_id`, `category_value` (z-score composite, ~−2..+4). Normalize: `(z+4)/8*100` clamped.
- hot/cold → `src.trend_tracker.compute_player_trends(pool, season_stats)` → cols incl. `player_id`, `trend_delta` (∈ ~[−3,3]), `trend_label` ("HOT"/"COLD"/"NEUTRAL"). Filter by label. Normalize delta: `(d+3)/6*100` clamped.
- breakout → `src.leaders.compute_breakout_scores_batch(pool)` → pool + `breakout_score` (already 0–100). Sort desc, top-N. `_value` = breakout_score (clamped).
- sell → `src.trend_tracker.detect_sell_high_candidates(pool, season_stats)` → hot subset + `sustainability_score`. Sort by `trend_delta` desc, top-N. Normalize delta.

**Contract convention:** snake_case; nested `PlayerRef`. Frontend `stats` is a `string[]` (e.g. `["24 HR","58 R",".322 AVG"]`) — NOT objects. `value` 0–100.

**Deferred/defaulted (documented):** `trend` is fixed per lens (overall=flat, hot=up, cold=down, breakout=up, sell=up) — momentum is implied by the lens, not computed per row. `note` is a per-lens default string (no per-row "leads league in HR" enrichment in Phase 1). `stats` come from the pool's `ytd_*` (season actuals); a ranked player absent from the pool degrades to a `"Player {id}"` ref + empty stats (graceful).

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = worktree root.

**Tech Stack:** FastAPI + Pydantic v2; pandas; pytest. `fastapi==0.137.1`/`httpx==0.28.1` pinned.

---

## File Structure

- `api/contracts/leaders.py` — **modify**: add `OverallLeaderRow`, `LeadersOverallResponse` (keep existing `LeaderRow`/`LeadersResponse`).
- `api/services/leaders_overall_service.py` — **create**: helpers + `LeadersOverallService`.
- `api/deps.py` — **modify**: add `get_leaders_overall_service()` + import.
- `api/routers/leaders.py` — **modify**: add `GET /leaders/overall`.
- `api/openapi.json` — **regenerate**.
- `tests/api/test_api_leaders_overall.py` — **create**: helper unit tests + fake-service contract test.

---

### Task 1: Contracts + pure helpers

**Files:**
- Modify: `api/contracts/leaders.py`
- Create: `api/services/leaders_overall_service.py`
- Test: `tests/api/test_api_leaders_overall.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/api/test_api_leaders_overall.py`:

```python
"""leaders/overall: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.leaders_overall_service import (
    _LENS_META,
    _norm_delta,
    _norm_z,
    _overall_stats,
    _to_overall_row,
)


def test_norm_z_maps_and_clamps():
    assert _norm_z(0.0) == 50.0  # (0+4)/8*100
    assert _norm_z(4.0) == 100.0
    assert _norm_z(-4.0) == 0.0
    assert _norm_z(99.0) == 100.0  # clamp
    assert _norm_z(float("nan")) == 0.0  # NaN-safe


def test_norm_delta_maps_and_clamps():
    assert _norm_delta(0.0) == 50.0  # (0+3)/6*100
    assert _norm_delta(3.0) == 100.0
    assert _norm_delta(-3.0) == 0.0
    assert _norm_delta(float("nan")) == 0.0


def test_overall_stats_hitter_and_pitcher():
    hrow = {"ytd_hr": 24, "ytd_r": 58, "ytd_avg": 0.322}
    assert _overall_stats(hrow, True) == ["24 HR", "58 R", ".322 AVG"]
    prow = {"ytd_k": 180, "ytd_era": 3.21, "ytd_whip": 1.05}
    assert _overall_stats(prow, False) == ["180 K", "3.21 ERA", "1.05 WHIP"]


def test_lens_meta_covers_all_five():
    assert set(_LENS_META) == {"overall", "hot", "cold", "breakout", "sell"}
    assert _LENS_META["overall"] == ("", "flat", _LENS_META["overall"][2])  # tag empty for overall
    assert _LENS_META["hot"][0] == "hot" and _LENS_META["hot"][1] == "up"
    assert _LENS_META["cold"][1] == "down"


def test_to_overall_row_enriches_and_stamps_lens():
    pool = pd.DataFrame([{
        "player_id": 1, "name": "Aaron Judge", "positions": "OF", "mlb_id": 592450,
        "team": "NYY", "is_hitter": True, "ytd_hr": 30, "ytd_r": 70, "ytd_avg": 0.31,
    }])
    row = {"player_id": 1, "_value": 88.0}
    item = _to_overall_row(2, row, pool, "hot")
    assert item.rank == 2
    assert item.value == 88.0
    assert item.player.mlb_id == 592450
    assert item.player.team_id == 147  # NYY
    assert item.hitter is True
    assert item.tag == "hot"
    assert item.trend == "up"
    assert item.note == _LENS_META["hot"][2]
    assert item.stats == ["30 HR", "70 R", ".310 AVG"]


def test_to_overall_row_missing_pool_row_degrades():
    item = _to_overall_row(1, {"player_id": 999, "_value": 50.0}, pd.DataFrame(), "overall")
    assert item.player.name == "Player 999"
    assert item.stats == []  # no pool row → no stats
    assert item.tag == ""
    assert item.value == 50.0
```

NOTE: `.322`/`.310` is the leading-zero-stripped AVG (baseball convention) — confirm `_overall_stats` produces that exact format; if you choose `format_stat`, verify its AVG output (it strips the leading zero) and keep the assertions matching reality.

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_leaders_overall.py -q`
Expected: FAIL — `ModuleNotFoundError: api.services.leaders_overall_service`.

- [ ] **Step 3: Add the contracts**

Append to `api/contracts/leaders.py` (keep the existing `LeaderRow`/`LeadersResponse`):

```python
class OverallLeaderRow(BaseModel):
    rank: int
    player: PlayerRef
    value: float = 0.0  # 0-100 lens score
    stats: list[str] = Field(default_factory=list)  # 3 key stats, e.g. ["24 HR","58 R",".322 AVG"]
    trend: str = "flat"  # up/down/flat
    tag: str = ""  # ""/hot/cold/breakout/sell
    note: str = ""
    hitter: bool = True


class LeadersOverallResponse(BaseModel):
    lens: str = "overall"
    rows: list[OverallLeaderRow] = Field(default_factory=list)
```

Ensure `api/contracts/leaders.py` imports `Field` (`from pydantic import BaseModel, Field`) and `PlayerRef` (already imported for `LeaderRow`).

- [ ] **Step 4: Create the service + helpers**

Create `api/services/leaders_overall_service.py`:

```python
"""leaders/overall service — a cross-category, 5-lens overall-value leaderboard
for the Research page (distinct from the per-category /api/leaders). Resilient:
cold env / unknown lens → empty rows. Composes existing src engines; src untouched."""

from __future__ import annotations

import logging
import math

from api.contracts.leaders import LeadersOverallResponse, OverallLeaderRow
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# lens -> (tag, trend, note). trend is fixed per lens (momentum implied by the lens).
_LENS_META: dict[str, tuple[str, str, str]] = {
    "overall": ("", "flat", "Overall value across all categories"),
    "hot": ("hot", "up", "Trending hot vs projection"),
    "cold": ("cold", "down", "Cooling off vs projection"),
    "breakout": ("breakout", "up", "Process metrics signal a breakout"),
    "sell": ("sell", "up", "Hot but low sustainability — regression risk"),
}

_HIT_STATS = (("HR", "ytd_hr", "int"), ("R", "ytd_r", "int"), ("AVG", "ytd_avg", "avg"))
_PIT_STATS = (("K", "ytd_k", "int"), ("ERA", "ytd_era", "f2"), ("WHIP", "ytd_whip", "f2"))


def _sf(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _norm_z(z) -> float:
    """z-score composite (~−4..+4) → 0-100."""
    return round(max(0.0, min(100.0, (_sf(z) + 4.0) / 8.0 * 100.0)), 1)


def _norm_delta(d) -> float:
    """trend_delta (~−3..+3) → 0-100."""
    return round(max(0.0, min(100.0, (_sf(d) + 3.0) / 6.0 * 100.0)), 1)


def _fmt(value, kind: str) -> str:
    fval = _sf(value)
    if kind == "int":
        return str(int(round(fval)))
    if kind == "avg":
        return f"{fval:.3f}"[1:] if 0.0 <= fval < 1.0 else f"{fval:.3f}"  # ".322"
    return f"{fval:.2f}"  # f2


def _overall_stats(row, hitter: bool) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    spec = _HIT_STATS if hitter else _PIT_STATS
    return [f"{_fmt(g(col), kind)} {label}" for label, col, kind in spec]


def _to_overall_row(rank: int, row, pool, lens: str) -> OverallLeaderRow:
    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    try:
        pid = int(g("player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    tag, trend, note = _LENS_META.get(lens, ("", "flat", ""))
    # Pull is_hitter + stats from the pool row (season actuals); graceful if absent.
    prow = None
    hitter = True
    try:
        import pandas as pd

        if isinstance(pool, pd.DataFrame) and not pool.empty:
            match = pool[pool["player_id"] == pid]
            if not match.empty:
                prow = match.iloc[0]
                ih = prow.get("is_hitter", True)
                hitter = True if (isinstance(ih, float) and math.isnan(ih)) else bool(ih)
    except Exception:
        prow = None
    return OverallLeaderRow(
        rank=rank,
        player=player_ref_from_pool(pid, pool, name=g("name") or g("player_name"), positions=g("positions")),
        value=round(_sf(g("_value")), 1),
        stats=_overall_stats(prow, hitter) if prow is not None else [],
        trend=trend,
        tag=tag,
        note=note,
        hitter=hitter,
    )


class LeadersOverallService:
    _VALID = ("overall", "hot", "cold", "breakout", "sell")

    def get_leaders_overall(self, lens: str = "overall", limit: int = 25) -> LeadersOverallResponse:
        lens = lens if lens in self._VALID else "overall"
        rows: list[OverallLeaderRow] = []
        try:
            ranked, pool = self._ranked(lens, limit)
            if ranked is not None and not ranked.empty:
                rows = [
                    _to_overall_row(i + 1, r, pool, lens)
                    for i, r in enumerate(ranked.head(limit).to_dict("records"))
                ]
        except Exception as exc:
            logger.warning("LeadersOverallService(%s) failed: %s", lens, exc)
            rows = []
        return LeadersOverallResponse(lens=lens, rows=rows)

    def _ranked(self, lens: str, limit: int):
        """Return (ranked_df_with_player_id_and__value, pool). Each branch normalizes
        its raw score into a uniform `_value` ∈ [0,100]."""
        import pandas as pd

        from src.database import load_player_pool

        pool = load_player_pool()
        if pool is None or pool.empty:
            return pd.DataFrame(), pd.DataFrame()

        if lens == "breakout":
            from src.leaders import compute_breakout_scores_batch

            bdf = compute_breakout_scores_batch(pool)
            if bdf is None or bdf.empty:
                return pd.DataFrame(), pool
            bdf = bdf.sort_values("breakout_score", ascending=False).reset_index(drop=True)
            bdf["_value"] = [round(max(0.0, min(100.0, _sf(v))), 1) for v in bdf["breakout_score"]]
            return bdf, pool

        season_stats = self._load_season_stats()
        if season_stats is None or season_stats.empty:
            return pd.DataFrame(), pool

        if lens == "overall":
            from src.leaders import compute_category_value_leaders

            vdf = compute_category_value_leaders(season_stats, top_n=max(limit, 25))
            if vdf is None or vdf.empty:
                return pd.DataFrame(), pool
            vdf = vdf.reset_index(drop=True)
            vdf["_value"] = [_norm_z(v) for v in vdf["category_value"]]
            return vdf, pool

        if lens in ("hot", "cold"):
            from src.trend_tracker import compute_player_trends

            tdf = compute_player_trends(pool, season_stats)
            if tdf is None or tdf.empty:
                return pd.DataFrame(), pool
            want = "HOT" if lens == "hot" else "COLD"
            tdf = tdf[tdf["trend_label"] == want].copy()
            tdf = tdf.sort_values("trend_delta", ascending=(lens == "cold")).reset_index(drop=True)
            tdf["_value"] = [_norm_delta(v) for v in tdf["trend_delta"]]
            return tdf, pool

        # sell
        from src.trend_tracker import detect_sell_high_candidates

        sdf = detect_sell_high_candidates(pool, season_stats)
        if sdf is None or sdf.empty:
            return pd.DataFrame(), pool
        sdf = sdf.sort_values("trend_delta", ascending=False).reset_index(drop=True)
        sdf["_value"] = [_norm_delta(v) for v in sdf["trend_delta"]]
        return sdf, pool

    @staticmethod
    def _load_season_stats():
        """Season-stats frame feeding the value/trend engines — the SAME shape the
        existing leaders_service uses."""
        import pandas as pd

        from src.database import get_connection

        conn = get_connection()
        try:
            return pd.read_sql_query(
                """
                SELECT
                    s.player_id, p.name, p.positions, p.is_hitter,
                    s.pa, s.ip, s.r, s.hr, s.rbi, s.sb, s.avg, s.obp,
                    s.w, s.l, s.sv, s.k, s.era, s.whip
                FROM season_stats s
                JOIN players p ON p.player_id = s.player_id
                WHERE s.season = 2026
                GROUP BY s.player_id
                """,
                conn,
            )
        finally:
            conn.close()
```

NOTE for the implementer: VERIFY against the real engines before finalizing —
(a) `compute_category_value_leaders` output has `player_id` + `category_value` (it does per the def); 
(b) `compute_player_trends`/`detect_sell_high_candidates` output rows carry `player_id` + `trend_delta` + `trend_label`; 
(c) `compute_breakout_scores_batch` output carries `player_id` + `breakout_score`. 
If a column name differs, adapt the branch (keep the `_value` normalization + the unit-test assertions identical). The `_load_season_stats` SQL is copied from `leaders_service.get_leaders` (proven to feed the sibling `compute_category_leaders`); keep the `2026` season + `GROUP BY s.player_id`.

- [ ] **Step 5: Run the helper unit tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_leaders_overall.py -q`
Expected: PASS (6 helper tests). (Fix `_fmt`'s AVG output vs the assertion if needed — the leading-zero strip.)

- [ ] **Step 6: Commit**

```bash
git add api/contracts/leaders.py api/services/leaders_overall_service.py tests/api/test_api_leaders_overall.py
git commit -m "feat(api): leaders/overall contracts + lens normalizers/mapper (LeadersOverallService)"
```

---

### Task 2: Wire the endpoint (route + DI + fake-service test + openapi)

**Files:**
- Modify: `api/deps.py`
- Modify: `api/routers/leaders.py`
- Test: append to `tests/api/test_api_leaders_overall.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing contract test**

Append to `tests/api/test_api_leaders_overall.py`:

```python
def test_leaders_overall_endpoint_returns_contract():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.leaders import LeadersOverallResponse, OverallLeaderRow
    from api.deps import get_leaders_overall_service
    from api.main import create_app

    class _Fake:
        def get_leaders_overall(self, lens="overall", limit=25):
            return LeadersOverallResponse(
                lens=lens,
                rows=[OverallLeaderRow(
                    rank=1,
                    player=PlayerRef(id=1, mlb_id=592450, name="Judge", positions="OF",
                                     team_abbr="NYY", team_id=147),
                    value=92.0, stats=["30 HR", "70 R", ".310 AVG"],
                    trend="up", tag="hot", note="Trending hot", hitter=True,
                )],
            )

    app = create_app()
    app.dependency_overrides[get_leaders_overall_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/leaders/overall?lens=hot").json()
        assert body["lens"] == "hot"
        r = body["rows"][0]
        assert r["player"]["mlb_id"] == 592450
        assert r["value"] == 92.0
        assert r["stats"] == ["30 HR", "70 R", ".310 AVG"]
        assert r["tag"] == "hot"
    finally:
        app.dependency_overrides.clear()
```

NOTE: confirm `get_leaders_overall_service` (after you add it) + `create_app` names.

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_leaders_overall.py::test_leaders_overall_endpoint_returns_contract -q`
Expected: FAIL — `ImportError: get_leaders_overall_service` / route 404.

- [ ] **Step 3: Add the DI provider**

In `api/deps.py`: add the import alongside the others —
```python
from api.services.leaders_overall_service import LeadersOverallService
```
and the provider (near `get_leaders_service`) —
```python
def get_leaders_overall_service() -> LeadersOverallService:
    return LeadersOverallService()
```

- [ ] **Step 4: Add the route**

In `api/routers/leaders.py`: extend imports + add the route (the router is already mounted — NO `main.py` change):
```python
from api.contracts.leaders import LeadersOverallResponse, LeadersResponse
from api.deps import get_leaders_overall_service, get_leaders_service
```
```python
@router.get("/leaders/overall", response_model=LeadersOverallResponse)
def get_leaders_overall(
    lens: str = "overall", limit: int = 25, service=Depends(get_leaders_overall_service)
) -> LeadersOverallResponse:
    return service.get_leaders_overall(lens, limit)
```

- [ ] **Step 5: Run the contract test**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_leaders_overall.py -q`
Expected: PASS (all 7).

- [ ] **Step 6: Regenerate OpenAPI**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe scripts/export_openapi.py`
Then: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS. Confirm `/api/leaders/overall` + `OverallLeaderRow`/`LeadersOverallResponse` schemas are present.

- [ ] **Step 7: Commit**

```bash
git add api/deps.py api/routers/leaders.py api/openapi.json tests/api/test_api_leaders_overall.py
git commit -m "feat(api): mount GET /api/leaders/overall (5-lens cross-category leaderboard)"
```

---

### Task 3: Full api suite + lint

- [ ] **Step 1: Full api suite**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS (prior 127 + the new tests). `test_no_logic_in_routers` green (the new GET only calls the service).

- [ ] **Step 2: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: clean.

- [ ] **Step 3: Commit (only if Step 1/2 changed files)**

```bash
git add -A && git commit -m "test(api): ruff + any updates for leaders/overall"
```

---

## Self-Review (completed by plan author)

**Spec coverage** (gap spec §6 Research): `GET /api/leaders/overall?lens=` returning a cross-category leaderboard, one lens per request ✔; per row `rank`/`player`(+mlb_id/team via PlayerRef)/`value`(0-100)/`stats`(3 strings)/`trend`/`tag`/`note`/`hitter` ✔; all 5 lenses ✔. Documented deferrals: `trend` fixed per-lens; `note` per-lens default; stats from pool `ytd_*`.

**Reuse / safety:** existing `/api/leaders` per-category endpoint UNCHANGED. Reuses `player_ref_from_pool` + the proven season-stats SQL + 4 existing engine functions (no engine change). One data-driven mapper + `_LENS_META` table (not 5 code paths). Service never raises (cold env / unknown lens → empty rows; unknown lens coerced to "overall"). NaN/inf-safe via `_sf`.

**Placeholder scan:** every step shows full code. The "verify engine output columns" note is a confirmation step (the unit-tested helpers + the fake-service test don't depend on the live engines; only the DB-backed `_ranked` does, which the implementer validates). ✔

**Type consistency:** `_norm_z`/`_norm_delta`/`_overall_stats`/`_to_overall_row`/`_LENS_META` used identically across tasks; `OverallLeaderRow`/`LeadersOverallResponse` field names match service + tests; NYY team_id=147 matches `_STATSAPI_TEAM_IDS`; the uniform `_value` column is produced by every `_ranked` branch and consumed by the one mapper. ✔
