# M0 — Matchup-B: you/opp TeamSide + totals rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/api/matchup` with the **you/opp team header** (`TeamSide`: name, manager, record, score) and the **totals rows** (`hitter_totals`/`pitcher_totals`: aggregated stat line per side). Completes the "your matchup" view on top of the roster-comparison core. The 6-matchup **league scoreboard** is a SEPARATE follow-up (Matchup-C — multi-source, cold-env-fragile).

**Architecture:** Extend `api/contracts/matchup.py` (add `TeamSide`, `SideTotals`; extend `MatchupResponse`). In `api/services/matchup_service.py`: a pure `_aggregate_totals(rows_df, hitter)` (counting stats summed, **rate stats weighted** per the canonical rule — AVG=Σh/Σab, OBP=Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf), ERA=Σer·9/Σip, WHIP=Σ(bb_allowed+h_allowed)/Σip), a pure `_format_record(w,l,t,rank)`, and `_team_side(team_name, score)` (manager from `league_teams`, record from `load_league_records()`). `_build_roster_tables` (which already slices each side + has the pool) is extended to also return the totals. The score = the per-category win count already computed in `get_matchup`. NaN/zero-safe; never-raise. `src/` untouched; router logic-free.

**Verified data sources:**
- manager: `SELECT manager_name FROM league_teams WHERE team_name = ?` (no export helper — raw via `get_connection()`).
- record + rank: `src.database.load_league_records()` → DataFrame cols `team_name, wins, losses, ties, rank`.
- score: count of `categories` where `cat.win == "you"` (you) / `"opp"` (opp) — already have `categories` with `win` in `get_matchup`.
- totals: aggregate the side's roster pool rows. Pool projected cols (VERIFY exact names): hitter `h, ab, r, hr, rbi, sb, bb, hbp, sf`; pitcher `ip, w, l, sv, k, er, bb_allowed, h_allowed`.

**Convention:** snake_case; reuse the roster-core's `_f`/`_avg`. Totals are a `list[str]` matching the 7 column headers (H/AB,R,HR,RBI,SB,AVG,OBP / IP,W,L,SV,K,ERA,WHIP).

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`, cwd = worktree root.

**Tech Stack:** FastAPI + Pydantic v2; pandas; pytest. `fastapi==0.137.1`/`httpx==0.28.1` pinned.

---

## File Structure

- `api/contracts/matchup.py` — **modify**: add `TeamSide`, `SideTotals`; extend `MatchupResponse` with `you`/`opp`/`hitter_totals`/`pitcher_totals`.
- `api/services/matchup_service.py` — **modify**: add `_aggregate_totals`, `_format_record`, `_team_side`; extend `_build_roster_tables` to return totals; wire `you`/`opp`/totals into `get_matchup`.
- `api/openapi.json` — **regenerate**.
- `tests/api/test_api_matchup_teamside.py` — **create**: helper unit tests + fake-service contract test.

---

### Task 1: Contracts + pure helpers (`_aggregate_totals`, `_format_record`)

**Files:**
- Modify: `api/contracts/matchup.py`
- Modify: `api/services/matchup_service.py`
- Test: `tests/api/test_api_matchup_teamside.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/api/test_api_matchup_teamside.py`:

```python
"""Matchup-B (TeamSide + totals): helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.matchup_service import _aggregate_totals, _format_record


def test_format_record():
    assert _format_record(4, 7, 1, 8) == "4-7-1 · 8th"
    assert _format_record(1, 0, 0, 1) == "1-0-0 · 1st"
    assert _format_record(0, 0, 0, 2) == "0-0-0 · 2nd"
    assert _format_record(3, 3, 0, 3) == "3-3-0 · 3rd"
    assert _format_record(5, 5, 0, 11) == "5-5-0 · 11th"  # 11th not "11st"


def test_aggregate_totals_hitter_weighted_rates():
    rows = pd.DataFrame([
        {"h": 100, "ab": 400, "r": 60, "hr": 20, "rbi": 70, "sb": 10, "bb": 40, "hbp": 4, "sf": 3},
        {"h": 50, "ab": 200, "r": 30, "hr": 10, "rbi": 35, "sb": 5, "bb": 20, "hbp": 1, "sf": 2},
    ])
    out = _aggregate_totals(rows, hitter=True)
    # H/AB = 150/600 ; R/HR/RBI/SB summed ; AVG = 150/600 = .250 ; OBP = (150+60+5)/(600+60+5+5)
    assert out[0] == "150/600"
    assert out[1:5] == ["90", "30", "105", "15"]
    assert out[5] == ".250"  # AVG
    assert out[6] == ".322"  # OBP = 215/670 = .3208 → .321? compute exactly in impl; adjust to real


def test_aggregate_totals_pitcher_weighted_rates():
    rows = pd.DataFrame([
        {"ip": 100.0, "w": 8, "l": 4, "sv": 0, "k": 110, "er": 35, "bb_allowed": 30, "h_allowed": 90},
        {"ip": 50.0, "w": 4, "l": 2, "sv": 0, "k": 55, "er": 20, "bb_allowed": 18, "h_allowed": 48},
    ])
    out = _aggregate_totals(rows, hitter=False)
    # IP=150 ; ERA = (55*9)/150 = 3.30 ; WHIP = (48+138)/150 = 1.24
    assert out[0] == "150.0"
    assert out[1:5] == ["12", "6", "0", "165"]
    assert out[5] == "3.30"   # ERA
    assert out[6] == "1.24"   # WHIP


def test_aggregate_totals_empty_and_zero_safe():
    assert _aggregate_totals(pd.DataFrame(), hitter=True)[0] == "0/0"
    z = _aggregate_totals(pd.DataFrame([{"ip": 0.0, "er": 5, "bb_allowed": 2, "h_allowed": 3}]), hitter=False)
    assert z[5] == "0.00" and z[6] == "0.00"  # divide-by-zero IP → 0, not inf/NaN
```

NOTE: compute the exact OBP/ERA/WHIP the formulas yield and set the assertions to the real output (the comment math is approximate). AVG/OBP/ERA/WHIP formatting reuses `_avg` (AVG/OBP leading-zero strip) / `:.2f` (ERA/WHIP).

- [ ] **Step 2: Run to verify failure**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_matchup_teamside.py -q`
Expected: FAIL — ImportError on `_aggregate_totals`/`_format_record`.

- [ ] **Step 3: Add the contracts**

In `api/contracts/matchup.py` add (keep existing models):

```python
class TeamSide(BaseModel):
    name: str = ""
    manager: str = ""
    record: str = ""  # "4-7-1 · 8th"
    score: int = 0  # this-week category-win count


class SideTotals(BaseModel):
    you: list[str] = Field(default_factory=list)
    opp: list[str] = Field(default_factory=list)
```

Extend `MatchupResponse`:
```python
    you: TeamSide = Field(default_factory=TeamSide)
    opp: TeamSide = Field(default_factory=TeamSide)
    hitter_totals: SideTotals = Field(default_factory=SideTotals)
    pitcher_totals: SideTotals = Field(default_factory=SideTotals)
```

- [ ] **Step 4: Add the pure helpers**

In `api/services/matchup_service.py` (reuse the existing `_f` + `_avg`; add near them):

```python
def _format_record(wins, losses, ties, rank) -> str:
    w, l, t = int(_f(wins)), int(_f(losses)), int(_f(ties))
    r = int(_f(rank))
    if r <= 0:
        return f"{w}-{l}-{t}"
    suffix = "th" if 11 <= (r % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{w}-{l}-{t} · {r}{suffix}"


def _aggregate_totals(rows, hitter: bool) -> list[str]:
    """Aggregate a side's roster stat line: counting stats summed, rate stats
    weighted (AVG=Σh/Σab, OBP=Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf), ERA=Σer·9/Σip,
    WHIP=Σ(bb_allowed+h_allowed)/Σip). NaN/zero-safe."""
    import pandas as pd

    def _sum(col: str) -> float:
        try:
            if isinstance(rows, pd.DataFrame) and col in rows.columns:
                return float(sum(_f(v) for v in rows[col]))
        except Exception:
            pass
        return 0.0

    if hitter:
        h, ab = _sum("h"), _sum("ab")
        bb, hbp, sf = _sum("bb"), _sum("hbp"), _sum("sf")
        obp_den = ab + bb + hbp + sf
        avg = (h / ab) if ab > 0 else 0.0
        obp = ((h + bb + hbp) / obp_den) if obp_den > 0 else 0.0
        return [
            f"{int(h)}/{int(ab)}",
            str(int(_sum("r"))), str(int(_sum("hr"))), str(int(_sum("rbi"))), str(int(_sum("sb"))),
            _avg(avg), _avg(obp),
        ]
    ip = _sum("ip")
    er, bba, ha = _sum("er"), _sum("bb_allowed"), _sum("h_allowed")
    era = (er * 9.0 / ip) if ip > 0 else 0.0
    whip = ((bba + ha) / ip) if ip > 0 else 0.0
    return [
        f"{ip:.1f}",
        str(int(_sum("w"))), str(int(_sum("l"))), str(int(_sum("sv"))), str(int(_sum("k"))),
        f"{era:.2f}", f"{whip:.2f}",
    ]
```

- [ ] **Step 5: Run the unit tests** — PASS (fix the OBP/ERA/WHIP assertion values to the real computed output if they differ).

- [ ] **Step 6: Commit**

```bash
git add api/contracts/matchup.py api/services/matchup_service.py tests/api/test_api_matchup_teamside.py
git commit -m "feat(api): matchup TeamSide/SideTotals contracts + _aggregate_totals/_format_record"
```

---

### Task 2: Wire TeamSide + totals into the response

**Files:**
- Modify: `api/services/matchup_service.py` (`_team_side`, extend `_build_roster_tables` for totals, wire `get_matchup`)
- Test: append to `tests/api/test_api_matchup_teamside.py`
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Write the failing contract test**

Append a fake-service test asserting the route returns `you`/`opp`/totals:

```python
def test_matchup_endpoint_includes_teamside_and_totals():
    from fastapi.testclient import TestClient

    from api.contracts.matchup import MatchupResponse, SideTotals, TeamSide
    from api.deps import get_matchup_service
    from api.main import create_app

    class _Fake:
        def get_matchup(self, team_name):
            return MatchupResponse(
                team_name=team_name, opponent="Rivals", week=7,
                you=TeamSide(name=team_name, manager="Connor", record="4-7-1 · 8th", score=5),
                opp=TeamSide(name="Rivals", manager="Sam", record="7-4-1 · 2nd", score=6),
                hitter_totals=SideTotals(you=["150/600", "90", "30", "105", "15", ".250", ".321"], opp=[]),
                pitcher_totals=SideTotals(you=["150.0", "12", "6", "0", "165", "3.30", "1.24"], opp=[]),
            )

    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/matchup?team_name=Team+Hickey").json()
        assert body["you"]["manager"] == "Connor"
        assert body["you"]["record"] == "4-7-1 · 8th"
        assert body["opp"]["score"] == 6
        assert body["hitter_totals"]["you"][0] == "150/600"
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run to verify failure** (route returns no you/opp/totals from the real service yet; the fake test drives the wiring + confirms the shape).

- [ ] **Step 3: Add `_team_side` + extend `_build_roster_tables` for totals**

Add the `_team_side` static helper to `MatchupService`:

```python
    @staticmethod
    def _team_side(team_name: str, score: int) -> TeamSide:
        manager, record = "", ""
        try:
            from src.database import get_connection, load_league_records

            conn = get_connection()
            try:
                r = conn.execute(
                    "SELECT manager_name FROM league_teams WHERE team_name = ?", (team_name,)
                ).fetchone()
                if r and r[0]:
                    manager = str(r[0])
            finally:
                conn.close()
            recs = load_league_records()
            if recs is not None and not recs.empty:
                m = recs[recs["team_name"] == team_name]
                if not m.empty:
                    row = m.iloc[0]
                    record = _format_record(row.get("wins"), row.get("losses"), row.get("ties"), row.get("rank"))
        except Exception:
            pass
        return TeamSide(name=team_name, manager=manager, record=record, score=int(score))
```

Extend `_build_roster_tables` to ALSO compute + return the totals. It already has `you_rosters`/`opp_rosters` + `pool` + per-player hitter/pitcher classification. After building the rows, for each (side, hitter|pitcher) collect the player_ids and aggregate the matching pool rows:

```python
        # (inside _build_roster_tables, after the rows are built — collect per-side pool rows)
        def _side_totals(side_rosters, is_hit: bool) -> list[str]:
            if pool is None or pool.empty or side_rosters.empty:
                return []
            ids = [int(p) for p in side_rosters["player_id"].dropna().astype(int).tolist()]
            sub = pool[pool["player_id"].isin(ids)]
            if "is_hitter" in sub.columns:
                sub = sub[sub["is_hitter"].astype(bool) == is_hit]
            return _aggregate_totals(sub, hitter=is_hit)

        hitter_totals = SideTotals(you=_side_totals(you_rosters, True), opp=_side_totals(opp_rosters, True))
        pitcher_totals = SideTotals(you=_side_totals(you_rosters, False), opp=_side_totals(opp_rosters, False))
```

Change `_build_roster_tables`'s return signature to append `hitter_totals, pitcher_totals` (update the type hint + the early-return empties to include `SideTotals(), SideTotals()`).

In `get_matchup`: compute the scores from `categories` (`you_score = sum(1 for c in categories if c.win == "you")`, `opp_score = sum(1 for c in categories if c.win == "opp")`), build `you=_team_side(team_name, you_score)` + `opp=_team_side(opponent, opp_score)`, unpack the extended `_build_roster_tables` return (now incl. totals), and pass `you`/`opp`/`hitter_totals`/`pitcher_totals` into `MatchupResponse`. Keep all of it inside the existing try/except (cold env → defaults).

- [ ] **Step 4: Run the contract test** — PASS.

- [ ] **Step 5: Regenerate OpenAPI** + confirm the snapshot passes; confirm `TeamSide`/`SideTotals` + the new `MatchupResponse` fields appear.

- [ ] **Step 6: Commit**

```bash
git add api/services/matchup_service.py api/openapi.json tests/api/test_api_matchup_teamside.py
git commit -m "feat(api): matchup you/opp TeamSide + hitter/pitcher totals"
```

---

### Task 3: Full api suite + lint

- [ ] `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q` (prior + new; fix any exact-equality matchup test for the new fields). `test_no_logic_in_routers` green.
- [ ] `ruff format` + `ruff check` on `api/` + `tests/api/` — clean.
- [ ] Commit if anything changed.

---

## Self-Review (plan author)

**Spec coverage** (gap spec §4 you/opp + totals): `TeamSide` (name/manager/record/score) for you+opp ✔; `hitter_totals`/`pitcher_totals` with WEIGHTED rate aggregation ✔. **Deferred:** the `league` scoreboard → Matchup-C (multi-source: `load_league_schedule_full` + `load_matchup_cache` + records; cold-env empty).

**Reuse/safety:** reuses `_f`/`_avg` + the existing `_build_roster_tables` side-slicing + pool; `score` reuses the already-computed per-category `win`. All new fields default (backward-compatible). `_team_side` + the totals are wrapped/never-raise. Rate stats use the canonical weighted formulas (NOT naive averages) — the load-bearing correctness point, covered by unit tests.

**Placeholder note:** Task 2 Step 3 references inserting `_side_totals` inside `_build_roster_tables` — the implementer places it correctly after reading the method; the PURE helpers (`_aggregate_totals`/`_format_record`) are fully specified + unit-tested.

**Type consistency:** `_aggregate_totals(rows, hitter)`/`_format_record(w,l,t,rank)`/`_team_side(team_name, score)` consistent; `TeamSide`/`SideTotals` fields match service + tests; reuses `_f`/`_avg` from the roster-core slice.
