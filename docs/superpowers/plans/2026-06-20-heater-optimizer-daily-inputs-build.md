# Optimizer Daily — Richer Inputs Build Plan (M1 enrichment)

> REQUIRED SUB-SKILL: executing-plans. Checkbox steps. TDD.

**Goal:** Feed `build_daily_dcv_table` the 4 inputs the M0 daily slice deferred — `park_factors`, `team_strength`, `confirmed_lineups`, `recent_form` — so daily DCV scoring uses park/platoon/form/lineup adjustments instead of only matchup + schedule. `POST /api/lineup/optimize` (mode=daily).

**Architecture:** Build the 4 inputs DIRECTLY from the engine helpers (lighter than the full `build_optimizer_context`), each graceful (failure → empty), and spread them into `pipeline.optimize(...)`. Additive; `src/` untouched; the standard-mode path unchanged; the daily path already degrades to empty in a cold env (so on current infra nothing changes until live data exists).

**Verified helper signatures (do NOT guess):**
- `src.database.load_park_factors(fallback=None) -> dict[team_code, factor_hitting]`
- `src.game_day.get_todays_lineups(schedule, deadline=None) -> dict`
- `src.game_day.get_team_strength(team_abbr) -> dict` (per team)
- `src.game_day.get_player_recent_form_cached(mlb_id) -> dict` (per player; the optimizer ctx keys `recent_form` by **player_id**, shared_data_layer.py:914)
- `pipeline.optimize(**kwargs)` forwards `park_factors`/`confirmed_lineups`/`recent_form`/`team_strength` to `build_daily_dcv_table` (guarded by `test_optimizer_pipeline_forwards_context.py`).

---

## Task 1: input builders (TDD)

**Files:** `api/services/lineup_service.py`, `tests/api/test_api_lineup_daily_inputs.py`

- [ ] **Step 1: failing test** — unit-test the builders with monkeypatched-at-source helpers (DB-free)

```python
from api.services.lineup_service import LineupService


def test_team_strength_built_from_schedule(monkeypatch):
    import src.game_day as gd
    import src.valuation as val
    monkeypatch.setattr(val, "team_name_to_abbr", lambda x: {"New York Yankees": "NYY", "Boston Red Sox": "BOS"}.get(x, x))
    monkeypatch.setattr(val, "canonicalize_team", lambda x: x)
    monkeypatch.setattr(gd, "get_team_strength", lambda a: {"wrc_plus": 110} if a == "NYY" else {"wrc_plus": 95})
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox"}]
    ts = LineupService()._team_strength(sched)
    assert ts.get("NYY") == {"wrc_plus": 110}
    assert ts.get("BOS") == {"wrc_plus": 95}


def test_recent_form_keyed_by_player_id(monkeypatch):
    import pandas as pd
    import src.game_day as gd
    monkeypatch.setattr(gd, "get_player_recent_form_cached", lambda mlb: {"form": "hot"} if mlb == 545361 else {})
    roster = pd.DataFrame([{"player_id": 7}, {"player_id": 9}])
    pool = pd.DataFrame([{"player_id": 7, "mlb_id": 545361}, {"player_id": 9, "mlb_id": 0}])
    rf = LineupService()._recent_form(roster, pool)
    assert rf == {7: {"form": "hot"}}  # keyed by player_id; pid 9 (no mlb_id) skipped


def test_daily_inputs_has_all_four_keys(monkeypatch):
    import src.database as db
    import src.game_day as gd
    monkeypatch.setattr(db, "load_park_factors", lambda fallback=None: {"NYY": 1.05})
    monkeypatch.setattr(gd, "get_todays_lineups", lambda sched, deadline=None: {"NYY": [1, 2, 3]})
    monkeypatch.setattr(gd, "get_team_strength", lambda a: {})
    monkeypatch.setattr(gd, "get_player_recent_form_cached", lambda mlb: {})
    out = LineupService()._daily_inputs(None, None, [])
    assert set(out) == {"park_factors", "team_strength", "confirmed_lineups", "recent_form"}
    assert out["park_factors"] == {"NYY": 1.05}
    assert out["confirmed_lineups"] == {"NYY": [1, 2, 3]}


def test_daily_inputs_graceful_on_failure(monkeypatch):
    import src.database as db
    monkeypatch.setattr(db, "load_park_factors", lambda fallback=None: (_ for _ in ()).throw(RuntimeError("db")))
    out = LineupService()._daily_inputs(None, None, [])
    assert out["park_factors"] == {}  # never raises
```

- [ ] **Step 2: run → FAIL**

- [ ] **Step 3: implement** — add these methods to `LineupService`:

```python
    def _daily_inputs(self, roster, pool, schedule) -> dict:
        """Build the richer daily-DCV inputs directly from the engine helpers — each
        graceful (failure → empty), so the daily path never breaks on missing live data."""
        park_factors: dict = {}
        try:
            from src.database import load_park_factors

            park_factors = load_park_factors() or {}
        except Exception:
            park_factors = {}
        confirmed_lineups: dict = {}
        try:
            from src.game_day import get_todays_lineups

            confirmed_lineups = get_todays_lineups(schedule) or {}
        except Exception:
            confirmed_lineups = {}
        return {
            "park_factors": park_factors,
            "confirmed_lineups": confirmed_lineups,
            "team_strength": self._team_strength(schedule),
            "recent_form": self._recent_form(roster, pool),
        }

    @staticmethod
    def _team_strength(schedule) -> dict:
        """{abbr: strength} for the teams playing today (from the schedule's team names)."""
        out: dict = {}
        try:
            from src.game_day import get_team_strength
            from src.valuation import canonicalize_team, team_name_to_abbr

            abbrs: set[str] = set()
            for g in schedule or []:
                if not isinstance(g, dict):
                    continue
                for k in ("home_name", "away_name", "home_team", "away_team"):
                    v = g.get(k)
                    if v:
                        abbr = canonicalize_team(team_name_to_abbr(str(v))).upper().strip()
                        if abbr:
                            abbrs.add(abbr)
            for a in abbrs:
                try:
                    ts = get_team_strength(a)
                    if ts:
                        out[a] = ts
                except Exception:
                    continue
        except Exception:
            return {}
        return out

    def _recent_form(self, roster, pool) -> dict:
        """{player_id: form} for the roster's players (form fetched by mlb_id, keyed by
        player_id to match the optimizer ctx's recent_form shape)."""
        import pandas as pd

        out: dict = {}
        try:
            from src.game_day import get_player_recent_form_cached

            if not isinstance(roster, pd.DataFrame) or roster.empty or "player_id" not in roster.columns:
                return {}
            mlb_by_pid: dict[int, int] = {}
            if (
                isinstance(pool, pd.DataFrame)
                and not pool.empty
                and "player_id" in pool.columns
                and "mlb_id" in pool.columns
            ):
                for pr in pool.to_dict("records"):
                    try:
                        mlb_by_pid[int(pr.get("player_id", 0) or 0)] = int(pr.get("mlb_id", 0) or 0)
                    except (TypeError, ValueError):
                        continue
            for r in roster.to_dict("records"):
                try:
                    pid = int(r.get("player_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                mlb = mlb_by_pid.get(pid, 0)
                if not pid or not mlb:
                    continue
                try:
                    form = get_player_recent_form_cached(mlb)
                    if form:
                        out[pid] = form
                except Exception:
                    continue
        except Exception:
            return {}
        return out
```

- [ ] **Step 4: run → PASS**

## Task 2: forward the inputs in `_optimize_daily`

- [ ] **Step 1: failing test** — a fake pipeline captures the kwargs

```python
def test_daily_forwards_rich_inputs(monkeypatch):
    import pandas as pd
    import src.game_day as gd
    import src.valuation as val
    from api.services import lineup_service as ls

    captured = {}

    class _FakePipeline:
        def __init__(self, *a, **k):
            pass

        def optimize(self, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr("src.optimizer.pipeline.LineupOptimizerPipeline", _FakePipeline)
    monkeypatch.setattr("src.valuation.LeagueConfig", lambda: object())
    monkeypatch.setattr(gd, "get_target_game_date", lambda: "2026-04-05")

    class _YDS:
        def get_rosters(self):
            return pd.DataFrame([{"player_id": 1}])

        def get_matchup(self):
            return None

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _YDS())
    monkeypatch.setattr(LineupService, "_schedule_today", staticmethod(lambda d: []))
    monkeypatch.setattr(LineupService, "_load_pool", staticmethod(lambda: None))
    monkeypatch.setattr(LineupService, "_daily_inputs", lambda self, r, p, s: {
        "park_factors": {"NYY": 1.05}, "team_strength": {"NYY": {}}, "confirmed_lineups": {"NYY": []}, "recent_form": {1: {}},
    })
    LineupService().optimize("Team Hickey", mode="daily")
    assert captured.get("park_factors") == {"NYY": 1.05}
    assert captured.get("confirmed_lineups") == {"NYY": []}
    assert captured.get("recent_form") == {1: {}}
    assert captured.get("team_strength") == {"NYY": {}}
```

- [ ] **Step 2: run → FAIL** (optimize currently passes only matchup+schedule_today)

- [ ] **Step 3: implement** — in `_optimize_daily`, move `pool = self._load_pool()` ABOVE the optimize call, build inputs, and spread them:

```python
            pool = self._load_pool()
            inputs = self._daily_inputs(roster, pool, schedule)
            pipeline = LineupOptimizerPipeline(roster, mode="daily", config=LeagueConfig())
            result = pipeline.optimize(
                matchup=matchup,
                schedule_today=schedule,
                park_factors=inputs["park_factors"],
                confirmed_lineups=inputs["confirmed_lineups"],
                recent_form=inputs["recent_form"],
                team_strength=inputs["team_strength"],
            )
            result = result if isinstance(result, dict) else {}
            slots, bench = self._daily_slots(
                result.get("daily_dcv"), result.get("daily_lineup"), roster, pool, schedule
            )
```
(remove the now-duplicate `pool = self._load_pool()` that was below the optimize call.)

- [ ] **Step 4: run → PASS**; full api suite + lint + openapi (no contract change → no diff).

- [ ] **Step 5: commit**

```bash
git add api/services/lineup_service.py tests/api/test_api_lineup_daily_inputs.py docs/superpowers/plans/2026-06-20-heater-optimizer-daily-inputs-build.md
git commit -m "feat(api): feed daily optimizer the richer DCV inputs (park/team-strength/lineups/form) — M1 enrich"
```

## Review gate
Dispatch `pr-review-toolkit:code-reviewer` (full engine context) — verify the helper signatures, the recent_form player_id keying matches `build_daily_dcv_table`'s expectation, the team-name→abbr canonicalization, graceful degradation, and that the standard path + the `_daily_slots` join are unaffected. Apply findings, then push.
```
