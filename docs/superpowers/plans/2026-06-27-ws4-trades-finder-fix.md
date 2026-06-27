# WS4 — Trades: Fix Finder + Enrich + Verify — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the empty Trade Finder (root cause: a raw dict lookup in `trade_finder_service` that fails on emoji/whitespace Yahoo team-name keys, compounded by passing `all_team_totals=None` which makes the engine early-return `[]`), enrich the `TradeSuggestion` contract with engine fields the service currently drops (`grade`, `partner_record`, `category_impacts`), wire the frontend `TradeCard` to render the real grade/impact, and verify Compare + Build end-to-end.

**Architecture:** The proven HEATER API slice pattern — contract (`api/contracts/trade_finder.py`) → service seam (`api/services/trade_finder_service.py`, the ONE place importing `src/`) → thin logic-free router (`api/routers/trade_finder.py`, already correct) → DI provider (`api/deps.py`, already wired) → fake-service DI test (`tests/api/test_api_trade_finder.py` via `app.dependency_overrides`) → regen `api/openapi.json` (`python scripts/export_openapi.py`, snapshot-guarded) → frontend adapter (`web/src/lib/api/adapters.ts`) + `web/src/lib/trades-data.ts` + `web/src/app/trades/page.tsx`. The `src/` engines (`find_trade_opportunities`, `grade_trade`, `_roster_category_totals`, `load_league_records`, `get_all_team_totals`) are UNCHANGED — this is wiring + a service-side cheap per-category diff. Shared contracts are reused, never redefined: `PlayerRef`/`StatItem` (`api/contracts/common.py`), `CategoryImpact` (`api/contracts/trade.py`), `make_player_ref`/`player_ref_from_pool` (`api/services/player_ref.py`).

**Tech Stack:** Python 3.12/3.14 + FastAPI `==0.137.1` + httpx `==0.28.1` (PINNED — openapi snapshot guard) + pydantic. Frontend: Next.js 16 + React 19 + TypeScript + pnpm. Windows PowerShell. Backend tests: `python -m pytest tests/api/test_api_trade_finder.py -v`. Frontend gate (from `web/`): `pnpm build` + `pnpm exec tsc --noEmit` + `pnpm run lint`.

**Key facts established by reading the real code (do not re-derive):**
- `api/routers/trade_finder.py` ALREADY uses `Depends(require_viewer_context)` + `resolve_required_team(ctx, team_name)`. BUT `resolve_required_team` only performs the emoji/whitespace reconcile when Clerk is configured; on the free beta (`CLERK_ISSUER` unset → `require_viewer_context` returns an empty `ViewerContext`), `effective_team(fallback)` returns the **raw client `team_name`** unchanged. So the router does not fix the mismatch — **the bug lives in the service**.
- `api/services/trade_finder_service.py:34` does `user_roster_ids = league_rosters.get(team_name, [])` — a RAW dict lookup. Yahoo roster keys carry emoji/whitespace ("🏆 Team Hickey"); the client `team_name` ("Team Hickey") won't match → `user_roster_ids = []`.
- The service also passes `all_team_totals=None` (line ~41). In `src/trade_finder.py:1895`, `find_trade_opportunities` does `if not league_rosters or not all_team_totals: return []`. So **even with a correct roster, `all_team_totals=None` alone forces an empty result.** Both must be fixed.
- `src/trade_finder.py:2083-2084`: `find_trade_opportunities` ALREADY sets `trade["grade"] = grade_trade(trade["user_sgp_gain"])` on every output dict. The service just doesn't read it. The grade fn is `src/engine/output/trade_evaluator.py:796 grade_trade(surplus_sgp)`. **No engine change needed for `grade`.**
- Engine output dicts (from `scan_1_for_1`/`scan_2_for_1`) carry: `giving_ids`, `receiving_ids`, `giving_names`, `receiving_names`, `user_sgp_gain`, `opponent_sgp_gain`, `acceptance_probability`, `acceptance_label`, `composite_score`, `trade_type`, `adp_fairness`, `grade`, `opponent_team`, `complementarity`, plus a scalar `category_fit`. **They do NOT carry a per-category SGP-delta breakdown** — that must be computed in the service (cheap roster-totals diff, ≤10 cards).
- `load_league_records()` (`src/database.py:3045`) → DataFrame with `team_name, wins, losses, ties, win_pct, rank`. Use for `partner_record`.
- `get_all_team_totals(...)` (`src/standings_utils.py:60`) → `{team_name: {cat: total}}` (Yahoo-standings-first, projection fallback, module-cached w/ 5-min TTL).
- `api/tenancy.py:28 normalize_team_name(name)` + `reconcile_team_name(assigned, roster_names)` already exist (emoji/whitespace-tolerant). The service will import `normalize_team_name` (api-internal, NOT `src/` — discipline preserved) to reconcile the requested team against the real roster keys.
- Pro-gate is DORMANT on the free beta: `api/gating.py require_pro` returns immediately when `not stripe_enabled()` (= `billing_env_configured()` False → Stripe unset). The finder route keeps `Depends(require_pro)` (harmless no-op now; live later). Compare (`api/routers/compare.py`) is `require_login`-gated only (also dormant when Clerk off). `trade/evaluate` keeps `enable_mc=false` default (frontend `web/src/lib/trades-data.ts:209` sends `enable_mc: false`).
- Openapi regen: `python scripts/export_openapi.py`. Snapshot guard: `tests/api/test_openapi_contract.py`. Frontend type regen (integrator): `pnpm gen:api`.
- **DB context:** the worktree/CI `draft_tool.db` is EMPTY — every backend test in this plan MUST be DB-free (monkeypatch the service's `src.*` imports / inject a fake service). Live reproduction + verification use the **live Railway API** or the **main-checkout 26 MB DB**, never the empty worktree DB.

---

## Task 1 — DEBUG: root-cause + document the empty Finder (investigation, no code change)

Reproduce and confirm the two compounding causes before touching code, per systematic-debugging. This task writes findings into the plan's verification log (below) and a short docstring-ready note; it does NOT change behavior.

**Files:**
- Read-only: `api/services/trade_finder_service.py` (lines 28-44), `src/trade_finder.py` (lines 1892-1896, 2079-2156), `api/tenancy.py` (lines 62-87).
- Repro scratch script (throwaway): `C:\Users\conno\AppData\Local\Temp\claude\C--Users-conno-Code-HEATER-v1-0-1\d09d33e5-1525-42ac-a260-172c476ee378\scratchpad\repro_finder.py`.

Steps:
- [ ] Re-read the service + engine guard. Confirm in writing the two failure modes:
  - (a) **Name mismatch:** `league_rosters.get(team_name, [])` returns `[]` when `team_name` lacks the emoji/whitespace the Yahoo roster key carries.
  - (b) **`all_team_totals=None`:** `find_trade_opportunities` returns `[]` at `src/trade_finder.py:1895` (`if not league_rosters or not all_team_totals: return []`) regardless of roster.
- [ ] Prove (a) in isolation WITHOUT the DB — write the scratch repro that constructs a fake `league_rosters = {"🏆 Team Hickey": [1, 2, 3]}` and shows `league_rosters.get("Team Hickey", [])` → `[]` while a `normalize_team_name`-keyed reconcile resolves it to `[1, 2, 3]`:
  ```python
  # scratchpad/repro_finder.py
  from api.tenancy import normalize_team_name

  league_rosters = {"🏆 Team Hickey": [1, 2, 3], "Over the Rembow": [4, 5]}
  requested = "Team Hickey"

  raw = league_rosters.get(requested, [])
  print("raw .get():", raw)  # [] — the bug

  norm = {normalize_team_name(k): k for k in league_rosters}
  resolved_key = norm.get(normalize_team_name(requested))
  print("reconciled key:", resolved_key, "->", league_rosters.get(resolved_key, []))  # 🏆 Team Hickey -> [1, 2, 3]
  assert raw == [] and league_rosters.get(resolved_key, []) == [1, 2, 3]
  ```
- [ ] Run it: `python "C:\Users\conno\AppData\Local\Temp\claude\C--Users-conno-Code-HEATER-v1-0-1\d09d33e5-1525-42ac-a260-172c476ee378\scratchpad\repro_finder.py"` — EXPECT it to print `raw .get(): []` then `🏆 Team Hickey -> [1, 2, 3]` and exit 0.
- [ ] (Live confirmation, optional but preferred) Hit the live API and confirm an empty result today: `GET https://celebrated-respect-production.up.railway.app/api/trade-finder?team_name=Team+Hickey&limit=10` → expect `{"team_name": "...", "suggestions": []}` (note: on the live API the Pro-gate is dormant + Clerk-gating may 401/return depending on token; if you can't reach it, the scratch repro + main-checkout DB run in Task 3's live-verify is sufficient). Record the observed status/body in the Verification Log below.
- [ ] Write a 4-line root-cause summary into the **Verification Log** section at the bottom of this file (no code commit for this task — investigation only). Confirm the fix plan: (1) reconcile the team name against actual roster keys in the service, (2) compute + pass `all_team_totals`.

_No commit — investigation task. The fix lands in Task 2._

---

## Task 2 — FIX: resolve the user's team (emoji/whitespace-tolerant) + pass `all_team_totals`

Fix the service so the user's roster resolves against the real Yahoo keys and `find_trade_opportunities` receives a non-None `all_team_totals`. DB-free test asserts both: the reconcile path maps a mismatched name to the right roster, and `all_team_totals` is computed + passed.

**Files:**
- `api/services/trade_finder_service.py` (rewrite `get_suggestions` body, lines 17-51; add a `_resolve_user_roster_ids` helper).
- `tests/api/test_api_trade_finder.py` (add DB-free tests that monkeypatch the service's `src.*` call sites).

Steps:

- [ ] **Write the failing tests first.** Append to `tests/api/test_api_trade_finder.py`. These inject fakes for the four `src.*` symbols the service imports inside `get_suggestions` (`load_player_pool`, `find_trade_opportunities`, `get_yahoo_data_service`, `get_all_team_totals`) by monkeypatching them where the service imports them (the service does local `from src... import ...` inside the method, so patch `src.database.load_player_pool`, `src.trade_finder.find_trade_opportunities`, `src.yahoo_data_service.get_yahoo_data_service`, `src.standings_utils.get_all_team_totals`). Capture the kwargs passed to the fake `find_trade_opportunities`:

  ```python
  import pandas as pd

  from api.services.trade_finder_service import TradeFinderService


  def _fake_pool():
      # minimal pool: ids 1,2,3 (user) + 4,5 (rival), with the cols the refs/diff need
      return pd.DataFrame(
          {
              "player_id": [1, 2, 3, 4, 5],
              "player_name": ["You A", "You B", "You C", "Riv A", "Riv B"],
              "positions": ["OF", "SP", "2B", "SS", "RP"],
              "mlb_id": [101, 102, 103, 104, 105],
              "team": ["NYM", "SEA", "KC", "CIN", "CLE"],
              "is_hitter": [1, 0, 1, 1, 0],
          }
      )


  class _FakeYds:
      def get_rosters(self):
          # emoji/whitespace team-name keys — the exact mismatch the bug hits
          return pd.DataFrame(
              {
                  "team_name": ["🏆 Team Hickey", "🏆 Team Hickey", "🏆 Team Hickey", "Over the Rembow", "Over the Rembow"],
                  "player_id": [1, 2, 3, 4, 5],
              }
          )


  def test_service_reconciles_emoji_team_name_and_passes_all_team_totals(monkeypatch):
      """The bug: raw .get('Team Hickey') misses the '🏆 Team Hickey' roster key →
      empty user_roster_ids; and all_team_totals=None forces find_trade_opportunities
      to early-return []. Assert the service reconciles the name AND passes non-None
      all_team_totals."""
      captured = {}

      def _fake_find(**kwargs):
          captured.update(kwargs)
          return [
              {
                  "giving_ids": [1],
                  "receiving_ids": [4],
                  "opponent_team": "Over the Rembow",
                  "user_sgp_gain": 1.2,
                  "grade": "B+",
                  "rationale": "ok",
              }
          ]

      monkeypatch.setattr("src.database.load_player_pool", _fake_pool)
      monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
      monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"🏆 Team Hickey": {"HR": 100.0}, "Over the Rembow": {"HR": 90.0}})
      monkeypatch.setattr("src.trade_finder.find_trade_opportunities", _fake_find)

      resp = TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)

      # reconciled the mismatched name → the user's real roster ids (NOT [])
      assert captured["user_roster_ids"] == [1, 2, 3]
      # all_team_totals passed through non-None (the second compounding bug)
      assert captured["all_team_totals"] is not None
      assert "Over the Rembow" in captured["all_team_totals"]
      # suggestion surfaced
      assert len(resp.suggestions) == 1
      assert resp.suggestions[0].partner_team == "Over the Rembow"


  def test_service_exact_name_match_still_works(monkeypatch):
      """An exact-match team_name (no emoji) must still resolve (no regression)."""
      captured = {}
      monkeypatch.setattr("src.database.load_player_pool", _fake_pool)

      class _ExactYds:
          def get_rosters(self):
              return pd.DataFrame({"team_name": ["Team Hickey", "Team Hickey", "Rival"], "player_id": [1, 2, 4]})

      monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _ExactYds())
      monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"Team Hickey": {}, "Rival": {}})

      def _fake_find(**kwargs):
          captured.update(kwargs)
          return []

      monkeypatch.setattr("src.trade_finder.find_trade_opportunities", _fake_find)
      TradeFinderService().get_suggestions(team_name="Team Hickey", limit=10)
      assert captured["user_roster_ids"] == [1, 2]


  def test_service_unresolvable_name_returns_empty_not_crash(monkeypatch):
      """A team_name that matches no roster key → empty suggestions, never a 500."""
      monkeypatch.setattr("src.database.load_player_pool", _fake_pool)

      class _Yds:
          def get_rosters(self):
              return pd.DataFrame({"team_name": ["Real Team"], "player_id": [1]})

      monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _Yds())
      monkeypatch.setattr("src.standings_utils.get_all_team_totals", lambda *a, **k: {"Real Team": {}})
      monkeypatch.setattr("src.trade_finder.find_trade_opportunities", lambda **k: [])
      resp = TradeFinderService().get_suggestions(team_name="Nonexistent Team", limit=10)
      assert resp.suggestions == []
  ```

- [ ] Run, EXPECT FAIL (the service still does the raw `.get()` and passes `all_team_totals=None`): `python -m pytest tests/api/test_api_trade_finder.py -v -k "reconciles or exact_name or unresolvable"`.

- [ ] **Minimal implementation.** Edit `api/services/trade_finder_service.py`: add the `normalize_team_name` import (api-internal, allowed), add `_resolve_user_roster_ids`, compute `all_team_totals`, and read the engine `grade` (the contract field lands in Task 3; for now keep mapping minimal but correct):

  ```python
  from api.tenancy import normalize_team_name
  ```

  Replace the body of `get_suggestions` (lines 18-47) with:

  ```python
          try:
              from src.database import load_player_pool
              from src.standings_utils import get_all_team_totals
              from src.trade_finder import find_trade_opportunities
              from src.valuation import LeagueConfig
              from src.yahoo_data_service import get_yahoo_data_service

              pool = load_player_pool()
              if pool is None or pool.empty:
                  return TradeFinderResponse(team_name=team_name)

              yds = get_yahoo_data_service()
              rosters_df = yds.get_rosters()
              league_rosters = self._build_league_rosters(rosters_df)
              if not league_rosters:
                  return TradeFinderResponse(team_name=team_name)

              # Resolve the user's team against the ACTUAL roster keys (Yahoo keys
              # carry emoji/whitespace, e.g. "🏆 Team Hickey"); a raw .get(team_name)
              # missed them → empty roster → zero suggestions. Use the resolved key
              # name downstream so all_team_totals / find_trade_opportunities agree.
              resolved_team, user_roster_ids = self._resolve_user_roster(team_name, league_rosters)
              if not user_roster_ids:
                  return TradeFinderResponse(team_name=team_name)

              # all_team_totals is REQUIRED: find_trade_opportunities early-returns []
              # when it's None (src/trade_finder.py). Compute it (Yahoo-standings-first,
              # projection fallback) and pass it.
              all_team_totals = get_all_team_totals(league_rosters=league_rosters, player_pool=pool)

              raw = find_trade_opportunities(
                  user_roster_ids=user_roster_ids,
                  player_pool=pool,
                  config=LeagueConfig(),
                  all_team_totals=all_team_totals or None,
                  user_team_name=resolved_team,
                  league_rosters=league_rosters,
                  max_results=limit,
              )

              suggestions = self._build_suggestions(raw, pool)
              return TradeFinderResponse(team_name=team_name, suggestions=suggestions)

          except Exception as exc:
              logger.warning("TradeFinderService.get_suggestions failed: %s", exc)
              return TradeFinderResponse(team_name=team_name)
  ```

  Add the resolver helper to the class:

  ```python
      @staticmethod
      def _resolve_user_roster(
          team_name: str, league_rosters: dict[str, list[int]]
      ) -> tuple[str, list[int]]:
          """Map the requested team_name to the EXACT roster key (emoji/whitespace
          tolerant), returning (resolved_key, roster_ids). Exact match wins; else a
          normalized match; else ("", []) so the caller returns empty (never a crash,
          never another team's roster)."""
          if team_name in league_rosters:
              return team_name, league_rosters[team_name]
          target = normalize_team_name(team_name)
          for key, ids in league_rosters.items():
              if normalize_team_name(key) == target:
                  return key, ids
          return "", []
  ```

- [ ] Run, EXPECT PASS: `python -m pytest tests/api/test_api_trade_finder.py -v -k "reconciles or exact_name or unresolvable"`.
- [ ] Run the full finder test file (no regression to the existing contract/DI/405 tests): `python -m pytest tests/api/test_api_trade_finder.py -v`.
- [ ] Run the router-discipline guard (service-only change, but cheap to confirm): `python -m pytest tests/api/test_no_logic_in_routers.py -v`.
- [ ] Commit:
  ```
  git add api/services/trade_finder_service.py tests/api/test_api_trade_finder.py
  git commit -m "fix(trade-finder): resolve emoji/whitespace team name + pass all_team_totals (empty Finder)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

## Task 3 — ENRICH contract: `grade` + `partner_record` (additive)

Add `grade` (read the engine's already-computed `trade['grade']`) and `partner_record` (from `load_league_records`, emoji-tolerant team match) to `TradeSuggestion`. Both additive/optional → backward-compatible.

**Files:**
- `api/contracts/trade_finder.py` (add `grade`, `partner_record` to `TradeSuggestion`).
- `api/services/trade_finder_service.py` (`_build_suggestions` reads `grade`; add `_partner_records` loader + map).
- `tests/api/test_api_trade_finder.py` (assert the new fields populate from the engine dict + records).

Steps:

- [ ] **Failing test first.** Append to `tests/api/test_api_trade_finder.py` — extend the existing `_FakeTradeFinderService` is not enough (the new mapping lives in the REAL service's `_build_suggestions`), so test `_build_suggestions` directly plus a records-monkeypatched end-to-end:

  ```python
  def test_build_suggestions_reads_engine_grade_and_partner_record(monkeypatch):
      """grade comes straight off the engine dict (find_trade_opportunities already
      sets trade['grade']); partner_record comes from load_league_records, matched
      emoji-tolerantly to the opponent_team."""
      svc = TradeFinderService()
      pool = _fake_pool()
      raw = [
          {
              "giving_ids": [1],
              "receiving_ids": [4],
              "opponent_team": "Over the Rembow",
              "user_sgp_gain": 1.2,
              "grade": "B+",
              "rationale": "ok",
          }
      ]
      # records keyed with an emoji variant to prove the tolerant match
      monkeypatch.setattr(
          "src.database.load_league_records",
          lambda: pd.DataFrame(
              {
                  "team_name": ["Over the Rembow"],
                  "wins": [11],
                  "losses": [1],
                  "ties": [0],
                  "rank": [1],
              }
          ),
      )
      out = svc._build_suggestions(raw, pool)
      assert out[0].grade == "B+"
      assert out[0].partner_record == "11-1-0 · 1st"


  def test_build_suggestions_missing_grade_and_records_degrade(monkeypatch):
      """No grade on the dict + no records table → grade '' and partner_record None,
      never a crash."""
      monkeypatch.setattr("src.database.load_league_records", lambda: pd.DataFrame())
      out = TradeFinderService()._build_suggestions(
          [{"giving_ids": [1], "receiving_ids": [4], "opponent_team": "Nobody", "user_sgp_gain": 0.4}],
          _fake_pool(),
      )
      assert out[0].grade == ""
      assert out[0].partner_record is None
  ```

- [ ] Run, EXPECT FAIL (`TradeSuggestion` has no `grade`/`partner_record`; `_build_suggestions` doesn't set them): `python -m pytest tests/api/test_api_trade_finder.py -v -k "engine_grade or missing_grade"`.

- [ ] **Implement the contract.** Edit `api/contracts/trade_finder.py`:

  ```python
  class TradeSuggestion(BaseModel):
      partner_team: str
      partner_record: str | None = None  # "11-1-0 · 1st" — from load_league_records
      grade: str = ""  # engine grade_trade(user_sgp_gain) — already computed by the finder
      giving: list[PlayerRef] = []  # players YOU give
      receiving: list[PlayerRef] = []  # players you receive
      net_sgp: float = 0.0
      rationale: str = ""
  ```

- [ ] **Implement the service mapping.** Edit `api/services/trade_finder_service.py`:
  - Add a `_partner_records()` static helper that loads `load_league_records()` once and returns `{normalize_team_name(team): "W-L-T · Nth"}` (with the ordinal suffix), degrading to `{}` on any failure.
  - In `_build_suggestions`, accept the records map (load it once before the loop), read `opp.get("grade", "")`, and look up `partner_record` by `normalize_team_name(partner_team)`.

  ```python
  @staticmethod
  def _ordinal(n: int) -> str:
      if 10 <= n % 100 <= 20:
          suf = "th"
      else:
          suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
      return f"{n}{suf}"

  @staticmethod
  def _partner_records() -> dict[str, str]:
      """{normalized_team_name: 'W-L-T · Nth'} from load_league_records; {} on any
      failure (records degrade to None, never crash)."""
      try:
          from src.database import load_league_records

          df = load_league_records()
          if df is None or df.empty:
              return {}
          out: dict[str, str] = {}
          for _, r in df.iterrows():
              name = str(r.get("team_name", "") or "")
              if not name.strip():
                  continue
              w, l, t = int(r.get("wins", 0) or 0), int(r.get("losses", 0) or 0), int(r.get("ties", 0) or 0)
              rank = int(r.get("rank", 0) or 0)
              rec = f"{w}-{l}-{t}"
              if rank > 0:
                  rec = f"{rec} · {TradeFinderService._ordinal(rank)}"
              out[normalize_team_name(name)] = rec
          return out
      except Exception:
          logger.warning("TradeFinderService._partner_records failed", exc_info=True)
          return {}
  ```

  Change `_build_suggestions` to a `@staticmethod`-free or keep static but load records inside it (it's already static; load records at the top of the method). In the loop, set:
  ```python
  records = TradeFinderService._partner_records()
  ...
  grade: str = str(opp.get("grade", "") or "")
  partner_record = records.get(normalize_team_name(partner_team))
  suggestions.append(
      TradeSuggestion(
          partner_team=partner_team,
          partner_record=partner_record,
          grade=grade,
          giving=giving,
          receiving=receiving,
          net_sgp=net_sgp,
          rationale=rationale,
      )
  )
  ```

- [ ] Run, EXPECT PASS: `python -m pytest tests/api/test_api_trade_finder.py -v -k "engine_grade or missing_grade"`.
- [ ] **Regenerate openapi** (the contract changed): `python scripts/export_openapi.py`.
- [ ] Run the openapi snapshot guard, EXPECT PASS: `python -m pytest tests/api/test_openapi_contract.py -v`.
- [ ] Run the full finder file again: `python -m pytest tests/api/test_api_trade_finder.py -v`.
- [ ] Commit:
  ```
  git add api/contracts/trade_finder.py api/services/trade_finder_service.py tests/api/test_api_trade_finder.py api/openapi.json
  git commit -m "feat(trade-finder): TradeSuggestion += grade + partner_record (additive)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

## Task 4 — ENRICH contract: `category_impacts` (cheap per-category SGP diff in the service)

The finder engine output dicts carry NO per-category breakdown (only a scalar `category_fit`). Compute `category_impacts` in the SERVICE as a lightweight before/after roster-totals SGP diff per category — reusing `_roster_category_totals` + `SGPCalculator.totals_sgp` (no MC, no LP per card; ≤10 suggestions, cheap). Reuse the SHARED `CategoryImpact` from `api/contracts/trade.py` (do NOT redefine).

**Files:**
- `api/contracts/trade_finder.py` (import + add `category_impacts: list[CategoryImpact]` — reuse the shared model).
- `api/services/trade_finder_service.py` (`_category_impacts(user_roster_ids, giving_ids, receiving_ids, pool, config)` helper; thread `user_roster_ids` into `_build_suggestions`).
- `tests/api/test_api_trade_finder.py` (assert per-category deltas computed for a known roster swap).

Steps:

- [ ] **Failing test first.** Append:

  ```python
  def test_category_impacts_computed_from_roster_diff():
      """category_impacts = per-category SGP delta of (roster - give + receive) vs
      roster, computed in the service (the finder engine doesn't surface it)."""
      svc = TradeFinderService()
      # pool with real-ish category columns so totals_sgp moves on a swap
      pool = pd.DataFrame(
          {
              "player_id": [1, 2, 3, 4],
              "player_name": ["You A", "You B", "You C", "Riv A"],
              "positions": ["OF", "OF", "OF", "SS"],
              "mlb_id": [101, 102, 103, 104],
              "team": ["NYM", "SEA", "KC", "CIN"],
              "is_hitter": [1, 1, 1, 1],
              "r": [80, 70, 60, 95],
              "hr": [20, 15, 10, 30],
              "rbi": [70, 60, 50, 90],
              "sb": [5, 3, 2, 30],
              "ab": [550, 540, 500, 560],
              "h": [150, 140, 130, 165],
              "bb": [50, 45, 40, 60],
              "hbp": [3, 2, 2, 4],
              "sf": [4, 3, 3, 5],
              "obp": [0.350, 0.330, 0.320, 0.380],
              "avg": [0.272, 0.259, 0.260, 0.295],
          }
      )
      user_roster_ids = [1, 2, 3]
      impacts = svc._category_impacts(user_roster_ids, [1], [4], pool)
      cats = {ci.cat for ci in impacts}
      # at minimum SB should be a large positive delta (give a 5-SB OF, get a 30-SB SS)
      sb = next((ci for ci in impacts if ci.cat == "SB"), None)
      assert sb is not None and sb.delta > 0
      assert "HR" in cats  # a counting cat is present


  def test_category_impacts_empty_when_no_user_roster():
      """No user roster ids → empty impacts, no crash."""
      svc = TradeFinderService()
      assert svc._category_impacts([], [1], [4], _fake_pool()) == []
  ```

- [ ] Run, EXPECT FAIL (`_category_impacts` doesn't exist): `python -m pytest tests/api/test_api_trade_finder.py -v -k "category_impacts"`.

- [ ] **Implement.** Edit `api/contracts/trade_finder.py`:
  ```python
  from api.contracts.trade import CategoryImpact
  ```
  Add to `TradeSuggestion`: `category_impacts: list[CategoryImpact] = []`.

  Edit `api/services/trade_finder_service.py` — add the helper (it imports the engine inside, NaN-safe, never raises):
  ```python
  @staticmethod
  def _category_impacts(
      user_roster_ids: list[int], giving_ids: list[int], receiving_ids: list[int], pool, config=None
  ) -> list:
      """Per-category SGP delta of the post-trade roster vs the current roster.
      Cheap totals diff (no MC/LP) — fine for ≤10 cards. Returns [] on any failure
      or empty roster."""
      from api.contracts.trade import CategoryImpact

      if not user_roster_ids:
          return []
      try:
          from src.in_season import _roster_category_totals
          from src.valuation import LeagueConfig, SGPCalculator

          cfg = config or LeagueConfig()
          calc = SGPCalculator(cfg)
          before_ids = list(user_roster_ids)
          after_ids = [pid for pid in before_ids if pid not in set(giving_ids)] + list(receiving_ids)
          before = _roster_category_totals(before_ids, pool)
          after = _roster_category_totals(after_ids, pool)
          before_sgp = calc.player_sgp(_as_series(before, cfg))  # see note below
          # NOTE: prefer per-category SGP via calc.player_sgp on totals; if the
          # calculator only exposes totals_sgp (scalar), compute per-cat deltas as the
          # signed raw-total delta scaled by the category's sgp denominator instead.
          out: list[CategoryImpact] = []
          for cat in cfg.all_categories:
              b = float(before.get(cat, 0.0) or 0.0)
              a = float(after.get(cat, 0.0) or 0.0)
              denom = abs(float(cfg.sgp_denominators.get(cat, 1.0) or 1.0)) or 1.0
              raw = a - b
              # inverse cats: a lower after-total is GOOD → flip sign so + = improvement
              if cat in cfg.inverse_stats:
                  raw = -raw
              delta = raw / denom
              if delta != delta:  # NaN guard
                  continue
              out.append(CategoryImpact(cat=cat, delta=round(delta, 3)))
          return out
      except Exception:
          logger.warning("TradeFinderService._category_impacts failed", exc_info=True)
          return []
  ```

  > **Implementer note (resolve during TDD):** `SGPCalculator` exposes `totals_sgp(totals, weights)` (scalar) and `player_sgp(series)` (per-category dict) — it does NOT have a per-category "totals → per-cat-SGP" method. The **simplest correct** per-category delta is the `(after - before)` raw category-total delta divided by that category's `sgp_denominators[cat]` (with inverse-stat sign flip) shown above — this is exactly how SGP per category is defined (`stat_delta / denom`). Drop the `calc.player_sgp(_as_series(...))` line and the `_as_series` reference; they are only a hint. Keep the denominator-based computation, which needs no new engine surface. Confirm `cfg.sgp_denominators` keys are UPPERCASE category names (they are) so `cfg.sgp_denominators.get(cat)` matches `cfg.all_categories`.

  Thread `user_roster_ids` into `_build_suggestions`: change its signature to `_build_suggestions(raw, pool, user_roster_ids)` and call `category_impacts=TradeFinderService._category_impacts(user_roster_ids, giving_ids, receiving_ids, pool)` in the constructed `TradeSuggestion`. Update the call site in `get_suggestions` to `self._build_suggestions(raw, pool, user_roster_ids)`. Update the existing `_build_suggestions`-direct tests from Task 3 to pass `user_roster_ids` (or default it to `None`/`[]` with a keyword and have those tests omit it → impacts `[]`, which they don't assert on).

- [ ] Run, EXPECT PASS: `python -m pytest tests/api/test_api_trade_finder.py -v -k "category_impacts"`.
- [ ] **Regenerate openapi:** `python scripts/export_openapi.py`; then `python -m pytest tests/api/test_openapi_contract.py -v`.
- [ ] Run the whole finder file: `python -m pytest tests/api/test_api_trade_finder.py -v`.
- [ ] Commit:
  ```
  git add api/contracts/trade_finder.py api/services/trade_finder_service.py tests/api/test_api_trade_finder.py api/openapi.json
  git commit -m "feat(trade-finder): TradeSuggestion += category_impacts (service-side per-cat SGP diff)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

## Task 5 — FRONTEND: render real grade / partner record / category impact (replace verdict-only)

Regenerate TS types from the new openapi, then make the Finder adapter + TradeCard use the now-real `grade`, `partner_record`, `category_impacts` instead of the verdict-only derivation. Keep `verdict` (derived from `net_sgp`) as a fallback string; `playoffDelta` stays mock-only (deferred — see Task 6).

**Files:**
- `web/src/lib/api/generated.ts` (regenerated — DO NOT hand-edit; `pnpm gen:api`).
- `web/src/lib/api/adapters.ts` (`apiTradeFinderToData`, lines ~508-523 — map grade/partnerRecord/impact).
- `web/src/lib/trades-data.ts` (`TradeRec` already has optional `grade`/`partnerRecord`/`impact` — confirm the `CatImpact` display formatting).
- `web/src/app/trades/page.tsx` (`TradeCard` already renders `rec.grade` / `rec.partnerRecord` / `ImpactLedger` when present — no change expected; verify).

Steps:

- [ ] Regenerate types (the openapi changed in Tasks 3-4): from `web/`, `pnpm gen:api`. Confirm `web/src/lib/api/generated.ts` now has `grade`, `partner_record`, `category_impacts` on the `TradeSuggestion` schema.
- [ ] Edit `apiTradeFinderToData` in `web/src/lib/api/adapters.ts` to surface the real fields. The frontend `CatImpact` expects `{ cat, delta: string (signed, display-ready), dir: "up"|"down" }`; the API `CategoryImpact` is `{ cat, delta: number }`. Add a small formatter (rate cats → 3-decimals, counting cats → integer-ish; sign-prefixed) and map `dir` from the delta sign:

  ```ts
  const RATE_CATS = new Set(["AVG", "OBP", "ERA", "WHIP"]);

  function fmtCatDelta(cat: string, delta: number): { delta: string; dir: "up" | "down" } {
    const dir = delta >= 0 ? "up" : "down";
    const sign = delta >= 0 ? "+" : "";
    const body = RATE_CATS.has(cat) ? Math.abs(delta).toFixed(3).replace(/^0/, "") : Math.round(Math.abs(delta)).toString();
    return { delta: `${sign}${RATE_CATS.has(cat) ? (delta < 0 ? "-" : "") : ""}${body}`, dir };
  }

  export function apiTradeFinderToData(api: ApiTradeFinderResponse): TradesData {
    const recs = (api.suggestions ?? []).map((s: ApiTradeSuggestion, i) => {
      const impact = (s.category_impacts ?? [])
        .filter((c) => (c.delta ?? 0) !== 0)
        .map((c) => ({ cat: c.cat, ...fmtCatDelta(c.cat, c.delta ?? 0) }));
      return {
        id: i,
        partner: s.partner_team ?? "",
        partnerRecord: s.partner_record ?? undefined,
        grade: s.grade || undefined, // real engine grade; falsy → card falls back to netSgp badge
        netSgp: s.net_sgp,
        verdict: verdictFromSgp(s.net_sgp ?? 0),
        give: (s.giving ?? []).map(toTradePlayer),
        get: (s.receiving ?? []).map(toTradePlayer),
        impact: impact.length > 0 ? impact : undefined,
        rationale: s.rationale ?? "",
      };
    });
    return { needs: [], recs };
  }
  ```
  Update the stale doc-comment above the function (it currently says grade/impact/partnerRecord "aren't in it → omitted") to note they are now mapped from the enriched contract; `playoffDelta` remains omitted (deferred).
- [ ] Confirm `web/src/app/trades/page.tsx` `TradeCard` already conditionally renders `rec.grade` (line ~151), `rec.partnerRecord` (line ~141), and `<ImpactLedger>` when `rec.impact` is present (line ~188) — NO change needed; just verify by reading. The `gradeColor` helper handles A/B/other.
- [ ] Update `web/src/lib/trades-data.ts` doc-comments (the file header + `TradeRec` field comments at lines 9-39 currently say grade/impact/partnerRecord are "mock-only / hidden on live data") to reflect they are now live from the enriched contract; `playoffDelta` stays mock-only.
- [ ] Frontend gate (from `web/`): `pnpm exec tsc --noEmit` → EXPECT clean; `pnpm run lint` → EXPECT clean; `pnpm build` → EXPECT success.
- [ ] Commit:
  ```
  git add web/src/lib/api/generated.ts web/src/lib/api/adapters.ts web/src/lib/trades-data.ts
  git commit -m "feat(web/trades): Finder cards render real grade + partner record + category impact

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

## Task 6 — VERIFY: Compare + Build end-to-end + document deferrals (no code unless a bug is found)

Confirm the two non-Finder trade surfaces work on live data and the safety properties hold. This task is verification + documentation; only write code (a follow-up task) if a real bug surfaces. Record results in the Verification Log.

**Files:** read-only — `api/routers/compare.py`, `api/routers/trade.py`, `web/src/components/trades/BuildPanel.tsx`, `web/src/components/trades/ComparePanel.tsx`, `web/src/lib/trades-data.ts`.

Steps:

- [ ] **Pro-gate dormant (free beta):** confirm `api/gating.py require_pro` returns early when `not stripe_enabled()` (`billing_env_configured()` False → Stripe unset). On the free beta the Finder's `Depends(require_pro)` and Build's gate are no-ops (friends never 402). Record: gate is dormant; no change.
- [ ] **Build keeps MC opt-in (no 45s hang):** confirm `web/src/lib/trades-data.ts:209` sends `enable_mc: false` and `api/contracts/trade.py TradeEvaluateRequest.enable_mc` defaults `False`. Read `BuildPanel.tsx` to confirm no UI path flips it on by default. Record: MC opt-in confirmed.
- [ ] **Compare works on live data:** `GET /api/compare?ids=545361,665742` against the live API (or local stack over the main-checkout DB) → EXPECT a `CompareResponse` with a table for both ids. Read `ComparePanel.tsx` to confirm the picker (player search) → `/compare` wiring. Record status/shape.
- [ ] **Build works:** `POST /api/trade/evaluate` `{team_name, giving_ids, receiving_ids, enable_mc:false}` against live (or local-stack/main-DB) → EXPECT a `TradeEvaluationResponse` with a real `grade` + `category_impacts`. Record.
- [ ] **Finder live-verify (the whole point):** run the API locally against the main-checkout 26 MB DB (`uvicorn api.main:create_app --factory --port 8000` from the main checkout, NOT the empty worktree) OR hit the live Railway API, then `GET /api/trade-finder?team_name=Team+Hickey&limit=10` → EXPECT non-empty `suggestions` with populated `grade`, `partner_record`, `category_impacts`. This is the regression-closing check (the worktree DB is empty → only the live/main-DB path proves the fix). Record before (Task 1) vs after counts.
- [ ] **Document the deferral:** per-suggestion `playoffDelta` is NOT added (a `simulate_trade_playoff_delta` sim per card is too slow — same call the evaluator's title-odds rerank gates behind `enable_title_odds_rerank`). The frontend `TradeRec.playoffDelta` stays mock-only. Note this in the Verification Log.
- [ ] If every check passes and no bug surfaced: no commit (verification only). If a bug surfaced, open a new TDD task (failing test → fix → commit) following the same slice pattern; do NOT fix inline without a test.

---

## Finish

- [ ] Run the relevant backend suite once more: `python -m pytest tests/api/test_api_trade_finder.py tests/api/test_no_logic_in_routers.py tests/api/test_openapi_contract.py -v`.
- [ ] Frontend gate from `web/`: `pnpm build` (the gate) + `pnpm exec tsc --noEmit` + `pnpm run lint`.
- [ ] Per the standing rule: merge to local `master` and push `origin/master` (the integrator pass reconciles `api/main.py`/`openapi.json`/`generated.ts` across workstreams — resolve generated files by REGENERATING, never hand-merging).
- [ ] Run `superpowers:requesting-code-review` (or the `silent-failure-hunter` agent) on the non-trivial backend change (the Finder service resolution + per-category diff): CI green ≠ no silent failures.

---

## Verification Log

_Fill in during execution._

- **Task 1 root cause (confirmed / observed):**
  - (a) name mismatch — repro output: …
  - (b) `all_team_totals=None` early-return at `src/trade_finder.py:1895` — …
  - Live `GET /api/trade-finder?team_name=Team+Hickey` (before fix): …
- **Task 6 live-verify (after fix):**
  - Finder suggestions count (before → after): …
  - `grade` / `partner_record` / `category_impacts` populated? …
  - Compare `/api/compare?ids=…`: …
  - Build `/api/trade/evaluate` (`enable_mc:false`): grade …, category_impacts …
  - Pro-gate dormant confirmed: …
  - Deferred: per-suggestion playoffDelta (sim-per-card too slow) — not added.
