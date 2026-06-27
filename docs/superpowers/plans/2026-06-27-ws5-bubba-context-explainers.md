# WS5 — Bubba: Auto Page-Context + Explainer Tools — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Bubba (the AI assistant) "omniscient" two ways: (1) it automatically sees the structured data each page rendered, and (2) it can explain how any number was computed by citing the optimizer constants registry and reusing the engines' existing metric breakdowns. Three explainer tools (`explain_constant`, `list_constants`, `explain_metric`) live in `src/ai/tools.py` so they work in BOTH the live Streamlit app and the FastAPI API (one-engine principle). The API gains a `page_context` request field; the React frontend auto-attaches a size-capped JSON of the current page's data to every Bubba message via a centralized `BubbaContext`.

**Architecture:**
- **One-engine, additive, frozen-reference-safe.** The new tools register in `tool_specs()` + dispatch in `dispatch_tool()`. The live Streamlit app calls `src/ai/providers.chat()` on its hot path; `chat()` is a thin DRAIN of `_chat_events`, guarded by `tests/test_ai_providers_streaming.py::test_chat_is_a_drain_returning_todays_exact_dict` (FROZEN REFERENCE). The tools are pure/read-only and only *added* to the registry — `chat()`'s return dict shape is unchanged. **Every task that touches `src/ai/` MUST keep that frozen-reference test green**, and Task 4 adds an explicit additive equivalence test proving `tool_specs()` now contains the three new tools while `chat()` still returns the byte-identical dict.
- **Explainer tools are DB-free where possible.** `explain_constant`/`list_constants` read the pure `CONSTANTS_REGISTRY` dict (`src/optimizer/constants_registry.py`, frozen `ConstantEntry` dataclass: `value`, `lower_bound`, `upper_bound`, `citation`, `module`, `sensitivity`, `description`). `explain_metric` assembles the formula string + per-component value + weight + input variables by REUSING the engines' existing breakdowns — for `stream_score` it mirrors `api/services/streaming_service.py::_factors` (the 6 registry-weighted factors), for `dcv`/`trade_grade`/`start_score` it composes the registry weights + a documented formula string. `explain_metric` accepts an optional `params` dict and is tested with fakes (no live DB), degrading to a graceful `{"error": ...}` on a bad/unsupported `kind`.
- **API page-context is a string field, fully backward-compatible.** `PageContext{page: str, data_json: str}` is optional on `ChatSendRequest`; when `None`, `_build_user_content`/system-prompt assembly is byte-identical to today (the existing `attached_text` tag flow is untouched and independent). The page data is folded into the user turn as an explicit, truncation-aware block so Bubba knows to reach for tools when truncated.
- **Frontend auto-context is centralized.** A `BubbaContext` provider (mounted in `web/src/app/layout.tsx` wrapping both `{children}` and `<Bubba/>`) holds `{pageId, data}`. `usePageData` publishes to it on each `loaded` state — one change covers all 14+ pages. `Bubba` reads the context and attaches a size-capped (≤16 KB) JSON serialization as `page_context` on every message (default ON, with a quiet toggle that defaults on). This is distinct from the manual `attached_text` select-to-tag flow (kept).
- **Proven API slice pattern:** contract (`api/contracts/chat.py`) → service (`api/services/chat_service.py`) → router already threads it (no logic) → regen `api/openapi.json` via `python scripts/export_openapi.py` (snapshot-guarded by `tests/api/test_openapi_contract.py`). `fastapi==0.137.1` + `httpx==0.28.1` are PINNED (the OpenAPI error-schema auto-generation makes an unpinned drift fail the snapshot).

**Tech Stack:** Python 3.12 (CI) / 3.14 (local) · FastAPI 0.137.1 · pydantic v2 · pytest (DB-free, worktree DB is empty) · Next.js 16 + React 19 + TypeScript 5.9 (pnpm) · Windows PowerShell.

---

## Ordering & shared-file notes

- **Backend first, frontend last.** Order: explainer tools (Tasks 1–4) → API page_context contract+service+openapi (Tasks 5–7) → frontend BubbaContext + Bubba wiring (Tasks 8–10).
- **`api/openapi.json` is regenerated exactly ONCE** (Task 7), after the only contract change (Task 5) lands. Do not hand-edit it.
- **Test file locations (verified):** `tests/test_ai_tools.py` (the `src/ai/tools.py` unit tests), `tests/test_ai_providers_streaming.py` (the `chat()` frozen-reference drain test), `tests/api/test_chat_service.py` (the DB-free ChatService tests). New backend tests go in those files (or a new `tests/test_ai_explainers.py` for the metric/equivalence additions, per the task steps).
- **Run backend tests** with: `python -m pytest tests/test_ai_tools.py tests/test_ai_providers_streaming.py tests/api/test_chat_service.py tests/api/test_openapi_contract.py -v`
- **Frontend gate** (from `web/`): `pnpm build` && `pnpm exec tsc --noEmit` && `pnpm run lint`.

---

## Task 1 — `explain_constant(name)` tool: registry entry lookup

**Files:**
- `src/ai/tools.py` (add `_fn` spec in `tool_specs()` after the `request_refresh` spec at line ~64; add a dispatch branch in `dispatch_tool()` before the final `return json.dumps({"error": ...})` at line ~149; add a private `_explain_constant` helper near the other `_get_*` helpers at end of file)
- `tests/test_ai_tools.py` (add tests after `test_dispatch_unknown_tool_returns_error`)

Steps:

- [ ] **Failing test** — append to `tests/test_ai_tools.py`:
  ```python
  def test_explain_constant_in_specs():
      from src.ai.tools import tool_specs

      names = {t["function"]["name"] for t in tool_specs()}
      assert "explain_constant" in names


  def test_dispatch_explain_constant_known():
      import json

      from src.ai.tools import dispatch_tool
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      name = next(iter(CONSTANTS_REGISTRY))  # any real registered constant
      out = json.loads(dispatch_tool("explain_constant", {"name": name}, user_id=99))
      entry = CONSTANTS_REGISTRY[name]
      assert out["name"] == name
      assert out["value"] == entry.value
      assert out["lower_bound"] == entry.lower_bound
      assert out["upper_bound"] == entry.upper_bound
      assert out["citation"] == entry.citation
      assert out["module"] == entry.module
      assert out["sensitivity"] == entry.sensitivity
      assert out["description"] == entry.description


  def test_dispatch_explain_constant_unknown_is_graceful():
      import json

      from src.ai.tools import dispatch_tool

      out = json.loads(dispatch_tool("explain_constant", {"name": "no_such_constant"}, user_id=99))
      assert "error" in out and "no_such_constant" in out["error"]
  ```
- [ ] **Run (expect FAIL)** — `python -m pytest tests/test_ai_tools.py -v -k explain_constant` (fails: tool not registered).
- [ ] **Minimal impl** — in `src/ai/tools.py`, add to the `specs` list in `tool_specs()` (after the `request_refresh` `_fn(...)`):
  ```python
  _fn(
      "explain_constant",
      "Explain ONE registered optimizer constant: its value, plausible bounds, "
      "research citation, source module, sensitivity tier, and a one-line description. "
      "Use this to cite how a tunable number (matchup weight, stabilization point, "
      "platoon split, etc.) was set when explaining 'how' a metric was computed.",
      {"name": {"type": "string", "description": "The constant's registry key, e.g. 'stream_score_w_matchup'."}},
      ["name"],
  ),
  ```
  Add the dispatch branch in `dispatch_tool()` (before the final unknown-tool `return`):
  ```python
  if name == "explain_constant":
      return _explain_constant(str(args.get("name", "")))
  ```
  Add the helper at the end of the file:
  ```python
  def _explain_constant(name: str) -> str:
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      key = name.strip()
      entry = CONSTANTS_REGISTRY.get(key)
      if entry is None:
          return json.dumps({"error": f"Unknown constant: {name!r}. Use list_constants to browse."})
      return json.dumps(
          {
              "name": key,
              "value": entry.value,
              "lower_bound": entry.lower_bound,
              "upper_bound": entry.upper_bound,
              "citation": entry.citation,
              "module": entry.module,
              "sensitivity": entry.sensitivity,
              "description": entry.description,
          },
          default=str,
      )
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/test_ai_tools.py -v -k explain_constant`.
- [ ] **Commit** — `git add src/ai/tools.py tests/test_ai_tools.py && git commit -m "feat(bubba): explain_constant tool over CONSTANTS_REGISTRY"`

---

## Task 2 — `list_constants(module?, sensitivity?)` tool: filtered discovery

**Files:**
- `src/ai/tools.py` (add spec in `tool_specs()` after the `explain_constant` spec; add dispatch branch in `dispatch_tool()`; add `_list_constants` helper)
- `tests/test_ai_tools.py` (add tests)

Steps:

- [ ] **Failing test** — append to `tests/test_ai_tools.py`:
  ```python
  def test_list_constants_in_specs():
      from src.ai.tools import tool_specs

      names = {t["function"]["name"] for t in tool_specs()}
      assert "list_constants" in names


  def test_dispatch_list_constants_all():
      import json

      from src.ai.tools import dispatch_tool
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      out = json.loads(dispatch_tool("list_constants", {}, user_id=99))
      assert out["count"] == len(CONSTANTS_REGISTRY)
      returned = {c["name"] for c in out["constants"]}
      assert returned == set(CONSTANTS_REGISTRY)
      first = out["constants"][0]
      assert set(first) == {"name", "description", "module", "sensitivity"}


  def test_dispatch_list_constants_filtered_by_sensitivity():
      import json

      from src.ai.tools import dispatch_tool
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      out = json.loads(dispatch_tool("list_constants", {"sensitivity": "high"}, user_id=99))
      expected = {k for k, v in CONSTANTS_REGISTRY.items() if v.sensitivity.upper() == "HIGH"}
      assert {c["name"] for c in out["constants"]} == expected


  def test_dispatch_list_constants_filtered_by_module():
      import json

      from src.ai.tools import dispatch_tool
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      mod = next(iter(CONSTANTS_REGISTRY.values())).module
      out = json.loads(dispatch_tool("list_constants", {"module": mod}, user_id=99))
      expected = {k for k, v in CONSTANTS_REGISTRY.items() if mod.lower() in v.module.lower()}
      assert {c["name"] for c in out["constants"]} == expected
  ```
- [ ] **Run (expect FAIL)** — `python -m pytest tests/test_ai_tools.py -v -k list_constants`.
- [ ] **Minimal impl** — in `src/ai/tools.py`, add to `tool_specs()`:
  ```python
  _fn(
      "list_constants",
      "Browse the registered optimizer constants (name + one-line description). "
      "Optionally filter by 'module' (substring, e.g. 'streaming.py') or 'sensitivity' "
      "(HIGH/MEDIUM/LOW). Use to discover which constant to explain_constant.",
      {
          "module": {"type": "string", "description": "Substring match against the source module."},
          "sensitivity": {"type": "string", "description": "HIGH, MEDIUM, or LOW."},
      },
  ),
  ```
  Add the dispatch branch:
  ```python
  if name == "list_constants":
      return _list_constants(args.get("module"), args.get("sensitivity"))
  ```
  Add the helper:
  ```python
  def _list_constants(module: str | None, sensitivity: str | None) -> str:
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      mod = (module or "").strip().lower()
      sens = (sensitivity or "").strip().upper()
      rows = []
      for key, entry in CONSTANTS_REGISTRY.items():
          if mod and mod not in entry.module.lower():
              continue
          if sens and entry.sensitivity.upper() != sens:
              continue
          rows.append(
              {
                  "name": key,
                  "description": entry.description,
                  "module": entry.module,
                  "sensitivity": entry.sensitivity,
              }
          )
      return json.dumps({"count": len(rows), "constants": rows}, default=str)
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/test_ai_tools.py -v -k list_constants`.
- [ ] **Commit** — `git add src/ai/tools.py tests/test_ai_tools.py && git commit -m "feat(bubba): list_constants discovery tool"`

---

## Task 3 — `explain_metric(kind, params)` tool: formula + components + weights

**Files:**
- `src/ai/tools.py` (add spec in `tool_specs()` after the `list_constants` spec; add dispatch branch in `dispatch_tool()`; add `_explain_metric` helper + a `_METRIC_FORMULAS` constant)
- `tests/test_ai_explainers.py` (new file — DB-free metric/breakdown assembly tests)

Design notes (read before coding):
- `explain_metric` returns `{kind, formula, components: [{key, label, weight, value, detail}], inputs: {...}, note}`.
- For `kind="stream_score"`: the six factors and their registry weights mirror `api/services/streaming_service.py::_factors` — keys `matchup`, `sgp`, `form`, `lineup`, `env`, `winprob`, each weight from `CONSTANTS_REGISTRY[f"stream_score_w_{key}"].value`. When `params` carries a `components` dict and/or input variables (`opp_wrc_plus`, `opp_k_pct`, `park_factor`, `net_sgp`, `win_probability`), surface them; otherwise return the formula + weights with `value=None` and a note that live values come from the streaming board / `/api/streaming/analyze`.
- For `kind="dcv"` and `kind="start_score"` and `kind="trade_grade"`: return the documented formula string + the relevant registry weights (filtered by the metric's module prefix) + any provided `params`. Keep these breakdown-assembly-only and DB-free; the live numeric values are carried inline on the page payloads (streaming `factors[]`, trade `category_impacts`, the daily DCV slot fields) — `explain_metric` explains the *recipe*, the page payloads carry the *values*.
- Unknown/unsupported `kind` → `{"error": "Unsupported metric kind: <kind>. Supported: stream_score, dcv, trade_grade, start_score."}`.
- Never raise — the outer `dispatch_tool` try/except already guards, but the helper should also degrade gracefully when a registry key is missing (weight → 0.0, like `_factors._w`).

Steps:

- [ ] **Failing test** — create `tests/test_ai_explainers.py`:
  ```python
  """explain_metric assembles the formula + registry weights + provided inputs.
  DB-free: the registry is pure; inputs are passed as params, not fetched."""

  import json

  from src.ai.tools import dispatch_tool, tool_specs


  def test_explain_metric_in_specs():
      names = {t["function"]["name"] for t in tool_specs()}
      assert "explain_metric" in names


  def test_explain_metric_stream_score_weights_match_registry():
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      out = json.loads(dispatch_tool("explain_metric", {"kind": "stream_score"}, user_id=99))
      assert out["kind"] == "stream_score"
      assert out["formula"]  # a non-empty formula string
      by_key = {c["key"]: c for c in out["components"]}
      assert set(by_key) == {"matchup", "sgp", "form", "lineup", "env", "winprob"}
      for key, comp in by_key.items():
          assert comp["weight"] == CONSTANTS_REGISTRY[f"stream_score_w_{key}"].value


  def test_explain_metric_surfaces_provided_component_values():
      out = json.loads(
          dispatch_tool(
              "explain_metric",
              {"kind": "stream_score", "params": {"components": {"matchup": 0.4, "sgp": -0.1}}},
              user_id=99,
          )
      )
      by_key = {c["key"]: c for c in out["components"]}
      assert by_key["matchup"]["value"] == 0.4
      assert by_key["sgp"]["value"] == -0.1
      # a component not supplied stays None (recipe, not a fabricated value)
      assert by_key["form"]["value"] is None


  def test_explain_metric_dcv_returns_formula_and_weights():
      out = json.loads(dispatch_tool("explain_metric", {"kind": "dcv"}, user_id=99))
      assert out["kind"] == "dcv" and out["formula"]
      assert isinstance(out["components"], list)


  def test_explain_metric_trade_grade_supported():
      out = json.loads(dispatch_tool("explain_metric", {"kind": "trade_grade"}, user_id=99))
      assert out["kind"] == "trade_grade" and out["formula"]


  def test_explain_metric_start_score_supported():
      out = json.loads(dispatch_tool("explain_metric", {"kind": "start_score"}, user_id=99))
      assert out["kind"] == "start_score" and out["formula"]


  def test_explain_metric_unknown_kind_is_graceful():
      out = json.loads(dispatch_tool("explain_metric", {"kind": "bogus"}, user_id=99))
      assert "error" in out and "bogus" in out["error"]


  def test_explain_metric_missing_kind_is_graceful():
      out = json.loads(dispatch_tool("explain_metric", {}, user_id=99))
      assert "error" in out
  ```
- [ ] **Run (expect FAIL)** — `python -m pytest tests/test_ai_explainers.py -v`.
- [ ] **Minimal impl** — in `src/ai/tools.py`, add to `tool_specs()`:
  ```python
  _fn(
      "explain_metric",
      "Explain HOW a HEATER score is computed: returns the formula string, each "
      "component's registry weight + value, and the input variables. kind is one of "
      "'stream_score' (pitcher streaming), 'dcv' (daily category value), 'trade_grade', "
      "'start_score' (start/sit). Pass live numbers in 'params' (e.g. "
      "{'components': {...}, 'net_sgp': 1.2}) to have them surfaced; omit to get the "
      "recipe + weights. Cite the constants it returns when answering 'how'.",
      {
          "kind": {"type": "string", "description": "stream_score | dcv | trade_grade | start_score"},
          "params": {"type": "object", "description": "Optional live inputs/components to surface."},
      },
      ["kind"],
  ),
  ```
  Add the dispatch branch:
  ```python
  if name == "explain_metric":
      return _explain_metric(str(args.get("kind", "")), args.get("params") or {})
  ```
  Add the helper + formula table at end of file:
  ```python
  # Per-metric formula string + the registry weight keys that compose it. The
  # stream_score key list MIRRORS api/services/streaming_service.py::_factors so the
  # explainer and the live board agree. dcv/start_score/trade_grade describe the
  # recipe; the live per-category numbers travel inline on each page's payload.
  _METRIC_SPECS: dict[str, dict] = {
      "stream_score": {
          "formula": "stream_score (0-100) = 100 * Σ w_k · factor_k for k in "
          "{matchup, sgp, form, lineup, env, winprob}, each factor_k in [-1, 1].",
          "weight_keys": [
              ("matchup", "Matchup"),
              ("sgp", "Streaming value"),
              ("form", "Recent form"),
              ("lineup", "Lineup exposure"),
              ("env", "Environment / park"),
              ("winprob", "Team win probability"),
          ],
          "weight_prefix": "stream_score_w_",
      },
      "dcv": {
          "formula": "daily_category_value = base_sgp_per_game · urgency_weight · "
          "matchup_multiplier · volume_factor; normalized to a 0-100 heat. Locked/IL/"
          "off-day starts get volume_factor 0.",
          "module": "daily_optimizer.py",
      },
      "trade_grade": {
          "formula": "grade derives from Phase-1 weighted surplus_sgp = Σ_cat "
          "(marginal_sgp(receiving) - marginal_sgp(giving)) · category_weight; "
          "inverse cats (L/ERA/WHIP) sign-flipped. Phase 1 is the grade authority.",
          "module": "engine",
      },
      "start_score": {
          "formula": "start_score (0-100) = weekly_projection · urgency · matchup_factors, "
          "risk-adjusted; home/away apply home_advantage/away_discount.",
          "module": "start_sit.py",
      },
  }


  def _explain_metric(kind: str, params: dict) -> str:
      k = (kind or "").strip().lower()
      spec = _METRIC_SPECS.get(k)
      if spec is None:
          return json.dumps(
              {
                  "error": f"Unsupported metric kind: {kind!r}. "
                  "Supported: stream_score, dcv, trade_grade, start_score."
              }
          )
      from src.optimizer.constants_registry import CONSTANTS_REGISTRY

      provided = params.get("components", {}) if isinstance(params.get("components"), dict) else {}
      components: list[dict] = []
      if spec.get("weight_keys"):
          prefix = spec["weight_prefix"]
          for key, label in spec["weight_keys"]:
              entry = CONSTANTS_REGISTRY.get(f"{prefix}{key}")
              components.append(
                  {
                      "key": key,
                      "label": label,
                      "weight": float(entry.value) if entry is not None else 0.0,
                      "value": provided.get(key),
                      "detail": entry.description if entry is not None else "",
                  }
              )
      else:
          mod = str(spec.get("module", "")).lower()
          for key, entry in CONSTANTS_REGISTRY.items():
              if mod and mod in entry.module.lower():
                  components.append(
                      {
                          "key": key,
                          "label": key,
                          "weight": float(entry.value),
                          "value": provided.get(key),
                          "detail": entry.description,
                      }
                  )
      inputs = {kk: vv for kk, vv in params.items() if kk != "components"}
      return json.dumps(
          {
              "kind": k,
              "formula": spec["formula"],
              "components": components,
              "inputs": inputs,
              "note": "Live per-category values travel inline on the page payload "
              "(streaming factors[], trade category_impacts, daily DCV slot fields); "
              "this tool explains the recipe + weights.",
          },
          default=str,
      )
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/test_ai_explainers.py -v`.
- [ ] **Commit** — `git add src/ai/tools.py tests/test_ai_explainers.py && git commit -m "feat(bubba): explain_metric tool (formula + registry weights + inputs)"`

---

## Task 4 — chat() equivalence guard: new tools are additive

**Files:**
- `tests/test_ai_explainers.py` (append the additive equivalence test)

Steps:

- [ ] **Failing test (expect PASS structurally, but add to lock it)** — append to `tests/test_ai_explainers.py`:
  ```python
  def test_new_explainer_tools_are_additive_to_specs():
      """The three explainer tools register WITHOUT removing any existing tool —
      the live Streamlit chat surface only grows."""
      from src.ai.tools import tool_specs

      names = {t["function"]["name"] for t in tool_specs()}
      # original surface still present
      assert {
          "query_data",
          "get_player",
          "compare_players",
          "get_my_team",
          "get_standings",
          "get_free_agents",
          "request_refresh",
      }.issubset(names)
      # new explainers added
      assert {"explain_constant", "list_constants", "explain_metric"}.issubset(names)


  def test_chat_drain_unchanged_with_explainers_registered(monkeypatch):
      """FROZEN REFERENCE (additive): with the explainer tools in the registry,
      src/ai/providers.chat() still drains to the byte-identical pre-WS5 dict. The
      live Streamlit app (src/ai/chat.py) is provably unchanged."""
      from types import SimpleNamespace

      from src.ai import providers

      def _delta(content=None):
          return SimpleNamespace(
              choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))]
          )

      def _rebuilt(content=None, tool_calls=None, in_tok=10, out_tok=5):
          return SimpleNamespace(
              choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
              usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
          )

      monkeypatch.setattr(providers, "_completion", lambda **kw: iter([_delta("Hi "), _delta("there")]))
      monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: _rebuilt(content="Hi there"))
      out = providers.chat(
          model="anthropic/claude-haiku-4-5",
          messages=[{"role": "user", "content": "hi"}],
          api_key="sk-test",
          user_id=99,
      )
      assert out == {"content": "Hi there", "tokens_in": 10, "tokens_out": 5, "tool_trace": []}
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/test_ai_explainers.py tests/test_ai_providers_streaming.py -v`. (Confirm `test_chat_is_a_drain_returning_todays_exact_dict` is STILL green — the original frozen reference.)
- [ ] **Run full tool surface** — `python -m pytest tests/test_ai_tools.py tests/test_ai_explainers.py tests/test_ai_providers.py tests/test_ai_providers_streaming.py -v`.
- [ ] **Commit** — `git add tests/test_ai_explainers.py && git commit -m "test(bubba): equivalence guard — explainer tools are additive, chat() drain unchanged"`

---

## Task 5 — `PageContext` contract + `ChatSendRequest.page_context`

**Files:**
- `api/contracts/chat.py` (add `PageContext` model after `ChatAttachment` at line ~12; add `page_context` field to `ChatSendRequest`)
- `tests/api/test_chat_service.py` (add a contract round-trip test — the service folding is Task 6)

Steps:

- [ ] **Failing test** — append to `tests/api/test_chat_service.py`:
  ```python
  def test_chat_send_request_accepts_page_context():
      from api.contracts.chat import ChatSendRequest, PageContext

      req = ChatSendRequest(
          message="why is this 0.28?",
          model="gpt-5",
          page_context=PageContext(page="optimizer", data_json='{"slots":[]}'),
      )
      assert req.page_context.page == "optimizer"
      assert req.page_context.data_json == '{"slots":[]}'


  def test_chat_send_request_page_context_defaults_none():
      from api.contracts.chat import ChatSendRequest

      req = ChatSendRequest(message="hi", model="gpt-5")
      assert req.page_context is None
  ```
- [ ] **Run (expect FAIL)** — `python -m pytest tests/api/test_chat_service.py -v -k page_context` (fails: `PageContext` does not exist).
- [ ] **Minimal impl** — in `api/contracts/chat.py`, add after `ChatAttachment`:
  ```python
  class PageContext(BaseModel):
      """Structured snapshot of what the page is currently displaying, attached to
      a chat turn so Bubba can 'see the screen'. `data_json` is a (size-capped,
      possibly truncated) JSON string the frontend serializes from the page's
      loaded data; `page` is the route id (e.g. 'optimizer')."""

      page: str = ""
      data_json: str = ""
  ```
  Add the field to `ChatSendRequest` (after `attachments`):
  ```python
      page_context: PageContext | None = None
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/api/test_chat_service.py -v -k page_context`.
- [ ] **Commit** — `git add api/contracts/chat.py tests/api/test_chat_service.py && git commit -m "feat(api): PageContext contract + ChatSendRequest.page_context (optional)"`

---

## Task 6 — ChatService folds page_context into the user turn + system prompt mentions it

**Files:**
- `src/ai/chat.py` (`build_system_prompt` — add the explainer/attached-data lines)
- `api/services/chat_service.py` (`_build_user_content` gains an optional `page_context` arg; `send`/`send_stream` accept + thread `page_context`; pass it into `_build_user_content`)
- `api/routers/chat.py` (`send`/`send_stream` pass `page_context=body.page_context` to the service)
- `tests/api/test_chat_service.py` (add folding tests)
- `tests/test_ai_tools.py` or a small `tests/test_ai_chat_prompt.py` (system-prompt mention test) — use a new `tests/test_ai_chat_prompt.py` to keep `build_system_prompt` coverage isolated

Design notes:
- `_build_user_content(message, attached_text, attachments, page_context=None)` — when `page_context` is present AND has non-empty `data_json`, prepend an explicit block to the text portion: `"[Data currently displayed on the <page> page (may be truncated — use explain_metric/query_data for more)]\n<data_json>\n\n"`. This composes WITH the existing `attached_text` wrapping (the manual tag flow) — both can be present. When `page_context` is `None`/empty, the returned content is byte-identical to today.
- The function still returns a STRING unless image attachments are present (then the multimodal parts list, with the page-context text folded into the `text` part) — exactly the existing branch, just with the augmented text.
- `build_system_prompt` gains two sentences: the page's displayed data may be attached, and Bubba has `explain_constant`/`list_constants`/`explain_metric` to show how any number was derived and should cite the constant/formula when explaining "how." Persona unchanged. **This string change does NOT affect the `chat()` frozen-reference test** (that test monkeypatches `_completion`/`_rebuild` and asserts the drain dict, not the prompt text), but it IS consumed by `api/services/chat_service.py` and `src/ai/chat.py` — keep it additive.

Steps:

- [ ] **Failing test (service folding)** — append to `tests/api/test_chat_service.py`:
  ```python
  def test_build_user_content_folds_page_context():
      from api.contracts.chat import PageContext

      out = cs._build_user_content(
          "why 0.28?",
          None,
          None,
          page_context=PageContext(page="optimizer", data_json='{"slots":[1,2]}'),
      )
      assert isinstance(out, str)
      assert "optimizer" in out
      assert '{"slots":[1,2]}' in out
      assert "why 0.28?" in out


  def test_build_user_content_no_page_context_is_byte_identical():
      # the existing no-tag, no-image path: bare message string, unchanged
      assert cs._build_user_content("hi", None, None) == "hi"
      assert cs._build_user_content("hi", None, None, page_context=None) == "hi"


  def test_build_user_content_page_context_and_attached_text_coexist():
      from api.contracts.chat import PageContext

      out = cs._build_user_content(
          "good start?",
          "Trout .312",
          None,
          page_context=PageContext(page="players", data_json='{"x":1}'),
      )
      assert "Trout .312" in out and '{"x":1}' in out and "good start?" in out


  def test_send_threads_page_context_into_user_turn(monkeypatch):
      from api.contracts.chat import PageContext

      captured = {}
      monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
      monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
      monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
      monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
      monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
      monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
      monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 1)
      monkeypatch.setattr(cs.history, "append_message", lambda *a, **k: 1)
      monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
      monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: None)

      def _chat(**k):
          captured.update(k)
          return {"content": "ok", "tokens_in": 1, "tokens_out": 1, "tool_trace": []}

      monkeypatch.setattr(cs.providers, "chat", _chat)
      ChatService().send(
          chat_user_id=1_000_000_001,
          message="why?",
          model="gpt-5",
          page_context=PageContext(page="streaming", data_json='{"board":[]}'),
      )
      content = captured["messages"][-1]["content"]
      assert "streaming" in content and '{"board":[]}' in content
  ```
- [ ] **Run (expect FAIL)** — `python -m pytest tests/api/test_chat_service.py -v -k "page_context or byte_identical"`.
- [ ] **Minimal impl (service)** — in `api/services/chat_service.py`, change `_build_user_content`:
  ```python
  def _build_user_content(message: str, attached_text: str | None, attachments: list | None, page_context=None):
      """Build the user-turn content. Returns a STRING (today's shape) unless image
      attachments are present, then an OpenAI-style multimodal parts list. Text tags
      are wrapped exactly like the live Streamlit app. An optional page_context
      (what the page is displaying) is prepended as an explicit, truncation-aware
      block; when absent the result is byte-identical to today."""
      prefix = ""
      if page_context is not None:
          page = getattr(page_context, "page", "") or "current"
          data_json = getattr(page_context, "data_json", "") or ""
          if data_json:
              prefix = (
                  f"[Data currently displayed on the {page} page "
                  "(may be truncated — use explain_metric/query_data for more)]\n"
                  f"{data_json}\n\n"
              )

      if attached_text:
          text = prefix + f"[Context the user selected on the page]\n{attached_text}\n\n[Question]\n{message}"
      else:
          text = prefix + message if prefix else message

      image_urls = [
          getattr(a, "data_url", None)
          for a in (attachments or [])
          if getattr(a, "kind", None) == "image" and getattr(a, "data_url", None)
      ]
      if not image_urls:
          return text  # byte-identical to today when no page_context + no text tag + no image
      return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
  ```
  Add `page_context=None` to the `send()` and `send_stream()` signatures (after `viewer_team`), and pass it through at BOTH `convo.append(... _build_user_content(message, attached_text, attachments))` call sites → `_build_user_content(message, attached_text, attachments, page_context)`.
- [ ] **Minimal impl (router)** — in `api/routers/chat.py`, add `page_context=body.page_context,` to BOTH `svc.send(...)` and `svc.send_stream(...)` keyword arg lists.
- [ ] **Minimal impl (system prompt)** — in `src/ai/chat.py::build_system_prompt`, change the middle paragraph to:
  ```python
      return (
          "You are HEATER's in-app fantasy-baseball assistant for a 12-team Yahoo H2H "
          "categories league (R,HR,RBI,SB,AVG,OBP / W,L,SV,K,ERA,WHIP; L/ERA/WHIP are "
          "inverse). Present data and analysis; do not give unsolicited personal trade "
          f"opinions — it is the user's team. The viewer's team is '{team}'. The user is "
          f"currently on the '{page}' page.\n\n"
          "The data currently displayed on the page may be attached to the message. "
          "You can call tools to read any app data — prefer the specific tools; use "
          "query_data (read-only SELECT) for anything else. To explain HOW a number was "
          "computed, use explain_metric (stream_score / dcv / trade_grade / start_score) "
          "and explain_constant / list_constants, and cite the constant value + formula. "
          "Only request_refresh when the user explicitly asks for fresh data.\n\n" + schema
      )
  ```
- [ ] **Failing test (system-prompt mention)** — create `tests/test_ai_chat_prompt.py`:
  ```python
  """build_system_prompt mentions the attached page data + the explainer tools."""

  from src.database import init_db


  def test_system_prompt_mentions_explainers_and_page_data(monkeypatch):
      monkeypatch.setenv("MULTI_USER", "1")
      init_db()
      from src.ai.chat import build_system_prompt

      sp = build_system_prompt("optimizer", "Team Hickey")
      assert "explain_metric" in sp
      assert "explain_constant" in sp
      assert "displayed on the page" in sp
      assert "Team Hickey" in sp and "optimizer" in sp
  ```
- [ ] **Run (expect PASS)** — `python -m pytest tests/api/test_chat_service.py tests/test_ai_chat_prompt.py -v`. Re-confirm the full chat-service suite + the frozen-reference drain are green: `python -m pytest tests/api/test_chat_service.py tests/api/test_chat_service_stream.py tests/test_ai_providers_streaming.py -v`.
- [ ] **Commit** — `git add src/ai/chat.py api/services/chat_service.py api/routers/chat.py tests/api/test_chat_service.py tests/test_ai_chat_prompt.py && git commit -m "feat(api): fold page_context into the chat user turn + explainer-aware system prompt"`

---

## Task 7 — Regenerate openapi.json (single integrator pass for the contract change)

**Files:**
- `api/openapi.json` (generated — do NOT hand-edit)

Steps:

- [ ] **Run regen** — `python scripts/export_openapi.py` (prints `wrote .../api/openapi.json`).
- [ ] **Run snapshot guard (expect PASS)** — `python -m pytest tests/api/test_openapi_contract.py -v`. If it fails, the regen did not run or pins drifted; confirm `fastapi==0.137.1` + `httpx==0.28.1` in `requirements.txt` before debugging anything else.
- [ ] **Sanity check** — `python -c "import json,pathlib; s=json.loads(pathlib.Path('api/openapi.json').read_text(encoding='utf-8')); assert 'PageContext' in s['components']['schemas']; print('PageContext in schema OK')"`
- [ ] **Regenerate TS types** (from `web/`) — `pnpm gen:api` (writes `src/lib/api/generated.ts` from the updated openapi). Then `pnpm exec tsc --noEmit` to confirm the regen is type-clean.
- [ ] **Commit** — `git add api/openapi.json web/src/lib/api/generated.ts && git commit -m "chore(api): regenerate openapi.json + TS types for PageContext"`

---

## Task 8 — Frontend `BubbaContext` provider + publish seam in `usePageData`

**Files:**
- `web/src/components/bubba/BubbaContext.tsx` (new — React context holding `{pageId, data}` + a `publishPageContext` setter + a `useBubbaContext` reader)
- `web/src/lib/use-page-data.ts` (publish `{pageId (from route), data}` on the `loaded` state)
- `web/src/app/layout.tsx` (wrap `{children}` + `<Bubba/>` in `<BubbaContextProvider>`)

Design notes:
- The provider is a `"use client"` component exporting `BubbaContextProvider`, a `useBubbaContext()` hook returning `{ pageId, data, publish }`, where `publish(pageId, data)` updates state. Keep the stored `data` as `unknown` (the page's loaded payload).
- `pageId` comes from the route. In `usePageData`, derive it from `window.location.pathname` (client-only; guard `typeof window`). Map `"/"` → `"team"`; otherwise strip the leading slash (e.g. `/optimizer` → `"optimizer"`). This mirrors the `forcedState()` direct-`window.location` read already in the file (avoids the Next 16 `useSearchParams` CSR-bailout pitfall noted in that file's comments).
- `usePageData` calls `useBubbaContext()` to get `publish`, and inside the `.then((data) => …)` success branch, when it sets `loaded`, also calls `publish(pageId, data)`. Because `usePageData` is now a context consumer, **the `BubbaContextProvider` must wrap every page** — it does, via `layout.tsx`. Guard `publish` defensively: if no provider is mounted (shouldn't happen), `useBubbaContext` returns a no-op `publish` so SSR/tests don't throw.
- Keep this lightweight: do not serialize here (the size-cap + JSON.stringify happens in Bubba at send time, Task 9), so the context just holds the live object.

Steps:

- [ ] **Create the provider** — `web/src/components/bubba/BubbaContext.tsx`:
  ```tsx
  "use client";

  import { createContext, useCallback, useContext, useMemo, useState } from "react";

  /** What Bubba "sees": the current route id + the data that page last loaded. */
  export interface BubbaPageContext {
    pageId: string;
    data: unknown;
  }

  interface BubbaContextValue {
    pageId: string;
    data: unknown;
    publish: (pageId: string, data: unknown) => void;
  }

  const Ctx = createContext<BubbaContextValue>({
    pageId: "",
    data: null,
    // Default no-op publish: if a consumer renders outside the provider (SSR /
    // tests), publishing is a harmless no-op rather than a throw.
    publish: () => {},
  });

  export function BubbaContextProvider({ children }: { children: React.ReactNode }) {
    const [pageCtx, setPageCtx] = useState<BubbaPageContext>({ pageId: "", data: null });
    const publish = useCallback((pageId: string, data: unknown) => {
      setPageCtx({ pageId, data });
    }, []);
    const value = useMemo(
      () => ({ pageId: pageCtx.pageId, data: pageCtx.data, publish }),
      [pageCtx, publish],
    );
    return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
  }

  export function useBubbaContext(): BubbaContextValue {
    return useContext(Ctx);
  }
  ```
- [ ] **Wire `usePageData`** — in `web/src/lib/use-page-data.ts`:
  - Add the import: `import { useBubbaContext } from "@/components/bubba/BubbaContext";`
  - Add a route-id helper near `forcedState()`:
    ```tsx
    /** The current page id for Bubba's context, derived from the route. */
    function currentPageId(): string {
      if (typeof window === "undefined") return "";
      const p = window.location.pathname;
      if (p === "/" ) return "team";
      return p.replace(/^\//, "").split("/")[0] || "team";
    }
    ```
  - Inside `usePageData`, after `const [epoch, setEpoch] = useState(0);`, add `const { publish } = useBubbaContext();`.
  - In the `.then((data) => { ... })` success branch, change the loaded set to also publish:
    ```tsx
        .then((data) => {
          if (!alive || data === undefined) return;
          if (data == null) {
            setState({ status: "empty" });
          } else {
            setState({ status: "loaded", data });
            publish(currentPageId(), data);
          }
        })
    ```
  - Add `publish` to the effect dependency array: `}, [fetcher, epoch, publish]);` (`publish` is `useCallback`-stable, so this does not re-fire the fetch).
- [ ] **Wrap in layout** — in `web/src/app/layout.tsx`:
  - Import: `import { BubbaContextProvider } from "@/components/bubba/BubbaContext";`
  - Wrap the inner content + Bubba (inside `<Providers>`):
    ```tsx
    <Providers>
      <BubbaContextProvider>
        {/* skip link + flex column + TopBar + main-content + Bubba unchanged, now inside the provider */}
        ...
        <Bubba />
      </BubbaContextProvider>
    </Providers>
    ```
    (Move the existing skip-link `<a>`, the `<div className="flex ...">…</div>`, and `<Bubba />` inside `<BubbaContextProvider>` — both `{children}` (via the flex column) and `<Bubba/>` must be descendants so the publisher and the reader share one context.)
- [ ] **Run (expect PASS)** — from `web/`: `pnpm exec tsc --noEmit` then `pnpm run lint`. (No new unit test infra in `web/`; the type+lint gate is the contract.)
- [ ] **Commit** — `git add web/src/components/bubba/BubbaContext.tsx web/src/lib/use-page-data.ts web/src/app/layout.tsx && git commit -m "feat(web): BubbaContext provider + usePageData publishes {pageId,data}"`

---

## Task 9 — Bubba auto-attaches size-capped page_context on every message

**Files:**
- `web/src/lib/api/bubba.ts` (add `page_context` to `ChatSendBody`)
- `web/src/components/bubba/Bubba.tsx` (read `useBubbaContext`; serialize + size-cap; attach to `sendStream`; add a quiet "Page context" toggle defaulting on)

Design notes:
- `ChatSendBody` gets `page_context?: { page: string; data_json: string };`.
- In `BubbaPanel`, read `const { pageId, data } = useBubbaContext();`. Add `const [pageContextOn, setPageContextOn] = useState(true);` (default ON — quiet toggle).
- Add a pure helper (module-level in `Bubba.tsx`) to build the capped JSON:
  ```ts
  const PAGE_CONTEXT_CAP = 16 * 1024; // 16 KB
  function buildPageContext(pageId: string, data: unknown): { page: string; data_json: string } | undefined {
    if (!pageId || data == null) return undefined;
    let json: string;
    try {
      json = JSON.stringify(data);
    } catch {
      return undefined; // non-serializable (cycles) — skip rather than throw
    }
    if (!json) return undefined;
    const data_json = json.length > PAGE_CONTEXT_CAP ? json.slice(0, PAGE_CONTEXT_CAP) + "…[truncated]" : json;
    return { page: pageId, data_json };
  }
  ```
- In `handleSend`, compute `const pageCtx = pageContextOn ? buildPageContext(pageId, data) : undefined;` and add `page_context: pageCtx,` to the `bubba.sendStream({...})` body (alongside `attached_text`/`attachments`). It is attached on EVERY message (including queued — unlike the manual tags which are cleared per-send; the page context reflects the live page, so re-reading it for a queued send is correct). Read `pageId`/`data` fresh inside `handleSend` from the hook values it already closes over; add `pageContextOn`, `pageId`, `data` to the `handleSend` `useCallback` deps.
- Add the toggle to the composer toggle row (next to Web/Research), reusing the existing `<Toggle>` component:
  ```tsx
  <Toggle label="Page" on={pageContextOn} onClick={() => setPageContextOn((v) => !v)} />
  ```
  with `title="Let Bubba see this page's data"` semantics (the `Toggle` already renders `aria-pressed`).

Steps:

- [ ] **Add to ChatSendBody** — in `web/src/lib/api/bubba.ts`, add to the `ChatSendBody` interface (after `attachments`):
  ```ts
    page_context?: { page: string; data_json: string };
  ```
- [ ] **Wire Bubba** — in `web/src/components/bubba/Bubba.tsx`:
  - Import: `import { useBubbaContext } from "./BubbaContext";`
  - Add the module-level `PAGE_CONTEXT_CAP` + `buildPageContext` helper (above `export function Bubba()`).
  - In `BubbaPanel`, add `const { pageId, data } = useBubbaContext();` and `const [pageContextOn, setPageContextOn] = useState(true);`.
  - In `handleSend`, before the `bubba.sendStream(` call, add `const pageCtx = pageContextOn ? buildPageContext(pageId, data) : undefined;` and add `page_context: pageCtx,` to the body object.
  - Add `pageContextOn, pageId, data` to the `handleSend` `useCallback` dependency array.
  - Add the `<Toggle label="Page" .../>` to the composer toggle row.
- [ ] **Run (expect PASS)** — from `web/`: `pnpm exec tsc --noEmit` && `pnpm run lint` && `pnpm build`.
- [ ] **Commit** — `git add web/src/lib/api/bubba.ts web/src/components/bubba/Bubba.tsx && git commit -m "feat(web): Bubba auto-attaches size-capped page_context (default on, quiet toggle)"`

---

## Task 10 — Final verification + live smoke

**Files:** none (verification only)

Steps:

- [ ] **Full backend gate** — `python -m pytest tests/test_ai_tools.py tests/test_ai_explainers.py tests/test_ai_chat_prompt.py tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/api/test_chat_service.py tests/api/test_chat_service_stream.py tests/api/test_openapi_contract.py tests/api/test_chat_tool_surface_readonly.py -v` — ALL green. Pay attention to: the frozen-reference `test_chat_is_a_drain_returning_todays_exact_dict` (live Streamlit path unchanged), the openapi snapshot, and the readonly tool-surface guard (`tests/api/test_chat_tool_surface_readonly.py` — confirm the three explainers are read-only/side-effect-free per its expectations; if it enumerates an allowed tool list, add the explainers there).
- [ ] **Frontend gate** — from `web/`: `pnpm build` && `pnpm exec tsc --noEmit` && `pnpm run lint`.
- [ ] **Live smoke (per the live-verification local-stack memory)** — start the API: `python -m uvicorn api.main:create_app --factory --port 8000` (it is a FACTORY). Then drive the React preview against it. Ask Bubba (a) "what's on this page?" — confirm the response reflects the attached page data; (b) "how was this stream score computed?" on `/streaming` — confirm Bubba calls `explain_metric`/`explain_constant` and cites the registry weights. Live-only Yahoo fields degrade gracefully on the empty local DB; the explainer tools are pure and work regardless.
- [ ] **Merge + push** (per the always-merge-and-push standing rule) — `git checkout master && git merge --no-ff <branch> && git push origin master`. Then run silent-failure-hunter on the non-trivial changes (the `explain_metric` assembly + the `_build_user_content` page-context folding) per the post-merge standing rule.

---

## Done-when

- `explain_constant`, `list_constants`, `explain_metric` are registered + dispatched in `src/ai/tools.py`, unit-tested DB-free, and graceful on bad input.
- The `chat()` frozen-reference drain test is GREEN (live Streamlit path provably unchanged); an additive equivalence test proves the explainers are additive.
- `ChatSendRequest.page_context` (`PageContext{page, data_json}`) is in the contract + `api/openapi.json` + `generated.ts`; `ChatService` folds it into the user turn (byte-identical when `None`); `build_system_prompt` mentions the attached page data + the explainer tools.
- The frontend `BubbaContextProvider` wraps `{children}` + `<Bubba/>`; `usePageData` publishes `{pageId, data}` on `loaded`; Bubba auto-attaches a ≤16 KB JSON `page_context` on every message with a default-on quiet toggle.
- `pnpm build` + `tsc` + `lint` pass; backend suite green; openapi snapshot current.
