# HEATER AI Chat Assistant — Design Spec

**Date:** 2026-06-11
**Status:** Approved (architecture + UX), pending spec review
**Author:** Connor + Claude
**Scope:** Embedded, persistent, multi-provider AI chat assistant on every HEATER page, connected to all app data via tool-use.

---

## 1. Goal

Add an AI chat assistant available on every page of the HEATER app that:

- Persists the conversation across page navigation (and across logout / device, per user).
- Lets each user bring their own API keys for multiple providers/models, switchable per message.
- Is connected to **all** app data — every SQLite table, every projection/stat/manually-computed figure, every refreshed or scraped source — through a tool surface over the existing service layer plus a guarded read-only SQL tool.
- Can **read, reference, and request refresh of** any data, plus (Phase 2) search the web and run a deep-research loop.
- Saves every user's conversation history, browsable from a dropdown in the chat window.
- (Phase 2) Lets the user toggle a mode to highlight any text on the page and attach it as direct context.

This spec covers **Phase 1 (MVP)** in implementation detail and **Phase 2** at outline resolution. Phase 1 ships and is verified before Phase 2 begins.

---

## 2. Resolved decisions

These were settled during brainstorming (research-backed; see `docs/superpowers/specs/` research notes and the 2026-06-11 deep-research run, 22 verified claims / 31 sources):

| Question | Decision | Rationale |
|---|---|---|
| **RAG vs tools** | **Tool-use over the service layer + one guarded read-only SQL tool.** No conversion of structured DBs to vector stores. | Industry consensus for structured data is function-calling, not RAG. RAG/vectors only earn their keep on unstructured text. |
| **Vector / FTS** | **SQLite FTS5** over unstructured text (news blurbs, player notes) only — Phase 2. No external vector DB. | FTS5 ships with SQLite, zero new infra. Covers the only genuinely unstructured corpus. |
| **Multi-provider** | **LiteLLM** (one OpenAI-compatible interface to Anthropic / OpenAI / Gemini / OpenRouter / Ollama). | Mature (50.1k★, v1.88.1 Jun 2026); all 5 target providers confirmed supported; built-in `completion_cost()`. |
| **Key encryption** | **Fernet**, mirroring `src/token_relay.py`. | Already the project's at-rest-secret pattern; symmetric authenticated encryption. |
| **No-key default** | **Admin shared key + per-user cost caps.** Users can override with their own key. | 12 non-technical leaguemates can chat immediately; cost stays bounded. |
| **Web search** | **App-side search tool** (DDGS free default, optional Tavily/Brave key) — Phase 2. | Works across every provider/model (provider-native search only works for that provider's models). |
| **Deep research** | App-side plan→search→fetch→synthesize loop, gated behind an explicit button — Phase 2. | Each run costs real tokens; the same loop pattern as GPT-Researcher. |
| **Highlight-attach** | Phase 2; custom **Components v2** (no-iframe) component using `window.getSelection()`. | Iframe components can't read the host page's selection; v2 runs without an iframe. |
| **Local model** | `qwen2.5:7b` (installed) for local use; `qwen3:30b-a3b` recommended pull (MoE, 3B active). Exposed to cloud users only via mini-PC tunnel — Phase 2. | Strongest open tool-caller; >8 tok/s on the Core Ultra 9 185H CPU. |
| **Rollout** | **Phased** — Phase 1 MVP, then Phase 2. | Lower risk; value sooner. |
| **Gating** | MULTI_USER-gated, inert when the flag is off (v1 byte-for-byte), exactly like `feedback`/`usage`/admin surfaces. | Preserves the project's hard v1-unchanged invariant. Local testing runs flag-on. |

---

## 3. Architecture

A new `src/ai/` sub-package (mirrors `src/optimizer/`, `src/engine/`). Each module has one clear job and a narrow interface.

```
src/ai/
  __init__.py
  keys.py          BYOK store: Fernet-encrypted per-user keys + admin shared key. Mirrors token_relay.
  providers.py     LiteLLM wrapper: chat(model, messages, tools, stream) → unified response/stream.
  router.py        Tier (simple|moderate|complex) → model map; admin-configurable defaults.
  schema_card.py   Builds a compact, cached text description of the DB schema for the system prompt.
  tools.py         Tool definitions + dispatch over the existing service layer.
  sql_tool.py      Guarded read-only SELECT executor (the "reach all data" tool).
  budget.py        Per-user token/cost caps + usage ledger writes.
  history.py       Save / load / list / rename / delete conversations in SQLite.
  chat.py          render_chat_widget(): orchestrates the fragment, calls providers, streams replies.
  chat_shell.py    Injects the window chrome (launcher button, drag/resize/minimize/close JS+CSS).
  search.py        [Phase 2] web_search (DDGS/Tavily/Brave) + deep_research loop.
  selection/       [Phase 2] custom no-iframe component for highlight-to-attach.
```

### 3.1 Message flow (Phase 1)

```
User types in the floating window (a single @st.fragment)
   │
   ▼
chat.py: append user msg to session_state + SQLite (history.py)
   │
   ▼
budget.py: check the user's remaining cap → block with a friendly message if exceeded
   │
   ▼
router.py: resolve the selected tier/model → providers.py (LiteLLM)
   │   system prompt = static persona + schema_card (cached) + viewer team context
   │   tools = tools.py definitions
   ▼
Agentic loop (providers.py):
   call model → if tool_calls: dispatch via tools.py (read tools + request_refresh + query_data)
              → append tool results → call again
              → else: stream the final answer via st.write_stream
   │
   ▼
budget.py: record tokens + cost (completion_cost()) into ai_usage_ledger
   │
   ▼
history.py: persist assistant msg; conversation stays in session_state (survives page nav)
```

### 3.2 Persistence model

- **Within a browser session, across pages:** `st.session_state["ai_chat"]` holds the active conversation. Streamlit `session_state` already survives `st.navigation` page changes — the conversation is *not* cleared when the user changes pages. (Confirmed: Streamlit's documented chat pattern stores message history in `session_state`.)
- **Across logout / device / restart:** every message is written through to SQLite (`ai_conversations` + `ai_messages`), keyed to `user_id`. The Conversations dropdown lists the user's saved threads; selecting one loads it back into `session_state`.

### 3.3 Why almost nothing is rebuilt

The existing service layer **is** the ideal tool surface — we wrap it, not replace it:

- `load_player_pool()` → player/stat reads
- `get_yahoo_data_service()` → rosters / standings / matchup / free agents / transactions
- `evaluate_trade(...)` → trade evaluation
- `LineupOptimizerPipeline(...)` → lineup optimization
- `get_connection()` (read-only variant) → the open-ended SQL tool

Only **three** additions to existing systems:
1. `forced_refresh_queue` table + one new step in the scheduler loop (§7).
2. FTS5 virtual tables over news text (Phase 2).
3. Per-page `render_chat_widget()` mount + a structural guard test (§9).

---

## 4. UI / UX — the chat window

### 4.1 Launcher

- A button labeled **"AI Chat"** (flame icon) pinned to the **top-right of every page**.
- Rendered by `chat_shell.py`. Clicking it reveals the window. The window DOM already exists (hidden) — opening is a **client-side** CSS toggle, **no Streamlit rerun**.

### 4.2 The window

A floating panel built on `streamlit-float` (a viewport-pinned Streamlit container) whose **contents are real Streamlit widgets** (so streaming, model picker, history dropdown, and chat input stay native and interactive). `chat_shell.py` injects CSS + JS to give it window behaviors:

| Control | Behavior | Rerun? |
|---|---|---|
| **Drag** | Title bar is the drag handle (pointer events move the container). | No (client-side) |
| **Resize** | Both axes — bottom-right grip / CSS `resize: both`. | No |
| **Minimize** | Collapses to just the title bar (or back to the launcher). | No |
| **Close** | Hides the window, shows the launcher. | No |
| **New chat (+)** | Starts a fresh conversation. | Fragment rerun |
| **Conversations ▾** | Dropdown of the user's saved threads; load one. | Fragment rerun |
| **Model picker** | Per-message model/tier selection. | No (read at send time) |
| **Send** | Sends the message. | **Fragment rerun only** (chat panel, not the page) |

- **Window position/size/open-state persist in `localStorage`** (client-side) so they survive reloads without a rerun. Open-state is also mirrored into `session_state` so the correct initial state renders server-side.
- The chat panel is wrapped in `@st.fragment` so a sent message reruns only the panel, never the whole page (confirmed Streamlit behavior).
- Replies render with `st.write_stream` (typewriter streaming).
- Styling follows the Combustion Index design system (`src/ui_shared.py` THEME tokens, orange `#ff6d00` primary, no emoji, inline SVG/Material-symbol icons) and is covered by the combustion lock conventions.

### 4.3 Mounting

Every interactive page calls `render_chat_widget()` once, immediately after `require_auth()` (and after `inject_custom_css()`), exactly mirroring the `render_feedback_widget()` / `log_page_view()` convention. A structural guard test enforces the call on every interactive page (§9). The widget is MULTI_USER-gated and inert when the flag is off.

---

## 5. Data model (new, additive)

All tables created in `init_db()` via `CREATE TABLE IF NOT EXISTS` (idempotent), same pattern as `feedback` / `usage_events` / the Plan-3 admin tables. All access through `get_connection()`.

```sql
CREATE TABLE IF NOT EXISTS ai_provider_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    provider      TEXT NOT NULL,          -- anthropic | openai | gemini | openrouter | ollama
    label         TEXT,                   -- user-facing nickname
    encrypted_key TEXT NOT NULL,          -- Fernet ciphertext
    created_at    TEXT NOT NULL,
    UNIQUE(user_id, provider, label)
);

CREATE TABLE IF NOT EXISTS ai_conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    title       TEXT NOT NULL,            -- auto-generated from first message, renamable
    model       TEXT,                     -- last model used
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES ai_conversations(id),
    role            TEXT NOT NULL,        -- user | assistant | tool | system
    content         TEXT NOT NULL,        -- JSON for tool/structured turns
    model           TEXT,
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_usage_ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(user_id),
    day        TEXT NOT NULL,             -- YYYY-MM-DD (UTC)
    tokens_in  INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd   REAL DEFAULT 0.0,
    UNIQUE(user_id, day)
);

CREATE TABLE IF NOT EXISTS forced_refresh_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,           -- a bootstrap source name, or 'all'
    requested_by INTEGER REFERENCES users(user_id),
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | error
    detail       TEXT,
    created_at   TEXT NOT NULL,
    completed_at TEXT
);
```

Admin-configurable settings live in the existing `app_settings` table (no new table): the default shared provider/model, the per-tier model map, and the per-user daily/monthly cost cap.

Phase 2 adds FTS5 virtual tables (e.g. `news_fts`) rebuilt after each news refresh.

---

## 6. Tool surface

Tools are defined in OpenAI function-calling schema (LiteLLM's lingua franca) in `tools.py` and dispatched to the existing service layer. The model receives a **schema-card** (`schema_card.py`) in its system prompt describing the DB so it can drive `query_data` accurately.

| Tool | Backed by | Returns | Side effect |
|---|---|---|---|
| `get_player(name_or_id)` | `load_player_pool()` | player row(s) incl. projections, YTD, statcast | none |
| `compare_players(ids)` | pool | side-by-side | none |
| `get_my_team()` | `get_yahoo_data_service().get_rosters()` + viewer team resolver | the viewer's roster + category standings | none |
| `get_standings()` / `get_matchup()` | yds | league standings / weekly matchup | none |
| `get_free_agents(filters)` | `yds.get_free_agents()` | FA list | none |
| `evaluate_trade(give, get)` | `trade_evaluator.evaluate_trade` | grade + deltas | none |
| `optimize_lineup(scope)` | `LineupOptimizerPipeline` | recommended lineup | none |
| `query_data(sql)` | **`sql_tool.py`** (read-only) | rows (capped) | none |
| `request_refresh(source)` | enqueue to `forced_refresh_queue` | acknowledgement + status | **write via queue** |
| `web_search(q)` / `deep_research(q)` | `search.py` | results / cited report | external [Phase 2] |

### 6.1 Guarded read-only SQL (`sql_tool.py`)

This is what makes the AI "100% connected to all data," including data that does not exist yet (new tables are automatically queryable):

- **Read-only connection:** open via `sqlite3.connect("file:<db>?mode=ro", uri=True)` — the OS/driver enforces read-only; writes are impossible even if a guard is bypassed.
- **SELECT-only allowlist:** reject anything whose first significant token isn't `SELECT` or `WITH`; reject multiple statements (no `;` chaining); reject `PRAGMA`, `ATTACH`, `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE`.
- **Row + time caps:** auto-append/enforce a `LIMIT` (e.g. 500) and set a busy/statement timeout.
- **Schema-card prompting:** the system prompt lists tables + columns so the model targets real columns (mitigates text-to-SQL's hallucinated-column / wrong-join failure modes).
- Errors return a structured `is_error` tool result so the model can self-correct rather than crash.

---

## 7. Scheduler bridge (the only write path)

The scheduler thread is the **sole SQLite writer** (single-replica invariant). The AI must never write directly. Instead:

- `request_refresh(source)` inserts a `pending` row into `forced_refresh_queue` and returns immediately with the row id/status.
- `src/scheduler.py::_refresh_once()` gets one new step at the top of each tick: drain `forced_refresh_queue` — for each `pending` row, mark `running`, call `bootstrap_all_data(force=True)` (or a single-source variant if/when one exists), mark `done`/`error`, stamp `completed_at`.
- The chat can poll the row's status and tell the user when the refresh lands.
- Under MULTI_USER the in-process scheduler already runs this loop; the new step protects all 12 users and respects the single-writer rule. Member sessions remain read-only (`viewer_can_write()` unchanged) — enqueuing is a normal insert the scheduler later acts on, not a member-initiated bootstrap.

---

## 8. Providers, keys, router, budget

### 8.1 Keys (`keys.py`)

- Encrypt/decrypt with Fernet using a key from an env var (e.g. `HEATER_AI_KEY`), mirroring `token_relay._fernet()`. If the env key is absent, BYOK storage is disabled with a clear admin message (fail closed, never store plaintext).
- `get_key(user_id, provider)` → the user's key if set, else the admin shared key (from `app_settings`), else `None`.
- Admin sets the shared key in Admin Controls (`pages/_admin_controls.py`), audit-logged with the **action only**, never the key text (mirrors the Plan-4 `yahoo_token_update` pattern + its AST guard).
- A per-user **AI Settings popover** inside the chat window (a gear icon in the title bar) lets each user add / label / remove their own provider keys.

### 8.2 Providers (`providers.py`)

- Thin wrapper around `litellm.completion(...)` exposing `chat(model, messages, tools=None, stream=False, api_key=...)`.
- Handles the agentic tool loop (call → dispatch tools → call) and the final streamed answer.
- Maps provider errors (auth, rate-limit, context-exceeded) to friendly chat messages; never leaks stack traces or keys.
- Ollama routes to a configurable base URL (local during dev; the mini-PC tunnel in Phase 2).

### 8.3 Router (`router.py`)

Default tier → model map (admin-overridable via `app_settings`), using verified June-2026 options:

| Tier | Default | Alternatives |
|---|---|---|
| **simple** | `claude-haiku-4-5` ($1/$5) | Gemini Flash, GPT-5.x-mini |
| **moderate** | `claude-sonnet-4-6` ($3/$15) | — |
| **complex** | `claude-opus-4-8` ($5/$25) | `claude-fable-5` ($10/$50) |

The user can also pick a specific model directly in the window; the tier is the default when they don't.

### 8.4 Budget (`budget.py`)

- Per-user daily and monthly USD caps (admin-set in `app_settings`; sensible defaults).
- Before each call: check `ai_usage_ledger` for the user's spend; if over cap, return a friendly "you've hit your AI limit for today" message instead of calling the model.
- After each call: add `completion_cost()` + token counts to the ledger. (Note: cost is computed via the explicit `completion_cost()` helper — there is no guaranteed automatic cost field.)
- Caps apply to the **admin shared key**. A user on their **own** key is exempt from the shared-key cap by default (they pay their own provider directly); the admin can still set an optional own-key ceiling.

---

## 9. Testing strategy

Follows the project's structural-invariant + unit-test conventions.

**Structural guards (string/AST-based, like `test_pages_have_feedback_and_usage.py`):**
- `test_pages_have_ai_chat.py` — every interactive page imports + calls `render_chat_widget()` after `require_auth()`.
- `test_ai_chat_backcompat.py` — MULTI_USER off ⇒ `render_chat_widget()` is a no-op, zero DB writes, no launcher (v1 byte-for-byte).
- `test_ai_tables.py` — `init_db()` creates the 5 new tables idempotently; all start empty/inert.
- `test_ai_sql_tool_readonly.py` — `sql_tool` rejects non-SELECT, multi-statement, PRAGMA/ATTACH/DDL/DML; enforces LIMIT; opens read-only (a write attempt raises).
- `test_ai_keys_encrypted.py` — keys are Fernet-encrypted at rest; plaintext never written; admin audit logs the action, never the key (AST-checked).
- `test_forced_refresh_queue.py` — `request_refresh` enqueues; the scheduler step drains pending→done; members never write directly.
- `test_ai_admin_key_gated.py` — only admins can set the shared key; `require_admin()` enforced.

**Unit tests:**
- `router` tier→model mapping + admin overrides.
- `budget` cap enforcement (block when over) + ledger accounting.
- `history` save/load/list/rename/delete round-trips, scoped to `user_id`.
- `providers` agentic loop with a mocked LiteLLM (no network; honors the conftest network guard).
- `schema_card` reflects current tables.

All network-touching tests honor the existing `tests/conftest.py` network guard (no real outbound sockets).

---

## 10. Dependencies

Add to `requirements.txt` (and the Railway `Dockerfile`):

- **Phase 1:** `litellm` (core; pulls a moderate dep tree — pin a known-good version; verify Python 3.12 CI + 3.14 local). `streamlit-float` for the floating window container (small, MIT — evaluate vendoring vs pip; the §14 fallback is a CSS-pinned container + drag/resize JS we own).
- **Phase 2:** `ddgs` (free web search), optional `tavily-python` (paid upgrade).

No external vector DB, no managed RAG service, no new datastore. FTS5 is built into SQLite.

---

## 11. Security & safety

- Keys encrypted at rest (Fernet); decrypted only in-process at call time; never logged, never sent to the client, never in audit text.
- Read-only SQL connection + SELECT allowlist + row/time caps contain text-to-SQL risk.
- The AI cannot write to the DB; refresh is mediated by the single-writer scheduler via a queue.
- MULTI_USER-gated and inert when off — preserves the v1 byte-for-byte invariant the whole codebase guards.
- Per-user cost caps bound spend on the admin shared key.
- Provider errors are surfaced as friendly messages; no stack traces or secrets leak to the chat.
- Conversations are scoped to `user_id`; no cross-user leakage.

---

## 12. Phase split

**Phase 1 (MVP) — this spec's implementation target:**
Floating draggable/resizable/minimizable window + top-right "AI Chat" launcher; persistent conversation (session_state + SQLite); BYOK + admin shared key (Fernet); tier router; read-only data tools + guarded SQL; `request_refresh` queue + scheduler step; Conversations history dropdown; per-user cost caps; structural + unit tests; deps + Dockerfile.

**Phase 2:**
`web_search` + `deep_research` loop; highlight-to-attach (Components v2, no-iframe); FTS5 over news/notes; local-model mini-PC tunnel; optional richer per-user analytics of AI usage in the Admin console.

---

## 13. Out of scope (for now)

- Voice input/output.
- Image generation.
- Fine-tuning or training any model.
- A standalone mobile app (the existing responsive web UI applies).
- Multi-tenant billing/invoicing (cost caps only).

---

## 14. Open risks & mitigations

| Risk | Mitigation |
|---|---|
| Draggable/resizable window fighting Streamlit reruns | All window chrome (drag/resize/min/close/open) is client-side JS with `localStorage`; only message-send triggers a *fragment* rerun. |
| LiteLLM dep weight / Python 3.14 local compat | Pin a known-good version; verify install on both 3.12 (CI) and 3.14 (local) before merge; it's pure-Python-friendly. |
| Text-to-SQL hallucination | Read-only conn + SELECT allowlist + schema-card + row caps + self-correcting error results. |
| Admin shared-key cost runaway | Per-user daily/monthly caps enforced before each call. |
| `streamlit-float` maintenance/compat (Phase 2) | Evaluate vendoring the small JS, or fall back to a CSS-pinned container + custom drag/resize JS we own. |
| Highlight-attach (Phase 2) needs no-iframe component | Components v2 documented `setStateValue`/`setTriggerValue` channel; prototype early in Phase 2. |

---

## 15. Success criteria (Phase 1)

1. "AI Chat" launcher appears top-right on every interactive page; opens a draggable, resizable, minimizable, closable window.
2. The conversation does **not** clear when navigating between pages.
3. A user can store keys for ≥2 providers and switch models per message; with no key, the admin shared key works (within caps).
4. The AI correctly answers a question that requires reading league data (e.g. "what's my SB rank?") by calling a tool — verified in the browser.
5. The AI can run an arbitrary read-only `query_data` and is provably unable to write.
6. `request_refresh` enqueues and the scheduler completes it; the chat reports completion.
7. History dropdown lists and reloads prior conversations for that user.
8. Cost caps block a user who exceeds the limit, with a friendly message.
9. Full test suite green, including the new structural guards; MULTI_USER-off path byte-for-byte unchanged.
