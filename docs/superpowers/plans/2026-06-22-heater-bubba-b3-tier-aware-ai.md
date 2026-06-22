# Bubba B3 — Tier-Aware Managed AI + Upgrade Nudge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Meter managed Bubba chat by subscription tier (Free gets a small daily taste, Pro a bigger allowance, own-key unlimited) and nudge a capped user to upgrade or add their own key — dormant until billing is turned on.

**Architecture:** `src/ai/budget.is_over_cap` gains an optional `cap_usd` override (no-op for the live Streamlit app). A new `api/services/ai_allowance.py` maps tier → daily managed cap. A `get_managed_ai_cap` dependency in `api/gating.py` (alongside `require_pro`, reusing its `verifier + SubscriptionStore` pattern — avoids the `deps↔identity` import cycle) resolves the caller's cap, returning `None` while billing is dormant. The chat router passes it to `ChatService` → `budget`. The over-cap streaming error gains `code:"over_cap"`; the frontend renders an upgrade nudge.

**Tech Stack:** FastAPI 0.137.1 (pinned), Pydantic v2, `src/ai/` (budget), pytest; Next.js 16 + React 19 + TS. No new deps.

**Coordination:** `api/` + one `src/ai/` function + `web/`. Two slices (A backend → B frontend). Dormant + additive — the live app is unchanged until `billing_env_configured()` is true.

---

## Grounding (verified in source — do NOT re-derive)

- `src/ai/budget.py::is_over_cap(user_id, on_own_key=False)`: `on_own_key` → `False`; else `spent_today(user_id) >= daily_cap_usd()`. **Used by the live Streamlit chat too** — keep the no-`cap_usd` path byte-identical.
- `api/gating.py` holds `require_pro` + `stripe_enabled()` (== `billing_env_configured()`), using `Header(authorization)` + `Depends(get_auth_verifier)` + `Depends(get_subscription_store)` and resolving `verifier.verify(authorization).clerk_user_id`. **`get_managed_ai_cap` mirrors this exactly** (gating.py imports `api.deps`/`api.auth`/`api.billing_config`/`api.stores.subscription_store` — no `api.identity`, so no cycle).
- `api/services/chat_service.py`: `send` + `send_stream` compute `on_own_key` then `budget.is_over_cap(chat_user_id, on_own_key=on_own_key)`. `send` over-cap → `self._empty("You've reached today's usage limit. Add your own API key for unlimited use.", conversation_id)`; `send_stream` over-cap → `yield _sse({"type": "error", "message": "You've reached today's usage limit. Add your own API key for unlimited use."})` then `return`. `ChatService.__init__(prompt_store=None)` exists (B2.4).
- `api/routers/chat.py`: `send` + `send_stream` are thin; `_uid(app_user)` 401s on `None`. Endpoint tests override deps in `tests/api/test_api_chat.py` (`_FakeChatService`, `_FakeUser`).
- The streaming `error` event is a plain SSE dict (no pydantic model). `web/src/lib/api/bubba.ts::BubbaStreamEvent` error variant is `{ type: "error"; message: string }`. `web/src/components/bubba/Bubba.tsx`: `UiMessage{role,content,isError?}`; the transcript maps `messages` to `<Bubble>`; the `error` event sets the last assistant turn `{isError:true, content:e.message}`; `setShowSettings(true)` opens the key panel.

**Venv python:** `.venv/Scripts/python.exe`. Frontend from `web/` with `pnpm`. Git from repo root `C:/Users/conno/Code/HEATER_v1.0.1`.

---

## File Structure

- **Modify** `src/ai/budget.py` — `is_over_cap` gains `cap_usd`.
- **Create** `api/services/ai_allowance.py` — tier → daily cap.
- **Modify** `api/gating.py` — `get_managed_ai_cap` resolver.
- **Modify** `api/services/chat_service.py` — `send`/`send_stream` thread `cap_usd` + `over_cap` code.
- **Modify** `api/routers/chat.py` — wire `Depends(get_managed_ai_cap)`.
- **Modify** `api/openapi.json` — regenerate (the auth-header dep may surface on the 2 send routes).
- **Modify** `web/src/lib/api/bubba.ts` — `BubbaStreamEvent.code?`.
- **Modify** `web/src/components/bubba/Bubba.tsx` — `over_cap` nudge.
- **Create** `tests/test_ai_budget_cap.py`, `tests/api/test_ai_allowance.py`, `tests/api/test_managed_ai_cap.py`; **Modify** `tests/api/test_chat_service.py`, `tests/api/test_api_chat.py`.

---

## SLICE A — backend

### Task 1: `budget.is_over_cap(cap_usd)` + the live-app guard

**Files:** Modify `src/ai/budget.py`; Test `tests/test_ai_budget_cap.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ai_budget_cap.py
"""is_over_cap's cap_usd override + the live-app-safe default (DB-free via
monkeypatched spent_today/daily_cap_usd)."""

import src.ai.budget as b


def test_no_cap_usd_uses_admin_cap_unchanged(monkeypatch):
    # The live Streamlit app calls is_over_cap(uid[, on_own_key]) with NO cap_usd —
    # this must stay identical to today (uses daily_cap_usd()).
    monkeypatch.setattr(b, "spent_today", lambda uid: 0.5)
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 1.0)
    assert b.is_over_cap(1) is False  # 0.5 < 1.0
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 0.4)
    assert b.is_over_cap(1) is True  # 0.5 >= 0.4


def test_cap_usd_override_ignores_admin_cap(monkeypatch):
    monkeypatch.setattr(b, "spent_today", lambda uid: 0.5)
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 999.0)  # ignored when cap_usd given
    assert b.is_over_cap(1, cap_usd=0.10) is True  # 0.5 >= 0.10
    assert b.is_over_cap(1, cap_usd=2.0) is False  # 0.5 < 2.0


def test_own_key_unlimited_regardless_of_cap(monkeypatch):
    monkeypatch.setattr(b, "spent_today", lambda uid: 9999.0)
    assert b.is_over_cap(1, on_own_key=True, cap_usd=0.01) is False
```

- [ ] **Step 2: Run — expect FAIL** (`is_over_cap() got an unexpected keyword argument 'cap_usd'`)

Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_budget_cap.py -q`

- [ ] **Step 3: Implement** — `src/ai/budget.py`, replace `is_over_cap`:

```python
def is_over_cap(user_id: int, on_own_key: bool = False, cap_usd: float | None = None) -> bool:
    if on_own_key:
        return False
    cap = cap_usd if cap_usd is not None else daily_cap_usd()
    return spent_today(user_id) >= cap
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_budget_cap.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/ai/budget.py tests/test_ai_budget_cap.py
git commit -m "feat(ai): is_over_cap optional cap_usd override (Bubba B3 slice A)"
```

---

### Task 2: `ai_allowance.py` (tier → daily cap)

**Files:** Create `api/services/ai_allowance.py`; Test `tests/api/test_ai_allowance.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_ai_allowance.py
from api.services.ai_allowance import managed_cap_for_tier


def test_pro_cap_exceeds_free_and_both_positive():
    assert managed_cap_for_tier("pro") > managed_cap_for_tier("free") > 0


def test_unknown_tier_gets_free_cap():
    assert managed_cap_for_tier("mystery") == managed_cap_for_tier("free")
```

- [ ] **Step 2: Run — expect FAIL.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_ai_allowance.py -q`

- [ ] **Step 3: Implement**

```python
# api/services/ai_allowance.py
"""Per-tier DAILY managed-AI cap (Bubba B3, lean). The shared-key ('managed')
daily $-allowance per subscription tier. BYO-key users are exempt upstream
(budget.is_over_cap's on_own_key short-circuit). Values are tunable; Plus/Max
slot in here when those tiers ship."""

from __future__ import annotations

_FREE_DAILY_CAP_USD = 0.10  # a small daily "taste" of managed AI on a cheap model
_TIER_DAILY_CAP_USD: dict[str, float] = {
    "free": _FREE_DAILY_CAP_USD,
    "pro": 2.00,
}


def managed_cap_for_tier(tier: str) -> float:
    """Daily managed-AI $ cap for a subscription tier. Unknown tier -> the free cap."""
    return _TIER_DAILY_CAP_USD.get(tier, _FREE_DAILY_CAP_USD)
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_ai_allowance.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/services/ai_allowance.py tests/api/test_ai_allowance.py
git commit -m "feat(api): ai_allowance per-tier managed-AI cap (Bubba B3 slice A)"
```

---

### Task 3: `get_managed_ai_cap` resolver (dormancy-aware)

**Files:** Modify `api/gating.py`; Test `tests/api/test_managed_ai_cap.py`

- [ ] **Step 1: Write the failing tests** (call the resolver directly with fakes — DB-free, no FastAPI)

```python
# tests/api/test_managed_ai_cap.py
import api.gating as g
from api.gating import get_managed_ai_cap
from api.services.ai_allowance import managed_cap_for_tier


class _Sub:
    def __init__(self, tier):
        self.tier = tier


class _Store:
    def __init__(self, sub=None, boom=False):
        self._sub, self._boom = sub, boom

    def get(self, clerk):  # noqa: ARG002
        if self._boom:
            raise RuntimeError("db down")
        return self._sub


class _Verifier:
    def __init__(self, clerk):
        self._clerk = clerk

    def verify(self, authorization):  # noqa: ARG002
        if self._clerk is None:
            raise RuntimeError("401")
        return type("P", (), {"clerk_user_id": self._clerk})()


def test_dormant_billing_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: False)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(_Sub("pro"))) is None


def test_live_pro_returns_pro_cap(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(_Sub("pro"))) == managed_cap_for_tier("pro")


def test_live_no_subscription_returns_free_cap(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(None)) == managed_cap_for_tier("free")


def test_live_unverifiable_caller_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap(None, _Verifier(None), _Store(_Sub("pro"))) is None


def test_live_store_error_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(boom=True)) is None
```

- [ ] **Step 2: Run — expect FAIL** (`cannot import name 'get_managed_ai_cap'`)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_managed_ai_cap.py -q`

- [ ] **Step 3: Implement** — `api/gating.py`. Add the imports at the top (after the existing imports):
```python
import logging

from api.services.ai_allowance import managed_cap_for_tier

logger = logging.getLogger(__name__)
```
and append the resolver:
```python
def get_managed_ai_cap(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
    store: SubscriptionStore = Depends(get_subscription_store),
) -> float | None:
    """The caller's DAILY managed-AI cap (shared-key spend), or None while billing
    is dormant -> budget.is_over_cap falls back to the admin cap (today's behavior).
    Never raises: an unverifiable caller / store error -> None (safe admin-cap
    fallback; the chat route's own require_app_user owns the 401)."""
    if not stripe_enabled():
        return None
    try:
        clerk = verifier.verify(authorization).clerk_user_id
    except Exception:
        return None  # unauthenticated/invalid — the route 401s independently
    if not clerk:
        return managed_cap_for_tier("free")
    try:
        sub = store.get(clerk)
    except Exception:
        logger.warning("get_managed_ai_cap subscription read failed", exc_info=True)
        return None
    return managed_cap_for_tier(sub.tier if sub else "free")
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_managed_ai_cap.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/gating.py tests/api/test_managed_ai_cap.py
git commit -m "feat(api): get_managed_ai_cap dormancy-aware tier->cap resolver (Bubba B3 slice A)"
```

---

### Task 4: Thread `cap_usd` through `ChatService` + the router + `over_cap` code

**Files:** Modify `api/services/chat_service.py`, `api/routers/chat.py`, `api/openapi.json`; Test `tests/api/test_chat_service.py`, `tests/api/test_api_chat.py`

- [ ] **Step 1: Write the failing service tests** — append to `tests/api/test_chat_service.py`:

```python
def test_send_stream_over_cap_emits_over_cap_code(monkeypatch):
    import json

    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])  # no own key -> managed
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: True)
    frames = list(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5", cap_usd=0.10))
    first = json.loads(frames[0][len("data: ") : -2])
    assert first["type"] == "error" and first["code"] == "over_cap"


def test_send_stream_threads_cap_usd_into_budget(monkeypatch):
    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])

    def _over(uid, on_own_key=False, cap_usd=None):
        captured["cap_usd"] = cap_usd
        return True  # short-circuit (over cap) so we don't need a fake provider

    monkeypatch.setattr(cs.budget, "is_over_cap", _over)
    list(ChatService().send_stream(chat_user_id=1, message="hi", model="gpt-5", cap_usd=0.42))
    assert captured["cap_usd"] == 0.42
```

- [ ] **Step 2: Run — expect FAIL** (`send_stream() got an unexpected keyword argument 'cap_usd'`)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`

- [ ] **Step 3: Thread `cap_usd` into `ChatService.send`.** Add the param to `send`'s signature (after `attachments`):
```python
        attached_text: str | None = None,
        attachments: list | None = None,
        cap_usd: float | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> dict:
```
and change `send`'s over-cap check + message:
```python
            on_own_key = any(k.get("provider") == provider for k in keys.list_keys(chat_user_id))
            if budget.is_over_cap(chat_user_id, on_own_key=on_own_key, cap_usd=cap_usd):
                return self._empty(
                    "You've hit today's AI limit. Upgrade for more, or add your own key for unlimited use.",
                    conversation_id,
                )
```

- [ ] **Step 4: Thread `cap_usd` into `send_stream`** — add the param (after `attachments`):
```python
        attached_text: str | None = None,
        attachments: list | None = None,
        cap_usd: float | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> Generator[str, None, None]:
```
and change its over-cap branch:
```python
            on_own_key = any(k.get("provider") == provider for k in keys.list_keys(chat_user_id))
            if budget.is_over_cap(chat_user_id, on_own_key=on_own_key, cap_usd=cap_usd):
                yield _sse(
                    {
                        "type": "error",
                        "code": "over_cap",
                        "message": "You've hit today's AI limit. Upgrade for more, or add your own key for unlimited use.",
                    }
                )
                return
```

- [ ] **Step 5: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`

- [ ] **Step 6: Write the failing router test** — append to `tests/api/test_api_chat.py`:

```python
def test_send_stream_receives_managed_cap_from_dep():
    from api.gating import get_managed_ai_cap

    captured = {}

    class _CapFake(_FakeChatService):
        def send_stream(self, **k):
            captured.update(k)
            yield (
                'data: {"type": "done", "content": "", "conversation_id": 1, '
                '"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "tool_trace": []}\n\n'
            )

    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_chat_service] = lambda: _CapFake()
    app.dependency_overrides[require_app_user] = lambda: _FakeUser()
    app.dependency_overrides[get_managed_ai_cap] = lambda: 0.33
    TestClient(app).post("/api/chat/send-stream", json={"message": "hi", "model": "gpt-5"})
    assert captured.get("cap_usd") == 0.33
```

- [ ] **Step 7: Run — expect FAIL** (`cap_usd` not passed)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py::test_send_stream_receives_managed_cap_from_dep -q`

- [ ] **Step 8: Wire the dependency** — `api/routers/chat.py`. Add the import:
```python
from api.gating import get_managed_ai_cap
```
In `send(...)`, add the dependency param + pass it:
```python
def send(
    body: ChatSendRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
    cap_usd: float | None = Depends(get_managed_ai_cap),
) -> ChatSendResponse:
    return ChatSendResponse(
        **svc.send(
            chat_user_id=_uid(app_user),
            message=body.message,
            model=body.model,
            conversation_id=body.conversation_id,
            web_search=body.web_search,
            deep_research=body.deep_research,
            reasoning_effort=body.reasoning_effort,
            attached_text=body.attached_text,
            attachments=body.attachments,
            cap_usd=cap_usd,
        )
    )
```
In `send_stream(...)`, the same — add `cap_usd: float | None = Depends(get_managed_ai_cap)` to the params and `cap_usd=cap_usd` to the `svc.send_stream(...)` call.

- [ ] **Step 9: Run — expect PASS** (router test + logic-free-router guard)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py -q`

- [ ] **Step 10: Regenerate OpenAPI + verify** (the resolver's `Header(authorization)` may surface an `authorization` header param on the 2 send routes — regen + commit whatever changes)

Run: `.venv/Scripts/python.exe scripts/export_openapi.py && .venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add api/services/chat_service.py api/routers/chat.py api/openapi.json tests/api/test_chat_service.py tests/api/test_api_chat.py
git commit -m "feat(api): thread tier cap into chat + over_cap signal (Bubba B3 slice A)"
```

---

## SLICE B — frontend nudge

### Task 5: `over_cap` upgrade nudge

**Files:** Modify `web/src/lib/api/bubba.ts`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add `code?` to the error event** — `web/src/lib/api/bubba.ts`, change the error variant of `BubbaStreamEvent`:
```typescript
  | { type: "error"; message: string; code?: string };
```

- [ ] **Step 2: Add `nudge` to `UiMessage`** — `web/src/components/bubba/Bubba.tsx`:
```typescript
interface UiMessage {
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
  nudge?: "over_cap";
}
```

- [ ] **Step 3: Tag the over-cap error** — in `handleSend`'s `error` branch, replace:
```typescript
          } else if (e.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = { role: "assistant", content: e.message, isError: true };
              return next;
            });
          }
```
with:
```typescript
          } else if (e.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = {
                role: "assistant",
                content: e.message,
                isError: true,
                ...(e.code === "over_cap" ? { nudge: "over_cap" as const } : {}),
              };
              return next;
            });
          }
```

- [ ] **Step 4: Render the nudge** — in the transcript map, replace:
```tsx
            {messages.map((m, i) => (
              <Bubble key={i} message={m} />
            ))}
```
with:
```tsx
            {messages.map((m, i) =>
              m.nudge === "over_cap" ? (
                <OverCapNudge key={i} message={m.content} onAddKey={() => setShowSettings(true)} />
              ) : (
                <Bubble key={i} message={m} />
              ),
            )}
```

- [ ] **Step 5: Add the `OverCapNudge` component** (next to `Bubble`, bottom of the file):
```tsx
function OverCapNudge({ message, onAddKey }: { message: string; onAddKey: () => void }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-2 rounded-2xl border border-heat/30 bg-heat/5 px-3 py-2 text-sm text-ink">
        <p>{message}</p>
        <div className="flex flex-wrap gap-2">
          <a
            href="/pricing"
            className="rounded-lg bg-heat px-2.5 py-1 text-xs font-semibold text-white hover:bg-heat-bright"
          >
            Upgrade
          </a>
          <button
            type="button"
            onClick={onAddKey}
            className="rounded-lg border border-line bg-canvas px-2.5 py-1 text-xs font-semibold text-ink hover:text-heat"
          >
            Add your own key
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add web/src/lib/api/bubba.ts web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba over-cap upgrade nudge (Bubba B3 slice B)"
```

---

### Task 6: Verify, preview, review, ship

- [ ] **Step 1: Full backend chat/AI suites + lint**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/api/ tests/test_ai_budget_cap.py tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/test_ai_thinking_params.py -q
python -m ruff check api/ src/ai/ tests/api/ tests/test_ai_budget_cap.py && python -m ruff format --check api/ src/ai/
```
Expected: PASS / no findings.

- [ ] **Step 2: Re-assert the read-only tool surface + the Pro-gating guard** (the gating module changed)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py tests/api/test_api_pro_gating.py -q`
Expected: PASS.

- [ ] **Step 3: Preview smoke.** `preview_start` the `heater-web` config; confirm the page loads with no console errors and the Bubba panel mounts cleanly. Stop the preview. (The live over-cap nudge needs Clerk + a real cap hit → owner verifies post-activation; locally verify no mount regression + the build.)

- [ ] **Step 4: Code review.** Dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on `git diff master...HEAD`. Verify: `is_over_cap` is byte-identical for the live app (no `cap_usd`); `get_managed_ai_cap` is dormant-safe (None while billing off) + never raises; `cap_usd` threads correctly; the over-cap path emits `over_cap`; the router stays logic-free; no write tool added; the live Streamlit budget path is unaffected. Apply findings.

- [ ] **Step 5: Reconcile + merge + push.** `git checkout master && git pull --no-rebase origin master`, then `git merge --no-ff feat/bubba-b3-tier-aware-ai` (or `--ff-only` if master hasn't moved), confirm the api suite green, `git push origin master`. Pre-push runs the structural suite.

- [ ] **Step 6: Docs.** Update `CLAUDE.md` (Bubba line) + the `project_bubba_ai_assistant` + `project_m2_auth_billing` memories to mark B3 (lean) shipped — managed AI now tier-aware (dormant); flip the spec `Status:` to shipped. Note the deferred Plus/Max + monthly-taste + admin-tunable caps.

---

## Self-Review

**1. Spec coverage**
- Tier-aware daily managed cap (Free taste / Pro allowance / own-key unlimited) → Task 1 (`cap_usd`) + Task 2 (`ai_allowance`) + Task 3 (resolver) + Task 4 (thread). ✅
- Dormant-safe (billing off → admin cap, live app unchanged) → Task 1 (no-`cap_usd` guard test) + Task 3 (`stripe_enabled()` → None). ✅
- `src/ai` stays tier-agnostic → only `budget.is_over_cap` gains a generic `cap_usd`; tier logic lives in `api/`. ✅
- Upgrade nudge with `code:"over_cap"` + Upgrade/Add-key CTAs → Task 4 (code) + Task 5 (nudge). ✅
- `get_managed_ai_cap` never raises → Task 3 (try/except → None) + tests. ✅
- Read-only invariant + Pro-gating guard re-assert → Task 6 Step 2. ✅
- Out-of-scope (Plus/Max, monthly taste, admin caps) → not in any task; noted in docs. ✅

**2. Placeholder scan** — every code step is complete. Concrete tunables flagged: the $0.10 / $2.00 caps (module constants, documented tunable). The plan corrects the spec's `deps.py` location → `gating.py` (with the cycle reason). No TBD/TODO.

**3. Type consistency** — `is_over_cap(user_id, on_own_key=False, cap_usd=None)` matches the `ChatService` calls (`budget.is_over_cap(chat_user_id, on_own_key=on_own_key, cap_usd=cap_usd)`) and the router's `cap_usd=Depends(get_managed_ai_cap)` → `svc.send(..., cap_usd=cap_usd)`. `managed_cap_for_tier(tier) -> float` ↔ `get_managed_ai_cap -> float | None`. The streaming `over_cap` error dict key (`code`) ↔ TS `BubbaStreamEvent.code?` ↔ `UiMessage.nudge` ↔ `OverCapNudge`. `get_managed_ai_cap(authorization, verifier, store)` positional order matches the direct-call tests.
