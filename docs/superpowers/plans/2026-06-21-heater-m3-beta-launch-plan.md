# M3 — Single-league paid beta: launch & deploy plan

> **For agentic workers:** the ONE code task (Part 1) is TDD with checkbox steps (`- [ ]`) — execute it with superpowers:executing-plans. Parts 2–4 are a deploy/onboarding runbook tagged by owner.

**Goal:** Put the existing FourzynBurn 12 onto the new React product (Vercel) + FastAPI API (Railway) on the CURRENT infra (SQLite, single replica, single hardcoded league), behind Clerk login + Stripe, with each person seeing THEIR OWN team — Streamlit still running as the fallback. Purpose = prove the new stack + cutover + billing end-to-end with real users (not revenue).

**Owner decisions baked in (2026-06-21):**
- **Reads require login at go-live** (each user only sees their own team; closes the `?team_name=` loophole).
- **Friends are free; validate checkout safely** (Pro gate stays OFF for the 12 → everything free; the Stripe checkout flow is proven separately in **test mode**, charging nobody).
- **Default Vercel/Railway URLs** for the beta (custom domain deferred to public launch / M5).

**Lane & gates:** Part 1 is mine (backend, `api/`, buildable now, **not** Postgres-gated, dormant until Clerk env is set). Parts 2–4 are cross-lane: 🟦 = CEO/backend (me), 🟧 = CMO/frontend, 🔧 = owner-only (M6). Nothing here touches the live SQLite data path or needs Postgres.

**Tech stack:** FastAPI + Clerk JWT (already built, dormant) + Stripe (already built, dormant) + Vercel + Railway. Builds on the shipped M3-1 (`docs/superpowers/plans/2026-06-21-heater-b4-m3-1-per-user-team-resolution.md`) + M2 (`docs/superpowers/specs/2026-06-20-heater-m2-*`).

---

## Part 1 — Code: the "require login" activation flip 🟦 (mine, buildable now)

**Why:** Today the personalized reads use `optional_app_user` → they STAY OPEN (anyone can pass `?team_name=`). At the paid-beta go-live they must require a valid login. The flip must be **env-gated + dormant**: require login **only when Clerk is configured** (`CLERK_ISSUER` set, an M6 owner step), else behave exactly as today so the live app + all 339 existing api tests stay green until activation.

**Behavior matrix (after the flip):**
| Clerk env | Request | Result |
|---|---|---|
| **unset** (today) | any | OPEN — `team_name` query param honored (byte-for-byte today). |
| **set** | no / invalid token | **401** (require login). |
| **set** | valid token, has assignment | resolves the user's team. |
| **set** | valid token, NO assignment | `team_name=None` → endpoint's "no team yet" state (NOT another user's team, NOT 401). |

**Files:**
- Modify: `api/auth.py` (add a shared `clerk_configured()` helper; reuse it in `get_auth_verifier`)
- Modify: `api/tenancy.py` (`require_viewer_context`: when Clerk configured AND no app_user → 401)
- Test: `tests/api/test_api_tenancy_resolver.py` (append the flip cases)

- [ ] **Step 1: Write the failing test (append to `tests/api/test_api_tenancy_resolver.py`)**

```python
def test_require_login_when_clerk_configured_and_no_token(monkeypatch):
    # Activation flip: Clerk configured + no identity → 401 (require login).
    monkeypatch.setenv("CLERK_ISSUER", "https://example.clerk.accounts.dev")
    app = _resolver_app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    # NOTE: do not override get_auth_verifier here — optional_app_user returns None
    # for a no-Authorization request regardless of verifier, and the flip keys off
    # the CLERK_ISSUER env, so the 401 comes from require_viewer_context.
    r = TestClient(app).get("/probe?team_name=Team%20Hickey")
    assert r.status_code == 401


def test_open_when_clerk_unset_even_without_token(monkeypatch):
    # Dormant: Clerk unset → reads stay OPEN (today's behavior).
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    app = _resolver_app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Team%20Hickey").json()
    assert body == {"team": "Team Hickey", "resolved": None}


def test_clerk_configured_valid_token_still_resolves(monkeypatch):
    # Clerk configured + valid token + assignment → resolves (no 401).
    monkeypatch.setenv("CLERK_ISSUER", "https://example.clerk.accounts.dev")
    app = _resolver_app()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    body = TestClient(app).get("/probe?team_name=x", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] == "Bronx Bombers"
```

- [ ] **Step 2: Run — expect the new tests to FAIL**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: `test_require_login_when_clerk_configured_and_no_token` FAILS (currently returns 200, not 401). The other two pass.

- [ ] **Step 3a: Add `clerk_configured()` to `api/auth.py`** (DRY — reuse in `get_auth_verifier`)

In `api/auth.py`, add near the top (after the imports / `_TOKEN_ENV`):
```python
def clerk_configured() -> bool:
    """True when Clerk auth is configured (CLERK_ISSUER set). The single predicate
    for 'is real user auth live' — used by get_auth_verifier AND the read-gate flip."""
    return bool(os.environ.get("CLERK_ISSUER", "").strip())
```
Then refactor `get_auth_verifier` to use it (behavior identical):
```python
def get_auth_verifier() -> AuthVerifier:
    if clerk_configured():
        issuer = os.environ["CLERK_ISSUER"].strip()
        return ClerkVerifier(
            issuer=issuer,
            audience=os.environ.get("CLERK_AUDIENCE", "").strip() or None,
            jwks_url=os.environ.get("CLERK_JWKS_URL", "").strip() or None,
        )
    return EnvTokenVerifier()
```

- [ ] **Step 3b: Gate reads in `api/tenancy.py::require_viewer_context`**

Add the import and the flip:
```python
from fastapi import Depends, HTTPException, status

from api.auth import clerk_configured
```
In `require_viewer_context`, replace the `if app_user is None:` branch:
```python
    if app_user is None:
        # Activation flip: once Clerk is live, an open read with no valid login is
        # rejected (each user must log in → only sees their own team). Dormant until
        # CLERK_ISSUER is set, so today's open reads + existing tests are unchanged.
        if clerk_configured():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return ViewerContext()
```
(The rest of `require_viewer_context` — league + membership lookup — is unchanged.)

- [ ] **Step 4: Run — expect all green**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: PASS (13).
Run: `python -m pytest tests/api/ -q`
Expected: PASS (all — existing tests set no `CLERK_ISSUER`, so reads stay open).

- [ ] **Step 5: Regenerate OpenAPI (the personalized routes gain a 401 response) + commit**

```bash
python scripts/export_openapi.py
python -m pytest tests/api/test_openapi_contract.py -q   # expect PASS after regen
python -m ruff check api/ --fix && python -m ruff format api/auth.py api/tenancy.py
git pull --no-rebase origin master
git add api/auth.py api/tenancy.py tests/api/test_api_tenancy_resolver.py api/openapi.json
git commit -m "feat(api): require login on personalized reads when Clerk configured (M3 activation, dormant)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin master
```

> **Code review before merge:** run a `silent-failure-hunter` pass (focus: the env-gate can't fail OPEN — i.e. a misread env must never silently leave reads open when Clerk IS configured).

---

## Part 2 — Deploy config

### 2.1 API on Railway 🟦 + 🔧
The FastAPI API runs as its **own** Railway service (the existing Streamlit service stays untouched). Start command (factory):
```
uvicorn api.main:create_app --factory --host 0.0.0.0 --port $PORT
```
🟦 **(me, buildable now):** add an API start surface so Railway has something to run — a `Procfile`-style or a small `api/Dockerfile` (mirroring the root Dockerfile's `--no-deps yfpy/streamlit-oauth` install), kept separate from the Streamlit Dockerfile. *(I can scaffold this in a follow-up; it's additive + doesn't affect Streamlit.)*

🔧 **(owner, M6):** create the Railway API service from this repo, set its env:
| Env var | Value | Why |
|---|---|---|
| `CLERK_ISSUER` (+ `CLERK_AUDIENCE`) | from the Clerk app | turns on login + the Part-1 read-gate |
| `HEATER_ADMIN_CLERK_IDS` | your Clerk user id | lets you assign teams |
| `HEATER_API_CORS_ORIGINS` | the Vercel URL | browser CORS |
| `HEATER_API_DB_PATH` | a path on a Railway **volume** | the api-owned `api_state.db` must persist across deploys |
| `HEATER_BETA_LEAGUE_EXTERNAL_ID` / `_NAME` | `469` / `FourzynBurn` | seeds the one beta league (defaults already match — only set to override) |
| Stripe vars | **LEAVE UNSET during the beta** | keeps the Pro gate OFF → free for friends (see Part 3) |

> The API and Streamlit can share the same Railway project but are **separate services**; only Streamlit writes the live `draft_tool.db`. The API reads `draft_tool.db` (read-only) + owns its separate `api_state.db`. Single-writer invariant preserved.

### 2.2 Frontend on Vercel 🟧 + 🔧
🔧 **(owner, M6):** deploy `web/` to Vercel; set `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` **and** `NEXT_PUBLIC_HEATER_LIVE=1` (must be set together) + the API base URL env the client reads.
🟧 **(CMO):** point the API client at the Railway API base; **send the Clerk token as `Authorization: Bearer` on every personalized call** (M2 frontend already wires `getToken()`); **drop the hardcoded `team_name`/`VIEWER_TEAM = "Team Hickey"`** so authed users get their resolved team (with the Part-1 gate, unauth'd users are bounced to login).

---

## Part 3 — Billing for the beta (free friends + safe checkout test) 🔧 + 🟦

**Beta runtime = Pro gate OFF.** Leave the Stripe env UNSET on the live API service → `billing_env_configured()` is false → `require_pro` is dormant → all 6 heavy endpoints are free for the 12. Everyone gets the full product free. (This is the M2 design's built-in dormancy — nothing to build.)

**Validate the checkout plumbing SEPARATELY, in Stripe TEST mode (charges nobody):**
🔧 (owner, M6, one-time): in a throwaway/staging API instance (or locally) set Stripe **test-mode** keys (`STRIPE_SECRET_KEY=sk_test_…`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRO_PRICE_ID`), then:
1. `POST /api/billing/checkout-session` → open the returned URL → pay with Stripe's test card `4242 4242 4242 4242`.
2. Confirm the webhook fires → `GET /api/billing/subscription` shows `pro`.
3. Confirm a Pro-gated endpoint returns 200 for that user and 402 for a non-Pro user.

That proves the entire Free→Pro path works before public launch, with zero real charges. **Do NOT set the Stripe vars on the friends' live beta service** (that would paywall them). Public launch (M5) flips to live keys.

---

## Part 4 — Onboarding runbook 🔧 (owner, M6) + verification 🟦

### 4.1 Onboard the 12 (owner)
1. In Clerk: revoke the `testuser`, invite the 12 leaguemates (email), each signs up → gets a Clerk user id.
2. Find each Clerk user id (Clerk dashboard → Users).
3. **Assign each user → their team** via the endpoint I shipped (you're the admin in `HEATER_ADMIN_CLERK_IDS`). For each member, with your own Clerk token as `Bearer`:
   ```bash
   curl -X POST "$API/api/admin/assignments" \
     -H "Authorization: Bearer $YOUR_CLERK_JWT" -H "Content-Type: application/json" \
     -d '{"clerk_user_id":"<their clerk id>","team_name":"<their exact team name>"}'
   ```
   - `GET "$API/api/admin/assignments"` lists current assignments + `available_teams` (copy the exact names from there).
   - If a name comes back `"validated": false`, the roster cache was cold — re-run once data has warmed so the name is verified.

### 4.2 Go-live verification (me 🟦, once the env is set)
- `GET $API/healthz` → 200.
- With NO token: `GET $API/api/me/team` → **401** (login required — the Part-1 flip is live).
- With a member's token: `GET $API/api/me/team` → **their** team (eyebrow/roster), not Team Hickey.
- Cross-check 2 members see different teams.
- Streamlit still reachable (fallback).

### 4.3 Rollback
Streamlit stays live throughout. If the React beta misbehaves, members keep using Streamlit; no data is at risk (the API only reads `draft_tool.db` + its own `api_state.db`). Fixing forward = redeploy the API/frontend; no migration to undo.

---

## "Done" for M3

12 users logged in via Clerk, each seeing their own team on the React product (Vercel) talking to the API (Railway), Pro features free for them, Stripe checkout proven in test mode, Streamlit still running as fallback. → unblocks the **M4** decision (multi-tenant infra for the public launch).

## What this plan does NOT cover (other milestones)
- M4 multi-tenant infra (Postgres/workers/connectors/per-user OAuth) — owner-gated + needs a real Postgres.
- M5 public launch, custom domain, multi-platform league connection, retiring Streamlit, marketing site.
- The CMO frontend screens themselves (M2 login/pricing UI, dropping the team hardcode) — 🟧 lane.

## Immediate buildable-now item
**Part 1 (the require-login activation flip)** is mine, non-gated, dormant, and the only code in this plan. Everything else is deploy config + owner runbook that can't run until you do the M6 setup (Clerk/Stripe/Railway/Vercel).
