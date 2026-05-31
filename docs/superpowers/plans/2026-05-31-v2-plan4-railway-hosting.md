# HEATER v2 Plan 4 — Railway Hosting + Scheduled-Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the HEATER Streamlit app to Railway as a hosted multi-user app (`MULTI_USER=1`) for all 12 FourzynBurn leaguemates, with an in-process scheduler thread keeping data fresh.

**Architecture:** Approach A — a single in-process background thread (`src/scheduler.py`, already built) runs `bootstrap_all_data(force=False)` as the *single writer*; the 12 user sessions are read-only consumers. SQLite (WAL) stays. One Railway service + one persistent volume, single replica (the in-process-writer + SQLite model requires exactly one writer process). No Postgres, no separate cron service (Railway attaches a volume to exactly one service, so a cron service can't share the SQLite volume). Every delta is additive and gated on `multi_user_enabled()`; flag-off stays v1 byte-for-byte.

**Tech Stack:** Docker (`python:3.12-slim`), Railway (`railway.toml`, DOCKERFILE builder, persistent volume), Streamlit headless server, SQLite WAL, existing `src/scheduler.py` daemon thread, existing `try_reconnect_yahoo()` headless OAuth path.

---

## File Structure

**New files:**
- `Dockerfile` — `python:3.12-slim` image; installs requirements then `--no-deps` yfpy/streamlit-oauth; runs headless Streamlit bound to `$PORT`.
- `railway.toml` — DOCKERFILE builder + deploy block (healthcheck `/_stcore/health`, `numReplicas = 1`, restart policy).
- `.dockerignore` — excludes venv/git/caches/db/token/logs/docs/tests; **keeps `data/seed/`** (Tier 3 fallbacks).
- `docs/deployment/railway-runbook.md` — step-by-step operator runbook for the Railway-side actions (account, env vars, volume, first deploy, Yahoo token seeding, troubleshooting).
- `tests/test_plan4_dockerfile_railway.py` — guards the three deploy artifacts.
- `tests/test_plan4_yahoo_token_paste.py` — guards `save_yahoo_token_json` + the admin paste control.
- `tests/test_plan4_scheduler_wiring.py` — guards `main()` starts the scheduler (flag-on) + idempotency.
- `tests/test_plan4_bootstrap_suppression.py` — guards the flag-on Home gate + `_latest_successful_refresh`.
- `tests/test_plan4_backcompat.py` — guards flag-off stays v1 (no scheduler, splash still renders).

**Modified files:**
- `src/yahoo_api.py` — add `save_yahoo_token_json(text) -> tuple[bool, str]` (validates + persists pasted token to the volume).
- `pages/_admin_controls.py` — add a "Yahoo token" admin section that calls the helper + audit-logs the action.
- `app.py` — import the scheduler + `get_refresh_log_snapshot`; add `_latest_successful_refresh()` + `_render_multiuser_home_gate()`; branch `render_single_user_app()` on the flag; start the scheduler in `main()` (flag-on only).
- `CLAUDE.md` — 5 new structural-invariant rows + a Plan 4 deployment note in the multi-user paragraph.

---

## Task 1: Deploy artifacts (Dockerfile, railway.toml, .dockerignore)

**Files:**
- Create: `Dockerfile`
- Create: `railway.toml`
- Create: `.dockerignore`
- Test: `tests/test_plan4_dockerfile_railway.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan4_dockerfile_railway.py`:

```python
"""Plan 4 guard: the three Railway deploy artifacts exist and carry the
load-bearing invariants (3.12-slim base, --no-deps yfpy/streamlit-oauth,
headless $PORT bind, single replica, healthcheck path, seed-dir preserved)."""

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_DOCKERFILE = _ROOT / "Dockerfile"
_RAILWAY = _ROOT / "railway.toml"
_DOCKERIGNORE = _ROOT / ".dockerignore"


def test_dockerfile_exists_and_uses_312_slim():
    assert _DOCKERFILE.exists(), "Dockerfile missing"
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "python:3.12-slim" in text, "must pin python:3.12-slim (CI parity)"


def test_dockerfile_installs_no_deps_yahoo_wrappers():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "--no-deps" in text, "yfpy/streamlit-oauth need --no-deps (dotenv pin conflict)"
    assert "yfpy" in text
    assert "streamlit-oauth" in text


def test_dockerfile_binds_port_headless_all_interfaces():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "PORT" in text, "must bind the platform-injected $PORT"
    assert "--server.headless" in text
    assert "0.0.0.0" in text, "must bind all interfaces inside the container"


def test_dockerignore_excludes_heavy_and_secret_paths():
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    for needle in (".venv", ".git", "__pycache__", "data/draft_tool.db", "data/yahoo_token.json"):
        assert needle in text, f".dockerignore must exclude {needle}"


def test_dockerignore_keeps_seed_dir():
    # Tier 3 fallbacks (catcher_framing, umpire_tendencies, park_factors) ship in
    # data/seed/ and MUST be in the image. A bare `data/` exclude would break them.
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    assert "data/" not in lines, "must not blanket-exclude data/ (would drop data/seed/)"
    assert "data/seed/" not in lines, "data/seed/ must remain in the image"


def test_railway_uses_dockerfile_builder():
    assert _RAILWAY.exists(), "railway.toml missing"
    text = _RAILWAY.read_text(encoding="utf-8")
    assert 'builder = "DOCKERFILE"' in text


def test_railway_has_healthcheck_path():
    text = _RAILWAY.read_text(encoding="utf-8")
    assert "/_stcore/health" in text, "Streamlit healthcheck path"


def test_railway_single_replica():
    text = _RAILWAY.read_text(encoding="utf-8")
    assert "numReplicas = 1" in text, "single-writer + SQLite require exactly one replica"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plan4_dockerfile_railway.py -v`
Expected: FAIL — all 8 tests error/fail (`Dockerfile missing`, `railway.toml missing`, `.dockerignore` read raises `FileNotFoundError`).

- [ ] **Step 3: Create the Dockerfile**

Create `Dockerfile`:

```dockerfile
# HEATER — Railway deploy image (v2 Plan 4).
# Python 3.12-slim matches CI (the app is tested on 3.12 sharded).
FROM python:3.12-slim

WORKDIR /app

# System deps: gcc/g++ for any source-built wheels (scipy/numpy ship wheels, but
# pulp/arviz transitive builds occasionally need a compiler on slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache: deps change less often than app code).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps "yfpy>=17.0" "streamlit-oauth>=0.1.14"

# App code.
COPY . .

EXPOSE 8501

# Shell form so ${PORT:-8501} expands: a bare local `docker run` (no $PORT) still
# binds 8501; on Railway the railway.toml startCommand overrides this CMD and
# uses the injected $PORT. Both are intentional — see the design spec §5.
CMD streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true
```

- [ ] **Step 4: Create railway.toml**

Create `railway.toml`:

```toml
# HEATER — Railway service config (v2 Plan 4).
# Single service, single volume, single replica (in-process scheduler is the
# sole SQLite writer — see the design spec). The startCommand here OVERRIDES the
# Dockerfile CMD and uses Railway's injected $PORT (always present on Railway).

[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true"
healthcheckPath = "/_stcore/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
numReplicas = 1
```

- [ ] **Step 5: Create .dockerignore**

Create `.dockerignore`:

```gitignore
# Plan 4 build context trim. KEEPS data/seed/ — Tier 3 fallbacks ship in the image.
.git
.gitignore
.venv
venv
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
htmlcov/
.coverage
# Runtime DB + WAL/SHM live on the Railway volume, not in the image.
data/draft_tool.db
data/draft_tool.db-wal
data/draft_tool.db-shm
# Secrets / tokens are seeded onto the volume at runtime, never baked in.
data/yahoo_token.json
.env
# Logs regenerate at runtime.
data/logs/
*.log
# Not needed in the runtime image.
docs/
tests/
.github/
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_plan4_dockerfile_railway.py -v`
Expected: PASS — all 8 tests green.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile railway.toml .dockerignore tests/test_plan4_dockerfile_railway.py
git commit -m "$(cat <<'EOF'
feat(deploy): Railway artifacts — Dockerfile + railway.toml + .dockerignore (v2 Plan 4)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `save_yahoo_token_json` helper in `src/yahoo_api.py`

**Files:**
- Modify: `src/yahoo_api.py` (add helper near `try_reconnect_yahoo`; `_AUTH_DIR` is defined at module scope ~line 133)
- Test: `tests/test_plan4_yahoo_token_paste.py`

**Why a standalone helper:** the validation/persist logic must be unit-testable without Streamlit. The page (Task 3) is a thin caller. The helper NEVER logs or echoes token contents.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan4_yahoo_token_paste.py`:

```python
"""Plan 4 guard: the admin Yahoo-token paste path. The helper validates pasted
JSON and persists it to the volume (_AUTH_DIR); the page is a thin admin-gated
caller that audit-logs only the ACTION, never the token contents."""

import ast
import json
from pathlib import Path

import pytest

from src.yahoo_api import save_yahoo_token_json

_PAGE = Path(__file__).parent.parent / "pages" / "_admin_controls.py"


def test_valid_token_written_to_auth_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    payload = {"refresh_token": "abc123", "access_token": "xyz", "token_type": "bearer"}
    ok, msg = save_yahoo_token_json(json.dumps(payload))
    assert ok is True
    written = tmp_path / "yahoo_token.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["refresh_token"] == "abc123"


def test_non_json_rejected_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json("this is not json {")
    assert ok is False
    assert "JSON" in msg
    assert not (tmp_path / "yahoo_token.json").exists()


def test_missing_refresh_token_rejected_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json(json.dumps({"access_token": "only-access"}))
    assert ok is False
    assert "refresh_token" in msg
    assert not (tmp_path / "yahoo_token.json").exists()


def test_non_object_json_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)
    ok, msg = save_yahoo_token_json(json.dumps(["not", "a", "dict"]))
    assert ok is False
    assert not (tmp_path / "yahoo_token.json").exists()


def test_page_admin_gated_and_logs_action_not_contents():
    """AST guard: page calls require_admin() + save_yahoo_token_json, logs the
    action string 'yahoo_token_update', and never passes the pasted text var to
    log_action (token contents must never be audit-logged)."""
    src = _PAGE.read_text(encoding="utf-8")
    assert "require_admin()" in src
    assert "save_yahoo_token_json" in src
    assert "yahoo_token_update" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "log_action":
            for arg in node.args:
                # No positional arg may be the pasted-text variable.
                assert getattr(arg, "id", None) != "_yahoo_token_text"


def test_admin_smoke_renders_yahoo_section():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "user_id": 1, "username": "admin", "is_admin": 1,
        "status": "active", "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run(timeout=60)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Yahoo" in s for s in subheaders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plan4_yahoo_token_paste.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_yahoo_token_json' from 'src.yahoo_api'`.

- [ ] **Step 3: Add the helper to `src/yahoo_api.py`**

`src/yahoo_api.py` already imports `json` and defines `_AUTH_DIR = Path(__file__).parent.parent / "data"`. Add this function immediately above `def try_reconnect_yahoo(` (~line 2099):

```python
def save_yahoo_token_json(text: str) -> tuple[bool, str]:
    """Validate pasted Yahoo OAuth token JSON and persist it to the volume.

    Backs the admin "Yahoo token" control (Plan 4). Validates that ``text`` is a
    JSON object carrying a non-empty ``refresh_token``, then writes it to
    ``_AUTH_DIR/yahoo_token.json`` (the same path ``try_reconnect_yahoo`` reads).
    Token contents are NEVER logged or echoed. Returns ``(ok, message)``.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, "Not valid JSON — paste the full contents of yahoo_token.json."
    if not isinstance(data, dict):
        return False, "Expected a JSON object (the yahoo_token.json dict)."
    if not data.get("refresh_token"):
        return False, "Missing 'refresh_token' — this does not look like a Yahoo token file."
    token_path = _AUTH_DIR / "yahoo_token.json"
    try:
        _AUTH_DIR.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:
        return False, f"Could not write token file: {exc}"
    return True, "Yahoo token saved."
```

- [ ] **Step 4: Run the helper tests (page tests still fail)**

Run: `python -m pytest tests/test_plan4_yahoo_token_paste.py -v`
Expected: the 4 helper tests PASS; `test_page_admin_gated...` and `test_admin_smoke...` still FAIL (page not wired yet — done in Task 3).

- [ ] **Step 5: Commit**

```bash
git add src/yahoo_api.py tests/test_plan4_yahoo_token_paste.py
git commit -m "$(cat <<'EOF'
feat(yahoo): save_yahoo_token_json helper for headless token seeding (v2 Plan 4)

Validates pasted Yahoo OAuth JSON (object + non-empty refresh_token) and writes
it to the volume path try_reconnect_yahoo reads. Never logs token contents.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Admin "Yahoo token" paste control in `pages/_admin_controls.py`

**Files:**
- Modify: `pages/_admin_controls.py` (add import + a new section before the "Audit log" section)
- Test: `tests/test_plan4_yahoo_token_paste.py` (the two page tests from Task 2 go green)

- [ ] **Step 1: Confirm the two page tests still fail**

Run: `python -m pytest tests/test_plan4_yahoo_token_paste.py::test_page_admin_gated_and_logs_action_not_contents tests/test_plan4_yahoo_token_paste.py::test_admin_smoke_renders_yahoo_section -v`
Expected: FAIL (no Yahoo section yet).

- [ ] **Step 2: Add the import**

In `pages/_admin_controls.py`, add to the imports block:

```python
from src.yahoo_api import save_yahoo_token_json
```

- [ ] **Step 3: Add the Yahoo-token section**

In `pages/_admin_controls.py`, insert this block immediately BEFORE the "Audit log" section (the `st.subheader("Audit log")` / `list_audit(...)` block near the end). `_admin_id` is already defined at the top of the page as `(current_user() or {}).get("user_id", 0)`:

```python
# --- Yahoo token -----------------------------------------------------------
st.subheader("Yahoo token")
st.caption(
    "Paste the contents of a locally-generated data/yahoo_token.json to seed "
    "headless Yahoo reconnect on this server. Stored on the persistent volume; "
    "never displayed back."
)
_yahoo_token_text = st.text_area(
    "yahoo_token.json contents", value="", key="yahoo_token_paste", height=160
)
if st.button("Save Yahoo token"):
    _ok, _msg = save_yahoo_token_json(_yahoo_token_text)
    if _ok:
        log_action(_admin_id, "yahoo_token_update")
        st.success(_msg)
    else:
        st.error(_msg)
```

- [ ] **Step 4: Run the full token-paste test file**

Run: `python -m pytest tests/test_plan4_yahoo_token_paste.py -v`
Expected: PASS — all 6 tests green (4 helper + 2 page).

- [ ] **Step 5: Commit**

```bash
git add pages/_admin_controls.py
git commit -m "$(cat <<'EOF'
feat(admin): Yahoo-token paste control on Admin Controls page (v2 Plan 4)

Admin-gated text area → save_yahoo_token_json; audit-logs the action
'yahoo_token_update' only (never the token contents).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `app.py` imports + `_latest_successful_refresh()`

**Files:**
- Modify: `app.py` (add `get_refresh_log_snapshot` to the `from src.database import (...)` block; add `from src.scheduler import start_background_refresh`; add the helper)
- Test: `tests/test_plan4_bootstrap_suppression.py`

**Why one authoritative signal:** the freshness badge AND the "warming up" gate both ask the same question — "is there a successful refresh row yet?" Routing both through `_latest_successful_refresh()` means they can never disagree (design spec §8–§9).

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan4_bootstrap_suppression.py`:

```python
"""Plan 4 guard: flag-on Home suppresses the v1 splash-screen bootstrap and
shows a 'warming up' gate driven by the single authoritative freshness signal
(_latest_successful_refresh). Flag-off Home is unchanged (covered in backcompat)."""

import ast
from pathlib import Path

_APP = Path(__file__).parent.parent / "app.py"


def _func_source(name: str) -> str:
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_APP.read_text(encoding="utf-8"), node)
    raise AssertionError(f"{name} not found in app.py")


def test_app_imports_refresh_snapshot():
    # The scheduler import is added in Task 6 (imported where used, to avoid an
    # F401 between tasks); the scheduler-wiring test guards that one.
    src = _APP.read_text(encoding="utf-8")
    assert "get_refresh_log_snapshot" in src


def test_render_single_user_app_branches_on_flag():
    body = _func_source("render_single_user_app")
    assert "multi_user_enabled()" in body, "Home must branch on the flag"
    assert "_render_multiuser_home_gate" in body, "flag-on path calls the gate"


def test_flag_on_path_suppresses_splash():
    """In render_single_user_app, the multi_user_enabled() branch must call the
    gate; render_splash_screen stays on the flag-off (else) side only."""
    body = _func_source("render_single_user_app")
    tree = ast.parse(body)
    # Find the `if multi_user_enabled():` node and assert render_splash_screen is
    # NOT called inside its body, but _render_multiuser_home_gate IS.
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(body, node.test) or ""
            if "multi_user_enabled" in test_src:
                found = True
                if_body_src = "\n".join(
                    ast.get_source_segment(body, n) or "" for n in node.body
                )
                assert "_render_multiuser_home_gate" in if_body_src
                assert "render_splash_screen" not in if_body_src
    assert found, "no `if multi_user_enabled():` branch in render_single_user_app"


def test_latest_successful_refresh_none_when_empty(monkeypatch):
    import app
    monkeypatch.setattr(app, "get_refresh_log_snapshot", lambda: [])
    assert app._latest_successful_refresh() is None


def test_latest_successful_refresh_none_when_only_errors(monkeypatch):
    import app
    monkeypatch.setattr(
        app, "get_refresh_log_snapshot",
        lambda: [{"status": "error", "last_refresh": "2026-05-31T10:00:00"}],
    )
    assert app._latest_successful_refresh() is None


def test_latest_successful_refresh_picks_latest_success(monkeypatch):
    import app
    monkeypatch.setattr(
        app, "get_refresh_log_snapshot",
        lambda: [
            {"status": "success", "last_refresh": "2026-05-31T09:00:00"},
            {"status": "error", "last_refresh": "2026-05-31T11:00:00"},
            {"status": "success", "last_refresh": "2026-05-31T10:30:00"},
        ],
    )
    latest = app._latest_successful_refresh()
    assert latest is not None
    assert latest.hour == 10 and latest.minute == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plan4_bootstrap_suppression.py -v`
Expected: FAIL — `test_app_imports_refresh_snapshot` fails (`get_refresh_log_snapshot` not imported yet) and the three `_latest_successful_refresh` tests fail (`app._latest_successful_refresh` missing).

- [ ] **Step 3: Add the import**

In `app.py`, add `get_refresh_log_snapshot` to the existing `from src.database import (...)` block (alphabetical or end of list). Do NOT add the scheduler import here — that lands in Task 6 where `main()` actually calls it (importing it now would be an unused import and fail the ruff pre-commit hook).

- [ ] **Step 4: Add the helper**

In `app.py`, add this module-level function (near the other module-level helpers, e.g. above `render_single_user_app`):

```python
def _latest_successful_refresh():
    """Most recent successful refresh timestamp, or None.

    Single authoritative 'is the data warm yet?' signal — shared by the
    freshness caption and the multi-user warming gate so they can never
    disagree (design spec §8-§9). Reads the same snapshot as the freshness
    badge; references the module-level get_refresh_log_snapshot so tests can
    monkeypatch it.
    """
    from datetime import datetime

    latest = None
    for row in get_refresh_log_snapshot():
        if row.get("status") != "success":
            continue
        ts = row.get("last_refresh")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest
```

- [ ] **Step 5: Run the helper tests (branch tests still fail)**

Run: `python -m pytest tests/test_plan4_bootstrap_suppression.py -v`
Expected: `test_app_imports_refresh_snapshot` + the three `_latest_successful_refresh` tests PASS; `test_render_single_user_app_branches_on_flag` + `test_flag_on_path_suppresses_splash` still FAIL (gate added in Task 5).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_plan4_bootstrap_suppression.py
git commit -m "$(cat <<'EOF'
feat(app): add _latest_successful_refresh freshness signal (v2 Plan 4)

Single authoritative 'data warm yet?' signal shared by the freshness caption
and the warming gate (next task), reading get_refresh_log_snapshot.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Flag-on Home gate + `render_single_user_app` branch

**Files:**
- Modify: `app.py` (add `_render_multiuser_home_gate()`; branch `render_single_user_app()` on the flag)
- Test: `tests/test_plan4_bootstrap_suppression.py` (the two branch tests go green)

**Why:** under `MULTI_USER=1` the per-session splash-screen bootstrap (`force=True`, 15-20 min) must NOT run — the scheduler thread is the sole writer. The Home page instead shows a freshness caption + an admin-only manual refresh, and a "warming up" notice until the first successful refresh row appears.

- [ ] **Step 1: Confirm the two branch tests still fail**

Run: `python -m pytest tests/test_plan4_bootstrap_suppression.py::test_render_single_user_app_branches_on_flag tests/test_plan4_bootstrap_suppression.py::test_flag_on_path_suppresses_splash -v`
Expected: FAIL.

- [ ] **Step 2: Add `_render_multiuser_home_gate()`**

In `app.py`, add above `render_single_user_app` (uses `_latest_successful_refresh` from Task 4; `_format_elapsed_hms` and `bootstrap_all_data` are already module-scope; `current_user`/`is_admin` come from the existing auth imports):

```python
def _render_multiuser_home_gate() -> bool:
    """Flag-on Home header. Returns True if data is warm enough to render the
    rest of the page, False if still warming up (caller should return early).

    No per-session bootstrap here — the in-process scheduler thread is the sole
    writer. Shows a freshness caption, an admin-only manual 'Refresh All Data'
    button, and a warming notice until the first successful refresh lands.
    """
    from datetime import datetime, UTC

    latest = _latest_successful_refresh()
    if latest is not None:
        if latest.tzinfo is None:
            mins = int((datetime.now() - latest).total_seconds() // 60)
        else:
            mins = int((datetime.now(UTC) - latest).total_seconds() // 60)
        st.caption(f"Data last refreshed {max(0, mins)} min ago.")

    user = current_user() or {}
    if user.get("is_admin"):
        with st.sidebar:
            if st.button("Refresh All Data", key="force_refresh_btn_mu", width="stretch"):
                start = datetime.now(UTC)
                with st.spinner("Refreshing all data..."):
                    yahoo_client = st.session_state.get("yahoo_client")
                    results = bootstrap_all_data(yahoo_client=yahoo_client, force=True)
                elapsed = (datetime.now(UTC) - start).total_seconds()
                st.session_state["bootstrap_results"] = results
                st.session_state["bootstrap_elapsed_secs"] = elapsed
                st.session_state["bootstrap_elapsed_hms"] = _format_elapsed_hms(elapsed)
                st.cache_data.clear()
                st.rerun()

    if latest is None:
        st.info(
            "Data warming up — first refresh in progress (~15-20 min); "
            "this page populates automatically."
        )
        return False
    return True
```

- [ ] **Step 3: Branch `render_single_user_app()`**

Replace the entire current `render_single_user_app()` (app.py ~2457-2492) with the version below. The ONLY changes are: the new `if multi_user_enabled(): ... return` wrapper at the top, and indenting the existing splash + refresh-button block into the `else:`. The trailing setup/draft routing (`st.session_state.page`) is byte-for-byte unchanged — verified to depend on `page` (set by `init_session()`), NOT on `bootstrap_complete`, so it runs fine when the splash is skipped:

```python
def render_single_user_app():
    """The v1 single-user experience: splash/bootstrap, refresh, draft flow.

    Called directly when MULTI_USER is off, and registered as the default
    "Draft Tool" Home page under st.navigation() when MULTI_USER is on.
    """
    if multi_user_enabled():
        # Hosted multi-user: the in-process scheduler is the sole writer, so we
        # do NOT run the per-session force=True splash bootstrap. Show the
        # warming gate and bail out until the first successful refresh lands.
        if not _render_multiuser_home_gate():
            return
    else:
        render_splash_screen()

        # Force Refresh button in sidebar (only after bootstrap is done)
        if st.session_state.get("bootstrap_complete"):
            with st.sidebar:
                if st.button("Refresh All Data", key="force_refresh_btn", width="stretch"):
                    import time as _time

                    with st.spinner("Refreshing all data sources..."):
                        yahoo_client = st.session_state.get("yahoo_client")
                        try:
                            st.cache_data.clear()
                        except Exception:
                            pass
                        _rf_start = _time.monotonic()
                        results = bootstrap_all_data(yahoo_client=yahoo_client, force=True)
                        _rf_elapsed = _time.monotonic() - _rf_start
                        st.session_state["bootstrap_results"] = results
                        st.session_state["bootstrap_elapsed_secs"] = float(_rf_elapsed)
                        st.session_state["bootstrap_elapsed_hms"] = _format_elapsed_hms(_rf_elapsed)
                    st.rerun()

    if st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "draft":
        render_draft_page()
```

- [ ] **Step 4: Run the full suppression test file**

Run: `python -m pytest tests/test_plan4_bootstrap_suppression.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "$(cat <<'EOF'
feat(app): flag-on Home warming gate suppresses per-session bootstrap (v2 Plan 4)

Under MULTI_USER the scheduler thread is the sole writer; Home shows a freshness
caption + admin-only manual refresh + a warming notice until the first success
row lands. Flag-off splash path unchanged (kept verbatim in the else branch).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Start the scheduler in `main()` (flag-on only)

**Files:**
- Modify: `app.py` (`main()` — call `start_background_refresh()` after the flag-off early return, before `require_auth()`)
- Test: `tests/test_plan4_scheduler_wiring.py`

**Why after the early return:** the flag-off path early-returns inside `main()` (`if not multi_user_enabled(): render_single_user_app(); return`). Placing `start_background_refresh()` AFTER that return guarantees v1 never starts a thread. `start_background_refresh()` is idempotent (process-global `_scheduler_running` + lock), so Streamlit re-running `main()` on every interaction starts exactly one thread.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan4_scheduler_wiring.py`:

```python
"""Plan 4 guard: main() starts the single-writer scheduler on the flag-ON path
only (after the flag-off early return), and start_background_refresh is
idempotent (one thread no matter how many times main() re-runs)."""

import ast
import threading
from pathlib import Path

_APP = Path(__file__).parent.parent / "app.py"


def test_app_imports_start_background_refresh():
    assert "from src.scheduler import start_background_refresh" in _APP.read_text(
        encoding="utf-8"
    )


def test_main_starts_scheduler_after_flag_off_return():
    """AST: within main(), the line that calls start_background_refresh() must
    come AFTER the `return` that ends the flag-off (not multi_user_enabled())
    fast path — so v1 never starts a thread."""
    src = _APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    main = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    # Line of the flag-off early return: the `return` inside an
    # `if not multi_user_enabled():` block.
    flag_off_return_line = None
    for node in ast.walk(main):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(src, node.test) or ""
            if "multi_user_enabled" in test_src and (
                "not " in test_src or "is False" in test_src
            ):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return):
                        flag_off_return_line = inner.lineno
    assert flag_off_return_line is not None, "no flag-off early return in main()"

    sched_call_line = None
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and getattr(
            node.func, "id", None
        ) == "start_background_refresh":
            sched_call_line = node.lineno
    assert sched_call_line is not None, "main() never calls start_background_refresh()"
    assert sched_call_line > flag_off_return_line, (
        "start_background_refresh() must be AFTER the flag-off early return"
    )


def test_start_background_refresh_idempotent(monkeypatch):
    """Calling start twice yields exactly one 'heater-refresh' thread."""
    import src.scheduler as scheduler
    # Make the refresh loop a no-op so the test doesn't touch the DB/network.
    monkeypatch.setattr(
        "src.data_bootstrap.bootstrap_all_data", lambda *a, **k: {}, raising=False
    )
    try:
        scheduler.start_background_refresh()
        scheduler.start_background_refresh()
        named = [t for t in threading.enumerate() if t.name == "heater-refresh"]
        assert len(named) == 1
    finally:
        scheduler.stop_background_refresh()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plan4_scheduler_wiring.py -v`
Expected: FAIL — `test_app_imports_start_background_refresh` fails (import added in Step 3) and `test_main_starts_scheduler_after_flag_off_return` fails (no call in main() yet). `test_start_background_refresh_idempotent` already passes — the scheduler module is pre-built.

- [ ] **Step 3: Wire the scheduler into `main()`**

In `app.py`, add the import near the other `from src.*` imports (it is used immediately by `main()` below, so no unused-import warning):

```python
from src.scheduler import start_background_refresh
```

Then in `main()`, AFTER the flag-off early return and BEFORE `require_auth()`:

```python
    if not multi_user_enabled():
        render_single_user_app()
        return

    start_background_refresh()  # single SQLite writer; idempotent + process-global
    require_auth()
    # ... existing flag-on body (view-as / broadcast / maintenance / heartbeat /
    #     st.navigation(...).run()) unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plan4_scheduler_wiring.py -v`
Expected: PASS — all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_plan4_scheduler_wiring.py
git commit -m "$(cat <<'EOF'
feat(app): start single-writer scheduler in main() on flag-on path (v2 Plan 4)

start_background_refresh() runs after the flag-off early return, so v1 never
spawns a thread; idempotent so Streamlit re-runs keep exactly one writer.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Flag-off back-compat guard

**Files:**
- Test: `tests/test_plan4_backcompat.py`

**Why:** the whole point of the gating is that `MULTI_USER` unset = v1 byte-for-byte. This is a behavioral guard: with the flag off, `main()` must render the single-user app and start zero scheduler threads.

- [ ] **Step 1: Write the test**

Create `tests/test_plan4_backcompat.py`:

```python
"""Plan 4 guard: MULTI_USER off ⇒ main() renders single-user v1 and NEVER starts
the scheduler. Mirrors the test_admin_backcompat.py pattern."""

import ast
from pathlib import Path

import pytest

_APP = Path(__file__).parent.parent / "app.py"


@pytest.fixture
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


def test_main_flag_off_renders_single_user_and_no_scheduler(_flag_off, monkeypatch):
    import app

    monkeypatch.setattr(app, "multi_user_enabled", lambda: False)

    calls = {"single_user": 0, "scheduler": 0}
    monkeypatch.setattr(app, "init_session", lambda: None)
    monkeypatch.setattr(app, "inject_custom_css", lambda: None)
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(
        app, "render_single_user_app",
        lambda: calls.__setitem__("single_user", calls["single_user"] + 1),
    )
    monkeypatch.setattr(
        app, "start_background_refresh",
        lambda: calls.__setitem__("scheduler", calls["scheduler"] + 1),
    )
    # require_auth must never be reached on the flag-off path; make it explode if it is.
    monkeypatch.setattr(
        app, "require_auth",
        lambda *a, **k: pytest.fail("require_auth reached on flag-off path"),
    )

    app.main()

    assert calls["single_user"] == 1, "flag-off must render the single-user app once"
    assert calls["scheduler"] == 0, "flag-off must NOT start the scheduler"


def test_flag_off_home_still_renders_splash():
    """AST: render_single_user_app keeps render_splash_screen on the flag-off
    (else) side — v1 splash bootstrap preserved."""
    src = _APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "render_single_user_app"
    )
    fn_src = ast.get_source_segment(src, fn)
    assert "render_splash_screen" in fn_src, "v1 splash must remain in Home"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_plan4_backcompat.py -v`
Expected: PASS — both tests green (the code from Tasks 5–6 already satisfies them).

- [ ] **Step 3: Run the whole Plan 4 test set together**

Run: `python -m pytest tests/test_plan4_dockerfile_railway.py tests/test_plan4_yahoo_token_paste.py tests/test_plan4_scheduler_wiring.py tests/test_plan4_bootstrap_suppression.py tests/test_plan4_backcompat.py -v`
Expected: PASS — all Plan 4 guards green (8 + 6 + 3 + 6 + 2 = 25 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_plan4_backcompat.py
git commit -m "$(cat <<'EOF'
test(plan4): flag-off back-compat — single-user render, zero scheduler threads

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLAUDE.md structural-invariant rows + deployment note

**Files:**
- Modify: `CLAUDE.md` (add 5 rows to the Structural Invariants table; add a Plan 4 sentence to the multi-user paragraph in Overview)

No test — documentation. The rows describe the guards written in Tasks 1–7 so future audits don't re-flag them.

- [ ] **Step 1: Add the 5 structural-invariant rows**

In `CLAUDE.md`, in the "Structural Invariants (machine-checked)" table, append these rows (after the last existing `test_*` row, before the spec-completion sub-tables):

```markdown
| `test_plan4_dockerfile_railway.py` | The three Railway deploy artifacts carry their load-bearing invariants: `Dockerfile` pins `python:3.12-slim` (CI parity) + installs `--no-deps` yfpy/streamlit-oauth + binds `$PORT`/headless/`0.0.0.0`; `.dockerignore` excludes `.venv`/`.git`/`__pycache__`/`data/draft_tool.db`/`data/yahoo_token.json` but KEEPS `data/seed/` (Tier 3 fallbacks); `railway.toml` uses `builder = "DOCKERFILE"`, healthcheck `/_stcore/health`, and `numReplicas = 1` (single-writer + SQLite) (v2 Plan 4) |
| `test_plan4_yahoo_token_paste.py` | `src/yahoo_api.save_yahoo_token_json(text)` validates pasted JSON (object + non-empty `refresh_token`) and writes to `_AUTH_DIR/yahoo_token.json`; rejects non-JSON / missing-token / non-object with NO file written. `pages/_admin_controls.py` calls `require_admin()` + the helper, audit-logs only the action `"yahoo_token_update"` (never the token text — AST-checked), and renders a "Yahoo token" subheader for admins (AppTest smoke) (v2 Plan 4) |
| `test_plan4_scheduler_wiring.py` | `app.py` imports `start_background_refresh`; `main()` calls it AFTER the flag-off early return (v1 never spawns a thread — AST-checked) and `start_background_refresh()` is idempotent (exactly one `heater-refresh` thread across repeat calls) (v2 Plan 4) |
| `test_plan4_bootstrap_suppression.py` | `render_single_user_app()` branches on `multi_user_enabled()`; the flag-on branch calls `_render_multiuser_home_gate()` and does NOT call `render_splash_screen` (no per-session bootstrap under MULTI_USER). `_latest_successful_refresh()` returns None when the refresh-log snapshot is empty or success-free, and the most-recent success timestamp otherwise (the single authoritative "data warm yet?" signal) (v2 Plan 4) |
| `test_plan4_backcompat.py` | `MULTI_USER` off ⇒ `main()` renders the single-user app exactly once and starts the scheduler ZERO times (`require_auth` never reached); flag-off Home keeps `render_splash_screen` (v1 byte-for-byte) (v2 Plan 4) |
```

- [ ] **Step 2: Add the Plan 4 deployment note**

In `CLAUDE.md` Overview, at the END of the existing multi-user paragraph (the one ending "...Every new surface is inert when the flag is off."), append:

```markdown
 **Plan 4 (Railway hosting)** ships the deploy surface: `Dockerfile` (`python:3.12-slim`, `--no-deps` yfpy/streamlit-oauth, headless `$PORT` bind), `railway.toml` (DOCKERFILE builder, `/_stcore/health`, `numReplicas = 1`), and `.dockerignore` (keeps `data/seed/`). Under `MULTI_USER=1` the in-process scheduler thread (`src/scheduler.py`, started once in `main()` after the flag-off early return) is the SOLE SQLite writer running `bootstrap_all_data(force=False)`; the 12 user sessions are read-only — `render_single_user_app()` shows a freshness caption + admin-only manual refresh + a "warming up" gate (driven by `_latest_successful_refresh()`, the same refresh-log signal as the badge) instead of the v1 per-session splash bootstrap. Admins seed headless Yahoo reconnect by pasting `yahoo_token.json` into the Admin Controls "Yahoo token" section (`save_yahoo_token_json` → volume). Single replica is a hard invariant (in-process writer + SQLite). Operator steps live in `docs/deployment/railway-runbook.md`. All Plan 4 deltas are flag-gated; flag-off stays v1 byte-for-byte.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude.md): Plan 4 deployment note + 5 structural-invariant rows (v2 Plan 4)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Deployment runbook + local image verification

**Files:**
- Create: `docs/deployment/railway-runbook.md`
- Verify: local `docker build` + `docker run` + `/_stcore/health`; full test suite; ruff

No automated test — this is the operator-facing runbook plus the final local verification that the image builds and serves.

- [ ] **Step 1: Write the runbook**

Create `docs/deployment/railway-runbook.md`:

````markdown
# HEATER on Railway — Deployment Runbook

Operator steps to deploy HEATER as a hosted multi-user app for the 12
FourzynBurn leaguemates. Claude wrote the `Dockerfile`, `railway.toml`,
`.dockerignore`, and all app wiring (v2 Plan 4); the steps below are the
Railway-side actions only YOU can perform (account, env vars, volume, OAuth
consent, deploy clicks).

## Architecture in one paragraph

One Railway **service**, one persistent **volume**, **single replica**. A
background thread inside the app (`src/scheduler.py`) is the only process that
writes the SQLite DB — it runs `bootstrap_all_data(force=False)` every 5 minutes
(refreshing only stale sources). All 12 users are read-only consumers. Because
the writer lives in-process and the DB is SQLite, you must NOT scale past one
replica.

## Prerequisites

- A [Railway](https://railway.app) account (Hobby plan ≈ $5/mo base + usage; this
  app's idle footprint lands around $7-15/mo).
- Your Yahoo OAuth app credentials (client id/secret) and a locally-generated
  `data/yahoo_token.json` (run the app once locally and complete Yahoo OAuth to
  produce it).
- This repo pushed to GitHub (Railway deploys from the repo).

## Step 1 — Create the service

1. Railway dashboard → **New Project** → **Deploy from GitHub repo** → pick
   `hicklax13/HEATER_v1.0.0`.
2. Railway detects `railway.toml` and uses the **DOCKERFILE** builder
   automatically. No build config needed.

## Step 2 — Add the persistent volume

1. Service → **Settings** → **Volumes** → **New Volume**.
2. Mount path: **`/app/data`**. (The app reads/writes `data/draft_tool.db`,
   `data/yahoo_token.json`, and `data/logs/` here. `data/seed/` ships in the
   image and is read-only baseline — the volume mount overlays the writable
   files alongside it.)
3. Size: 1 GB is plenty (the DB is tens of MB).

> **Volume + seed interaction:** the image bakes `data/seed/`. Mounting a volume
> at `/app/data` shadows the *directory*, but the seed files were COPYed into the
> image under `/app/data/seed/`. On Railway the volume is empty on first boot, so
> the app re-creates `draft_tool.db` there; the seed files remain available from
> the image layer because the volume only overlays paths that exist on it. If you
> ever see missing-seed errors, copy `data/seed/` onto the volume once via the
> Railway shell.

## Step 3 — Set environment variables

Service → **Variables** → add:

| Variable | Value | Notes |
|----------|-------|-------|
| `MULTI_USER` | `1` | **Required** — turns on the hosted multi-user surface |
| `ADMIN_USERNAME` | _your choice_ | Seeds the admin account (idempotent) |
| `ADMIN_PASSWORD` | _strong secret_ | Seeds the admin account |
| `ADMIN_TEAM_NAME` | `Team Hickey` | Your Yahoo team name |
| `YAHOO_LEAGUE_ID` | _your league id_ | Read by `try_reconnect_yahoo` for headless reconnect |
| `YAHOO_CLIENT_ID` | _client id_ | Yahoo OAuth app credential (fallback when not embedded in the token JSON) |
| `YAHOO_CLIENT_SECRET` | _client secret_ | Yahoo OAuth app credential (fallback when not embedded in the token JSON) |

> Railway injects `PORT` automatically — do NOT set it. The `railway.toml`
> `startCommand` binds `$PORT`.

## Step 4 — First deploy

1. **Deploy**. The image builds (~3-6 min first time).
2. Healthcheck hits `/_stcore/health`; Streamlit answers as soon as the port
   binds (seconds) — it does NOT wait for the 15-20 min first data refresh.
3. Open the service URL. Log in with the `ADMIN_*` credentials.
4. The Home page shows **"Data warming up — first refresh in progress (~15-20
   min)…"** This is expected: the scheduler thread is running its first
   `bootstrap_all_data`. The page populates automatically once the first
   successful refresh lands.

## Step 5 — Seed the Yahoo token

Headless Yahoo reconnect needs `data/yahoo_token.json` on the volume.

1. Open `data/yahoo_token.json` from your LOCAL machine in a text editor; copy
   its full contents.
2. In the hosted app (as admin): **Admin → Controls → Yahoo token**.
3. Paste the JSON into the text area → **Save Yahoo token**. You'll see "Yahoo
   token saved." (The token is written to the volume and never displayed back.)
4. The next scheduler cycle (or a manual **Refresh All Data**) reconnects Yahoo
   headlessly.

## Step 6 — Invite leaguemates

1. Share the service URL.
2. Each leaguemate **registers** → lands in `pending`.
3. You **approve** them and assign their Yahoo team: **Admin → Console → pending
   users**.

## Operations

- **Manual refresh:** Admin sees a **Refresh All Data** button in the sidebar
  (forces a full `bootstrap_all_data(force=True)`). Regular users do not.
- **Freshness:** Home shows "Data last refreshed N min ago." The scheduler
  re-checks every 5 minutes and refreshes only stale sources.
- **Logs:** Railway → service → **Deployments** → **View Logs**. Bootstrap and
  scheduler activity print here; persistent bootstrap log is on the volume at
  `data/logs/bootstrap.log`.
- **Restart:** safe anytime. On restart the warming gate reappears briefly until
  the first refresh row is re-read from the (persisted) DB; if the DB already has
  fresh rows the page renders immediately.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Stuck on "Data warming up" > 30 min | First bootstrap erroring | Check Deploy logs for tracebacks; verify `YAHOO_*` vars; the 3-tier waterfalls fall back to seed data, so most failures still produce a partial first refresh |
| Yahoo data never loads | Token not seeded / expired | Re-do Step 5 with a fresh local `yahoo_token.json` |
| "database is locked" in logs | More than one replica | Service → Settings → ensure **Replicas = 1** (also pinned in `railway.toml`) |
| Login fails for admin | `ADMIN_*` vars unset/typo'd | Verify the three `ADMIN_*` variables; redeploy |
| Healthcheck failing at deploy | App crashed at import | Check logs for the traceback; usually a missing env var or a dep that didn't install |

## Cost note

Hobby plan base + usage-based compute/volume. A single always-on small service
with a 1 GB volume typically runs **$7-15/month**. Railway bills compute by the
second; the app is lightweight at idle (Streamlit + a 5-min scheduler tick).
````

- [ ] **Step 2: Build the image locally**

Run: `docker build -t heater:plan4 .`
Expected: build succeeds through all layers (apt deps → pip install → `--no-deps` yfpy/streamlit-oauth → `COPY . .`). If Docker isn't available locally, note that in the verification and rely on CI/Railway to build.

- [ ] **Step 3: Run the image locally with the flag on**

Run:
```bash
docker run --rm -e MULTI_USER=1 -e ADMIN_USERNAME=admin -e ADMIN_PASSWORD=test123 -e ADMIN_TEAM_NAME="Team Hickey" -p 8501:8501 heater:plan4
```
Expected: container starts; logs show Streamlit binding `0.0.0.0:8501`; the scheduler thread logs its first refresh-loop start.

- [ ] **Step 4: Verify the healthcheck**

In another shell: `curl -fsS http://localhost:8501/_stcore/health`
Expected: `ok` (HTTP 200) within seconds of container start — BEFORE the bootstrap finishes (proves the healthcheck/bootstrap decoupling). Then open `http://localhost:8501` in a browser: login renders; after login Home shows the "Data warming up" gate. Stop the container (Ctrl-C).

- [ ] **Step 5: Run the full test suite + ruff**

Run:
```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest tests/test_plan4_dockerfile_railway.py tests/test_plan4_yahoo_token_paste.py tests/test_plan4_scheduler_wiring.py tests/test_plan4_bootstrap_suppression.py tests/test_plan4_backcompat.py -v
python -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q
```
Expected: ruff clean; 25 Plan 4 tests green; full suite green (~4534 tests). The conftest network guard keeps the in-process scheduler's bootstrap from making real network calls during the idempotency test.

- [ ] **Step 6: Commit**

```bash
git add docs/deployment/railway-runbook.md
git commit -m "$(cat <<'EOF'
docs(deploy): Railway operator runbook (v2 Plan 4)

Step-by-step: service + volume + env vars + first deploy + Yahoo token seeding
+ invite flow + ops + troubleshooting. Single-replica invariant called out.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- All 25 Plan 4 guard tests pass; full suite + ruff green.
- `docker build` succeeds; `docker run -e MULTI_USER=1` serves; `/_stcore/health` returns `ok` within seconds (before bootstrap completes).
- Flag-off: `main()` renders single-user v1, starts zero scheduler threads, Home still shows the splash (back-compat guard green).
- Flag-on: scheduler thread is the sole writer; Home shows freshness + warming gate, no per-session bootstrap; admin can paste a Yahoo token.
- `docs/deployment/railway-runbook.md` covers every Railway-side operator step.
- After ship: update memory `project_v2_rollout.md` (Plan 4 shipped → all 4 v2 plans complete).

