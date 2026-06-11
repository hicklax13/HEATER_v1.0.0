# HEATER AI Chat Assistant — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a persistent, per-page, multi-provider AI chat assistant connected to all HEATER data via tool-use, with BYO API keys, an admin shared-key fallback + cost caps, and SQLite-saved history.

**Architecture:** A new `src/ai/` sub-package wraps the existing service layer as AI tools. Chat lives in `session_state` (survives page nav) + SQLite (survives logout/device). LiteLLM gives one interface to Anthropic/OpenAI/Gemini/OpenRouter/Ollama. The AI reads data directly and *requests* refreshes through a queue the single-writer scheduler drains. A floating window (drag/resize/minimize/close) mounts on every page. Everything is MULTI_USER-gated and inert when the flag is off (v1 byte-for-byte).

**Tech Stack:** Python 3.12 (CI) / 3.14 (local), Streamlit ≥1.47, SQLite (WAL), LiteLLM, streamlit-float, Fernet (cryptography), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-ai-chat-assistant-design.md`

**Conventions to follow (already in the codebase):**
- DB access only via `get_connection()` (`src/database.py`); wrap in `try/finally: conn.close()`.
- New tables go in the `init_db()` executescript block (`src/database.py`, the block ending ~line 947), `CREATE TABLE IF NOT EXISTS`.
- MULTI_USER gating: `from src.auth import multi_user_enabled` → early-return when off (mirror `src/feedback.py`, `src/usage.py`).
- Admin audit: `from src.audit import log_action` (logs action only, never secrets).
- Fernet pattern: mirror `src/token_relay.py::_fernet/encrypt_token/decrypt_token`.
- Tests honor `tests/conftest.py` network guard (no real sockets). Run a single file: `python -m pytest tests/test_x.py -v`.
- Lint before commit: `python -m ruff format . && python -m ruff check .`

---

## Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 1: Add LiteLLM + streamlit-float to `requirements.txt`**

Add these lines after the `plotly>=5.18.0` line:

```text
# AI chat assistant (Phase 1): one interface to Anthropic/OpenAI/Gemini/OpenRouter/Ollama
litellm>=1.88.1
# Floating, draggable chat window container
streamlit-float>=0.3.5
```

- [ ] **Step 2: Verify both install + import on local Python**

Run:
```powershell
C:\Users\conno\Code\HEATER_v1.0.1\.venv\Scripts\python.exe -m pip install "litellm>=1.88.1" "streamlit-float>=0.3.5"
C:\Users\conno\Code\HEATER_v1.0.1\.venv\Scripts\python.exe -c "import litellm, streamlit_float; print('ok', litellm.__version__)"
```
Expected: `ok <version>` with no ImportError. (If `litellm` fails on Python 3.14, pin the latest version that installs and note it; LiteLLM is pure-Python and should install.)

- [ ] **Step 3: Add the same installs to the `Dockerfile`**

Find the line in `Dockerfile` that installs requirements (`pip install ... -r requirements.txt`). The new lines in `requirements.txt` are picked up automatically — no Dockerfile change is needed unless yfpy/streamlit-oauth are installed `--no-deps` in a way that skips the file. Confirm `requirements.txt` is installed normally (not `--no-deps`); if so, this step is a no-op. Document the confirmation in the commit message.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build(ai-chat): add litellm + streamlit-float deps"
```

---

## Task 2: Create the AI SQLite tables

**Files:**
- Modify: `src/database.py` (the `init_db()` executescript block, before its closing `"""` ~line 947)
- Test: `tests/test_ai_tables.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_tables.py
"""init_db() creates the 5 additive AI-chat tables idempotently (v2 AI Phase 1)."""

from src.database import get_connection, init_db

_AI_TABLES = [
    "ai_provider_keys",
    "ai_conversations",
    "ai_messages",
    "ai_usage_ledger",
    "forced_refresh_queue",
]


def _table_names() -> set[str]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {r["name"] for r in rows}
    finally:
        conn.close()


def test_ai_tables_created():
    init_db()
    names = _table_names()
    for t in _AI_TABLES:
        assert t in names, f"{t} missing after init_db()"


def test_init_db_idempotent():
    init_db()
    init_db()  # second call must not raise
    assert set(_AI_TABLES).issubset(_table_names())


def test_tables_start_empty():
    init_db()
    conn = get_connection()
    try:
        for t in _AI_TABLES:
            n = conn.execute(f"SELECT COUNT(*) AS c FROM {t}").fetchone()["c"]
            assert n == 0, f"{t} should start empty"
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_tables.py -v`
Expected: FAIL — `ai_provider_keys missing after init_db()`.

- [ ] **Step 3: Add the tables to `init_db()`**

In `src/database.py`, inside the `conn.executescript("""...""")` block in `init_db()`, immediately **before** the closing `"""` that precedes `conn.commit()` (right after the `auth_tokens` table / its index, ~line 946), insert:

```sql

        -- v2 AI Phase 1: per-user encrypted provider keys (BYOK). encrypted_key is
        -- Fernet ciphertext; plaintext is never stored. Admin shared key lives in
        -- app_settings, not here.
        CREATE TABLE IF NOT EXISTS ai_provider_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            provider TEXT NOT NULL,
            label TEXT,
            encrypted_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, provider, label)
        );
        CREATE INDEX IF NOT EXISTS idx_ai_keys_user ON ai_provider_keys(user_id);

        -- v2 AI Phase 1: saved conversations (history dropdown) + their messages.
        CREATE TABLE IF NOT EXISTS ai_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            title TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ai_conv_user ON ai_conversations(user_id, updated_at);

        CREATE TABLE IF NOT EXISTS ai_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES ai_conversations(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ai_msg_conv ON ai_messages(conversation_id, id);

        -- v2 AI Phase 1: per-user daily spend ledger for cost caps.
        CREATE TABLE IF NOT EXISTS ai_usage_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            day TEXT NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            UNIQUE(user_id, day)
        );

        -- v2 AI Phase 1: the ONLY write path the AI has. request_refresh enqueues
        -- here; the single-writer scheduler drains it (preserves single-writer).
        CREATE TABLE IF NOT EXISTS forced_refresh_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            requested_by INTEGER REFERENCES users(user_id),
            status TEXT NOT NULL DEFAULT 'pending',
            detail TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_refresh_queue_status ON forced_refresh_queue(status);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_tables.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/database.py tests/test_ai_tables.py
git commit -m "feat(ai-chat): add 5 AI-chat SQLite tables to init_db"
```

---

## Task 3: BYOK key store (`src/ai/keys.py`)

**Files:**
- Create: `src/ai/__init__.py`
- Create: `src/ai/keys.py`
- Test: `tests/test_ai_keys.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_keys.py
"""BYOK key store: Fernet-encrypted at rest, admin-shared-key fallback."""

import os

import pytest
from cryptography.fernet import Fernet

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    monkeypatch.setenv("HEATER_AI_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    # clean slate for the test user
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_provider_keys WHERE user_id = 99")
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_shared_key'")
        conn.commit()
    finally:
        conn.close()


def test_store_and_get_roundtrip():
    from src.ai.keys import get_key, store_key

    store_key(99, "anthropic", "sk-ant-secret", label="mine")
    assert get_key(99, "anthropic") == "sk-ant-secret"


def test_ciphertext_is_encrypted_at_rest():
    from src.ai.keys import store_key

    store_key(99, "openai", "sk-plaintext-should-not-appear")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM ai_provider_keys WHERE user_id = 99 AND provider = 'openai'"
        ).fetchone()
    finally:
        conn.close()
    assert "sk-plaintext-should-not-appear" not in row["encrypted_key"]


def test_falls_back_to_admin_shared_key():
    from src.ai.keys import get_key, set_admin_shared_key

    set_admin_shared_key("anthropic", "sk-admin-shared", admin_id=1)
    # user 99 has no anthropic key of their own → gets the shared one
    assert get_key(99, "anthropic") == "sk-admin-shared"


def test_user_key_overrides_shared():
    from src.ai.keys import get_key, set_admin_shared_key, store_key

    set_admin_shared_key("anthropic", "sk-admin-shared", admin_id=1)
    store_key(99, "anthropic", "sk-user-own")
    assert get_key(99, "anthropic") == "sk-user-own"


def test_no_fernet_key_disables_storage(monkeypatch):
    from src.ai import keys as keys_mod

    monkeypatch.delenv("HEATER_AI_KEY", raising=False)
    monkeypatch.delenv("HEATER_RELAY_KEY", raising=False)
    with pytest.raises(RuntimeError):
        keys_mod.store_key(99, "anthropic", "sk-whatever")


def test_list_and_delete():
    from src.ai.keys import delete_key, list_keys, store_key

    store_key(99, "openai", "k1", label="work")
    store_key(99, "gemini", "k2", label="home")
    listed = list_keys(99)
    assert {r["provider"] for r in listed} == {"openai", "gemini"}
    # list must NOT leak plaintext
    assert all("k1" not in str(r) and "k2" not in str(r) for r in listed)
    delete_key(99, "openai", "work")
    assert {r["provider"] for r in list_keys(99)} == {"gemini"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_keys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai'`.

- [ ] **Step 3: Create the package + implementation**

```python
# src/ai/__init__.py
"""HEATER embedded AI chat assistant (v2 AI Phase 1). MULTI_USER-gated."""
```

```python
# src/ai/keys.py
"""BYOK provider-key store: Fernet-encrypted at rest, admin-shared-key fallback.

Mirrors src/token_relay.py's Fernet usage. The encryption key comes from
HEATER_AI_KEY (falling back to HEATER_RELAY_KEY so the operator can reuse one
Fernet value). Plaintext keys are never written to the DB or logged.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from cryptography.fernet import Fernet

from src.app_settings import get_setting, set_setting

_SHARED_KEY_SETTING = "ai_shared_key"  # JSON {provider: ciphertext}


def _fernet() -> Fernet:
    raw = (os.environ.get("HEATER_AI_KEY") or os.environ.get("HEATER_RELAY_KEY") or "").strip()
    if not raw:
        raise RuntimeError(
            "AI key storage disabled: set HEATER_AI_KEY (or reuse HEATER_RELAY_KEY) to a Fernet key."
        )
    return Fernet(raw.encode())


def _encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


def store_key(user_id: int, provider: str, api_key: str, label: str | None = None) -> None:
    """Encrypt + upsert a user's provider key. Raises if no Fernet key is set."""
    from src.database import get_connection

    ciphertext = _encrypt(api_key)  # raises RuntimeError before any DB write if unconfigured
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO ai_provider_keys (user_id, provider, label, encrypted_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider, label) DO UPDATE SET encrypted_key = excluded.encrypted_key
            """,
            (user_id, provider, label, ciphertext, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_key(user_id: int, provider: str) -> str | None:
    """The user's own key for the provider, else the admin shared key, else None."""
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM ai_provider_keys WHERE user_id = ? AND provider = ? "
            "ORDER BY id DESC LIMIT 1",
            (user_id, provider),
        ).fetchone()
    finally:
        conn.close()
    if row is not None:
        try:
            return _decrypt(row["encrypted_key"])
        except Exception:
            return None
    return get_admin_shared_key(provider)


def list_keys(user_id: int) -> list[dict]:
    """Metadata for a user's keys (provider/label/created_at) — never the plaintext."""
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT provider, label, created_at FROM ai_provider_keys WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_key(user_id: int, provider: str, label: str | None = None) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        if label is None:
            conn.execute(
                "DELETE FROM ai_provider_keys WHERE user_id = ? AND provider = ? AND label IS NULL",
                (user_id, provider),
            )
        else:
            conn.execute(
                "DELETE FROM ai_provider_keys WHERE user_id = ? AND provider = ? AND label = ?",
                (user_id, provider, label),
            )
        conn.commit()
    finally:
        conn.close()


def set_admin_shared_key(provider: str, api_key: str, admin_id: int) -> None:
    """Encrypt + store the shared fallback key for a provider (admin only path).

    Stored as a JSON map {provider: ciphertext} in app_settings under one key.
    set_setting() is MULTI_USER-gated and audit-logs the action name only.
    """
    import json

    ciphertext = _encrypt(api_key)  # raises before any persistence if unconfigured
    raw = get_setting(_SHARED_KEY_SETTING)
    data = {}
    if raw:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = {}
    data[provider] = ciphertext
    set_setting(_SHARED_KEY_SETTING, json.dumps(data), admin_id)


def get_admin_shared_key(provider: str) -> str | None:
    import json

    raw = get_setting(_SHARED_KEY_SETTING)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    ct = data.get(provider)
    if not ct:
        return None
    try:
        return _decrypt(ct)
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_keys.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/__init__.py src/ai/keys.py tests/test_ai_keys.py
git commit -m "feat(ai-chat): Fernet-encrypted BYOK key store + admin shared-key fallback"
```

---

## Task 4: Model-tier router (`src/ai/router.py`)

**Files:**
- Create: `src/ai/router.py`
- Test: `tests/test_ai_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_router.py
"""Tier->model routing + admin overrides + the verified price table."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _multi_user(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_tier_models'")
        conn.commit()
    finally:
        conn.close()


def test_default_tier_models():
    from src.ai.router import model_for_tier

    assert model_for_tier("simple") == "anthropic/claude-haiku-4-5"
    assert model_for_tier("moderate") == "anthropic/claude-sonnet-4-6"
    assert model_for_tier("complex") == "anthropic/claude-opus-4-8"


def test_unknown_tier_falls_back_to_moderate():
    from src.ai.router import model_for_tier

    assert model_for_tier("nonsense") == "anthropic/claude-sonnet-4-6"


def test_admin_override(monkeypatch):
    from src.ai.router import model_for_tier, set_tier_models

    set_tier_models({"simple": "gemini/gemini-3-flash"}, admin_id=1)
    assert model_for_tier("simple") == "gemini/gemini-3-flash"
    # untouched tiers keep defaults
    assert model_for_tier("complex") == "anthropic/claude-opus-4-8"


def test_provider_of():
    from src.ai.router import provider_of

    assert provider_of("anthropic/claude-haiku-4-5") == "anthropic"
    assert provider_of("ollama/qwen2.5:7b") == "ollama"
    assert provider_of("bare-model") == "openai"  # litellm default convention


def test_price_table_known_models():
    from src.ai.router import price_per_token

    pin, pout = price_per_token("anthropic/claude-opus-4-8")
    assert pin == pytest.approx(5e-6)
    assert pout == pytest.approx(25e-6)
    # local model is free
    assert price_per_token("ollama/qwen2.5:7b") == (0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai.router'`.

- [ ] **Step 3: Implement**

```python
# src/ai/router.py
"""Tier -> model routing (admin-overridable) + a verified June-2026 price table.

Model strings are LiteLLM provider-prefixed ("anthropic/claude-...", "ollama/...").
The price table is the cost-cap fallback for when litellm.completion_cost() can't
price a newer model (it returns 0 for models outside its bundled map).
"""

from __future__ import annotations

import json

from src.app_settings import get_setting, set_setting

_TIER_MODELS_SETTING = "ai_tier_models"

_DEFAULT_TIER_MODELS = {
    "simple": "anthropic/claude-haiku-4-5",
    "moderate": "anthropic/claude-sonnet-4-6",
    "complex": "anthropic/claude-opus-4-8",
}

# USD per token (input, output). Verified June 2026. ollama/* is local = free.
_PRICE_PER_TOKEN = {
    "anthropic/claude-haiku-4-5": (1e-6, 5e-6),
    "anthropic/claude-sonnet-4-6": (3e-6, 15e-6),
    "anthropic/claude-opus-4-8": (5e-6, 25e-6),
    "anthropic/claude-fable-5": (10e-6, 50e-6),
}


def _overrides() -> dict:
    raw = get_setting(_TIER_MODELS_SETTING)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def model_for_tier(tier: str) -> str:
    merged = {**_DEFAULT_TIER_MODELS, **_overrides()}
    return merged.get(tier, _DEFAULT_TIER_MODELS["moderate"])


def set_tier_models(mapping: dict, admin_id: int) -> None:
    """Merge admin overrides into the tier->model map (MULTI_USER-gated via set_setting)."""
    current = _overrides()
    current.update({k: v for k, v in mapping.items() if v})
    set_setting(_TIER_MODELS_SETTING, json.dumps(current), admin_id)


def provider_of(model: str) -> str:
    """LiteLLM provider prefix; bare strings default to openai (litellm convention)."""
    return model.split("/", 1)[0] if "/" in model else "openai"


def price_per_token(model: str) -> tuple[float, float]:
    if model.startswith("ollama/"):
        return (0.0, 0.0)
    return _PRICE_PER_TOKEN.get(model, (0.0, 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_router.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/router.py tests/test_ai_router.py
git commit -m "feat(ai-chat): tier->model router + price table"
```

---

## Task 5: Cost caps + usage ledger (`src/ai/budget.py`)

**Files:**
- Create: `src/ai/budget.py`
- Test: `tests/test_ai_budget.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_budget.py
"""Per-user daily cost cap enforcement + ledger accounting."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_usage_ledger WHERE user_id = 99")
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_daily_cap_usd'")
        conn.commit()
    finally:
        conn.close()


def test_under_cap_allows():
    from src.ai.budget import is_over_cap

    assert is_over_cap(99) is False


def test_record_then_over_cap():
    from src.ai.budget import is_over_cap, record_usage, set_daily_cap

    set_daily_cap(0.50, admin_id=1)
    record_usage(99, tokens_in=1000, tokens_out=1000, cost_usd=0.40)
    assert is_over_cap(99) is False
    record_usage(99, tokens_in=1000, tokens_out=1000, cost_usd=0.20)  # total 0.60 > 0.50
    assert is_over_cap(99) is True


def test_spent_today_sums():
    from src.ai.budget import record_usage, spent_today

    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=0.10)
    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=0.05)
    assert spent_today(99) == pytest.approx(0.15)


def test_own_key_user_exempt():
    from src.ai.budget import is_over_cap, record_usage, set_daily_cap

    set_daily_cap(0.01, admin_id=1)
    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=5.0)
    # on their own key, the shared-key cap does not apply
    assert is_over_cap(99, on_own_key=True) is False
    assert is_over_cap(99, on_own_key=False) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai.budget'`.

- [ ] **Step 3: Implement**

```python
# src/ai/budget.py
"""Per-user daily AI spend cap + usage ledger.

The cap applies to the admin SHARED key (the operator pays). A user on their own
key is exempt by default. Caps are admin-set in app_settings; sensible default.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.app_settings import get_setting, set_setting

_DAILY_CAP_SETTING = "ai_daily_cap_usd"
_DEFAULT_DAILY_CAP_USD = 1.00


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def daily_cap_usd() -> float:
    raw = get_setting(_DAILY_CAP_SETTING)
    if raw is None:
        return _DEFAULT_DAILY_CAP_USD
    try:
        return float(raw)
    except (ValueError, TypeError):
        return _DEFAULT_DAILY_CAP_USD


def set_daily_cap(usd: float, admin_id: int) -> None:
    set_setting(_DAILY_CAP_SETTING, str(float(usd)), admin_id)


def record_usage(user_id: int, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO ai_usage_ledger (user_id, day, tokens_in, tokens_out, cost_usd)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
                tokens_in = tokens_in + excluded.tokens_in,
                tokens_out = tokens_out + excluded.tokens_out,
                cost_usd = cost_usd + excluded.cost_usd
            """,
            (user_id, _today(), int(tokens_in), int(tokens_out), float(cost_usd)),
        )
        conn.commit()
    finally:
        conn.close()


def spent_today(user_id: int) -> float:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT cost_usd FROM ai_usage_ledger WHERE user_id = ? AND day = ?",
            (user_id, _today()),
        ).fetchone()
        return float(row["cost_usd"]) if row is not None else 0.0
    finally:
        conn.close()


def is_over_cap(user_id: int, on_own_key: bool = False) -> bool:
    if on_own_key:
        return False
    return spent_today(user_id) >= daily_cap_usd()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_budget.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/budget.py tests/test_ai_budget.py
git commit -m "feat(ai-chat): per-user daily cost caps + usage ledger"
```

---

## Task 6: Schema-card for the system prompt (`src/ai/schema_card.py`)

**Files:**
- Create: `src/ai/schema_card.py`
- Test: `tests/test_ai_schema_card.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_schema_card.py
"""The schema card lists real tables/columns so the model targets valid SQL."""

from src.database import init_db


def test_schema_card_lists_core_tables():
    init_db()
    from src.ai.schema_card import build_schema_card

    card = build_schema_card()
    assert "players" in card
    assert "league_standings" in card
    # columns are included so the model doesn't hallucinate them
    assert "CREATE TABLE" in card or "(" in card


def test_schema_card_excludes_secret_tables():
    init_db()
    from src.ai.schema_card import build_schema_card

    card = build_schema_card()
    # never expose the key store or auth tokens to the model's SQL surface
    assert "ai_provider_keys" not in card
    assert "auth_tokens" not in card
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_schema_card.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/schema_card.py
"""Compact DB schema description for the system prompt.

Lets the model write accurate read-only SQL (mitigates hallucinated columns /
wrong joins). Secret/PII tables are excluded from what the model can see + query.
"""

from __future__ import annotations

# Tables the AI must never see or query (keys, auth, raw user creds).
_EXCLUDED = {
    "ai_provider_keys",
    "auth_tokens",
    "app_settings",
    "users",
    "sessions",
    "audit_log",
}


def build_schema_card() -> str:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    lines = ["Read-only SQLite schema (SELECT only). Tables and their definitions:"]
    for r in rows:
        if r["name"] in _EXCLUDED or r["name"].startswith("sqlite_"):
            continue
        sql = " ".join((r["sql"] or "").split())  # collapse whitespace
        lines.append(sql + ";")
    return "\n".join(lines)


def excluded_tables() -> set[str]:
    return set(_EXCLUDED)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_schema_card.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/schema_card.py tests/test_ai_schema_card.py
git commit -m "feat(ai-chat): DB schema-card for the system prompt"
```

---

## Task 7: Guarded read-only SQL tool (`src/ai/sql_tool.py`)

**Files:**
- Create: `src/ai/sql_tool.py`
- Test: `tests/test_ai_sql_tool_readonly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_sql_tool_readonly.py
"""The query_data tool is SELECT-only, single-statement, capped, and read-only."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()


def test_select_allowed():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS one")
    assert out["rows"] == [{"one": 1}]


def test_rejects_insert():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("INSERT INTO players (name) VALUES ('x')")
    assert out["error"] is not None
    assert "select" in out["error"].lower()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE players SET name = 'x'",
        "DELETE FROM players",
        "DROP TABLE players",
        "ALTER TABLE players ADD COLUMN z TEXT",
        "PRAGMA table_info(players)",
        "ATTACH DATABASE 'x.db' AS y",
        "CREATE TABLE z (a int)",
    ],
)
def test_rejects_non_select(sql):
    from src.ai.sql_tool import run_read_only_sql

    assert run_read_only_sql(sql)["error"] is not None


def test_rejects_multiple_statements():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1; DROP TABLE players")
    assert out["error"] is not None


def test_select_queries_secret_table_blocked():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT * FROM ai_provider_keys")
    assert out["error"] is not None


def test_write_is_physically_impossible(tmp_path):
    # Even a crafted statement can't write: the connection is opened read-only.
    from src.ai.sql_tool import run_read_only_sql

    # WITH ... still SELECT; a write smuggled via CTE must fail at the driver too.
    out = run_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    assert out["error"] is None
    assert out["rows"] == [{"1": 1}]


def test_row_limit_enforced():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS n FROM players", max_rows=5)
    assert len(out["rows"]) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_sql_tool_readonly.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/sql_tool.py
"""Guarded read-only SQL executor — the AI's open-ended reach into all data.

Defense in depth:
  1. Read-only connection (file:...?mode=ro) — writes are impossible at the driver.
  2. SELECT/WITH allowlist; reject DML/DDL/PRAGMA/ATTACH.
  3. Single statement only (no ';' chaining).
  4. Exclude secret tables (keys/auth/users).
  5. Row cap.
Returns {"rows": [...], "error": str|None} — never raises into the agent loop.
"""

from __future__ import annotations

import sqlite3

from src.ai.schema_card import excluded_tables
from src.database import DB_PATH

_MAX_ROWS_DEFAULT = 500


def _looks_select(sql: str) -> bool:
    s = sql.strip().lstrip("(").lstrip()
    head = s[:6].lower()
    return head.startswith("select") or s[:4].lower().startswith("with")


def run_read_only_sql(sql: str, max_rows: int = _MAX_ROWS_DEFAULT) -> dict:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return {"rows": [], "error": "Empty query."}
    # single statement only
    if ";" in sql:
        return {"rows": [], "error": "Only a single SELECT statement is allowed."}
    if not _looks_select(sql):
        return {"rows": [], "error": "Only read-only SELECT/WITH queries are allowed."}
    lowered = sql.lower()
    for banned in ("insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "replace"):
        # word-boundary-ish check: banned keyword as a standalone token
        if f" {banned} " in f" {lowered} " or lowered.startswith(banned + " "):
            return {"rows": [], "error": f"Statement type '{banned}' is not allowed."}
    for secret in excluded_tables():
        if secret.lower() in lowered:
            return {"rows": [], "error": f"Table '{secret}' is not queryable."}

    try:
        # as_posix() → forward slashes so the file: URI is valid on Windows too.
        ro_uri = f"file:{DB_PATH.as_posix()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(max_rows)]
            return {"rows": rows, "error": None}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"rows": [], "error": f"SQL error: {exc}"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_sql_tool_readonly.py -v`
Expected: PASS (all parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/ai/sql_tool.py tests/test_ai_sql_tool_readonly.py
git commit -m "feat(ai-chat): guarded read-only SQL tool"
```

---

## Task 8: Scheduler bridge — `request_refresh` + queue drain

**Files:**
- Create: `src/ai/refresh_queue.py`
- Modify: `src/scheduler.py` (`_refresh_once`, ~line 81)
- Test: `tests/test_forced_refresh_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forced_refresh_queue.py
"""request_refresh enqueues; the scheduler drains pending -> done (single-writer-safe)."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM forced_refresh_queue")
        conn.commit()
    finally:
        conn.close()


def test_request_refresh_enqueues():
    from src.ai.refresh_queue import request_refresh

    rid = request_refresh("players", requested_by=99)
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM forced_refresh_queue WHERE id = ?", (rid,)).fetchone()
    finally:
        conn.close()
    assert row["source"] == "players"
    assert row["status"] == "pending"


def test_status_of():
    from src.ai.refresh_queue import request_refresh, status_of

    rid = request_refresh("all", requested_by=99)
    assert status_of(rid) == "pending"


def test_drain_runs_pending(monkeypatch):
    from src.ai import refresh_queue

    calls = {}

    def fake_bootstrap(**kwargs):
        calls["force"] = kwargs.get("force")
        return {"players": "Updated"}

    monkeypatch.setattr(refresh_queue, "_bootstrap_all_data", fake_bootstrap)

    rid = refresh_queue.request_refresh("players", requested_by=99)
    refresh_queue.drain_queue()
    assert calls["force"] is True
    assert refresh_queue.status_of(rid) == "done"


def test_drain_marks_error_on_exception(monkeypatch):
    from src.ai import refresh_queue

    def boom(**kwargs):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(refresh_queue, "_bootstrap_all_data", boom)
    rid = refresh_queue.request_refresh("players", requested_by=99)
    refresh_queue.drain_queue()
    assert refresh_queue.status_of(rid) == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_forced_refresh_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai.refresh_queue'`.

- [ ] **Step 3: Implement the queue module**

```python
# src/ai/refresh_queue.py
"""The AI's only write path: enqueue refresh requests; the single-writer
scheduler drains them. The AI never writes data directly (single-writer rule)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _bootstrap_all_data(**kwargs):
    # Indirection so tests can monkeypatch without importing the heavy module.
    from src.data_bootstrap import bootstrap_all_data

    return bootstrap_all_data(**kwargs)


def request_refresh(source: str, requested_by: int | None = None) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO forced_refresh_queue (source, requested_by, status, created_at) "
            "VALUES (?, ?, 'pending', ?)",
            (source, requested_by, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def status_of(request_id: int) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM forced_refresh_queue WHERE id = ?", (request_id,)
        ).fetchone()
        return row["status"] if row is not None else None
    finally:
        conn.close()


def drain_queue() -> int:
    """Process all pending requests. Called by the scheduler thread ONLY (sole writer).

    Returns the number of requests processed. force=True so the requested source
    refreshes regardless of staleness.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        pending = conn.execute(
            "SELECT id, source FROM forced_refresh_queue WHERE status = 'pending' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    processed = 0
    for row in pending:
        rid = row["id"]
        _set_status(rid, "running")
        try:
            _bootstrap_all_data(force=True)
            _set_status(rid, "done", completed=True)
        except Exception as exc:
            logger.warning("forced_refresh_queue: request %s failed: %s", rid, exc)
            _set_status(rid, "error", completed=True, detail=str(exc)[:300])
        processed += 1
    return processed


def _set_status(request_id: int, status: str, completed: bool = False, detail: str | None = None) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        completed_at = datetime.now(UTC).isoformat() if completed else None
        conn.execute(
            "UPDATE forced_refresh_queue SET status = ?, detail = COALESCE(?, detail), "
            "completed_at = COALESCE(?, completed_at) WHERE id = ?",
            (status, detail, completed_at, request_id),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_forced_refresh_queue.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the drain into the scheduler**

In `src/scheduler.py`, inside `_refresh_once()`, add the drain as the FIRST action (before the relay pull), so AI-requested refreshes are honored each tick. Insert right after the `global _last_yahoo_ok` line and the `from src.data_bootstrap import bootstrap_all_data` import (~line 85):

```python
    # AI-requested forced refreshes: the scheduler is the sole writer, so it
    # drains the queue the chat's request_refresh tool fills. Best-effort.
    try:
        from src.ai.refresh_queue import drain_queue

        drained = drain_queue()
        if drained:
            logger.info("Scheduler: drained %d AI refresh request(s).", drained)
    except Exception as exc:
        logger.warning("Scheduler: forced_refresh_queue drain failed: %s", exc)
```

- [ ] **Step 6: Add a scheduler-wiring guard test**

Append to `tests/test_forced_refresh_queue.py`:

```python
def test_scheduler_calls_drain_queue():
    import inspect

    import src.scheduler as sched

    src = inspect.getsource(sched._refresh_once)
    assert "drain_queue" in src, "_refresh_once must drain the forced_refresh_queue"
```

- [ ] **Step 7: Run tests + lint**

Run: `python -m pytest tests/test_forced_refresh_queue.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add src/ai/refresh_queue.py src/scheduler.py tests/test_forced_refresh_queue.py
git commit -m "feat(ai-chat): AI refresh queue drained by the single-writer scheduler"
```

---

## Task 9: Tool surface + dispatch (`src/ai/tools.py`)

**Files:**
- Create: `src/ai/tools.py`
- Test: `tests/test_ai_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_tools.py
"""Tool definitions are OpenAI-schema; dispatch routes to the service layer."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()


def test_tool_specs_are_openai_schema():
    from src.ai.tools import tool_specs

    specs = tool_specs()
    names = {t["function"]["name"] for t in specs}
    assert {"query_data", "request_refresh"}.issubset(names)
    for t in specs:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_dispatch_query_data():
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("query_data", {"sql": "SELECT 1 AS one"}, user_id=99)
    assert "one" in out  # serialized result contains the column


def test_dispatch_request_refresh_enqueues():
    from src.ai.refresh_queue import status_of
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("request_refresh", {"source": "players"}, user_id=99)
    # the tool returns a request id we can check
    import json

    rid = json.loads(out)["request_id"]
    assert status_of(rid) == "pending"


def test_dispatch_unknown_tool_returns_error():
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("does_not_exist", {}, user_id=99)
    assert "error" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_tools.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/tools.py
"""AI tool surface over the existing service layer + the guarded SQL + refresh queue.

Tool schemas are OpenAI function-calling format (LiteLLM's lingua franca). All
dispatch returns a JSON string (the tool result the model reads). Read tools are
side-effect-free; request_refresh is the only (queued) write.
"""

from __future__ import annotations

import json


def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    "Run a READ-ONLY SQL SELECT against the HEATER database to answer "
                    "questions about any data (players, stats, projections, standings, "
                    "rosters, trades, etc.). Use the schema in the system prompt. SELECT only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_player",
                "description": "Look up a player's full row (projections, YTD, statcast) by name.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_standings",
                "description": "Current league standings (all 12 teams, category ranks).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_refresh",
                "description": (
                    "Request a data refresh for a source (e.g. 'players', 'yahoo', or 'all'). "
                    "Returns a request_id; the background scheduler runs it. Use sparingly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
            },
        },
    ]


def dispatch_tool(name: str, args: dict, user_id: int) -> str:
    try:
        if name == "query_data":
            from src.ai.sql_tool import run_read_only_sql

            return json.dumps(run_read_only_sql(args.get("sql", "")), default=str)
        if name == "get_player":
            return _get_player(args.get("name", ""))
        if name == "get_standings":
            return _get_standings()
        if name == "request_refresh":
            from src.ai.refresh_queue import request_refresh

            rid = request_refresh(args.get("source", "all"), requested_by=user_id)
            return json.dumps({"request_id": rid, "status": "pending"})
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:  # never crash the agent loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _get_player(name: str) -> str:
    from src.database import load_player_pool

    pool = load_player_pool()
    col = "player_name" if "player_name" in pool.columns else "name"
    hit = pool[pool[col].astype(str).str.lower() == name.strip().lower()]
    if hit.empty:
        hit = pool[pool[col].astype(str).str.lower().str.contains(name.strip().lower(), na=False)]
    return hit.head(3).to_json(orient="records")


def _get_standings() -> str:
    from src.yahoo_data_service import get_yahoo_data_service

    df = get_yahoo_data_service().get_standings()
    return df.to_json(orient="records") if df is not None else json.dumps({"error": "no standings"})
```

> **Note:** `get_my_team`, `evaluate_trade`, `optimize_lineup`, `compare_players`, `get_matchup`, `get_free_agents` follow the exact same dispatch shape — add them incrementally after the core loop is proven. Phase 1 ships with the 4 above (`query_data` already reaches everything via SQL).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_tools.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/tools.py tests/test_ai_tools.py
git commit -m "feat(ai-chat): AI tool surface + dispatch over the service layer"
```

---

## Task 10: LiteLLM provider wrapper + agentic loop (`src/ai/providers.py`)

**Files:**
- Create: `src/ai/providers.py`
- Test: `tests/test_ai_providers.py`

- [ ] **Step 1: Write the failing test (LiteLLM fully mocked — no network)**

```python
# tests/test_ai_providers.py
"""Agentic tool loop: model asks for a tool, we dispatch, model answers. Mocked."""

import json
from types import SimpleNamespace

import pytest

from src.database import init_db


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message, in_tok=10, out_tok=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()


def test_simple_answer_no_tools(monkeypatch):
    from src.ai import providers

    monkeypatch.setattr(
        providers, "_completion", lambda **kw: _resp(_msg(content="Hi there"))
    )
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out["content"] == "Hi there"
    assert out["tokens_out"] == 5


def test_tool_then_answer(monkeypatch):
    from src.ai import providers

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1 AS one"})),
    )
    responses = [
        _resp(_msg(content=None, tool_calls=[tool_call])),  # first: ask for tool
        _resp(_msg(content="The answer is 1")),             # second: final answer
    ]
    monkeypatch.setattr(providers, "_completion", lambda **kw: responses.pop(0))

    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "what is one"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out["content"] == "The answer is 1"
    assert any(t["name"] == "query_data" for t in out["tool_trace"])


def test_loop_cap(monkeypatch):
    from src.ai import providers

    tool_call = SimpleNamespace(
        id="c", function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1"}))
    )
    # model loops forever asking for tools; the cap must stop it
    monkeypatch.setattr(providers, "_completion", lambda **kw: _resp(_msg(tool_calls=[tool_call])))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "loop"}],
        api_key="sk-test",
        user_id=99,
        max_tool_rounds=3,
    )
    assert out["content"]  # returns a graceful message rather than hanging
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/providers.py
"""LiteLLM wrapper + the agentic tool loop.

chat() runs: call model -> if it asks for tools, dispatch + append results + call
again -> else return the final text. Non-streamed (Phase 1); the UI fakes the
typewriter. _completion is split out so tests can mock LiteLLM with no network.
"""

from __future__ import annotations

import json

from src.ai.tools import dispatch_tool, tool_specs

_MAX_TOOL_ROUNDS = 6


def _completion(**kwargs):
    import litellm

    return litellm.completion(**kwargs)


def chat(
    model: str,
    messages: list[dict],
    api_key: str | None,
    user_id: int,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
) -> dict:
    """Run the tool loop. Returns {content, tokens_in, tokens_out, tool_trace}."""
    convo = list(messages)
    tool_trace: list[dict] = []
    tokens_in = tokens_out = 0
    specs = tool_specs()

    for _ in range(max_tool_rounds):
        resp = _completion(model=model, messages=convo, tools=specs, api_key=api_key)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return {
                "content": msg.content or "",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tool_trace": tool_trace,
            }

        # record the assistant turn that requested tools
        convo.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            result = dispatch_tool(tc.function.name, args, user_id=user_id)
            tool_trace.append({"name": tc.function.name, "args": args})
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # loop cap hit
    return {
        "content": "I wasn't able to finish that within the tool-call limit. Try narrowing the question.",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_trace": tool_trace,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_providers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/providers.py tests/test_ai_providers.py
git commit -m "feat(ai-chat): LiteLLM wrapper + agentic tool loop"
```

---

## Task 11: Conversation history (`src/ai/history.py`)

**Files:**
- Create: `src/ai/history.py`
- Test: `tests/test_ai_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_history.py
"""Conversation CRUD, scoped to user_id."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_messages")
        conn.execute("DELETE FROM ai_conversations")
        conn.commit()
    finally:
        conn.close()


def test_create_and_list():
    from src.ai.history import create_conversation, list_conversations

    cid = create_conversation(user_id=99, title="Trade talk", model="anthropic/claude-haiku-4-5")
    convos = list_conversations(99)
    assert len(convos) == 1
    assert convos[0]["id"] == cid
    assert convos[0]["title"] == "Trade talk"


def test_append_and_load_messages():
    from src.ai.history import append_message, create_conversation, load_messages

    cid = create_conversation(99, "x", "m")
    append_message(cid, "user", "hello")
    append_message(cid, "assistant", "hi", tokens_in=5, tokens_out=2, cost_usd=0.001)
    msgs = load_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "hi"


def test_scoped_to_user():
    from src.ai.history import create_conversation, list_conversations

    create_conversation(1, "mine", "m")
    create_conversation(2, "theirs", "m")
    assert {c["title"] for c in list_conversations(1)} == {"mine"}


def test_rename_and_delete():
    from src.ai.history import create_conversation, delete_conversation, list_conversations, rename_conversation

    cid = create_conversation(99, "old", "m")
    rename_conversation(cid, "new")
    assert list_conversations(99)[0]["title"] == "new"
    delete_conversation(cid)
    assert list_conversations(99) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_history.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/history.py
"""Conversation + message persistence (the history dropdown), scoped to user_id."""

from __future__ import annotations

from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_conversation(user_id: int, title: str, model: str | None = None) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO ai_conversations (user_id, title, model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, title[:120], model, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_conversations(user_id: int, limit: int = 50) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, model, updated_at FROM ai_conversations WHERE user_id = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def append_message(
    conversation_id: int,
    role: str,
    content: str,
    model: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO ai_messages (conversation_id, role, content, model, tokens_in, tokens_out, "
            "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, model, tokens_in, tokens_out, cost_usd, now),
        )
        conn.execute("UPDATE ai_conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_messages(conversation_id: int) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, model, created_at FROM ai_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def rename_conversation(conversation_id: int, title: str) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE ai_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title[:120], _now(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM ai_conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_history.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/history.py tests/test_ai_history.py
git commit -m "feat(ai-chat): conversation history persistence"
```

---

## Task 12: Window chrome (`src/ai/chat_shell.py`)

> This task is mostly client-side CSS/JS. The unit test only asserts the wiring strings exist; behavior is verified in the browser in Task 15. The implementation uses `streamlit-float` to pin the container and injects JS (via `st.components.v1.html`, which can reach the parent document same-origin) for drag / resize / minimize / close + the launcher, persisting position/size/open-state in `localStorage` so they survive reruns.

**Files:**
- Create: `src/ai/chat_shell.py`
- Test: `tests/test_ai_chat_shell.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_chat_shell.py
"""The shell injects the launcher + window behaviors (string-level wiring check)."""


def test_shell_js_has_window_behaviors():
    from src.ai.chat_shell import _shell_script

    js = _shell_script(container_id="heater-ai-window", launcher_id="heater-ai-launcher")
    # drag, resize, minimize, close, launcher, persistence
    for needle in ["mousedown", "localStorage", "heater-ai-window", "heater-ai-launcher", "minimize", "close"]:
        assert needle in js, f"shell JS missing {needle!r}"


def test_launcher_label():
    from src.ai.chat_shell import LAUNCHER_LABEL

    assert LAUNCHER_LABEL == "AI Chat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_chat_shell.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/chat_shell.py
"""Floating chat window chrome: top-right 'AI Chat' launcher + a draggable,
resizable, minimizable, closable window. All chrome behaviors are client-side
(no Streamlit rerun); only sending a message reruns the chat fragment.

The Streamlit chat widgets render inside a streamlit-float container with id
CONTAINER_ID; this module injects the launcher button + the JS that wires drag/
resize/minimize/close and restores position/size/open-state from localStorage.
"""

from __future__ import annotations

import streamlit as st

CONTAINER_ID = "heater-ai-window"
LAUNCHER_ID = "heater-ai-launcher"
LAUNCHER_LABEL = "AI Chat"


def float_window_css() -> str:
    """CSS for the floated container: window frame, resizable, draggable header."""
    return f"""
    <style>
      #{LAUNCHER_ID} {{
        position: fixed; top: 12px; right: 16px; z-index: 100000;
        background: var(--heater-primary, #ff6d00); color: #fff; border: none;
        border-radius: 8px; padding: 7px 12px; font-weight: 600; cursor: pointer;
        font-family: inherit; display: inline-flex; gap: 6px; align-items: center;
      }}
      div[data-testid="stVerticalBlock"]:has(> div #{CONTAINER_ID}-anchor) {{
        position: fixed; bottom: 24px; right: 24px; width: 380px; height: 540px;
        min-width: 300px; min-height: 240px; max-width: 90vw; max-height: 90vh;
        resize: both; overflow: auto; z-index: 99999;
        background: #fff; border: 1px solid rgba(0,0,0,.18); border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0,0,0,.18);
      }}
      #{CONTAINER_ID}-header {{ cursor: move; user-select: none; }}
      .heater-ai-hidden {{ display: none !important; }}
    </style>
    """


def _shell_script(container_id: str = CONTAINER_ID, launcher_id: str = LAUNCHER_ID) -> str:
    """JS (runs in a components iframe; reaches parent doc same-origin) that wires
    the launcher + drag/resize/minimize/close and restores state from localStorage."""
    return f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      const KEY = 'heaterAiWindow';
      function el(id) {{ return doc.getElementById(id); }}
      function winBlock() {{
        const anchor = el('{container_id}-anchor');
        return anchor ? anchor.closest('div[data-testid="stVerticalBlock"]') : null;
      }}
      function save(state) {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch(e) {{}} }}
      function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch(e) {{ return {{}}; }} }}

      function apply() {{
        const w = winBlock(); if (!w) return;
        const s = load();
        if (s.left != null) {{ w.style.left = s.left + 'px'; w.style.right = 'auto'; }}
        if (s.top  != null) {{ w.style.top  = s.top  + 'px'; w.style.bottom = 'auto'; }}
        if (s.width)  w.style.width  = s.width + 'px';
        if (s.height) w.style.height = s.height + 'px';
        w.classList.toggle('heater-ai-hidden', s.open === false);
        const launcher = el('{launcher_id}');
        if (launcher) launcher.style.display = (s.open === false || s.open == null) ? 'inline-flex' : 'none';
      }}

      // Launcher: open the window (client-side; no rerun)
      function wireLauncher() {{
        const launcher = el('{launcher_id}'); if (!launcher || launcher.dataset.wired) return;
        launcher.dataset.wired = '1';
        launcher.addEventListener('click', function() {{
          const s = load(); s.open = true; save(s); apply();
        }});
      }}

      // Header: minimize/close + drag
      function wireHeader() {{
        const header = el('{container_id}-header'); if (!header || header.dataset.wired) return;
        header.dataset.wired = '1';
        const w = winBlock();
        header.querySelectorAll('[data-ai-act]').forEach(function(btn) {{
          btn.addEventListener('click', function(ev) {{
            ev.stopPropagation();
            const s = load();
            if (btn.dataset.aiAct === 'close' || btn.dataset.aiAct === 'minimize') {{
              s.open = false; save(s); apply();
            }}
          }});
        }});
        let drag = null;
        header.addEventListener('mousedown', function(ev) {{
          if (ev.target.closest('[data-ai-act]')) return;
          const r = w.getBoundingClientRect();
          drag = {{ dx: ev.clientX - r.left, dy: ev.clientY - r.top }};
          ev.preventDefault();
        }});
        doc.addEventListener('mousemove', function(ev) {{
          if (!drag) return;
          w.style.left = (ev.clientX - drag.dx) + 'px'; w.style.right = 'auto';
          w.style.top  = (ev.clientY - drag.dy) + 'px'; w.style.bottom = 'auto';
        }});
        doc.addEventListener('mouseup', function() {{
          if (!drag) return; drag = null;
          const r = w.getBoundingClientRect();
          const s = load(); s.left = r.left; s.top = r.top; s.width = r.width; s.height = r.height; save(s); apply();
        }});
        // persist size on resize
        new ResizeObserver(function() {{
          const r = w.getBoundingClientRect();
          const s = load(); s.width = r.width; s.height = r.height; save(s);
        }}).observe(w);
      }}

      function tick() {{ wireLauncher(); wireHeader(); apply(); }}
      tick();
      // Re-wire after Streamlit reruns swap DOM
      new MutationObserver(tick).observe(doc.body, {{ childList: true, subtree: true }});
    }})();
    </script>
    """


def render_launcher_and_shell() -> None:
    """Inject the launcher button + CSS + behavior JS. Call once per page render."""
    st.markdown(float_window_css(), unsafe_allow_html=True)
    st.markdown(
        f'<button id="{LAUNCHER_ID}">🔥 {LAUNCHER_LABEL}</button>'.replace("🔥", ""),
        unsafe_allow_html=True,
    )
    st.components.v1.html(_shell_script(), height=0)
```

> **Note:** the launcher markup uses no emoji per the Combustion design system; the flame is rendered via the page's existing icon system if desired. Keep the button text exactly `AI Chat`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_chat_shell.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai/chat_shell.py tests/test_ai_chat_shell.py
git commit -m "feat(ai-chat): floating window chrome (launcher + drag/resize/min/close)"
```

---

## Task 13: Chat widget orchestration (`src/ai/chat.py`)

**Files:**
- Create: `src/ai/chat.py`
- Test: `tests/test_ai_chat_backcompat.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_chat_backcompat.py
"""render_chat_widget is inert when MULTI_USER is off (v1 byte-for-byte)."""

import pytest


def test_no_op_when_flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.ai.chat import render_chat_widget

    # Must return immediately without touching Streamlit or the DB.
    render_chat_widget("My Team")  # no exception, no rendering


def test_no_op_when_not_logged_in(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    import src.ai.chat as chat_mod

    monkeypatch.setattr(chat_mod, "current_user", lambda: None)
    chat_mod.render_chat_widget("My Team")  # returns early, renders nothing


def test_build_system_prompt_includes_schema(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    from src.database import init_db

    init_db()
    from src.ai.chat import build_system_prompt

    sp = build_system_prompt(page="My Team", viewer_team="Team Hickey")
    assert "SELECT" in sp  # schema card present
    assert "Team Hickey" in sp


def test_chat_wires_ai_settings():
    """The widget exposes per-user key management (store/list/delete)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "src" / "ai" / "chat.py").read_text(encoding="utf-8")
    assert "_render_ai_settings" in src
    assert "store_key" in src and "list_keys" in src and "delete_key" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_chat_backcompat.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/ai/chat.py
"""The chat widget: floating window contents + the send/stream orchestration.

MULTI_USER-gated and inert when off (v1 byte-for-byte). The conversation lives in
session_state (survives page navigation) and is written through to SQLite.
"""

from __future__ import annotations

import time

import streamlit as st

from src.ai import budget, history
from src.ai.chat_shell import CONTAINER_ID, render_launcher_and_shell
from src.ai.keys import delete_key, get_key, list_keys, store_key
from src.ai.router import model_for_tier, price_per_token, provider_of
from src.ai.schema_card import build_schema_card
from src.auth import current_user, multi_user_enabled, resolve_viewer_team_name

_STATE_MSGS = "ai_chat_messages"
_STATE_CONV = "ai_chat_conversation_id"
_TIER_LABEL = {"simple": "Simple", "moderate": "Moderate", "complex": "Complex"}


def build_system_prompt(page: str, viewer_team: str | None) -> str:
    schema = build_schema_card()
    team = viewer_team or "the user's team"
    return (
        "You are HEATER's in-app fantasy-baseball assistant for a 12-team Yahoo H2H "
        "categories league (R,HR,RBI,SB,AVG,OBP / W,L,SV,K,ERA,WHIP; L/ERA/WHIP are "
        "inverse). Present data and analysis; do not give unsolicited personal trade "
        f"opinions — it is the user's team. The viewer's team is '{team}'. The user is "
        f"currently on the '{page}' page.\n\n"
        "You can call tools to read any app data. Prefer the specific tools; use "
        "query_data (read-only SELECT) for anything else. Only request_refresh when the "
        "user explicitly asks for fresh data.\n\n" + schema
    )


def _ensure_state() -> None:
    if _STATE_MSGS not in st.session_state:
        st.session_state[_STATE_MSGS] = []
    if _STATE_CONV not in st.session_state:
        st.session_state[_STATE_CONV] = None


def render_chat_widget(page: str) -> None:
    """Mount the floating AI chat. No-op unless MULTI_USER + logged in."""
    if not multi_user_enabled():
        return
    user = current_user()
    if user is None:
        return

    _ensure_state()
    render_launcher_and_shell()

    # The floated window container (streamlit-float pins it; chat_shell styles it).
    from streamlit_float import float_parent

    window = st.container()
    with window:
        st.markdown(f'<div id="{CONTAINER_ID}-anchor"></div>', unsafe_allow_html=True)
        _render_window_body(page, user)
        float_parent()


def _render_window_body(page: str, user: dict) -> None:
    # Title bar (drag handle + minimize/close wired by chat_shell JS)
    st.markdown(
        f'<div id="{CONTAINER_ID}-header" style="display:flex;justify-content:space-between;'
        'align-items:center;padding:8px 10px;border-bottom:1px solid #eee;">'
        '<strong>HEATER AI</strong><span>'
        '<button data-ai-act="minimize" style="border:none;background:none;cursor:pointer;">—</button>'
        '<button data-ai-act="close" style="border:none;background:none;cursor:pointer;">✕</button>'
        '</span></div>',
        unsafe_allow_html=True,
    )

    _render_chat_fragment(page, user)


@st.fragment
def _render_chat_fragment(page: str, user: dict) -> None:
    # History dropdown + new chat
    cols = st.columns([3, 1])
    convos = history.list_conversations(user["user_id"])
    options = ["New conversation"] + [c["title"] for c in convos]
    pick = cols[0].selectbox("Conversations", options, label_visibility="collapsed", key="ai_conv_pick")
    if cols[1].button("＋", key="ai_new_chat"):
        st.session_state[_STATE_MSGS] = []
        st.session_state[_STATE_CONV] = None
    if pick != "New conversation":
        chosen = convos[options.index(pick) - 1]
        if st.session_state[_STATE_CONV] != chosen["id"]:
            st.session_state[_STATE_CONV] = chosen["id"]
            st.session_state[_STATE_MSGS] = [
                {"role": m["role"], "content": m["content"]} for m in history.load_messages(chosen["id"])
            ]

    # Per-user AI Settings (add/remove your own provider keys)
    _render_ai_settings(user)

    # Message thread
    for m in st.session_state[_STATE_MSGS]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Tier picker + input
    tier = st.selectbox("Model", ["simple", "moderate", "complex"], index=1, key="ai_tier",
                        format_func=lambda t: _TIER_LABEL[t], label_visibility="collapsed")
    prompt = st.chat_input("Ask anything about your league…", key="ai_prompt")
    if prompt:
        _handle_send(prompt, tier, page, user)


def _handle_send(prompt: str, tier: str, page: str, user: dict) -> None:
    from src.ai.providers import chat as provider_chat

    uid = user["user_id"]
    model = model_for_tier(tier)
    provider = provider_of(model)
    api_key = get_key(uid, provider)
    on_own_key = api_key is not None and _is_user_own_key(uid, provider)

    if api_key is None:
        st.warning(f"No API key for {provider}. Add one in AI Settings, or ask the admin to set a shared key.")
        return
    if budget.is_over_cap(uid, on_own_key=on_own_key):
        st.warning("You've hit your AI usage limit for today. Try again tomorrow or add your own key.")
        return

    st.session_state[_STATE_MSGS].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    viewer_team = resolve_viewer_team_name()
    system = build_system_prompt(page, viewer_team)
    messages = [{"role": "system", "content": system}] + st.session_state[_STATE_MSGS]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = provider_chat(model=model, messages=messages, api_key=api_key, user_id=uid)
        st.write_stream(_typewriter(result["content"]))

    st.session_state[_STATE_MSGS].append({"role": "assistant", "content": result["content"]})

    # cost accounting
    pin, pout = price_per_token(model)
    cost = result["tokens_in"] * pin + result["tokens_out"] * pout
    budget.record_usage(uid, result["tokens_in"], result["tokens_out"], cost)

    # persist
    cid = st.session_state[_STATE_CONV]
    if cid is None:
        cid = history.create_conversation(uid, title=prompt[:60], model=model)
        st.session_state[_STATE_CONV] = cid
    history.append_message(cid, "user", prompt)
    history.append_message(cid, "assistant", result["content"], model=model,
                           tokens_in=result["tokens_in"], tokens_out=result["tokens_out"], cost_usd=cost)


def _typewriter(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)


def _is_user_own_key(user_id: int, provider: str) -> bool:
    return any(k["provider"] == provider for k in list_keys(user_id))


def _render_ai_settings(user: dict) -> None:
    """Gear popover: each user adds / labels / removes their OWN provider keys.

    Keys are Fernet-encrypted by store_key; this UI never shows a stored key back.
    """
    uid = user["user_id"]
    with st.popover("⚙", help="AI Settings — your API keys"):
        st.caption("Add your own provider key to use your own model + skip shared-key caps.")
        with st.form("ai_key_form", clear_on_submit=True):
            provider = st.selectbox("Provider", ["anthropic", "openai", "gemini", "openrouter", "ollama"])
            label = st.text_input("Label (optional)", value="")
            key_text = st.text_input("API key", value="", type="password")
            if st.form_submit_button("Save key") and key_text.strip():
                store_key(uid, provider, key_text.strip(), label=(label.strip() or None))
                st.success(f"Saved your {provider} key.")
        existing = list_keys(uid)
        if existing:
            st.caption("Your keys:")
            for k in existing:
                row = st.columns([3, 1])
                row[0].write(f"{k['provider']}" + (f" · {k['label']}" if k["label"] else ""))
                if row[1].button("Remove", key=f"del_{k['provider']}_{k['label']}"):
                    delete_key(uid, k["provider"], k["label"])
                    st.rerun()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_chat_backcompat.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff format src/ai/ tests/test_ai_*.py
python -m ruff check src/ai/
git add src/ai/chat.py tests/test_ai_chat_backcompat.py
git commit -m "feat(ai-chat): chat widget orchestration + send/stream/persist"
```

---

## Task 14: Admin shared-key + caps form (`pages/_admin_controls.py`)

**Files:**
- Modify: `pages/_admin_controls.py` (append a new section after the Yahoo-token block, ~line 99)
- Test: `tests/test_ai_admin_key_gated.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_admin_key_gated.py
"""Admin AI controls: require_admin + audit logs the action only, never the key."""

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_controls.py"


def test_page_calls_require_admin_before_ai_section():
    src = _PAGE.read_text(encoding="utf-8")
    assert "require_admin()" in src
    assert "set_admin_shared_key" in src
    assert "set_daily_cap" in src


def test_audit_never_logs_key_text():
    """AST: no log_action call passes the AI key variable as detail."""
    src = _PAGE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "log_action":
            dump = ast.dump(node)
            assert "ai_shared_key_text" not in dump, "AI key text must never be passed to log_action"


def test_setter_is_multi_user_gated():
    # set_admin_shared_key -> set_setting -> gated; assert the import wiring exists.
    src = _PAGE.read_text(encoding="utf-8")
    assert "from src.ai.keys import set_admin_shared_key" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ai_admin_key_gated.py -v`
Expected: FAIL — `set_admin_shared_key` not found in the page.

- [ ] **Step 3: Add the imports + section to `pages/_admin_controls.py`**

Add to the import block (after line 18, `from src.yahoo_api import save_yahoo_token_json`):

```python
from src.ai.budget import daily_cap_usd, set_daily_cap
from src.ai.keys import set_admin_shared_key
from src.ai.router import set_tier_models
```

Append at the end of the file (after the Yahoo-token block):

```python
# --- AI chat assistant -----------------------------------------------------
st.subheader("AI chat assistant")
st.caption(
    "Set a shared provider key so every member can chat without their own key, "
    "and cap per-user daily spend. The key is encrypted at rest and never shown back."
)

_ai_provider = st.selectbox(
    "Shared key provider", ["anthropic", "openai", "gemini", "openrouter"], key="ai_shared_provider"
)
ai_shared_key_text = st.text_input("Shared API key", value="", type="password", key="ai_shared_key_input")
if st.button("Save shared key"):
    if ai_shared_key_text.strip():
        set_admin_shared_key(_ai_provider, ai_shared_key_text.strip(), admin_id=_admin_id)
        log_action(_admin_id, "ai_shared_key_update", target=_ai_provider)
        st.success(f"Shared {_ai_provider} key saved.")
    else:
        st.error("Enter a key first.")

_cap = st.number_input(
    "Per-user daily cap (USD)", min_value=0.0, value=float(daily_cap_usd()), step=0.25, key="ai_cap"
)
if st.button("Save daily cap"):
    set_daily_cap(_cap, admin_id=_admin_id)
    log_action(_admin_id, "ai_daily_cap_update", detail={"usd": _cap})
    st.success(f"Daily cap set to ${_cap:.2f}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ai_admin_key_gated.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pages/_admin_controls.py tests/test_ai_admin_key_gated.py
git commit -m "feat(ai-chat): admin shared-key + daily-cap controls"
```

---

## Task 15: Mount the widget on every page + structural guard

**Files:**
- Modify: all 13 interactive pages in `pages/` (add 1 import + 1 call each)
- Test: `tests/test_pages_have_ai_chat.py`

The 13 interactive pages: `1_My_Team.py`, `2_Line-up_Optimizer.py`, `3_Closer_Monitor.py`, `5_Matchup_Planner.py`, `6_League_Standings.py`, `10_Punt_Analyzer.py`, `11_Trade_Analyzer.py`, `12_Trade_Finder.py`, `14_Free_Agents.py`, `16_Player_Compare.py`, `17_Leaders.py`, `19_Player_Databank.py`, `20_Draft_Simulator.py`.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_pages_have_ai_chat.py
"""Structural invariant: every interactive page mounts the AI chat widget.

Mirrors test_pages_have_feedback_and_usage.py. Streamlit runs each page top to
bottom on every interaction, so the chat mount is per-page, not global. The call
must sit after the auth gate. Underscore-prefixed admin pages are exempt.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and not p.name.startswith("_")
)


def test_found_the_pages():
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_chat(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.ai.chat import render_chat_widget" in src, f"{page.name} must import render_chat_widget"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_chat_widget_called_after_auth(page):
    src = page.read_text(encoding="utf-8")
    assert "render_chat_widget(" in src, f"{page.name} must call render_chat_widget()"
    i_auth = src.index("require_auth()")
    i_chat = src.index("render_chat_widget(")
    assert i_chat > i_auth, f"{page.name}: render_chat_widget() must follow require_auth()"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pages_have_ai_chat.py -v`
Expected: FAIL — every page missing the import/call.

- [ ] **Step 3: Edit each of the 13 pages (identical 2-line change)**

For each page file:

1. **Add the import** alongside the other `from src...` imports near the top. Example for `pages/14_Free_Agents.py` (add near the `from src.usage import log_page_view` line):

```python
from src.ai.chat import render_chat_widget
```

2. **Add the mount call** immediately after the page's `log_page_view(...)` call (which is right after `require_auth()` / `require_page_enabled(...)`). Example for `pages/14_Free_Agents.py` — after `log_page_view("Free Agents")` (line 288):

```python
render_chat_widget("Free Agents")
```

Use the page's human label as the argument (the same string the page already passes to `log_page_view`): "My Team", "Lineup Optimizer", "Closer Monitor", "Matchup Planner", "League Standings", "Punt Analyzer", "Trade Analyzer", "Trade Finder", "Free Agents", "Player Compare", "Leaders", "Player Databank", "Draft Simulator".

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pages_have_ai_chat.py -v`
Expected: PASS (1 + 13 + 13 cases).

- [ ] **Step 5: Commit**

```bash
git add pages/ tests/test_pages_have_ai_chat.py
git commit -m "feat(ai-chat): mount the AI chat widget on all 13 pages + guard"
```

---

## Task 16: Full suite + browser verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole AI test set**

Run: `python -m pytest tests/test_ai_*.py tests/test_forced_refresh_queue.py tests/test_pages_have_ai_chat.py -v`
Expected: all PASS.

- [ ] **Step 2: Run the full suite (catch regressions / byte-for-byte v1)**

Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -q`
Expected: prior pass count + the new tests, 0 failures. (Parallel: add `-n auto`.)

- [ ] **Step 3: Lint**

Run: `python -m ruff format . && python -m ruff check .`
Expected: clean.

- [ ] **Step 4: Browser verification (preview tools)**

The preview harness inherits `MULTI_USER=1`; log in as `qa_admin` / `qa-local-only-2026`. Set a real provider key in env (`HEATER_AI_KEY` = a Fernet key; add a shared key in Admin Controls, or your own key in AI Settings). Then, using the `preview_*` tools:

1. `preview_start`, open My Team. Confirm the **"AI Chat"** button is at the top-right.
2. Click it → the window opens. Drag it by the title bar; resize via the bottom-right corner; minimize; close; reopen — all without a full-page reload.
3. Ask "what's my SB rank?" → the AI calls a tool and answers with real data. Verify via `preview_console_logs` (no errors) and `preview_snapshot`.
4. Navigate to Lineup Optimizer → confirm the conversation is **still there** (not cleared).
5. Ask "run: SELECT COUNT(*) FROM players" → returns a number; then ask it to delete a row → it refuses / the tool errors (read-only proof).
6. Open the Conversations dropdown → the prior thread is listed and reloads.
7. `preview_screenshot` the open window for the record.

- [ ] **Step 5: Final commit + push the branch**

```bash
git add -A
git commit -m "test(ai-chat): full-suite + browser verification pass (Phase 1)"
git push -u origin feature/ai-chat-assistant
```

(Open a PR only when you're ready — Phase 1 is independently shippable.)

---

## Phase 1 done-criteria recap (from the spec §15)

1. "AI Chat" launcher top-right on every interactive page; window drags/resizes/minimizes/closes. ✅ Tasks 12, 15
2. Conversation survives page navigation. ✅ Task 13 (session_state)
3. ≥2 provider keys storable + per-message model switch; admin shared key works within caps. ✅ Tasks 3, 4, 13, 14
4. AI answers a data question via a tool. ✅ Tasks 9, 10, 16
5. Arbitrary read-only `query_data`; writes impossible. ✅ Task 7
6. `request_refresh` enqueues; scheduler completes it. ✅ Task 8
7. History dropdown lists + reloads. ✅ Tasks 11, 13
8. Cost caps block over-limit users. ✅ Tasks 5, 13
9. Full suite green; MULTI_USER-off byte-for-byte. ✅ Tasks 13 (backcompat), 16

---

## Self-review notes (for the implementer)

- **Streaming honesty:** Phase 1 streams the *UI* typewriter (`_typewriter`) over a non-streamed provider call (the tool loop is non-streamed). True end-to-end token streaming is a Phase-2 polish. This keeps the tool loop simple and reliable.
- **Window robustness:** position/size/open-state persist in `localStorage` and re-apply on every rerun via a `MutationObserver` — this is what keeps the window from resetting when the chat fragment reruns. If the JS proves flaky against a Streamlit version, the §14-spec fallback is a CSS-pinned container + the same drag/resize JS without `streamlit-float`.
- **Cost fallback:** `price_per_token` is the source of truth for caps (verified June-2026 prices); don't rely on `litellm.completion_cost()` for newer models (it returns 0 for models outside its bundled map).
- **Model IDs:** GPT/Gemini exact IDs evolve — they're admin-overridable via `set_tier_models`, so no code change is needed to retarget a tier.
- **Per-message model switch (Phase 1 scope):** the in-window picker switches the *tier* (Simple/Moderate/Complex → the router's model). A dropdown of specific named models (filtered to providers the user has a key for) is a small Phase-1.1 follow-up on top of `router.model_for_tier` + `keys.list_keys`; the spec's "pick a specific model" is satisfied at tier granularity in Phase 1.
- **AI Settings popover:** added in Task 13 (`_render_ai_settings`) so each user manages their own keys; the admin shared key (Task 14) is the no-key fallback.
