# Yahoo Token Relay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a fresh Yahoo access token on the Railway server at all times by refreshing it on the owner's residential mini PC and relaying it (encrypted) via a GitHub gist, so the server never calls the OAuth refresh endpoint that Yahoo blocks for datacenter IPs.

**Architecture:** A mini-PC script refreshes the token (residential IP), encrypts it (Fernet), and PATCHes it into a secret gist every 30 min. The Railway scheduler pulls + decrypts that gist each cycle and writes it to the volume *before* its existing reconnect runs — so yfpy sees a valid token and never refreshes. All server-side relay code is dormant unless `HEATER_TOKEN_RELAY_URL` + `HEATER_RELAY_KEY` are set (v1 byte-for-byte preserved).

**Tech Stack:** Python 3.12 (server) / 3.14 (mini PC), `requests`, `cryptography` (Fernet), GitHub Gist API, Windows Task Scheduler, pytest, Streamlit, SQLite, Railway.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/yahoo_api.py` | Add `refresh_yahoo_token()` (direct refresh + surfaces the real Yahoo error) | Modify |
| `src/token_relay.py` | Encrypt/decrypt helpers, `pull_relayed_token()`, relay gating | Create |
| `src/logging_setup.py` | `configure_src_logging()` — idempotent stdout/file handler attach | Create |
| `src/scheduler.py` | Extract `_refresh_once()`, call `pull_relayed_token()` before reconnect, degrade-once logging | Modify |
| `app.py` | Call `configure_src_logging()` at top of `main()` | Modify |
| `src/data_bootstrap.py` | Replace import-time logging block with `configure_src_logging()` call | Modify |
| `scripts/yahoo_token_relay.py` | Mini-PC relay entrypoint (refresh → encrypt → upload) | Create |
| `requirements.txt` | Add `cryptography` | Modify |
| `tests/test_refresh_yahoo_token.py` | Tests for the refresh helper | Create |
| `tests/test_token_relay.py` | Tests for encrypt/decrypt + pull + gating + no-secret-logging | Create |
| `tests/test_logging_setup.py` | Runtime test that `src.*` INFO reaches stdout; idempotency | Create |
| `tests/test_scheduler_relay_wiring.py` | `_refresh_once` pulls before reconnect; degrade-once | Create |
| `tests/test_src_logs_to_stdout.py` | Update stale source-grep guard to point at new module | Modify |

---

## Task 1: `refresh_yahoo_token()` — direct refresh that surfaces the real Yahoo error

**Files:**
- Modify: `src/yahoo_api.py` (add function near the other standalone helpers, after `exchange_code_for_token`)
- Test: `tests/test_refresh_yahoo_token.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refresh_yahoo_token.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_api import refresh_yahoo_token

_TOKEN = {
    "consumer_key": "ck", "consumer_secret": "cs", "refresh_token": "rt",
    "access_token": "old", "token_time": 1.0, "token_type": "bearer", "guid": "g",
}


def _resp(status, payload=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    r.json.return_value = payload if payload is not None else {}
    r.text = text
    return r


def test_refresh_success_merges_new_access_token():
    new = {"access_token": "NEW", "expires_in": 3600, "token_type": "bearer"}
    with patch("src.yahoo_api._requests.post", return_value=_resp(200, new)):
        out = refresh_yahoo_token(_TOKEN)
    assert out is not None
    assert out["access_token"] == "NEW"
    assert out["refresh_token"] == "rt"          # preserved
    assert out["token_time"] > 1.0               # advanced
    assert out["consumer_key"] == "ck"           # untouched


def test_refresh_rotated_refresh_token_is_persisted():
    new = {"access_token": "NEW", "refresh_token": "RT2"}
    with patch("src.yahoo_api._requests.post", return_value=_resp(200, new)):
        out = refresh_yahoo_token(_TOKEN)
    assert out["refresh_token"] == "RT2"


def test_refresh_invalid_consumer_key_returns_none_and_logs_real_error(caplog):
    err = {"error": "INVALID_CONSUMER_KEY", "error_description": "Client ID does not exist"}
    with patch("src.yahoo_api._requests.post", return_value=_resp(400, err)):
        out = refresh_yahoo_token(_TOKEN)
    assert out is None
    assert "INVALID_CONSUMER_KEY" in caplog.text          # the REAL error, surfaced


def test_refresh_missing_fields_returns_none():
    assert refresh_yahoo_token({"consumer_key": "ck"}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_refresh_yahoo_token.py -v`
Expected: FAIL — `ImportError: cannot import name 'refresh_yahoo_token'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/yahoo_api.py` (after `exchange_code_for_token`, before `create_streamlit_oauth_component`):

```python
def refresh_yahoo_token(token_dict: dict) -> dict | None:
    """Refresh a Yahoo OAuth access token via a direct token-endpoint POST.

    Works from residential IPs (the mini-PC relay). On the Railway datacenter IP
    Yahoo rejects this with INVALID_CONSUMER_KEY — which this function now SURFACES
    in the log instead of letting it surface as a bare KeyError elsewhere.

    Returns the input dict merged with the new ``access_token``/``token_time`` (and
    any rotated ``refresh_token``), or ``None`` on any failure.
    """
    import base64

    ck = token_dict.get("consumer_key")
    cs = token_dict.get("consumer_secret")
    rt = token_dict.get("refresh_token")
    if not (ck and cs and rt):
        logger.warning("refresh_yahoo_token: token missing consumer_key/secret/refresh_token.")
        return None

    basic = base64.b64encode(f"{ck}:{cs}".encode()).decode()
    try:
        resp = _requests.post(
            "https://api.login.yahoo.com/oauth2/get_token",
            headers={
                "Authorization": "Basic " + basic,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "refresh_token", "redirect_uri": "oob", "refresh_token": rt},
            timeout=20,
        )
    except Exception as exc:
        logger.warning("refresh_yahoo_token: request failed: %s", type(exc).__name__)
        return None

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", "")
        except Exception:
            err = (resp.text or "")[:120]
        logger.warning(
            "refresh_yahoo_token: Yahoo refused the refresh (HTTP %s, error=%s). If this is the "
            "Railway server, its datacenter IP is blocked from refreshing — the mini-PC relay must "
            "supply the token.",
            resp.status_code,
            err,
        )
        return None

    data = resp.json()
    if "access_token" not in data:
        logger.warning("refresh_yahoo_token: response missing access_token (keys=%s).", sorted(data.keys()))
        return None

    merged = dict(token_dict)
    merged["access_token"] = data["access_token"]
    merged["token_time"] = time.time()
    if data.get("refresh_token"):
        merged["refresh_token"] = data["refresh_token"]
    if data.get("expires_in"):
        merged["expires_in"] = data["expires_in"]
    if data.get("token_type"):
        merged["token_type"] = data["token_type"]
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_refresh_yahoo_token.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/yahoo_api.py tests/test_refresh_yahoo_token.py
git commit -m "feat(yahoo): refresh_yahoo_token direct helper that surfaces the real Yahoo error"
```

---

## Task 2: Add `cryptography` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt` (in the analytics/core section, one line):

```
cryptography>=42.0
```

- [ ] **Step 2: Install it into the local venv**

Run: `.venv\Scripts\python.exe -m pip install "cryptography>=42.0"`
Expected: "Successfully installed cryptography-…"

- [ ] **Step 3: Verify import**

Run: `.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print('ok', len(Fernet.generate_key()))"`
Expected: `ok 44`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add cryptography (Fernet) for token relay encryption"
```

---

## Task 3: `src/token_relay.py` — encryption + `pull_relayed_token()`

**Files:**
- Create: `src/token_relay.py`
- Test: `tests/test_token_relay.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_token_relay.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import token_relay

_TOKEN = {"access_token": "AT", "refresh_token": "RT", "consumer_key": "ck",
          "consumer_secret": "cs", "token_time": 2000.0, "token_type": "bearer"}


def test_encrypt_decrypt_roundtrip():
    f = Fernet(Fernet.generate_key())
    blob = token_relay.encrypt_token(_TOKEN, f)
    assert _TOKEN["access_token"] not in blob          # ciphertext, not plaintext
    assert token_relay.decrypt_token(blob, f) == _TOKEN


def test_pull_is_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("HEATER_TOKEN_RELAY_URL", raising=False)
    monkeypatch.delenv("HEATER_RELAY_KEY", raising=False)
    with patch("src.token_relay._requests.get") as g:
        assert token_relay.pull_relayed_token() is False
        g.assert_not_called()                          # v1: no network at all


def test_pull_writes_when_newer(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    written = {}
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: {"token_time": 1000.0})
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: written.update(t) or True)
    resp = MagicMock(); resp.text = blob; resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is True
    assert written["access_token"] == "AT"


def test_pull_skips_when_not_newer(monkeypatch):
    key = Fernet.generate_key(); f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)              # token_time 2000
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: {"token_time": 9999.0})
    wrote = []
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: wrote.append(t) or True)
    resp = MagicMock(); resp.text = blob; resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is False
    assert wrote == []


def test_pull_never_logs_token_contents(monkeypatch, caplog):
    key = Fernet.generate_key(); f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: None)
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: True)
    resp = MagicMock(); resp.text = blob; resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        token_relay.pull_relayed_token()
    assert "AT" not in caplog.text and "RT" not in caplog.text


def test_pull_handles_decrypt_failure(monkeypatch):
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", Fernet.generate_key().decode())
    resp = MagicMock(); resp.text = "not-cipher"; resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_token_relay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.token_relay'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/token_relay.py
"""Encrypted Yahoo-token relay (mini PC ⇄ GitHub gist ⇄ Railway server).

Root cause this solves: Yahoo blocks OAuth token refresh from Railway's datacenter
IP. The mini PC refreshes the token from a residential IP and uploads it (encrypted)
to a secret gist; the server pulls it here. All server-side behavior is DORMANT
unless HEATER_TOKEN_RELAY_URL + HEATER_RELAY_KEY are set (v1 byte-for-byte).
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests as _requests
from cryptography.fernet import Fernet

from src.yahoo_api import _AUTH_DIR, _write_token_file

logger = logging.getLogger(__name__)

GIST_FILENAME = "heater_yahoo_token.enc"
_relay_healthy: bool | None = None  # for degrade/recover transition logging


def _fernet() -> Fernet | None:
    key = os.environ.get("HEATER_RELAY_KEY", "").strip()
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except Exception:
        logger.warning("token_relay: HEATER_RELAY_KEY is not a valid Fernet key.")
        return None


def encrypt_token(token_dict: dict, fernet: Fernet) -> str:
    return fernet.encrypt(json.dumps(token_dict).encode()).decode()


def decrypt_token(ciphertext: str, fernet: Fernet) -> dict:
    return json.loads(fernet.decrypt(ciphertext.encode()).decode())


def relay_enabled() -> bool:
    return bool(os.environ.get("HEATER_TOKEN_RELAY_URL") and os.environ.get("HEATER_RELAY_KEY"))


def _read_local_token() -> dict | None:
    p = _AUTH_DIR / "yahoo_token.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def pull_relayed_token() -> bool:
    """Fetch + decrypt the relayed token from the gist raw URL; write it to the
    volume if it is newer than what's on disk. No-op (returns False) when the relay
    is not configured. Never logs token contents."""
    url = os.environ.get("HEATER_TOKEN_RELAY_URL", "").strip()
    f = _fernet()
    if not url or f is None:
        return False

    try:
        resp = _requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("pull_relayed_token: could not fetch relay token: %s", type(exc).__name__)
        return False

    try:
        token = decrypt_token(resp.text.strip(), f)
    except Exception:
        logger.warning("pull_relayed_token: decrypt failed (HEATER_RELAY_KEY mismatch?).")
        return False

    if not (token.get("access_token") and token.get("refresh_token")):
        logger.warning("pull_relayed_token: relayed token missing access/refresh token.")
        return False

    existing = _read_local_token()
    new_tt = float(token.get("token_time") or 0)
    old_tt = float((existing or {}).get("token_time") or 0)
    if existing and new_tt <= old_tt:
        return False

    if not _write_token_file(token):
        return False

    age_min = max(0, int((time.time() - new_tt) / 60)) if new_tt else -1
    logger.info("pull_relayed_token: wrote relayed token (age ~%dm).", age_min)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_token_relay.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/token_relay.py tests/test_token_relay.py
git commit -m "feat(relay): encrypted token relay pull + Fernet helpers (dormant unless configured)"
```

---

## Task 4: Scheduler wiring — pull before reconnect + degrade-once logging

**Files:**
- Modify: `src/scheduler.py` (extract `_refresh_once`, add pull + transition logging)
- Test: `tests/test_scheduler_relay_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler_relay_wiring.py
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.scheduler as sch


def test_refresh_once_pulls_before_reconnect():
    calls = []
    with patch("src.token_relay.pull_relayed_token", side_effect=lambda: calls.append("pull")), \
         patch("src.yahoo_api.try_reconnect_yahoo", side_effect=lambda: calls.append("reconnect") or None), \
         patch("src.data_bootstrap.bootstrap_all_data", return_value={}):
        sch._refresh_once()
    assert calls == ["pull", "reconnect"]


def test_degrade_logged_once_then_recover(caplog):
    sch._last_yahoo_ok = None
    with patch("src.token_relay.pull_relayed_token", return_value=False), \
         patch("src.data_bootstrap.bootstrap_all_data", return_value={}):
        # two failed cycles -> degrade logged ONCE
        with patch("src.yahoo_api.try_reconnect_yahoo", return_value=None):
            sch._refresh_once()
            sch._refresh_once()
        assert caplog.text.count("Yahoo sync degraded") == 1
        # recovery -> logged once
        caplog.clear()

        class _C:
            def persist_current_token(self):  # noqa: D401
                return True

        with patch("src.yahoo_api.try_reconnect_yahoo", return_value=_C()):
            sch._refresh_once()
        assert caplog.text.count("Yahoo sync recovered") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scheduler_relay_wiring.py -v`
Expected: FAIL — `AttributeError: module 'src.scheduler' has no attribute '_refresh_once'`

- [ ] **Step 3: Write minimal implementation**

In `src/scheduler.py`, add a module global near the top (after `_CHECK_INTERVAL_SECONDS`):

```python
_last_yahoo_ok: bool | None = None  # for degrade/recover transition logging
```

Replace the body of `_refresh_loop` (the `while` block) by extracting a single-cycle
helper. The new `_refresh_loop` plus the new `_refresh_once`:

```python
def _refresh_once() -> None:
    """One scheduler cycle: pull the relayed token, reconnect, bootstrap. Extracted
    from the loop so it is unit-testable."""
    global _last_yahoo_ok
    from src.data_bootstrap import bootstrap_all_data

    # Relay: refresh the on-disk token from the gist BEFORE reconnecting, so yfpy
    # sees a valid token and never calls Yahoo's (datacenter-blocked) refresh.
    try:
        from src.token_relay import pull_relayed_token

        pull_relayed_token()
    except Exception as exc:
        logger.warning("Scheduler: relay token pull failed: %s", exc)

    yahoo_client = None
    try:
        from src.yahoo_api import try_reconnect_yahoo

        yahoo_client = try_reconnect_yahoo()
    except Exception as exc:
        logger.warning("Scheduler Yahoo reconnect failed: %s", exc)

    # Degrade/recover transition — log ONCE, not every cycle (no error storm).
    if yahoo_client is None and _last_yahoo_ok is not False:
        logger.warning("Yahoo sync degraded — relayed token stale? Is the mini-PC relay running?")
        _last_yahoo_ok = False
    elif yahoo_client is not None and _last_yahoo_ok is not True:
        if _last_yahoo_ok is False:
            logger.info("Yahoo sync recovered.")
        _last_yahoo_ok = True

    results = bootstrap_all_data(yahoo_client=yahoo_client, force=False)
    if yahoo_client is not None:
        try:
            yahoo_client.persist_current_token()
        except Exception:
            logger.warning("Scheduler: persisting refreshed Yahoo token failed.", exc_info=True)
    refreshed = [k for k, v in results.items() if v != "Fresh"]
    if refreshed:
        logger.info("Background refresh updated: %s", refreshed)


def _refresh_loop():
    """Main scheduler loop — see _refresh_once for one cycle's work."""
    while _scheduler_running:
        try:
            _refresh_once()
        except Exception as e:
            logger.warning("Background refresh error: %s", e)
        if _stop_event.wait(timeout=_CHECK_INTERVAL_SECONDS):
            break
```

(Delete the old inline loop body that `_refresh_once` now contains.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scheduler_relay_wiring.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the existing scheduler/Plan-4 guards to ensure no regression**

Run: `python -m pytest tests/test_plan4_scheduler_wiring.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/scheduler.py tests/test_scheduler_relay_wiring.py
git commit -m "feat(scheduler): pull relayed token before reconnect; degrade/recover logged once"
```

---

## Task 5: Observability — idempotent `configure_src_logging()` wired into `main()`

**Files:**
- Create: `src/logging_setup.py`
- Modify: `app.py` (call at top of `main()`), `src/data_bootstrap.py` (replace import-time block)
- Test: `tests/test_logging_setup.py`, and update `tests/test_src_logs_to_stdout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_setup.py
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _clear_src_handlers():
    lg = logging.getLogger("src")
    for h in list(lg.handlers):
        lg.removeHandler(h)


def test_configure_attaches_stdout_handler_at_runtime(monkeypatch):
    monkeypatch.delenv("HEATER_DISABLE_FILE_LOG", raising=False)
    monkeypatch.setitem(sys.modules, "pytest", sys.modules["pytest"])  # pytest IS loaded
    from src.logging_setup import configure_src_logging, _MARKER

    _clear_src_handlers()
    # Force the real attach even though pytest is loaded, by calling the internal path:
    configure_src_logging(_force=True)
    handlers = logging.getLogger("src").handlers
    assert any(getattr(h, _MARKER, False) for h in handlers)

    # Runtime proof: a src.* INFO record reaches the stdout handler's stream.
    buf = io.StringIO()
    for h in handlers:
        if getattr(h, _MARKER, False):
            h.stream = buf
    logging.getLogger("src.demo").info("hello-relay")
    assert "hello-relay" in buf.getvalue()


def test_configure_is_idempotent(monkeypatch):
    from src.logging_setup import configure_src_logging, _MARKER

    _clear_src_handlers()
    configure_src_logging(_force=True)
    configure_src_logging(_force=True)
    marked = [h for h in logging.getLogger("src").handlers if getattr(h, _MARKER, False)]
    assert len(marked) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_logging_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.logging_setup'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/logging_setup.py
"""Idempotent logging setup for the ``src`` logger.

Why this exists: e7d7150 attached a stdout handler as an *import side-effect* of
src.data_bootstrap, which does not reliably take effect on the Railway deploy (the
scheduler's INFO lines never reached the console). This module exposes an explicit,
idempotent setup that ``main()`` calls at startup, so the handler is attached in the
real runtime path regardless of import order.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MARKER = "_heater_src_stdout"


def configure_src_logging(_force: bool = False) -> None:
    """Attach a stdout handler (and a best-effort rotating file handler) to the
    ``src`` logger at INFO level, exactly once. Skipped under pytest unless _force."""
    under_pytest = "pytest" in sys.modules or os.environ.get("HEATER_DISABLE_FILE_LOG") == "1"
    if under_pytest and not _force:
        return

    src_logger = logging.getLogger("src")
    if any(getattr(h, _MARKER, False) for h in src_logger.handlers):
        return  # already configured

    src_logger.setLevel(logging.INFO)
    src_logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    setattr(stream, _MARKER, True)
    src_logger.addHandler(stream)

    try:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(log_dir / "bootstrap.log", maxBytes=5_000_000, backupCount=3)
        fileh.setFormatter(fmt)
        src_logger.addHandler(fileh)
    except Exception:
        pass  # file logging is best-effort; stdout is what matters on Railway
```

- [ ] **Step 4: Wire it into `main()` and `data_bootstrap`**

In `app.py`, at the **very top** of `def main():` (before `init_session()`):

```python
def main():
    from src.logging_setup import configure_src_logging

    configure_src_logging()
    init_session()
    ...
```

In `src/data_bootstrap.py`, replace the import-time logging block (the
`if not _UNDER_PYTEST:` block that builds handlers, lines ~33-50) with:

```python
if not _UNDER_PYTEST:
    from src.logging_setup import configure_src_logging

    configure_src_logging()
```

(Keep the `_UNDER_PYTEST` definition above it unchanged.)

- [ ] **Step 5: Update the stale source-grep guard**

Replace `tests/test_src_logs_to_stdout.py` body with a check that points at the new
module and asserts the runtime function exists:

```python
# tests/test_src_logs_to_stdout.py
"""Guard: src.* logs must reach stdout via the idempotent configure_src_logging()
(2026-06-06: moved out of data_bootstrap's import side-effect, which didn't take
effect on Railway). Runtime behavior is covered by tests/test_logging_setup.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_logging_setup_module_exists_and_targets_src_stdout():
    from src import logging_setup

    src = (Path(__file__).parent.parent / "src" / "logging_setup.py").read_text(encoding="utf-8")
    assert "StreamHandler(sys.stdout)" in src
    assert 'getLogger("src")' in src
    assert hasattr(logging_setup, "configure_src_logging")


def test_main_calls_configure_src_logging():
    app = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
    assert "configure_src_logging()" in app
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_logging_setup.py tests/test_src_logs_to_stdout.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/logging_setup.py app.py src/data_bootstrap.py tests/test_logging_setup.py tests/test_src_logs_to_stdout.py
git commit -m "fix(logging): idempotent configure_src_logging() called from main() (reliable Railway stdout)"
```

---

## Task 6: Mini-PC relay script

**Files:**
- Create: `scripts/yahoo_token_relay.py`
- Test: `tests/test_relay_script.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_relay_script.py
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

relay = importlib.import_module("scripts.yahoo_token_relay")


def test_main_returns_1_when_no_token_file(monkeypatch, tmp_path):
    monkeypatch.setattr(relay, "TOKEN_FILE", tmp_path / "missing.json")
    assert relay.main() == 1


def test_main_returns_3_when_no_relay_key(monkeypatch, tmp_path):
    tok = tmp_path / "yahoo_token.json"
    tok.write_text('{"consumer_key":"ck","consumer_secret":"cs","refresh_token":"rt"}')
    monkeypatch.setattr(relay, "TOKEN_FILE", tok)
    monkeypatch.delenv("HEATER_RELAY_KEY", raising=False)
    fresh = {"access_token": "NEW", "refresh_token": "rt", "consumer_key": "ck",
             "consumer_secret": "cs", "token_time": 1.0}
    with patch.object(relay, "refresh_yahoo_token", return_value=fresh), \
         patch.object(relay, "_write_token_file", return_value=True):
        assert relay.main() == 3


def test_main_uploads_on_success(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    tok = tmp_path / "yahoo_token.json"
    tok.write_text('{"consumer_key":"ck","consumer_secret":"cs","refresh_token":"rt"}')
    monkeypatch.setattr(relay, "TOKEN_FILE", tok)
    monkeypatch.setenv("HEATER_RELAY_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("HEATER_GIST_ID", "gid")
    monkeypatch.setenv("HEATER_GIST_PAT", "pat")
    fresh = {"access_token": "NEW", "refresh_token": "rt", "consumer_key": "ck",
             "consumer_secret": "cs", "token_time": 1.0}
    resp = MagicMock(); resp.status_code = 200
    with patch.object(relay, "refresh_yahoo_token", return_value=fresh), \
         patch.object(relay, "_write_token_file", return_value=True), \
         patch.object(relay.requests, "patch", return_value=resp) as p:
        assert relay.main() == 0
    p.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_relay_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.yahoo_token_relay'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/yahoo_token_relay.py
"""HEATER mini-PC Yahoo-token relay.

Run every 30 min by Windows Task Scheduler. Refreshes the Yahoo token from this
machine's (residential) IP — which Yahoo accepts, unlike the Railway datacenter IP —
encrypts it, and uploads it to a secret GitHub gist that the Railway server reads.

Env required: HEATER_RELAY_KEY (Fernet key), HEATER_GIST_ID, HEATER_GIST_PAT (gist scope).
Never prints the token or any secret.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.token_relay import _fernet, encrypt_token  # noqa: E402
from src.yahoo_api import _AUTH_DIR, _write_token_file, refresh_yahoo_token  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("yahoo_token_relay")

TOKEN_FILE = _AUTH_DIR / "yahoo_token.json"
GIST_FILENAME = "heater_yahoo_token.enc"


def main() -> int:
    if not TOKEN_FILE.exists():
        logger.error("No token file at %s — seed it once via the local OAuth flow.", TOKEN_FILE)
        return 1
    token = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))

    fresh = refresh_yahoo_token(token)
    if fresh is None:
        logger.error("Refresh failed (see the Yahoo error above).")
        return 2
    _write_token_file(fresh)  # keep the local seed current

    f = _fernet()
    if f is None:
        logger.error("HEATER_RELAY_KEY not set / invalid — cannot encrypt.")
        return 3

    gist_id = os.environ.get("HEATER_GIST_ID", "").strip()
    pat = os.environ.get("HEATER_GIST_PAT", "").strip()
    if not (gist_id and pat):
        logger.error("HEATER_GIST_ID / HEATER_GIST_PAT not set — cannot upload.")
        return 4

    ciphertext = encrypt_token(fresh, f)
    resp = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {pat}", "Accept": "application/vnd.github+json"},
        json={"files": {GIST_FILENAME: {"content": ciphertext}}},
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        logger.error("Gist upload failed: HTTP %s", resp.status_code)
        return 5

    logger.info("Relay OK: token refreshed + uploaded (token_time advanced).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_relay_script.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/yahoo_token_relay.py tests/test_relay_script.py
git commit -m "feat(relay): mini-PC relay script (refresh on residential IP, encrypt, push to gist)"
```

---

## Task 7: Full-suite + structural-guard regression check

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -q -n auto --dist loadfile`
Expected: all pass (prior baseline ~4869) + the new tests; 0 failures.

- [ ] **Step 2: Run ruff**

Run: `python -m ruff check . && python -m ruff format --check .`
Expected: clean (fix + re-run if needed).

- [ ] **Step 3: Commit any formatting**

```bash
git add -A && git commit -m "style: ruff format for token-relay changes" || echo "nothing to format"
```

---

## Task 8: Code review (three reviewers)

- [ ] **Step 1: silent-failure-hunter** — invoke `pr-review-toolkit:silent-failure-hunter` on the diff (`git diff master...fix/yahoo-token-relay`). Focus: every `except` in `refresh_yahoo_token`, `pull_relayed_token`, the relay script, and the scheduler must log actionably and never silently swallow. Fix findings (TDD).
- [ ] **Step 2: code-reviewer** — invoke `pr-review-toolkit:code-reviewer` on the same diff. Fix findings.
- [ ] **Step 3: coderabbit** — invoke `coderabbit:code-review`. Triage + fix real findings.
- [ ] **Step 4: Re-run** `python -m pytest --ignore=tests/test_cheat_sheet.py -q` after fixes. Expected: PASS.
- [ ] **Step 5: Commit** any review fixes with clear messages.

---

## Task 9: One-time setup (owner + me) — gist, keys, env, Task Scheduler

> These are ops steps, executed with the owner. No code; confirm each with the owner.

- [ ] **Step 1 (owner, guided):** Create a GitHub PAT with **only `gist`** scope at
  github.com → Settings → Developer settings → Personal access tokens. Do **not** paste it
  in chat — set it directly (next steps).
- [ ] **Step 2 (me):** Generate a Fernet key locally:
  `.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
  (Capture for the owner to place into env vars — handled so it isn't echoed in prose.)
- [ ] **Step 3 (me, with owner's PAT in env):** Create the secret gist via API with an initial
  `heater_yahoo_token.enc` placeholder; record the gist id + the stable raw URL
  `https://gist.githubusercontent.com/<user>/<id>/raw/heater_yahoo_token.enc`.
- [ ] **Step 4 (mini PC env):** Set `HEATER_RELAY_KEY`, `HEATER_GIST_ID`, `HEATER_GIST_PAT`
  as persistent user env vars (`setx`), so Task Scheduler runs see them.
- [ ] **Step 5 (mini PC):** Smoke-run the relay once:
  `.venv\Scripts\python.exe scripts\yahoo_token_relay.py` → expect `Relay OK`; verify the gist
  updated and `data/yahoo_token.json` `token_time` advanced.
- [ ] **Step 6 (mini PC):** Register a Task Scheduler job (every 30 min, run whether logged in or
  not, using the venv python + the script path); confirm a scheduled run produces `Relay OK`.
- [ ] **Step 7 (owner, guided):** In Railway → Variables, add `HEATER_TOKEN_RELAY_URL` (the raw
  gist URL) and `HEATER_RELAY_KEY` (same Fernet key). Confirm before saving (this triggers a redeploy).

---

## Task 10: Deploy + LIVE verification (the real bar)

> REQUIRED SUB-SKILL: superpowers:verification-before-completion. Green tests are necessary but NOT sufficient.

- [ ] **Step 1 (confirm with owner):** Push `fix/yahoo-token-relay` and merge/deploy to Railway
  (auto-deploys on push to master; ~3–5 min; resets sessions). Confirm before pushing.
- [ ] **Step 2:** Via Railway Console, watch `data/logs/bootstrap.log` and stdout for, each cycle:
  `pull_relayed_token: wrote relayed token (age ~Xm)` and a **successful** reconnect with **no**
  `INVALID_CONSUMER_KEY` / "authentication returned False".
- [ ] **Step 3:** Confirm the gist updates ~every 30 min and the server token's `token_time`
  advances (Console one-liner printing only `token_time` + age).
- [ ] **Step 4:** Watch the app's **Data Freshness** panel: all 7 rows stay **Live/Cached** for
  **≥ 2 hours including at least one hour boundary** (≥ 1 relay refresh). The "Warming up / No
  matchup data" state must be gone.
- [ ] **Step 5:** If any step fails, return to systematic-debugging — do NOT claim done.

---

## Task 11: Lock it in — CLAUDE.md + guard summary

> REQUIRED SUB-SKILL: claude-md-management:revise-claude-md

- [ ] **Step 1:** Record in `CLAUDE.md`: the **real** root cause (Yahoo blocks OAuth refresh from
  the Railway datacenter IP; misleading `INVALID_CONSUMER_KEY`), the relay architecture, the new
  env vars, and the new guard tests (token_relay, logging_setup, scheduler_relay_wiring,
  refresh_yahoo_token). Update the "Known Design Choices" note that previously framed this as
  hang/throttle.
- [ ] **Step 2:** Commit.

---

## Task 12: hookify secret-guard hook (owner opted in)

- [ ] **Step 1:** Use the `hookify` plugin to add a hook that blocks the Yahoo token, client
  secret, Fernet key, and PAT from being printed to logs or committed (pattern-based pre-commit /
  output guard). Verify it triggers on a planted dummy and not on normal code.
- [ ] **Step 2:** Commit the hook config.

---

## Task 13: Finish the branch

> REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch

- [ ] **Step 1:** Ensure full suite green + live verification complete.
- [ ] **Step 2:** Open the PR / merge to master per the owner's choice (confirm the push). Then
  proceed to onboarding (revoke `testuser` → invite 12 → approve/assign) per the live-onboarding
  memory.

---

## Self-Review (writing-plans)

- **Spec coverage:** relay (T6,T9) · gist drop-box (T3,T9) · server reader + scheduler wiring
  (T3,T4) · refresh helper / real-error surfacing (T1) · observability (T5) · no-storm (T4) ·
  security/encryption (T3,T9,T12) · config/env (T9,T10) · cryptography dep (T2) · TDD tests
  (T1,T3,T4,T5,T6) · reviews (T8) · live verification (T10) · invariants/back-compat
  (T3 no-op, T4 guard) · CLAUDE.md (T11). All spec sections map to a task.
- **Placeholder scan:** every code step contains complete code; commands have expected output.
- **Type consistency:** `refresh_yahoo_token(dict)->dict|None`, `pull_relayed_token()->bool`,
  `encrypt_token/decrypt_token`, `_read_local_token`, `_write_token_file`, `_fernet`,
  `configure_src_logging`, `_refresh_once`, `_last_yahoo_ok`, env names
  (`HEATER_TOKEN_RELAY_URL`/`HEATER_RELAY_KEY`/`HEATER_GIST_ID`/`HEATER_GIST_PAT`) — consistent
  across tasks.
