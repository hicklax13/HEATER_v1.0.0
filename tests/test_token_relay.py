import json
import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import token_relay


@pytest.fixture(autouse=True)
def _reset_relay_healthy():
    """Reset the module-level relay health flag before each test so the
    degrade/recover transition logging is deterministic and order-independent."""
    token_relay._relay_healthy = None
    yield


_TOKEN = {
    "access_token": "AT",
    "refresh_token": "RT",
    "consumer_key": "ck",
    "consumer_secret": "cs",
    "token_time": 2000.0,
    "token_type": "bearer",
}


def test_encrypt_decrypt_roundtrip():
    f = Fernet(Fernet.generate_key())
    blob = token_relay.encrypt_token(_TOKEN, f)
    assert _TOKEN["access_token"] not in blob
    assert token_relay.decrypt_token(blob, f) == _TOKEN


def test_pull_is_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("HEATER_TOKEN_RELAY_URL", raising=False)
    monkeypatch.delenv("HEATER_RELAY_KEY", raising=False)
    with patch("src.token_relay._requests.get") as g:
        assert token_relay.pull_relayed_token() is False
        g.assert_not_called()


def test_pull_writes_when_newer(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    written = {}
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: {"token_time": 1000.0})
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: written.update(t) or True)
    resp = MagicMock()
    resp.text = blob
    resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is True
    assert written["access_token"] == "AT"


def test_pull_skips_when_not_newer(monkeypatch):
    key = Fernet.generate_key()
    f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: {"token_time": 9999.0})
    wrote = []
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: wrote.append(t) or True)
    resp = MagicMock()
    resp.text = blob
    resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is False
    assert wrote == []


def test_pull_warns_once_when_relay_stale(monkeypatch, caplog):
    """When the gist stops advancing past the stale threshold (mini-PC relay down),
    warn exactly once on the healthy->stale transition — not every cycle."""
    caplog.set_level(logging.WARNING)
    key = Fernet.generate_key()
    f = Fernet(key)
    stale = dict(_TOKEN)
    stale["token_time"] = time.time() - 3600  # 60 min old -> past the 45m warn threshold
    blob = token_relay.encrypt_token(stale, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: {"token_time": stale["token_time"]})
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: True)
    resp = MagicMock()
    resp.text = blob
    resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is False
        assert token_relay.pull_relayed_token() is False
    assert caplog.text.count("relay gist not advancing") == 1


def test_pull_never_logs_token_contents(monkeypatch, caplog):
    key = Fernet.generate_key()
    f = Fernet(key)
    blob = token_relay.encrypt_token(_TOKEN, f)
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", key.decode())
    monkeypatch.setattr(token_relay, "_read_local_token", lambda: None)
    monkeypatch.setattr(token_relay, "_write_token_file", lambda t: True)
    resp = MagicMock()
    resp.text = blob
    resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        token_relay.pull_relayed_token()
    assert "AT" not in caplog.text and "RT" not in caplog.text


def test_pull_handles_decrypt_failure(monkeypatch):
    monkeypatch.setenv("HEATER_TOKEN_RELAY_URL", "https://gist/raw")
    monkeypatch.setenv("HEATER_RELAY_KEY", Fernet.generate_key().decode())
    resp = MagicMock()
    resp.text = "not-cipher"
    resp.raise_for_status.return_value = None
    with patch("src.token_relay._requests.get", return_value=resp):
        assert token_relay.pull_relayed_token() is False
