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
    fresh = {
        "access_token": "NEW",
        "refresh_token": "rt",
        "consumer_key": "ck",
        "consumer_secret": "cs",
        "token_time": 1.0,
    }
    with (
        patch.object(relay, "refresh_yahoo_token", return_value=fresh),
        patch.object(relay, "_write_token_file", return_value=True),
    ):
        assert relay.main() == 3


def test_main_uploads_on_success(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    tok = tmp_path / "yahoo_token.json"
    tok.write_text('{"consumer_key":"ck","consumer_secret":"cs","refresh_token":"rt"}')
    monkeypatch.setattr(relay, "TOKEN_FILE", tok)
    monkeypatch.setenv("HEATER_RELAY_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("HEATER_GIST_ID", "gid")
    monkeypatch.setenv("HEATER_GIST_PAT", "pat")
    fresh = {
        "access_token": "NEW",
        "refresh_token": "rt",
        "consumer_key": "ck",
        "consumer_secret": "cs",
        "token_time": 1.0,
    }
    resp = MagicMock()
    resp.status_code = 200
    with (
        patch.object(relay, "refresh_yahoo_token", return_value=fresh),
        patch.object(relay, "_write_token_file", return_value=True),
        patch.object(relay.requests, "patch", return_value=resp) as p,
    ):
        assert relay.main() == 0
    p.assert_called_once()
