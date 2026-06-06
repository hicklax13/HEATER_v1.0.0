import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_api import refresh_yahoo_token

_TOKEN = {
    "consumer_key": "ck",
    "consumer_secret": "cs",
    "refresh_token": "rt",
    "access_token": "old",
    "token_time": 1.0,
    "token_type": "bearer",
    "guid": "g",
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
    assert out["refresh_token"] == "rt"
    assert out["token_time"] > 1.0
    assert out["consumer_key"] == "ck"


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
    assert "INVALID_CONSUMER_KEY" in caplog.text


def test_refresh_missing_fields_returns_none():
    assert refresh_yahoo_token({"consumer_key": "ck"}) is None
