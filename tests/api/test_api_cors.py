"""CORS guard: the API must send CORS headers so the Next.js frontend (a
different origin) can call it from the browser. Allowed origins come from
HEATER_API_CORS_ORIGINS (comma-separated), defaulting to local Next.js dev
origins. Disallowed origins get no allow-origin header (deny-by-default)."""

from starlette.testclient import TestClient

from api.main import create_app

_DEV_ORIGIN = "http://localhost:3000"
_ENV = "HEATER_API_CORS_ORIGINS"


def test_allowed_dev_origin_is_echoed(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)  # use built-in localhost defaults
    client = TestClient(create_app())
    resp = client.get("/healthz", headers={"Origin": _DEV_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == _DEV_ORIGIN


def test_preflight_options_succeeds_for_allowed_origin(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    client = TestClient(create_app())
    resp = client.options(
        "/api/standings",
        headers={"Origin": _DEV_ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == _DEV_ORIGIN


def test_disallowed_origin_gets_no_allow_header(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    client = TestClient(create_app())
    resp = client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200  # the request itself still succeeds...
    assert "access-control-allow-origin" not in resp.headers  # ...but the browser is told no


def test_env_var_overrides_allowed_origins(monkeypatch):
    monkeypatch.setenv(_ENV, "https://app.heater.example, https://staging.heater.example")
    client = TestClient(create_app())
    resp = client.get("/healthz", headers={"Origin": "https://app.heater.example"})
    assert resp.headers.get("access-control-allow-origin") == "https://app.heater.example"
    # the default dev origin is NOT allowed once the env list is set
    resp2 = client.get("/healthz", headers={"Origin": _DEV_ORIGIN})
    assert "access-control-allow-origin" not in resp2.headers
