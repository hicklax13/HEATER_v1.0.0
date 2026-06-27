from fastapi.testclient import TestClient

from api.main import create_app


def test_security_headers_present_on_every_response():
    client = TestClient(create_app())
    res = client.get("/healthz")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "no-referrer"
    assert res.headers.get("X-Permitted-Cross-Domain-Policies") == "none"
    assert "Permissions-Policy" in res.headers


def test_existing_response_body_unchanged():
    # Hardening must not alter payloads.
    client = TestClient(create_app())
    res = client.get("/healthz")
    assert res.json() == {"status": "ok"}
