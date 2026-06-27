from fastapi.testclient import TestClient

from api.main import create_app


def test_response_has_request_id_header():
    client = TestClient(create_app())
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID")  # non-empty


def test_provided_request_id_is_echoed():
    client = TestClient(create_app())
    res = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert res.headers.get("X-Request-ID") == "abc-123"
