from fastapi.testclient import TestClient

from api.main import create_app


def test_livez_is_liveness():
    client = TestClient(create_app())
    res = client.get("/livez")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"


def test_readyz_reports_dependency_checks():
    client = TestClient(create_app())
    res = client.get("/readyz")
    body = res.json()
    assert "checks" in body
    assert "database" in body["checks"]
    # Status code mirrors the dependency checks.
    if all(body["checks"].values()):
        assert res.status_code == 200
        assert body["status"] == "ready"
    else:
        assert res.status_code == 503
        assert body["status"] == "not_ready"


def test_healthz_liveness_backcompat_unchanged():
    # The existing /healthz liveness probe must keep returning {"status": "ok"}.
    client = TestClient(create_app())
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
