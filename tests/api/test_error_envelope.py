import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import HeaterError, install_error_handlers


@pytest.fixture
def app_with_routes():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom-heater")
    def boom_heater():
        raise HeaterError(
            "provider_unavailable",
            "League data could not be refreshed.",
            status_code=503,
            retryable=True,
            dependency="yahoo",
        )

    @app.get("/boom-unhandled")
    def boom_unhandled():
        raise ValueError("kaboom")

    return app


def test_heater_error_renders_envelope(app_with_routes):
    client = TestClient(app_with_routes)
    res = client.get("/boom-heater")
    assert res.status_code == 503
    body = res.json()
    assert body["error"]["code"] == "provider_unavailable"
    assert body["error"]["message"] == "League data could not be refreshed."
    assert body["error"]["retryable"] is True
    assert body["error"]["dependency"] == "yahoo"
    assert "request_id" in body["error"]


def test_unhandled_exception_is_enveloped_not_leaked(app_with_routes):
    # raise_server_exceptions=False so the handler runs instead of TestClient re-raising.
    client = TestClient(app_with_routes, raise_server_exceptions=False)
    res = client.get("/boom-unhandled")
    assert res.status_code == 500
    body = res.json()
    assert body["error"]["code"] == "internal_error"
    assert "kaboom" not in res.text  # the raw exception message is not leaked
    assert body["error"]["retryable"] is True
