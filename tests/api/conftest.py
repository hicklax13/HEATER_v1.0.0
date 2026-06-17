import pytest
from starlette.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
