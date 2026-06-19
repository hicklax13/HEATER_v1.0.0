import pytest
from fastapi import HTTPException

from api.auth import EnvTokenVerifier, Principal, _bearer

_ENV = "HEATER_API_WRITE_TOKEN"


def test_bearer_parsing():
    assert _bearer(None) is None
    assert _bearer("") is None
    assert _bearer("Bearer abc") == "abc"
    assert _bearer("bearer abc") == "abc"  # scheme is case-insensitive
    assert _bearer("Bearer   ") is None  # empty token after scheme
    assert _bearer("Token abc") is None  # wrong scheme
    assert _bearer("abc") is None  # no scheme


def test_verify_denies_by_default_when_secret_unset(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        EnvTokenVerifier().verify("Bearer anything")
    assert ei.value.status_code == 401
    assert "not configured" in ei.value.detail.lower()


def test_verify_rejects_missing_and_bad_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    for header in (None, "", "Bearer", "Bearer wrong", "Token s3cret"):
        with pytest.raises(HTTPException) as ei:
            EnvTokenVerifier().verify(header)
        assert ei.value.status_code == 401


def test_verify_accepts_matching_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    principal = EnvTokenVerifier().verify("Bearer s3cret")
    assert isinstance(principal, Principal)
    assert principal.subject == "api-token"
