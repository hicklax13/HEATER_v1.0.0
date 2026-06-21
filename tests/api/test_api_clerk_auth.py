"""ClerkVerifier unit tests — DB-free + network-free.

A local RS256 keypair is generated per test and the public key is INJECTED via
signing_key_resolver, so no JWKS network call happens (the conftest network guard
stays satisfied). Covers the full negative space: expired / bad-sig / wrong-iss /
wrong-aud / garbage / non-ASCII / missing-sub, plus the env-vs-Clerk selector."""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from api.auth import ClerkVerifier, EnvTokenVerifier, Principal, get_auth_verifier

_ISS = "https://heater.clerk.accounts.dev"


def _keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv, priv.public_key()


def _token(priv, *, iss=_ISS, sub="user_abc", aud=None, exp_delta=3600, kid="kid1"):
    now = int(time.time())
    payload = {"iss": iss, "sub": sub, "iat": now, "exp": now + exp_delta}
    if aud is not None:
        payload["aud"] = aud
    return jwt.encode(payload, priv, algorithm="RS256", headers={"kid": kid})


def _verifier(pub, *, issuer=_ISS, audience=None):
    # Inject the signing key so no JWKS network call happens (conftest guard).
    return ClerkVerifier(issuer=issuer, audience=audience, signing_key_resolver=lambda _t: pub)


def test_principal_carries_clerk_user_id():
    p = Principal(subject="user_abc", clerk_user_id="user_abc")
    assert p.clerk_user_id == "user_abc"
    assert Principal(subject="api-token").clerk_user_id is None  # optional, back-compat


def test_valid_token_returns_principal():
    priv, pub = _keypair()
    p = _verifier(pub).verify(f"Bearer {_token(priv)}")
    assert isinstance(p, Principal)
    assert p.subject == "user_abc"
    assert p.clerk_user_id == "user_abc"


def test_expired_token_denied():
    # Expire well beyond the verifier's 30s clock-skew leeway so it is truly stale.
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify(f"Bearer {_token(priv, exp_delta=-120)}")
    assert ei.value.status_code == 401


def test_bad_signature_denied():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(other_pub).verify(f"Bearer {_token(priv)}")
    assert ei.value.status_code == 401


def test_wrong_issuer_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub, issuer="https://evil.example").verify(f"Bearer {_token(priv)}")
    assert ei.value.status_code == 401


def test_wrong_audience_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub, audience="heater-api").verify(f"Bearer {_token(priv, aud='other')}")
    assert ei.value.status_code == 401


def test_correct_audience_accepted():
    priv, pub = _keypair()
    p = _verifier(pub, audience="heater-api").verify(f"Bearer {_token(priv, aud='heater-api')}")
    assert p.clerk_user_id == "user_abc"


def test_missing_or_garbage_bearer_denied():
    priv, pub = _keypair()
    v = _verifier(pub)
    for header in (None, "", "Bearer", "Bearer   ", "Token x"):
        with pytest.raises(HTTPException) as ei:
            v.verify(header)
        assert ei.value.status_code == 401


def test_non_ascii_bearer_denies_not_crashes():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify("Bearer café")  # non-ASCII → clean 401, not 500
    assert ei.value.status_code == 401


def test_missing_subject_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify(f"Bearer {_token(priv, sub='')}")
    assert ei.value.status_code == 401


def test_selector_returns_clerk_when_issuer_set(monkeypatch):
    monkeypatch.setenv("CLERK_ISSUER", _ISS)
    assert isinstance(get_auth_verifier(), ClerkVerifier)


def test_selector_returns_env_verifier_when_issuer_unset(monkeypatch):
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    assert isinstance(get_auth_verifier(), EnvTokenVerifier)
