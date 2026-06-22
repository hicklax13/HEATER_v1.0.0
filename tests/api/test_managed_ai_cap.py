import api.gating as g
from api.gating import get_managed_ai_cap
from api.services.ai_allowance import managed_cap_for_tier


class _Sub:
    def __init__(self, tier):
        self.tier = tier


class _Store:
    def __init__(self, sub=None, boom=False):
        self._sub, self._boom = sub, boom

    def get(self, clerk):  # noqa: ARG002
        if self._boom:
            raise RuntimeError("db down")
        return self._sub


class _Verifier:
    def __init__(self, clerk):
        self._clerk = clerk

    def verify(self, authorization):  # noqa: ARG002
        if self._clerk is None:
            raise RuntimeError("401")
        return type("P", (), {"clerk_user_id": self._clerk})()


def test_dormant_billing_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: False)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(_Sub("pro"))) is None


def test_live_pro_returns_pro_cap(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(_Sub("pro"))) == managed_cap_for_tier("pro")


def test_live_no_subscription_returns_free_cap(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(None)) == managed_cap_for_tier("free")


def test_live_unverifiable_caller_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap(None, _Verifier(None), _Store(_Sub("pro"))) is None


def test_live_store_error_returns_none(monkeypatch):
    monkeypatch.setattr(g, "stripe_enabled", lambda: True)
    assert get_managed_ai_cap("Bearer x", _Verifier("u1"), _Store(boom=True)) is None
