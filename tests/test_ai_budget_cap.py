"""is_over_cap's cap_usd override + the live-app-safe default (DB-free via
monkeypatched spent_today/daily_cap_usd)."""

import src.ai.budget as b


def test_no_cap_usd_uses_admin_cap_unchanged(monkeypatch):
    # The live Streamlit app calls is_over_cap(uid[, on_own_key]) with NO cap_usd —
    # this must stay identical to today (uses daily_cap_usd()).
    monkeypatch.setattr(b, "spent_today", lambda uid: 0.5)
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 1.0)
    assert b.is_over_cap(1) is False  # 0.5 < 1.0
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 0.4)
    assert b.is_over_cap(1) is True  # 0.5 >= 0.4


def test_cap_usd_override_ignores_admin_cap(monkeypatch):
    monkeypatch.setattr(b, "spent_today", lambda uid: 0.5)
    monkeypatch.setattr(b, "daily_cap_usd", lambda: 999.0)  # ignored when cap_usd given
    assert b.is_over_cap(1, cap_usd=0.10) is True  # 0.5 >= 0.10
    assert b.is_over_cap(1, cap_usd=2.0) is False  # 0.5 < 2.0


def test_own_key_unlimited_regardless_of_cap(monkeypatch):
    monkeypatch.setattr(b, "spent_today", lambda uid: 9999.0)
    assert b.is_over_cap(1, on_own_key=True, cap_usd=0.01) is False
