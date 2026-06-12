"""Per-user daily cost cap enforcement + ledger accounting."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_usage_ledger WHERE user_id = 99")
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_daily_cap_usd'")
        conn.commit()
    finally:
        conn.close()


def test_under_cap_allows():
    from src.ai.budget import is_over_cap

    assert is_over_cap(99) is False


def test_record_then_over_cap():
    from src.ai.budget import is_over_cap, record_usage, set_daily_cap

    set_daily_cap(0.50, admin_id=1)
    record_usage(99, tokens_in=1000, tokens_out=1000, cost_usd=0.40)
    assert is_over_cap(99) is False
    record_usage(99, tokens_in=1000, tokens_out=1000, cost_usd=0.20)  # total 0.60 > 0.50
    assert is_over_cap(99) is True


def test_spent_today_sums():
    from src.ai.budget import record_usage, spent_today

    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=0.10)
    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=0.05)
    assert spent_today(99) == pytest.approx(0.15)


def test_own_key_user_exempt():
    from src.ai.budget import is_over_cap, record_usage, set_daily_cap

    set_daily_cap(0.01, admin_id=1)
    record_usage(99, tokens_in=10, tokens_out=10, cost_usd=5.0)
    # on their own key, the shared-key cap does not apply
    assert is_over_cap(99, on_own_key=True) is False
    assert is_over_cap(99, on_own_key=False) is True
