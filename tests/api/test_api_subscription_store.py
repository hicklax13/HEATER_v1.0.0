import logging
import sqlite3

import pytest

from api.stores.subscription_store import InMemorySubscriptionStore, SqliteSubscriptionStore, Subscription


def _sub(clerk="u1", customer="cus_1", tier="pro", status="active"):
    return Subscription(
        clerk_user_id=clerk, stripe_customer_id=customer, tier=tier, status=status, updated_at="2026-06-20T00:00:00Z"
    )


def test_inmemory_upsert_get_idempotent():
    store = InMemorySubscriptionStore()
    store.upsert(_sub())
    store.upsert(_sub(status="canceled", tier="free"))  # same clerk id overwrites
    got = store.get("u1")
    assert got is not None
    assert got.status == "canceled"
    assert got.tier == "free"


def test_inmemory_get_by_customer():
    store = InMemorySubscriptionStore()
    store.upsert(_sub(clerk="u1", customer="cus_X"))
    assert store.get_by_customer("cus_X").clerk_user_id == "u1"
    assert store.get_by_customer("nope") is None


def test_inmemory_missing_get_is_none():
    assert InMemorySubscriptionStore().get("ghost") is None


def test_sqlite_upsert_get_and_by_customer(tmp_path):
    store = SqliteSubscriptionStore(db_path=str(tmp_path / "api_state.db"))
    store.upsert(_sub(clerk="u9", customer="cus_9"))
    assert store.get("u9").tier == "pro"
    assert store.get_by_customer("cus_9").clerk_user_id == "u9"
    store.upsert(_sub(clerk="u9", customer="cus_9", status="canceled", tier="free"))
    assert store.get("u9").status == "canceled"  # upsert overwrites, not duplicates


def test_sqlite_persists_across_instances(tmp_path):
    db = str(tmp_path / "api_state.db")
    SqliteSubscriptionStore(db_path=db).upsert(_sub(clerk="u2"))
    assert SqliteSubscriptionStore(db_path=db).get("u2") is not None


def test_sqlite_logs_and_propagates_on_failure(tmp_path, monkeypatch, caplog):
    store = SqliteSubscriptionStore(db_path=str(tmp_path / "x.db"))

    class _BoomConn:
        closed = False

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        def commit(self):  # pragma: no cover
            pass

        def close(self):
            self.closed = True

    boom = _BoomConn()
    monkeypatch.setattr(store, "_connect", lambda: boom)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(sqlite3.OperationalError):
            store.upsert(_sub())
    assert "upsert failed" in caplog.text
    assert boom.closed is True
