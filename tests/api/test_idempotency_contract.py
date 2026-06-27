from api.idempotency import (
    IDEMPOTENCY_HEADER,
    InMemoryIdempotencyStore,
)


def test_header_name_is_standard():
    assert IDEMPOTENCY_HEADER == "Idempotency-Key"


def test_in_memory_store_roundtrip():
    store = InMemoryIdempotencyStore()
    assert store.get("k1") is None
    store.put("k1", {"ok": True})
    assert store.get("k1") == {"ok": True}


def test_in_memory_store_does_not_overwrite_on_replay():
    store = InMemoryIdempotencyStore()
    store.put("k1", {"v": 1})
    # Replays return the first stored result; put() with the same key is a no-op.
    store.put("k1", {"v": 2})
    assert store.get("k1") == {"v": 1}
