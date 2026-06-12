"""init_db() creates the 5 additive AI-chat tables idempotently (v2 AI Phase 1)."""

from src.database import get_connection, init_db

_AI_TABLES = [
    "ai_provider_keys",
    "ai_conversations",
    "ai_messages",
    "ai_usage_ledger",
    "forced_refresh_queue",
]


def _table_names() -> set[str]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {r["name"] for r in rows}
    finally:
        conn.close()


def test_ai_tables_created():
    init_db()
    names = _table_names()
    for t in _AI_TABLES:
        assert t in names, f"{t} missing after init_db()"


def test_init_db_idempotent():
    init_db()
    init_db()  # second call must not raise
    assert set(_AI_TABLES).issubset(_table_names())


def test_tables_queryable():
    """Each AI table exists and is queryable. (Not asserting emptiness — the
    parallel suite shares one on-disk DB, so sibling test files may have rows.)"""
    init_db()
    conn = get_connection()
    try:
        for t in _AI_TABLES:
            n = conn.execute(f"SELECT COUNT(*) AS c FROM {t}").fetchone()["c"]
            assert isinstance(n, int) and n >= 0, f"{t} should be queryable"
    finally:
        conn.close()
