"""Launch blocker (2026-06-16 security review): the AI chat's ``query_data``
read-only SQL tool must NOT be able to read other members' per-user data.

``run_read_only_sql`` (src/ai/sql_tool.py) gates table access SOLELY by the
``_EXCLUDED`` set in ``src/ai/schema_card.py`` (a substring reject). When
Phase 7 added the per-user tables (``user_saved_views``, ``user_watchlist``)
and the pre-existing ``feedback``/``usage_events``/``page_visits`` tables were
never added to ``_EXCLUDED``, any logged-in league member could ask the AI to
``SELECT ... FROM user_saved_views`` and read every rival's saved lineups,
trade scenarios, watchlists, feedback, and activity — cross-member data
exfiltration via a one-line chat prompt.
"""

from __future__ import annotations

from src.ai.schema_card import build_schema_card, excluded_tables
from src.ai.sql_tool import run_read_only_sql
from src.database import get_connection, init_db

# Per-user tables that must be invisible + unqueryable to the AI SQL tool.
_USER_SCOPED = ["feedback", "user_saved_views", "user_watchlist", "usage_events", "page_visits"]


def test_known_per_user_tables_excluded():
    for tbl in _USER_SCOPED:
        assert tbl in excluded_tables(), (
            f"{tbl!r} is a per-user table and MUST be in schema_card._EXCLUDED "
            "so the AI query_data tool cannot read other members' rows"
        )


def test_ai_sql_rejects_queries_against_per_user_tables():
    for tbl in _USER_SCOPED:
        res = run_read_only_sql(f"SELECT * FROM {tbl}")
        assert res["error"] is not None and not res["rows"], (
            f"AI read-only SQL must refuse SELECT FROM {tbl} (cross-member leak)"
        )


def test_schema_card_does_not_advertise_per_user_tables():
    card = build_schema_card()
    for tbl in _USER_SCOPED:
        assert tbl not in card, f"schema card must not expose {tbl!r} to the model"


def test_no_user_id_scoped_table_is_queryable():
    """Forward-looking guard: ANY table carrying a ``user_id`` column is
    per-user data and must be excluded — catches future per-user tables."""
    init_db()
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        offenders = []
        for t in tables:
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')")]
            if "user_id" in cols and t not in excluded_tables():
                offenders.append(t)
    finally:
        conn.close()
    assert not offenders, (
        f"tables with a user_id column must be in schema_card._EXCLUDED (AI query_data leak risk): {offenders}"
    )
