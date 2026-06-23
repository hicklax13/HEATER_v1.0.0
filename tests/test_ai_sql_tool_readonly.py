"""The query_data tool is SELECT-only, single-statement, capped, and read-only."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()


def test_select_allowed():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS one")
    assert out["rows"] == [{"one": 1}]


def test_rejects_insert():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("INSERT INTO players (name) VALUES ('x')")
    assert out["error"] is not None
    assert "select" in out["error"].lower()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE players SET name = 'x'",
        "DELETE FROM players",
        "DROP TABLE players",
        "ALTER TABLE players ADD COLUMN z TEXT",
        "PRAGMA table_info(players)",
        "ATTACH DATABASE 'x.db' AS y",
        "CREATE TABLE z (a int)",
    ],
)
def test_rejects_non_select(sql):
    from src.ai.sql_tool import run_read_only_sql

    assert run_read_only_sql(sql)["error"] is not None


def test_rejects_multiple_statements():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1; DROP TABLE players")
    assert out["error"] is not None


def test_select_queries_secret_table_blocked():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT * FROM ai_provider_keys")
    assert out["error"] is not None


def test_read_only_cte_select_allowed():
    # A `WITH ... SELECT` is still a read: the SELECT-only gate must PERMIT it
    # (a CTE is not a smuggled write) and return its rows. Write rejection is
    # covered by test_rejects_insert / test_rejects_non_select — this asserts
    # the gate doesn't over-block a legitimate CTE query.
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    assert out["error"] is None
    assert out["rows"] == [{"1": 1}]


def test_row_limit_enforced():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS n FROM players", max_rows=5)
    assert len(out["rows"]) <= 5


@pytest.mark.parametrize("bad", [123, ["SELECT 1"], b"SELECT 1", None, {"a": 1}])
def test_non_string_sql_returns_error_not_raise(bad):
    """Contract: a non-str query returns an error dict, never raises."""
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql(bad)
    assert out["rows"] == [] and out["error"] is not None


@pytest.mark.parametrize("cap", [-1, 0, None, 2.5, float("nan"), "lots"])
def test_bad_max_rows_does_not_raise(cap):
    """Contract: an odd max_rows is clamped/defaulted, never raises."""
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS one", max_rows=cap)
    assert isinstance(out, dict) and out["error"] is None
    assert out["rows"] == [{"one": 1}]
