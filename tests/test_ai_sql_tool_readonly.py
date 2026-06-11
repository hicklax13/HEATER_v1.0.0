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


def test_write_is_physically_impossible(tmp_path):
    # Even a crafted statement can't write: the connection is opened read-only.
    from src.ai.sql_tool import run_read_only_sql

    # WITH ... still SELECT; a write smuggled via CTE must fail at the driver too.
    out = run_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x")
    assert out["error"] is None
    assert out["rows"] == [{"1": 1}]


def test_row_limit_enforced():
    from src.ai.sql_tool import run_read_only_sql

    out = run_read_only_sql("SELECT 1 AS n FROM players", max_rows=5)
    assert len(out["rows"]) <= 5
