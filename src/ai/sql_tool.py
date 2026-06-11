"""Guarded read-only SQL executor — the AI's open-ended reach into all data.

Defense in depth:
  1. Read-only connection (file:...?mode=ro) — writes are impossible at the driver.
  2. SELECT/WITH allowlist; reject DML/DDL/PRAGMA/ATTACH.
  3. Single statement only (no ';' chaining).
  4. Exclude secret tables (keys/auth/users).
  5. Row cap.
Returns {"rows": [...], "error": str|None} — never raises into the agent loop.
"""

from __future__ import annotations

import sqlite3

from src.ai.schema_card import excluded_tables
from src.database import DB_PATH

_MAX_ROWS_DEFAULT = 500


def _looks_select(sql: str) -> bool:
    s = sql.strip().lstrip("(").lstrip()
    head = s[:6].lower()
    return head.startswith("select") or s[:4].lower().startswith("with")


def run_read_only_sql(sql: str, max_rows: int = _MAX_ROWS_DEFAULT) -> dict:
    # Contract: never raise into the agent loop. The caller is an LLM (via the
    # tool layer) and may pass a non-string sql or an odd max_rows — validate
    # both up front so string ops / fetchmany can't throw.
    if not isinstance(sql, str):
        return {"rows": [], "error": "Query must be a string."}
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError, OverflowError):
        max_rows = _MAX_ROWS_DEFAULT
    if max_rows < 1:
        max_rows = _MAX_ROWS_DEFAULT
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return {"rows": [], "error": "Empty query."}
    # single statement only
    if ";" in sql:
        return {"rows": [], "error": "Only a single SELECT statement is allowed."}
    if not _looks_select(sql):
        return {"rows": [], "error": "Only read-only SELECT/WITH queries are allowed."}
    lowered = sql.lower()
    for banned in ("insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "replace"):
        # word-boundary-ish check: banned keyword as a standalone token
        if f" {banned} " in f" {lowered} " or lowered.startswith(banned + " "):
            return {"rows": [], "error": f"Statement type '{banned}' is not allowed."}
    for secret in excluded_tables():
        if secret.lower() in lowered:
            return {"rows": [], "error": f"Table '{secret}' is not queryable."}

    try:
        # as_posix() → forward slashes so the file: URI is valid on Windows too.
        ro_uri = f"file:{DB_PATH.as_posix()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(max_rows)]
            return {"rows": rows, "error": None}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"rows": [], "error": f"SQL error: {exc}"}
