"""Compact DB schema description for the system prompt.

Lets the model write accurate read-only SQL (mitigates hallucinated columns /
wrong joins). Secret/PII tables are excluded from what the model can see + query.
"""

from __future__ import annotations

# Tables the AI must never see or query (keys, auth, raw user creds).
_EXCLUDED = {
    "ai_provider_keys",
    "auth_tokens",
    "app_settings",
    "users",
    "sessions",
    "audit_log",
}


def build_schema_card() -> str:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    lines = ["Read-only SQLite schema (SELECT only). Tables and their definitions:"]
    for r in rows:
        if r["name"] in _EXCLUDED or r["name"].startswith("sqlite_"):
            continue
        sql = " ".join((r["sql"] or "").split())  # collapse whitespace
        lines.append(sql + ";")
    return "\n".join(lines)


def excluded_tables() -> set[str]:
    return set(_EXCLUDED)
