"""Compact DB schema description for the system prompt.

Lets the model write accurate read-only SQL (mitigates hallucinated columns /
wrong joins). Secret/PII tables are excluded from what the model can see + query.
"""

from __future__ import annotations

# Tables the AI must never see or query. Two groups:
#   - secrets / PII / auth: keys, tokens, settings, user creds, sessions, audit.
#   - the chat's own meta-tables: conversations/messages/usage/queue. Excluding
#     these stops a user's AI from reading OTHER users' chat history or usage via
#     the query_data SQL tool (it's keyed to baseball data, not the chat plumbing).
_EXCLUDED = {
    "ai_provider_keys",
    "ai_conversations",
    "ai_messages",
    "ai_usage_ledger",
    "forced_refresh_queue",
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
