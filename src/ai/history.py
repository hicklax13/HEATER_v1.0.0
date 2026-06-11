"""Conversation + message persistence (the history dropdown), scoped to user_id."""

from __future__ import annotations

from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_conversation(user_id: int, title: str, model: str | None = None) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO ai_conversations (user_id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, title[:120], model, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_conversations(user_id: int, limit: int = 50) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, model, updated_at FROM ai_conversations WHERE user_id = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def append_message(
    conversation_id: int,
    role: str,
    content: str,
    model: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO ai_messages (conversation_id, role, content, model, tokens_in, tokens_out, "
            "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, model, tokens_in, tokens_out, cost_usd, now),
        )
        conn.execute("UPDATE ai_conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_messages(conversation_id: int) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, model, created_at FROM ai_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def rename_conversation(conversation_id: int, title: str) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE ai_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title[:120], _now(), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM ai_conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()
