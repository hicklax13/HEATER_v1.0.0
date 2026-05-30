"""Admin audit log — records privileged actions. MULTI_USER-gated."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.auth import multi_user_enabled


def log_action(admin_id: int, action: str, target: str | None = None, detail: dict | None = None) -> None:
    if not multi_user_enabled():
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO audit_log (admin_id, action, target, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                admin_id,
                action,
                target,
                json.dumps(detail) if detail is not None else None,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_audit(limit: int = 200, action: str | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        sql = "SELECT a.*, u.username AS admin_username FROM audit_log a LEFT JOIN users u ON u.user_id = a.admin_id"
        params: list = []
        if action is not None:
            sql += " WHERE a.action = ?"
            params.append(action)
        sql += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
