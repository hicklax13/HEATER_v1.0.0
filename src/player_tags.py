# src/player_tags.py
"""Player tag management — Sleeper, Target, Avoid, Breakout, Bust."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.database import get_connection

VALID_TAGS: list[str] = ["Sleeper", "Target", "Avoid", "Breakout", "Bust"]

TAG_COLORS: dict[str, str] = {
    "Sleeper": "#6c63ff",  # purple
    "Target": "#2d6a4f",  # green
    "Avoid": "#e63946",  # red
    "Breakout": "#ff6d00",  # orange
    "Bust": "#6b7280",  # gray
}


def add_tag(player_id: int, tag: str, note: str = "") -> bool:
    """Add a tag to a player. Returns True if inserted, False if duplicate or invalid."""
    if tag not in VALID_TAGS:
        return False
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO player_tags (player_id, tag, note) VALUES (?, ?, ?)",
            (player_id, tag, note),
        )
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def remove_tag(player_id: int, tag: str) -> bool:
    """Remove a tag from a player. Returns True if deleted."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM player_tags WHERE player_id = ? AND tag = ?",
            (player_id, tag),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_tags(player_id: int) -> list[dict]:
    """Get all tags for a player."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT tag, note, created_at FROM player_tags WHERE player_id = ?",
            (player_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_tagged_players(tag_filter: str | None = None) -> pd.DataFrame:
    """Get all tagged players, optionally filtered by tag type."""
    conn = get_connection()
    try:
        sql = "SELECT player_id, tag, note, created_at FROM player_tags"
        params: list = []
        if tag_filter is not None:
            sql += " WHERE tag = ?"
            params.append(tag_filter)
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def render_tag_badges_html(tags: list[dict]) -> str:
    """Render tag badges as inline HTML spans."""
    if not tags:
        return ""
    parts = []
    for t in tags:
        color = TAG_COLORS.get(t["tag"], "#6b7280")
        parts.append(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
            f"font-size:11px;font-weight:600;color:#fff;background:{color};"
            f'margin-right:4px;">{t["tag"]}</span>'
        )
    return "".join(parts)
