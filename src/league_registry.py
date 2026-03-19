"""Multi-league CRUD, context switching, data scoping."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from src.database import get_connection


@dataclass
class LeagueInfo:
    league_id: str
    platform: str
    league_name: str
    num_teams: int
    scoring_format: str
    yahoo_league_id: str | None
    created_at: str
    is_active: bool


def register_league(
    platform: str = "manual",
    league_name: str = "My League",
    num_teams: int = 12,
    scoring_format: str = "h2h_categories",
    yahoo_league_id: str | None = None,
) -> str:
    """Register a new league, returns league_id (UUID). First league becomes active."""
    league_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        # Check if any leagues exist
        count = conn.execute("SELECT COUNT(*) FROM leagues").fetchone()[0]
        is_active = 1 if count == 0 else 0

        conn.execute(
            """INSERT INTO leagues (league_id, platform, league_name, num_teams,
               scoring_format, yahoo_league_id, created_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                league_id,
                platform,
                league_name,
                num_teams,
                scoring_format,
                yahoo_league_id,
                datetime.now(UTC).isoformat(),
                is_active,
            ),
        )
        conn.commit()
        return league_id
    finally:
        conn.close()


def get_league(league_id: str) -> LeagueInfo | None:
    """Get a league by its ID. Returns None if not found."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM leagues WHERE league_id = ?", (league_id,)).fetchone()
        if not row:
            return None
        return LeagueInfo(
            league_id=row[0],
            platform=row[1],
            league_name=row[2],
            num_teams=row[3],
            scoring_format=row[4],
            yahoo_league_id=row[5],
            created_at=row[6],
            is_active=bool(row[7]),
        )
    finally:
        conn.close()


def list_leagues() -> list[LeagueInfo]:
    """List all registered leagues ordered by creation time."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM leagues ORDER BY created_at").fetchall()
        return [
            LeagueInfo(
                league_id=r[0],
                platform=r[1],
                league_name=r[2],
                num_teams=r[3],
                scoring_format=r[4],
                yahoo_league_id=r[5],
                created_at=r[6],
                is_active=bool(r[7]),
            )
            for r in rows
        ]
    finally:
        conn.close()


def set_active_league(league_id: str) -> bool:
    """Set the given league as active (deactivates all others). Returns False if not found."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT league_id FROM leagues WHERE league_id = ?", (league_id,)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE leagues SET is_active = 0")
        conn.execute("UPDATE leagues SET is_active = 1 WHERE league_id = ?", (league_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def get_active_league_id() -> str:
    """Return the active league_id, or 'default' if none exists."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT league_id FROM leagues WHERE is_active = 1").fetchone()
        return row[0] if row else "default"
    finally:
        conn.close()


def delete_league(league_id: str) -> bool:
    """Delete a league by ID. Returns True if deleted, False if not found."""
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM leagues WHERE league_id = ?", (league_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
