"""QA seeder: create one active QA user per team + qa_admin.

LOCAL-ONLY — never push this output to production. Re-runnable (idempotent).

Usage:
    .venv\\Scripts\\python.exe scripts\\qa_seed_local.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── Constants ─────────────────────────────────────────────────────────────────

QA_PASSWORD = "qa-local-only-2026"


# ── Slug helper (importable by test) ─────────────────────────────────────────


def qa_username(team_name: str) -> str:
    """Deterministic QA username from a team name.

    Lowercases alphanumerics, replaces everything else (emoji, spaces,
    punctuation) with underscores.  Example: '🏆 Team Hickey' -> 'qa___team_hickey'.
    """
    return "qa_" + "".join(c.lower() if c.isalnum() else "_" for c in team_name)


# ── Admin team resolution ─────────────────────────────────────────────────────


def _resolve_admin_team(teams: list[str]) -> str:
    """Return the exact team_name to assign qa_admin.

    Priority:
    1. league_teams WHERE is_user_team = 1 (first non-null result)
    2. Team whose name contains 'Team Hickey'
    3. First team alphabetically
    """
    from src.database import get_connection

    try:
        conn = get_connection()
        try:
            row = conn.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1 LIMIT 1").fetchone()
            if row and row[0]:
                return row[0]
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] league_teams query failed ({exc}), falling back.")

    # Fallback 1: substring match
    for t in teams:
        if "Team Hickey" in t:
            return t

    # Fallback 2: first alphabetically
    return teams[0]


# ── Core seeding logic ────────────────────────────────────────────────────────


def _upsert_qa_user(username: str, team_name: str, display_name: str) -> str:
    """Ensure a QA user is active with the correct team_name.

    Returns 'created' or 'updated' depending on whether the row was new.
    """
    from src.auth import approve_user, create_user, get_user

    existing = get_user(username)
    if existing is None:
        create_user(username, QA_PASSWORD, display_name=display_name)
        action = "created"
    else:
        action = "updated"

    # Always call approve_user to converge status + team_name regardless.
    approve_user(username, team_name, approved_by="qa_seed_local")
    return action


def _ensure_admin(username: str) -> None:
    """Promote username to is_admin=1 via direct DB update."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET is_admin=1 WHERE username = ? COLLATE NOCASE",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def seed(*, min_teams: int = 12) -> None:
    """Seed QA users.  Exits with code 1 if team count < min_teams."""
    from src.database import load_league_rosters

    df = load_league_rosters()
    teams: list[str] = sorted(set(df["team_name"].dropna()))

    if len(teams) < min_teams:
        print(
            f"ERROR: Only {len(teams)} team(s) found in league_rosters "
            f"(need >= {min_teams}).\n"
            "Run a full data refresh first (use the app's 'Refresh All Data' "
            "sidebar button or the documented bootstrap), then re-run this seeder."
        )
        sys.exit(1)

    print(f"Found {len(teams)} teams — seeding QA users...")

    admin_team = _resolve_admin_team(teams)
    print(f"  qa_admin will be assigned to: {admin_team!r}")

    created_count = 0
    updated_count = 0

    for team in teams:
        username = qa_username(team)
        display_name = f"QA: {team}"
        action = _upsert_qa_user(username, team, display_name)
        if action == "created":
            created_count += 1
        else:
            updated_count += 1
        print(f"  [{action:7s}] {username!r:40s}  team={team!r}")

    # Seed qa_admin (not tied to a slug — fixed username).
    admin_action = _upsert_qa_user("qa_admin", admin_team, "QA Admin")
    _ensure_admin("qa_admin")
    if admin_action == "created":
        created_count += 1
    else:
        updated_count += 1
    print(f"  [{admin_action:7s}] 'qa_admin'                               team={admin_team!r}  [admin=1]")

    total = len(teams) + 1  # team users + qa_admin
    print(f"\nDone. created={created_count}  already-present/updated={updated_count}  total QA users={total}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Windows consoles often default to cp1252, which cannot encode the emoji in
    # some team names (e.g. '🏆 Team Hickey'). An unguarded print would crash the
    # seeder mid-run — potentially before qa_admin is created. Replace unencodable
    # characters instead of crashing. Guarded + only on direct invocation, so
    # importing qa_username for the test never touches pytest's output capture.
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 - best-effort console hardening
        pass
    seed()
