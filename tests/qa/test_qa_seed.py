"""Phase 0 gate: verify local DB has >= 12 teams and all QA users are seeded.

Must be run SERIALLY against the real data/draft_tool.db (no -n flag).
The xdist conftest redirects parallel workers to empty per-worker DBs, so
these assertions would always fail under xdist. Run with:

    .venv\\Scripts\\python.exe -m pytest tests/qa/test_qa_seed.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load qa_username from the seeder by file path — avoids needing scripts/__init__.py.
_repo_root = Path(__file__).resolve().parent.parent.parent
_seeder_path = _repo_root / "scripts" / "qa_seed_local.py"

_spec = importlib.util.spec_from_file_location("qa_seed_local", _seeder_path)
_module = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["qa_seed_local"] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]

qa_username = _module.qa_username


def test_at_least_12_distinct_teams() -> None:
    """load_league_rosters() must have >= 12 distinct non-null team_names."""
    from src.database import load_league_rosters

    df = load_league_rosters()
    distinct_teams = set(df["team_name"].dropna())
    assert len(distinct_teams) >= 12, (
        f"Expected >= 12 distinct team_names in league_rosters, got {len(distinct_teams)}: "
        f"{sorted(str(t) for t in distinct_teams)}"
    )


def test_qa_user_per_team() -> None:
    """Each team has an active QA user whose team_name matches exactly."""
    from src.auth import get_user
    from src.database import load_league_rosters

    df = load_league_rosters()
    distinct_teams = sorted(set(df["team_name"].dropna()))

    missing: list[str] = []
    wrong_status: list[str] = []
    wrong_team: list[str] = []

    for team in distinct_teams:
        username = qa_username(team)
        user = get_user(username)
        if user is None:
            missing.append(f"{username!r} (team={team!r})")
            continue
        if user["status"] != "active":
            wrong_status.append(f"{username!r}: status={user['status']!r}")
        if user["team_name"] != team:
            wrong_team.append(f"{username!r}: team_name={user['team_name']!r}, expected={team!r}")

    errors: list[str] = []
    if missing:
        errors.append(f"Missing QA users: {missing}")
    if wrong_status:
        errors.append(f"Wrong status: {wrong_status}")
    if wrong_team:
        errors.append(f"Wrong team_name: {wrong_team}")

    assert not errors, "\n".join(errors)


def test_qa_admin_exists_active_and_is_admin() -> None:
    """qa_admin must exist, be active, and have is_admin truthy."""
    from src.auth import get_user

    user = get_user("qa_admin")
    assert user is not None, "qa_admin user does not exist — run scripts/qa_seed_local.py"
    assert user["status"] == "active", f"qa_admin status={user['status']!r}, expected 'active'"
    assert user["is_admin"], f"qa_admin is_admin={user['is_admin']!r}, expected truthy"
