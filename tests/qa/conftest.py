"""QA suite conftest: team_names fixture + skip gate for unseeded data.

The entire tests/qa/ directory requires:
  - >= 12 distinct teams in league_rosters (real production/QA data seeded)
  - qa_admin user present and active

When those conditions are not met (fresh clone, empty xdist worker DB, etc.)
every test in tests/qa/ is skipped with a clear message.

Run SERIALLY — no -n flag:
    .venv\\Scripts\\python.exe -m pytest tests/qa/ -q
"""

from __future__ import annotations

import os

import pytest

# The qa suite is auth-on by design — set MULTI_USER before any module
# that calls multi_user_enabled() at import time.
os.environ["MULTI_USER"] = "1"

# ── Skip-gate constants ───────────────────────────────────────────────────────

_SKIP_REASON = (
    "QA data not seeded or DB is an empty xdist worker copy. "
    "Run: .venv\\Scripts\\python.exe scripts\\qa_seed_local.py (serially)"
)

_MIN_TEAMS = 12


# ── Skip gate: session-scoped autouse ────────────────────────────────────────


def _check_qa_data_present() -> tuple[bool, str]:
    """Return (ready, reason).  ready=True means the suite should run."""
    try:
        from src.auth import get_user
        from src.database import load_league_rosters

        df = load_league_rosters()
        teams = sorted(set(df["team_name"].dropna()))
        if len(teams) < _MIN_TEAMS:
            return False, (f"Only {len(teams)} team(s) in league_rosters (need >= {_MIN_TEAMS}). " + _SKIP_REASON)

        admin = get_user("qa_admin")
        if admin is None:
            return False, f"qa_admin user not found. {_SKIP_REASON}"

        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"DB check failed ({exc}). {_SKIP_REASON}"


@pytest.fixture(scope="session", autouse=True)
def _qa_data_gate():
    """Skip the entire qa suite when the local DB is not seeded."""
    ready, reason = _check_qa_data_present()
    if not ready:
        pytest.skip(reason)


# ── team_names fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def team_names() -> list[str]:
    """Sorted list of the 12 exact team names from league_rosters."""
    from src.database import load_league_rosters

    df = load_league_rosters()
    return sorted(set(df["team_name"].dropna()))
