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

import pytest

# NOTE: MULTI_USER is NOT set at import time on purpose. run_page_as_team()
# scopes the flag to each page run (set on entry, restored on exit), so the qa
# suite never leaks MULTI_USER=1 into MULTI_USER-off tests when run in the same
# process as the main suite.

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


# ── Harness fixture (the canonical way fleet tests get the page runner) ───────


@pytest.fixture(scope="session")
def run_page_as_team():
    """Return the per-team page runner from tests/qa/_harness.py.

    Fleet test files should depend on THIS fixture rather than importing
    _harness.py directly — it loads the harness by file path (sidestepping the
    broken installed ``tests`` package in .venv) once per session and hands back
    the ``run_page_as_team`` callable:

        def test_my_page(run_page_as_team, team_names):
            r = run_page_as_team("pages/1_My_Team.py", team_names[0])
            assert r.ran and r.exception is None and not r.errors
    """
    import importlib.util
    import sys
    from pathlib import Path

    harness_path = Path(__file__).resolve().parent / "_harness.py"
    if "qa_harness" in sys.modules:
        return sys.modules["qa_harness"].run_page_as_team

    spec = importlib.util.spec_from_file_location("qa_harness", harness_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qa_harness"] = mod
    spec.loader.exec_module(mod)
    return mod.run_page_as_team


# ── Generous per-test timeout for the render-heavy qa suite ──────────────────


def pytest_collection_modifyitems(config, items):
    """Give qa tests a generous timeout when they don't declare their own.

    The per-page deep modules and the ownership module render all 12 teams in a
    module-scoped fixture (the 2-page ownership fixture renders 24).  That batch
    wall-time blows the repo-global ``timeout = 120`` (pyproject.toml
    ``pytest-timeout``), which on Windows uses the ``thread`` method and kills the
    whole worker process mid-render.  Apply a 3000s ceiling to any qa item that
    doesn't set its own ``@pytest.mark.timeout`` (the smoke suite sets its own
    1200s).  3000s covers the worst realistic batch (24 renders * 180s
    per-render AppTest cap = 4320s never occurs because real renders are ~15-90s;
    3000s is comfortable headroom).  The per-render AppTest ``default_timeout=180``
    still bounds any single hung render.

    Path-filtered to ``tests/qa`` so that running the FULL repo suite (where this
    sub-package conftest is still loaded) never weakens the 120s ceiling for
    non-qa tests.
    """
    for item in items:
        path = str(getattr(item, "fspath", "")).replace("\\", "/")
        if "/tests/qa/" not in path and "/qa/" not in path:
            continue
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(3000))
