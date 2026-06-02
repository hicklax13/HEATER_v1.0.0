"""Broad per-team smoke suite: render EVERY page for EVERY team.

16 test items total:
  - 13 parametrized by MEMBER_PAGES: each test loops all 12 teams as a member
    and collects failures.
  - 3 parametrized by ADMIN_PAGES: each test renders the page once as qa_admin.

Warnings (st.warning) are intentionally NOT asserted on — Streamlit deprecation
notices and fallback banners are expected noise.  Deep value-plausibility checks
come in a later task.

Run SERIALLY (no -n flag).  Requires QA data seeded:
    .venv\\Scripts\\python.exe scripts\\qa_seed_local.py
"""

from __future__ import annotations

import pytest

from src.nav import _ADMIN_PAGES, PAGE_REGISTRY

MEMBER_PAGES = [e["path"] for e in PAGE_REGISTRY]
ADMIN_PAGES = [p["path"] for p in _ADMIN_PAGES]


@pytest.mark.parametrize("page_path", MEMBER_PAGES, ids=MEMBER_PAGES)
@pytest.mark.timeout(1200)  # 12 teams × 90s harness timeout each = 1080s max
def test_member_page_loads_for_all_teams(page_path, run_page_as_team, team_names):
    """Render a member page for all 12 teams; collect and report every failure."""
    failures = []
    for team in team_names:
        r = run_page_as_team(page_path, team, is_admin=False)
        if not r.ran:
            failures.append((team, "did-not-run", r.exception))
        elif r.exception:
            failures.append((team, "st.exception", r.exception))
        elif r.errors:
            failures.append((team, "st.error", r.errors))
    assert not failures, f"{page_path} broke for {len(failures)} team(s):\n" + "\n".join(
        f"  - {t} [{kind}]: {msg}" for t, kind, msg in failures
    )


@pytest.mark.parametrize("page_path", ADMIN_PAGES, ids=ADMIN_PAGES)
def test_admin_page_loads_for_admin(page_path, run_page_as_team):
    """Render an admin page as qa_admin and flag crashes or st.error output."""
    r = run_page_as_team(page_path, team_name="admin", is_admin=True)
    problems = []
    if not r.ran:
        problems.append(("did-not-run", r.exception))
    elif r.exception:
        problems.append(("st.exception", r.exception))
    elif r.errors:
        problems.append(("st.error", r.errors))
    assert not problems, f"{page_path} (admin) broke: {problems}"
