"""Smoke test: prove the per-team AppTest harness works on My Team (1 page).

This file is also the CANONICAL TEMPLATE for the Phase-1 fleet test files:
depend on the ``run_page_as_team`` + ``team_names`` fixtures from conftest.py —
no import boilerplate, no installed-`tests`-package collision.

Run SERIALLY — no -n flag.  Requires the QA users to be seeded:
    .venv\\Scripts\\python.exe scripts\\qa_seed_local.py
"""

from __future__ import annotations


def test_my_team_runs_for_admin(run_page_as_team):
    r = run_page_as_team("pages/1_My_Team.py", team_name="admin", is_admin=True)
    assert r.ran, f"page raised before render: {r.exception}"
    assert r.exception is None, f"page exception: {r.exception}"
    assert not r.errors, f"st.error on page: {r.errors}"


def test_my_team_runs_for_a_member(team_names, run_page_as_team):
    team = team_names[0]  # a real, exact team name from the DB
    r = run_page_as_team("pages/1_My_Team.py", team_name=team, is_admin=False)
    assert r.ran, f"page raised before render: {r.exception}"
    assert r.exception is None, f"page exception: {r.exception}"
    assert not r.errors, f"st.error on page: {r.errors}"
