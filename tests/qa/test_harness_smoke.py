"""Smoke test: prove the per-team AppTest harness works on My Team (1 page).

Run SERIALLY — no -n flag.  Requires the QA users to be seeded:
    .venv\\Scripts\\python.exe scripts\\qa_seed_local.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load _harness by file path — avoids the installed `tests` package in
# .venv/Lib/site-packages/tests/ taking priority over our local tests/ dir.
_qa_dir = Path(__file__).resolve().parent
_harness_path = _qa_dir / "_harness.py"

_spec = importlib.util.spec_from_file_location("qa_harness", _harness_path)
_harness_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules.setdefault("qa_harness", _harness_mod)
if sys.modules["qa_harness"] is _harness_mod:
    _spec.loader.exec_module(_harness_mod)  # type: ignore[union-attr]

run_page_as_team = _harness_mod.run_page_as_team


def test_my_team_runs_for_admin():
    r = run_page_as_team("pages/1_My_Team.py", team_name="admin", is_admin=True)
    assert r.ran, f"page raised before render: {r.exception}"
    assert r.exception is None, f"page exception: {r.exception}"
    assert not r.errors, f"st.error on page: {r.errors}"


def test_my_team_runs_for_a_member(team_names):
    team = team_names[0]  # a real, exact team name from the DB
    r = run_page_as_team("pages/1_My_Team.py", team_name=team, is_admin=False)
    assert r.ran, f"page raised before render: {r.exception}"
    assert r.exception is None, f"page exception: {r.exception}"
    assert not r.errors, f"st.error on page: {r.errors}"
