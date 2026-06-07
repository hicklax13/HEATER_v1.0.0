"""BR-9 (2026-06-07): FA Ownership Heat Index must reflect percent_owned.

The Heat Index (Hot/Warm/Breakout counts) and the per-row HEAT column on
pages/14_Free_Agents.py showed all zeros / blank because the heat formula was
momentum-only (delta_7d * 20 + recent_adds * 2). Both inputs stay ~0 until a
week of ownership history accrues, so every FA scored heat=0 even though
percent_owned IS populated (Yahoo FA fetch → ownership_trends, also visible as
"% Ros" in Player Databank).

These tests pin: heat is driven by percent_owned as the base signal, with
delta_7d / recent_adds as momentum boosts. The pure formula
``_compute_heat_score`` is extracted from the page source (the page module runs
top-level Streamlit code on import, so we exec only the helper definition).
"""

from __future__ import annotations

import ast
from pathlib import Path

_FA_PAGE = Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"


def _load_compute_heat_score():
    """Extract and exec just the _compute_heat_score function from the page."""
    src = _FA_PAGE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_heat_score":
            ns: dict = {}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(_FA_PAGE), "exec"), ns)  # noqa: S102
            return ns["_compute_heat_score"]
    raise AssertionError("_compute_heat_score not found in pages/14_Free_Agents.py")


def test_heat_reflects_ownership_with_no_momentum():
    """With delta_7d=0 and recent_adds=0 (the early-season reality), heat must
    still scale with percent_owned — not collapse to 0."""
    compute = _load_compute_heat_score()
    # 70% owned, no momentum → Hot (>=7).
    assert compute(70.0, 0.0, 0) >= 7, "70%-owned FA should be Hot even with no momentum"
    # 40% owned → Warm band (4-6).
    assert 4 <= compute(40.0, 0.0, 0) <= 6, "40%-owned FA should be Warm"
    # 0% owned, no momentum → cold (0).
    assert compute(0.0, 0.0, 0) == 0


def test_heat_momentum_boosts_low_owned_breakout():
    """A low-owned FA with positive ownership momentum should climb above a
    same-ownership FA with no momentum (breakout semantics)."""
    compute = _load_compute_heat_score()
    quiet = compute(20.0, 0.0, 0)
    rising = compute(20.0, 0.20, 5)  # +20% in 7d, 5 recent adds
    assert rising > quiet, "Momentum must increase heat over a quiet same-ownership FA"


def test_heat_clamped_to_0_10():
    """Heat is always within [0, 10] regardless of extreme inputs."""
    compute = _load_compute_heat_score()
    assert compute(100.0, 1.0, 50) == 10
    assert compute(-5.0, 0.0, 0) == 0
    assert compute(None, None, None) == 0  # NaN/None-safe


def test_load_ownership_heat_uses_percent_owned(tmp_path):
    """End-to-end: seeded ownership_trends with percent_owned but delta_7d=0
    must produce non-zero heat for high-ownership FAs (the BR-9 bug repro)."""
    import sqlite3

    import src.database as db_mod

    db_path = tmp_path / "heat.db"
    original = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    try:
        db_mod.init_db()
        conn = sqlite3.connect(db_path)
        try:
            # Two FAs: one heavily owned (75%), one barely (5%); both delta_7d=0.
            conn.execute(
                "INSERT INTO ownership_trends (player_id, date, percent_owned, delta_7d) VALUES (?, ?, ?, ?)",
                (101, "2026-06-07", 75.0, 0.0),
            )
            conn.execute(
                "INSERT INTO ownership_trends (player_id, date, percent_owned, delta_7d) VALUES (?, ?, ?, ?)",
                (102, "2026-06-07", 5.0, 0.0),
            )
            conn.commit()
        finally:
            conn.close()

        # Exec the helper section of the page (defs only, up to the first
        # top-level Streamlit call) so _load_ownership_heat is importable.
        src = _FA_PAGE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        wanted = {"_compute_heat_score", "_load_ownership_heat"}
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        ns = {"pd": __import__("pandas"), "get_connection": db_mod.get_connection}
        module = ast.Module(body=defs, type_ignores=[])
        exec(compile(module, str(_FA_PAGE), "exec"), ns)  # noqa: S102
        load_heat = ns["_load_ownership_heat"]

        df = load_heat([101, 102])
        assert not df.empty, "ownership_trends seeded → heat frame must be non-empty"
        heat_by_pid = dict(zip(df["player_id"], df["heat"]))
        assert heat_by_pid[101] >= 7, f"75%-owned FA should be Hot, got heat={heat_by_pid[101]}"
        assert heat_by_pid[101] > heat_by_pid[102], "Higher ownership → higher heat"
    finally:
        db_mod.DB_PATH = original
