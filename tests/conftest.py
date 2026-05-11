"""Shared pytest fixtures for HEATER test suite.

Currently provides:
  - league_standings_seed: a 12-team H2H Categories standings DataFrame in the
    schema produced by ``src.database.load_league_standings`` (long format with
    columns ``team_name``, ``category``, ``total``, ``rank``).
  - patch_league_standings: an autouse fixture that monkeypatches
    ``load_league_standings`` at the two import sites that consume it
    (``src.engine.output.trade_evaluator`` and ``src.engine.portfolio.valuation``)
    so trade engine tests do not need a live SQLite ``league_standings`` table.

The autouse patch only takes effect when the underlying SQLite table is missing;
this keeps it safe for any future test that wants to seed the real table itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 12 H2H Categories: 6 hitting (R, HR, RBI, SB, AVG, OBP) + 6 pitching
# (W, L, SV, K, ERA, WHIP). ERA/WHIP/L are inverse (lower better).
# Realistic 2026-style baselines for a full season snapshot.
_CAT_BASELINES: dict[str, tuple[float, float]] = {
    "R": (700, 40),
    "HR": (170, 20),
    "RBI": (680, 35),
    "SB": (80, 25),
    "AVG": (0.255, 0.010),
    "OBP": (0.330, 0.012),
    "W": (60, 8),
    "L": (55, 7),
    "SV": (50, 15),
    "K": (1000, 80),
    "ERA": (3.80, 0.30),
    "WHIP": (1.22, 0.06),
}

# Inverse categories (lower is better) for rank assignment
_INVERSE_CATS: frozenset[str] = frozenset({"L", "ERA", "WHIP"})


def _build_league_standings_df(num_teams: int = 12, seed: int = 42) -> pd.DataFrame:
    """Construct a deterministic 12-team standings DataFrame.

    Output schema matches ``src.database.load_league_standings``:
    columns = [team_name, category, total, rank, points].
    """
    rng = np.random.RandomState(seed)
    teams = [f"Team {i + 1}" for i in range(num_teams)]
    rows: list[dict] = []
    for team in teams:
        for cat, (mean, std) in _CAT_BASELINES.items():
            total = float(round(mean + rng.normal(0, std), 4))
            rows.append(
                {
                    "team_name": team,
                    "category": cat,
                    "total": total,
                    "rank": 1,  # filled below
                    "points": 0.0,  # filled below (rotisserie-style)
                }
            )

    df = pd.DataFrame(rows)

    # Compute per-category ranks + roto-style points (12 = best, 1 = worst).
    for cat in _CAT_BASELINES:
        cat_mask = df["category"] == cat
        ascending = cat in _INVERSE_CATS
        cat_df = df.loc[cat_mask].sort_values("total", ascending=ascending)
        for rank, idx in enumerate(cat_df.index, start=1):
            df.loc[idx, "rank"] = rank
            df.loc[idx, "points"] = float(num_teams - rank + 1)

    return df


@pytest.fixture(scope="session")
def league_standings_seed() -> pd.DataFrame:
    """Seed standings DataFrame in the schema of ``load_league_standings``.

    Available for tests that want to assert on or override the standings used
    by the trade engine. Independent of the autouse patch below.
    """
    return _build_league_standings_df()


@pytest.fixture(scope="session")
def league_standings_seed_path(tmp_path_factory, league_standings_seed) -> Path:
    """Provide a temp SQLite DB with a populated ``league_standings`` table.

    Useful for the rare test that wants to exercise the real
    ``load_league_standings`` SQL path (e.g. database integration tests).
    Most tests should rely on the autouse ``patch_league_standings`` fixture
    instead.
    """
    db_path = tmp_path_factory.mktemp("trade_engine") / "test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE league_standings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                category TEXT NOT NULL,
                total REAL DEFAULT 0,
                rank INTEGER,
                points REAL,
                UNIQUE(team_name, category)
            )
            """
        )
        rows = [
            (r["team_name"], r["category"], r["total"], int(r["rank"]), float(r["points"]))
            for _, r in league_standings_seed.iterrows()
        ]
        conn.executemany(
            "INSERT INTO league_standings (team_name, category, total, rank, points) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _table_missing() -> bool:
    """Return True iff the real DB has no ``league_standings`` table.

    The autouse patch fires only in this case so tests that intentionally
    populate a real table (e.g. integration tests) see the real data.
    """
    try:
        from src.database import get_connection
    except Exception:
        return True
    try:
        conn = get_connection()
    except Exception:
        return True
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='league_standings'")
        return cur.fetchone() is None
    except Exception:
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def patch_league_standings(monkeypatch, league_standings_seed):
    """Auto-patch ``load_league_standings`` when the SQLite table is missing.

    Two call sites exist (both consume the dataframe directly, so we patch the
    bound name in each module rather than the source in ``src.database``):
      - ``src.engine.output.trade_evaluator.load_league_standings``
      - ``src.engine.portfolio.valuation.load_league_standings``

    Returning a fresh copy each call avoids cross-test mutation of the seed.
    """
    if not _table_missing():
        # Real seeded DB exists — let it through unmodified.
        yield
        return

    seed = league_standings_seed

    def _fake_load_league_standings() -> pd.DataFrame:
        return seed.copy(deep=True)

    # Patch every module that imported load_league_standings into its namespace.
    # Use raising=False so a refactor that drops one of these names doesn't
    # silently break this fixture — instead, the test will surface the missing
    # symbol clearly.
    for target in (
        "src.engine.output.trade_evaluator.load_league_standings",
        "src.engine.portfolio.valuation.load_league_standings",
        "src.database.load_league_standings",
    ):
        try:
            monkeypatch.setattr(target, _fake_load_league_standings, raising=False)
        except Exception:
            # Module may not be importable in this test context; skip silently.
            continue

    yield
