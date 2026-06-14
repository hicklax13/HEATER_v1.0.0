"""DB-E2: load_park_factors() reads the refreshed park_factors DB table.

The lineup optimizer previously read the module-level frozen alias
``src.data_bootstrap.PARK_FACTORS`` (== the emergency dict) directly, so a
successful Tier-1 park-factor refresh never reached lineup/matchup decisions.

``database.load_park_factors(fallback=...)`` centralizes the "DB table first,
frozen dict only when the table is empty/unavailable" lookup so optimizer
call sites route through the live table. These tests pin that contract:

  - populated table -> the DB values are returned (NOT the fallback);
  - empty/missing table -> the supplied fallback dict is returned;
  - no fallback + empty table -> {}.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZER_PAGE = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


_FROZEN_FALLBACK = {"COL": 1.134, "SD": 0.959, "BOS": 1.042}


class TestLoadParkFactors:
    def test_reads_db_table_when_populated(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):
            from src.database import load_park_factors, upsert_park_factors

            # Simulate a successful Tier-1 refresh writing non-emergency values.
            upsert_park_factors(
                [
                    {"team_code": "COL", "factor_hitting": 1.120, "factor_pitching": 1.10},
                    {"team_code": "SD", "factor_hitting": 0.960, "factor_pitching": 0.97},
                ]
            )

            result = load_park_factors(fallback=_FROZEN_FALLBACK)

            # DB values must win over the frozen fallback.
            assert result["COL"] == pytest.approx(1.120, abs=1e-4)
            assert result["SD"] == pytest.approx(0.960, abs=1e-4)
            assert result["COL"] != pytest.approx(_FROZEN_FALLBACK["COL"], abs=1e-4)

    def test_returns_fallback_when_table_empty(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):
            from src.database import load_park_factors

            result = load_park_factors(fallback=_FROZEN_FALLBACK)

            assert result == _FROZEN_FALLBACK

    def test_returns_empty_dict_when_empty_and_no_fallback(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):
            from src.database import load_park_factors

            assert load_park_factors() == {}


class TestOptimizerPageReadsDbTable:
    """The lineup optimizer must route park-factor lookups through the DB table
    reader (frozen dict only as fallback), NOT pass the static PARK_FACTORS
    alias directly to its engines.
    """

    def test_page_routes_park_factors_through_db_reader(self):
        src = OPTIMIZER_PAGE.read_text(encoding="utf-8")

        # The DB-table-backed resolver must exist and consume load_park_factors.
        assert "def _effective_park_factors(" in src
        assert "load_park_factors(" in src

        # No park_factors= kwarg may still pass the bare frozen PARK_FACTORS
        # alias. Call sites resolve via _effective_park_factors(); the threaded
        # optimize compute (R-6) receives that already-resolved value as a param
        # and forwards a lowercase `park_factors` variable, which is fine.
        for lineno, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("park_factors=") or stripped.startswith("park_factors =("):
                assert "PARK_FACTORS" not in line or "_effective_park_factors" in line, (
                    f"pages/2_Line-up_Optimizer.py:{lineno} passes the bare frozen "
                    f"PARK_FACTORS alias instead of the DB-backed resolver: {line.strip()!r}"
                )
