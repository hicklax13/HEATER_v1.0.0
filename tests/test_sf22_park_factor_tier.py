"""SF-22: _bootstrap_park_factors must persist Tier 1 data when available.

Audit finding (2026-04-18):
    Even when pybaseball returns valid park-factor data, the function falls
    through to ``source_dict = _PARK_FACTORS_EMERGENCY_2026`` because the
    overwrite ``if isinstance(data, dict)`` never fires (Tier 1 returns a
    DataFrame, not a dict). Net: the emergency constants get persisted while
    refresh_log advertises ``tier='primary'`` — telemetry lies.

Required behavior:
    - When the pybaseball fetcher returns a usable dict of park factors,
      that dict gets persisted AND refresh_log records ``tier='primary'``.
    - When the fetcher returns nothing usable, the emergency dict is
      persisted AND refresh_log records ``tier='emergency'``.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


def _get_tier(source: str) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT tier FROM refresh_log WHERE source = ?", (source,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


_PATCHED_TIER1: dict[str, float] = {
    "ARI": 1.111,
    "ATL": 1.222,
    "BAL": 0.888,
    "BOS": 1.333,
    "CHC": 0.777,
    "CWS": 1.444,
    "CIN": 1.555,
    "CLE": 0.666,
    "COL": 1.666,
    "DET": 1.777,
    "HOU": 0.555,
    "KC": 1.888,
    "LAA": 1.999,
    "LAD": 0.444,
    "MIA": 1.123,
    "MIL": 0.333,
    "MIN": 1.234,
    "NYM": 0.222,
    "NYY": 1.345,
    "ATH": 1.456,
    "PHI": 0.111,
    "PIT": 1.567,
    "SD": 0.999,
    "SF": 1.678,
    "SEA": 1.789,
    "STL": 1.890,
    "TB": 1.901,
    "TEX": 0.987,
    "TOR": 1.012,
    "WSH": 1.023,
}


class TestParkFactorsTier1Honored:
    """When the live fetcher returns a valid dict, persist that dict, mark tier='primary'."""

    def test_primary_dict_is_persisted(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):

            def fake_fetch_with_fallback(*args, **kwargs):
                return _PATCHED_TIER1, "primary"

            with patch("src.data_fetch_utils.fetch_with_fallback", side_effect=fake_fetch_with_fallback):
                from src.data_bootstrap import (
                    _PARK_FACTORS_EMERGENCY_2026,
                    BootstrapProgress,
                    _bootstrap_park_factors,
                )

                progress = BootstrapProgress()
                _bootstrap_park_factors(progress)

                conn = get_connection()
                try:
                    rows = conn.execute("SELECT team_code, factor_hitting FROM park_factors").fetchall()
                finally:
                    conn.close()

                persisted = {team: round(float(pf), 4) for team, pf in rows}

                differing = [
                    t for t in _PATCHED_TIER1 if t in persisted and persisted[t] != round(_PATCHED_TIER1[t], 4)
                ]
                assert not differing, (
                    f"Tier 1 data was IGNORED — {len(differing)} teams stored emergency values "
                    f"instead of the patched fetcher data: {differing[:5]}"
                )

                emergency_match = [
                    t
                    for t in _PATCHED_TIER1
                    if t in persisted
                    and t in _PARK_FACTORS_EMERGENCY_2026
                    and persisted[t] == round(_PARK_FACTORS_EMERGENCY_2026[t], 4)
                    and round(_PARK_FACTORS_EMERGENCY_2026[t], 4) != round(_PATCHED_TIER1[t], 4)
                ]
                assert not emergency_match, (
                    f"Persisted values match emergency dict for {len(emergency_match)} teams: {emergency_match[:5]}"
                )

    def test_refresh_log_records_primary_tier(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):

            def fake_fetch_with_fallback(*args, **kwargs):
                return _PATCHED_TIER1, "primary"

            with patch("src.data_fetch_utils.fetch_with_fallback", side_effect=fake_fetch_with_fallback):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_park_factors

                progress = BootstrapProgress()
                _bootstrap_park_factors(progress)

                tier = _get_tier("park_factors")
                assert tier == "primary", f"Expected refresh_log.tier='primary', got {tier!r}"


class TestParkFactorsEmergencyFallback:
    """When the fetcher gives nothing usable, persist emergency and mark tier='emergency'."""

    def test_emergency_data_persisted_when_fetcher_empty(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):

            def fake_fetch_with_fallback(*args, **kwargs):
                from src.data_bootstrap import _PARK_FACTORS_EMERGENCY_2026

                return _PARK_FACTORS_EMERGENCY_2026, "emergency"

            with patch("src.data_fetch_utils.fetch_with_fallback", side_effect=fake_fetch_with_fallback):
                from src.data_bootstrap import (
                    _PARK_FACTORS_EMERGENCY_2026,
                    BootstrapProgress,
                    _bootstrap_park_factors,
                )

                progress = BootstrapProgress()
                _bootstrap_park_factors(progress)

                conn = get_connection()
                try:
                    rows = conn.execute("SELECT team_code, factor_hitting FROM park_factors").fetchall()
                finally:
                    conn.close()

                persisted = {team: round(float(pf), 4) for team, pf in rows}
                for team, expected in _PARK_FACTORS_EMERGENCY_2026.items():
                    if team in persisted:
                        assert persisted[team] == round(expected, 4), (
                            f"{team} mismatch: persisted={persisted[team]} expected={round(expected, 4)}"
                        )

                tier = _get_tier("park_factors")
                assert tier == "emergency"


class TestParkFactorsDataframeFallthrough:
    """If fetch_with_fallback returns a non-dict (e.g. DataFrame leaks), use emergency, log emergency."""

    def test_non_dict_data_uses_emergency_and_logs_emergency(self, temp_db):
        import pandas as pd

        with patch("src.database.DB_PATH", temp_db):
            df = pd.DataFrame({"Team": ["BOS"], "OPS+": [105]})

            def fake_fetch_with_fallback(*args, **kwargs):
                return df, "primary"

            with patch("src.data_fetch_utils.fetch_with_fallback", side_effect=fake_fetch_with_fallback):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_park_factors

                progress = BootstrapProgress()
                _bootstrap_park_factors(progress)

                tier = _get_tier("park_factors")
                assert tier == "emergency", (
                    f"DataFrame is unusable — tier should be 'emergency' (downgraded), got {tier!r}"
                )
