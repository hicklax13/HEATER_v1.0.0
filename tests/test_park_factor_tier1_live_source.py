"""DB-C1: _bootstrap_park_factors Tier 1 must return a real {team: factor} dict.

Audit finding (2026-06-07):
    The old ``_tier1_pybaseball`` called ``team_batting(year)`` and returned a
    pandas DataFrame of team BATTING stats — not park-factor data and not a
    dict. The ``if isinstance(data, dict)`` guard therefore NEVER fired, so the
    function ALWAYS fell through to ``_PARK_FACTORS_EMERGENCY_2026`` and the
    live tier was permanent dead code.

These tests exercise the REAL Tier 1 fetcher with only the HTTP layer mocked
(``requests.get``) so the parser/normalizer runs end-to-end:

  (a) a realistic FanGraphs-style park-factor payload flows through, gets
      persisted as ``{team: factor}``, and refresh_log records tier='primary';
  (b) a 403 / unparseable HTTP response makes Tier 1 return ``{}`` so the
      waterfall falls through cleanly to the emergency dict (tier='emergency') —
      the old failure mode stays handled.
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


# A realistic FanGraphs guts.aspx?type=pf HTML fragment. The real page renders
# the park-factor grid as an HTML table; rows carry the team abbreviation and a
# "Basic" (1yr) park-factor column scaled to 100 (e.g. Coors ~112, Petco ~96).
# We include OAK to prove the abbr→ATH normalization runs, plus garbage rows the
# parser must ignore.
_FAKE_FG_PF_HTML = """
<html><body>
<table class="rgMasterTable">
<thead><tr><th>Season</th><th>Team</th><th>Basic</th><th>HR</th></tr></thead>
<tbody>
<tr><td>2026</td><td>COL</td><td>112</td><td>118</td></tr>
<tr><td>2026</td><td>BOS</td><td>104</td><td>101</td></tr>
<tr><td>2026</td><td>SD</td><td>96</td><td>92</td></tr>
<tr><td>2026</td><td>SEA</td><td>94</td><td>90</td></tr>
<tr><td>2026</td><td>OAK</td><td>103</td><td>99</td></tr>
<tr><td>2026</td><td>NYY</td><td>99</td><td>108</td></tr>
<tr><td>Average</td><td>League</td><td>100</td><td>100</td></tr>
<tr><td>2026</td><td>ZZZ</td><td>not-a-number</td><td>x</td></tr>
</tbody>
</table>
</body></html>
"""


class _FakeResp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class TestParkFactorTier1LiveSource:
    def test_realistic_payload_flows_through_as_primary(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):

            def fake_get(url, *args, **kwargs):
                return _FakeResp(200, _FAKE_FG_PF_HTML)

            with patch("requests.get", side_effect=fake_get):
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

            # tier must be primary — the real fetcher produced a usable dict.
            assert _get_tier("park_factors") == "primary"

            # Parsed values: FanGraphs "Basic" / 100. COL=1.12, SD=0.96, etc.
            assert persisted["COL"] == pytest.approx(1.12, abs=1e-4)
            assert persisted["SD"] == pytest.approx(0.96, abs=1e-4)
            assert persisted["BOS"] == pytest.approx(1.04, abs=1e-4)

            # OAK must be normalized to the project's ATH code.
            assert "ATH" in persisted, "OAK was not normalized to ATH"
            assert persisted["ATH"] == pytest.approx(1.03, abs=1e-4)
            assert "OAK" not in persisted

            # Values must NOT be the frozen emergency dict (proves Tier 1 won).
            assert persisted["COL"] != round(_PARK_FACTORS_EMERGENCY_2026["COL"], 4)

    def test_http_403_falls_through_to_emergency(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):

            def fake_get(url, *args, **kwargs):
                return _FakeResp(403, "Forbidden")

            with patch("requests.get", side_effect=fake_get):
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

            # 403 → Tier 1 returns {} → emergency dict persisted, tier='emergency'.
            assert _get_tier("park_factors") == "emergency"
            assert persisted["COL"] == round(_PARK_FACTORS_EMERGENCY_2026["COL"], 4)
