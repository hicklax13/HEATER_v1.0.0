"""Tests for the 2026-04-17 optimizer trust audit fixes.

Covers:
  - _normalize_pitcher_name helper (daily_optimizer)
  - DCV gate: non-probable SPs must get volume_factor = 0.0
  - Matchup multiplier: opponent/venue resolution (no more opponent_team="")
  - FA streaming drop-candidate: slot-aware selection
  - _format_fetch_error helper (data_bootstrap)
  - _format_elapsed_hms helper (app)
  - SQLite WAL mode + busy_timeout (database)
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_roster_statuses(monkeypatch):
    """Prevent build_daily_dcv_table from reading real DB roster statuses.

    _load_roster_statuses() queries league_rosters. In a test environment
    the real DB may contain IL/DTD/NA statuses for the test player IDs,
    which would zero out health_factor and mask the bugs we're testing.
    """
    monkeypatch.setattr(
        "src.trade_intelligence._load_roster_statuses",
        lambda: {},
    )


# ── _normalize_pitcher_name ──────────────────────────────────────────


class TestNormalizePitcherName:
    def test_basic_lowercase_strip(self) -> None:
        from src.optimizer.daily_optimizer import _normalize_pitcher_name

        assert _normalize_pitcher_name("Chris Sale") == "chris sale"
        assert _normalize_pitcher_name("  Chris Sale  ") == "chris sale"

    def test_diacritics_stripped(self) -> None:
        from src.optimizer.daily_optimizer import _normalize_pitcher_name

        assert _normalize_pitcher_name("José Ramírez") == "jose ramirez"
        assert _normalize_pitcher_name("Andrés Giménez") == "andres gimenez"

    def test_suffix_stripped(self) -> None:
        from src.optimizer.daily_optimizer import _normalize_pitcher_name

        assert _normalize_pitcher_name("Luis Garcia Jr.") == "luis garcia"
        assert _normalize_pitcher_name("Luis Garcia Jr") == "luis garcia"
        assert _normalize_pitcher_name("Ken Griffey III") == "ken griffey"
        assert _normalize_pitcher_name("Cal Ripken Sr.") == "cal ripken"

    def test_punctuation_removed(self) -> None:
        from src.optimizer.daily_optimizer import _normalize_pitcher_name

        assert _normalize_pitcher_name("A.J. Puk") == "aj puk"
        assert _normalize_pitcher_name("J.T. Realmuto") == "jt realmuto"

    def test_empty_safe(self) -> None:
        from src.optimizer.daily_optimizer import _normalize_pitcher_name

        assert _normalize_pitcher_name("") == ""
        assert _normalize_pitcher_name(None) == ""  # type: ignore[arg-type]


# ── DCV gate: non-probable SP → volume_factor = 0.0 ──────────────────


class TestDcvNonProbableSpGate:
    """Verify that a pure SP not in today's probable-starters list gets
    volume_factor = 0.0, which zeroes their total_dcv so they cannot be
    mistakenly started. Prior bug: volume was 0.9 (ghost value) which let
    non-probable SPs land in P slots with positive DCV."""

    def _make_roster(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Tanner Bibee",
                    "positions": "SP,P",
                    "team": "CLE",
                    "is_hitter": 0,
                    "w": 14,
                    "l": 9,
                    "era": 3.50,
                    "whip": 1.15,
                    "k": 180,
                    "ip": 170,
                    "sv": 0,
                    "throws": "R",
                },
                {
                    "player_id": 2,
                    "name": "Chris Sale",
                    "positions": "SP,P",
                    "team": "ATL",
                    "is_hitter": 0,
                    "w": 12,
                    "l": 8,
                    "era": 3.55,
                    "whip": 1.05,
                    "k": 170,
                    "ip": 160,
                    "sv": 0,
                    "throws": "L",
                },
            ]
        )

    def _make_schedule(self) -> list[dict]:
        return [
            {
                "home_name": "CLEVELAND GUARDIANS",
                "away_name": "DETROIT TIGERS",
                "home_probable_pitcher": "Tanner Bibee",
                "away_probable_pitcher": "Jack Flaherty",
            },
            {
                "home_name": "ATLANTA BRAVES",
                "away_name": "NEW YORK METS",
                "home_probable_pitcher": "Spencer Schwellenbach",
                "away_probable_pitcher": "Sean Manaea",
            },
        ]

    def test_probable_sp_gets_volume_1(self) -> None:
        from src.optimizer.daily_optimizer import build_daily_dcv_table

        roster = self._make_roster()
        schedule = self._make_schedule()
        park_factors = {"CLE": 0.98, "DET": 0.96, "ATL": 1.01, "NYM": 0.95}
        dcv = build_daily_dcv_table(
            roster=roster,
            matchup=None,
            schedule_today=schedule,
            park_factors=park_factors,
        )
        bibee = dcv[dcv["player_id"] == 1].iloc[0]
        # Bibee is the probable starter — volume should be 1.0
        assert bibee["volume_factor"] == pytest.approx(1.0)

    def test_non_probable_pure_sp_gets_volume_0(self) -> None:
        from src.optimizer.daily_optimizer import build_daily_dcv_table

        roster = self._make_roster()
        schedule = self._make_schedule()
        park_factors = {"CLE": 0.98, "DET": 0.96, "ATL": 1.01, "NYM": 0.95}
        dcv = build_daily_dcv_table(
            roster=roster,
            matchup=None,
            schedule_today=schedule,
            park_factors=park_factors,
        )
        sale = dcv[dcv["player_id"] == 2].iloc[0]
        # Sale's team plays but he's NOT probable → volume must be 0
        assert sale["volume_factor"] == pytest.approx(0.0)
        # Total DCV must also be 0 (volume=0 zeroes everything)
        assert sale["total_dcv"] == pytest.approx(0.0)


# ── Matchup multiplier venue resolution ─────────────────────────────


class TestMatchupMultiplierOpponentResolution:
    """Confirm the bug is fixed: previously opponent_team="" was hardcoded,
    which made park_factor_adjustment fall back to the player's team park
    regardless of home/away. Now the call site resolves venue from schedule.
    """

    def test_home_hitter_uses_home_park(self) -> None:
        from src.optimizer.daily_optimizer import build_daily_dcv_table

        roster = pd.DataFrame(
            [
                {
                    "player_id": 10,
                    "name": "Mickey Moniak",
                    "positions": "OF,Util",
                    "team": "COL",
                    "is_hitter": 1,
                    "r": 70,
                    "hr": 20,
                    "rbi": 65,
                    "sb": 8,
                    "avg": 0.260,
                    "obp": 0.310,
                    "pa": 500,
                    "ab": 470,
                    "hits": 122,
                    "bb": 22,
                    "hbp": 3,
                    "sf": 5,
                    "bats": "L",
                    "throws": "",
                }
            ]
        )
        schedule = [
            {
                "home_name": "COLORADO ROCKIES",
                "away_name": "ARIZONA DIAMONDBACKS",
                "home_probable_pitcher": "Kyle Freeland",
                "away_probable_pitcher": "Zac Gallen",
            }
        ]
        park_factors = {"COL": 1.38, "ARI": 1.04}
        dcv = build_daily_dcv_table(
            roster=roster,
            matchup=None,
            schedule_today=schedule,
            park_factors=park_factors,
        )
        moniak = dcv[dcv["player_id"] == 10].iloc[0]
        # Moniak is HOME at COL → venue = COL → park factor ~1.38 applied
        assert moniak["matchup_mult"] > 1.15  # not neutral
        assert moniak["volume_factor"] > 0  # active hitter

    def test_away_hitter_uses_opponent_park(self) -> None:
        """COL hitter visiting Miami: venue = MIA park, not COL's 1.38."""
        from src.optimizer.daily_optimizer import build_daily_dcv_table

        roster = pd.DataFrame(
            [
                {
                    "player_id": 11,
                    "name": "Mickey Moniak",
                    "positions": "OF,Util",
                    "team": "COL",
                    "is_hitter": 1,
                    "r": 70,
                    "hr": 20,
                    "rbi": 65,
                    "sb": 8,
                    "avg": 0.260,
                    "obp": 0.310,
                    "pa": 500,
                    "ab": 470,
                    "hits": 122,
                    "bb": 22,
                    "hbp": 3,
                    "sf": 5,
                    "bats": "L",
                    "throws": "",
                }
            ]
        )
        schedule = [
            {
                "home_name": "MIAMI MARLINS",
                "away_name": "COLORADO ROCKIES",
                "home_probable_pitcher": "Sandy Alcantara",
                "away_probable_pitcher": "Kyle Freeland",
            }
        ]
        park_factors = {"COL": 1.38, "MIA": 0.88}
        dcv = build_daily_dcv_table(
            roster=roster,
            matchup=None,
            schedule_today=schedule,
            park_factors=park_factors,
        )
        moniak = dcv[dcv["player_id"] == 11].iloc[0]
        # Moniak visiting MIA → venue is MIA (0.88), NOT COL (1.38)
        # The exact multiplier also includes platoon and weather, but park
        # contribution should be < 1.0 (pitcher-friendly), not boosted.
        assert moniak["matchup_mult"] < 1.10


# ── FA streaming drop-candidate slot awareness ───────────────────────


class TestFaDropSlotAware:
    def test_parse_positions_comma(self) -> None:
        from src.optimizer.fa_recommender import _parse_positions

        assert _parse_positions("2B,OF,Util") == {"2B", "OF", "UTIL"}

    def test_parse_positions_slash(self) -> None:
        from src.optimizer.fa_recommender import _parse_positions

        assert _parse_positions("2B/OF/Util") == {"2B", "OF", "UTIL"}

    def test_parse_positions_empty(self) -> None:
        from src.optimizer.fa_recommender import _parse_positions

        assert _parse_positions("") == set()
        assert _parse_positions(None) == set()  # type: ignore[arg-type]

    def test_parse_positions_whitespace(self) -> None:
        from src.optimizer.fa_recommender import _parse_positions

        assert _parse_positions("  2B  , OF,  Util  ") == {"2B", "OF", "UTIL"}


# ── _format_fetch_error helper ───────────────────────────────────────


class TestFormatFetchError:
    def test_fangraphs_403(self) -> None:
        from src.data_bootstrap import _format_fetch_error

        exc = Exception("403 Client Error: Forbidden for url: leaders-legacy.aspx")
        out = _format_fetch_error(exc, "FanGraphs Stuff+")
        assert "Skipped" in out and "403" in out and "FanGraphs Stuff+" in out

    def test_leaders_legacy_trigger(self) -> None:
        from src.data_bootstrap import _format_fetch_error

        exc = Exception("Error accessing 'https://www.fangraphs.com/leaders-legacy.aspx'")
        out = _format_fetch_error(exc, "FanGraphs")
        assert "Skipped" in out

    def test_429(self) -> None:
        from src.data_bootstrap import _format_fetch_error

        exc = Exception("429 Too Many Requests")
        out = _format_fetch_error(exc, "MLB Stats API")
        assert "Skipped" in out and "429" in out

    def test_timeout(self) -> None:
        from src.data_bootstrap import _format_fetch_error

        exc = Exception("Request timed out after 30s")
        out = _format_fetch_error(exc, "FanGraphs")
        assert "Skipped" in out and "timed out" in out.lower()

    def test_generic_error_passthrough(self) -> None:
        from src.data_bootstrap import _format_fetch_error

        exc = KeyError("some_column")
        out = _format_fetch_error(exc, "FanGraphs")
        assert out.startswith("Error")


# ── _format_elapsed_hms helper ───────────────────────────────────────


class TestFormatElapsedHms:
    def _import_fn(self):
        # app.py lives at repo root; ensure it's importable.
        import importlib

        return importlib.import_module("app")._format_elapsed_hms

    def test_zero(self) -> None:
        fn = self._import_fn()
        assert fn(0.0) == "00:00:00"

    def test_minutes(self) -> None:
        fn = self._import_fn()
        assert fn(137.0) == "00:02:17"

    def test_hours(self) -> None:
        fn = self._import_fn()
        assert fn(3723.5) == "01:02:03"

    def test_negative_clamped(self) -> None:
        fn = self._import_fn()
        assert fn(-5.0) == "00:00:00"

    def test_non_numeric_safe(self) -> None:
        fn = self._import_fn()
        assert fn("not a number") == "00:00:00"  # type: ignore[arg-type]


# ── SQLite WAL + busy_timeout ────────────────────────────────────────


class TestGetConnectionPragmas:
    """Verify WAL mode + busy_timeout fire on every new connection.
    Root-cause fix for 6 bootstrap phases that failed with
    'database is locked' during parallel ThreadPoolExecutor writes."""

    def test_wal_mode_enabled(self, tmp_path, monkeypatch) -> None:
        # Redirect DB_PATH to a temp file so we don't touch the real DB.
        import src.database as db_mod

        tmp_db = tmp_path / "test.db"
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_db)
        conn = db_mod.get_connection()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            # WAL is persistent so both direct pragma call and re-open confirm it
            assert str(mode).lower() == "wal"
        finally:
            conn.close()

    def test_busy_timeout_set(self, tmp_path, monkeypatch) -> None:
        import src.database as db_mod

        tmp_db = tmp_path / "test.db"
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_db)
        conn = db_mod.get_connection()
        try:
            timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert int(timeout_ms) >= 30000
        finally:
            conn.close()
