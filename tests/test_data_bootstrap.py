"""Tests for the zero-interaction data bootstrap pipeline."""

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


# ---------------------------------------------------------------------------
# Task 1: check_staleness()
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def test_no_record_is_stale(self, temp_db):
        """No refresh_log entry → stale."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import check_staleness

            assert check_staleness("season_stats", max_age_hours=24) is True

    def test_fresh_record_not_stale(self, temp_db):
        """Recent refresh → not stale."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import check_staleness, update_refresh_log

            update_refresh_log("season_stats", "success")
            assert check_staleness("season_stats", max_age_hours=24) is False

    def test_expired_record_is_stale(self, temp_db):
        """Old refresh → stale."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import check_staleness, get_connection

            conn = get_connection()
            old_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO refresh_log (source, last_refresh, status) VALUES (?, ?, ?)",
                ("season_stats", old_time, "success"),
            )
            conn.commit()
            conn.close()
            assert check_staleness("season_stats", max_age_hours=24) is True

    def test_zero_max_age_always_stale(self, temp_db):
        """max_age_hours=0 means always refresh."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import check_staleness, update_refresh_log

            update_refresh_log("test_source", "success")
            assert check_staleness("test_source", max_age_hours=0) is True


# ---------------------------------------------------------------------------
# Task 2: upsert_player_bulk()
# ---------------------------------------------------------------------------


class TestUpsertPlayerBulk:
    def test_inserts_new_players(self, temp_db):
        """Bulk upsert creates players."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_player_bulk

            players = [
                {"name": "Shohei Ohtani", "team": "LAD", "positions": "Util", "is_hitter": True},
                {"name": "Mike Trout", "team": "LAA", "positions": "OF", "is_hitter": True},
            ]
            count = upsert_player_bulk(players)
            assert count == 2
            conn = get_connection()
            rows = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            conn.close()
            assert rows == 2

    def test_updates_existing_player(self, temp_db):
        """Second upsert updates team, doesn't duplicate."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_player_bulk

            players = [{"name": "Shohei Ohtani", "team": "LAD", "positions": "Util", "is_hitter": True}]
            upsert_player_bulk(players)
            players[0]["team"] = "NYY"
            upsert_player_bulk(players)
            conn = get_connection()
            row = conn.execute("SELECT team FROM players WHERE name = 'Shohei Ohtani'").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM players WHERE name = 'Shohei Ohtani'").fetchone()[0]
            conn.close()
            assert row[0] == "NYY"
            assert count == 1

    def test_empty_list(self, temp_db):
        """Empty player list returns 0."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import upsert_player_bulk

            assert upsert_player_bulk([]) == 0


# ---------------------------------------------------------------------------
# Task 3: upsert_injury_history_bulk() and upsert_park_factors()
# ---------------------------------------------------------------------------


class TestUpsertInjuryHistoryBulk:
    def test_inserts_records(self, temp_db):
        """Bulk upsert injury history records."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_injury_history_bulk, upsert_player_bulk

            upsert_player_bulk([{"name": "Mike Trout", "team": "LAA", "positions": "OF", "is_hitter": True}])
            conn = get_connection()
            pid = conn.execute("SELECT player_id FROM players WHERE name = 'Mike Trout'").fetchone()[0]
            conn.close()
            records = [
                {"player_id": pid, "season": 2024, "games_played": 29, "games_available": 162},
                {"player_id": pid, "season": 2023, "games_played": 82, "games_available": 162},
            ]
            count = upsert_injury_history_bulk(records)
            assert count == 2

    def test_updates_on_duplicate(self, temp_db):
        """Second upsert for same player+season updates, doesn't duplicate."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_injury_history_bulk, upsert_player_bulk

            upsert_player_bulk([{"name": "Test Player", "team": "NYY", "positions": "SS", "is_hitter": True}])
            conn = get_connection()
            pid = conn.execute("SELECT player_id FROM players WHERE name = 'Test Player'").fetchone()[0]
            conn.close()
            records = [{"player_id": pid, "season": 2024, "games_played": 100, "games_available": 162}]
            upsert_injury_history_bulk(records)
            records[0]["games_played"] = 50
            upsert_injury_history_bulk(records)
            conn = get_connection()
            rows = conn.execute(
                "SELECT games_played FROM injury_history WHERE player_id = ? AND season = 2024",
                (pid,),
            ).fetchall()
            conn.close()
            assert len(rows) == 1
            assert rows[0][0] == 50


class TestUpsertParkFactors:
    def test_inserts_factors(self, temp_db):
        """Bulk upsert park factors."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_park_factors

            factors = [
                {"team_code": "COL", "factor_hitting": 1.38, "factor_pitching": 1.38},
                {"team_code": "MIA", "factor_hitting": 0.88, "factor_pitching": 0.88},
            ]
            count = upsert_park_factors(factors)
            assert count == 2
            conn = get_connection()
            row = conn.execute("SELECT factor_hitting FROM park_factors WHERE team_code = 'COL'").fetchone()
            conn.close()
            assert abs(row[0] - 1.38) < 0.01

    def test_updates_existing_factors(self, temp_db):
        """Second upsert updates factor value."""
        with patch("src.database.DB_PATH", temp_db):
            from src.database import get_connection, upsert_park_factors

            factors = [{"team_code": "COL", "factor_hitting": 1.38, "factor_pitching": 1.38}]
            upsert_park_factors(factors)
            factors[0]["factor_hitting"] = 1.40
            upsert_park_factors(factors)
            conn = get_connection()
            row = conn.execute("SELECT factor_hitting FROM park_factors WHERE team_code = 'COL'").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM park_factors WHERE team_code = 'COL'").fetchone()[0]
            conn.close()
            assert abs(row[0] - 1.40) < 0.01
            assert count == 1


# ---------------------------------------------------------------------------
# Task 4: fetch_all_mlb_players()
# ---------------------------------------------------------------------------


class TestFetchAllMlbPlayers:
    def test_returns_correct_structure(self):
        """fetch_all_mlb_players returns DataFrame with required columns."""
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = {
                "people": [
                    {
                        "id": 660271,
                        "fullName": "Shohei Ohtani",
                        "currentTeam": {"abbreviation": "LAD"},
                        "primaryPosition": {"abbreviation": "DH", "type": "Hitter"},
                        "active": True,
                    },
                    {
                        "id": 477132,
                        "fullName": "Gerrit Cole",
                        "currentTeam": {"abbreviation": "NYY"},
                        "primaryPosition": {"abbreviation": "P", "type": "Pitcher"},
                        "active": True,
                    },
                ]
            }
            from src.live_stats import fetch_all_mlb_players

            df = fetch_all_mlb_players(season=2026)
            assert len(df) == 2
            assert set(df.columns) >= {"mlb_id", "name", "team", "positions", "is_hitter"}
            ohtani = df[df["name"] == "Shohei Ohtani"].iloc[0]
            assert ohtani["is_hitter"] is True or ohtani["is_hitter"] == 1
            cole = df[df["name"] == "Gerrit Cole"].iloc[0]
            assert cole["is_hitter"] is False or cole["is_hitter"] == 0

    def test_empty_api_response(self):
        """Empty API response returns empty DataFrame."""
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = {"people": []}
            from src.live_stats import fetch_all_mlb_players

            df = fetch_all_mlb_players(season=2026)
            assert len(df) == 0

    def test_filters_inactive_players(self):
        """Inactive players are excluded."""
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = {
                "people": [
                    {
                        "id": 1,
                        "fullName": "Active Player",
                        "currentTeam": {"abbreviation": "NYY"},
                        "primaryPosition": {"abbreviation": "SS", "type": "Hitter"},
                        "active": True,
                    },
                    {
                        "id": 2,
                        "fullName": "Inactive Player",
                        "currentTeam": {"abbreviation": "BOS"},
                        "primaryPosition": {"abbreviation": "1B", "type": "Hitter"},
                        "active": False,
                    },
                ]
            }
            from src.live_stats import fetch_all_mlb_players

            df = fetch_all_mlb_players(season=2026)
            assert len(df) == 1
            assert df.iloc[0]["name"] == "Active Player"


# ---------------------------------------------------------------------------
# Task 5: fetch_historical_stats()
# ---------------------------------------------------------------------------


class TestFetchHistoricalStats:
    def test_multiple_seasons(self):
        """fetch_historical_stats returns data for multiple seasons."""
        with patch("src.live_stats.statsapi") as mock_api:

            def mock_get(endpoint, params=None, **kwargs):
                if endpoint == "teams":
                    return {"teams": [{"id": 147}]}
                if endpoint == "team_roster":
                    return {
                        "roster": [
                            {
                                "person": {
                                    "fullName": "Test Player",
                                    "currentTeam": {"abbreviation": "NYY"},
                                    "stats": [
                                        {
                                            "group": {"displayName": "hitting"},
                                            "splits": [
                                                {
                                                    "stat": {
                                                        "plateAppearances": 500,
                                                        "atBats": 450,
                                                        "hits": 120,
                                                        "runs": 70,
                                                        "homeRuns": 25,
                                                        "rbi": 80,
                                                        "stolenBases": 10,
                                                        "avg": ".267",
                                                        "gamesPlayed": 140,
                                                    }
                                                }
                                            ],
                                        }
                                    ],
                                },
                                "position": {"type": "Outfielder"},
                            }
                        ]
                    }
                return {}

            mock_api.get.side_effect = mock_get
            from src.live_stats import fetch_historical_stats

            results = fetch_historical_stats(seasons=[2023, 2024])
            assert len(results) == 2
            assert 2023 in results and 2024 in results

    def test_defaults_to_one_year(self):
        """Default seasons are [2025] (2024 excluded per historical filter)."""
        with patch("src.live_stats.statsapi") as mock_api:

            def mock_get(endpoint, params=None, **kwargs):
                if endpoint == "teams":
                    return {"teams": [{"id": 147}]}
                if endpoint == "team_roster":
                    return {
                        "roster": [
                            {
                                "person": {
                                    "fullName": "Test",
                                    "currentTeam": {"abbreviation": "NYY"},
                                    "stats": [
                                        {
                                            "group": {"displayName": "hitting"},
                                            "splits": [
                                                {
                                                    "stat": {
                                                        "plateAppearances": 100,
                                                        "atBats": 90,
                                                        "hits": 25,
                                                        "runs": 10,
                                                        "homeRuns": 5,
                                                        "rbi": 15,
                                                        "stolenBases": 2,
                                                        "avg": ".278",
                                                        "gamesPlayed": 50,
                                                    }
                                                }
                                            ],
                                        }
                                    ],
                                },
                                "position": {"type": "Outfielder"},
                            }
                        ]
                    }
                return {}

            mock_api.get.side_effect = mock_get
            from src.live_stats import fetch_historical_stats

            results = fetch_historical_stats()
            assert len(results) == 1


# ---------------------------------------------------------------------------
# Task 6: fetch_injury_data_bulk()
# ---------------------------------------------------------------------------


class TestFetchInjuryDataBulk:
    def test_converts_historical_to_records(self):
        """Converts historical stats into injury_history records."""
        from src.live_stats import fetch_injury_data_bulk

        historical = {
            2023: pd.DataFrame(
                [
                    {"player_name": "Mike Trout", "team": "LAA", "games_played": 82, "is_hitter": True},
                ]
            ),
            2024: pd.DataFrame(
                [
                    {"player_name": "Mike Trout", "team": "LAA", "games_played": 29, "is_hitter": True},
                ]
            ),
        }
        records = fetch_injury_data_bulk(historical)
        assert len(records) == 2
        assert all("season" in r and "games_played" in r for r in records)
        assert records[0]["games_available"] == 162

    def test_skips_zero_games(self):
        """Players with 0 games are excluded."""
        from src.live_stats import fetch_injury_data_bulk

        historical = {
            2023: pd.DataFrame(
                [
                    {"player_name": "No Games", "team": "NYY", "games_played": 0, "is_hitter": True},
                ]
            ),
        }
        records = fetch_injury_data_bulk(historical)
        assert len(records) == 0

    def test_empty_input(self):
        """Empty historical dict returns empty list."""
        from src.live_stats import fetch_injury_data_bulk

        records = fetch_injury_data_bulk({})
        assert records == []


# ---------------------------------------------------------------------------
# Task 7: StalenessConfig + PARK_FACTORS
# ---------------------------------------------------------------------------


class TestStalenessConfig:
    def test_defaults(self):
        """StalenessConfig has sensible defaults."""
        from src.data_bootstrap import StalenessConfig

        cfg = StalenessConfig()
        assert cfg.players_hours == 168
        assert cfg.live_stats_hours == 1
        assert cfg.projections_hours == 24
        assert cfg.historical_hours == 720
        assert cfg.park_factors_hours == 720
        assert cfg.yahoo_hours == 0.5
        assert cfg.game_day_hours == 2
        assert cfg.team_strength_hours == 24


class TestParkFactorsConstant:
    def test_has_30_teams(self):
        """PARK_FACTORS has all 30 MLB teams."""
        from src.data_bootstrap import PARK_FACTORS

        assert len(PARK_FACTORS) == 30

    def test_coors_field_hitter_friendly(self):
        """Coors Field is hitter-friendly."""
        from src.data_bootstrap import PARK_FACTORS

        assert PARK_FACTORS["COL"] > 1.2

    def test_miami_pitcher_friendly(self):
        """loanDepot park is pitcher-friendly."""
        from src.data_bootstrap import PARK_FACTORS

        assert PARK_FACTORS["MIA"] < 0.95


# ---------------------------------------------------------------------------
# Task 8: Bootstrap sub-functions
# ---------------------------------------------------------------------------


class TestBootstrapPlayers:
    def test_populates_players_table(self, temp_db):
        """_bootstrap_players populates players table."""
        with patch("src.database.DB_PATH", temp_db):
            with patch("src.live_stats.statsapi") as mock_api:
                mock_api.get.return_value = {
                    "people": [
                        {
                            "id": 1,
                            "fullName": "Test Hitter",
                            "active": True,
                            "currentTeam": {"abbreviation": "NYY"},
                            "primaryPosition": {"abbreviation": "SS", "type": "Hitter"},
                        },
                        {
                            "id": 2,
                            "fullName": "Test Pitcher",
                            "active": True,
                            "currentTeam": {"abbreviation": "BOS"},
                            "primaryPosition": {"abbreviation": "P", "type": "Pitcher"},
                        },
                    ]
                }
                from src.data_bootstrap import BootstrapProgress, _bootstrap_players

                progress = BootstrapProgress()
                result = _bootstrap_players(progress)
                assert "2" in result or "Saved" in result


class TestBootstrapParkFactors:
    def test_saves_all_teams(self, temp_db):
        """_bootstrap_park_factors saves all 30 teams."""
        with patch("src.database.DB_PATH", temp_db):
            from src.data_bootstrap import BootstrapProgress, _bootstrap_park_factors

            progress = BootstrapProgress()
            result = _bootstrap_park_factors(progress)
            assert "30" in result


class TestBootstrapProjections:
    def test_delegates_to_refresh_if_stale(self, temp_db):
        """_bootstrap_projections calls refresh_if_stale."""
        with patch("src.database.DB_PATH", temp_db):
            with patch("src.data_pipeline.refresh_if_stale", return_value=True) as mock_refresh:
                from src.data_bootstrap import BootstrapProgress, _bootstrap_projections

                progress = BootstrapProgress()
                _bootstrap_projections(progress)
                mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# Task 9: bootstrap_all_data() orchestrator
# ---------------------------------------------------------------------------


class TestBootstrapAllData:
    def test_skips_fresh_sources(self, temp_db):
        """bootstrap_all_data skips fresh sources."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.data_bootstrap._bootstrap_players") as mp,
                patch("src.data_bootstrap._bootstrap_park_factors") as mpf,
                patch("src.data_bootstrap._bootstrap_projections") as mpr,
                patch("src.data_bootstrap._bootstrap_live_stats") as ml,
                patch("src.data_bootstrap._bootstrap_historical") as mh,
                patch("src.data_bootstrap._bootstrap_injury_data") as mi,
                patch("src.data_bootstrap._bootstrap_yahoo") as my,
                patch("src.data_bootstrap._bootstrap_extended_roster") as mer,
                patch("src.database.check_staleness", return_value=False),
            ):
                from src.data_bootstrap import bootstrap_all_data

                results = bootstrap_all_data()
                # When nothing is stale, bootstrap functions should NOT be called
                mp.assert_not_called()
                mpf.assert_not_called()
                mer.assert_not_called()
                assert isinstance(results, dict)
                assert results["players"] == "Fresh"

    def test_force_refreshes_all(self, temp_db):
        """force=True calls all bootstrap functions."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.data_bootstrap._bootstrap_players", return_value="ok") as mp,
                patch("src.data_bootstrap._bootstrap_park_factors", return_value="ok") as mpf,
                patch("src.data_bootstrap._bootstrap_projections", return_value="ok") as mpr,
                patch("src.data_bootstrap._bootstrap_live_stats", return_value="ok") as ml,
                patch("src.data_bootstrap._bootstrap_historical", return_value="ok") as mh,
                patch("src.data_bootstrap._bootstrap_injury_data", return_value="ok") as mi,
                patch("src.data_bootstrap._bootstrap_yahoo", return_value="ok") as my,
                patch("src.data_bootstrap._bootstrap_extended_roster", return_value="ok") as mer,
            ):
                from src.data_bootstrap import bootstrap_all_data

                results = bootstrap_all_data(force=True)
                mp.assert_called_once()
                mpf.assert_called_once()
                mpr.assert_called_once()
                ml.assert_called_once()
                mer.assert_called_once()  # 2026-04-17 reorder: now runs at Phase 3b
                assert results["players"] == "ok"

    def test_progress_callback(self, temp_db):
        """on_progress callback is invoked."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.data_bootstrap._bootstrap_players", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_park_factors", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_projections", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_live_stats", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_historical", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_injury_data", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_yahoo", return_value="ok"),
                patch("src.data_bootstrap._bootstrap_extended_roster", return_value="ok"),
            ):
                from src.data_bootstrap import bootstrap_all_data

                progress_calls = []
                results = bootstrap_all_data(
                    force=True,
                    on_progress=lambda p: progress_calls.append(p.pct),
                )
                assert len(progress_calls) >= 7  # At least one per phase
                assert progress_calls[-1] == 1.0  # Final call is 100%


# ---------------------------------------------------------------------------
# Task 12: Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_bootstrap_then_query_players(self, temp_db):
        """Full pipeline: bootstrap players → query returns data."""
        with patch("src.database.DB_PATH", temp_db):
            with patch("src.live_stats.statsapi") as mock_api:
                mock_api.get.return_value = {
                    "people": [
                        {
                            "id": 1,
                            "fullName": "Test Hitter",
                            "active": True,
                            "currentTeam": {"abbreviation": "NYY"},
                            "primaryPosition": {"abbreviation": "SS", "type": "Hitter"},
                        },
                    ]
                }
                with patch("src.data_pipeline.refresh_if_stale", return_value=True):
                    from src.data_bootstrap import bootstrap_all_data
                    from src.database import get_connection

                    results = bootstrap_all_data(force=True)
                    assert "Error" not in str(results.get("players", ""))
                    conn = get_connection()
                    count = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
                    conn.close()
                    assert count >= 1
