"""Test live stats module — MLB Stats API + pybaseball integration."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (2, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def test_match_player_id(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Aaron Judge", "NYY")
    assert pid == 1


def test_match_player_id_not_found(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Nonexistent Player", "XXX")
    assert pid is None


@patch("src.live_stats.statsapi")
def test_fetch_season_stats_structure(mock_statsapi, temp_db):
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 147}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "fullName": "Aaron Judge",
                            "currentTeam": {"abbreviation": "NYY"},
                            "stats": [
                                {
                                    "group": {"displayName": "hitting"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "plateAppearances": 500,
                                                "atBats": 450,
                                                "hits": 130,
                                                "runs": 80,
                                                "homeRuns": 35,
                                                "rbi": 90,
                                                "stolenBases": 5,
                                                "avg": ".289",
                                                "gamesPlayed": 120,
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

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    assert "player_name" in df.columns
    assert "hr" in df.columns


def test_get_refresh_age():
    from src.live_stats import _get_refresh_age_hours

    age = _get_refresh_age_hours("nonexistent_source")
    assert age > 24


@patch("src.live_stats.statsapi")
def test_refresh_all_stats(mock_statsapi, temp_db):
    from src.live_stats import refresh_all_stats

    mock_statsapi.get.return_value = {"people": []}

    result = refresh_all_stats(force=True)
    assert isinstance(result, dict)
    assert "season_stats" in result


# ── SF-3: Two-Way Player detection ──────────────────────────────


def test_two_way_player_is_pitcher():
    """SF-3: Two-Way Players must be detected as pitchers for stat parsing."""
    pos_type = "Two-Way Player"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True


def test_regular_pitcher_still_detected():
    """Regular pitchers must still be detected as pitchers."""
    pos_type = "Pitcher"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is True


def test_hitter_not_detected_as_pitcher():
    """Hitters must NOT be detected as pitchers."""
    pos_type = "Hitter"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is False


def test_outfielder_not_detected_as_pitcher():
    """Outfielders must NOT be detected as pitchers."""
    pos_type = "Outfielder"
    is_pitcher = pos_type in ("Pitcher", "Two-Way Player")
    assert is_pitcher is False


@patch("src.live_stats.statsapi")
def test_two_way_player_emits_both_rows(mock_statsapi, temp_db):
    """SF-3: A Two-Way Player should emit both hitting and pitching rows."""
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 17}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "id": 660271,
                            "fullName": "Shohei Ohtani",
                            "currentTeam": {"abbreviation": "LAD"},
                            "stats": [
                                {
                                    "group": {"displayName": "hitting"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "plateAppearances": 300,
                                                "atBats": 270,
                                                "hits": 85,
                                                "runs": 50,
                                                "homeRuns": 20,
                                                "rbi": 55,
                                                "stolenBases": 10,
                                                "avg": ".315",
                                                "gamesPlayed": 70,
                                            }
                                        }
                                    ],
                                },
                                {
                                    "group": {"displayName": "pitching"},
                                    "splits": [
                                        {
                                            "stat": {
                                                "wins": 8,
                                                "losses": 2,
                                                "era": "2.50",
                                                "whip": "0.95",
                                                "strikeOuts": 100,
                                                "saves": 0,
                                                "inningsPitched": "90.0",
                                                "gamesPlayed": 15,
                                            }
                                        }
                                    ],
                                },
                            ],
                        },
                        "position": {"type": "Two-Way Player"},
                    }
                ]
            }
        return {}

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    # Two-Way Player should produce 2 rows: one hitting, one pitching
    ohtani_rows = df[df["player_name"] == "Shohei Ohtani"]
    assert len(ohtani_rows) == 2, f"Expected 2 rows for Ohtani, got {len(ohtani_rows)}"
    # One row should have is_hitter=True (hitting stats), one is_hitter=False (pitching)
    assert ohtani_rows["is_hitter"].sum() == 1, "Expected exactly 1 hitter row"


@patch("src.live_stats.statsapi")
def test_two_way_player_zero_stats_emits_both(mock_statsapi, temp_db):
    """SF-3: Two-Way Player with no stats should still emit both rows."""
    from src.live_stats import fetch_season_stats

    def mock_get(endpoint, params=None, **kwargs):
        if endpoint == "teams":
            return {"teams": [{"id": 17}]}
        if endpoint == "team_roster":
            return {
                "roster": [
                    {
                        "person": {
                            "id": 660271,
                            "fullName": "Shohei Ohtani",
                            "currentTeam": {"abbreviation": "LAD"},
                            "stats": [],  # No stats yet (e.g. IL)
                        },
                        "position": {"type": "Two-Way Player"},
                    }
                ]
            }
        return {}

    mock_statsapi.get.side_effect = mock_get

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    ohtani_rows = df[df["player_name"] == "Shohei Ohtani"]
    assert len(ohtani_rows) == 2, f"Expected 2 zero-stat rows for Ohtani, got {len(ohtani_rows)}"


# ── H1 (PR A): TWP routing in save_season_stats_to_db ───────────


def _insert_ohtani(temp_db, stored_is_hitter: int = 1) -> None:
    """Insert Ohtani with TWP positions for use by save-side tests."""
    conn = sqlite3.connect(temp_db)
    try:
        conn.execute(
            "INSERT INTO players (player_id, mlb_id, name, team, positions, is_hitter) "
            "VALUES (3, 660271, 'Shohei Ohtani', 'LAD', 'DH,SP,TWP', ?)",
            (stored_is_hitter,),
        )
        conn.commit()
    finally:
        conn.close()


def _row(**overrides):
    """Build a stats_df row dict with sensible zero defaults for unused fields."""
    base = {
        "player_name": "",
        "team": "",
        "mlb_id": 0,
        "is_hitter": False,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "ip": 0.0,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "era": 0.0,
        "whip": 0.0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "games_played": 0,
    }
    base.update(overrides)
    return base


def test_twp_pitcher_row_routed_not_skipped(temp_db):
    """H1: A TWP player's pitcher-stats row must NOT be rejected by the type guard."""
    from src.live_stats import save_season_stats_to_db

    _insert_ohtani(temp_db, stored_is_hitter=1)
    pitching_row = _row(
        player_name="Shohei Ohtani",
        team="LAD",
        mlb_id=660271,
        is_hitter=False,
        ip=44.0,
        w=4,
        l=2,
        k=55,
        era=3.20,
        whip=1.05,
        er=16,
        games_played=8,
    )
    saved = save_season_stats_to_db(pd.DataFrame([pitching_row]), season=2026)
    assert saved == 1, "TWP pitcher row should be saved, not skipped by type guard"

    conn = sqlite3.connect(temp_db)
    try:
        cur = conn.execute(
            "SELECT ip, w, l, k, era, whip, er, ab, h, hr FROM season_stats WHERE player_id=3 AND season=2026"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None
    ip, w, l, k, era, whip, er, ab, h, hr = row
    assert ip == 44.0 and w == 4 and l == 2 and k == 55 and round(era, 2) == 3.20 and er == 16
    # Hitting cols should be untouched (no prior hitter row written), which is 0.
    assert ab == 0 and h == 0 and hr == 0


def test_twp_hitter_row_preserves_pitching_cols(temp_db):
    """H1: A TWP player's hitting-stats row must NOT zero out previously-written pitching cols."""
    from src.live_stats import save_season_stats_to_db

    _insert_ohtani(temp_db, stored_is_hitter=1)
    # First: write the pitching half.
    save_season_stats_to_db(
        pd.DataFrame(
            [
                _row(
                    player_name="Shohei Ohtani",
                    team="LAD",
                    mlb_id=660271,
                    is_hitter=False,
                    ip=90.0,
                    w=8,
                    k=100,
                    era=2.50,
                    er=25,
                    games_played=15,
                )
            ]
        ),
        season=2026,
    )
    # Then: write the hitting half.
    save_season_stats_to_db(
        pd.DataFrame(
            [
                _row(
                    player_name="Shohei Ohtani",
                    team="LAD",
                    mlb_id=660271,
                    is_hitter=True,
                    pa=300,
                    ab=270,
                    h=85,
                    hr=20,
                    rbi=55,
                    avg=0.315,
                    games_played=70,
                )
            ]
        ),
        season=2026,
    )

    conn = sqlite3.connect(temp_db)
    try:
        cur = conn.execute(
            "SELECT ab, h, hr, rbi, ip, w, k, era, er FROM season_stats WHERE player_id=3 AND season=2026"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None
    ab, h, hr, rbi, ip, w, k, era, er = row
    # Hitting cols set by the second call:
    assert ab == 270 and h == 85 and hr == 20 and rbi == 55
    # Pitching cols preserved from the first call (the bug being fixed would zero these):
    assert ip == 90.0 and w == 8 and k == 100 and round(era, 2) == 2.50 and er == 25


def test_non_twp_type_mismatch_still_rejected(temp_db):
    """H1 regression guard: Bellinger-style type guard must remain intact for non-TWP players."""
    from src.live_stats import save_season_stats_to_db

    # Cole is a regular pitcher (player_id=2, positions='SP'). A hitter-stats
    # row claiming to be his should still be rejected.
    bad_row = _row(
        player_name="Gerrit Cole",
        team="NYY",
        is_hitter=True,
        pa=400,
        ab=350,
        h=100,
        hr=15,
        rbi=60,
        avg=0.286,
        games_played=100,
    )
    saved = save_season_stats_to_db(pd.DataFrame([bad_row]), season=2026)
    assert saved == 0, "Non-TWP type-mismatch row must still be rejected"


def test_is_twp_helper():
    """H1 helper: _is_twp recognizes both the explicit TWP token and the multi-eligibility pattern."""
    from src.live_stats import _is_twp

    assert _is_twp("DH,SP,TWP") is True  # Ohtani
    assert _is_twp("TWP") is True  # explicit only
    assert _is_twp("SP,1B") is True  # pitcher + hitter eligibility
    assert _is_twp("OF") is False
    assert _is_twp("SP") is False
    assert _is_twp("SP,RP") is False  # both pitcher tokens, no hitter eligibility
    assert _is_twp("") is False
    assert _is_twp(None) is False
