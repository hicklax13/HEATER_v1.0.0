"""Stream board guards: live/final games lock, future dates carry confidence.

Phase 1 of the Pitcher Streaming Analyzer plan. A pitcher whose game is in
progress or final can no longer be streamed — the board must keep the row
visible (transparency) but mark it non-actionable, mirroring the
LOCKED_GAME_STATUSES convention from the DCV engine. Future-date boards never
lock; they carry the two_start.py probable-confidence tier instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from src.optimizer.stream_analyzer import build_stream_board
from src.valuation import LeagueConfig


def _pool():
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Streamer A",
                "team": "SEA",
                "positions": "SP",
                "is_hitter": 0,
                "throws": "R",
                "era": 3.40,
                "whip": 1.10,
                "k": 170.0,
                "w": 11.0,
                "ip": 168.0,
                "percent_owned": 14.0,
            },
            {
                "player_id": 2,
                "player_name": "Streamer B",
                "team": "BOS",
                "positions": "SP",
                "is_hitter": 0,
                "throws": "L",
                "era": 3.90,
                "whip": 1.22,
                "k": 150.0,
                "w": 9.0,
                "ip": 160.0,
                "percent_owned": 9.0,
            },
        ]
    )


def _ctx(**overrides):
    base = {
        "player_pool": _pool(),
        "user_roster_ids": [],
        "league_rostered_ids": set(),
        "team_strength": {},
        "park_factors": {"SEA": 0.95, "BOS": 1.04},
        "weather": {},
        "two_start_pitchers": [],
        "recent_form": {},
        "todays_schedule": [],
        "config": LeagueConfig(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _schedule(date_str, status_b="Scheduled"):
    return [
        {
            "game_date": date_str,
            "home_name": "Seattle Mariners",
            "away_name": "Chicago White Sox",
            "home_probable_pitcher": "Streamer A",
            "away_probable_pitcher": "",
            "status": "Scheduled",
        },
        {
            "game_date": date_str,
            "home_name": "Boston Red Sox",
            "away_name": "New York Yankees",
            "home_probable_pitcher": "Streamer B",
            "away_probable_pitcher": "",
            "status": status_b,
        },
    ]


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def test_in_progress_game_locks_row_but_stays_visible():
    today = _today()
    board = build_stream_board(_ctx(), today, schedule=_schedule(today, status_b="In Progress"))
    assert len(board) == 2, "locked rows must stay on the board (transparency)"
    by_name = board.set_index("player_name")
    assert by_name.loc["Streamer B", "status"] == "LOCKED"
    assert not by_name.loc["Streamer B", "actionable"]
    assert by_name.loc["Streamer A", "status"] == "PROBABLE"
    assert by_name.loc["Streamer A", "actionable"]


def test_final_game_marked_final():
    today = _today()
    board = build_stream_board(_ctx(), today, schedule=_schedule(today, status_b="Final"))
    by_name = board.set_index("player_name")
    assert by_name.loc["Streamer B", "status"] == "FINAL"
    assert not by_name.loc["Streamer B", "actionable"]


def test_future_date_never_locks_and_carries_confidence():
    future = (datetime.now(UTC) + timedelta(days=6)).strftime("%Y-%m-%d")
    board = build_stream_board(_ctx(), future, schedule=_schedule(future, status_b="In Progress"))
    assert (board["status"] != "LOCKED").all(), (
        "future dates cannot lock — a stale/garbled status string must not make tomorrow's start non-actionable"
    )
    assert board["actionable"].all()
    assert set(board["confidence"]) <= {"HIGH", "MEDIUM", "LOW"}
    assert (board["confidence"] == "LOW").all(), "+6 days out is a LOW-confidence probable"


def test_rostered_pitchers_excluded_by_default():
    today = _today()
    ctx = _ctx(league_rostered_ids={2})
    board = build_stream_board(ctx, today, schedule=_schedule(today))
    assert list(board["player_name"]) == ["Streamer A"]


def test_include_rostered_adds_my_sps():
    today = _today()
    ctx = _ctx(league_rostered_ids={2}, user_roster_ids=[2])
    board = build_stream_board(ctx, today, schedule=_schedule(today), include_rostered=True)
    by_name = board.set_index("player_name")
    assert by_name.loc["Streamer B", "rostered"]
    assert not by_name.loc["Streamer A", "rostered"]


def test_two_start_flag_and_sgp_volume():
    today = _today()
    one = build_stream_board(_ctx(), today, schedule=_schedule(today))
    two = build_stream_board(_ctx(two_start_pitchers=[1]), today, schedule=_schedule(today))
    one_a = one.set_index("player_name").loc["Streamer A"]
    two_a = two.set_index("player_name").loc["Streamer A"]
    assert one_a["num_starts"] == 1
    assert two_a["num_starts"] == 2
    assert two_a["net_sgp"] > one_a["net_sgp"], "a two-start week carries more marginal SGP than a single start"


def test_empty_schedule_returns_empty_board():
    board = build_stream_board(_ctx(), _today(), schedule=[])
    assert board.empty


def test_board_sorted_by_score_descending():
    today = _today()
    board = build_stream_board(_ctx(), today, schedule=_schedule(today))
    scores = list(board["stream_score"])
    assert scores == sorted(scores, reverse=True)


def test_board_enriches_off_roster_opponent_strength(monkeypatch):
    """ctx.team_strength only covers the user's rostered teams — the board
    must enrich missing opponents via game_day.get_team_strength so
    off-roster opponents don't all score as a neutral 100 wRC+ / 22 K%
    (2026-06-10 live finding)."""
    import src.game_day as game_day

    fetched = {}

    def _fake_ts(abbr):
        fetched[abbr] = True
        return {"wrc_plus": 85.0, "k_pct": 26.0}

    monkeypatch.setattr(game_day, "get_team_strength", _fake_ts)
    today = _today()
    board = build_stream_board(_ctx(team_strength={}), today, schedule=_schedule(today))
    assert fetched, "missing opponents must be fetched"
    assert (board["opp_wrc_plus"] == 85.0).all()


def test_board_neutral_when_strength_fetch_fails(monkeypatch):
    """Fetch failures degrade to the neutral line — never raise."""
    import src.game_day as game_day

    def _boom(abbr):
        raise RuntimeError("statsapi down")

    monkeypatch.setattr(game_day, "get_team_strength", _boom)
    today = _today()
    board = build_stream_board(_ctx(team_strength={}), today, schedule=_schedule(today))
    assert len(board) == 2
    assert (board["opp_wrc_plus"] == 100.0).all()


def test_board_prefers_ctx_team_strength(monkeypatch):
    """Teams already in ctx.team_strength are NOT re-fetched."""
    import src.game_day as game_day

    def _boom(abbr):
        raise AssertionError(f"should not fetch {abbr}")

    monkeypatch.setattr(game_day, "get_team_strength", _boom)
    today = _today()
    strength = {
        "CWS": {"wrc_plus": 80.0, "k_pct": 27.0},
        "NYY": {"wrc_plus": 115.0, "k_pct": 19.0},
        "SEA": {"wrc_plus": 100.0, "k_pct": 22.0},
        "BOS": {"wrc_plus": 100.0, "k_pct": 22.0},
    }
    board = build_stream_board(_ctx(team_strength=strength), today, schedule=_schedule(today))
    by_name = board.set_index("player_name")
    assert by_name.loc["Streamer A", "opp_wrc_plus"] == 80.0
    assert by_name.loc["Streamer B", "opp_wrc_plus"] == 115.0
