from api.services.live_boxscore import fetch_live_player_lines, reset_cache


def setup_function(_):
    reset_cache()


def _fake_box(_gid):
    # Real boxscore_data shape (verified against statsapi).
    return {
        "home": {
            "players": {
                "ID100": {  # a hitter: 2-for-4, HR, 2 RBI, SB, 1 BB
                    "person": {"id": 100, "fullName": "Bat Man"},
                    "stats": {
                        "batting": {
                            "hits": 2,
                            "atBats": 4,
                            "runs": 1,
                            "homeRuns": 1,
                            "rbi": 2,
                            "stolenBases": 1,
                            "baseOnBalls": 1,
                            "strikeOuts": 0,
                        },
                        "pitching": {},
                    },
                },
                "ID200": {  # a pitcher: 6.1 IP, won, 7 K, 2 ER, 5 H, 1 BB
                    "person": {"id": 200, "fullName": "Ace Pitcher"},
                    "stats": {
                        "batting": {},
                        "pitching": {
                            "inningsPitched": "6.1",
                            "strikeOuts": 7,
                            "earnedRuns": 2,
                            "hits": 5,
                            "baseOnBalls": 1,
                            "note": "(W, 5)",
                        },
                    },
                },
            }
        },
        "away": {"players": {}},
    }


_SCHED = [
    {
        "game_id": 1,
        "status": "Final",
        "inning_state": "",
        "current_inning": 9,
        "home_score": 5,
        "away_score": 3,
        "home_name": "A",
        "away_name": "B",
    }
]


def test_hitter_line_from_boxscore():
    m = fetch_live_player_lines(_SCHED, date_key="t1", boxscore_fn=_fake_box)
    assert 100 in m
    line = m[100]["hitter"]  # [H/AB, R, HR, RBI, SB, AVG, OBP]
    assert line[0] == "2/4"
    assert line[1] == "1" and line[2] == "1" and line[3] == "2" and line[4] == "1"
    assert line[5] == ".500"  # 2/4
    assert m[100]["pitcher"] == []


def test_pitcher_line_from_boxscore():
    m = fetch_live_player_lines(_SCHED, date_key="t2", boxscore_fn=_fake_box)
    assert 200 in m
    line = m[200]["pitcher"]  # [IP, W, L, SV, K, ERA, WHIP]
    assert line[0] == "6.3"  # 6.1 outs -> 6.333 -> "6.3"
    assert line[1] == "1" and line[2] == "0" and line[3] == "0"  # W=1 from note
    assert line[4] == "7"  # K
    assert line[5] == "2.84"  # 2*9/6.333
    assert line[6] == "0.95"  # (1+5)/6.333
    assert m[200]["hitter"] == []


def test_status_string_present():
    m = fetch_live_player_lines(_SCHED, date_key="t3", boxscore_fn=_fake_box)
    assert "Final" in m[100]["status"]


def test_scheduled_games_skipped():
    sched = [{"game_id": 9, "status": "Scheduled", "home_name": "A", "away_name": "B"}]
    assert fetch_live_player_lines(sched, date_key="t4", boxscore_fn=_fake_box) == {}


def test_fetch_failure_is_graceful():
    def _boom(_gid):
        raise RuntimeError("api down")

    assert fetch_live_player_lines(_SCHED, date_key="t5", boxscore_fn=_boom) == {}
