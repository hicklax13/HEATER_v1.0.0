"""Real-class coverage for api/services/live_boxscore.py pure parsers.

All tests are DB-free: they call the module-level private functions directly
with synthetic statsapi-shaped dicts.  No network, no live DB required.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Ensure src.live_stats._ip_outs_to_decimal is importable without hitting
# the full src bootstrap.  We only need the one function; if the module
# is already in sys.modules (e.g. from a prior import) that's fine too.
# We DO NOT stub it — the real _ip_outs_to_decimal is what we want to test
# via the boxscore parsers.
# ---------------------------------------------------------------------------
from api.services.live_boxscore import (
    _HITTER_KEY,
    _PITCHER_KEY,
    _avg,
    _f,
    _has_batting,
    _hitter_line,
    _pitcher_line,
    _state_of,
    _status_str,
    fetch_live_player_lines,
    reset_cache,
)

# ---------------------------------------------------------------------------
# _f — NaN/inf/type safety
# ---------------------------------------------------------------------------


def test_f_converts_numeric_string():
    assert _f("3") == 3.0
    assert _f("0") == 0.0


def test_f_uses_default_on_none():
    assert _f(None) == 0.0
    assert _f(None, default=99.0) == 99.0


def test_f_uses_default_on_nan():
    assert _f(float("nan")) == 0.0


def test_f_uses_default_on_inf():
    assert _f(float("inf")) == 0.0
    assert _f(float("-inf")) == 0.0


def test_f_uses_default_on_bad_string():
    assert _f("bad") == 0.0


# ---------------------------------------------------------------------------
# _avg — formatting helper
# ---------------------------------------------------------------------------


def test_avg_strips_leading_zero():
    assert _avg(0.275) == ".275"


def test_avg_zero():
    assert _avg(0.0) == ".000"


def test_avg_above_one():
    # e.g. a weird edge input — should just format as "1.000" (no strip)
    assert _avg(1.0) == "1.000"


# ---------------------------------------------------------------------------
# _hitter_line — stat line parsing
# ---------------------------------------------------------------------------


def _hitter_bat(h=2, ab=4, runs=1, hr=1, rbi=2, sb=0, bb=1) -> dict:
    return {
        "hits": h,
        "atBats": ab,
        "runs": runs,
        "homeRuns": hr,
        "rbi": rbi,
        "stolenBases": sb,
        "baseOnBalls": bb,
    }


def test_hitter_line_basic():
    line = _hitter_line(_hitter_bat(h=2, ab=4, runs=1, hr=1, rbi=2, sb=0, bb=1))
    # [h/ab, R, HR, RBI, SB, AVG, OBP]
    assert line[0] == "2/4"  # hits/AB
    assert line[1] == "1"  # R
    assert line[2] == "1"  # HR
    assert line[3] == "2"  # RBI
    assert line[4] == "0"  # SB
    # AVG = 2/4 = 0.500 → ".500"
    assert line[5] == ".500"
    # OBP = (2+1)/(4+1) = 0.600 → ".600"
    assert line[6] == ".600"


def test_hitter_line_perfect_avg():
    line = _hitter_line(_hitter_bat(h=3, ab=3, bb=0))
    assert line[5] == "1.000"  # AVG = 1.0, no leading-zero strip


def test_hitter_line_no_at_bats():
    """0 AB → AVG=0.000, OBP=0.000, no ZeroDivision."""
    line = _hitter_line(_hitter_bat(h=0, ab=0, bb=0))
    assert line[5] == ".000"
    assert line[6] == ".000"


def test_hitter_line_walk_only():
    """0 AB, 1 BB → OBP = 1/(0+1) = 1.000."""
    bat = {"hits": 0, "atBats": 0, "runs": 0, "homeRuns": 0, "rbi": 0, "stolenBases": 0, "baseOnBalls": 1}
    line = _hitter_line(bat)
    assert line[6] == "1.000"


def test_hitter_line_missing_fields_default_zero():
    """Partial dict — missing keys degrade to 0 without raising."""
    line = _hitter_line({})
    assert line[0] == "0/0"
    assert line[1] == "0"
    assert line[5] == ".000"


def test_hitter_line_none_values_default_zero():
    bat = {k: None for k in ("hits", "atBats", "runs", "homeRuns", "rbi", "stolenBases", "baseOnBalls")}
    line = _hitter_line(bat)
    assert line[0] == "0/0"
    assert line[5] == ".000"


# ---------------------------------------------------------------------------
# _pitcher_line — stat line parsing
# ---------------------------------------------------------------------------


def _pitcher_pit(ip="6.0", er=2, bb=1, h=5, k=7, note="(W, 5)") -> dict:
    return {
        "inningsPitched": ip,
        "earnedRuns": er,
        "baseOnBalls": bb,
        "hits": h,
        "strikeOuts": k,
        "note": note,
    }


def test_pitcher_line_win():
    line = _pitcher_line(_pitcher_pit(ip="6.0", er=2, bb=1, h=5, k=7, note="(W, 5)"))
    # [IP, W, L, SV, K, ERA, WHIP]
    assert line[0] == "6.0"
    assert line[1] == "1"  # W
    assert line[2] == "0"  # L
    assert line[3] == "0"  # SV
    assert line[4] == "7"  # K
    # ERA = 2*9/6 = 3.00
    assert line[5] == "3.00"
    # WHIP = (1+5)/6 = 1.00
    assert line[6] == "1.00"


def test_pitcher_line_loss():
    line = _pitcher_line(_pitcher_pit(note="(L, 3)"))
    assert line[1] == "0"  # W
    assert line[2] == "1"  # L
    assert line[3] == "0"  # SV


def test_pitcher_line_save():
    line = _pitcher_line(_pitcher_pit(ip="1.0", er=0, bb=0, h=1, k=1, note="(S, 12)"))
    assert line[1] == "0"  # W
    assert line[2] == "0"  # L
    assert line[3] == "1"  # SV
    # ERA = 0*9/1 = 0.00
    assert line[5] == "0.00"
    # WHIP = (0+1)/1 = 1.00
    assert line[6] == "1.00"


def test_pitcher_line_fractional_ip():
    """'6.2' = 6 + 2 outs = 6.667 IP via _ip_outs_to_decimal."""
    line = _pitcher_line(_pitcher_pit(ip="6.2", er=3, bb=2, h=6, k=8, note=""))
    # ip = 6 + 2/3 ≈ 6.6667
    # ERA = 3*9/6.6667 ≈ 4.05
    # WHIP = (2+6)/6.6667 ≈ 1.20
    era_val = float(line[5])
    whip_val = float(line[6])
    assert abs(era_val - 4.05) < 0.02, f"ERA {era_val} != ~4.05"
    assert abs(whip_val - 1.20) < 0.02, f"WHIP {whip_val} != ~1.20"


def test_pitcher_line_zero_ip():
    """0 IP → ERA=0.00, WHIP=0.00, no ZeroDivision."""
    line = _pitcher_line(_pitcher_pit(ip="0.0", er=0, bb=0, h=0, k=0))
    assert line[5] == "0.00"
    assert line[6] == "0.00"


def test_pitcher_line_missing_fields():
    """Empty dict — no raise; IP defaults to 0.0, everything else 0."""
    line = _pitcher_line({})
    assert line[0] == "0.0"
    assert line[1] == "0"
    assert line[5] == "0.00"


def test_pitcher_line_none_ip():
    """None IP → _ip_outs_to_decimal returns 0.0; no raise."""
    pit = _pitcher_pit()
    pit["inningsPitched"] = None
    line = _pitcher_line(pit)
    assert line[0] == "0.0"


# ---------------------------------------------------------------------------
# _has_batting
# ---------------------------------------------------------------------------


def test_has_batting_true_on_atbats():
    assert _has_batting({"atBats": 3}) is True


def test_has_batting_true_on_hits():
    assert _has_batting({"atBats": 0, "hits": 1}) is True


def test_has_batting_false_all_zero():
    assert _has_batting({"atBats": 0, "hits": 0, "runs": 0, "baseOnBalls": 0}) is False


def test_has_batting_false_empty():
    assert _has_batting({}) is False


# ---------------------------------------------------------------------------
# _state_of — needs src.game_day constants; stub them
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def stub_game_day(monkeypatch):
    """Stub src.game_day with the canonical status sets the service uses."""
    gd_mod = types.ModuleType("src.game_day")
    gd_mod.FINAL_GAME_STATUSES = {"final", "game over", "completed early"}
    gd_mod.LOCKED_GAME_STATUSES = {"in progress", "live"}
    monkeypatch.setitem(sys.modules, "src.game_day", gd_mod)
    return gd_mod


def test_state_of_final(stub_game_day):
    assert _state_of("Final") == "final"


def test_state_of_live(stub_game_day):
    assert _state_of("In Progress") == "live"


def test_state_of_sched(stub_game_day):
    assert _state_of("Preview") == "sched"


def test_state_of_empty(stub_game_day):
    assert _state_of("") == "sched"


# ---------------------------------------------------------------------------
# _status_str
# ---------------------------------------------------------------------------


def test_status_str_final():
    game = {"away_score": 3, "home_score": 5}
    assert _status_str(game, "final") == "Final · 3-5"


def test_status_str_live_with_inning():
    game = {"away_score": 2, "home_score": 1, "inning_state": "Top", "current_inning": "7"}
    s = _status_str(game, "live")
    assert "Top 7" in s
    assert "2-1" in s


def test_status_str_missing_scores_default_zero():
    game = {}
    s = _status_str(game, "final")
    assert "0-0" in s


# ---------------------------------------------------------------------------
# fetch_live_player_lines — integration with a fake boxscore_fn
# ---------------------------------------------------------------------------


def _fake_boxscore_fn(game_id):
    """Returns a minimal statsapi boxscore_data dict for game_id 999."""
    return {
        "home": {
            "players": {
                "ID101": {
                    "person": {"id": 101},
                    "stats": {
                        "batting": {
                            "hits": 2,
                            "atBats": 4,
                            "runs": 1,
                            "homeRuns": 0,
                            "rbi": 1,
                            "stolenBases": 0,
                            "baseOnBalls": 0,
                        },
                        "pitching": {},
                    },
                },
                "ID202": {
                    "person": {"id": 202},
                    "stats": {
                        "batting": {},
                        "pitching": {
                            "inningsPitched": "6.0",
                            "earnedRuns": 2,
                            "baseOnBalls": 1,
                            "hits": 5,
                            "strikeOuts": 7,
                            "note": "(W, 5)",
                        },
                    },
                },
            }
        },
        "away": {"players": {}},
    }


def test_fetch_live_player_lines_hitter(stub_game_day):
    reset_cache()
    schedule = [{"game_id": 999, "status": "Final", "away_score": 3, "home_score": 5}]
    result = fetch_live_player_lines(schedule, date_key="test-hitter", boxscore_fn=_fake_boxscore_fn)

    assert 101 in result
    entry = result[101]
    assert entry["hitter"][0] == "2/4"  # H/AB
    assert entry["hitter"][5] == ".500"  # AVG
    assert entry["status"] == "Final · 3-5"


def test_fetch_live_player_lines_pitcher(stub_game_day):
    reset_cache()
    schedule = [{"game_id": 999, "status": "Final", "away_score": 3, "home_score": 5}]
    result = fetch_live_player_lines(schedule, date_key="test-pitcher", boxscore_fn=_fake_boxscore_fn)

    assert 202 in result
    entry = result[202]
    assert entry["pitcher"][0] == "6.0"  # IP
    assert entry["pitcher"][1] == "1"  # W
    assert entry["pitcher"][5] == "3.00"  # ERA = 2*9/6


def test_fetch_live_player_lines_skips_scheduled_games(stub_game_day):
    reset_cache()
    schedule = [{"game_id": 999, "status": "Preview", "away_score": 0, "home_score": 0}]
    result = fetch_live_player_lines(schedule, date_key="test-sched", boxscore_fn=_fake_boxscore_fn)
    assert result == {}


def test_fetch_live_player_lines_boxscore_raises_skipped(stub_game_day):
    reset_cache()

    def bad_boxscore(gid):
        raise ConnectionError("network error")

    schedule = [{"game_id": 999, "status": "Final", "away_score": 0, "home_score": 0}]
    # Should not raise; failed game contributes nothing
    result = fetch_live_player_lines(schedule, date_key="test-fail", boxscore_fn=bad_boxscore)
    assert result == {}


def test_fetch_live_player_lines_ttl_cache(stub_game_day):
    reset_cache()
    calls = []

    def counting_boxscore(gid):
        calls.append(gid)
        return _fake_boxscore_fn(gid)

    schedule = [{"game_id": 999, "status": "Final", "away_score": 0, "home_score": 0}]
    fetch_live_player_lines(schedule, date_key="ttl-key", boxscore_fn=counting_boxscore)
    fetch_live_player_lines(schedule, date_key="ttl-key", boxscore_fn=counting_boxscore)
    # Second call should be served from cache — boxscore_fn called only once
    assert len(calls) == 1


def test_fetch_live_player_lines_empty_schedule(stub_game_day):
    reset_cache()
    result = fetch_live_player_lines([], date_key="empty", boxscore_fn=_fake_boxscore_fn)
    assert result == {}


def test_fetch_live_player_lines_none_schedule(stub_game_day):
    reset_cache()
    result = fetch_live_player_lines(None, date_key="none", boxscore_fn=_fake_boxscore_fn)
    assert result == {}


# ---------------------------------------------------------------------------
# TWP (two-way player / Ohtani-class) merge: same mlb_id appears as BOTH
# batter and pitcher in the same game — both stat lines must survive on one
# entry (no line is clobbered).
# ---------------------------------------------------------------------------


def _fake_boxscore_twp(game_id):
    """Boxscore where mlb_id 660271 (Ohtani-class) is a batter on the home
    side AND a pitcher on the away side (e.g. DH in one half-inning,
    started on the mound — statsapi surfaces the player once per team-side
    with all their stats; a real TWP day has batting stats on the batting
    side and pitching stats on the pitching side).  We put them in *both*
    home and away players dicts so the loop encounters the same pid twice."""
    return {
        "home": {
            "players": {
                "ID660271": {
                    "person": {"id": 660271},
                    "stats": {
                        "batting": {
                            "hits": 1,
                            "atBats": 3,
                            "runs": 1,
                            "homeRuns": 1,
                            "rbi": 2,
                            "stolenBases": 0,
                            "baseOnBalls": 0,
                        },
                        "pitching": {},  # no pitching on this side
                    },
                }
            }
        },
        "away": {
            "players": {
                "ID660271": {
                    "person": {"id": 660271},
                    "stats": {
                        "batting": {},  # no batting on this side
                        "pitching": {
                            "inningsPitched": "5.2",
                            "earnedRuns": 1,
                            "baseOnBalls": 2,
                            "hits": 4,
                            "strikeOuts": 9,
                            "note": "(W, 3)",
                        },
                    },
                }
            }
        },
    }


def test_fetch_live_player_lines_twp_both_lines_populated(stub_game_day):
    """TWP merge: the single entry for Ohtani-class pid must carry BOTH the
    hitter line (from the batting-side encounter) AND the pitcher line (from
    the pitching-side encounter).  Neither key should be clobbered."""
    reset_cache()
    schedule = [{"game_id": 777, "status": "Final", "away_score": 2, "home_score": 5}]
    result = fetch_live_player_lines(schedule, date_key="test-twp", boxscore_fn=_fake_boxscore_twp)

    assert 660271 in result, "TWP pid must be present in result"
    entry = result[660271]

    # Exactly ONE entry for the pid (the merge worked — not two separate keys)
    assert isinstance(entry, dict)

    # Hitter line is populated (1/3, R=1, HR=1, RBI=2, SB=0, AVG=.333, OBP=.333)
    hitter = entry[_HITTER_KEY]
    assert hitter, "hitter line must not be empty for a TWP"
    assert hitter[0] == "1/3", f"expected '1/3', got {hitter[0]}"
    assert hitter[2] == "1", f"expected HR=1, got {hitter[2]}"  # HR

    # Pitcher line is populated (5.2 IP, W, 9 K)
    pitcher = entry[_PITCHER_KEY]
    assert pitcher, "pitcher line must not be empty for a TWP"
    assert pitcher[1] == "1", f"expected W=1, got {pitcher[1]}"  # W
    assert pitcher[4] == "9", f"expected K=9, got {pitcher[4]}"  # K
    # _pitcher_line formats IP as f"{ip:.1f}" where ip = _ip_outs_to_decimal("5.2")
    # = 5 + 2/3 ≈ 5.6667 → formatted as "5.7"
    assert pitcher[0] == "5.7", f"expected IP='5.7', got {pitcher[0]}"
