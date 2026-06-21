"""Matchup live-line override — DB-free unit test of _apply_live_lines."""

from api.contracts.common import PlayerRef
from api.contracts.matchup import MatchPlayer, RosterRow
from api.services import matchup_service as ms


def _mp(mlb_id, state, stats):
    return MatchPlayer(
        player=PlayerRef(id=1, mlb_id=mlb_id, name="P", positions="OF"),
        pos="OF",
        status="vs SF · In Progress",
        state=state,
        stats=stats,
    )


def test_apply_live_lines_overrides_stats_and_status(monkeypatch):
    mp = _mp(100, "live", ["0/0", "0", "0", "0", "0", ".000", ".000"])
    row = RosterRow(slot="OF", you=mp, opp=None)
    monkeypatch.setattr(
        ms,
        "fetch_live_player_lines",
        lambda *a, **k: {
            100: {"hitter": ["2/4", "1", "1", "2", "1", ".500", ".600"], "pitcher": [], "status": "Top 5 · 3-2"}
        },
    )
    ms._apply_live_lines([row], [], schedule=[{"game_id": 1, "status": "In Progress"}], date_key="k")
    assert row.you.stats[0] == "2/4"
    assert "Top 5" in row.you.status


def test_apply_live_lines_skips_scheduled_players(monkeypatch):
    # A player whose game hasn't started (state="sched") keeps the projected line.
    mp = _mp(100, "sched", ["PROJ"])
    row = RosterRow(slot="OF", you=mp, opp=None)
    monkeypatch.setattr(
        ms, "fetch_live_player_lines", lambda *a, **k: {100: {"hitter": ["2/4"], "pitcher": [], "status": "x"}}
    )
    ms._apply_live_lines([row], [], schedule=[], date_key="k")
    assert row.you.stats == ["PROJ"]  # not overridden


def test_apply_live_lines_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ms, "fetch_live_player_lines", _boom)
    mp = _mp(100, "live", ["PROJ"])
    row = RosterRow(slot="OF", you=mp, opp=None)
    ms._apply_live_lines([row], [], schedule=[], date_key="k")  # no exception
    assert row.you.stats == ["PROJ"]  # untouched on failure
