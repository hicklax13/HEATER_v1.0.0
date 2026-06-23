"""Matchup live-line override — DB-free unit test of _apply_live_lines."""

from api.contracts.common import PlayerRef, StatItem
from api.contracts.matchup import MatchPlayer, RosterRow
from api.services import matchup_service as ms

_HITTER_COLUMNS = ["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"]


def _stat_items(*values) -> list[StatItem]:
    """Build list[StatItem] from values, using HITTER_COLUMNS for labels."""
    return [StatItem(label=col, value=str(v)) for col, v in zip(_HITTER_COLUMNS, values)]


def _mp(mlb_id, state, stats):
    return MatchPlayer(
        player=PlayerRef(id=1, mlb_id=mlb_id, name="P", positions="OF"),
        pos="OF",
        status="vs SF · In Progress",
        state=state,
        stats=stats,  # must be list[StatItem]
    )


def test_apply_live_lines_overrides_stats_and_status(monkeypatch):
    mp = _mp(100, "live", _stat_items("0/0", "0", "0", "0", "0", ".000", ".000"))
    row = RosterRow(slot="OF", you=mp, opp=None)
    monkeypatch.setattr(
        ms,
        "fetch_live_player_lines",
        lambda *a, **k: {
            100: {"hitter": ["2/4", "1", "1", "2", "1", ".500", ".600"], "pitcher": [], "status": "Top 5 · 3-2"}
        },
    )
    ms._apply_live_lines([row], [], schedule=[{"game_id": 1, "status": "In Progress"}], date_key="k")
    # stats are now list[StatItem]; the override wraps live strings into StatItem
    assert row.you.stats[0].value == "2/4"
    assert row.you.stats[0].label == "H/AB"
    assert "Top 5" in row.you.status


def test_apply_live_lines_skips_scheduled_players(monkeypatch):
    # A player whose game hasn't started (state="sched") keeps the projected line.
    mp = _mp(100, "sched", _stat_items("PROJ", "0", "0", "0", "0", ".000", ".000"))
    row = RosterRow(slot="OF", you=mp, opp=None)
    monkeypatch.setattr(
        ms, "fetch_live_player_lines", lambda *a, **k: {100: {"hitter": ["2/4"], "pitcher": [], "status": "x"}}
    )
    ms._apply_live_lines([row], [], schedule=[], date_key="k")
    assert row.you.stats[0].value == "PROJ"  # not overridden


def test_apply_live_lines_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ms, "fetch_live_player_lines", _boom)
    mp = _mp(100, "live", _stat_items("PROJ", "0", "0", "0", "0", ".000", ".000"))
    row = RosterRow(slot="OF", you=mp, opp=None)
    ms._apply_live_lines([row], [], schedule=[], date_key="k")  # no exception
    assert row.you.stats[0].value == "PROJ"  # untouched on failure
