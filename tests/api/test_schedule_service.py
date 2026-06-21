"""DB-free tests for the probable-pitcher grid pure logic (band / availability /
cell mapping / grid assembly) with synthetic board rows + a fake roster map."""

from api.services.schedule_service import _assemble_grid, _availability, _band, _to_cell


def test_band_thresholds():
    assert _band(75) == "easy"
    assert _band(60) == "easy"  # boundary
    assert _band(50) == "medium"
    assert _band(40) == "tough"  # boundary
    assert _band(30) == "tough"


def test_availability_yours_taken_available():
    roster = {10: "Team Hickey", 20: "Rivals"}
    user = {10}
    assert _availability(10, roster, user) == ("yours", "Team Hickey")
    assert _availability(20, roster, user) == ("taken", "Rivals")
    assert _availability(99, roster, user) == ("available", None)


def _row(pid, team, **kw):
    base = {
        "player_id": pid,
        "player_name": f"P{pid}",
        "mlb_id": 600000 + pid,
        "team": team,
        "opponent": "SF",
        "is_home": True,
        "stream_score": 65.0,
        "num_starts": 1,
        "status": "OK",
        "confidence": "HIGH",
    }
    base.update(kw)
    return base


def test_to_cell_maps_difficulty_twostart_availability():
    cell = _to_cell(_row(1, "LAD", stream_score=70.0, num_starts=2), {1: "Team Hickey"}, {1})
    assert cell.difficulty == 70.0 and cell.band == "easy"
    assert cell.two_start is True
    assert cell.availability == "yours" and cell.rostered_by == "Team Hickey"
    assert cell.pitcher is not None and cell.pitcher.id == 1 and cell.pitcher.mlb_id == 600001


def test_to_cell_nan_score_is_safe():
    cell = _to_cell(_row(2, "NYY", stream_score=float("nan"), num_starts=float("nan")), {}, set())
    assert cell.difficulty == 0.0 and cell.band == "tough"
    assert cell.two_start is False  # NaN num_starts -> default 1


def test_assemble_grid_offday_none_and_twostart_two_cells():
    date_list = ["2026-06-21", "2026-06-22"]
    boards = [
        [_row(1, "LAD", num_starts=2), _row(5, "NYY")],
        [_row(1, "LAD", num_starts=2)],
    ]
    grid = _assemble_grid(boards, date_list, roster_map={}, user_ids=set())
    assert grid.days == date_list
    teams = {t.team: t for t in grid.teams}
    assert set(teams) == {"LAD", "NYY"}
    # two-start pitcher fills both day cells
    assert teams["LAD"].cells[0] is not None and teams["LAD"].cells[1] is not None
    assert teams["LAD"].cells[0].two_start is True
    # NYY pitches day 0 only -> day 1 is an off-day (None)
    assert teams["NYY"].cells[0] is not None and teams["NYY"].cells[1] is None
    assert teams["NYY"].cells[0].availability == "available"  # not rostered


def test_assemble_grid_skips_blank_team_rows():
    grid = _assemble_grid([[_row(1, ""), _row(2, "LAD")]], ["2026-06-21"], {}, set())
    assert [t.team for t in grid.teams] == ["LAD"]  # blank-team row dropped
