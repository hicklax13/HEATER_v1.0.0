"""DB-free tests for the probable-pitcher grid pure logic (band / availability /
cell mapping / grid assembly) with synthetic board rows + a fake roster map."""

from api.services.schedule_service import (
    _assemble_grid,
    _assemble_hitter_grid,
    _availability,
    _band,
    _to_cell,
    _to_hitter_cell,
)


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


def _sp_row(pid, team, opponent, **kw):
    """A synthetic build_stream_board row (the STARTER's row)."""
    base = {
        "player_id": pid,
        "player_name": f"SP{pid}",
        "mlb_id": 600000 + pid,
        "team": team,  # the starter's team
        "opponent": opponent,  # the batting team this starter faces
        "throws": "R",
        "is_home": True,  # the STARTER is home -> batting team is away
        "park_factor": 1.0,
        "opp_wrc_plus": 100.0,
        "opp_k_pct": 22.0,
        "status": "PROBABLE",
        "confidence": "HIGH",
    }
    base.update(kw)
    return base


def test_to_hitter_cell_remaps_to_batting_team_and_inverts_home():
    # SP on LAD (home) vs SF -> the cell belongs to SF, and SF is AWAY.
    batting_team, cell = _to_hitter_cell(_sp_row(1, "LAD", "SF"), {"xfip": 5.2})
    assert batting_team == "SF"
    assert cell.is_home is False  # batting team away (SP was home)
    assert cell.opponent == "LAD"  # the team SF plays = the SP's team
    assert cell.opp_sp is not None and cell.opp_sp.mlb_id == 600001
    assert cell.opp_sp_throws == "R"
    assert 0.0 <= cell.difficulty <= 100.0
    assert cell.band in {"easy", "medium", "tough"}


def test_to_hitter_cell_weak_pitcher_scores_higher_than_ace():
    _, easy = _to_hitter_cell(_sp_row(1, "LAD", "SF"), {"k_bb_pct": 0.03, "xfip": 5.4, "csw_pct": 0.24})
    _, tough = _to_hitter_cell(_sp_row(2, "LAD", "SF"), {"k_bb_pct": 0.27, "xfip": 2.7, "csw_pct": 0.34})
    assert easy.difficulty > tough.difficulty


def test_assemble_hitter_grid_offday_totals_rank_and_yours():
    dates = ["2026-06-21", "2026-06-22"]
    # Day 0: SP(NYY) faces BOS (R), SP(LAD) faces SF (L-hander).
    # Day 1: SP(NYY) faces BOS again (R). SF has an off day on day 1.
    boards = [
        [_sp_row(1, "NYY", "BOS", throws="R"), _sp_row(2, "LAD", "SF", throws="L")],
        [_sp_row(1, "NYY", "BOS", throws="R")],
    ]
    pool = {600001: {"k_bb_pct": 0.05, "xfip": 5.0}, 600002: {"k_bb_pct": 0.28, "xfip": 2.6}}
    grid = _assemble_hitter_grid(boards, dates, pool, user_hitter_teams={"BOS"})
    assert grid.days == dates
    rows = {r.team: r for r in grid.teams}
    assert set(rows) == {"BOS", "SF"}
    # BOS bats both days (vs a weak RHP) -> 2 games, both vs RHP, none LHP.
    assert rows["BOS"].totals.games == 2
    assert rows["BOS"].totals.vs_rhp == 2 and rows["BOS"].totals.vs_lhp == 0
    # SF bats day 0 only -> off-day cell is None.
    assert rows["SF"].cells[1] is None and rows["SF"].totals.games == 1
    assert rows["SF"].totals.vs_lhp == 1
    # BOS (weak RHP both days) is an easier schedule than SF (ace LHP) -> rank 1.
    assert rows["BOS"].matchups_rank == 1 and rows["SF"].matchups_rank == 2
    # Yours tag: viewer rosters >=1 BOS hitter.
    assert rows["BOS"].availability == "yours" and rows["SF"].availability == "other"


def test_assemble_hitter_grid_skips_blank_batting_team():
    grid = _assemble_hitter_grid([[_sp_row(1, "LAD", "")]], ["2026-06-21"], {}, set())
    assert grid.teams == []
