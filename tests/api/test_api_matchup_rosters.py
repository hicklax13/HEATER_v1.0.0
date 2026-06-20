"""Matchup roster tables: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd
import pytest

from api.services.matchup_service import (
    MatchupService,
    _cat_win,
    _date_tabs,
    _fmt_hitter_stats,
    _fmt_pitcher_stats,
    _game_state,
    _pair_rows,
    _to_match_player,
)

# ---------------------------------------------------------------------------
# Helpers for the roster-build tests
# ---------------------------------------------------------------------------


def _make_pool(*rows):
    """Build a minimal pool DataFrame."""
    defaults = dict(
        player_id=0,
        name="",
        positions="",
        mlb_id=0,
        team="",
        is_hitter=True,
        h=0,
        ab=0,
        r=0,
        hr=0,
        rbi=0,
        sb=0,
        avg=0.0,
        obp=0.0,
        ip=0.0,
        w=0,
        l=0,
        sv=0,
        k=0,
        era=0.0,
        whip=0.0,
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_roster(*rows):
    """Build a minimal league_rosters DataFrame."""
    defaults = dict(
        team_name="Team Hickey",
        player_id=0,
        roster_slot="",
        selected_position="",
        status="active",
        editorial_team_abbr="",
    )
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _run_build(roster_df, pool_df=None):
    """Run _build_roster_tables with mocked I/O (no DB/Yahoo/schedule calls).

    _build_roster_tables imports src.* inside the function body, so we patch
    those modules in sys.modules to intercept the inline `import` calls.
    """
    import sys
    import types
    import unittest.mock as mock

    if pool_df is None:
        pool_df = pd.DataFrame()

    mock_yds = mock.MagicMock()
    mock_yds.get_rosters.return_value = roster_df

    # Patch src.database.load_player_pool via a fake module inserted into sys.modules.
    fake_db = types.ModuleType("src.database")
    fake_db.load_player_pool = mock.MagicMock(return_value=pool_df)

    fake_game_day = types.ModuleType("src.game_day")
    fake_game_day.get_target_game_date = mock.MagicMock(return_value="2026-06-19")
    fake_game_day.FINAL_GAME_STATUSES = {"final", "game over"}
    fake_game_day.LOCKED_GAME_STATUSES = {"in progress", "live"}

    fake_statsapi = types.ModuleType("statsapi")
    fake_statsapi.schedule = mock.MagicMock(return_value=[])

    # Build a minimal fake for src.valuation (needs TEAM_NAME_TO_ABBR).
    fake_valuation = sys.modules.get("src.valuation") or types.ModuleType("src.valuation")
    if not hasattr(fake_valuation, "TEAM_NAME_TO_ABBR"):
        fake_valuation.TEAM_NAME_TO_ABBR = {}

    overrides = {
        "src.database": fake_db,
        "src.game_day": fake_game_day,
        "statsapi": fake_statsapi,
        "src.valuation": fake_valuation,
    }
    orig = {k: sys.modules.get(k) for k in overrides}
    sys.modules.update(overrides)
    try:
        hitters, pitchers, date_tabs, h_cols, p_cols, _ht, _pt = MatchupService._build_roster_tables(
            "Team Hickey", "Rivals", mock_yds, week=7
        )
    finally:
        for k, v in orig.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return hitters, pitchers, date_tabs, h_cols, p_cols


def test_fmt_hitter_stats():
    row = {"h": 120, "ab": 410, "r": 70, "hr": 24, "rbi": 80, "sb": 12, "avg": 0.293, "obp": 0.371}
    assert _fmt_hitter_stats(row) == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_fmt_pitcher_stats():
    row = {"ip": 180.0, "w": 14, "l": 7, "sv": 0, "k": 200, "era": 3.21, "whip": 1.05}
    assert _fmt_pitcher_stats(row) == ["180.0", "14", "7", "0", "200", "3.21", "1.05"]


def test_fmt_stats_nan_safe():
    assert _fmt_hitter_stats({"h": float("nan"), "ab": None})[0] == "0/0"


def test_game_state_maps_status():
    sched = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "Final"}]
    state, status = _game_state("NYY", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})
    assert state == "final"
    sched2 = [{"home_name": "New York Yankees", "away_name": "Boston Red Sox", "status": "In Progress"}]
    assert _game_state("BOS", sched2, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "live"
    # no game today → none
    assert _game_state("LAD", sched, {"NYY": "New York Yankees", "BOS": "Boston Red Sox"})[0] == "none"


def test_cat_win_respects_inverse():
    assert _cat_win(5.0, 3.0, inverse=False) == "you"  # higher wins
    assert _cat_win(5.0, 3.0, inverse=True) == "opp"  # lower wins (ERA/WHIP/L)
    assert _cat_win(3.0, 3.0, inverse=False) == ""  # tie


def test_date_tabs_starts_live_totals():
    tabs = _date_tabs(7)
    assert tabs[0] == "Live"
    assert tabs[1] == "Totals"
    assert len(tabs) >= 3


def test_to_match_player_builds_ref_and_stats():
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "positions": "OF",
                "mlb_id": 592450,
                "team": "NYY",
                "is_hitter": True,
                "h": 120,
                "ab": 410,
                "r": 70,
                "hr": 24,
                "rbi": 80,
                "sb": 12,
                "avg": 0.293,
                "obp": 0.371,
            }
        ]
    )
    mp = _to_match_player(1, "OF", pool, hitter=True, state="final", status="Final")
    assert mp.player.mlb_id == 592450
    assert mp.player.team_id == 147
    assert mp.pos == "OF"
    assert mp.state == "final"
    assert mp.stats == ["120/410", "70", "24", "80", "12", ".293", ".371"]


def test_pair_rows_zips_sides_and_pads():
    import pandas as pd

    pool = pd.DataFrame(
        [
            {"player_id": i, "name": f"P{i}", "positions": "OF", "mlb_id": i, "team": "NYY", "is_hitter": True}
            for i in (1, 2, 3)
        ]
    )
    you = [_to_match_player(1, "OF", pool, True, "none", ""), _to_match_player(2, "OF", pool, True, "none", "")]
    opp = [_to_match_player(3, "OF", pool, True, "none", "")]
    rows = _pair_rows(you, opp, ["OF", "OF"])
    assert len(rows) == 2
    assert rows[0].you is not None and rows[0].opp is not None
    assert rows[1].you is not None and rows[1].opp is None  # opp padded


def test_matchup_endpoint_includes_roster_tables():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.matchup import MatchPlayer, MatchupResponse, RosterRow
    from api.deps import get_matchup_service
    from api.main import create_app

    class _Fake:
        def get_matchup(self, team_name):
            mp = MatchPlayer(
                player=PlayerRef(id=1, mlb_id=592450, name="Judge", positions="OF", team_abbr="NYY", team_id=147),
                pos="OF",
                state="final",
                stats=["1/4", "1", "0", "0", "0", ".250", ".300"],
            )
            return MatchupResponse(
                team_name=team_name,
                opponent="Rivals",
                week=7,
                hitter_columns=["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"],
                pitcher_columns=["IP", "W", "L", "SV", "K", "ERA", "WHIP"],
                date_tabs=["Live", "Totals"],
                hitters=[RosterRow(slot="OF", you=mp, opp=None)],
            )

    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/matchup?team_name=Team+Hickey").json()
        assert body["hitter_columns"][0] == "H/AB"
        assert body["hitters"][0]["you"]["player"]["mlb_id"] == 592450
        assert body["hitters"][0]["you"]["stats"][0] == "1/4"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Bug-fix tests (must FAIL before the fix, PASS after)
# ---------------------------------------------------------------------------


def test_swingman_sp_rp_lands_in_pitcher_table():
    """Bug (a)/(c): swingman with roster_slot='SP,RP' and is_hitter=False must
    land in the PITCHER table, not the hitter table.  Before the fix it was
    classified as a hitter because _is_pitcher_slot('SP,RP') was False
    (the slot contained a comma so none of the exact matches hit).
    Classification MUST use is_hitter from the pool, NOT the slot string.
    """
    pool = _make_pool(
        dict(player_id=10, name="Shohei Ohtani", is_hitter=False, team="LAA"),
    )
    roster = _make_roster(
        dict(
            team_name="Team Hickey",
            player_id=10,
            roster_slot="SP,RP",  # eligible positions — the old buggy source
            selected_position="SP",  # the assigned slot
            status="active",
        ),
    )
    hitters, pitchers, *_ = _run_build(roster, pool)
    # Must be in pitchers, NOT in hitters
    h_ids = [r.you.player.id for r in hitters if r.you] if hitters else []
    p_ids = [r.you.player.id for r in pitchers if r.you] if pitchers else []
    assert 10 not in h_ids, "swingman wrongly landed in hitter table"
    assert 10 in p_ids, "swingman must be in pitcher table"


def test_bench_player_included_with_bn_slot():
    """Bug (b): BN players were dropped (continue on BN). They must appear."""
    pool = _make_pool(
        dict(player_id=20, name="Bench Hitter", is_hitter=True, team="NYY"),
    )
    roster = _make_roster(
        dict(
            team_name="Team Hickey",
            player_id=20,
            roster_slot="1B,3B",  # eligible positions
            selected_position="BN",  # assigned slot → bench
            status="active",
        ),
    )
    hitters, pitchers, *_ = _run_build(roster, pool)
    h_ids = [r.you.player.id for r in hitters if r.you] if hitters else []
    assert 20 in h_ids, "BN player must be included in hitter table"
    # Slot/pos in the row should be 'BN', not the eligible-positions string
    matching = [r for r in hitters if r.you and r.you.player.id == 20]
    assert matching, "no row found for BN hitter"
    assert matching[0].you.pos == "BN", f"expected pos='BN', got {matching[0].you.pos!r}"


def test_selected_position_used_as_slot_not_roster_slot():
    """Bug (b) slot label: emitted slot must come from selected_position, not roster_slot."""
    pool = _make_pool(
        dict(player_id=30, name="SS Guy", is_hitter=True, team="BOS"),
    )
    roster = _make_roster(
        dict(
            team_name="Team Hickey",
            player_id=30,
            roster_slot="2B,SS",  # eligible positions string
            selected_position="SS",  # what manager actually slotted them as
            status="active",
        ),
    )
    hitters, pitchers, *_ = _run_build(roster, pool)
    matching = [r for r in hitters if r.you and r.you.player.id == 30]
    assert matching, "SS player not found in hitters"
    # slot must be 'SS', not the comma-separated eligible string
    assert matching[0].slot == "SS", f"expected slot='SS', got {matching[0].slot!r}"
    assert matching[0].you.pos == "SS", f"expected pos='SS', got {matching[0].you.pos!r}"


def test_il_player_included_with_correct_slot():
    """Bug (b)/(slot): IL pitcher must appear AND have slot='IL10' (from selected_position).
    Before the fix, selected_position='IL10' was ignored and roster_slot='SP' was used,
    so the slot was wrong even if the player appeared by accident.
    After the fix, selected_position is used, so slot='IL10'.
    """
    pool = _make_pool(
        dict(player_id=40, name="IL Pitcher", is_hitter=False, team="HOU"),
    )
    roster = _make_roster(
        dict(
            team_name="Team Hickey",
            player_id=40,
            roster_slot="SP",  # eligible positions — must NOT be used as slot
            selected_position="IL10",  # assigned IL slot — must be used
            status="IL10",
        ),
    )
    hitters, pitchers, *_ = _run_build(roster, pool)
    p_ids = [r.you.player.id for r in pitchers if r.you] if pitchers else []
    assert 40 in p_ids, "IL pitcher must be included in pitcher table"
    matching = [r for r in pitchers if r.you and r.you.player.id == 40]
    assert matching[0].you.pos == "IL10", f"expected pos='IL10' from selected_position, got {matching[0].you.pos!r}"


@pytest.mark.parametrize(
    "status,expected_badge",
    [
        ("IL10", "IL"),
        ("IL15", "IL"),
        ("IL60", "IL"),
        ("il10", "IL"),  # case-insensitive
        ("NA", "IL"),
        ("na", "IL"),
        ("DTD", "DTD"),
        ("Day-to-Day", "DTD"),
        ("active", None),
        ("Active", None),
        ("", None),
    ],
)
def test_badge_from_status(status, expected_badge):
    """Bug (d): badge must be populated from status, not always None."""
    pool = _make_pool(
        dict(player_id=50, name="Test Player", is_hitter=True, team="NYY"),
    )
    roster = _make_roster(
        dict(
            team_name="Team Hickey",
            player_id=50,
            roster_slot="OF",
            selected_position="OF",
            status=status,
        ),
    )
    hitters, pitchers, *_ = _run_build(roster, pool)
    all_players = [r.you for r in (hitters or []) if r.you] + [r.you for r in (pitchers or []) if r.you]
    matching = [mp for mp in all_players if mp.player.id == 50]
    assert matching, f"player not found in output for status={status!r}"
    assert matching[0].badge == expected_badge, (
        f"status={status!r}: expected badge={expected_badge!r}, got {matching[0].badge!r}"
    )
