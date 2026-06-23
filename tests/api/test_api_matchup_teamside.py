"""Matchup-B (TeamSide + totals): helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.matchup_service import _aggregate_totals, _format_record


def test_format_record():
    assert _format_record(4, 7, 1, 8) == "4-7-1 · 8th"
    assert _format_record(1, 0, 0, 1) == "1-0-0 · 1st"
    assert _format_record(0, 0, 0, 2) == "0-0-0 · 2nd"
    assert _format_record(3, 3, 0, 3) == "3-3-0 · 3rd"
    assert _format_record(5, 5, 0, 11) == "5-5-0 · 11th"  # 11th not "11st"


def test_aggregate_totals_hitter_weighted_rates():
    rows = pd.DataFrame(
        [
            {"h": 100, "ab": 400, "r": 60, "hr": 20, "rbi": 70, "sb": 10, "bb": 40, "hbp": 4, "sf": 3},
            {"h": 50, "ab": 200, "r": 30, "hr": 10, "rbi": 35, "sb": 5, "bb": 20, "hbp": 1, "sf": 2},
        ]
    )
    out = _aggregate_totals(rows, hitter=True)
    # Returns list[StatItem] now — check values
    values = [s.value for s in out]
    # H/AB = 150/600 ; R/HR/RBI/SB summed ; AVG = 150/600 = .250 ; OBP = 215/670 = .321
    assert values[0] == "150/600"
    assert values[1:5] == ["90", "30", "105", "15"]
    assert values[5] == ".250"  # AVG
    assert values[6] == ".321"  # OBP = 215/670 = 0.3209 -> ".321"
    # labels are also checked
    assert out[0].label == "H/AB"


def test_aggregate_totals_pitcher_weighted_rates():
    rows = pd.DataFrame(
        [
            {"ip": 100.0, "w": 8, "l": 4, "sv": 0, "k": 110, "er": 35, "bb_allowed": 30, "h_allowed": 90},
            {"ip": 50.0, "w": 4, "l": 2, "sv": 0, "k": 55, "er": 20, "bb_allowed": 18, "h_allowed": 48},
        ]
    )
    out = _aggregate_totals(rows, hitter=False)
    values = [s.value for s in out]
    # IP=150 ; ERA = (55*9)/150 = 3.30 ; WHIP = (48+138)/150 = 1.24
    assert values[0] == "150.0"
    assert values[1:5] == ["12", "6", "0", "165"]
    assert values[5] == "3.30"  # ERA
    assert values[6] == "1.24"  # WHIP
    assert out[0].label == "IP"


def test_aggregate_totals_empty_and_zero_safe():
    out_empty = _aggregate_totals(pd.DataFrame(), hitter=True)
    assert out_empty[0].value == "0/0"
    z = _aggregate_totals(pd.DataFrame([{"ip": 0.0, "er": 5, "bb_allowed": 2, "h_allowed": 3}]), hitter=False)
    assert z[5].value == "0.00" and z[6].value == "0.00"  # divide-by-zero IP -> 0, not inf/NaN


def test_matchup_endpoint_includes_teamside_and_totals():
    from fastapi.testclient import TestClient

    from api.contracts.common import Record, StatItem
    from api.contracts.matchup import MatchupResponse, SideTotals, TeamSide
    from api.deps import get_matchup_service
    from api.main import create_app

    _H_COLS = ["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"]
    _P_COLS = ["IP", "W", "L", "SV", "K", "ERA", "WHIP"]

    class _Fake:
        def get_matchup(self, team_name):
            return MatchupResponse(
                team_name=team_name,
                opponent="Rivals",
                week=7,
                you=TeamSide(
                    name=team_name,
                    manager="Connor",
                    record="4-7-1 · 8th",
                    record_wlt=Record(wins=4, losses=7, ties=1),
                    score=5,
                ),
                opp=TeamSide(name="Rivals", manager="Sam", record="7-4-1 · 2nd", score=6),
                hitter_totals=SideTotals(
                    you=[
                        StatItem(label=col, value=v)
                        for col, v in zip(_H_COLS, ["150/600", "90", "30", "105", "15", ".250", ".321"])
                    ],
                    opp=[],
                ),
                pitcher_totals=SideTotals(
                    you=[
                        StatItem(label=col, value=v)
                        for col, v in zip(_P_COLS, ["150.0", "12", "6", "0", "165", "3.30", "1.24"])
                    ],
                    opp=[],
                ),
            )

    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/matchup?team_name=Team+Hickey").json()
        assert body["you"]["manager"] == "Connor"
        assert body["you"]["record"] == "4-7-1 · 8th"
        assert body["you"]["record_wlt"]["wins"] == 4
        assert body["opp"]["score"] == 6
        # totals are now list[{label, value}] dicts in JSON
        assert body["hitter_totals"]["you"][0]["label"] == "H/AB"
        assert body["hitter_totals"]["you"][0]["value"] == "150/600"
    finally:
        app.dependency_overrides.clear()
