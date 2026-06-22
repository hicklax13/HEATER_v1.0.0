"""Type-design contract hardening tests (Task U4).

Each Literal field is tested to:
  - REJECT unknown values (pydantic ValidationError)
  - ACCEPT all valid members of the Literal set
Each structured type (Record, StatItem, ToolTraceEntry) is tested for construction.
Mapper coercion is tested end-to-end: feeding a raw engine row with a WEIRD
status/confidence must produce status=="" and confidence=="" (graceful, no raise).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# common.py — Record
# ---------------------------------------------------------------------------


def test_record_defaults():
    from api.contracts.common import Record

    r = Record()
    assert r.wins == 0 and r.losses == 0 and r.ties == 0


def test_record_construction():
    from api.contracts.common import Record

    r = Record(wins=5, losses=3, ties=1)
    assert r.wins == 5 and r.losses == 3 and r.ties == 1


# ---------------------------------------------------------------------------
# streaming.py — Literal status / confidence + BudgetStrip ip_pace
# ---------------------------------------------------------------------------


def test_stream_candidate_status_accepts_known():
    from api.contracts.common import PlayerRef
    from api.contracts.streaming import StreamCandidate

    p = PlayerRef(id=1, name="X", positions="SP")
    for val in ("", "PROBABLE", "LOCKED", "FINAL", "OPEN"):
        c = StreamCandidate(player=p, status=val)
        assert c.status == val


def test_stream_candidate_status_rejects_unknown():
    from api.contracts.common import PlayerRef
    from api.contracts.streaming import StreamCandidate

    p = PlayerRef(id=1, name="X", positions="SP")
    with pytest.raises(ValidationError):
        StreamCandidate(player=p, status="WEIRD")


def test_stream_candidate_confidence_accepts_known():
    from api.contracts.common import PlayerRef
    from api.contracts.streaming import StreamCandidate

    p = PlayerRef(id=1, name="X", positions="SP")
    for val in ("", "HIGH", "MEDIUM", "LOW"):
        c = StreamCandidate(player=p, confidence=val)
        assert c.confidence == val


def test_stream_candidate_confidence_rejects_unknown():
    from api.contracts.common import PlayerRef
    from api.contracts.streaming import StreamCandidate

    p = PlayerRef(id=1, name="X", positions="SP")
    with pytest.raises(ValidationError):
        StreamCandidate(player=p, confidence="EXTREME")


def test_budget_strip_ip_pace_none_by_default():
    from api.contracts.streaming import BudgetStrip

    b = BudgetStrip()
    assert b.ip_pace is None


def test_budget_strip_ip_pace_accepts_float():
    from api.contracts.streaming import BudgetStrip

    b = BudgetStrip(ip_pace=42.5)
    assert b.ip_pace == 42.5


# ---------------------------------------------------------------------------
# matchup.py — Literal win, StatItem stats, Record on TeamSide/SideTotals
# ---------------------------------------------------------------------------


def test_matchup_category_win_accepts_known():
    from api.contracts.matchup import MatchupCategory

    for val in ("", "you", "opp"):
        mc = MatchupCategory(cat="HR", you=10.0, opp=8.0, win_prob=0.8, win=val)
        assert mc.win == val


def test_matchup_category_win_rejects_unknown():
    from api.contracts.matchup import MatchupCategory

    with pytest.raises(ValidationError):
        MatchupCategory(cat="HR", you=10.0, opp=8.0, win_prob=0.8, win="tie")


def test_match_player_stats_accepts_stat_items():
    from api.contracts.common import PlayerRef, StatItem
    from api.contracts.matchup import MatchPlayer

    p = PlayerRef(id=1, name="X", positions="OF")
    stats = [StatItem(label="HR", value="18"), StatItem(label="AVG", value=".300")]
    mp = MatchPlayer(player=p, stats=stats)
    assert len(mp.stats) == 2
    assert mp.stats[0].label == "HR"
    assert mp.stats[0].value == "18"


def test_match_player_stats_rejects_plain_strings():
    """list[str] is NOT accepted — Pydantic must validate each item as StatItem."""
    from api.contracts.common import PlayerRef
    from api.contracts.matchup import MatchPlayer

    p = PlayerRef(id=1, name="X", positions="OF")
    with pytest.raises(ValidationError):
        MatchPlayer(player=p, stats=["1/4", "1", "0"])


def test_team_side_record_wlt_additive():
    """record (display str) is kept; record_wlt is new and optional."""
    from api.contracts.common import Record
    from api.contracts.matchup import TeamSide

    ts = TeamSide(name="Team A", record="4-7-1 · 8th", record_wlt=Record(wins=4, losses=7, ties=1))
    assert ts.record == "4-7-1 · 8th"
    assert ts.record_wlt is not None
    assert ts.record_wlt.wins == 4


def test_team_side_record_wlt_defaults_none():
    from api.contracts.matchup import TeamSide

    ts = TeamSide(name="Team A")
    assert ts.record_wlt is None


def test_side_totals_accepts_stat_items():
    from api.contracts.common import StatItem
    from api.contracts.matchup import SideTotals

    items = [StatItem(label="HR", value="10"), StatItem(label="AVG", value=".280")]
    st = SideTotals(you=items, opp=[])
    assert st.you[0].label == "HR"


def test_side_totals_rejects_strings():
    from api.contracts.matchup import SideTotals

    with pytest.raises(ValidationError):
        SideTotals(you=["150/600", "90"])


# ---------------------------------------------------------------------------
# my_team.py — Literal trend, tag, StatusChip.status, OpsCard.status
# ---------------------------------------------------------------------------


def test_mover_trend_accepts_known():
    from api.contracts.common import PlayerRef
    from api.contracts.my_team import Mover

    p = PlayerRef(id=1, name="X", positions="OF")
    for val in ("up", "down", "flat"):
        m = Mover(player=p, trend=val)
        assert m.trend == val


def test_mover_trend_rejects_unknown():
    from api.contracts.common import PlayerRef
    from api.contracts.my_team import Mover

    p = PlayerRef(id=1, name="X", positions="OF")
    with pytest.raises(ValidationError):
        Mover(player=p, trend="rising")


def test_mover_tag_accepts_known():
    from api.contracts.common import PlayerRef
    from api.contracts.my_team import Mover

    p = PlayerRef(id=1, name="X", positions="OF")
    for val in ("hot", "cold", ""):
        m = Mover(player=p, tag=val)
        assert m.tag == val


def test_mover_tag_rejects_unknown():
    from api.contracts.common import PlayerRef
    from api.contracts.my_team import Mover

    p = PlayerRef(id=1, name="X", positions="OF")
    with pytest.raises(ValidationError):
        Mover(player=p, tag="WARM")


def test_status_chip_status_accepts_known():
    from api.contracts.my_team import StatusChip

    for val in ("ok", "warn", "info"):
        sc = StatusChip(label="IL", value=2, status=val)
        assert sc.status == val


def test_status_chip_status_rejects_unknown():
    from api.contracts.my_team import StatusChip

    with pytest.raises(ValidationError):
        StatusChip(label="IL", value=2, status="error")


def test_ops_card_status_accepts_known():
    from api.contracts.my_team import OpsCard

    for val in ("ok", "warn", "danger"):
        oc = OpsCard(key="ip_pace", label="IP", value=40.0, total=54.0, status=val)
        assert oc.status == val


def test_ops_card_status_rejects_unknown():
    from api.contracts.my_team import OpsCard

    with pytest.raises(ValidationError):
        OpsCard(key="ip_pace", label="IP", value=40.0, total=54.0, status="critical")


# ---------------------------------------------------------------------------
# playoff.py — projected_record_wlt additive field
# ---------------------------------------------------------------------------


def test_playoff_team_record_wlt_additive():
    from api.contracts.common import Record
    from api.contracts.playoff import PlayoffTeam

    pt = PlayoffTeam(
        team="Team A",
        projected_record="10-5-0",
        projected_record_wlt=Record(wins=10, losses=5, ties=0),
    )
    assert pt.projected_record == "10-5-0"
    assert pt.projected_record_wlt is not None
    assert pt.projected_record_wlt.wins == 10


def test_playoff_team_record_wlt_defaults_none():
    from api.contracts.playoff import PlayoffTeam

    pt = PlayoffTeam(team="Team A")
    assert pt.projected_record_wlt is None


# ---------------------------------------------------------------------------
# chat.py — ToolTraceEntry typed
# ---------------------------------------------------------------------------


def test_tool_trace_entry_defaults():
    from api.contracts.chat import ToolTraceEntry

    t = ToolTraceEntry()
    assert t.name == ""
    assert t.args == {}


def test_tool_trace_entry_construction():
    from api.contracts.chat import ToolTraceEntry

    t = ToolTraceEntry(name="web_search", args={"query": "fantasy baseball"})
    assert t.name == "web_search"
    assert t.args["query"] == "fantasy baseball"


def test_chat_send_response_tool_trace_typed():
    """tool_trace must accept list[ToolTraceEntry], not raw list[dict]."""
    from api.contracts.chat import ChatSendResponse, ToolTraceEntry

    resp = ChatSendResponse(
        content="hello",
        conversation_id=None,
        tool_trace=[ToolTraceEntry(name="search", args={"q": "x"})],
    )
    assert resp.tool_trace[0].name == "search"


def test_chat_send_response_tool_trace_coerces_dicts():
    """Pydantic v2 coerces dicts → ToolTraceEntry (name/args extracted)."""
    from api.contracts.chat import ChatSendResponse

    resp = ChatSendResponse(
        content="hello",
        conversation_id=None,
        tool_trace=[{"name": "web_search", "args": {"q": "HR leaders"}}],
    )
    assert resp.tool_trace[0].name == "web_search"
    assert resp.tool_trace[0].args == {"q": "HR leaders"}


# ---------------------------------------------------------------------------
# Mapper coercion: streaming service degrades unknown status/confidence
# ---------------------------------------------------------------------------


def test_streaming_mapper_coerces_unknown_status_and_confidence():
    """StreamingService._to_candidate with a WEIRD status/confidence must not raise
    and must degrade to "" for both fields."""
    from api.services.streaming_service import StreamingService

    row = {
        "player_id": 1,
        "mlb_id": 1,
        "player_name": "Tester",
        "team": "NYY",
        "opponent": "BOS",
        "is_home": True,
        "stream_score": 75.0,
        "status": "WEIRD_STATUS",
        "confidence": "MEGA_HIGH",
        "num_starts": 1,
        "actionable": True,
        "net_sgp": 0.5,
        "opp_wrc_plus": 100.0,
        "opp_k_pct": 22.0,
        "park_factor": 1.0,
        "expected_ip": 6.0,
        "expected_k": 6.0,
        "expected_er": 2.0,
        "win_probability": 0.55,
        "percent_owned": 50.0,
        "risk_flags": [],
        "components": {},
    }
    cand = StreamingService._to_candidate(row, rank=1)
    assert cand.status == "", f"Expected '' got {cand.status!r}"
    assert cand.confidence == "", f"Expected '' got {cand.confidence!r}"


def test_streaming_mapper_preserves_valid_status_and_confidence():
    """Valid values pass through unchanged."""
    from api.services.streaming_service import StreamingService

    row = {
        "player_id": 2,
        "mlb_id": 2,
        "player_name": "Valid",
        "team": "DET",
        "opponent": "CWS",
        "is_home": False,
        "stream_score": 80.0,
        "status": "PROBABLE",
        "confidence": "HIGH",
        "num_starts": 1,
        "actionable": True,
        "net_sgp": 1.0,
        "opp_wrc_plus": 90.0,
        "opp_k_pct": 23.0,
        "park_factor": 0.97,
        "expected_ip": 6.0,
        "expected_k": 7.0,
        "expected_er": 2.0,
        "win_probability": 0.60,
        "percent_owned": 85.0,
        "risk_flags": [],
        "components": {},
    }
    cand = StreamingService._to_candidate(row, rank=2)
    assert cand.status == "PROBABLE"
    assert cand.confidence == "HIGH"


def test_streaming_mapper_coerces_empty_status_and_confidence():
    """Empty strings (the default) are valid and pass through."""
    from api.services.streaming_service import StreamingService

    row = {
        "player_id": 3,
        "mlb_id": 3,
        "player_name": "Empty",
        "team": "LAD",
        "opponent": "SF",
        "is_home": True,
        "stream_score": 70.0,
        "status": "",
        "confidence": "",
        "num_starts": 1,
        "actionable": True,
        "net_sgp": 0.3,
        "opp_wrc_plus": 95.0,
        "opp_k_pct": 21.0,
        "park_factor": 0.98,
        "expected_ip": 5.5,
        "expected_k": 5.0,
        "expected_er": 2.0,
        "win_probability": 0.52,
        "percent_owned": 60.0,
        "risk_flags": [],
        "components": {},
    }
    cand = StreamingService._to_candidate(row, rank=3)
    assert cand.status == ""
    assert cand.confidence == ""
