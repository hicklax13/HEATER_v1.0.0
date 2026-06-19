"""Per-site PlayerRef enrichment tests (M0 slice 1).

Each test injects a fabricated pool row / grid dict and asserts the service's
mapping function fills mlb_id / team_abbr / team_id.
"""

from __future__ import annotations

import pandas as pd

from api.services.closers_service import CloserService
from api.services.databank_service import DatabankService
from api.services.streaming_service import StreamingService
from api.services.trade_finder_service import _build_player_refs as finder_refs
from api.services.trade_service import _build_player_refs as trade_refs


def _pool_one(pid: int, mlb_id: int, team: str) -> pd.DataFrame:
    return pd.DataFrame([{"player_id": pid, "name": "Test Player", "positions": "OF", "mlb_id": mlb_id, "team": team}])


def test_trade_build_player_refs_enriches_from_pool():
    refs = trade_refs([5], _pool_one(5, 605141, "LAD"))
    assert len(refs) == 1
    assert refs[0].mlb_id == 605141
    assert refs[0].team_abbr == "LAD"
    assert refs[0].team_id == 119


def test_trade_finder_build_player_refs_enriches_from_pool():
    refs = finder_refs([8], _pool_one(8, 660271, "LAA"))
    assert refs[0].mlb_id == 660271
    assert refs[0].team_abbr == "LAA"
    assert refs[0].team_id == 108


def test_databank_build_ref_enriches_from_pool():
    ref = DatabankService._build_ref(7, _pool_one(7, 456789, "BOS"))
    assert ref.mlb_id == 456789
    assert ref.team_abbr == "BOS"
    assert ref.team_id == 111


def test_closers_to_entry_enriches_closer_ref():
    row = {
        "team": "NYY",
        "closer_name": "Devin Williams",
        "mlb_id": 642207,
        "job_security": 0.8,
        "setup_names": [],
    }
    entry = CloserService._to_entry(row)
    assert entry.closer is not None
    assert entry.closer.mlb_id == 642207
    assert entry.closer.team_abbr == "NYY"
    assert entry.closer.team_id == 147


def test_streaming_to_candidate_enriches_team_only():
    row = {"player_id": 9, "player_name": "Tarik Skubal", "team": "DET", "stream_score": 80.0}
    cand = StreamingService._to_candidate(row)
    assert cand.player.team_abbr == "DET"
    assert cand.player.team_id == 116
    assert cand.player.mlb_id is None  # stream board carries no mlb_id


def test_enrichment_survives_mixed_null_pool():
    # A row with mlb_id=None forces the mlb_id column to float64 (the real pool shape),
    # so the populated row's mlb_id arrives as 592450.0 and must coerce to int 592450.
    # The null row must degrade cleanly (no "nan" strings).
    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "Real", "positions": "OF", "mlb_id": 592450, "team": "NYY"},
            {"player_id": 2, "name": "NoIds", "positions": "SP", "mlb_id": None, "team": None},
        ]
    )
    refs = trade_refs([1, 2], pool)
    by_id = {r.id: r for r in refs}
    assert by_id[1].mlb_id == 592450  # float64 592450.0 -> int
    assert by_id[1].team_abbr == "NYY"
    assert by_id[1].team_id == 147
    assert by_id[2].mlb_id is None
    assert by_id[2].team_abbr is None
    assert by_id[2].team_id is None


def test_draft_to_recs_enriches_from_engine_results():
    import pandas as pd

    from api.services.draft_service import DraftService

    results = pd.DataFrame(
        [
            {
                "player_id": 11,
                "player_name": "Corbin Carroll",
                "positions": "OF",
                "mlb_id": 682998,
                "team": "ARI",
                "overall_rank": 1,
                "composite_value": 95.0,
                "mc_mean_sgp": 3.2,
                "confidence_level": "high",
                "buy_fair_avoid": "buy",
            }
        ]
    )
    recs = DraftService._to_recs(results)
    assert recs[0].player.mlb_id == 682998
    assert recs[0].player.team_abbr == "ARI"
    assert recs[0].player.team_id == 109


def test_compare_enriches_from_pool(monkeypatch):
    import pandas as pd

    import src.database as db
    from api.services.compare_service import CompareService

    fake_pool = pd.DataFrame(
        [
            {
                "player_id": 3,
                "name": "Mookie Betts",
                "positions": "OF",
                "mlb_id": 605141,
                "team": "LAD",
                "r": 90,
                "hr": 20,
                "rbi": 70,
                "sb": 10,
                "avg": 0.300,
                "obp": 0.380,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
            }
        ]
    )
    monkeypatch.setattr(db, "load_player_pool", lambda: fake_pool)
    resp = CompareService().compare([3])
    assert resp.players[0].player.mlb_id == 605141
    assert resp.players[0].player.team_abbr == "LAD"
    assert resp.players[0].player.team_id == 119


def test_leaders_to_leader_row_enriches():
    from api.services.leaders_service import LeadersService

    row = {"player_id": 21, "name": "Bobby Witt Jr.", "positions": "SS", "mlb_id": 677951, "team": "KC", "hr": 24}
    leader_row = LeadersService._to_leader_row(1, row, "hr")
    assert leader_row.rank == 1
    assert leader_row.value == 24.0
    assert leader_row.player.mlb_id == 677951
    assert leader_row.player.team_abbr == "KC"
    assert leader_row.player.team_id == 118


def test_lineup_to_slots_enriches_via_pool():
    import pandas as pd

    from api.services.lineup_service import LineupService

    pool = pd.DataFrame([{"player_id": 31, "name": "X", "positions": "OF", "mlb_id": 660271, "team": "LAA"}])
    result = {
        "lineup": [
            {"slot": "OF", "player_id": 31, "player_name": "Shohei Ohtani", "positions": "OF", "action": "START"}
        ]
    }
    slots = LineupService._to_slots(result, pool)
    assert slots[0].player.mlb_id == 660271
    assert slots[0].player.team_abbr == "LAA"
    assert slots[0].player.team_id == 108
    assert slots[0].player.name == "Shohei Ohtani"  # engine name preferred


def test_lineup_to_slots_without_pool_still_works():
    from api.services.lineup_service import LineupService

    result = {"lineup": [{"slot": "OF", "player_id": 31, "player_name": "X", "positions": "OF", "action": "START"}]}
    slots = LineupService._to_slots(result)  # pool defaults to None
    assert slots[0].player.mlb_id is None
    assert slots[0].player.id == 31


def test_fa_to_rec_enriches_and_fixes_id_key():
    import pandas as pd

    from api.services.fa_service import FreeAgentService

    pool = pd.DataFrame(
        [
            {"player_id": 41, "name": "Add Guy", "positions": "2B", "mlb_id": 700000, "team": "NYM"},
            {"player_id": 42, "name": "Drop Guy", "positions": "SP", "mlb_id": 800000, "team": "SF"},
        ]
    )
    move = {
        "add_id": 41,
        "add_name": "Add Guy",
        "add_positions": "2B",
        "drop_id": 42,
        "drop_name": "Drop Guy",
        "drop_positions": "SP",
        "net_sgp_delta": 1.5,
    }
    rec = FreeAgentService._to_rec(move, pool)
    assert rec.add.id == 41  # was always 0 before the id-key fix
    assert rec.add.mlb_id == 700000
    assert rec.add.team_abbr == "NYM"
    assert rec.add.team_id == 121
    assert rec.drop is not None
    assert rec.drop.id == 42
    assert rec.drop.mlb_id == 800000
    assert rec.drop.team_id == 137
