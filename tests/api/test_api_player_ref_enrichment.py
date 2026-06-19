"""Per-site PlayerRef enrichment tests (M0 slice 1).

Each test injects a fabricated pool row / grid dict and asserts the service's
mapping function fills mlb_id / team_abbr / team_id.
"""

from __future__ import annotations

import pandas as pd

from api.services.databank_service import DatabankService
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
