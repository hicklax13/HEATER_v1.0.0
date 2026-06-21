"""Closer 'Saves Finder' field enrichment — DB-free unit test of _to_entry."""

import pandas as pd

from api.services.closers_service import CloserService

_ROW = {
    "team": "NYY",
    "closer_name": "Devin Williams",
    "setup_names": ["Luke Weaver"],
    "job_security": 0.82,
    "security_color": "green",
    "projected_sv": 34.0,
    "era": 2.10,
    "whip": 0.95,
    "mlb_id": 642207,
}


def test_entry_exposes_actionable_numeric_fields():
    entry = CloserService._to_entry(_ROW, None)
    assert entry.job_security == 0.82
    assert entry.security_color == "green"
    assert entry.projected_sv == 34.0
    assert entry.era == 2.10
    assert entry.whip == 0.95
    # backward-compatible: the string confidence label still derives from job_security
    assert entry.confidence == "Firm"


def test_handcuff_resolved_from_pool_when_unique():
    pool = pd.DataFrame([{"player_id": 500, "name": "Luke Weaver", "mlb_id": 608648, "team": "NYY", "positions": "RP"}])
    entry = CloserService._to_entry(_ROW, pool)
    assert len(entry.handcuffs) == 1
    assert entry.handcuffs[0].name == "Luke Weaver"
    assert entry.handcuffs[0].mlb_id == 608648  # rich, clickable
    assert entry.handcuffs[0].id == 500


def test_handcuff_falls_back_to_name_only_when_ambiguous():
    # Two pool rows with the same name → can't safely pick one (Muncy-DNA guard).
    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "Luke Weaver", "mlb_id": 608648, "team": "NYY", "positions": "RP"},
            {"player_id": 2, "name": "Luke Weaver", "mlb_id": 999999, "team": "SEA", "positions": "RP"},
        ]
    )
    entry = CloserService._to_entry(_ROW, pool)
    assert entry.handcuffs[0].name == "Luke Weaver"
    assert entry.handcuffs[0].mlb_id is None  # ambiguous → name-only, not a guessed id
    assert entry.handcuffs[0].id == 0


def test_handcuff_name_only_when_no_pool():
    entry = CloserService._to_entry(_ROW, None)
    assert entry.handcuffs[0].name == "Luke Weaver"
    assert entry.handcuffs[0].id == 0
