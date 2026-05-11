"""Test BUG-022 fix: il_manager.detect_il_changes does not flag everyone on cold start."""

import pandas as pd


def _roster():
    return pd.DataFrame(
        [
            {"player_id": 1, "name": "Player A", "status": "IL10", "positions": "OF"},
            {"player_id": 2, "name": "Player B", "status": "IL15", "positions": "P"},
            {"player_id": 3, "name": "Player C", "status": "active", "positions": "1B"},
            {"player_id": 4, "name": "Player D", "status": "DTD", "positions": "SS"},
        ]
    )


def test_cold_start_emits_no_changes():
    """When last_known_status is None (cold start), no changes should be
    emitted. Caller cannot distinguish 'new IL transition' from 'pre-existing
    IL' without prior baseline. (BUG-022 fix.)"""
    from src.il_manager import detect_il_changes

    changes = detect_il_changes(_roster(), last_known_status=None)
    assert changes == [], (
        f"BUG-022: cold start should emit empty list (no baseline), but emitted {len(changes)} change(s)."
    )


def test_real_transition_emits_change():
    """When last_known_status shows a player was 'active' and now is 'IL10',
    that's a real transition — emit a change."""
    from src.il_manager import detect_il_changes

    last_known = {1: "active", 2: "IL15", 3: "active", 4: "DTD"}
    changes = detect_il_changes(_roster(), last_known_status=last_known)
    pids = [c["player_id"] for c in changes]
    assert pids == [1], f"Expected only pid 1 flagged; got {pids}"


def test_empty_dict_treated_as_explicit_empty_baseline():
    """Caller passing {} explicitly is treated as 'no prior IL state' —
    every current IL player IS a new transition."""
    from src.il_manager import detect_il_changes

    changes = detect_il_changes(_roster(), last_known_status={})
    pids = sorted(c["player_id"] for c in changes)
    assert pids == [1, 2, 4], f"Expected pids 1,2,4 flagged; got {pids}"
