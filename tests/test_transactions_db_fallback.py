"""Read-only members must see league ownership trends (adds/drops) in Player
Databank. get_adds_drops_map gated on is_connected() (always False for members)
AND get_transactions had an empty DB fallback -> members saw nothing. Wire a
real transactions DB fallback (joined to player names) + drop the gate.
2026-06-03 follow-up to the Data Freshness goal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_connection, load_transactions
from src.yahoo_data_service import YahooDataService, _get_state_store


def _seed_player_and_txn(player_id: int, name: str, txn_type: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO players (player_id, name, positions) VALUES (?, ?, 'OF')",
            (player_id, name),
        )
        conn.execute(
            "INSERT INTO transactions (player_id, type, team_from, team_to, timestamp) VALUES (?, ?, ?, ?, ?)",
            (player_id, txn_type, "", "Team Hickey", "2026-06-01"),
        )
        conn.commit()
    finally:
        conn.close()


def test_load_transactions_joins_player_names():
    """The DB stores player_id; the loader joins players to surface player_name
    (the column get_adds_drops_map consumes)."""
    _seed_player_and_txn(990001, "Test Adder", "add")
    df = load_transactions()
    assert not df.empty
    assert "player_name" in df.columns
    assert "type" in df.columns
    row = df[df["player_name"] == "Test Adder"]
    assert not row.empty
    assert row.iloc[0]["type"] == "add"


def test_get_transactions_db_fallback_when_disconnected():
    """A read-only member (no live client) still gets transactions from SQLite."""
    _seed_player_and_txn(990002, "Fallback Guy", "drop")
    _get_state_store().clear()  # force the DB-fallback path, not session cache
    yds = YahooDataService(yahoo_client=None)  # not connected
    txns = yds.get_transactions()
    assert not txns.empty
    assert "Fallback Guy" in txns["player_name"].values
