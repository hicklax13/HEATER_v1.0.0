"""Bonus audit (10a): _bootstrap_contracts must report honest refresh_log status.

Audit finding (2026-04-18):
    The function writes refresh_log status='success' even when 0 players matched
    the BB-Ref contract-year list. UI shows green-success while no contract
    flags were actually persisted.

Required behavior:
    - 0 contract-year names returned (or 0 players matched)  → status='no_data'
    - >=1 player flagged                                      → status='success'
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


def _get_status(source: str) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT status FROM refresh_log WHERE source = ?", (source,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


class TestContractsEmpty:
    """When fetch returns empty set, status must NOT be 'success'."""

    def test_empty_fetch_writes_no_data(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):
            with patch("src.contract_data.fetch_contract_year_players", return_value=set()):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_contracts

                progress = BootstrapProgress()
                _bootstrap_contracts(progress)

                status = _get_status("contracts")
                assert status == "no_data", f"Expected 'no_data' when fetcher returned empty set, got {status!r}"


class TestContractsZeroMatched:
    """When fetcher returns names but none match any DB player, status must NOT be 'success'."""

    def test_no_matches_writes_no_data(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):
            conn = get_connection()
            try:
                conn.execute(
                    """INSERT INTO players (player_id, name, mlb_id, is_hitter, team, positions)
                       VALUES (1, 'Joe Schmoe', 100001, 1, 'TST', 'OF')"""
                )
                conn.commit()
            finally:
                conn.close()

            unmatched = {"nonexistent_player_alpha", "nonexistent_player_beta"}
            with patch("src.contract_data.fetch_contract_year_players", return_value=unmatched):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_contracts

                progress = BootstrapProgress()
                _bootstrap_contracts(progress)

                status = _get_status("contracts")
                assert status == "no_data", (
                    f"Expected 'no_data' when 0 players matched the contract-year list, got {status!r}"
                )


class TestContractsSuccess:
    """When at least one player matches, status must be 'success'."""

    def test_match_writes_success(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):
            conn = get_connection()
            try:
                conn.execute(
                    """INSERT INTO players (player_id, name, mlb_id, is_hitter, team, positions)
                       VALUES (1, 'Aaron Judge', 592450, 1, 'NYY', 'OF')"""
                )
                conn.commit()
            finally:
                conn.close()

            with patch(
                "src.contract_data.fetch_contract_year_players",
                return_value={"aaron judge"},
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_contracts

                progress = BootstrapProgress()
                _bootstrap_contracts(progress)

                status = _get_status("contracts")
                assert status == "success", f"Expected 'success' when at least one player matched, got {status!r}"

                conn = get_connection()
                try:
                    row = conn.execute("SELECT contract_year FROM players WHERE name = 'Aaron Judge'").fetchone()
                finally:
                    conn.close()
                assert row[0] == 1
