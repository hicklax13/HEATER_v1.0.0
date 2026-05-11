"""SF-7: tests for shipped seed files + 3-tier waterfall in
catcher_framing / umpire_tendencies bootstrap phases.

The waterfall is:
    Tier 1 (primary)    — pybaseball / FanGraphs / statsapi (live)
    Tier 2 (fallback)   — Savant scrape with browser headers (catchers only;
                          Savant has no public umpire leaderboard)
    Tier 3 (emergency)  — shipped 2024 baseline at data/seed/*.json

These tests verify:
1. The two seed files exist, parse, and contain a sane number of entries.
2. When EVERY live fetch path is mocked to fail, the bootstrap functions
   load the seed file, persist rows to the DB, and write a refresh_log row
   with ``tier='emergency'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Repo location helpers ──────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
CATCHER_SEED = REPO_ROOT / "data" / "seed" / "catcher_framing_2024.json"
UMPIRE_SEED = REPO_ROOT / "data" / "seed" / "umpire_tendencies_2024.json"


# ── 1. Seed-file existence + structural integrity ──────────────────────


class TestSeedFileExistence:
    def test_catcher_framing_seed_exists_and_parses(self):
        assert CATCHER_SEED.exists(), f"missing seed: {CATCHER_SEED}"
        payload = json.loads(CATCHER_SEED.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert "catchers" in payload, "seed must have 'catchers' key"
        assert isinstance(payload["catchers"], list)

    def test_catcher_framing_seed_has_enough_entries(self):
        payload = json.loads(CATCHER_SEED.read_text(encoding="utf-8"))
        assert len(payload["catchers"]) >= 20, f"need >=20 catchers, have {len(payload['catchers'])}"

    def test_catcher_seed_entries_have_required_fields(self):
        payload = json.loads(CATCHER_SEED.read_text(encoding="utf-8"))
        required = {"player_name", "team", "framing_runs", "games"}
        for i, row in enumerate(payload["catchers"]):
            missing = required - set(row.keys())
            assert not missing, f"catcher #{i} ({row}) missing {missing}"
            assert isinstance(row["framing_runs"], (int, float))
            # Sanity: framing runs should be in plausible Savant range.
            assert -25.0 <= row["framing_runs"] <= 25.0, (
                f"{row['player_name']}: framing_runs={row['framing_runs']} outside plausible Savant range"
            )

    def test_umpire_tendencies_seed_exists_and_parses(self):
        assert UMPIRE_SEED.exists(), f"missing seed: {UMPIRE_SEED}"
        payload = json.loads(UMPIRE_SEED.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert "umpires" in payload, "seed must have 'umpires' key"
        assert isinstance(payload["umpires"], list)

    def test_umpire_tendencies_seed_has_enough_entries(self):
        payload = json.loads(UMPIRE_SEED.read_text(encoding="utf-8"))
        assert len(payload["umpires"]) >= 20, f"need >=20 umpires, have {len(payload['umpires'])}"

    def test_umpire_seed_entries_have_required_fields(self):
        payload = json.loads(UMPIRE_SEED.read_text(encoding="utf-8"))
        required = {"name", "games", "k_pct_diff", "bb_pct_diff"}
        for i, row in enumerate(payload["umpires"]):
            missing = required - set(row.keys())
            assert not missing, f"umpire #{i} ({row}) missing {missing}"
            # Sanity: deltas should be in plausible Savant range.
            assert -0.05 <= row["k_pct_diff"] <= 0.05, f"{row['name']}: k_pct_diff={row['k_pct_diff']} outside ±0.05"
            assert -0.05 <= row["bb_pct_diff"] <= 0.05, f"{row['name']}: bb_pct_diff={row['bb_pct_diff']} outside ±0.05"

    def test_seed_provenance_metadata_present(self):
        for seed in (CATCHER_SEED, UMPIRE_SEED):
            payload = json.loads(seed.read_text(encoding="utf-8"))
            assert payload.get("source"), f"{seed.name} missing 'source'"
            assert payload.get("scraped_date"), f"{seed.name} missing 'scraped_date'"
            assert payload.get("season") == 2024, f"{seed.name} should be season=2024"


# ── 2. Loader helpers (no DB writes, no network) ───────────────────────


class TestSeedLoaderHelpers:
    def test_load_catcher_framing_seed_returns_rows(self):
        from src.data_bootstrap import _load_catcher_framing_seed

        rows = _load_catcher_framing_seed()
        assert rows is not None
        assert len(rows) >= 20
        # Normalized shape: name, framing_runs, games, pop_time, cs_pct.
        sample = rows[0]
        assert "name" in sample
        assert "framing_runs" in sample
        assert "games" in sample

    def test_load_umpire_tendencies_seed_returns_rows(self):
        from src.data_bootstrap import _load_umpire_tendencies_seed

        rows = _load_umpire_tendencies_seed()
        assert rows is not None
        assert len(rows) >= 20
        sample = rows[0]
        assert "name" in sample
        assert "k_pct" in sample
        assert "k_pct_delta" in sample
        # Reconstructed absolute K% should be a sensible MLB rate.
        assert 0.15 <= sample["k_pct"] <= 0.35, f"absolute k_pct={sample['k_pct']} outside MLB plausible range"

    def test_load_seed_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """If the seed file path is absent, helpers return None (graceful)."""
        # Point cwd at an empty tmp dir so the relative seed lookup misses.
        monkeypatch.chdir(tmp_path)
        from src.data_bootstrap import (
            _load_catcher_framing_seed,
            _load_umpire_tendencies_seed,
        )

        assert _load_catcher_framing_seed() is None
        assert _load_umpire_tendencies_seed() is None


# ── 3. Bootstrap waterfall — Tier 3 fires when all live sources fail ──


@pytest.fixture
def temp_db_with_catchers(tmp_path):
    """Initialize a temp DB and populate it with a few catchers so the
    name→player_id lookup in the bootstrap function succeeds.

    Returns the DB path; uses ``patch`` so callers re-patch DB_PATH inside
    the test ``with`` block.
    """
    db_path = tmp_path / "test_sf7.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import get_connection, init_db, upsert_player_bulk

        init_db()
        # Seed the players table with catchers whose names match the seed file
        # so the name→pid mapping connects rows.
        upsert_player_bulk(
            [
                {"name": "Patrick Bailey", "team": "SF", "positions": "C", "is_hitter": True},
                {"name": "Cal Raleigh", "team": "SEA", "positions": "C", "is_hitter": True},
                {"name": "Jose Trevino", "team": "NYY", "positions": "C", "is_hitter": True},
                {"name": "Adley Rutschman", "team": "BAL", "positions": "C", "is_hitter": True},
                {"name": "Will Smith", "team": "LAD", "positions": "C", "is_hitter": True},
                {"name": "Salvador Perez", "team": "KC", "positions": "C", "is_hitter": True},
                {"name": "J.T. Realmuto", "team": "PHI", "positions": "C", "is_hitter": True},
                {"name": "William Contreras", "team": "MIL", "positions": "C", "is_hitter": True},
                {"name": "Sean Murphy", "team": "ATL", "positions": "C", "is_hitter": True},
                {"name": "Yainer Diaz", "team": "HOU", "positions": "C", "is_hitter": True},
                {"name": "Jake Rogers", "team": "DET", "positions": "C", "is_hitter": True},
                {"name": "Carson Kelly", "team": "CHC", "positions": "C", "is_hitter": True},
                {"name": "Tyler Stephenson", "team": "CIN", "positions": "C", "is_hitter": True},
                {"name": "Jonah Heim", "team": "TEX", "positions": "C", "is_hitter": True},
                {"name": "Bo Naylor", "team": "CLE", "positions": "C", "is_hitter": True},
            ]
        )
        get_connection().close()
        yield db_path


@pytest.fixture
def temp_db_blank(tmp_path):
    db_path = tmp_path / "test_sf7_blank.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


class TestCatcherFramingTierWaterfall:
    def test_tier3_seed_loads_when_all_live_paths_fail(self, temp_db_with_catchers):
        """Mock every live fetch to fail; verify Tier 3 (seed) populates the
        catcher_framing table and writes refresh_log.tier='emergency'."""
        from src.data_bootstrap import BootstrapProgress, _bootstrap_catcher_framing

        with (
            patch("src.database.DB_PATH", temp_db_with_catchers),
            # Force Tier 1 pybaseball Savant to raise.
            patch(
                "pybaseball.statcast_catcher_framing",
                side_effect=Exception("403 simulated"),
            ),
            # Force Tier 1 pybaseball FanGraphs to raise.
            patch("pybaseball.batting_stats", side_effect=Exception("403 simulated")),
            # Force Tier 2 Savant scrape to return None (network failure).
            patch(
                "src.data_bootstrap._fetch_catcher_framing_savant_scrape",
                return_value=None,
            ),
            # Stub statsapi import path so the statsapi sub-fallback is a no-op.
            patch.dict("sys.modules", {}),
        ):
            # We additionally make statsapi.player_stat_data raise if invoked.
            try:
                import statsapi as _statsapi  # noqa: F401

                _statsapi_patch = patch(
                    "statsapi.player_stat_data",
                    side_effect=Exception("simulated"),
                )
                _statsapi_patch.start()
            except ImportError:
                _statsapi_patch = None

            try:
                progress = BootstrapProgress()
                result = _bootstrap_catcher_framing(progress)
            finally:
                if _statsapi_patch is not None:
                    _statsapi_patch.stop()

        # Result string should mention emergency tier
        assert "emergency" in result.lower() or "saved" in result.lower(), f"unexpected result: {result}"

        # Verify DB row count + refresh_log tier
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db_with_catchers):
            conn = get_connection()
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM catcher_framing").fetchone()[0]
                tier_row = conn.execute(
                    "SELECT tier, status FROM refresh_log WHERE source='catcher_framing'"
                ).fetchone()
            finally:
                conn.close()

        assert cnt > 0, "catcher_framing table should have rows from seed"
        assert tier_row is not None, "refresh_log row should exist"
        assert tier_row[0] == "emergency", f"refresh_log.tier should be 'emergency', got {tier_row[0]!r}"

    def test_tier1_success_does_not_load_seed(self, temp_db_with_catchers):
        """When pybaseball returns data, the seed file is NOT loaded."""
        import pandas as pd

        from src.data_bootstrap import BootstrapProgress, _bootstrap_catcher_framing

        # Pretend pybaseball.statcast_catcher_framing returns a real DataFrame.
        fake_df = pd.DataFrame(
            [
                {"Name": "Patrick Bailey", "framing_runs": 10.5, "G": 100},
                {"Name": "Cal Raleigh", "framing_runs": 8.3, "G": 120},
            ]
        )

        with (
            patch("src.database.DB_PATH", temp_db_with_catchers),
            patch("pybaseball.statcast_catcher_framing", return_value=fake_df),
            patch("src.data_bootstrap._load_catcher_framing_seed") as load_seed_spy,
        ):
            progress = BootstrapProgress()
            _bootstrap_catcher_framing(progress)
            load_seed_spy.assert_not_called()

        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db_with_catchers):
            conn = get_connection()
            try:
                tier_row = conn.execute("SELECT tier FROM refresh_log WHERE source='catcher_framing'").fetchone()
            finally:
                conn.close()
        assert tier_row[0] == "primary"


class TestUmpireTendenciesTierWaterfall:
    def test_tier3_seed_loads_when_statsapi_returns_empty(self, temp_db_blank):
        """Mock statsapi.schedule to return [] (no games); verify Tier 3 loads."""
        from src.data_bootstrap import BootstrapProgress, _bootstrap_umpire_tendencies

        with (
            patch("src.database.DB_PATH", temp_db_blank),
            patch("statsapi.schedule", return_value=[]),
        ):
            progress = BootstrapProgress()
            result = _bootstrap_umpire_tendencies(progress)

        assert "saved" in result.lower() or "emergency" in result.lower(), f"unexpected result: {result}"

        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db_blank):
            conn = get_connection()
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM umpire_tendencies").fetchone()[0]
                tier_row = conn.execute(
                    "SELECT tier, status FROM refresh_log WHERE source='umpire_tendencies'"
                ).fetchone()
            finally:
                conn.close()

        assert cnt > 0, "umpire_tendencies should have rows from seed"
        assert tier_row is not None
        assert tier_row[0] == "emergency", f"refresh_log.tier should be 'emergency', got {tier_row[0]!r}"

    def test_tier3_seed_uses_2024_season_tag(self, temp_db_blank):
        """Seed-loaded rows should be tagged season=2024 to flag origin."""
        from src.data_bootstrap import BootstrapProgress, _bootstrap_umpire_tendencies

        with (
            patch("src.database.DB_PATH", temp_db_blank),
            patch("statsapi.schedule", return_value=[]),
        ):
            progress = BootstrapProgress()
            _bootstrap_umpire_tendencies(progress)

        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db_blank):
            conn = get_connection()
            try:
                seasons = conn.execute("SELECT DISTINCT season FROM umpire_tendencies").fetchall()
            finally:
                conn.close()

        seasons = [s[0] for s in seasons]
        assert 2024 in seasons, f"expected season=2024, got {seasons}"
