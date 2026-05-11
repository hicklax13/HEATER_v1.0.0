"""SF-16: Hardcoded TTL literals must be configurable via StalenessConfig.

Bug: Seven phases pass integer literals to check_staleness():
- adp_sources: 24
- depth_charts: 168
- contracts: 720
- park_factors_dynamic: 168
- bat_speed: 168
- forty_man: 168
- game_logs: 1

Fix: Add fields to StalenessConfig and read TTLs from config.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


_MOCKED_PHASES = (
    "_bootstrap_players",
    "_bootstrap_park_factors",
    "_bootstrap_projections",
    "_bootstrap_live_stats",
    "_bootstrap_historical",
    "_bootstrap_injury_data",
    "_bootstrap_yahoo",
    "_bootstrap_extended_roster",
    "_bootstrap_adp_sources",
    "_bootstrap_depth_charts",
    "_bootstrap_contracts",
    "_bootstrap_news",
    "_bootstrap_prospects",
    "_bootstrap_news_intel",
    "_bootstrap_ecr_consensus",
    "_bootstrap_game_day",
    "_bootstrap_team_strength",
    "_bootstrap_stuff_plus",
    "_bootstrap_batting_stats",
    "_bootstrap_sprint_speed",
    "_bootstrap_dynamic_park_factors",
    "_bootstrap_bat_speed",
    "_bootstrap_forty_man",
    "_bootstrap_umpire_tendencies",
    "_bootstrap_catcher_framing",
    "_bootstrap_pvb_splits",
    "_bootstrap_game_logs",
    "_bootstrap_injury_writeback",
    "_bootstrap_draft_results",
)


def _mock_all_phases(stack: ExitStack) -> None:
    for name in _MOCKED_PHASES:
        stack.enter_context(patch(f"src.data_bootstrap.{name}", return_value="ok"))


class TestSF16StalenessConfigFields:
    def test_has_new_fields_with_defaults(self):
        """StalenessConfig must expose all 7 new TTL fields with the documented defaults."""
        from src.data_bootstrap import StalenessConfig

        cfg = StalenessConfig()
        assert cfg.adp_sources_hours == 24
        assert cfg.depth_charts_hours == 168
        assert cfg.contracts_hours == 720
        assert cfg.dynamic_park_factors_hours == 168
        assert cfg.bat_speed_hours == 168
        assert cfg.forty_man_hours == 168
        assert cfg.game_logs_hours == 1


class TestSF16ConfigPlumbedThrough:
    @pytest.mark.parametrize(
        "field, source",
        [
            ("adp_sources_hours", "adp_sources"),
            ("depth_charts_hours", "depth_charts"),
            ("contracts_hours", "contracts"),
            ("dynamic_park_factors_hours", "park_factors_dynamic"),
            ("bat_speed_hours", "bat_speed"),
            ("forty_man_hours", "forty_man"),
            ("game_logs_hours", "game_logs"),
        ],
    )
    def test_check_staleness_uses_configured_ttl(self, temp_db, field, source):
        """When user passes custom StalenessConfig, the configured TTL is sent to check_staleness()."""
        with patch("src.database.DB_PATH", temp_db):
            calls: list[tuple] = []

            def _record(src, hours):
                calls.append((src, hours))
                return False

            from src.data_bootstrap import StalenessConfig, bootstrap_all_data

            kwargs = {field: 999.5}
            cfg = StalenessConfig(**kwargs)

            with ExitStack() as stack:
                _mock_all_phases(stack)
                stack.enter_context(patch("src.database.check_staleness", side_effect=_record))

                bootstrap_all_data(force=False, staleness=cfg)

            matching = [c for c in calls if c[0] == source]
            assert matching, f"check_staleness was never called with source={source!r}"
            assert any(c[1] == 999.5 for c in matching), (
                f"Expected check_staleness({source!r}, 999.5) — got {matching}. Field {field!r} is not plumbed through."
            )
