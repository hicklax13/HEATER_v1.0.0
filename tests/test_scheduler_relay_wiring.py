import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.scheduler as sch


def test_refresh_once_pulls_before_reconnect():
    calls = []
    with (
        patch("src.token_relay.pull_relayed_token", side_effect=lambda: calls.append("pull")),
        patch("src.yahoo_api.try_reconnect_yahoo", side_effect=lambda: calls.append("reconnect") or None),
        patch("src.data_bootstrap.bootstrap_all_data", return_value={}),
    ):
        sch._refresh_once()
    assert calls == ["pull", "reconnect"]


def test_degrade_logged_once_then_recover(caplog):
    caplog.set_level(logging.INFO)
    sch._last_yahoo_ok = None
    with (
        patch("src.token_relay.pull_relayed_token", return_value=False),
        patch("src.data_bootstrap.bootstrap_all_data", return_value={}),
    ):
        with patch("src.yahoo_api.try_reconnect_yahoo", return_value=None):
            sch._refresh_once()
            sch._refresh_once()
        assert caplog.text.count("Yahoo sync degraded") == 1
        caplog.clear()

        class _C:
            def persist_current_token(self):
                return True

        with patch("src.yahoo_api.try_reconnect_yahoo", return_value=_C()):
            sch._refresh_once()
        assert caplog.text.count("Yahoo sync recovered") == 1
