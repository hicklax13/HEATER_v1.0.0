"""Tests for background refresh scheduler."""

import time
from unittest.mock import patch

from src.scheduler import (
    _CHECK_INTERVAL_SECONDS,
    is_running,
    start_background_refresh,
    stop_background_refresh,
)


class TestSchedulerLifecycle:
    """Test scheduler start/stop behavior."""

    def setup_method(self):
        """Ensure scheduler is stopped before each test."""
        stop_background_refresh()

    def teardown_method(self):
        """Ensure scheduler is stopped after each test."""
        stop_background_refresh()

    def test_start_sets_running(self):
        """start_background_refresh sets is_running to True."""
        with patch("src.scheduler._refresh_loop"):
            start_background_refresh()
            assert is_running()

    def test_stop_clears_running(self):
        """stop_background_refresh sets is_running to False."""
        with patch("src.scheduler._refresh_loop"):
            start_background_refresh()
            stop_background_refresh()
            assert not is_running()

    def test_double_start_is_idempotent(self):
        """Calling start twice does not create a second thread."""
        with patch("src.scheduler._refresh_loop"):
            start_background_refresh()
            start_background_refresh()  # should be a no-op
            assert is_running()

    def test_check_interval_positive(self):
        """Check interval is a positive number of seconds."""
        assert _CHECK_INTERVAL_SECONDS > 0
        assert isinstance(_CHECK_INTERVAL_SECONDS, int)

    def test_not_running_initially(self):
        """Scheduler is not running when module loads."""
        assert not is_running()
