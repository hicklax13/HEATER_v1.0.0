"""Regression guard for SFH L (2026-05-20): bootstrap.log test isolation.

The module ``src/data_bootstrap.py`` attaches a ``RotatingFileHandler`` to
the "src" logger at import time so production bootstrap runs persist a log
to ``data/logs/bootstrap.log`` for post-mortem analysis (SF-14).

Pre-fix, that handler was attached unconditionally. Pytest imports of any
``src.*`` module triggered the attach, and tests that exercise fallback
paths with mocked DB raises (e.g. ``test_wave8b_silent_failures`` patching
``get_connection`` to ``side_effect=RuntimeError("DB out")``) wrote their
fake "DB out" log lines into the production file. Operators reviewing
``bootstrap.log`` saw 70+ "DB out" warnings and assumed a real DB outage.

Post-fix, the attach is gated on ``"pytest" in sys.modules`` (with an
override env var ``HEATER_DISABLE_FILE_LOG=1`` for non-pytest test
runners). This file guards the contract.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler


def test_no_file_handler_attached_under_pytest():
    """Under pytest, ``src.data_bootstrap`` must NOT attach the
    RotatingFileHandler to the 'src' logger."""
    assert "pytest" in sys.modules, "this test must run under pytest"

    # Force a fresh import so we exercise the gated attach branch.
    import src.data_bootstrap  # noqa: F401

    src_logger = logging.getLogger("src")
    file_handlers = [h for h in src_logger.handlers if isinstance(h, RotatingFileHandler)]

    assert file_handlers == [], (
        "src logger has a RotatingFileHandler attached during pytest run — "
        "test log output will leak into data/logs/bootstrap.log and "
        "pollute production post-mortem analysis. "
        f"Found: {file_handlers}"
    )


def test_under_pytest_sentinel_is_true():
    """Sanity: the module's _UNDER_PYTEST sentinel is True when this test
    runs, so the guard branch above is the one being exercised."""
    import src.data_bootstrap as boot

    assert boot._UNDER_PYTEST is True, (
        "_UNDER_PYTEST should be True inside a pytest run; if this fails the "
        "detection logic (sys.modules check) regressed."
    )
