"""Runtime guard for the src.* stdout logging setup (2026-06-06).

e7d7150 attached the stdout handler as an import side-effect of data_bootstrap,
which did not reliably take effect on Railway (the scheduler's INFO lines never
reached the console). configure_src_logging() is the explicit, idempotent setup
that main() calls at startup. These tests verify the RUNTIME behavior (a src.*
INFO record actually reaches the stdout handler) rather than grepping source text.
"""

import io
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def _restore_src_logger():
    """Save the 'src' logger's handlers/level/propagate, clear them for the test,
    and restore afterward so configure_src_logging() can't pollute other tests
    (notably the caplog-based tests that rely on src.* propagating to root)."""
    lg = logging.getLogger("src")
    saved_handlers = list(lg.handlers)
    saved_level = lg.level
    saved_propagate = lg.propagate
    for h in list(lg.handlers):
        lg.removeHandler(h)
    yield lg
    for h in list(lg.handlers):
        lg.removeHandler(h)
    for h in saved_handlers:
        lg.addHandler(h)
    lg.setLevel(saved_level)
    lg.propagate = saved_propagate


def test_configure_attaches_stdout_handler_at_runtime(_restore_src_logger):
    from src.logging_setup import _MARKER, configure_src_logging

    configure_src_logging(_force=True)
    lg = _restore_src_logger
    marked = [h for h in lg.handlers if getattr(h, _MARKER, False)]
    assert marked, "a stdout handler with the marker should be attached"

    # Runtime proof: a src.* INFO record reaches the stdout handler's stream.
    buf = io.StringIO()
    marked[0].stream = buf
    logging.getLogger("src.demo").info("hello-relay")
    assert "hello-relay" in buf.getvalue()


def test_configure_is_idempotent(_restore_src_logger):
    from src.logging_setup import _MARKER, configure_src_logging

    configure_src_logging(_force=True)
    configure_src_logging(_force=True)
    marked = [h for h in _restore_src_logger.handlers if getattr(h, _MARKER, False)]
    assert len(marked) == 1


def test_no_file_handler_under_pytest_even_with_force(_restore_src_logger):
    """Even with _force, the RotatingFileHandler must NOT be attached under pytest
    (keeps test logs out of data/logs/bootstrap.log — companion to
    test_bootstrap_log_isolation)."""
    from logging.handlers import RotatingFileHandler

    from src.logging_setup import configure_src_logging

    configure_src_logging(_force=True)
    file_handlers = [h for h in _restore_src_logger.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers == []
