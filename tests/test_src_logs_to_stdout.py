"""2026-06-06 observability: src.* logs (scheduler, yahoo_api reconnect/persist) must
ALSO emit to STDOUT so the deploy console (Railway) shows production diagnostics.

The bootstrap RotatingFileHandler alone routed every src.* log to a file on the volume
(`data/logs/bootstrap.log`), so the live "Yahoo token dies ~1h after paste" failures were
invisible in Railway's console — we were debugging blind. This guards that a stdout
StreamHandler is wired up alongside the file handler (still inside the under-pytest guard
so test runs aren't spammed; companion to test_bootstrap_log_isolation).
"""

from pathlib import Path

_BOOTSTRAP = Path(__file__).resolve().parent.parent / "src" / "data_bootstrap.py"


def test_src_logger_emits_to_stdout():
    src = _BOOTSTRAP.read_text(encoding="utf-8")
    assert "StreamHandler(sys.stdout)" in src, (
        "src.* logs must emit to stdout so the deploy console captures the scheduler + "
        "Yahoo reconnect/persist diagnostics (not just the on-volume bootstrap.log file)"
    )


def test_stdout_handler_attached_to_src_logger_under_guard():
    src = _BOOTSTRAP.read_text(encoding="utf-8")
    # Must stay inside the `if not _UNDER_PYTEST:` block and target the "src" logger.
    assert "if not _UNDER_PYTEST:" in src
    guard_idx = src.index("if not _UNDER_PYTEST:")
    stream_idx = src.index("StreamHandler(sys.stdout)")
    assert stream_idx > guard_idx, "stdout handler must be set up inside the under-pytest guard"
    assert 'getLogger("src")' in src
