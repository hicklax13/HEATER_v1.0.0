"""Idempotent logging setup for the ``src`` logger.

Why this exists: e7d7150 attached a stdout handler as an *import side-effect* of
``src.data_bootstrap``, which did not reliably take effect on the Railway deploy
(the scheduler's INFO lines never reached the console). This module exposes an
explicit, idempotent setup that ``main()`` calls at startup, so the handler is
attached in the real runtime path regardless of import order.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MARKER = "_heater_src_stdout"


def configure_src_logging(_force: bool = False) -> None:
    """Attach a stdout handler (and, in production, a rotating ``bootstrap.log``
    file handler) to the ``src`` logger at INFO level — exactly once.

    Idempotent: repeat calls are no-ops once the marked stdout handler exists.
    Skipped entirely under pytest unless ``_force`` is set; and even with
    ``_force`` the file handler is NEVER attached under pytest, so test log output
    can't leak into ``data/logs/bootstrap.log`` (companion to
    ``test_bootstrap_log_isolation``).
    """
    under_pytest = "pytest" in sys.modules or os.environ.get("HEATER_DISABLE_FILE_LOG") == "1"
    if under_pytest and not _force:
        return

    src_logger = logging.getLogger("src")
    if any(getattr(h, _MARKER, False) for h in src_logger.handlers):
        return  # already configured

    src_logger.setLevel(logging.INFO)
    # Own the src.* handlers fully so a WARNING isn't also emitted via the root
    # lastResort handler (which would double-print, unformatted, on the console).
    src_logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    setattr(stream, _MARKER, True)
    src_logger.addHandler(stream)

    if under_pytest:
        return  # never attach the file handler under pytest (test isolation)

    try:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(log_dir / "bootstrap.log", maxBytes=5_000_000, backupCount=3)
        fileh.setFormatter(fmt)
        src_logger.addHandler(fileh)
    except Exception:
        # File logging is best-effort; stdout (above) is what matters on Railway.
        pass
