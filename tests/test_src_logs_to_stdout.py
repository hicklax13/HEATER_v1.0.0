"""Guard: src.* logs must reach stdout via the idempotent configure_src_logging()
(2026-06-06: moved out of data_bootstrap's import side-effect, which didn't reliably
take effect on Railway). Runtime behavior is covered by tests/test_logging_setup.py;
this file guards the wiring (module exists + main() calls it)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_logging_setup_module_exists_and_targets_src_stdout():
    from src import logging_setup

    src = (Path(__file__).parent.parent / "src" / "logging_setup.py").read_text(encoding="utf-8")
    assert "StreamHandler(sys.stdout)" in src
    assert 'getLogger("src")' in src
    assert hasattr(logging_setup, "configure_src_logging")


def test_main_calls_configure_src_logging():
    app = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
    assert "configure_src_logging()" in app
