"""SFH M-4 structural guard: `src/matchup_context.py` must NOT have a
module-level `get_yahoo_data_service` binding.

Why this matters: `tests/test_unified_weights.py::isolated_yahoo_data`
patches `src.yahoo_data_service.get_yahoo_data_service` to return a mock.
This works ONLY because every caller in `matchup_context.py` does the
import INSIDE a function body — `from src.yahoo_data_service import
get_yahoo_data_service` evaluated at call time hits the patched binding.

If a future refactor hoisted the import to module-level:

    from src.yahoo_data_service import get_yahoo_data_service  # ← module-level

then `matchup_context.get_yahoo_data_service` would be the ORIGINAL
function captured at module-import time, and the test patch would have
no effect — tests would silently start reading real SQLite cached
matchup data again (the very regression PR #52 fixed).

This test ensures that any future hoist is caught immediately.
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_matchup_context_imports_yds_inside_functions_only():
    """No top-level `from src.yahoo_data_service import get_yahoo_data_service`."""
    path = Path(__file__).resolve().parent.parent / "src" / "matchup_context.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    offenders: list[str] = []
    for node in tree.body:  # body == module-level statements only
        if isinstance(node, ast.ImportFrom):
            if node.module == "src.yahoo_data_service":
                for alias in node.names:
                    if alias.name == "get_yahoo_data_service":
                        offenders.append(
                            f"src/matchup_context.py:{node.lineno}: "
                            f"`from src.yahoo_data_service import get_yahoo_data_service` "
                            f"is hoisted to module level — this would bypass the "
                            f"`isolated_yahoo_data` test fixture in tests/test_unified_weights.py. "
                            f"Keep the import inside each function body (see PR #52)."
                        )

    assert not offenders, "\n".join(offenders)
