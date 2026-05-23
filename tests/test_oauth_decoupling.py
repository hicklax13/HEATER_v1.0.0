"""T1.21 / Batch G: OAuth decoupling structural-invariant guards.

These tests enforce the architectural constraint that the optimize button
in pages/2_Line-up_Optimizer.py does NOT trigger a forced Yahoo API refresh.
Forcing refresh on every optimize click was the root cause of the 159-second
compute hang documented in docs/superpowers/specs/2026-05-22-heater-v2-optimizer-design.md.

Decoupling makes optimize fast (uses cached data) and gives users explicit
control via the "Refresh Yahoo Data" button.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"
YDS_PATH = REPO_ROOT / "src" / "yahoo_data_service.py"


def _find_optimize_clicked_blocks(tree: ast.Module) -> list[ast.If]:
    """Return all `if optimize_clicked:` blocks in the parsed page."""
    blocks: list[ast.If] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "optimize_clicked":
                blocks.append(node)
    return blocks


def test_no_force_refresh_in_optimize_handler():
    """The `if optimize_clicked:` block must NOT call any YahooDataService
    method with force_refresh=True.

    Per T1.21 (Batch G of HEATER v2). Root cause: yds.get_rosters(
    force_refresh=True) at the top of the optimize handler triggers a
    Yahoo OAuth refresh that can hang ~150s in the OAuth retry loop when
    the access token has expired.
    """
    assert PAGE_PATH.exists(), f"Page not found: {PAGE_PATH}"
    source = PAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    blocks = _find_optimize_clicked_blocks(tree)
    assert len(blocks) >= 1, (
        f"Could not find `if optimize_clicked:` block in {PAGE_PATH} — page structure may have changed"
    )

    violations: list[str] = []
    for block in blocks:
        for sub in ast.walk(block):
            if isinstance(sub, ast.Call):
                for kw in sub.keywords:
                    # Literal True only — variable-bound force_refresh is not detected,
                    # but the production pattern is always the literal form.
                    if kw.arg == "force_refresh" and isinstance(kw.value, ast.Constant):
                        if kw.value.value is True:
                            violations.append(f"line {sub.lineno}: {ast.unparse(sub)}")

    assert not violations, (
        "force_refresh=True found inside `if optimize_clicked:` block — "
        "violates T1.21. Move forced refreshes to the explicit "
        "'Refresh Yahoo Data' button. Violations:\n  " + "\n  ".join(violations)
    )
