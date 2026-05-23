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
import time
from pathlib import Path
from unittest.mock import MagicMock

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


def test_refresh_yahoo_button_present():
    """`pages/2_Line-up_Optimizer.py` must contain an explicit button to
    force-refresh Yahoo data. Replaces the implicit force_refresh=True
    that used to live in the optimize click handler (T1.21).

    The button must invoke YahooDataService.force_refresh_all() so that
    rosters, standings, matchup, free agents, transactions, settings, and
    schedule are all refreshed in one click.
    """
    assert PAGE_PATH.exists(), f"Page not found: {PAGE_PATH}"
    source = PAGE_PATH.read_text(encoding="utf-8")

    # Look for a Streamlit button referencing "Refresh Yahoo"
    assert "Refresh Yahoo Data" in source, (
        "pages/2_Line-up_Optimizer.py must contain an explicit "
        "'Refresh Yahoo Data' button (T1.21). The button replaces the "
        "implicit force_refresh=True that was dropped from the optimize "
        "click handler."
    )

    # Verify the button calls force_refresh_all (the YDS method that
    # refreshes all 7 data types in one shot)
    assert "force_refresh_all()" in source, (
        "The 'Refresh Yahoo Data' button must call yds.force_refresh_all() "
        "so all 7 data caches are invalidated, not just one."
    )


def test_get_cached_times_out_when_yahoo_hangs():
    """YahooDataService._get_cached must NOT hang forever when fetch_fn
    is slow (T1.21 root cause: yfpy OAuth retry loop can take 100s+).

    When fetch_fn exceeds the 15-second budget, _get_cached must fall
    back to the db_fallback_fn return value rather than blocking the
    page render.
    """
    import pandas as pd

    from src.yahoo_data_service import YahooDataService

    # Mock client that pretends to be connected
    mock_client = MagicMock()
    mock_client.is_authenticated = True

    service = YahooDataService.__new__(YahooDataService)
    service._client = mock_client
    service._stats = MagicMock()
    service._PREFIX = "_yds_test_"

    # Slow fetch (would take 60s)
    def slow_fetch():
        time.sleep(60)
        return pd.DataFrame({"team_name": ["Hung"]})

    # Fast fallback
    def fast_fallback():
        return pd.DataFrame({"team_name": ["Cached"]})

    start = time.time()
    result = service._get_cached(
        key="test_hang",
        ttl=300,
        fetch_fn=slow_fetch,
        db_fallback_fn=fast_fallback,
        force=True,  # force=True so we definitely call fetch_fn
    )
    elapsed = time.time() - start

    assert elapsed < 20, (
        f"_get_cached blocked for {elapsed:.1f}s when fetch_fn was slow. "
        f"Expected <20s (15s budget + 5s grace). Timeout protection is missing."
    )
    assert not result.empty, "Expected fallback DataFrame, got empty"
    assert result.iloc[0]["team_name"] == "Cached", f"Expected fallback 'Cached', got '{result.iloc[0]['team_name']}'"
