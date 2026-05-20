"""FA-engine P1 PR1 (2026-05-20): Free Agents page uses recommend_fa_moves
as the primary recommendation engine.

Background:
  The FA page (pages/14_Free_Agents.py) previously called
  src.waiver_wire.compute_add_drop_recommendations — an older engine
  missing opponent context, urgency weighting, news warnings, and slot-
  aware drop selection. The newer engine src.optimizer.fa_recommender
  .recommend_fa_moves has all of these but was only wired to the Lineup
  Optimizer page. The Crochet/Kirk bad-recommendation bug pattern is
  directly caused by the old engine's lack of opponent context.

This guard pins the rewire:
  * FA page imports recommend_fa_moves
  * FA page imports build_optimizer_context
  * FA page calls recommend_fa_moves before the legacy fallback
  * Output-shape adapter translates add_id/drop_id/sustainability into
    add_player_id/drop_player_id/sustainability_score
  * Legacy engine is kept as fallback in the except branch (defensive)

A future refactor that reverts to calling compute_add_drop_recommendations
as primary will fail this test.
"""

from __future__ import annotations

import pathlib

_PAGE_PATH = pathlib.Path("pages/14_Free_Agents.py")
_WAIVER_PATH = pathlib.Path("src/waiver_wire.py")


def _page_source() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


def test_fa_page_imports_recommend_fa_moves():
    """Free Agents page must import the new engine."""
    src = _page_source()
    assert "from src.optimizer.fa_recommender import recommend_fa_moves" in src, (
        "FA page must import recommend_fa_moves from src.optimizer.fa_recommender. "
        "This is the primary engine for FA recommendations going forward."
    )


def test_fa_page_imports_build_optimizer_context():
    """Free Agents page must import the context builder."""
    src = _page_source()
    assert "from src.optimizer.shared_data_layer import build_optimizer_context" in src, (
        "FA page must import build_optimizer_context. recommend_fa_moves requires an OptimizerDataContext as input."
    )


def test_fa_page_calls_recommend_fa_moves_as_primary():
    """recommend_fa_moves must be CALLED, not just imported. And it must
    be called BEFORE the legacy fallback (so it's primary, not secondary)."""
    src = _page_source()
    assert "recommend_fa_moves(_ctx_fa" in src, (
        "Expected `recommend_fa_moves(_ctx_fa, max_moves=...)` call. "
        "If the variable name changed, update this guard accordingly."
    )


def test_fa_page_uses_rest_of_season_scope():
    """The FA page is for ROS decisions, not weekly streamers — scope
    must be 'rest_of_season'. Other valid scopes (today, rest_of_week)
    are used by other pages."""
    src = _page_source()
    assert 'scope="rest_of_season"' in src or "scope='rest_of_season'" in src, (
        "FA page must build context with scope='rest_of_season'. The other "
        "scopes (today, rest_of_week) are for the Lineup Optimizer page."
    )


def test_fa_page_translates_output_shape_keys():
    """The output adapter must rename engine keys to FA-page keys:
      add_id          → add_player_id
      drop_id         → drop_player_id
      sustainability  → sustainability_score
    These are needed by the rendering code + the IL-stash filter from PR #89."""
    src = _page_source()
    for required in (
        '"add_player_id": m.get("add_id"',
        '"drop_player_id": m.get("drop_id"',
        '"sustainability_score": m.get("sustainability"',
    ):
        assert required in src, (
            f"Expected output-shape adapter line: {required!r}. The page's "
            "rendering code and the IL-stash filter expect the renamed keys."
        )


def test_legacy_waiver_wire_kept_as_fallback():
    """If recommend_fa_moves throws, the page should fall back to the
    legacy engine so the FA page degrades gracefully rather than going
    completely empty. The legacy call should appear in an `except` block."""
    src = _page_source()
    # Both engines should be referenced; the new engine first (primary),
    # the old engine inside the except branch (fallback).
    assert "recommend_fa_moves" in src
    assert "compute_add_drop_recommendations" in src
    # The legacy fallback should be after the new engine call.
    new_idx = src.find("recommend_fa_moves(")
    legacy_idx = src.find("compute_add_drop_recommendations(")
    assert new_idx > 0 and legacy_idx > 0, "Both engines must be referenced — new as primary, old as fallback."
    assert new_idx < legacy_idx, (
        "The new engine recommend_fa_moves must be called BEFORE the legacy "
        "compute_add_drop_recommendations fallback. Otherwise the legacy is "
        "primary and the rewire is incomplete."
    )


def test_legacy_function_has_deprecation_warning():
    """compute_add_drop_recommendations must emit a DeprecationWarning
    so any other callers (test code, future code paths) get the signal
    that this function is going away."""
    src = _WAIVER_PATH.read_text(encoding="utf-8")
    # The deprecation should be in the docstring AND raised at runtime.
    assert "deprecated" in src.lower(), "Legacy function must be marked deprecated in docstring."
    assert "DeprecationWarning" in src, (
        "compute_add_drop_recommendations must raise DeprecationWarning at "
        "runtime — passive docstring deprecation isn't enough; existing "
        "callers need to see the signal."
    )
    assert "warnings.warn" in src, (
        "Expected warnings.warn(...) call in compute_add_drop_recommendations to actually emit the DeprecationWarning."
    )
