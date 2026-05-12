"""Wave 9 / Task 6: Player Compare page exposes a level filter."""

from pathlib import Path


def test_player_compare_page_has_level_filter():
    """Player Compare must support filtering the candidate-player picker
    by level so users don't accidentally compare an MLB regular against
    an AAA prospect with no Yahoo ownership context."""
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    matches = list(pages_dir.glob("*Player_Compare*.py"))
    assert matches, "Player Compare page not found in pages/"

    text = matches[0].read_text(encoding="utf-8")
    assert ("level" in text.lower()) and ('"AAA"' in text or '"AA"' in text or "selectbox" in text.lower()), (
        f"Wave 9 regression: Player Compare page {matches[0].name} lacks a level filter."
    )
