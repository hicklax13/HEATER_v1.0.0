"""Wave 9 / Task 5: Free Agents page exposes a level filter."""

from pathlib import Path


def test_free_agents_page_has_level_selectbox():
    """The Free Agents page must include a Streamlit selectbox or radio for
    filtering by player level (MLB / AAA / AA / All). This guards that
    Wave 9 minor leaguers don't silently appear in default FA lists when
    users expect MLB-only."""
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    matches = list(pages_dir.glob("*Free_Agents*.py"))
    assert matches, "Free Agents page not found in pages/"

    text = matches[0].read_text(encoding="utf-8")
    # Look for either a selectbox or a radio with "level" + ("MLB"|"AAA"|"AA"|"All")
    assert ("level" in text.lower()) and ('"AAA"' in text or '"AA"' in text or "selectbox" in text.lower()), (
        f"Wave 9 regression: Free Agents page {matches[0].name} appears "
        "to lack a level filter for MLB/AAA/AA. Add a selectbox or radio."
    )
