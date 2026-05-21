"""SFH FA-IL guard (2026-05-20): unified IL-stash protection on the
Free Agents page.

Background:
  The top "Recommended Adds/Drops" headline and table were rendering
  raw recommendations without applying the IL-stash protection that
  the bottom "Recommended Drops" section had. Garrett Crochet
  (IL15, top-30 SP) was appearing as a drop candidate in the headline
  pickup — "Add Kyle Teel / Drop Shane Bieber" — even though the
  bottom section knew to protect IL stashes.

  The previous protection only matched IL_STASH_NAMES (hardcoded
  ``{"Shane Bieber", "Spencer Strider"}``) and substring-matched
  ``"10-day"`` / ``"day-to-day"`` in news il_status — silently letting
  IL15 / IL60 players slip through every layer.

This guard pins the unified filter:
  * pages/14_Free_Agents.py imports IL_STASH_NAMES from src.alerts
  * Expanded protect-substring set covers 10-day / 15-day / 60-day / DTD
  * Authoritative roster-status lookup against league_rosters.status
    in (IL10, IL15, IL60, IL, DTD)
  * filtered_recs is reassigned via list comprehension that excludes
    drops whose name is in protected_drop_names

A future refactor that removes any of these layers fails this test.
"""

from __future__ import annotations

import pathlib

_PAGE_PATH = pathlib.Path("pages/14_Free_Agents.py")


def _page_source() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


def test_il_stash_names_imported_on_fa_page():
    """IL stash names symbol must be imported from src.alerts.

    Use AST instead of substring matching so the test is robust against
    ruff's import-formatting choices (single-line vs multi-line). After
    FA PR7 the import line was widened to include `get_il_stash_names`
    alongside `IL_STASH_NAMES`, which ruff format may split across
    multiple lines — both forms are valid.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(_PAGE_PATH).read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.alerts":
            names = {alias.name for alias in node.names}
            if "IL_STASH_NAMES" in names:
                found = True
                break
    assert found, (
        "Free Agents page must import IL_STASH_NAMES from src.alerts (any form). "
        "Without it, the IL stash protection layer is missing."
    )


def test_protected_drop_names_set_built():
    """The unified filter must build a `protected_drop_names` set."""
    src = _page_source()
    assert "protected_drop_names" in src, (
        "Expected variable `protected_drop_names` (set used to filter "
        "filtered_recs before both top and bottom sections render). If "
        "this got renamed, update this guard accordingly."
    )


def test_il_protect_substrings_cover_long_term_il():
    """The substring list MUST include 15-day and 60-day, not just
    10-day / day-to-day (the original buggy filter only covered short-
    term IL — Crochet IL15 and Bieber IL60 fell through)."""
    src = _page_source()
    for required in ("10-day", "15-day", "60-day", "day-to-day"):
        assert required in src, (
            f"FA-IL filter must match {required!r} in news il_status — "
            "the original code only matched 10-day / day-to-day so IL15/"
            "IL60 long-term injuries slipped through and got recommended "
            "as drop candidates."
        )


def test_authoritative_roster_status_check_present():
    """The unified filter also reads league_rosters.status authoritatively
    (not just relying on news strings, which can be stale or absent)."""
    src = _page_source()
    # Must query league_rosters joined to players, filtering on IL statuses.
    assert "league_rosters" in src and "IL10" in src and "IL15" in src and "IL60" in src, (
        "Expected authoritative IL check via league_rosters.status IN "
        "(IL10, IL15, IL60, ...) joined to players. This is the only "
        "layer that uses Yahoo's own roster status instead of news scraping."
    )


def test_filtered_recs_filtered_by_protected_names():
    """The filter must actually be applied — filtered_recs must be
    reassigned via comprehension excluding protected drops."""
    src = _page_source()
    # Look for the pattern: filtered_recs = [r for r in filtered_recs if r.get("drop_name", ...
    assert "protected_drop_names" in src and "drop_name" in src, (
        "Expected filtered_recs to be reassigned to exclude drops whose name is in protected_drop_names."
    )
    # Look for the specific filter idiom
    assert any(
        snippet in src
        for snippet in [
            'if r.get("drop_name", "") not in protected_drop_names',
            "if r.get('drop_name', '') not in protected_drop_names",
        ]
    ), (
        "Expected the literal filter comprehension. If the idiom changed, "
        "update this guard to match the new pattern but make sure the "
        "filter is still applied to filtered_recs."
    )


def test_filter_applied_before_section_1_render():
    """The filter must run BEFORE the top section renders — otherwise
    only the bottom section is protected (the regression we're fixing).

    The filter is applied right after `filtered_recs = _apply_pos_filter_recs(...)`
    and before the `if not filtered_recs:` branch that opens Section 1.
    AST-checking exact ordering is brittle; we settle for "the filter
    text appears earlier in the file than the top section's render."
    """
    src = _page_source()
    filter_idx = src.find("not in protected_drop_names")
    top_section_idx = src.find('"Recommended Adds/Drops"')
    # Both must exist...
    assert filter_idx > 0, "IL stash filter text not found"
    assert top_section_idx > 0, "Top section marker not found"
    # ...and the filter must appear BEFORE the section header
    # ALSO appears in a markdown render (so check the actual application).
    # The filter is applied right after _apply_pos_filter_recs, which is
    # after the markdown headers — but the filter REASSIGNS filtered_recs
    # before the loop iterates it (lines 487+). So instead of position
    # checking, verify the filter is structurally above the "for rec in
    # filtered_recs" loop.
    for_loop_idx = src.find("for rec in filtered_recs:")
    assert for_loop_idx > 0
    assert filter_idx < for_loop_idx, (
        "IL stash filter must reassign filtered_recs BEFORE the first "
        "`for rec in filtered_recs` loop. Otherwise the protection is "
        "skipped for the top section."
    )
