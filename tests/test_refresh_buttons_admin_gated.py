"""Refresh/Sync buttons that write to the shared DB must be gated to writers.

Under MULTI_USER the scheduler is the sole SQLite writer and members are
read-only, so member-visible "refresh" buttons that write-through to the DB are
a multi-writer hazard ("database is locked" / corruption). Each must be gated by
``viewer_can_write()`` (true in single-user v1, or for admins under MULTI_USER).
2026-06-01 pre-launch audit, launch-blocker #1.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

# (page filename, exact gated-button substring that must be present)
_GATED_BUTTONS = [
    ("1_My_Team.py", 'viewer_can_write() and st.button("Refresh Stats")'),
    ("1_My_Team.py", 'viewer_can_write() and st.button("Sync Yahoo"'),
    ("2_Line-up_Optimizer.py", "viewer_can_write() and st.button("),
    ("17_Leaders.py", 'viewer_can_write() and st.button("Refresh Data"'),
]


@pytest.mark.parametrize("page,snippet", _GATED_BUTTONS)
def test_refresh_button_is_writer_gated(page, snippet):
    src = (_PAGES_DIR / page).read_text(encoding="utf-8")
    assert snippet in src, (
        f"{page}: a DB-writing refresh button must be gated with `{snippet}` so "
        "members can't trigger shared-DB writes under MULTI_USER"
    )


@pytest.mark.parametrize("page", ["1_My_Team.py", "2_Line-up_Optimizer.py", "17_Leaders.py"])
def test_page_imports_viewer_can_write(page):
    src = (_PAGES_DIR / page).read_text(encoding="utf-8")
    assert "viewer_can_write" in src and "from src.auth import" in src, (
        f"{page} must import viewer_can_write from src.auth"
    )
