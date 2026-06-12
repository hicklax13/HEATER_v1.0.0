"""nav registry stays in sync with disk; build_pages assembles role-aware st.Page groups."""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def test_filter_enabled_pages_drops_disabled():
    from src.nav import filter_enabled_pages

    keys = ["1_My_Team", "17_Leaders"]
    assert filter_enabled_pages(keys, {"1_My_Team": False}) == ["17_Leaders"]


def test_filter_enabled_pages_absence_is_enabled():
    from src.nav import filter_enabled_pages

    assert filter_enabled_pages(["a", "b"], {}) == ["a", "b"]


def test_registry_matches_disk():
    from src.nav import PAGE_REGISTRY

    disk_stems = {p.stem for p in _PAGES_DIR.glob("*.py") if not p.name.startswith("_")}
    registry_keys = {e["key"] for e in PAGE_REGISTRY}
    assert registry_keys == disk_stems


class _FakePage:
    def __init__(self, page, title=None, default=False, icon=None):
        self.page = page
        self.title = title
        self.default = default


def _patch_pages(monkeypatch, flags):
    monkeypatch.setattr("streamlit.Page", _FakePage)
    monkeypatch.setattr("src.feature_flags.list_page_flags", lambda: flags)


def test_build_pages_groups(monkeypatch):
    _patch_pages(monkeypatch, {})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 1}, draft_page=lambda: None)
    assert set(groups) == {"Home", "Season", "Admin"}
    assert len(groups["Season"]) == 14
    assert len(groups["Admin"]) == 3


def test_build_pages_no_admin_for_non_admin(monkeypatch):
    _patch_pages(monkeypatch, {})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    assert "Admin" not in groups


def test_build_pages_respects_disabled_flag(monkeypatch):
    _patch_pages(monkeypatch, {"page:1_My_Team": False})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    season_paths = [p.page for p in groups["Season"]]
    assert "pages/1_My_Team.py" not in season_paths
