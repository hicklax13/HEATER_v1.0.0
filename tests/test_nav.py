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


def _patch_pages(monkeypatch, flags, in_season=True):
    monkeypatch.setattr("streamlit.Page", _FakePage)
    monkeypatch.setattr("src.feature_flags.list_page_flags", lambda: flags)
    monkeypatch.setattr("src.nav.is_in_season", lambda: in_season)


# ── Legacy group-name tests updated for new structure ──────────────────────────


def test_build_pages_groups_in_season(monkeypatch):
    """In-season: groups are Season + Preseason (+ Admin for admins), no Home."""
    _patch_pages(monkeypatch, {}, in_season=True)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 1}, draft_page=lambda: None)
    assert set(groups) == {"Season", "Preseason", "Admin"}
    assert len(groups["Admin"]) == 3


def test_build_pages_groups_preseason(monkeypatch):
    """Pre-season: Draft Tool is default under Home, Season has 13 pages."""
    _patch_pages(monkeypatch, {}, in_season=False)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 1}, draft_page=lambda: None)
    assert set(groups) == {"Home", "Season", "Admin"}
    # Season has all 14 registry entries (Draft Simulator stays in Season pre-season)
    # Draft Tool is in Home
    assert len(groups["Home"]) == 1
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


# ── New in-season nav tests ────────────────────────────────────────────────────


def test_build_pages_in_season_my_team_is_default(monkeypatch):
    """In-season: My Team page must be the default (default=True)."""
    _patch_pages(monkeypatch, {}, in_season=True)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    season_pages = groups["Season"]
    defaults = [p for p in season_pages if p.default]
    assert len(defaults) == 1, "Exactly one default page expected"
    assert defaults[0].title == "My Team"


def test_build_pages_in_season_draft_tool_in_preseason_group(monkeypatch):
    """In-season: Draft Tool is in the Preseason group, not Season."""
    _patch_pages(monkeypatch, {}, in_season=True)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    assert "Preseason" in groups
    preseason_titles = [p.title for p in groups["Preseason"]]
    assert "Draft Tool" in preseason_titles
    season_titles = [p.title for p in groups["Season"]]
    assert "Draft Tool" not in season_titles


def test_build_pages_in_season_draft_simulator_in_preseason_group(monkeypatch):
    """In-season: Draft Simulator is in the Preseason group, not Season."""
    _patch_pages(monkeypatch, {}, in_season=True)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    preseason_titles = [p.title for p in groups["Preseason"]]
    assert "Draft Simulator" in preseason_titles
    season_titles = [p.title for p in groups["Season"]]
    assert "Draft Simulator" not in season_titles


def test_build_pages_in_season_season_has_13_pages(monkeypatch):
    """In-season: Season group has 13 pages (Draft Simulator moved to Preseason)."""
    _patch_pages(monkeypatch, {}, in_season=True)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    assert len(groups["Season"]) == 13


def test_build_pages_preseason_draft_tool_is_default(monkeypatch):
    """Pre-season: Draft Tool must be the default page."""
    _patch_pages(monkeypatch, {}, in_season=False)
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    home_pages = groups["Home"]
    defaults = [p for p in home_pages if p.default]
    assert len(defaults) == 1
    assert defaults[0].title == "Draft Tool"


def test_draft_simulator_eyebrow_is_scouting():
    """Draft Simulator page must use SCOUTING as the eyebrow, not PRESEASON."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "pages" / "20_Draft_Simulator.py").read_text(encoding="utf-8")
    # Look for render_page_header call with eyebrow argument
    tree = ast.parse(src)
    found_scouting = False
    found_preseason = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            if fname == "render_page_header":
                for kw in node.keywords:
                    if kw.arg == "eyebrow" and isinstance(kw.value, ast.Constant):
                        val = kw.value.value
                        if val == "SCOUTING":
                            found_scouting = True
                        if val == "PRESEASON":
                            found_preseason = True
    assert found_scouting, "render_page_header eyebrow must be 'SCOUTING'"
    assert not found_preseason, "render_page_header eyebrow must NOT be 'PRESEASON'"
