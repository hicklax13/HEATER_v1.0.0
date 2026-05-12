"""BUG-009 fix: pages/9_League_Standings + pages/7_Player_Compare route
roster fetches through get_yahoo_data_service, not raw SQL helpers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scan_for_call(path: Path, fn_name: str) -> list[tuple[int, str]]:
    """Return (line_no, line) for non-comment, non-import lines that call fn_name(."""
    if not path.exists():
        return []
    bad: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(("from ", "import ")):
            continue
        if f"{fn_name}(" in line:
            bad.append((lineno, stripped))
    return bad


def test_league_standings_uses_yds_for_rosters():
    """pages/9_League_Standings.py must NOT call load_league_rosters directly.
    Use yds.get_rosters() instead. load_league_records/schedule remain OK
    because YDS doesn't expose those."""
    bad = _scan_for_call(REPO_ROOT / "pages" / "9_League_Standings.py", "load_league_rosters")
    assert not bad, (
        f"BUG-009 regression: pages/9_League_Standings.py calls "
        f"load_league_rosters() directly. Route through yds.get_rosters(). "
        f"Offenders: {bad}"
    )


def test_player_compare_uses_yds_for_rosters():
    """pages/7_Player_Compare.py must NOT call load_league_rosters directly."""
    bad = _scan_for_call(REPO_ROOT / "pages" / "7_Player_Compare.py", "load_league_rosters")
    assert not bad, (
        f"BUG-009 regression: pages/7_Player_Compare.py calls load_league_rosters() directly. Offenders: {bad}"
    )
