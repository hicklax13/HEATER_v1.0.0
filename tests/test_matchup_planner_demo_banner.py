"""BUG-024 fix: Matchup Planner page surfaces a warning when falling back to demo roster."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "12_Matchup_Planner.py"


def test_demo_roster_path_emits_warning():
    """The fallback path that uses pool.head(...) as a demo roster must be
    accompanied by an st.warning/st.info call so the user knows the
    ratings are against demo data, not their real team.

    Heuristic: find any `pool.head(` line in the file; the immediately
    surrounding ±5 lines must contain `st.warning(` or `st.info(` or `st.error(`.
    """
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    lines = text.splitlines()
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "pool.head(" in line:
            start = max(0, lineno - 6)
            end = min(len(lines), lineno + 5)
            context = "\n".join(lines[start:end])
            if not any(marker in context for marker in ("st.warning(", "st.info(", "st.error(")):
                offenders.append((lineno, stripped))
    assert not offenders, (
        f"BUG-024 regression: pool.head(...) used as demo roster without "
        f"a user-visible st.warning/st.info banner within ±5 lines. "
        f"Offenders: {offenders}"
    )
