"""BUG-019 fix: team-name → team-abbrev maps in Line-up Optimizer are consistent."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"


def test_athletics_maps_to_ath_everywhere():
    """Both `"Athletics"` and `"Oakland Athletics"` must map to `"ATH"`
    (canonical 2026 MLB Stats API team code, per CLAUDE.md and Wave 1).
    The earlier inconsistency (`"Oakland Athletics" → "OAK"` only) made
    the Streaming tab silently drop Athletics matchups (MLB API returns
    `"Athletics"`)."""
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    # 2026-05-18 Section 3 D3: also accept the indexing form
    # `_STREAM_TEAM_ABBREVS["Athletics"] = "ATH"` (the refactor that
    # routes through valuation.team_name_to_abbr but keeps explicit
    # Athletics-name registrations because both "Athletics" and
    # "Oakland Athletics" must map to "ATH").
    pat_dict = re.compile(r'"(Oakland\s+Athletics|Athletics)"\s*:\s*"([A-Z]+)"')
    pat_index = re.compile(r'_STREAM_TEAM_ABBREVS\["(Oakland\s+Athletics|Athletics)"\]\s*=\s*"([A-Z]+)"')
    mismatches: list[tuple[int, str, str]] = []
    seen_athletics_keys: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in pat_dict.finditer(line):
            name, code = m.group(1), m.group(2)
            seen_athletics_keys.add(name)
            if code != "ATH":
                mismatches.append((lineno, name, code))
        for m in pat_index.finditer(line):
            name, code = m.group(1), m.group(2)
            seen_athletics_keys.add(name)
            if code != "ATH":
                mismatches.append((lineno, name, code))
    assert not mismatches, (
        f"BUG-019 regression: Athletics mapped to wrong code. "
        f"Per Wave 1 finding (D1A-008), all Athletics name forms should "
        f"map to 'ATH' (canonical 2026 MLB Stats API). Offenders: {mismatches}"
    )
    assert seen_athletics_keys, (
        "No Athletics name→abbrev mapping found in pages/2_Line-up_Optimizer.py — did someone remove the team entirely?"
    )
