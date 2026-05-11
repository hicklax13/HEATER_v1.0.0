"""pages/1_My_Team.py must use LeagueConfig for inverse-stat sets."""

import re
from pathlib import Path


def test_no_inverse_set_dropping_l():
    """Inverse-stat sets that drop 'L' silently treat Losses as positive-good."""
    text = Path("pages/1_My_Team.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    # Specifically the {"ERA", "WHIP"} pattern as an INVERSE set
    bad = re.findall(
        r'_INVERSE_CATS\s*[:=].*?\{["\']ERA["\']\s*,\s*["\']WHIP["\']\}',
        text_no_comments,
    )
    assert bad == [], (
        f"Found inverse-stat constants that drop 'L': {bad}\nUse _LC.inverse_stats which includes L, ERA, WHIP."
    )


def test_no_full_hardcoded_category_list():
    """The 12-cat list literal is forbidden; derive from LeagueConfig."""
    text = Path("pages/1_My_Team.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    bad = re.findall(
        r'\[["\']R["\']\s*,\s*["\']HR["\']\s*,\s*["\']RBI["\']\s*,\s*["\']SB["\']',
        text_no_comments,
    )
    assert bad == [], f"Hardcoded 12-cat list: {bad}"
