"""app.py must derive all category lists from LeagueConfig."""

import re
from pathlib import Path


def test_app_no_hardcoded_full_category_list():
    """Twelve-cat literal lists are forbidden in app.py."""
    text = Path("app.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    # Pattern matches the 12-cat list literal verbatim
    bad = re.findall(
        r'\[?["\']R["\']\s*,\s*["\']HR["\']\s*,\s*["\']RBI["\']\s*,\s*["\']SB["\']\s*,'
        r'\s*["\']AVG["\']\s*,\s*["\']OBP["\']',
        text_no_comments,
    )
    assert bad == [], f"Hardcoded category list literals found in app.py: {bad}"


def test_app_inverse_set_includes_L():
    """app.py inverse-stat checks must include 'L' (Losses are inverse)."""
    text = Path("app.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    # Find any {"ERA","WHIP"} (without L) literal — these are bugs
    bad = re.findall(r'\{["\']ERA["\']\s*,\s*["\']WHIP["\']\}', text_no_comments)
    assert bad == [], (
        f"Found inverse-stat sets that drop L (Losses are inverse): {bad}\n"
        "Use _LC.inverse_stats which includes L, ERA, WHIP."
    )
