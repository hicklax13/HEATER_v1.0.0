"""sensitivity.py CATEGORIES must come from LeagueConfig."""

import re
from pathlib import Path

from src.engine.game_theory.sensitivity import CATEGORIES, INVERSE_CATEGORIES
from src.valuation import LeagueConfig


def test_categories_match_league_config():
    cfg = LeagueConfig()
    # Match either canonical order OR justified custom order with same membership
    assert set(CATEGORIES) == set(cfg.all_categories), (
        f"CATEGORIES set membership diverges from LeagueConfig:\n"
        f"  sensitivity: {sorted(CATEGORIES)}\n"
        f"  LeagueConfig: {sorted(cfg.all_categories)}"
    )


def test_inverse_categories_match_league_config():
    cfg = LeagueConfig()
    assert set(INVERSE_CATEGORIES) == set(cfg.inverse_stats), (
        "INVERSE_CATEGORIES diverges from LeagueConfig.inverse_stats"
    )


def test_no_hardcoded_inverse_drops_L():
    text = Path("src/engine/game_theory/sensitivity.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    bad = re.findall(r'\{["\']ERA["\']\s*,\s*["\']WHIP["\']\}', text_no_comments)
    assert bad == [], f"Found inverse sets dropping L: {bad}"
