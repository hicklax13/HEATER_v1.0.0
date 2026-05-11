"""Engine modules must not define _LC singletons.

C6 cleanup of SF-21: the engine portfolio + game_theory modules previously
held import-time ``_LC = LeagueConfig()`` singletons that masked stale
denominators when callers were updated to pass live config. These guards lock
the deletion in so the singleton cannot creep back via copy/paste.

C7 (DEFAULT_SGP_DENOMS removal in opponent_valuation.py) was deferred:
the constant uses different numeric values than ``LeagueConfig().sgp_denominators``
and existing tests in tests/test_trade_engine_math.py assert math against those
specific values. Trade evaluator already passes ``config.sgp_denominators``
explicitly, so the live-config path works for production callers. Future task
should migrate the no-config callers (the test file) and then drop the dict.
"""

import re
from pathlib import Path


def test_no_lc_singleton_in_category_analysis():
    text = Path("src/engine/portfolio/category_analysis.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    # Match module-level _LC = LeagueConfig() or _LC = _LC_Class()
    bad = re.findall(r"^_LC\s*=\s*", text_no_comments, re.MULTILINE)
    assert bad == [], f"Found _LC singleton in category_analysis.py: {bad}"


def test_no_lc_singleton_in_copula():
    text = Path("src/engine/portfolio/copula.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    bad = re.findall(r"^_LC\s*=\s*", text_no_comments, re.MULTILINE)
    assert bad == [], f"Found _LC singleton in copula.py: {bad}"


def test_no_lc_singleton_in_opponent_valuation():
    """C6 portion of opponent_valuation.py — the _LC singleton MUST be gone.

    DEFAULT_SGP_DENOMS dict is intentionally retained (see module docstring
    and the file's own comment block); this test only enforces the singleton
    deletion.
    """
    text = Path("src/engine/game_theory/opponent_valuation.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    bad = re.findall(r"^_LC\s*=\s*", text_no_comments, re.MULTILINE)
    assert bad == [], f"Found _LC singleton in opponent_valuation.py: {bad}"
