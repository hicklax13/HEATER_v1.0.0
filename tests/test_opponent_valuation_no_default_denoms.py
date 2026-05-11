"""C7 followup: opponent_valuation must not ship a DEFAULT_SGP_DENOMS fallback.

Background: C6 deleted the module-level ``_LC = LeagueConfig()`` singleton in
``src/engine/game_theory/opponent_valuation.py`` so live LeagueConfig values
flow through. C7 follows up by deleting the residual ``DEFAULT_SGP_DENOMS``
dict (whose values were stale: HR=12.0 vs LeagueConfig 13.0, K=25.0 vs 45.0,
SV=5.0 vs 9.0, etc.). This guard locks the deletion in so a future copy/paste
cannot reintroduce a numeric fallback that silently disagrees with
``LeagueConfig().sgp_denominators``.

The corollary contract enforced here: ``estimate_opponent_valuations`` and
``player_market_value`` REQUIRE ``sgp_denominators`` to be passed by the
caller. Production caller (``src/engine/output/trade_evaluator.py``) already
passes ``config.sgp_denominators`` explicitly.
"""

import re
from pathlib import Path

import pytest


def test_no_default_sgp_denoms_constant():
    """The DEFAULT_SGP_DENOMS dict must be removed from opponent_valuation.py."""
    text = Path("src/engine/game_theory/opponent_valuation.py").read_text(encoding="utf-8")
    text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    text_no_strings = re.sub(r'"""[\s\S]*?"""', "", text_no_comments)
    bad = re.findall(r"DEFAULT_SGP_DENOMS\s*=\s*\{", text_no_strings)
    assert bad == [], f"DEFAULT_SGP_DENOMS still defined: {bad}"


def test_estimate_opponent_valuations_requires_denoms():
    """Calling without sgp_denominators must raise (no silent fallback)."""
    from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

    with pytest.raises((TypeError, ValueError)):
        estimate_opponent_valuations(
            player_projections={"HR": 30},
            all_team_totals={"A": {"HR": 100}, "B": {"HR": 200}},
            your_team_id="A",
        )  # type: ignore[call-arg]


def test_estimate_opponent_valuations_rejects_empty_denoms():
    """Passing an empty dict must raise — empty is treated as missing."""
    from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

    with pytest.raises(ValueError):
        estimate_opponent_valuations(
            player_projections={"HR": 30},
            all_team_totals={"A": {"HR": 100}, "B": {"HR": 200}},
            your_team_id="A",
            sgp_denominators={},
        )


def test_player_market_value_requires_denoms():
    """player_market_value also requires sgp_denominators."""
    from src.engine.game_theory.opponent_valuation import player_market_value

    with pytest.raises((TypeError, ValueError)):
        player_market_value(
            player_projections={"HR": 30},
            all_team_totals={"A": {"HR": 100}, "B": {"HR": 200}},
            your_team_id="A",
        )  # type: ignore[call-arg]
