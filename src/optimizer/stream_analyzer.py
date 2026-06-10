"""Pitcher Streaming Analyzer engine.

Composition layer for the Pitcher Streaming page
(design: docs/superpowers/specs/2026-06-09-pitcher-streaming-analyzer-design.md).

Composes the existing canonical pieces — ``streaming.py`` (marginal SGP,
Bayesian stream score), ``two_start.py`` (pitcher-vs-opponent matchup score),
``matchup_adjustments.py`` (park, PvB), ``game_day.py`` (schedule, locked
statuses, team strength) — into a single 0-100 Stream Score per scheduled
start, plus the board/deep-dive/replay data structures the page renders.

Engine purity rules (guarded by tests/test_stream_analyzer_*.py):
- No streamlit imports.
- All tunable weights/thresholds read from CONSTANTS_REGISTRY at call time.
- The weekly add budget derives from league_rules.WEEKLY_TRANSACTION_LIMIT,
  never streaming.py's legacy alias.
- SGP flows through compute_streaming_value / SGPCalculator paths only —
  inverse stats (ERA/WHIP/L) must never get inline sign math here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.game_day import _NEUTRAL_DEFAULTS as _TEAM_NEUTRAL
from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

logger = logging.getLogger(__name__)

# The six Stream Score components, in registry order. Weights are read from
# CONSTANTS_REGISTRY at call time (never cached at import) so calibration
# tooling that patches the registry takes effect without a restart.
_SCORE_WEIGHT_KEYS: tuple[str, ...] = (
    "stream_score_w_matchup",
    "stream_score_w_sgp",
    "stream_score_w_form",
    "stream_score_w_lineup",
    "stream_score_w_env",
    "stream_score_w_winprob",
)


def _score_weights() -> dict[str, float]:
    """Return component-name → weight, read from the registry at call time."""
    return {key.removeprefix("stream_score_w_"): float(_CR[key].value) for key in _SCORE_WEIGHT_KEYS}


def get_opponent_offense_context(
    team_abbr: str,
    vs_throws: str | None,
    team_strength: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Opposing-offense snapshot for one scheduled start.

    Prefers a vs-handedness split (``wrc_plus_vs_lhp`` / ``wrc_plus_vs_rhp``
    keys, present when the optional team-splits fetch ran) and falls back to
    the overall team line; ``split_source`` tells the UI which one it got so
    the fallback can be footnoted.

    Args:
        team_abbr: Opposing team code (canonical MLB Stats API form).
        vs_throws: The streaming pitcher's throwing hand ("L"/"R"), or None.
        team_strength: ``ctx.team_strength``-shaped mapping
            (team → {"wrc_plus", "k_pct", "bb_pct", ...}; k_pct in percent).

    Returns:
        {"wrc_plus", "k_pct", "bb_pct", "iso", "l14_wrc_plus", "split_source"}
        — ``iso`` / ``l14_wrc_plus`` are None when the source lacks them;
        missing teams get the neutral league-average line.
    """
    entry: dict[str, Any] = {}
    if team_strength:
        entry = dict(team_strength.get(team_abbr) or {})

    hand = (vs_throws or "").upper()
    split_suffix = {"L": "vs_lhp", "R": "vs_rhp"}.get(hand)

    split_source = "overall"
    wrc_plus = entry.get("wrc_plus")
    k_pct = entry.get("k_pct")
    if split_suffix:
        split_wrc = entry.get(f"wrc_plus_{split_suffix}")
        if split_wrc is not None:
            wrc_plus = split_wrc
            k_pct = entry.get(f"k_pct_{split_suffix}", k_pct)
            split_source = "vs_hand"

    if wrc_plus is None:
        wrc_plus = _TEAM_NEUTRAL["wrc_plus"]
    if k_pct is None:
        k_pct = _TEAM_NEUTRAL["k_pct"]

    return {
        "wrc_plus": float(wrc_plus),
        "k_pct": float(k_pct),
        "bb_pct": float(entry.get("bb_pct", _TEAM_NEUTRAL["bb_pct"])),
        "iso": entry.get("iso"),
        "l14_wrc_plus": entry.get("l14_wrc_plus"),
        "split_source": split_source,
    }
