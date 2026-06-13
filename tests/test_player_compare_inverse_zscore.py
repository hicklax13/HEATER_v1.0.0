"""TDD tests for Task 5.1 (Player Compare slice):
Inverse-stat z-scores (ERA, WHIP, L) must show HIGHER values for the better player,
and the display table must annotate these stats with "(lower is better)".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

PAGE = Path(__file__).resolve().parents[1] / "pages" / "16_Player_Compare.py"
PAGE_TEXT = PAGE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit-level: compare_players z-score sign correctness
# ---------------------------------------------------------------------------


def _make_minimal_pool(stats_a: dict, stats_b: dict) -> pd.DataFrame:
    """Build a 4-row pool (2 test players + 2 filler pitchers) for compare_players."""
    base_pitcher = {
        "player_id": 0,
        "player_name": "Filler",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 10,
        "l": 8,
        "sv": 0,
        "k": 150,
        "era": 4.00,
        "whip": 1.30,
        "ip": 150,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    rows = []
    for pid, stats in [(1, stats_a), (2, stats_b)]:
        row = {**base_pitcher, "player_id": pid}
        row.update(stats)
        rows.append(row)
    # add filler rows for a non-degenerate mean/std
    for fid in [3, 4, 5, 6]:
        row = {**base_pitcher, "player_id": fid, "player_name": f"Filler{fid}"}
        rows.append(row)

    return pd.DataFrame(rows)


def test_better_era_gives_higher_zscore():
    """A pitcher with a lower (better) ERA must get a higher z-score than a worse pitcher.

    This verifies that compare_players already applies the sign flip correctly
    so the page display does not need to re-flip.
    """
    from src.in_season import compare_players
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()

    elite = {
        "player_id": 1,
        "player_name": "Elite SP",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 14,
        "l": 4,
        "sv": 0,
        "k": 200,
        "era": 2.50,
        "whip": 1.00,
        "ip": 180,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    mediocre = {
        "player_id": 2,
        "player_name": "Mediocre SP",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 8,
        "l": 10,
        "sv": 0,
        "k": 120,
        "era": 5.00,
        "whip": 1.50,
        "ip": 150,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }

    fillers = [
        {
            "player_id": fid,
            "player_name": f"Filler{fid}",
            "is_hitter": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.0,
            "obp": 0.0,
            "w": 10,
            "l": 8,
            "sv": 0,
            "k": 150,
            "era": 4.00,
            "whip": 1.30,
            "ip": 150,
            "ab": 0,
            "h": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
        }
        for fid in range(3, 10)
    ]

    pool = pd.DataFrame([elite, mediocre] + fillers)
    result = compare_players(1, 2, pool, cfg)

    assert "error" not in result, f"compare_players returned error: {result.get('error')}"

    z_elite_era = result["z_scores_a"]["ERA"]
    z_mediocre_era = result["z_scores_b"]["ERA"]

    assert z_elite_era > z_mediocre_era, (
        f"Task 5.1: Elite pitcher (ERA 2.50) z-score={z_elite_era:.3f} must be "
        f"higher than mediocre pitcher (ERA 5.00) z-score={z_mediocre_era:.3f}. "
        "For inverse stats, lower ERA → higher z-score (quality-adjusted)."
    )
    assert z_elite_era > 0, (
        f"Task 5.1: Elite pitcher (ERA 2.50) z-score should be positive (above average), got {z_elite_era:.3f}"
    )


def test_better_whip_gives_higher_zscore():
    """A pitcher with lower (better) WHIP must get a higher z-score."""
    from src.in_season import compare_players
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()

    elite = {
        "player_id": 1,
        "player_name": "Elite SP",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 12,
        "l": 6,
        "sv": 0,
        "k": 180,
        "era": 3.20,
        "whip": 0.90,
        "ip": 170,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    mediocre = {
        "player_id": 2,
        "player_name": "Mediocre SP",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 8,
        "l": 10,
        "sv": 0,
        "k": 100,
        "era": 4.50,
        "whip": 1.60,
        "ip": 140,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    fillers = [
        {
            "player_id": fid,
            "player_name": f"Filler{fid}",
            "is_hitter": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.0,
            "obp": 0.0,
            "w": 10,
            "l": 8,
            "sv": 0,
            "k": 150,
            "era": 4.00,
            "whip": 1.30,
            "ip": 150,
            "ab": 0,
            "h": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
        }
        for fid in range(3, 10)
    ]
    pool = pd.DataFrame([elite, mediocre] + fillers)
    result = compare_players(1, 2, pool, cfg)

    assert "error" not in result

    z_elite_whip = result["z_scores_a"]["WHIP"]
    z_mediocre_whip = result["z_scores_b"]["WHIP"]

    assert z_elite_whip > z_mediocre_whip, (
        f"Task 5.1: Elite pitcher (WHIP 0.90) z-score={z_elite_whip:.3f} must be "
        f"higher than mediocre pitcher (WHIP 1.60) z-score={z_mediocre_whip:.3f}."
    )


def test_fewer_losses_gives_higher_zscore():
    """A pitcher with fewer (better) Losses must get a higher z-score."""
    from src.in_season import compare_players
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()
    if "L" not in cfg.inverse_stats:
        pytest.skip("L not in inverse_stats for this config")

    ace = {
        "player_id": 1,
        "player_name": "Ace",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 14,
        "l": 3,
        "sv": 0,
        "k": 200,
        "era": 2.80,
        "whip": 1.00,
        "ip": 180,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    loser = {
        "player_id": 2,
        "player_name": "Loser",
        "is_hitter": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "obp": 0.0,
        "w": 6,
        "l": 14,
        "sv": 0,
        "k": 100,
        "era": 5.20,
        "whip": 1.60,
        "ip": 140,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
    }
    fillers = [
        {
            "player_id": fid,
            "player_name": f"Filler{fid}",
            "is_hitter": 0,
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.0,
            "obp": 0.0,
            "w": 10,
            "l": 8,
            "sv": 0,
            "k": 150,
            "era": 4.00,
            "whip": 1.30,
            "ip": 150,
            "ab": 0,
            "h": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
        }
        for fid in range(3, 10)
    ]
    pool = pd.DataFrame([ace, loser] + fillers)
    result = compare_players(1, 2, pool, cfg)

    assert "error" not in result

    z_ace_l = result["z_scores_a"]["L"]
    z_loser_l = result["z_scores_b"]["L"]

    assert z_ace_l > z_loser_l, (
        f"Task 5.1: Ace (L=3) z-score={z_ace_l:.3f} must be higher than "
        f"Loser (L=14) z-score={z_loser_l:.3f}. Fewer losses = higher z-score."
    )


# ---------------------------------------------------------------------------
# Page structural checks: category breakdown table + annotation
# ---------------------------------------------------------------------------


def test_category_breakdown_table_present():
    """The 'Category Breakdown' section must exist on the page."""
    assert "Category Breakdown" in PAGE_TEXT, (
        "Task 5.1: 'Category Breakdown' section header not found in 16_Player_Compare.py"
    )


def test_inverse_stat_annotation_in_breakdown_section():
    """The Category Breakdown section must annotate inverse stats.

    Acceptable forms:
      "(lower is better)"  or  "lower is better"  or  "(lower = better)"
    The annotation must appear in the breakdown section (after 'Category Breakdown').
    """
    breakdown_start = PAGE_TEXT.find("Category Breakdown")
    assert breakdown_start != -1

    # Look 3000 chars after the heading (covers the table + captions)
    after_breakdown = PAGE_TEXT[breakdown_start : breakdown_start + 3000]
    lower_text = after_breakdown.lower()

    has_annotation = "lower is better" in lower_text or "lower = better" in lower_text or "(lower)" in lower_text
    assert has_annotation, (
        "Task 5.1: No '(lower is better)' annotation found near the Category Breakdown section. "
        "Add a caption or column note so users know higher z-score = better even for ERA/WHIP/L."
    )


def test_inverse_stats_named_in_annotation():
    """The annotation must explicitly mention at least one inverse stat (ERA, WHIP, or L)."""
    breakdown_start = PAGE_TEXT.find("Category Breakdown")
    assert breakdown_start != -1
    after_breakdown = PAGE_TEXT[breakdown_start : breakdown_start + 3000]

    has_stat_name = bool(re.search(r"\b(ERA|WHIP|Losses?)\b", after_breakdown, re.IGNORECASE))
    assert has_stat_name, (
        "Task 5.1: The inverse-stat annotation must name at least one inverse stat "
        "(ERA, WHIP, or L/Losses) so users know which categories 'lower is better' applies to."
    )


def test_category_breakdown_z_score_format():
    """The Category Breakdown table must format z-scores with sign (+/-) and 2 decimals."""
    breakdown_start = PAGE_TEXT.find("Category Breakdown")
    assert breakdown_start != -1
    after_breakdown = PAGE_TEXT[breakdown_start : breakdown_start + 1500]

    # Format should still be +.2f (sign shown) so users see magnitude clearly
    has_format = "+.2f" in after_breakdown or ":+.2f" in after_breakdown
    assert has_format, (
        "Task 5.1: Category Breakdown z-score format must use +.2f (signed) so users "
        "can read magnitude. If this changed, restore it."
    )
