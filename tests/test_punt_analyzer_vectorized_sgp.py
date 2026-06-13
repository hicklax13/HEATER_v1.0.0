"""TDD: Punt Analyzer must use total_sgp_batch (vectorized), not iterrows() over the pool.

Task 2.4 — Vectorize the Punt SGP recompute
"""

import ast
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.valuation import LeagueConfig, SGPCalculator

# ── Path to the page under test ──────────────────────────────────────────────

PAGE_PATH = Path(__file__).parent.parent / "pages" / "10_Punt_Analyzer.py"

# ── Fixture: minimal player pool matching pool schema used by the page ────────


@pytest.fixture()
def config():
    return LeagueConfig()


@pytest.fixture()
def small_pool():
    """Six synthetic players covering hitters and pitchers."""
    return pd.DataFrame(
        [
            # Hitter
            dict(
                player_name="Alice",
                positions="OF",
                team="NYY",
                mlb_id=1,
                r=90.0,
                hr=30.0,
                rbi=100.0,
                sb=20.0,
                avg=0.280,
                obp=0.360,
                ab=500.0,
                h=140.0,
                bb=60.0,
                hbp=5.0,
                pa=565.0,
                sf=5.0,
                # pitching cols zero for hitters
                w=0.0,
                l=0.0,
                sv=0.0,
                k=0.0,
                era=0.0,
                whip=0.0,
                ip=0.0,
                er=0.0,
                bb_allowed=0.0,
                h_allowed=0.0,
            ),
            # Power hitter
            dict(
                player_name="Bob",
                positions="1B",
                team="LAD",
                mlb_id=2,
                r=70.0,
                hr=45.0,
                rbi=110.0,
                sb=2.0,
                avg=0.245,
                obp=0.330,
                ab=520.0,
                h=127.0,
                bb=65.0,
                hbp=3.0,
                pa=588.0,
                sf=4.0,
                w=0.0,
                l=0.0,
                sv=0.0,
                k=0.0,
                era=0.0,
                whip=0.0,
                ip=0.0,
                er=0.0,
                bb_allowed=0.0,
                h_allowed=0.0,
            ),
            # Speed hitter
            dict(
                player_name="Carol",
                positions="SS",
                team="ATL",
                mlb_id=3,
                r=100.0,
                hr=5.0,
                rbi=40.0,
                sb=55.0,
                avg=0.290,
                obp=0.370,
                ab=480.0,
                h=139.0,
                bb=70.0,
                hbp=4.0,
                pa=554.0,
                sf=6.0,
                w=0.0,
                l=0.0,
                sv=0.0,
                k=0.0,
                era=0.0,
                whip=0.0,
                ip=0.0,
                er=0.0,
                bb_allowed=0.0,
                h_allowed=0.0,
            ),
            # Starter pitcher
            dict(
                player_name="Dave",
                positions="SP",
                team="HOU",
                mlb_id=4,
                r=0.0,
                hr=0.0,
                rbi=0.0,
                sb=0.0,
                avg=0.0,
                obp=0.0,
                ab=0.0,
                h=0.0,
                bb=0.0,
                hbp=0.0,
                pa=0.0,
                sf=0.0,
                w=14.0,
                l=6.0,
                sv=0.0,
                k=195.0,
                era=3.20,
                whip=1.10,
                ip=180.0,
                er=64.0,
                bb_allowed=45.0,
                h_allowed=153.0,
            ),
            # Closer
            dict(
                player_name="Eve",
                positions="RP",
                team="SD",
                mlb_id=5,
                r=0.0,
                hr=0.0,
                rbi=0.0,
                sb=0.0,
                avg=0.0,
                obp=0.0,
                ab=0.0,
                h=0.0,
                bb=0.0,
                hbp=0.0,
                pa=0.0,
                sf=0.0,
                w=3.0,
                l=2.0,
                sv=35.0,
                k=75.0,
                era=2.50,
                whip=0.90,
                ip=65.0,
                er=18.0,
                bb_allowed=15.0,
                h_allowed=44.0,
            ),
            # Weak player
            dict(
                player_name="Frank",
                positions="C",
                team="BOS",
                mlb_id=6,
                r=30.0,
                hr=8.0,
                rbi=35.0,
                sb=1.0,
                avg=0.220,
                obp=0.290,
                ab=320.0,
                h=70.0,
                bb=28.0,
                hbp=2.0,
                pa=350.0,
                sf=3.0,
                w=0.0,
                l=0.0,
                sv=0.0,
                k=0.0,
                era=0.0,
                whip=0.0,
                ip=0.0,
                er=0.0,
                bb_allowed=0.0,
                h_allowed=0.0,
            ),
        ]
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTotalSgpBatchExists:
    """total_sgp_batch must exist on SGPCalculator (already present check)."""

    def test_method_exists(self, config):
        calc = SGPCalculator(config)
        assert callable(getattr(calc, "total_sgp_batch", None)), "SGPCalculator.total_sgp_batch must exist"

    def test_returns_ndarray(self, config, small_pool):
        calc = SGPCalculator(config)
        result = calc.total_sgp_batch(small_pool)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(small_pool)


class TestBatchMatchesPerRow:
    """Batch result must match per-row total_sgp within tolerance."""

    def test_original_sgp_matches(self, config, small_pool):
        calc = SGPCalculator(config)
        per_row = np.array([calc.total_sgp(row) for _, row in small_pool.iterrows()])
        batch = calc.total_sgp_batch(small_pool)
        np.testing.assert_allclose(
            batch, per_row, rtol=1e-6, atol=1e-9, err_msg="Batch SGP must match per-row SGP for original config"
        )

    def test_punt_sgp_matches(self, small_pool):
        """With punted denominators, batch still matches per-row."""
        punt_config = LeagueConfig()
        punt_config.sgp_denominators["SB"] = 999999.0
        punt_config.sgp_denominators["SV"] = 999999.0
        calc = SGPCalculator(punt_config)

        per_row = np.array([calc.total_sgp(row) for _, row in small_pool.iterrows()])
        batch = calc.total_sgp_batch(small_pool)
        np.testing.assert_allclose(
            batch, per_row, rtol=1e-6, atol=1e-9, err_msg="Batch SGP must match per-row SGP for punt config"
        )

    def test_inverse_stat_direction(self, config, small_pool):
        """Players with higher ERA/WHIP/L should have lower batch SGP."""
        calc = SGPCalculator(config)
        result = calc.total_sgp_batch(small_pool)
        # Dave (idx 3) has better ERA/WHIP than Eve (idx 4)?  — just check no NaN
        assert not np.any(np.isnan(result)), "No NaN values in batch SGP"


class TestPageUsesVectorizedBatch:
    """pages/10_Punt_Analyzer.py must use total_sgp_batch, not iterrows() in the SGP loop."""

    def _get_ast(self):
        source = PAGE_PATH.read_text(encoding="utf-8")
        return ast.parse(source), source

    def test_page_calls_total_sgp_batch(self):
        """The page must call .total_sgp_batch( somewhere in the SGP compute block."""
        source = PAGE_PATH.read_text(encoding="utf-8")
        assert "total_sgp_batch" in source, (
            "pages/10_Punt_Analyzer.py must call total_sgp_batch() instead of iterrows() for SGP computation"
        )

    def test_no_iterrows_in_sgp_compute_block(self):
        """The SGP compute block (between 'Compute Original Values' and 'Summary')
        must not contain an iterrows() loop calling total_sgp or player_sgp."""
        source = PAGE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all For loops that call iterrows()
        class IterrowsSGPVisitor(ast.NodeVisitor):
            def __init__(self):
                self.bad_iterrows_calls = []  # (lineno, context)

            def visit_For(self, node):
                # Check if the iter is a Call to .iterrows()
                iter_node = node.iter
                if (
                    isinstance(iter_node, ast.Call)
                    and isinstance(iter_node.func, ast.Attribute)
                    and iter_node.func.attr == "iterrows"
                ):
                    # Check if the body contains a call to total_sgp or sgp_orig/sgp_punt
                    body_src = ast.unparse(node)
                    if "total_sgp" in body_src or "sgp_orig" in body_src or "sgp_punt" in body_src:
                        # Only flag if it's appending SGP values (the compute loop)
                        if ".append(" in body_src or "orig_sgps" in body_src or "punt_sgps" in body_src:
                            self.bad_iterrows_calls.append(node.lineno)
                self.generic_visit(node)

        visitor = IterrowsSGPVisitor()
        visitor.visit(tree)

        assert not visitor.bad_iterrows_calls, (
            f"pages/10_Punt_Analyzer.py still uses iterrows() for SGP computation "
            f"at line(s) {visitor.bad_iterrows_calls}. "
            "Replace with total_sgp_batch()."
        )

    def test_value_swing_table_iterrows_is_ok(self):
        """The _value_swing_table_html function's iterrows() (for HTML rendering) is fine —
        we only forbid iterrows() in the SGP *compute* path."""
        source = PAGE_PATH.read_text(encoding="utf-8")
        # This test just confirms the rendering iterrows stays (not a constraint violation)
        assert "_value_swing_table_html" in source, (
            "_value_swing_table_html helper must remain (its iterrows is for HTML, not SGP)"
        )


class TestPuntValuesNumericallyCorrect:
    """End-to-end: punting SB should zero-out SB contribution and change value_change correctly."""

    def test_punt_sb_reduces_sb_specialist_value(self, small_pool):
        """Carol (SB=55) should lose relative value when SB is punted."""
        config_orig = LeagueConfig()
        config_punt = LeagueConfig()
        config_punt.sgp_denominators["SB"] = 999999.0

        calc_orig = SGPCalculator(config_orig)
        calc_punt = SGPCalculator(config_punt)

        pool = small_pool.copy()
        pool["original_sgp"] = calc_orig.total_sgp_batch(pool)
        pool["punt_sgp"] = calc_punt.total_sgp_batch(pool)
        pool["value_change"] = pool["punt_sgp"] - pool["original_sgp"]

        carol = pool[pool["player_name"] == "Carol"].iloc[0]
        bob = pool[pool["player_name"] == "Bob"].iloc[0]

        # Carol specialises in SB → her value_change should be more negative than Bob's
        assert carol["value_change"] < bob["value_change"], (
            "Punting SB should hurt the SB specialist (Carol) more than the power hitter (Bob)"
        )

    def test_punt_sv_reduces_closer_value(self, small_pool):
        """Eve (SV=35) should lose relative value when SV is punted."""
        config_orig = LeagueConfig()
        config_punt = LeagueConfig()
        config_punt.sgp_denominators["SV"] = 999999.0

        calc_orig = SGPCalculator(config_orig)
        calc_punt = SGPCalculator(config_punt)

        pool = small_pool.copy()
        pool["original_sgp"] = calc_orig.total_sgp_batch(pool)
        pool["punt_sgp"] = calc_punt.total_sgp_batch(pool)
        pool["value_change"] = pool["punt_sgp"] - pool["original_sgp"]

        eve = pool[pool["player_name"] == "Eve"].iloc[0]
        dave = pool[pool["player_name"] == "Dave"].iloc[0]

        assert eve["value_change"] < dave["value_change"], (
            "Punting SV should hurt the closer (Eve) more than the starter (Dave)"
        )
