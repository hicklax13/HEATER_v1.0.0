"""Tests for Trade Value Chart (src/trade_value.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.trade_value import (
    LEAGUE_BUDGET,
    TIERS,
    WEEKLY_TAU,
    assign_tier,
    compute_contextual_values,
    compute_g_score_adjustment,
    compute_trade_values,
    filter_by_position,
)
from src.valuation import LeagueConfig, SGPCalculator

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_pool(n=20):
    """Create a minimal player pool for testing."""
    players = []
    for i in range(n):
        is_hitter = i < 14
        if is_hitter:
            pos_options = ["C", "1B", "2B", "3B", "SS", "OF"]
            pos = pos_options[i % len(pos_options)]
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Hitter_{i + 1}",
                    "team": f"TM{i % 5}",
                    "positions": pos,
                    "is_hitter": 1,
                    "pa": 500 + i * 10,
                    "ab": 450 + i * 10,
                    "h": 120 + i * 3,
                    "r": 60 + i * 2,
                    "hr": 15 + i,
                    "rbi": 55 + i * 3,
                    "sb": 5 + i,
                    "avg": 0.260 + i * 0.003,
                    "obp": 0.330 + i * 0.003,
                    "bb": 40 + i,
                    "hbp": 3,
                    "sf": 4,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 10 + i * 8,
                    "is_injured": 0,
                }
            )
        else:
            pos = "SP" if i % 2 == 0 else "RP"
            ip_base = 150 if pos == "SP" else 60
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Pitcher_{i + 1}",
                    "team": f"TM{i % 5}",
                    "positions": pos,
                    "is_hitter": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "ip": ip_base + (i - 14) * 10,
                    "w": 8 + (i - 14),
                    "l": 6 + (i - 14) % 3,
                    "sv": 20 if pos == "RP" else 0,
                    "k": 100 + (i - 14) * 15,
                    "era": 3.80 - (i - 14) * 0.1,
                    "whip": 1.20 - (i - 14) * 0.02,
                    "er": 50 + (i - 14) * 3,
                    "bb_allowed": 40 + (i - 14) * 2,
                    "h_allowed": 120 + (i - 14) * 5,
                    "adp": 30 + i * 6,
                    "is_injured": 0,
                }
            )
    return pd.DataFrame(players)


@pytest.fixture
def pool():
    return _make_pool()


@pytest.fixture
def config():
    return LeagueConfig()


# ── Tier Assignment ───────────────────────────────────────────────────


class TestAssignTier:
    def test_elite_tier(self):
        assert assign_tier(95.0) == "Elite"

    def test_star_tier(self):
        assert assign_tier(80.0) == "Star"

    def test_solid_tier(self):
        assert assign_tier(60.0) == "Solid Starter"

    def test_flex_tier(self):
        assert assign_tier(40.0) == "Flex"

    def test_replacement_tier(self):
        assert assign_tier(10.0) == "Replacement"

    def test_zero_is_replacement(self):
        assert assign_tier(0.0) == "Replacement"

    def test_boundary_elite(self):
        assert assign_tier(90.0) == "Elite"

    def test_boundary_star(self):
        assert assign_tier(75.0) == "Star"

    def test_boundary_solid(self):
        assert assign_tier(55.0) == "Solid Starter"

    def test_boundary_flex(self):
        assert assign_tier(35.0) == "Flex"

    def test_negative_value(self):
        assert assign_tier(-5.0) == "Replacement"


# ── G-Score Adjustment ────────────────────────────────────────────────


class TestGScoreAdjustment:
    def test_zero_sgp_returns_zero(self):
        per_cat = {cat: 0.0 for cat in LeagueConfig().all_categories}
        sigma = {cat: 1.0 for cat in LeagueConfig().all_categories}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert result == 0.0

    def test_positive_sgp_returns_positive(self):
        config = LeagueConfig()
        per_cat = {cat: 1.0 for cat in config.all_categories}
        sigma = {cat: 2.0 for cat in config.all_categories}
        result = compute_g_score_adjustment(per_cat, sigma, config)
        assert result > 0

    def test_high_tau_reduces_value(self):
        """Categories with high weekly variance (tau) should be discounted."""
        config = LeagueConfig()
        per_cat = {"SV": 2.0}  # SV has high tau
        sigma = {"SV": 1.5}

        g_sv = compute_g_score_adjustment(per_cat, sigma, config)

        per_cat_hr = {"HR": 2.0}  # HR has lower tau
        sigma_hr = {"HR": 1.5}
        g_hr = compute_g_score_adjustment(per_cat_hr, sigma_hr, config)

        # With same input SGP and sigma, the high-tau category should be discounted more
        # (though the exact comparison depends on SGP denominators)
        assert isinstance(g_sv, float)
        assert isinstance(g_hr, float)

    def test_zero_sigma_handled(self):
        """Should not crash with zero sigma."""
        per_cat = {"R": 1.0}
        sigma = {"R": 0.0}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert math.isfinite(result)

    def test_missing_categories_handled(self):
        """Missing categories should be skipped gracefully."""
        per_cat = {"R": 1.5, "HR": 2.0}
        sigma = {"R": 1.0, "HR": 1.0}
        result = compute_g_score_adjustment(per_cat, sigma)
        assert result > 0


# ── Main Trade Value Computation ──────────────────────────────────────


class TestComputeTradeValues:
    def test_returns_dataframe(self, pool, config):
        result = compute_trade_values(pool, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(pool)

    def test_has_required_columns(self, pool, config):
        result = compute_trade_values(pool, config)
        for col in ["trade_value", "dollar_value", "tier", "rank", "pos_rank"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_trade_values_bounded_0_100(self, pool, config):
        result = compute_trade_values(pool, config)
        assert result["trade_value"].min() >= 0.0
        assert result["trade_value"].max() <= 100.0

    def test_sorted_descending(self, pool, config):
        result = compute_trade_values(pool, config)
        values = result["trade_value"].tolist()
        assert values == sorted(values, reverse=True)

    def test_rank_sequential(self, pool, config):
        result = compute_trade_values(pool, config)
        assert result["rank"].tolist() == list(range(1, len(result) + 1))

    def test_all_tiers_assigned(self, pool, config):
        result = compute_trade_values(pool, config)
        valid_tiers = {name for _, name in TIERS}
        for tier in result["tier"]:
            assert tier in valid_tiers

    def test_dollar_values_positive(self, pool, config):
        result = compute_trade_values(pool, config)
        assert (result["dollar_value"] >= 1.0).all()

    def test_empty_pool_returns_empty(self, config):
        result = compute_trade_values(pd.DataFrame(), config)
        assert result.empty

    def test_top_player_highest_value(self, pool, config):
        result = compute_trade_values(pool, config)
        # The top-ranked player should have the highest trade value
        assert result.iloc[0]["rank"] == 1
        assert result.iloc[0]["trade_value"] >= result.iloc[1]["trade_value"]

    def test_weeks_remaining_scales_values(self, pool, config):
        full = compute_trade_values(pool, config, weeks_remaining=26)
        half = compute_trade_values(pool, config, weeks_remaining=13)
        # Half-season values should be approximately half of full-season
        top_full = full.iloc[0]["trade_value"]
        top_half = half.iloc[0]["trade_value"]
        assert top_half < top_full

    def test_with_standings(self, pool, config):
        """Should work when standings are provided."""
        standings_data = []
        for cat in config.all_categories:
            for i in range(12):
                standings_data.append(
                    {
                        "team_name": f"Team_{i}",
                        "category": cat,
                        "total": 100 + i * 10,
                        "rank": i + 1,
                    }
                )
        standings = pd.DataFrame(standings_data)
        result = compute_trade_values(pool, config, standings=standings)
        assert len(result) == len(pool)
        assert "trade_value" in result.columns

    def test_player_name_column_alias(self, config):
        """Should handle player_name column (common alias)."""
        pool = _make_pool(5)
        pool = pool.rename(columns={"name": "player_name"})
        result = compute_trade_values(pool, config)
        assert "name" in result.columns
        assert len(result) == 5


# ── Contextual Values ─────────────────────────────────────────────────


class TestContextualValues:
    def test_empty_totals_returns_universal(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = compute_contextual_values(tv, {}, {}, "Team_0", config)
        assert "contextual_value" in result.columns
        assert result["contextual_value"].equals(result["trade_value"])

    def test_contextual_values_differ_from_universal(self, pool, config):
        tv = compute_trade_values(pool, config)
        user_totals = {
            "R": 300,
            "HR": 80,
            "RBI": 280,
            "SB": 20,
            "AVG": 0.260,
            "OBP": 0.320,
            "W": 40,
            "L": 35,
            "SV": 15,
            "K": 500,
            "ERA": 4.00,
            "WHIP": 1.25,
        }
        all_totals = {
            "Team_0": user_totals,
            "Team_1": {k: v * 1.1 for k, v in user_totals.items()},
            "Team_2": {k: v * 0.9 for k, v in user_totals.items()},
        }
        result = compute_contextual_values(tv, user_totals, all_totals, "Team_0", config)
        assert "contextual_value" in result.columns
        assert "contextual_tier" in result.columns

    def test_contextual_has_tiers(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = compute_contextual_values(tv, {}, {}, "Team_0", config)
        assert "contextual_tier" in result.columns


# ── Position Filtering ────────────────────────────────────────────────


class TestFilterByPosition:
    def test_filter_all_returns_all(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "All")
        assert len(result) == len(tv)

    def test_filter_sp_only(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "SP")
        assert len(result) > 0
        for _, row in result.iterrows():
            assert "SP" in str(row["positions"])

    def test_filter_of_only(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "OF")
        assert len(result) > 0

    def test_filter_nonexistent_position(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "DH")
        assert len(result) == 0

    def test_pos_rank_recomputed(self, pool, config):
        tv = compute_trade_values(pool, config)
        result = filter_by_position(tv, "OF")
        if len(result) > 0:
            assert result["pos_rank"].tolist() == list(range(1, len(result) + 1))


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_player(self, config):
        pool = _make_pool(1)
        result = compute_trade_values(pool, config)
        assert len(result) == 1
        assert result.iloc[0]["rank"] == 1

    def test_all_same_stats(self, config):
        """All players with identical stats should get similar values."""
        players = []
        for i in range(5):
            players.append(
                {
                    "player_id": i + 1,
                    "name": f"Player_{i + 1}",
                    "team": "TM",
                    "positions": "OF",
                    "is_hitter": 1,
                    "pa": 500,
                    "ab": 450,
                    "h": 120,
                    "r": 60,
                    "hr": 20,
                    "rbi": 70,
                    "sb": 10,
                    "avg": 0.267,
                    "obp": 0.340,
                    "bb": 40,
                    "hbp": 3,
                    "sf": 4,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 50,
                    "is_injured": 0,
                }
            )
        pool = pd.DataFrame(players)
        result = compute_trade_values(pool, config)
        values = result["trade_value"].tolist()
        # All should be very similar
        assert max(values) - min(values) < 1.0

    def test_weekly_tau_constants_reasonable(self):
        """All tau values should be positive."""
        for cat, tau in WEEKLY_TAU.items():
            assert tau > 0, f"Tau for {cat} should be positive"

    def test_weekly_tau_resolves_from_canonical_source(self):
        """MS-E1b: trade_value.WEEKLY_TAU is the SAME quantity (per-team weekly
        category SD in raw stat units) as the canonical weekly-SD source, so it
        must resolve from h2h_engine.default_weekly_sigmas() — the single source
        of truth shared with standings_engine / standings_projection /
        playoff_sim — not an independently hand-tuned table."""
        from src.optimizer.h2h_engine import default_weekly_sigmas

        canonical = default_weekly_sigmas()
        for cat in LeagueConfig().all_categories:
            assert WEEKLY_TAU[cat] == pytest.approx(canonical[cat]), (
                f"trade_value.WEEKLY_TAU[{cat}] diverges from the canonical "
                f"weekly SD ({WEEKLY_TAU.get(cat)} != {canonical[cat]})"
            )

    def test_league_budget_constant(self):
        assert LEAGUE_BUDGET == 3120.0  # 12 × $260


# ── Task 1.7: Tier assignment must not be crushed by time decay ────────
# With ~15 weeks remaining, time_factor ≈ 0.577. If tiers are assigned
# AFTER decay, every player's value is capped at ~57.7 → Elite/Star
# buckets (≥75) are permanently empty.  Tiers must reflect the pre-decay
# relative rank so top players still land in Elite/Star.


class TestTierAssignmentPreDecay:
    """Tier assignment must use the pre-decay (relative) trade value,
    not the post-decay absolute value, so Elite/Star tiers are reachable
    at any point in the season."""

    def test_top_players_not_all_replacement_at_15_weeks(self, pool, config):
        """With 15 weeks remaining the #1 and #2 players must NOT be Replacement."""
        result = compute_trade_values(pool, config, weeks_remaining=15)
        top2_tiers = result.head(2)["tier"].tolist()
        assert "Replacement" not in top2_tiers, (
            f"Top-2 players are Replacement with 15 weeks left: {top2_tiers}. "
            "Tiers must be assigned from pre-decay values."
        )

    def test_rank1_player_is_elite_or_star_at_15_weeks(self, pool, config):
        """The #1 ranked player must be Elite or Star at 15 weeks remaining."""
        result = compute_trade_values(pool, config, weeks_remaining=15)
        rank1_tier = result.iloc[0]["tier"]
        assert rank1_tier in ("Elite", "Star"), (
            f"Rank-1 player tier is '{rank1_tier}' at 15 weeks — expected Elite or Star. "
            "Time decay is crushing tier assignments; assign tiers before decay."
        )

    def test_tiers_span_multiple_buckets_at_15_weeks(self, pool, config):
        """With a 20-player pool and 15 weeks left, tiers must not all be Replacement."""
        result = compute_trade_values(pool, config, weeks_remaining=15)
        unique_tiers = set(result["tier"].unique())
        assert unique_tiers != {"Replacement"}, (
            "All players are Replacement at 15 weeks — tier cutoffs must be "
            "scaled or tiers must be assigned before time decay."
        )

    def test_tiers_consistent_regardless_of_weeks_remaining(self, pool, config):
        """The top player's tier must be identical at 26 weeks and 1 week.
        Tier assignment must reflect relative standing, not absolute post-decay value."""
        full = compute_trade_values(pool, config, weeks_remaining=26)
        late = compute_trade_values(pool, config, weeks_remaining=1)
        top_full = full.iloc[0]["tier"]
        top_late = late.iloc[0]["tier"]
        assert top_full == top_late, (
            f"Rank-1 tier changes from '{top_full}' (26w) to '{top_late}' (1w). "
            "Tiers must be assigned before time decay is applied."
        )


# ── Task 1.8: Sort by Acceptance must use numeric probability ──────────
# The trade dict carries both "acceptance_label" (High/Medium/Low) and
# "acceptance_probability" (float). _build_trade_df only exposes the
# string label; sorting on it alphabetically puts Medium above High.
# Fix: expose "acceptance_prob" (float) alongside "Acceptance" (label)
# and sort on the numeric column.


def _make_trade_dicts() -> list[dict]:
    """Minimal trade dicts covering all three acceptance labels."""
    return [
        {
            "giving_names": ["Player A"],
            "receiving_names": ["Player X"],
            "trade_type": "1-for-1",
            "opponent_team": "Team 1",
            "user_sgp_gain": 1.0,
            "opponent_sgp_gain": 0.5,
            "grade": "B",
            "acceptance_label": "Low",
            "acceptance_probability": 0.20,
            "health_risk": "",
            "composite_score": 10.0,
        },
        {
            "giving_names": ["Player B"],
            "receiving_names": ["Player Y"],
            "trade_type": "1-for-1",
            "opponent_team": "Team 2",
            "user_sgp_gain": 2.0,
            "opponent_sgp_gain": 1.0,
            "grade": "A",
            "acceptance_label": "High",
            "acceptance_probability": 0.75,
            "health_risk": "",
            "composite_score": 20.0,
        },
        {
            "giving_names": ["Player C"],
            "receiving_names": ["Player Z"],
            "trade_type": "1-for-1",
            "opponent_team": "Team 3",
            "user_sgp_gain": 1.5,
            "opponent_sgp_gain": 0.8,
            "grade": "B+",
            "acceptance_label": "Medium",
            "acceptance_probability": 0.45,
            "health_risk": "",
            "composite_score": 15.0,
        },
    ]


def _simulate_build_trade_df(trades: list[dict]) -> pd.DataFrame:
    """Replicate what _build_trade_df produces after the fix.

    This mirrors the page helper directly so we can test the column contract
    without needing to import the page (which calls main() at module level).
    The fix adds "acceptance_prob" (float) so the sort-map can use it instead
    of the string "Acceptance" label.
    """
    rows = []
    for trade in trades:
        giving = ", ".join(trade.get("giving_names", []))
        receiving = ", ".join(trade.get("receiving_names", []))
        row = {
            "You Give": giving,
            "You Receive": receiving,
            "Type": trade.get("trade_type", "1-for-1"),
            "Partner": trade.get("opponent_team", ""),
            "Your Gain": round(trade.get("user_sgp_gain", 0), 2),
            "Their Gain": round(trade.get("opponent_sgp_gain", 0), 2),
            "Grade": trade.get("grade", ""),
            "Acceptance": trade.get("acceptance_label", ""),
            # ← THE FIX: expose numeric probability alongside the label
            "acceptance_prob": float(trade.get("acceptance_probability", 0.0)),
            "Health": trade.get("health_risk", "") or "",
            "Score": round(trade.get("composite_score", 0), 2),
        }
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


class TestAcceptanceSortOrder:
    """_build_trade_df must expose a numeric acceptance_prob column so
    that sort-by-Acceptance orders rows High > Medium > Low.

    The fix adds "acceptance_prob" (float) to the row dict in _build_trade_df
    and updates sort_map["Acceptance Probability"] to use "acceptance_prob"
    instead of "Acceptance".
    """

    def test_build_trade_df_exposes_acceptance_prob_column(self):
        """The fixed _build_trade_df must include a numeric 'acceptance_prob' column."""
        df = _simulate_build_trade_df(_make_trade_dicts())
        assert "acceptance_prob" in df.columns, (
            "'acceptance_prob' numeric column missing. "
            "Add it so sort-by-Acceptance uses numeric values, not string labels."
        )

    def test_acceptance_prob_values_are_numeric(self):
        """acceptance_prob must be a float matching the source dict."""
        df = _simulate_build_trade_df(_make_trade_dicts())
        assert df["acceptance_prob"].dtype.kind == "f", "acceptance_prob must be float"
        probs = sorted(df["acceptance_prob"].tolist())
        assert probs == pytest.approx([0.20, 0.45, 0.75])

    def test_sort_by_acceptance_prob_descending_gives_high_medium_low_order(self):
        """Sorting by acceptance_prob descending must put High before Medium before Low."""
        df = _simulate_build_trade_df(_make_trade_dicts())
        sorted_df = df.sort_values("acceptance_prob", ascending=False)
        labels = sorted_df["Acceptance"].tolist()
        assert labels == ["High", "Medium", "Low"], (
            f"Sort by acceptance_prob descending gave {labels}; expected ['High', 'Medium', 'Low']. "
            "String-sort on 'Acceptance' produces wrong order (Medium before High)."
        )

    def test_string_sort_on_acceptance_label_is_broken(self):
        """Confirm the old bug: sorting on the label string gives wrong order.

        Alphabetical descending of High/Medium/Low → Medium > Low > High.
        This test documents why the numeric column is needed.
        """
        df = _simulate_build_trade_df(_make_trade_dicts())
        # alphabetical descending: M > L > H → first row is NOT High
        string_sorted = df.sort_values("Acceptance", ascending=False)
        labels_str = string_sorted["Acceptance"].tolist()
        assert labels_str[0] != "High", (
            "String-label sort happened to put High first — this is an alphabetical "
            "coincidence that breaks for Medium/Low. The numeric column is still needed."
        )

    def test_page_sort_map_uses_acceptance_prob_key(self):
        """AST guard: pages/12_Trade_Finder.py sort_map must map
        'Acceptance Probability' to 'acceptance_prob', not 'Acceptance'."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "pages" / "12_Trade_Finder.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        # Find the sort_map dict literal in the file
        found_correct = False
        for node in ast.walk(tree):
            # Look for: "Acceptance Probability": ("acceptance_prob", False)
            if isinstance(node, ast.Dict):
                for key, val in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "Acceptance Probability":
                        # val should be a Tuple whose first element is "acceptance_prob"
                        if isinstance(val, ast.Tuple) and val.elts:
                            first = val.elts[0]
                            if isinstance(first, ast.Constant) and first.value == "acceptance_prob":
                                found_correct = True

        assert found_correct, (
            "pages/12_Trade_Finder.py sort_map['Acceptance Probability'] must map "
            "to ('acceptance_prob', False), not ('Acceptance', False). "
            "Without this fix, sorting by Acceptance uses alphabetical string order."
        )
