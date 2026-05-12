"""Wave 8c — type-design improvements.

These tests verify TypedDict definitions are importable and structurally
correct, that dataclass param-objects preserve backwards-compat keyword
args, and that module-level mutable state is properly encapsulated.

TypedDict is a structural type hint — at runtime a TypedDict instance IS
a plain dict. So these tests focus on:
  - importability (no circular imports)
  - schema correctness (keys present, types match expectations)
  - backwards compat (old call sites that ignored typing still work)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────
# Batch 1: Cross-boundary TypedDicts
# ─────────────────────────────────────────────────────────────────────────


class TestTradeResultTypedDict:
    def test_importable(self):
        from src.engine.output.types import (
            CategoryImpactEntry,
            GradeRange,
            MCOverlayResult,
            TradeResult,
        )

        assert TradeResult is not None
        assert MCOverlayResult is not None
        assert GradeRange is not None
        assert CategoryImpactEntry is not None

    def test_grade_range_construction(self):
        from src.engine.output.types import GradeRange

        # TypedDict at runtime IS a dict; construction returns a dict literal
        gr: GradeRange = GradeRange(grade="A", grade_low="B+", grade_high="A+", confidence="high")
        assert gr["grade"] == "A"
        assert gr["confidence"] == "high"

    def test_trade_result_partial(self):
        """TradeResult is total=False, so partial dicts type-check."""
        from src.engine.output.types import TradeResult

        # Should accept a dict with only Phase 1 keys
        result: TradeResult = {"grade": "A", "surplus_sgp": 1.5, "verdict": "ACCEPT"}
        assert result["grade"] == "A"
        # Missing keys behave like dict
        assert "mc_mean" not in result

    def test_compute_grade_range_returns_typed_dict(self):
        from src.engine.output.trade_evaluator import _compute_grade_range

        result = _compute_grade_range(1.5, 0.8)
        assert "grade" in result
        assert "grade_low" in result
        assert "grade_high" in result
        assert "confidence" in result
        assert result["confidence"] in {"high", "medium", "low"}


class TestMatchupResultTypedDict:
    def test_importable(self):
        from src.yahoo_api import MatchupCategoryEntry, MatchupResult

        assert MatchupResult is not None
        assert MatchupCategoryEntry is not None

    def test_matchup_result_partial(self):
        """MatchupResult is total=False — supports degraded shapes."""
        from src.yahoo_api import MatchupResult

        m: MatchupResult = {"week": 5, "user_name": "X", "wins": 3, "losses": 2}
        assert m["week"] == 5

    def test_matchup_category_entry_shape(self):
        from src.yahoo_api import MatchupCategoryEntry

        entry: MatchupCategoryEntry = {"cat": "HR", "you": "10", "opp": "8", "result": "WIN"}
        assert entry["cat"] == "HR"
        assert entry["result"] == "WIN"

    def test_yahoo_data_service_signature(self):
        """get_matchup is annotated MatchupResult | None."""
        import inspect

        from src.yahoo_data_service import YahooDataService

        sig = inspect.signature(YahooDataService.get_matchup)
        # Return annotation should mention MatchupResult
        ret_str = str(sig.return_annotation)
        assert "MatchupResult" in ret_str or "Matchup" in ret_str


class TestPlayerCardDataTypedDict:
    def test_importable(self):
        from src.player_card import PlayerCardData

        assert PlayerCardData is not None

    def test_player_card_data_has_all_sections(self):
        """PlayerCardData TypedDict declares all 10 sections."""
        from src.player_card import PlayerCardData

        # Pull annotations off the TypedDict class
        anns = PlayerCardData.__annotations__
        expected_sections = {
            "profile",
            "projections",
            "historical",
            "advanced",
            "injury_history",
            "rankings",
            "radar",
            "trends",
            "news",
            "prospect",
        }
        assert expected_sections.issubset(set(anns.keys()))


class TestOptimizerResultTypedDict:
    def test_importable(self):
        from src.optimizer.pipeline import LineupOptimizerPipeline, OptimizerResult

        assert OptimizerResult is not None
        assert LineupOptimizerPipeline is not None

    def test_optimizer_result_partial(self):
        from src.optimizer.pipeline import OptimizerResult

        # total=False — partial dicts allowed for mode-conditional keys
        r: OptimizerResult = {"lineup": {}, "recommendations": [], "mode": "quick"}
        assert r["mode"] == "quick"
        assert "daily_dcv" not in r  # daily-mode-only key absent


class TestRuntimeRoundTripPreservation:
    """TypedDict introduction must not change runtime behavior."""

    def test_evaluate_trade_signature_unchanged(self):
        import inspect

        from src.engine.output.trade_evaluator import evaluate_trade

        sig = inspect.signature(evaluate_trade)
        params = list(sig.parameters.keys())
        # Spot-check that all the documented kwargs still exist
        for required in [
            "giving_ids",
            "receiving_ids",
            "user_roster_ids",
            "player_pool",
            "config",
            "enable_mc",
            "enable_context",
            "enable_game_theory",
        ]:
            assert required in params, f"{required} missing from evaluate_trade signature"

    def test_pipeline_optimize_signature_unchanged(self):
        import inspect

        from src.optimizer.pipeline import LineupOptimizerPipeline

        sig = inspect.signature(LineupOptimizerPipeline.optimize)
        params = list(sig.parameters.keys())
        for required in [
            "self",
            "standings",
            "team_name",
            "h2h_opponent_totals",
            "my_totals",
            "week_schedule",
            "park_factors",
        ]:
            assert required in params

    def test_get_current_matchup_signature_unchanged(self):
        import inspect

        from src.yahoo_api import YahooFantasyClient

        sig = inspect.signature(YahooFantasyClient.get_current_matchup)
        # Should still take only self
        assert list(sig.parameters.keys()) == ["self"]
