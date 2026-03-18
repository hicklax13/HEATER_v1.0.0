"""Tests for ML ensemble model and news sentiment scoring."""

import numpy as np
import pandas as pd
import pytest

from src.ml_ensemble import (
    MAX_CORRECTION,
    MIN_TRAINING_SAMPLES,
    XGBOOST_AVAILABLE,
    DraftMLEnsemble,
    get_ml_ensemble,
)
from src.news_sentiment import (
    HIGH_IMPACT_NEGATIVE,
    HIGH_IMPACT_POSITIVE,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    SentimentResult,
    analyze_news_sentiment,
    batch_sentiment,
    compute_news_sentiment,
    sentiment_adjustment,
)

# ── DraftMLEnsemble Tests ────────────────────────────────────────────


class TestDraftMLEnsembleNoModel:
    """Test DraftMLEnsemble behavior with no trained model."""

    def test_no_model_predict_returns_zeros(self):
        """DraftMLEnsemble() with no model returns zero predictions."""
        ensemble = DraftMLEnsemble()
        features = pd.DataFrame({"age": [25, 30], "is_hitter": [True, False]})
        result = ensemble.predict_value(features)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_model_available_false_when_no_model(self):
        """MODEL_AVAILABLE is False when no model loaded."""
        # Reset class-level state
        DraftMLEnsemble.MODEL_AVAILABLE = False
        ensemble = DraftMLEnsemble()
        assert not ensemble.is_ready
        assert not DraftMLEnsemble.MODEL_AVAILABLE

    def test_no_model_path_fallback(self):
        """None model_path creates a fallback-only instance."""
        ensemble = DraftMLEnsemble(model_path=None)
        assert ensemble._model is None
        assert not ensemble.is_ready

    def test_nonexistent_model_path_fallback(self):
        """Non-existent model file creates a fallback-only instance."""
        ensemble = DraftMLEnsemble(model_path="/nonexistent/model.json")
        assert ensemble._model is None
        assert not ensemble.is_ready

    def test_save_model_returns_false_when_no_model(self):
        """save_model() returns False when no model exists."""
        ensemble = DraftMLEnsemble()
        assert ensemble.save_model("/tmp/test_model.json") is False

    def test_predict_batch_returns_series(self):
        """predict_batch returns a Series aligned to input index."""
        ensemble = DraftMLEnsemble()
        pool = pd.DataFrame(
            {"age": [25, 28, 32], "is_hitter": [True, True, False]},
            index=[10, 20, 30],
        )
        result = ensemble.predict_batch(pool)
        assert isinstance(result, pd.Series)
        assert result.name == "ml_correction"
        assert list(result.index) == [10, 20, 30]
        assert all(result == 0.0)


class TestFeaturePreparation:
    """Test feature extraction and validation."""

    def test_missing_columns_filled_with_zeros(self):
        """Missing feature columns are filled with 0."""
        ensemble = DraftMLEnsemble()
        df = pd.DataFrame({"age": [25, 30]})  # Missing most columns
        features = ensemble._prepare_features(df)
        assert "park_factor" in features.columns
        assert features["park_factor"].tolist() == [0, 0]
        assert "health_score" in features.columns
        assert features["health_score"].tolist() == [0, 0]

    def test_is_hitter_cast_to_int(self):
        """Boolean is_hitter is cast to integer."""
        ensemble = DraftMLEnsemble()
        df = pd.DataFrame({"is_hitter": [True, False, True]})
        features = ensemble._prepare_features(df)
        assert features["is_hitter"].dtype in (np.int64, np.int32, int)
        assert features["is_hitter"].tolist() == [1, 0, 1]

    def test_non_numeric_coerced(self):
        """Non-numeric values are coerced to 0."""
        ensemble = DraftMLEnsemble()
        df = pd.DataFrame({"age": ["twenty", 30, None], "is_hitter": [1, 0, 1]})
        features = ensemble._prepare_features(df)
        assert features["age"].tolist() == [0.0, 30.0, 0.0]

    def test_all_feature_columns_present(self):
        """All FEATURE_COLUMNS appear in output."""
        ensemble = DraftMLEnsemble()
        df = pd.DataFrame({"age": [25]})
        features = ensemble._prepare_features(df)
        for col in DraftMLEnsemble.FEATURE_COLUMNS:
            assert col in features.columns


class TestTrainWithoutXGBoost:
    """Test training behavior when xgboost is not installed."""

    def test_train_returns_skip_without_xgboost(self):
        """train() returns skip status when xgboost unavailable."""
        if XGBOOST_AVAILABLE:
            pytest.skip("xgboost is installed — testing fallback not possible")
        ensemble = DraftMLEnsemble()
        result = ensemble.train(pd.DataFrame())
        assert result["status"] == "skipped"
        assert "xgboost" in result["reason"]

    def test_train_insufficient_data(self):
        """train() returns skip when data is below MIN_TRAINING_SAMPLES."""
        if not XGBOOST_AVAILABLE:
            pytest.skip("xgboost not installed")
        ensemble = DraftMLEnsemble()
        small_data = pd.DataFrame(
            {
                "age": range(10),
                "is_hitter": [True] * 10,
                "park_factor": [1.0] * 10,
                "projection_spread": [0.5] * 10,
                "health_score": [0.9] * 10,
                "position_scarcity": [0.3] * 10,
                "residual": np.random.randn(10),
            }
        )
        result = ensemble.train(small_data)
        assert result["status"] == "skipped"
        assert "insufficient" in result["reason"]

    def test_train_missing_target_column(self):
        """train() returns skip when target column is missing."""
        if not XGBOOST_AVAILABLE:
            pytest.skip("xgboost not installed")
        ensemble = DraftMLEnsemble()
        data = pd.DataFrame({"age": range(100), "is_hitter": [True] * 100})
        result = ensemble.train(data, target_col="nonexistent")
        assert result["status"] == "skipped"
        assert "nonexistent" in result["reason"]


class TestPredictionBounds:
    """Test that predictions are bounded correctly."""

    def test_predictions_bounded_to_max_correction(self):
        """Predictions are clipped to [-MAX_CORRECTION, MAX_CORRECTION]."""
        ensemble = DraftMLEnsemble()
        # Without a model, predictions are zeros (within bounds)
        features = pd.DataFrame({"age": [25]})
        result = ensemble.predict_value(features)
        assert all(result >= -MAX_CORRECTION)
        assert all(result <= MAX_CORRECTION)

    def test_empty_features_returns_empty_array(self):
        """Empty DataFrame returns empty array."""
        ensemble = DraftMLEnsemble()
        result = ensemble.predict_value(pd.DataFrame())
        assert len(result) == 0


class TestGetMLEnsemble:
    """Test the factory function."""

    def test_factory_returns_instance(self):
        """get_ml_ensemble returns DraftMLEnsemble instance."""
        ensemble = get_ml_ensemble()
        assert isinstance(ensemble, DraftMLEnsemble)

    def test_factory_with_path(self):
        """get_ml_ensemble with nonexistent path returns fallback instance."""
        ensemble = get_ml_ensemble("/nonexistent/model.json")
        assert isinstance(ensemble, DraftMLEnsemble)
        assert not ensemble.is_ready


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_no_model_returns_empty_dict(self):
        """compute_feature_importance with None model returns empty dict."""
        result = DraftMLEnsemble.compute_feature_importance(None)
        assert result == {}


# ── News Sentiment Tests ─────────────────────────────────────────────


class TestComputeNewsSentiment:
    """Test the basic sentiment scoring function."""

    def test_positive_keywords_yield_positive_score(self):
        """Positive keywords produce a positive sentiment score."""
        news = ["Player showing impressive power in spring training"]
        score = compute_news_sentiment(news)
        assert score > 0.0

    def test_negative_keywords_yield_negative_score(self):
        """Negative keywords produce a negative sentiment score."""
        news = ["Player suffered a strain during practice"]
        score = compute_news_sentiment(news)
        assert score < 0.0

    def test_empty_list_returns_zero(self):
        """Empty news list returns 0.0."""
        assert compute_news_sentiment([]) == 0.0

    def test_no_keywords_returns_zero(self):
        """News without any matching keywords returns 0.0."""
        news = ["Player attended team meeting today"]
        assert compute_news_sentiment(news) == 0.0

    def test_mixed_keywords_balanced(self):
        """Equal positive and negative keywords produce ~0.0 score."""
        news = ["Player is healthy but has tightness in shoulder"]
        score = compute_news_sentiment(news)
        # "healthy" = +1, "tightness" = -1 → 0.0
        assert score == pytest.approx(0.0)

    def test_score_clamped_to_range(self):
        """Score is always in [-1.0, +1.0]."""
        # All positive
        news = ["breakout impressive dominant"]
        score = compute_news_sentiment(news)
        assert -1.0 <= score <= 1.0

        # All negative
        news = ["surgery setback fracture torn"]
        score = compute_news_sentiment(news)
        assert -1.0 <= score <= 1.0

    def test_multiple_items_aggregated(self):
        """Multiple news items are aggregated together."""
        news = [
            "Player looking dominant in spring",
            "Named everyday starter",
            "Impressive batting practice numbers",
        ]
        score = compute_news_sentiment(news)
        assert score > 0.5  # Strong positive signal

    def test_case_insensitive(self):
        """Keyword matching is case-insensitive."""
        news_upper = ["Player is HEALTHY and IMPRESSIVE"]
        news_lower = ["player is healthy and impressive"]
        assert compute_news_sentiment(news_upper) == compute_news_sentiment(news_lower)


class TestAnalyzeNewsSentiment:
    """Test the detailed sentiment analysis function."""

    def test_empty_returns_zero_result(self):
        """Empty list returns SentimentResult with all zeros."""
        result = analyze_news_sentiment([])
        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.positive_count == 0
        assert result.negative_count == 0
        assert result.confidence == 0.0

    def test_high_impact_positive_double_counted(self):
        """High-impact positive keywords count double."""
        news = ["Player having a breakout season"]
        result = analyze_news_sentiment(news)
        assert result.positive_count == 2  # "breakout" counts double
        assert any("+breakout" in f for f in result.high_impact_flags)

    def test_high_impact_negative_double_counted(self):
        """High-impact negative keywords count double."""
        news = ["Player needs surgery on elbow"]
        result = analyze_news_sentiment(news)
        assert result.negative_count == 2  # "surgery" counts double
        assert any("-surgery" in f for f in result.high_impact_flags)

    def test_confidence_scales_with_signals(self):
        """Confidence increases with more keyword matches."""
        few = analyze_news_sentiment(["Player is healthy"])
        many = analyze_news_sentiment(
            [
                "Player is healthy and impressive, named everyday starter",
                "Looking dominant with breakout power in spring",
            ]
        )
        assert many.confidence > few.confidence


class TestSentimentAdjustment:
    """Test the sentiment-to-multiplier conversion."""

    def test_zero_sentiment_returns_one(self):
        """Zero sentiment produces a neutral 1.0 multiplier."""
        assert sentiment_adjustment(0.0) == 1.0

    def test_positive_sentiment_above_one(self):
        """Positive sentiment produces multiplier > 1.0."""
        assert sentiment_adjustment(1.0) > 1.0

    def test_negative_sentiment_below_one(self):
        """Negative sentiment produces multiplier < 1.0."""
        assert sentiment_adjustment(-1.0) < 1.0

    def test_custom_weight(self):
        """Custom weight scales the adjustment magnitude."""
        assert sentiment_adjustment(1.0, weight=0.10) == pytest.approx(1.10)
        assert sentiment_adjustment(-1.0, weight=0.10) == pytest.approx(0.90)


class TestBatchSentiment:
    """Test batch sentiment processing."""

    def test_batch_returns_dict(self):
        """batch_sentiment returns a dict mapping player_id to score."""
        player_news = {
            1: ["Player is healthy and impressive"],
            2: ["Player suffered a strain"],
            3: [],
        }
        results = batch_sentiment(player_news)
        assert isinstance(results, dict)
        assert 1 in results
        assert 2 in results
        assert 3 in results
        assert results[1] > 0  # positive
        assert results[2] < 0  # negative
        assert results[3] == 0.0  # no news

    def test_empty_dict_returns_empty(self):
        """Empty input dict returns empty output dict."""
        assert batch_sentiment({}) == {}


class TestKeywordLists:
    """Validate keyword list integrity."""

    def test_no_keyword_overlap(self):
        """Positive and negative keyword lists should not overlap."""
        overlap = set(POSITIVE_KEYWORDS) & set(NEGATIVE_KEYWORDS)
        assert not overlap, f"Overlapping keywords: {overlap}"

    def test_high_impact_subsets(self):
        """High-impact keywords are subsets of their parent lists."""
        for kw in HIGH_IMPACT_POSITIVE:
            assert kw in POSITIVE_KEYWORDS, f"{kw} not in POSITIVE_KEYWORDS"
        for kw in HIGH_IMPACT_NEGATIVE:
            assert kw in NEGATIVE_KEYWORDS, f"{kw} not in NEGATIVE_KEYWORDS"
