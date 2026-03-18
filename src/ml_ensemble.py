"""XGBoost ensemble model for draft value correction.

Optional dependency -- graceful fallback to zeros when xgboost unavailable.
Follows the PYMC_AVAILABLE pattern from src/bayesian.py for optional deps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Maximum correction magnitude (in SGP units)
MAX_CORRECTION: float = 2.0

# Default weight when blending ML correction into pick_score
ML_WEIGHT: float = 0.1

# Minimum number of training samples required
MIN_TRAINING_SAMPLES: int = 50

# XGBoost training hyperparameters (conservative to avoid overfitting)
DEFAULT_PARAMS: dict = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Number of boosting rounds for raw Booster API
DEFAULT_NUM_BOOST_ROUNDS: int = 100


class DraftMLEnsemble:
    """XGBoost supplemental signal for draft valuations.

    The model learns what projections miss by training on:
      target = actual_fantasy_value - projected_fantasy_value

    Used as SUPPLEMENTAL signal (weight=0.1), not primary.
    Falls back to zero corrections when model unavailable.
    """

    FEATURE_COLUMNS: list[str] = [
        "age",
        "is_hitter",
        "park_factor",
        "projection_spread",
        "health_score",
        "position_scarcity",
    ]

    def __init__(self, model_path: str | None = None):
        """Load pre-trained model or use fallback.

        Args:
            model_path: Path to serialized model file. If None or file
                doesn't exist, falls back to zero predictions.
        """
        self._model = None
        if model_path and XGBOOST_AVAILABLE:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    self._model = xgb.Booster()
                    self._model.load_model(str(model_file))
                    logger.info("Loaded ML model from %s", model_path)
                except Exception as exc:
                    logger.warning("Could not load ML model: %s", exc)
                    self._model = None
            else:
                logger.debug("Model file not found: %s", model_path)

    def predict_value(self, features: pd.DataFrame) -> np.ndarray:
        """Predict draft value corrections.

        Returns array of corrections (positive = projections undervalue,
        negative = projections overvalue). Falls back to zeros.

        Args:
            features: DataFrame with feature columns (age, is_hitter, etc.).

        Returns:
            Array of correction values clamped to [-MAX_CORRECTION, MAX_CORRECTION].
        """
        if self._model is None or not XGBOOST_AVAILABLE:
            return np.zeros(len(features))

        try:
            feature_matrix = self._prepare_features(features)
            dmat = xgb.DMatrix(feature_matrix)
            predictions = self._model.predict(dmat)
            return np.clip(predictions, -MAX_CORRECTION, MAX_CORRECTION)
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            return np.zeros(len(features))

    def predict_batch(self, player_pool: pd.DataFrame) -> pd.Series:
        """Convenience wrapper returning a named Series aligned to the pool index.

        Args:
            player_pool: Full player pool DataFrame.

        Returns:
            Series named 'ml_correction' with same index as input.
        """
        corrections = self.predict_value(player_pool)
        return pd.Series(corrections, index=player_pool.index, name="ml_correction")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate feature columns.

        Missing columns are filled with 0. Boolean is_hitter is cast to int.
        All values coerced to numeric with NaN replaced by 0.
        """
        features = pd.DataFrame(index=df.index)
        for col in self.FEATURE_COLUMNS:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                features[col] = 0
        # Convert boolean to int for XGBoost compatibility
        if "is_hitter" in features.columns:
            features["is_hitter"] = features["is_hitter"].astype(int)
        return features

    def train(
        self,
        historical_data: pd.DataFrame,
        target_col: str = "residual",
        params: dict | None = None,
    ) -> dict:
        """Train model on historical player seasons.

        Features: age, is_hitter, park_factor, projection_spread,
                  health_score, position_scarcity
        Target: actual - projected (the residual)

        Args:
            historical_data: DataFrame with feature columns + target column.
            target_col: Name of the target column (actual - projected).
            params: Optional XGBoost parameters override.

        Returns:
            Dict with training metrics or skip status.
        """
        if not XGBOOST_AVAILABLE:
            return {"status": "skipped", "reason": "xgboost not installed"}

        if len(historical_data) < MIN_TRAINING_SAMPLES:
            return {
                "status": "skipped",
                "reason": f"insufficient data ({len(historical_data)} < {MIN_TRAINING_SAMPLES})",
            }

        if target_col not in historical_data.columns:
            return {"status": "skipped", "reason": f"target column '{target_col}' not found"}

        try:
            features = self._prepare_features(historical_data)
            target = pd.to_numeric(historical_data[target_col], errors="coerce")

            # Remove rows where target is NaN (truly missing residuals)
            valid_mask = target.notna()
            features = features[valid_mask]
            target = target[valid_mask]

            if len(features) < MIN_TRAINING_SAMPLES:
                return {
                    "status": "skipped",
                    "reason": f"insufficient valid samples ({len(features)} < {MIN_TRAINING_SAMPLES})",
                }

            # Build DMatrix and train
            dtrain = xgb.DMatrix(features, label=target)
            train_params = dict(DEFAULT_PARAMS)
            if params:
                train_params.update(params)

            # Extract n_estimators for num_boost_round (Booster API uses this)
            num_rounds = train_params.pop("n_estimators", DEFAULT_NUM_BOOST_ROUNDS)

            self._model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_rounds,
            )
            # Compute training metrics
            preds = self._model.predict(dtrain)
            residuals = target.values - preds
            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae = float(np.mean(np.abs(residuals)))

            return {
                "status": "trained",
                "n_samples": len(features),
                "rmse": rmse,
                "mae": mae,
                "n_features": len(self.FEATURE_COLUMNS),
            }

        except Exception as exc:
            logger.warning("ML training failed: %s", exc)
            self._model = None
            return {"status": "failed", "reason": str(exc)}

    def save_model(self, path: str) -> bool:
        """Serialize model to disk.

        Args:
            path: File path to save the model.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._model is None:
            return False
        try:
            self._model.save_model(path)
            logger.info("Saved ML model to %s", path)
            return True
        except Exception as exc:
            logger.warning("Could not save ML model: %s", exc)
            return False

    @staticmethod
    def compute_feature_importance(model) -> dict[str, float]:
        """Extract feature importance from a trained model.

        Args:
            model: Trained XGBoost Booster.

        Returns:
            Dict mapping feature name to importance score.
        """
        if model is None or not XGBOOST_AVAILABLE:
            return {}
        try:
            return dict(model.get_score(importance_type="gain"))
        except Exception:
            return {}

    @property
    def MODEL_AVAILABLE(self) -> bool:
        """Backward-compat property. Use is_ready instead."""
        return self.is_ready

    @property
    def is_ready(self) -> bool:
        """Whether the model is loaded and ready for predictions."""
        return self._model is not None and XGBOOST_AVAILABLE


# ── Module-level convenience ─────────────────────────────────────────


def get_ml_ensemble(model_path: str | None = None) -> DraftMLEnsemble:
    """Factory function for creating a DraftMLEnsemble instance.

    Args:
        model_path: Optional path to pre-trained model.

    Returns:
        DraftMLEnsemble instance (may be fallback-only).
    """
    return DraftMLEnsemble(model_path=model_path)
