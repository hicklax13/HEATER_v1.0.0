"""XGBoost ensemble model for draft value correction.

Optional dependency -- graceful fallback to zeros when xgboost unavailable.
Follows the PYMC_AVAILABLE pattern from src/bayesian.py for optional deps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


# ── Statcast XGBoost Regression ─────────────────────────────────────

# Feature columns for Statcast-based regression model
STATCAST_FEATURES: list[str] = [
    "ev_mean",
    "barrel_pct",
    "hard_hit_pct",
    "xwoba",
    "xba",
    "sprint_speed",
    "age",
    "health_score",
]


def train_ensemble(
    historical_stats: pd.DataFrame,
    historical_projections: pd.DataFrame,
    config: Any = None,
) -> Any:
    """Train XGBoost model on historical data to learn projection residuals.

    The model learns what projections miss by training on Statcast features
    to predict the gap between actual and projected SGP values.

    Features: exit_velocity (ev_mean), barrel_rate (barrel_pct),
              hard_hit_pct, xwoba, xba, sprint_speed, age, health_score
    Target: actual_sgp - projected_sgp (what projections missed)

    Args:
        historical_stats: DataFrame with actual stats + Statcast columns,
            must have ``player_id`` column.
        historical_projections: DataFrame with projected stats,
            must have ``player_id`` column.
        config: Optional LeagueConfig for SGP computation. Uses default if None.

    Returns:
        Trained XGBRegressor model object, or None if xgboost is unavailable
        or training data is insufficient (<50 samples).
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not installed -- skipping ensemble training")
        return None

    # Lazy import to avoid circular dependency
    from src.valuation import LeagueConfig, SGPCalculator

    if config is None:
        config = LeagueConfig()

    # Validate inputs
    if historical_stats is None or historical_projections is None:
        logger.warning("Missing historical data -- skipping ensemble training")
        return None

    if "player_id" not in historical_stats.columns or "player_id" not in historical_projections.columns:
        logger.warning("Missing player_id column -- skipping ensemble training")
        return None

    # Merge on player_id
    merged = historical_stats.merge(
        historical_projections,
        on="player_id",
        suffixes=("_actual", "_proj"),
        how="inner",
    )

    if len(merged) < MIN_TRAINING_SAMPLES:
        logger.warning(
            "Insufficient training samples (%d < %d) -- skipping",
            len(merged),
            MIN_TRAINING_SAMPLES,
        )
        return None

    # Compute SGP for actual and projected stats
    sgp_calc = SGPCalculator(config)

    actual_sgp_vals = []
    proj_sgp_vals = []

    for _, row in merged.iterrows():
        # Build Series with actual stat columns (strip _actual suffix)
        actual_cols = {c.replace("_actual", ""): row[c] for c in merged.columns if c.endswith("_actual")}
        actual_series = pd.Series(actual_cols)
        # Ensure is_hitter is present
        if "is_hitter" not in actual_series.index:
            actual_series["is_hitter"] = row.get("is_hitter", row.get("is_hitter_actual", 1))

        proj_cols = {c.replace("_proj", ""): row[c] for c in merged.columns if c.endswith("_proj")}
        proj_series = pd.Series(proj_cols)
        if "is_hitter" not in proj_series.index:
            proj_series["is_hitter"] = row.get("is_hitter", row.get("is_hitter_proj", 1))

        try:
            actual_sgp_vals.append(sgp_calc.total_sgp(actual_series))
        except Exception:
            actual_sgp_vals.append(0.0)

        try:
            proj_sgp_vals.append(sgp_calc.total_sgp(proj_series))
        except Exception:
            proj_sgp_vals.append(0.0)

    merged["actual_sgp"] = actual_sgp_vals
    merged["proj_sgp"] = proj_sgp_vals
    merged["target"] = merged["actual_sgp"] - merged["proj_sgp"]

    # Build feature matrix from Statcast columns
    features = pd.DataFrame(index=merged.index)
    for col in STATCAST_FEATURES:
        # Try multiple suffixed versions then raw column name
        if f"{col}_actual" in merged.columns:
            features[col] = pd.to_numeric(merged[f"{col}_actual"], errors="coerce")
        elif col in merged.columns:
            features[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            features[col] = np.nan

    # Fill missing values with column medians
    for col in features.columns:
        median_val = features[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        features[col] = features[col].fillna(median_val)

    target = merged["target"].values

    # Drop rows where target is NaN
    valid_mask = ~np.isnan(target)
    features = features[valid_mask]
    target = target[valid_mask]

    if len(features) < MIN_TRAINING_SAMPLES:
        logger.warning(
            "Insufficient valid samples after cleanup (%d < %d)",
            len(features),
            MIN_TRAINING_SAMPLES,
        )
        return None

    # Train XGBRegressor
    model = xgb.XGBRegressor(
        objective=DEFAULT_PARAMS["objective"],
        max_depth=DEFAULT_PARAMS["max_depth"],
        learning_rate=DEFAULT_PARAMS["eta"],
        n_estimators=DEFAULT_PARAMS["n_estimators"],
        subsample=DEFAULT_PARAMS["subsample"],
        colsample_bytree=DEFAULT_PARAMS["colsample_bytree"],
        min_child_weight=DEFAULT_PARAMS["min_child_weight"],
        reg_alpha=DEFAULT_PARAMS["reg_alpha"],
        reg_lambda=DEFAULT_PARAMS["reg_lambda"],
        random_state=42,
    )
    model.fit(features, target)

    logger.info(
        "Trained Statcast ensemble on %d samples with %d features",
        len(features),
        len(STATCAST_FEATURES),
    )
    return model


def predict_corrections(
    model: Any,
    current_pool: pd.DataFrame,
    config: Any = None,
) -> pd.Series:
    """Predict SGP corrections for current players using a trained model.

    Uses Statcast features from the current player pool to predict how much
    projections over- or under-value each player.

    Args:
        model: Trained XGBRegressor from ``train_ensemble()``, or None.
        current_pool: DataFrame with player data including Statcast columns.
            Must have ``player_id`` column.
        config: Optional LeagueConfig (unused, reserved for future).

    Returns:
        pd.Series indexed by player_id with correction values clamped to
        [-MAX_CORRECTION, +MAX_CORRECTION] (plus/minus 2.0 SGP).
        Returns all-zeros Series if model is None or xgboost unavailable.
    """
    # Determine index
    if current_pool is not None and "player_id" in current_pool.columns:
        index = current_pool["player_id"]
    elif current_pool is not None:
        index = current_pool.index
    else:
        return pd.Series(dtype=float, name="ml_correction")

    # Fallback: zeros
    if model is None or not XGBOOST_AVAILABLE:
        return pd.Series(
            np.zeros(len(current_pool)),
            index=index,
            name="ml_correction",
        )

    # Build feature matrix
    features = pd.DataFrame(index=current_pool.index)
    for col in STATCAST_FEATURES:
        if col in current_pool.columns:
            features[col] = pd.to_numeric(current_pool[col], errors="coerce")
        else:
            features[col] = np.nan

    # Fill missing with column medians
    for col in features.columns:
        median_val = features[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        features[col] = features[col].fillna(median_val)

    try:
        predictions = model.predict(features)
        clamped = np.clip(predictions, -MAX_CORRECTION, MAX_CORRECTION)
        return pd.Series(clamped, index=index, name="ml_correction")
    except Exception as exc:
        logger.warning("Statcast prediction failed: %s", exc)
        return pd.Series(
            np.zeros(len(current_pool)),
            index=index,
            name="ml_correction",
        )
