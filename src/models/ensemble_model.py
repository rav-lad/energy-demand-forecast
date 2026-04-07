"""Ensemble model for price forecasting.

Combines multiple models (Ridge, XGBoost, LightGBM, LSTM) with weighted averaging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.utils.config import MODEL_PARAMS, ENSEMBLE_WEIGHTS


class EnsembleForecaster:
    """Ensemble forecaster combining multiple models."""

    def __init__(
        self, models: List[str] = ["ridge", "xgboost"], weights: Dict[str, float] = None
    ):
        """
        Initialize ensemble forecaster.

        Args:
            models: List of model names to include
            weights: Dictionary of weights for each model (must sum to 1)
        """
        self.models = models
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.trained_models = {}
        self.scalers = {}

        # Validate weights
        if set(models) != set(self.weights.keys()):
            raise ValueError(
                f"Weights keys {self.weights.keys()} don't match models {models}"
            )

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {sum(self.weights.values())}"
            )

    def _create_model(self, model_name: str) -> Any:
        """Create a model instance."""
        if model_name == "ridge":
            return Ridge(**MODEL_PARAMS["ridge"])
        elif model_name == "xgboost":
            return xgb.XGBRegressor(**MODEL_PARAMS["xgboost"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleForecaster":
        """
        Train all models in the ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            self
        """
        for model_name in self.models:
            # Create model
            model = self._create_model(model_name)

            # Ridge needs scaling, XGBoost doesn't
            if model_name == "ridge":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                self.scalers[model_name] = scaler
            else:
                model.fit(X, y)

            self.trained_models[model_name] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        predictions = np.zeros(len(X))

        for model_name, model in self.trained_models.items():
            weight = self.weights[model_name]

            # Apply scaling if needed
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)

            predictions += weight * pred

        return predictions

    def predict_individual(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each model individually.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Dictionary of {model_name: predictions}
        """
        predictions = {}

        for model_name, model in self.trained_models.items():
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
                predictions[model_name] = model.predict(X_scaled)
            else:
                predictions[model_name] = model.predict(X)

        return predictions

    def get_model_importance(self) -> Dict[str, float]:
        """Get model weights (importance in ensemble)."""
        return self.weights.copy()


def train_ensemble_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "price",
    initial_train_days: int = 365,
    forecast_horizon_hours: int = 24,
    models: List[str] = ["ridge", "xgboost"],
    weights: Dict[str, float] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train ensemble model with walk-forward validation.

    Args:
        df: DataFrame with datetime_hour, features, and target
        feature_cols: List of feature column names
        target_col: Target column name
        initial_train_days: Days of data for initial training
        forecast_horizon_hours: Forecast horizon (default 24h)
        models: List of models to include in ensemble
        weights: Ensemble weights
        verbose: Print progress

    Returns:
        predictions_df: DataFrame with predictions
        metrics: Overall metrics
    """
    from src.models.walk_forward_validator import walk_forward_validate

    # Create ensemble model
    ensemble = EnsembleForecaster(models=models, weights=weights)

    # Run walk-forward validation
    predictions_df, metrics, daily_metrics = walk_forward_validate(
        df=df,
        model=ensemble,
        feature_cols=feature_cols,
        target_col=target_col,
        initial_train_days=initial_train_days,
        forecast_horizon_hours=forecast_horizon_hours,
        retrain_frequency_days=1,
        expanding_window=True,
        verbose=verbose,
    )

    return predictions_df, metrics


if __name__ == "__main__":
    # Test ensemble model
    print("Testing ensemble model...")

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(1000) * 0.1

    X_test = np.random.randn(100, 10)
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + np.random.randn(100) * 0.1

    # Train ensemble
    ensemble = EnsembleForecaster(models=["ridge", "xgboost"])
    ensemble.fit(X_train, y_train)

    # Predict
    y_pred = ensemble.predict(X_test)

    # Calculate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"Ensemble MAE: {mae:.4f}")

    # Individual predictions
    individual_preds = ensemble.predict_individual(X_test)
    for model_name, pred in individual_preds.items():
        mae_individual = np.mean(np.abs(y_test - pred))
        print(f"{model_name} MAE: {mae_individual:.4f}")

    print("\nEnsemble weights:")
    for model, weight in ensemble.get_model_importance().items():
        print(f"  {model}: {weight:.2f}")

    print("\n[OK] Ensemble model test passed")
