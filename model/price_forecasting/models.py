"""Price forecasting models using advanced machine learning techniques.

This module implements multiple approaches for electricity price forecasting:
- LightGBM Quantile Regression (probabilistic forecasts)
- Ensemble methods combining multiple models
- Base forecaster interface for consistency
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class PriceForecaster(ABC):
    """Abstract base class for price forecasting models.

    All price forecasting models should inherit from this class and implement
    the fit and predict methods.
    """

    def __init__(self):
        """Initialize base forecaster."""
        self.model = None
        self.scaler = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the forecasting model.

        Args:
            X: Feature DataFrame.
            y: Target series (prices).
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate price forecasts.

        Args:
            X: Feature DataFrame.

        Returns:
            np.ndarray: Predicted prices.
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Feature DataFrame.
            y: True prices.

        Returns:
            dict: Performance metrics.
        """
        y_pred = self.predict(X)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "mape": np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100,
        }

        # Directional accuracy
        if len(y) > 1:
            y_diff = np.diff(y)
            pred_diff = np.diff(y_pred)
            metrics["directional_accuracy"] = (
                np.mean(np.sign(y_diff) == np.sign(pred_diff)) * 100
            )

        return metrics


class LightGBMQuantileForecaster(PriceForecaster):
    """LightGBM-based quantile regression forecaster.

    This model provides probabilistic forecasts by predicting multiple quantiles,
    enabling uncertainty quantification and risk-aware trading decisions.

    Attributes:
        quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9]).
        models: Dictionary mapping quantiles to trained models.
        params: LightGBM training parameters.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
    ):
        """Initialize LightGBM Quantile Forecaster.

        Args:
            quantiles: List of quantiles to predict.
            n_estimators: Number of boosting iterations.
            learning_rate: Boosting learning rate.
            max_depth: Maximum tree depth.
            num_leaves: Maximum number of leaves in one tree.
            min_data_in_leaf: Minimum samples per leaf.
            feature_fraction: Fraction of features to use per tree.
            bagging_fraction: Fraction of data to use per iteration.
            bagging_freq: Frequency of bagging.
            random_state: Random seed.
        """
        super().__init__()

        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.models = {}

        self.params = {
            "objective": "quantile",
            "metric": "quantile",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "random_state": random_state,
            "verbose": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train quantile models for each specified quantile.

        Args:
            X: Feature DataFrame.
            y: Target series (prices).
        """
        # Fit a model for each quantile
        for q in self.quantiles:
            params_q = self.params.copy()
            params_q["alpha"] = q

            model = lgb.LGBMRegressor(**params_q)
            model.fit(X, y)

            self.models[q] = model

        self.is_fitted = True

    def predict(self, X: pd.DataFrame, quantile: float = 0.5) -> np.ndarray:
        """Generate price forecast for a specific quantile.

        Args:
            X: Feature DataFrame.
            quantile: Quantile to predict (default: 0.5 for median).

        Returns:
            np.ndarray: Predicted prices at specified quantile.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if quantile not in self.models:
            raise ValueError(
                f"Quantile {quantile} not available. "
                f"Available quantiles: {list(self.models.keys())}"
            )

        return self.models[quantile].predict(X)

    def predict_interval(
        self, X: pd.DataFrame, lower: float = 0.1, upper: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty interval.

        Args:
            X: Feature DataFrame.
            lower: Lower quantile for prediction interval.
            upper: Upper quantile for prediction interval.

        Returns:
            tuple: (median_prediction, lower_bound, upper_bound)
        """
        median = self.predict(X, quantile=0.5)
        lower_bound = self.predict(X, quantile=lower)
        upper_bound = self.predict(X, quantile=upper)

        return median, lower_bound, upper_bound

    def get_feature_importance(self, quantile: float = 0.5) -> pd.Series:
        """Get feature importance for a specific quantile model.

        Args:
            quantile: Quantile model to extract importance from.

        Returns:
            pd.Series: Feature importances sorted by value.
        """
        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not available")

        model = self.models[quantile]
        importance = pd.Series(
            model.feature_importances_, index=model.feature_name_
        ).sort_values(ascending=False)

        return importance


class EnsemblePriceForecaster(PriceForecaster):
    """Ensemble forecaster combining multiple models.

    This forecaster combines LightGBM, Random Forest, and Ridge regression
    using weighted averaging for robust predictions.

    Attributes:
        lgbm_model: LightGBM quantile model.
        rf_model: Random Forest model.
        ridge_model: Ridge regression model.
        weights: Weights for model averaging.
    """

    def __init__(
        self,
        lgbm_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        ridge_params: Optional[Dict] = None,
        weights: Optional[List[float]] = None,
    ):
        """Initialize ensemble forecaster.

        Args:
            lgbm_params: Parameters for LightGBM model.
            rf_params: Parameters for Random Forest model.
            ridge_params: Parameters for Ridge regression.
            weights: Model weights for averaging [lgbm, rf, ridge].
        """
        super().__init__()

        # Default parameters
        if lgbm_params is None:
            lgbm_params = {"random_state": 42}

        if rf_params is None:
            rf_params = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1,
            }

        if ridge_params is None:
            ridge_params = {"alpha": 1.0, "random_state": 42}

        # Initialize models
        self.lgbm_model = LightGBMQuantileForecaster(**lgbm_params)
        self.rf_model = RandomForestRegressor(**rf_params)
        self.ridge_model = Ridge(**ridge_params)
        self.scaler = StandardScaler()

        # Weights for averaging (default: equal weights)
        self.weights = weights if weights is not None else [0.5, 0.3, 0.2]

        if len(self.weights) != 3:
            raise ValueError("Weights must be a list of 3 values for [lgbm, rf, ridge]")

        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train all ensemble models.

        Args:
            X: Feature DataFrame.
            y: Target series (prices).
        """
        # Standardize features for Ridge (others are tree-based, don't need scaling)
        X_scaled = self.scaler.fit_transform(X)

        # Train all models
        self.lgbm_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.ridge_model.fit(X_scaled, y)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble forecast.

        Args:
            X: Feature DataFrame.

        Returns:
            np.ndarray: Weighted average predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        # Get predictions from each model
        pred_lgbm = self.lgbm_model.predict(X, quantile=0.5)
        pred_rf = self.rf_model.predict(X)
        pred_ridge = self.ridge_model.predict(X_scaled)

        # Weighted average
        ensemble_pred = (
            self.weights[0] * pred_lgbm
            + self.weights[1] * pred_rf
            + self.weights[2] * pred_ridge
        )

        return ensemble_pred

    def get_individual_predictions(
        self, X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model.

        Args:
            X: Feature DataFrame.

        Returns:
            dict: Predictions from each model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        return {
            "lgbm": self.lgbm_model.predict(X, quantile=0.5),
            "random_forest": self.rf_model.predict(X),
            "ridge": self.ridge_model.predict(X_scaled),
            "ensemble": self.predict(X),
        }
