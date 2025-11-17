"""
Naive Baseline Models for Price Forecasting

Implements simple baseline models to compare against ML models.
Essential for validating that complex models actually add value.

Baselines implemented:
1. Persistence (Last Value): y(t+h) = y(t)
2. Historical Mean: y(t+h) = mean(y[t-window:t])
3. Seasonal Naive: y(t+h) = y(t-24h) (same hour yesterday)
4. Seasonal Mean: y(t+h) = mean(y[same_hour, last_N_days])
5. Moving Average: y(t+h) = MA(y, window)

Usage:
    baseline = PersistenceBaseline()
    predictions = baseline.fit_predict(train_data, test_data)

    comparator = BaselineComparator()
    results = comparator.compare_all(y_train, y_test, ml_predictions)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class BaselineMetrics:
    """Metrics for baseline comparison."""

    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float

    def __str__(self) -> str:
        return (
            f"{self.model_name:20} | "
            f"MAE: {self.mae:7.2f} | "
            f"RMSE: {self.rmse:7.2f} | "
            f"R²: {self.r2:6.3f} | "
            f"MAPE: {self.mape:6.2f}% | "
            f"Dir Acc: {self.directional_accuracy:5.1f}%"
        )


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def fit(self, y_train: pd.Series) -> "BaselineModel":
        """Fit the baseline model."""
        pass

    @abstractmethod
    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Generate predictions."""
        pass

    def fit_predict(
        self, y_train: pd.Series, y_test_index: pd.DatetimeIndex, horizon: int = 24
    ) -> np.ndarray:
        """Fit and predict for test set."""
        self.fit(y_train)

        # Combine train and test for rolling predictions
        all_data = y_train.copy()
        predictions = []

        for i in range(len(y_test_index)):
            # Predict next value
            pred = self.predict(all_data, horizon=horizon)
            predictions.append(pred[0] if isinstance(pred, np.ndarray) else pred)

            # Note: For true walk-forward, we'd append actual value here
            # For now, using one-step-ahead prediction

        return np.array(predictions)


class PersistenceBaseline(BaselineModel):
    """
    Persistence model: y(t+h) = y(t)

    Also known as "naive forecast" or "random walk".
    """

    def __init__(self, lag: int = 1):
        """
        Args:
            lag: Number of periods to lag (default: 1 = last value)
        """
        self.lag = lag
        self.last_value = None

    def fit(self, y_train: pd.Series) -> "PersistenceBaseline":
        """Store last value from training."""
        self.last_value = y_train.iloc[-self.lag]
        return self

    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Predict using last observed value."""
        last_val = y_history.iloc[-self.lag]
        return np.full(horizon, last_val)


class HistoricalMeanBaseline(BaselineModel):
    """
    Historical mean: y(t+h) = mean(y[t-window:t])
    """

    def __init__(self, window: int = 168):
        """
        Args:
            window: Lookback window in hours (default: 168 = 1 week)
        """
        self.window = window
        self.mean_value = None

    def fit(self, y_train: pd.Series) -> "HistoricalMeanBaseline":
        """Calculate mean of last window."""
        self.mean_value = y_train.iloc[-self.window :].mean()
        return self

    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Predict using rolling mean."""
        if len(y_history) < self.window:
            mean_val = y_history.mean()
        else:
            mean_val = y_history.iloc[-self.window :].mean()
        return np.full(horizon, mean_val)


class SeasonalNaiveBaseline(BaselineModel):
    """
    Seasonal naive: y(t+h) = y(t-season)

    For hourly electricity prices, season = 24 (same hour yesterday).
    """

    def __init__(self, season: int = 24):
        """
        Args:
            season: Seasonal period (default: 24 hours)
        """
        self.season = season

    def fit(self, y_train: pd.Series) -> "SeasonalNaiveBaseline":
        """No fitting required for seasonal naive."""
        return self

    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Predict using value from same time last season."""
        if len(y_history) < self.season:
            # Fallback to last value if not enough history
            return np.full(horizon, y_history.iloc[-1])

        predictions = []
        for h in range(horizon):
            # Get value from one season ago
            idx = -self.season - h
            if abs(idx) <= len(y_history):
                predictions.append(y_history.iloc[idx])
            else:
                predictions.append(y_history.iloc[-1])

        return np.array(predictions)


class SeasonalMeanBaseline(BaselineModel):
    """
    Seasonal mean: y(t+h) = mean(y[same_hour, last_N_days])

    Averages values from the same hour across multiple days.
    """

    def __init__(self, season: int = 24, n_seasons: int = 7):
        """
        Args:
            season: Seasonal period (24 for hourly)
            n_seasons: Number of past seasons to average (7 = 1 week)
        """
        self.season = season
        self.n_seasons = n_seasons

    def fit(self, y_train: pd.Series) -> "SeasonalMeanBaseline":
        """No fitting required."""
        return self

    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Predict using mean of same seasonal periods."""
        predictions = []

        for h in range(horizon):
            # Collect values from same hour in past N days
            seasonal_values = []
            for i in range(1, self.n_seasons + 1):
                idx = -(self.season * i) - h
                if abs(idx) <= len(y_history):
                    seasonal_values.append(y_history.iloc[idx])

            if seasonal_values:
                predictions.append(np.mean(seasonal_values))
            else:
                predictions.append(y_history.iloc[-1])

        return np.array(predictions)


class MovingAverageBaseline(BaselineModel):
    """
    Simple moving average: y(t+h) = MA(y, window)
    """

    def __init__(self, window: int = 24):
        """
        Args:
            window: MA window size (default: 24 hours)
        """
        self.window = window

    def fit(self, y_train: pd.Series) -> "MovingAverageBaseline":
        """No fitting required for MA."""
        return self

    def predict(self, y_history: pd.Series, horizon: int = 1) -> np.ndarray:
        """Predict using moving average."""
        if len(y_history) < self.window:
            ma = y_history.mean()
        else:
            ma = y_history.iloc[-self.window :].mean()

        return np.full(horizon, ma)


class BaselineComparator:
    """Compare multiple baseline models and ML models."""

    def __init__(self):
        self.baselines = {
            "Persistence": PersistenceBaseline(lag=1),
            "Persistence_24h": PersistenceBaseline(lag=24),
            "Historical_Mean": HistoricalMeanBaseline(window=168),
            "Seasonal_Naive": SeasonalNaiveBaseline(season=24),
            "Seasonal_Mean": SeasonalMeanBaseline(season=24, n_seasons=7),
            "Moving_Average_24h": MovingAverageBaseline(window=24),
        }

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> BaselineMetrics:
        """Calculate all metrics for a model."""
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return BaselineMetrics(
                model_name=model_name,
                mae=np.nan,
                rmse=np.nan,
                r2=np.nan,
                mape=np.nan,
                directional_accuracy=np.nan,
            )

        # Basic metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)

        # MAPE
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

        # Directional accuracy (for trading)
        if len(y_true_clean) > 1:
            true_direction = np.diff(y_true_clean) > 0
            pred_direction = np.diff(y_pred_clean) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = np.nan

        return BaselineMetrics(
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            directional_accuracy=directional_accuracy,
        )

    def compare_all(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        ml_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Compare all baselines and optionally ML models.

        Args:
            y_train: Training data
            y_test: Test data (actual values)
            ml_predictions: Dict of {model_name: predictions}

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        # Test baselines
        print("\n" + "=" * 100)
        print("BASELINE MODEL COMPARISON")
        print("=" * 100)

        for name, model in self.baselines.items():
            # Generate predictions
            predictions = model.fit_predict(y_train, y_test.index)

            # Calculate metrics
            metrics = self.calculate_metrics(y_test.values, predictions, name)
            results.append(metrics)

            print(metrics)

        # Test ML models if provided
        if ml_predictions:
            print("\n" + "-" * 100)
            print("MACHINE LEARNING MODELS")
            print("-" * 100)

            for name, predictions in ml_predictions.items():
                metrics = self.calculate_metrics(y_test.values, predictions, name)
                results.append(metrics)
                print(metrics)

        print("=" * 100 + "\n")

        # Convert to DataFrame
        df = pd.DataFrame([vars(m) for m in results])
        df = df.sort_values("rmse")

        return df

    def plot_comparison(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        ml_predictions: Optional[Dict[str, np.ndarray]] = None,
        n_points: int = 168,
    ):
        """Plot predictions from all models."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Plot each baseline
        for idx, (name, model) in enumerate(self.baselines.items()):
            ax = axes[idx]

            # Generate predictions
            predictions = model.fit_predict(y_train, y_test.index)

            # Plot
            ax.plot(
                y_test.index[:n_points], y_test.values[:n_points], label="Actual", lw=2
            )
            ax.plot(
                y_test.index[:n_points],
                predictions[:n_points],
                label="Predicted",
                alpha=0.7,
                ls="--",
            )

            # Calculate metrics
            metrics = self.calculate_metrics(y_test.values, predictions, name)

            ax.set_title(f"{name}\nRMSE: {metrics.rmse:.2f}, R²: {metrics.r2:.3f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (EUR/MWh)")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic price data
    np.random.seed(42)

    # Create hourly data for 30 days
    dates = pd.date_range("2024-01-01", periods=30 * 24, freq="H")

    # Synthetic price with daily seasonality + noise
    hour_of_day = dates.hour
    daily_pattern = 40 + 15 * np.sin(2 * np.pi * hour_of_day / 24)
    noise = np.random.normal(0, 5, len(dates))
    prices = daily_pattern + noise

    # Create series
    price_series = pd.Series(prices, index=dates)

    # Split train/test
    split = int(len(price_series) * 0.8)
    y_train = price_series[:split]
    y_test = price_series[split:]

    # Compare baselines
    comparator = BaselineComparator()
    results = comparator.compare_all(y_train, y_test)

    print("\n📊 Results Summary:")
    print(results.to_string(index=False))

    # Plot comparison
    fig = comparator.plot_comparison(y_train, y_test, n_points=168)
    fig.savefig("outputs/baseline_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✅ Plot saved to outputs/baseline_comparison.png")
