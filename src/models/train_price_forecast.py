"""Training script for price forecasting models with MLflow tracking.

This script trains and evaluates price forecasting models with comprehensive
experiment tracking using MLflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from mlops import MLflowTracker
from src.models.data_loader import prepare_price_forecasting_dataset
from src.models.models import (
    EnsemblePriceForecaster,
    LightGBMQuantileForecaster,
)

warnings.filterwarnings("ignore")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series,
    title: str = "Price Forecast",
    n_points: int = 500,
) -> plt.Figure:
    """Plot actual vs predicted prices.

    Args:
        y_true: True prices.
        y_pred: Predicted prices.
        dates: Datetime index.
        title: Plot title.
        n_points: Number of points to display.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Limit to last n_points for visibility
    idx = slice(-n_points, None)

    # Time series plot
    axes[0].plot(dates.iloc[idx], y_true[idx], label="Actual", alpha=0.7, linewidth=1.5)
    axes[0].plot(
        dates.iloc[idx], y_pred[idx], label="Predicted", alpha=0.7, linewidth=1.5
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price (EUR/MWh)")
    axes[0].set_title(f"{title} - Time Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[1].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Price (EUR/MWh)")
    axes[1].set_ylabel("Predicted Price (EUR/MWh)")
    axes[1].set_title(f"{title} - Scatter Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_quantile_predictions(
    y_true: np.ndarray,
    pred_median: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    dates: pd.Series,
    n_points: int = 300,
) -> plt.Figure:
    """Plot quantile prediction intervals.

    Args:
        y_true: True prices.
        pred_median: Median predictions.
        pred_lower: Lower bound predictions.
        pred_upper: Upper bound predictions.
        dates: Datetime index.
        n_points: Number of points to display.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Limit to last n_points
    idx = slice(-n_points, None)

    ax.plot(
        dates.iloc[idx],
        y_true[idx],
        label="Actual",
        linewidth=2,
        color="black",
        alpha=0.7,
    )
    ax.plot(
        dates.iloc[idx],
        pred_median[idx],
        label="Median Forecast",
        linewidth=2,
        color="blue",
    )
    ax.fill_between(
        dates.iloc[idx],
        pred_lower[idx],
        pred_upper[idx],
        alpha=0.3,
        color="blue",
        label="80% Prediction Interval",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Probabilistic Price Forecast with Uncertainty Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def train_lgbm_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates_test: pd.Series,
    track_mlflow: bool = True,
) -> LightGBMQuantileForecaster:
    """Train LightGBM Quantile forecasting model.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        dates_test: Test datetime index.
        track_mlflow: Whether to track with MLflow.

    Returns:
        LightGBMQuantileForecaster: Trained model.
    """
    print("=" * 70)
    print("Training LightGBM Quantile Forecaster")
    print("=" * 70)

    if track_mlflow:
        tracker = MLflowTracker("price_forecasting", run_name="lgbm-quantile")
        tracker.__enter__()

    # Model parameters
    params = {
        "quantiles": [0.1, 0.5, 0.9],
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "random_state": 42,
    }

    if track_mlflow:
        tracker.log_params(params)

    # Train model
    model = LightGBMQuantileForecaster(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train, quantile=0.5)
    y_test_pred = model.predict(X_test, quantile=0.5)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    # Log metrics
    if track_mlflow:
        tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Log model
        tracker.log_model(model, "model", registered_model_name="price_forecast_lgbm")

        # Log plots
        fig = plot_predictions(y_test, y_test_pred, dates_test, "LightGBM Quantile")
        tracker.log_figure(fig, "predictions_lgbm.png")
        plt.close(fig)

        # Plot quantile predictions
        pred_median, pred_lower, pred_upper = model.predict_interval(X_test)
        fig_quantile = plot_quantile_predictions(
            y_test, pred_median, pred_lower, pred_upper, dates_test
        )
        tracker.log_figure(fig_quantile, "quantile_predictions.png")
        plt.close(fig_quantile)

        # Log feature importance
        importance = model.get_feature_importance(quantile=0.5)
        tracker.log_dict(importance.head(20).to_dict(), "feature_importance.json")

        tracker.__exit__(None, None, None)

    # Print results
    print(f"\nTrain RMSE: {train_metrics['rmse']:.2f} EUR/MWh")
    print(f"Train R²:   {train_metrics['r2']:.4f}")
    print(f"Test RMSE:  {test_metrics['rmse']:.2f} EUR/MWh")
    print(f"Test R²:    {test_metrics['r2']:.4f}")
    print(f"Test MAPE:  {test_metrics['mape']:.2f}%")
    print(f"Test DA:    {test_metrics['directional_accuracy']:.2f}%")

    return model


def train_ensemble_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates_test: pd.Series,
    track_mlflow: bool = True,
) -> EnsemblePriceForecaster:
    """Train ensemble forecasting model.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        dates_test: Test datetime index.
        track_mlflow: Whether to track with MLflow.

    Returns:
        EnsemblePriceForecaster: Trained model.
    """
    print("\n" + "=" * 70)
    print("Training Ensemble Forecaster")
    print("=" * 70)

    if track_mlflow:
        tracker = MLflowTracker("price_forecasting", run_name="ensemble")
        tracker.__enter__()

    # Model parameters
    params = {
        "weights": [0.5, 0.3, 0.2],  # LightGBM, RandomForest, Ridge
    }

    if track_mlflow:
        tracker.log_params(params)

    # Train model
    model = EnsemblePriceForecaster(weights=params["weights"])
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    # Log metrics
    if track_mlflow:
        tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Log model
        tracker.log_model(
            model, "model", registered_model_name="price_forecast_ensemble"
        )

        # Log plots
        fig = plot_predictions(y_test, y_test_pred, dates_test, "Ensemble")
        tracker.log_figure(fig, "predictions_ensemble.png")
        plt.close(fig)

        # Log individual model predictions
        individual_preds = model.get_individual_predictions(X_test)
        for name, preds in individual_preds.items():
            if name != "ensemble":
                rmse = np.sqrt(np.mean((y_test - preds) ** 2))
                tracker.log_metric(f"test_{name}_rmse", rmse)

        tracker.__exit__(None, None, None)

    # Print results
    print(f"\nTrain RMSE: {train_metrics['rmse']:.2f} EUR/MWh")
    print(f"Train R²:   {train_metrics['r2']:.4f}")
    print(f"Test RMSE:  {test_metrics['rmse']:.2f} EUR/MWh")
    print(f"Test R²:    {test_metrics['r2']:.4f}")
    print(f"Test MAPE:  {test_metrics['mape']:.2f}%")
    print(f"Test DA:    {test_metrics['directional_accuracy']:.2f}%")

    return model


def walk_forward_validation(df: pd.DataFrame, feature_cols: list, n_splits: int = 5):
    """Perform walk-forward validation for price forecasting.

    Args:
        df: Complete dataset with features and target.
        feature_cols: List of feature column names.
        n_splits: Number of train-test splits.
    """
    print("\n" + "=" * 70)
    print("Walk-Forward Validation")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        print(f"\nFold {fold}/{n_splits}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["price"]
        X_test = test_df[feature_cols]
        y_test = test_df["price"]

        # Train LightGBM model
        model = LightGBMQuantileForecaster(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        metrics["fold"] = fold
        results.append(metrics)

        print(f"  Test RMSE: {metrics['rmse']:.2f}")
        print(f"  Test R²:   {metrics['r2']:.4f}")

    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Walk-Forward Validation Summary")
    print("=" * 70)
    print(
        f"Average Test RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}"
    )
    print(
        f"Average Test R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}"
    )
    print(
        f"Average Test MAPE: {results_df['mape'].mean():.2f}% ± {results_df['mape'].std():.2f}%"
    )


def main(
    data_path: str = "data/modified_data/energy_hourly_regional.csv",
    model_type: str = "both",
    track_mlflow: bool = True,
    run_walk_forward: bool = False,
):
    """Main training pipeline.

    Args:
        data_path: Path to hourly load data.
        model_type: Model to train ('lgbm', 'ensemble', or 'both').
        track_mlflow: Whether to track experiments with MLflow.
        run_walk_forward: Whether to run walk-forward validation.
    """
    print("\n" + "=" * 70)
    print("PRICE FORECASTING MODEL TRAINING")
    print("=" * 70)

    # Load and prepare data
    print("\nLoading and preparing dataset...")
    print("[WARNING]  CRITICAL: Using REAL ENTSO-E price data (simulate_prices=False)")
    print("   Simulated prices are NOT allowed in production training.")

    # [OK] EXPLICITLY USE REAL DATA - NO SIMULATION
    df, feature_cols = prepare_price_forecasting_dataset(
        data_path,
        simulate_prices=False,  # [OK] CRITICAL: Must use real data
        strict_validation=True,  # [OK] Crash if simulated data detected
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Date range: {df['datetime_hour'].min()} to {df['datetime_hour'].max()}")

    # Train-test split (80/20 split, maintaining time order)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["price"]
    X_test = test_df[feature_cols]
    y_test = test_df["price"]
    dates_test = test_df["datetime_hour"]

    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size:  {len(test_df)}")

    # Train models
    if model_type in ["lgbm", "both"]:
        train_lgbm_quantile_model(
            X_train, y_train, X_test, y_test, dates_test, track_mlflow
        )

    if model_type in ["ensemble", "both"]:
        train_ensemble_model(X_train, y_train, X_test, y_test, dates_test, track_mlflow)

    # Walk-forward validation
    if run_walk_forward:
        walk_forward_validation(df, feature_cols)

    print("\n" + "=" * 70)
    print("[OK] Training complete!")
    print("=" * 70)

    if track_mlflow:
        print("\nView results in MLflow UI:")
        print(
            "  mlflow ui --backend-store-uri file:///home/user/energy-demand-forecast/mlruns"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train price forecasting models")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/modified_data/energy_hourly_regional.csv",
        help="Path to hourly load data",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "ensemble", "both"],
        default="both",
        help="Model type to train",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_type=args.model,
        track_mlflow=not args.no_mlflow,
        run_walk_forward=args.walk_forward,
    )
