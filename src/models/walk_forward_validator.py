"""Walk-forward validation utility for price forecasting models.

This module provides a reusable walk-forward validation framework that:
1. Prevents data leakage
2. Simulates realistic trading conditions (daily retrain + predict)
3. Calculates comprehensive metrics
4. Supports any sklearn-compatible model

The workflow:
- Day 1: Train on historical data → Predict next 24h
- Day 2: Retrain with Day 1 actual data → Predict next 24h
- Day 3: Retrain with Days 1-2 actual data → Predict next 24h
- ... continue for entire test period

This mimics production: every day we get new data and retrain before trading.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def calculate_price_forecast_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive metrics for price forecasting.

    Args:
        y_true: Actual prices
        y_pred: Predicted prices

    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # sMAPE: symmetric, handles near-zero prices (common in power markets)
    mape = (
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        * 100
    )
    r2 = r2_score(y_true, y_pred)

    # Direction accuracy (did we predict price increase/decrease correctly?)
    price_changes_true = np.diff(y_true)
    price_changes_pred = np.diff(y_pred)
    direction_accuracy = (
        np.mean(np.sign(price_changes_true) == np.sign(price_changes_pred)) * 100
    )

    # Quantile metrics (useful for risk management)
    errors = y_pred - y_true

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "bias": np.mean(errors),  # Systematic over/under prediction
        "error_std": np.std(errors),
        "error_q05": np.percentile(errors, 5),
        "error_q95": np.percentile(errors, 95),
        "max_overpredict": np.max(errors),
        "max_underpredict": np.min(errors),
    }

    return metrics


def walk_forward_validate(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    target_col: str = "price",
    initial_train_days: int = 365,
    forecast_horizon_hours: int = 24,
    retrain_frequency_days: int = 1,
    expanding_window: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict]]:
    """Perform walk-forward validation with daily retraining.

    This function simulates realistic trading conditions:
    - Each day, we train on all historical data available up to that point
    - We predict the next 24 hours
    - Next day, we incorporate the actual data and retrain

    CRITICAL: No data leakage - we only use data available at prediction time.

    Args:
        df: DataFrame with datetime, features, and target
        model: Sklearn-compatible model (must have fit/predict methods)
        feature_cols: List of feature column names
        target_col: Target column name
        initial_train_days: Days of data for initial training
        forecast_horizon_hours: Forecast horizon (default 24h)
        retrain_frequency_days: How often to retrain (default 1 = daily)
        expanding_window: If True, use all past data. If False, use fixed window.
        verbose: Print progress

    Returns:
        - predictions_df: DataFrame with datetime, actual, predicted, errors
        - metrics: Dictionary with overall metrics
        - daily_metrics: List of metrics for each prediction day
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    # Validate inputs
    assert target_col in df.columns, f"Target '{target_col}' not in DataFrame"
    for col in feature_cols:
        assert col in df.columns, f"Feature '{col}' not in DataFrame"

    # Split into initial train and walk-forward test
    initial_train_end_idx = initial_train_days * 24  # Convert days to hours

    if initial_train_end_idx >= len(df):
        raise ValueError(
            f"Not enough data: need {initial_train_days} days for initial training"
        )

    # Store predictions
    predictions = []
    daily_metrics_list = []

    # Initial training
    train_df = df.iloc[:initial_train_end_idx].copy()
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    if verbose:
        logger.info(
            f"\n{'='*80}\n"
            f"WALK-FORWARD VALIDATION\n"
            f"{'='*80}\n"
            f"Model: {model.__class__.__name__}\n"
            f"Initial training: {len(train_df)} hours ({initial_train_days} days)\n"
            f"Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}\n"
            f"Features: {len(feature_cols)}\n"
            f"Forecast horizon: {forecast_horizon_hours}h\n"
            f"Retrain frequency: {retrain_frequency_days} day(s)\n"
            f"Window type: {'expanding' if expanding_window else 'rolling'}\n"
        )

    # Fit initial model
    model.fit(X_train, y_train)

    if verbose:
        train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        logger.info(f"Initial training MAE: {train_mae:.2f} EUR/MWh")

    # Walk forward through test period - ALIGN WITH CALENDAR DAYS
    # Find first complete calendar day after training period
    first_test_datetime = df.iloc[initial_train_end_idx]["datetime"]
    first_pred_date = (first_test_datetime.normalize() + pd.Timedelta(days=1)).date()

    if verbose:
        logger.info(f"First prediction date: {first_pred_date} (00:00-23:00)")

    current_date = pd.Timestamp(first_pred_date)
    iteration = 0
    days_since_retrain = 0

    while current_date <= df["datetime"].max():
        iteration += 1

        # Get all hours for current calendar day (00:00-23:00)
        next_date = current_date + pd.Timedelta(days=1)
        day_mask = (df["datetime"] >= current_date) & (df["datetime"] < next_date)
        test_df = df[day_mask].copy()

        # Only process full days (24 hours)
        if len(test_df) != forecast_horizon_hours:
            if verbose and len(test_df) > 0:
                logger.info(
                    f"Skipping {current_date.date()}: only {len(test_df)}/24 hours available"
                )
            current_date = next_date
            continue

        # Predict
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        y_pred = model.predict(X_test)

        # Store predictions
        for i, (idx, row) in enumerate(test_df.iterrows()):
            predictions.append(
                {
                    "datetime": row["datetime"],
                    "actual": y_test[i],
                    "predicted": y_pred[i],
                    "error": y_pred[i] - y_test[i],
                    "abs_error": abs(y_pred[i] - y_test[i]),
                    "iteration": iteration,
                }
            )

        # Calculate metrics for this prediction window
        window_metrics = calculate_price_forecast_metrics(y_test, y_pred)
        window_metrics["datetime"] = current_date
        window_metrics["iteration"] = iteration
        daily_metrics_list.append(window_metrics)

        # Move to next day
        current_date = next_date
        days_since_retrain += 1

        # Retrain model (every N days)
        if days_since_retrain >= retrain_frequency_days:
            # Find the index of current_date in df
            train_end_mask = df["datetime"] < current_date
            current_idx = train_end_mask.sum()

            if expanding_window:
                # Use all data up to current point
                train_df = df.iloc[:current_idx].copy()
            else:
                # Use fixed window (last initial_train_days)
                start_idx = max(0, current_idx - initial_train_days * 24)
                train_df = df.iloc[start_idx:current_idx].copy()

            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values

            model.fit(X_train, y_train)
            days_since_retrain = 0

            if verbose and iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: Retrained on {len(train_df)} hours "
                    f"({train_df['datetime'].min()} to {train_df['datetime'].max()})"
                )

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Calculate overall metrics
    overall_metrics = calculate_price_forecast_metrics(
        predictions_df["actual"].values, predictions_df["predicted"].values
    )

    overall_metrics["n_predictions"] = len(predictions_df)
    overall_metrics["n_iterations"] = iteration
    overall_metrics["coverage_hours"] = (
        predictions_df["datetime"].max() - predictions_df["datetime"].min()
    ).total_seconds() / 3600

    if verbose:
        logger.info(
            f"\n{'='*80}\n"
            f"WALK-FORWARD RESULTS\n"
            f"{'='*80}\n"
            f"Predictions: {len(predictions_df):,} hours\n"
            f"Iterations: {iteration}\n"
            f"Date range: {predictions_df['datetime'].min()} to {predictions_df['datetime'].max()}\n"
            f"\n"
            f"Overall Metrics:\n"
            f"  MAE:  {overall_metrics['mae']:.2f} EUR/MWh\n"
            f"  RMSE: {overall_metrics['rmse']:.2f} EUR/MWh\n"
            f"  MAPE: {overall_metrics['mape']:.2f}%\n"
            f"  R²:   {overall_metrics['r2']:.3f}\n"
            f"  Direction Accuracy: {overall_metrics['direction_accuracy']:.2f}%\n"
            f"  Bias: {overall_metrics['bias']:.2f} EUR/MWh\n"
        )

    return predictions_df, overall_metrics, daily_metrics_list


def plot_walk_forward_results(
    predictions_df: pd.DataFrame, daily_metrics: List[Dict], model_name: str = "Model"
):
    """Plot walk-forward validation results.

    Args:
        predictions_df: DataFrame from walk_forward_validate
        daily_metrics: List of daily metrics
        model_name: Name of the model for plot titles
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"{model_name} - Walk-Forward Validation Results",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Actual vs Predicted (time series)
    ax = axes[0, 0]
    sample_size = min(7 * 24, len(predictions_df))  # Show first week
    sample_df = predictions_df.head(sample_size)
    ax.plot(
        sample_df["datetime"],
        sample_df["actual"],
        label="Actual",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        sample_df["datetime"],
        sample_df["predicted"],
        label="Predicted",
        alpha=0.7,
        linewidth=1.5,
    )
    ax.set_title("Actual vs Predicted (First Week)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # 2. Scatter plot
    ax = axes[0, 1]
    ax.scatter(predictions_df["actual"], predictions_df["predicted"], alpha=0.3, s=10)
    ax.plot(
        [predictions_df["actual"].min(), predictions_df["actual"].max()],
        [predictions_df["actual"].min(), predictions_df["actual"].max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual Price (EUR/MWh)")
    ax.set_ylabel("Predicted Price (EUR/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Error distribution
    ax = axes[1, 0]
    ax.hist(predictions_df["error"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
    ax.axvline(
        predictions_df["error"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f'Mean error: {predictions_df["error"].mean():.2f}',
    )
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (EUR/MWh)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Error over time
    ax = axes[1, 1]
    ax.plot(
        predictions_df["datetime"], predictions_df["error"], alpha=0.5, linewidth=0.5
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("Prediction Error Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (EUR/MWh)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # 5. MAE over time (daily metrics)
    if daily_metrics:
        ax = axes[2, 0]
        daily_df = pd.DataFrame(daily_metrics)
        ax.plot(
            daily_df["datetime"],
            daily_df["mae"],
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
        ax.axhline(
            daily_df["mae"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Mean MAE: {daily_df["mae"].mean():.2f}',
        )
        ax.set_title("MAE Evolution Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # 6. Metrics summary table
    ax = axes[2, 1]
    ax.axis("off")

    metrics_summary = [
        ["Metric", "Value"],
        ["MAE", f"{predictions_df['abs_error'].mean():.2f} EUR/MWh"],
        ["RMSE", f"{np.sqrt(np.mean(predictions_df['error']**2)):.2f} EUR/MWh"],
        ["Bias", f"{predictions_df['error'].mean():.2f} EUR/MWh"],
        ["Error Std", f"{predictions_df['error'].std():.2f} EUR/MWh"],
        ["Max Over-predict", f"{predictions_df['error'].max():.2f} EUR/MWh"],
        ["Max Under-predict", f"{predictions_df['error'].min():.2f} EUR/MWh"],
        ["N Predictions", f"{len(predictions_df):,}"],
    ]

    table = ax.table(
        cellText=metrics_summary, loc="center", cellLoc="left", colWidths=[0.5, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()
    return fig


def save_walk_forward_results(
    predictions_df: pd.DataFrame,
    metrics: Dict[str, float],
    daily_metrics: List[Dict],
    model_name: str,
    output_dir: str = "research_notebooks/reports",
):
    """Save walk-forward validation results to files.

    Args:
        predictions_df: DataFrame with predictions
        metrics: Overall metrics dictionary
        daily_metrics: List of daily metrics
        model_name: Name of the model
        output_dir: Output directory
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_file = output_path / f"{model_name}_predictions.csv"
    predictions_df.to_csv(pred_file, index=False)

    # Save daily metrics
    daily_file = output_path / f"{model_name}_daily_metrics.csv"
    pd.DataFrame(daily_metrics).to_csv(daily_file, index=False)

    # Save overall metrics
    metrics_file = output_path / f"{model_name}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Walk-Forward Validation Results: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info(
        f"\n[OK] Results saved:\n"
        f"   Predictions: {pred_file}\n"
        f"   Daily metrics: {daily_file}\n"
        f"   Overall metrics: {metrics_file}"
    )


if __name__ == "__main__":
    # Test with dummy data
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 80)
    print("TESTING WALK-FORWARD VALIDATOR")
    print("=" * 80)

    # Create dummy dataset
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="H")
    n = len(dates)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "price": 50
            + 20 * np.sin(2 * np.pi * dates.hour / 24)
            + np.random.normal(0, 10, n),
            "feature1": np.random.normal(0, 1, n),
            "feature2": np.random.normal(0, 1, n),
            "feature3": np.random.normal(0, 1, n),
        }
    )

    # Test with simple linear regression
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    feature_cols = ["feature1", "feature2", "feature3"]

    predictions_df, metrics, daily_metrics = walk_forward_validate(
        df=df,
        model=model,
        feature_cols=feature_cols,
        target_col="price",
        initial_train_days=180,
        forecast_horizon_hours=24,
        retrain_frequency_days=7,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    print(predictions_df.head(10))

    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
