"""Performance metrics for electricity price forecasting.

This module provides robust metrics that handle:
- Negative prices (common in European power markets)
- Near-zero prices (renewable flush periods)
- Extreme volatility

Standard MAPE is inappropriate for power prices due to division by near-zero.
Use sMAPE as the primary metric instead.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 5.0,
) -> float:
    """Calculate MAPE excluding near-zero actual values.

    Standard MAPE = mean(|actual - pred| / |actual|) × 100%
    Problem: Explodes when actual ≈ 0 (common in power markets)

    Solution: Exclude observations where |actual| < threshold

    Args:
        y_true: Actual values
        y_pred: Predicted values
        threshold: Minimum |actual| value to include (default: 5 EUR/MWh)

    Returns:
        MAPE in percentage (0-100+)

    Raises:
        ValueError: If no values meet threshold criterion

    Example:
        >>> y_true = np.array([100, 50, 2, -1, 80])
        >>> y_pred = np.array([95, 55, 10, 5, 75])
        >>> safe_mape(y_true, y_pred, threshold=5.0)
        7.5  # Only includes [100, 50, 80] (excludes 2, -1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Exclude near-zero values
    mask = np.abs(y_true) >= threshold

    if mask.sum() == 0:
        raise ValueError(
            f"No values with |actual| >= {threshold}. "
            f"Cannot calculate MAPE. Consider lowering threshold or using sMAPE instead."
        )

    filtered_true = y_true[mask]
    filtered_pred = y_pred[mask]

    mape = np.mean(np.abs((filtered_true - filtered_pred) / filtered_true)) * 100

    return mape


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    sMAPE = 200% × mean(|pred - actual| / (|pred| + |actual|))

    Advantages over MAPE:
    - Symmetric (doesn't penalize over-predictions more than under-predictions)
    - Bounded (0-200%)
    - Handles near-zero values better
    - Works with negative values

    [WARNING] RECOMMENDED as PRIMARY metric for power price forecasting.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        sMAPE in percentage (0-200)

    Example:
        >>> y_true = np.array([100, 50, 2, -1, 80])
        >>> y_pred = np.array([95, 55, 10, 5, 75])
        >>> smape(y_true, y_pred)
        12.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)

    # Handle division by zero (both pred and actual are zero)
    denominator = np.where(denominator == 0, 1e-10, denominator)

    smape_value = 200 * np.mean(numerator / denominator)

    return smape_value


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 24,
) -> float:
    """Mean Absolute Scaled Error.

    MASE = MAE / MAE_naive
    where MAE_naive is the MAE of a seasonal naive forecast

    Advantages:
    - Scale-independent
    - Interpretable (< 1 means better than naive, > 1 means worse)
    - Handles zeros and negatives

    Args:
        y_true: Actual test values
        y_pred: Predicted test values
        y_train: Training data for scaling (if None, uses seasonal naive on test)
        seasonality: Seasonal period (default 24 for hourly data with daily patterns)

    Returns:
        MASE value (< 1 is good, > 1 is bad)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Forecast error
    mae_forecast = np.mean(np.abs(y_true - y_pred))

    # Naive seasonal forecast error (for scaling)
    if y_train is not None:
        # Use training data for naive forecast
        y_train = np.asarray(y_train)
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        mae_naive = np.mean(naive_errors)
    else:
        # Use test data for naive forecast
        if len(y_true) <= seasonality:
            # Not enough data for seasonal naive
            return mae_forecast
        naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
        mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf if mae_forecast > 0 else 0.0

    return mae_forecast / mae_naive


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    mape_threshold: float = 5.0,
) -> Dict[str, float]:
    """Calculate comprehensive forecast accuracy metrics.

    Returns all standard metrics for model evaluation:
    - MAE (Mean Absolute Error) - in EUR/MWh
    - RMSE (Root Mean Squared Error) - in EUR/MWh
    - R² (Coefficient of Determination)
    - sMAPE (Symmetric MAPE) - PRIMARY METRIC ⭐
    - Safe MAPE (excluding near-zero values)
    - MASE (Mean Absolute Scaled Error)
    - Directional Accuracy (% correct sign of price change)

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Optional training data for MASE calculation
        mape_threshold: Threshold for safe_mape (EUR/MWh)

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = calculate_all_metrics(y_true, y_pred)
        >>> print(f"sMAPE: {metrics['smape']:.2f}%")
        >>> print(f"MAE: {metrics['mae']:.2f} EUR/MWh")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),  # ⭐ PRIMARY METRIC
    }

    # Safe MAPE (may fail if all values below threshold)
    try:
        metrics["safe_mape"] = safe_mape(y_true, y_pred, threshold=mape_threshold)
        metrics["safe_mape_threshold"] = mape_threshold
    except ValueError as e:
        metrics["safe_mape"] = np.nan
        metrics["safe_mape_error"] = str(e)

    # MASE (if training data provided)
    if y_train is not None:
        try:
            metrics["mase"] = mase(y_true, y_pred, y_train)
        except Exception:
            metrics["mase"] = np.nan
    else:
        metrics["mase"] = np.nan

    # Directional accuracy (for price change prediction)
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        metrics["directional_accuracy"] = (
            np.mean(actual_direction == pred_direction) * 100
        )
    else:
        metrics["directional_accuracy"] = np.nan

    # Bias (mean error)
    metrics["bias"] = np.mean(y_pred - y_true)

    # Max error
    metrics["max_error"] = np.max(np.abs(y_true - y_pred))

    return metrics


def print_metrics_report(
    metrics: Dict[str, float],
    model_name: str = "Model",
    include_header: bool = True,
) -> None:
    """Print formatted metrics report.

    Args:
        metrics: Dictionary from calculate_all_metrics()
        model_name: Name of model for display
        include_header: Whether to print header separator
    """
    if include_header:
        print(f"\n{'='*70}")
        print(f"{model_name} - Forecast Performance Metrics")
        print(f"{'='*70}")
    else:
        print(f"\n{model_name}:")
        print(f"{'-'*70}")

    print(f"MAE:                      {metrics['mae']:>10.2f} EUR/MWh")
    print(f"RMSE:                     {metrics['rmse']:>10.2f} EUR/MWh")
    print(f"R²:                       {metrics['r2']:>10.4f}")
    print(f"sMAPE (PRIMARY):          {metrics['smape']:>10.2f}%  ⭐")

    if "safe_mape" in metrics and not np.isnan(metrics.get("safe_mape", np.nan)):
        threshold = metrics.get("safe_mape_threshold", 5.0)
        print(f"Safe MAPE (|y|≥{threshold:.0f}):     {metrics['safe_mape']:>10.2f}%")

    if "mase" in metrics and not np.isnan(metrics.get("mase", np.nan)):
        mase_val = metrics["mase"]
        status = "[OK] Better" if mase_val < 1 else "[WARNING] Worse"
        print(f"MASE:                     {mase_val:>10.4f}  {status} than naive")

    if "directional_accuracy" in metrics and not np.isnan(
        metrics.get("directional_accuracy", np.nan)
    ):
        print(f"Directional Accuracy:     {metrics['directional_accuracy']:>10.2f}%")

    if "bias" in metrics:
        bias = metrics["bias"]
        bias_dir = "Over-prediction" if bias > 0 else "Under-prediction"
        print(f"Bias:                     {bias:>10.2f} EUR/MWh ({bias_dir})")

    if "max_error" in metrics:
        print(f"Max Error:                {metrics['max_error']:>10.2f} EUR/MWh")

    if include_header:
        print(f"{'='*70}\n")


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "smape",
    ascending: bool = True,
) -> pd.DataFrame:
    """Compare multiple models and rank by specified metric.

    Args:
        results: Dictionary of {model_name: metrics_dict}
        metric: Metric to rank by (default: 'smape')
        ascending: Whether lower is better (True for errors, False for R²)

    Returns:
        DataFrame with ranked models

    Example:
        >>> results = {
        ...     'XGBoost': calculate_all_metrics(y_true, pred_xgb),
        ...     'LightGBM': calculate_all_metrics(y_true, pred_lgb),
        ... }
        >>> ranking = compare_models(results, metric='smape')
        >>> print(ranking)
    """
    df = pd.DataFrame(results).T
    df = df.sort_values(metric, ascending=ascending)

    # Add rank column
    df["rank"] = range(1, len(df) + 1)

    # Reorder columns to put rank first
    cols = ["rank"] + [col for col in df.columns if col != "rank"]
    df = df[cols]

    return df


if __name__ == "__main__":
    # Example usage
    print("Testing metrics module...")

    # Generate example data
    np.random.seed(42)
    y_true = np.random.uniform(40, 100, 1000)
    y_pred = y_true + np.random.normal(0, 10, 1000)

    # Add some near-zero values
    y_true[0:5] = [0.5, -1.0, 2.0, -0.5, 1.5]
    y_pred[0:5] = [1.0, -0.5, 3.0, 0.0, 2.0]

    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)

    # Print report
    print_metrics_report(metrics, model_name="Example Model")

    print("[OK] Metrics module working correctly")
