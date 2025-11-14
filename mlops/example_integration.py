"""Example MLflow integration with energy forecasting models.

This script demonstrates how to integrate MLflow tracking into the existing
energy forecasting pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mlops import MLflowTracker


def generate_sample_data(n_samples=10000):
    """Generate sample energy demand data for testing.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        tuple: (X, y) feature matrix and target vector.
    """
    np.random.seed(42)

    # Simulate hourly energy data
    hours = np.arange(n_samples) % 24
    days = np.arange(n_samples) // 24
    temperature = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.randn(
        n_samples
    ) * 3
    load_lag1 = np.random.randn(n_samples) * 50 + 500
    load_lag24 = np.random.randn(n_samples) * 50 + 500

    # Create features
    X = pd.DataFrame(
        {
            "hour": hours,
            "day_of_week": days % 7,
            "temperature": temperature,
            "load_lag1": load_lag1,
            "load_lag24": load_lag24,
        }
    )

    # Generate target (energy load)
    y = (
        300
        + 50 * np.sin(2 * np.pi * hours / 24)  # Daily pattern
        + 100 * (days % 7 < 5)  # Weekday effect
        - 5 * temperature  # Temperature effect
        + 0.3 * load_lag1
        + 0.2 * load_lag24
        + np.random.randn(n_samples) * 30
    )

    return X, y


def train_baseline_model():
    """Train baseline Random Forest model with MLflow tracking."""
    print("=" * 70)
    print("EXAMPLE 1: Baseline Model Training with MLflow")
    print("=" * 70)

    # Generate data
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MLflow tracking
    with MLflowTracker("load_forecasting", run_name="rf-baseline") as tracker:
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        }
        tracker.log_params(params)

        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Log metrics using built-in method
        print("\nLogging training metrics...")
        train_metrics = tracker.log_forecast_metrics(
            y_train, y_train_pred, prefix="train_"
        )

        print("\nLogging test metrics...")
        test_metrics = tracker.log_forecast_metrics(
            y_test, y_test_pred, prefix="test_"
        )

        # Log model
        print("\nLogging model artifact...")
        tracker.log_model(model, "model", registered_model_name="energy_load_rf")

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Train RMSE: {train_metrics['train_rmse']:.2f}")
        print(f"Train R²:   {train_metrics['train_r2']:.4f}")
        print(f"Test RMSE:  {test_metrics['test_rmse']:.2f}")
        print(f"Test R²:    {test_metrics['test_r2']:.4f}")
        print(f"Test MAPE:  {test_metrics['test_mape']:.2f}%")
        print(f"Test DA:    {test_metrics['test_directional_accuracy']:.2f}%")
        print("=" * 70)


def compare_multiple_models():
    """Compare multiple model configurations with MLflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Model Comparison with MLflow")
    print("=" * 70)

    # Generate data
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Different configurations to test
    configs = [
        {"n_estimators": 50, "max_depth": 5, "name": "rf-shallow"},
        {"n_estimators": 100, "max_depth": 10, "name": "rf-medium"},
        {"n_estimators": 200, "max_depth": 15, "name": "rf-deep"},
    ]

    results = []

    for config in configs:
        name = config.pop("name")
        print(f"\nTraining {name}...")

        with MLflowTracker("model_comparison", run_name=name) as tracker:
            # Log parameters
            tracker.log_params(config)

            # Train
            model = RandomForestRegressor(**config, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            # Log metrics
            metrics = tracker.log_forecast_metrics(y_test, y_pred, prefix="test_")

            results.append({"model": name, **metrics})

    # Display comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    comparison_df = pd.DataFrame(results)
    print(
        comparison_df[["model", "test_rmse", "test_r2", "test_mape"]].to_string(
            index=False
        )
    )
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n🚀 MLflow Integration Examples")
    print("\nThese examples demonstrate professional experiment tracking.")
    print("After running, view results with: mlflow ui\n")

    # Example 1: Basic training with tracking
    train_baseline_model()

    # Example 2: Model comparison
    compare_multiple_models()

    print("\n✅ Examples completed!")
    print("\nTo view results in MLflow UI:")
    print("  mlflow ui --backend-store-uri file:///home/user/energy-demand-forecast/mlruns")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()
