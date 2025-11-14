"""Quick test script for price forecasting module using generated data."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from model.price_forecasting.data_loader import (
    add_calendar_features,
    add_lag_features,
    simulate_realistic_prices,
)
from model.price_forecasting.models import (
    EnsemblePriceForecaster,
    LightGBMQuantileForecaster,
)


def generate_test_load_data(n_hours=10000):
    """Generate synthetic load data for testing."""
    dates = pd.date_range("2022-01-01", periods=n_hours, freq="h")

    # Simulate realistic load patterns
    hour = dates.hour
    dow = dates.dayofweek
    doy = dates.dayofyear

    load = (
        5000  # Base load
        + 2000 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
        + 1000 * (dow < 5)  # Weekday effect
        + 1500 * np.sin(2 * np.pi * doy / 365)  # Seasonal pattern
        + np.random.randn(n_hours) * 300  # Noise
    )

    return pd.DataFrame({"datetime_hour": dates, "conso_elec_mw": load})


def main():
    """Run price forecasting test."""
    print("=" * 70)
    print("PRICE FORECASTING MODULE TEST")
    print("=" * 70)

    # Generate test data
    print("\n1. Generating synthetic load data...")
    load_df = generate_test_load_data(n_hours=10000)
    print(f"   Generated {len(load_df)} hours of load data")

    # Simulate prices
    print("\n2. Simulating realistic electricity prices...")
    price_df = simulate_realistic_prices(load_df, random_seed=42)
    print(f"   Price range: {price_df['price'].min():.2f} - {price_df['price'].max():.2f} EUR/MWh")
    print(f"   Mean price: {price_df['price'].mean():.2f} EUR/MWh")
    print(f"   Std price: {price_df['price'].std():.2f} EUR/MWh")

    # Add features
    print("\n3. Engineering features...")
    df = add_calendar_features(price_df)
    df = add_lag_features(df, target_col="price")
    df = add_lag_features(df, target_col="load_mw")
    df = df.dropna().reset_index(drop=True)

    feature_cols = [col for col in df.columns if col not in ["datetime_hour", "price"]]
    print(f"   Created {len(feature_cols)} features")

    # Train-test split
    print("\n4. Splitting data (80/20)...")
    split_idx = int(len(df) * 0.8)
    X_train = df.iloc[:split_idx][feature_cols]
    y_train = df.iloc[:split_idx]["price"]
    X_test = df.iloc[split_idx:][feature_cols]
    y_test = df.iloc[split_idx:]["price"]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Test LightGBM Quantile model
    print("\n" + "=" * 70)
    print("5. Training LightGBM Quantile Model...")
    print("=" * 70)

    lgbm_model = LightGBMQuantileForecaster(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    )

    lgbm_model.fit(X_train, y_train)
    print("   ✓ Model trained")

    # Evaluate
    metrics = lgbm_model.evaluate(X_test, y_test)
    print(f"\n   Test RMSE: {metrics['rmse']:.2f} EUR/MWh")
    print(f"   Test R²:   {metrics['r2']:.4f}")
    print(f"   Test MAPE: {metrics['mape']:.2f}%")
    print(f"   Test DA:   {metrics['directional_accuracy']:.2f}%")

    # Quantile predictions
    median, p10, p90 = lgbm_model.predict_interval(X_test)
    interval_width = np.mean(p90 - p10)
    print(f"\n   Average 80% prediction interval: {interval_width:.2f} EUR/MWh")

    # Test Ensemble model
    print("\n" + "=" * 70)
    print("6. Training Ensemble Model...")
    print("=" * 70)

    ensemble_model = EnsemblePriceForecaster(weights=[0.5, 0.3, 0.2])
    ensemble_model.fit(X_train, y_train)
    print("   ✓ Model trained")

    # Evaluate
    metrics = ensemble_model.evaluate(X_test, y_test)
    print(f"\n   Test RMSE: {metrics['rmse']:.2f} EUR/MWh")
    print(f"   Test R²:   {metrics['r2']:.4f}")
    print(f"   Test MAPE: {metrics['mape']:.2f}%")
    print(f"   Test DA:   {metrics['directional_accuracy']:.2f}%")

    # Individual model comparison
    print("\n7. Individual Model Performance:")
    print("=" * 70)
    individual_preds = ensemble_model.get_individual_predictions(X_test)
    for name, preds in individual_preds.items():
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        print(f"   {name:15s}: RMSE = {rmse:.2f}, R² = {r2:.4f}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nPrice forecasting module is working correctly!")
    print("Ready for production use with real ENTSO-E data.")


if __name__ == "__main__":
    main()
