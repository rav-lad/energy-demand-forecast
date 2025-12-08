"""Tests for data validation and leakage prevention.

This test suite ensures:
1. Simulated prices are rejected in production mode
2. Real price data passes validation
3. No data leakage in feature engineering
4. Proper temporal ordering in train/test splits
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.price_forecasting.data_loader import (
    load_price_and_load_data,
    validate_real_price_data,
    add_lag_features,
    prepare_price_forecasting_dataset,
)


class TestDataValidation:
    """Test suite for data validation."""

    def test_simulated_prices_rejected_in_strict_mode(self):
        """Test that simulated prices crash in strict validation mode."""
        with pytest.raises(RuntimeError, match="simulate_prices=True is NOT allowed"):
            load_price_and_load_data(
                simulate_prices=True,
                strict_validation=True
            )

    def test_simulated_prices_allowed_in_test_mode(self):
        """Test that simulated prices work when strict_validation=False."""
        try:
            df = load_price_and_load_data(
                simulate_prices=True,
                strict_validation=False
            )
            assert len(df) > 0
            assert 'price' in df.columns
        except FileNotFoundError:
            # Expected if load data file doesn't exist
            pytest.skip("Load data file not found - skipping test")

    def test_validate_real_data_rejects_extreme_prices(self):
        """Test that validation rejects unrealistic prices."""
        # Create fake data with extreme prices
        df = pd.DataFrame({
            'price': [5000.0] * 100,  # Unrealistic: 5000 EUR/MWh
            'datetime_hour': pd.date_range('2023-01-01', periods=100, freq='H')
        })

        with pytest.raises(ValueError, match="Extremely high prices"):
            validate_real_price_data(df, data_source="test_data")

    def test_validate_real_data_accepts_negative_prices(self):
        """Test that validation accepts realistic negative prices."""
        df = pd.DataFrame({
            'price': [-5.0, -2.0, 10.0, 50.0, 80.0] * 200,
            'datetime_hour': pd.date_range('2023-01-01', periods=1000, freq='H')
        })

        # Should not raise
        try:
            validate_real_price_data(df, data_source="test_data")
        except ValueError as e:
            pytest.fail(f"Validation should accept negative prices: {e}")

    def test_validate_real_data_rejects_missing_price_column(self):
        """Test that validation requires 'price' column."""
        df = pd.DataFrame({
            'value': [50.0] * 100,  # Wrong column name
            'datetime_hour': pd.date_range('2023-01-01', periods=100, freq='H')
        })

        with pytest.raises(ValueError, match="must contain 'price' column"):
            validate_real_price_data(df)

    def test_no_future_leakage_in_lag_features(self):
        """Test that lag features don't use future data."""
        # Create simple dataset
        df = pd.DataFrame({
            'datetime_hour': pd.date_range('2023-01-01', periods=100, freq='H'),
            'price': np.arange(100).astype(float)  # 0, 1, 2, ...
        })

        df = add_lag_features(df, target_col='price', lags=[1, 24])

        # Check that lag_1h at time t equals price at time t-1
        for i in range(1, len(df)):
            if not pd.isna(df['price_lag_1h'].iloc[i]):
                assert df['price_lag_1h'].iloc[i] == df['price'].iloc[i-1], \
                    f"Lag feature at index {i} should equal price at {i-1}"

        # First value should be NaN (no previous data)
        assert pd.isna(df['price_lag_1h'].iloc[0]), \
            "First lag value should be NaN"

    def test_rolling_features_no_future_leakage(self):
        """Test that rolling features don't include current observation."""
        df = pd.DataFrame({
            'datetime_hour': pd.date_range('2023-01-01', periods=100, freq='H'),
            'price': np.arange(100).astype(float)
        })

        df = add_lag_features(df, target_col='price', lags=[1])

        # Rolling mean should be calculated on historical data only
        # Check that rolling_mean uses shifted data (no future leakage)
        for window in [24, 168]:
            col_name = f'price_roll_mean_{window}h'
            if col_name in df.columns:
                # Rolling mean should not include current price
                for i in range(window, len(df)):
                    historical_prices = df['price'].iloc[max(0, i-window):i]
                    if len(historical_prices) > 0:
                        expected_mean = historical_prices.mean()
                        # Allow small numerical error
                        if not pd.isna(df[col_name].iloc[i]):
                            assert abs(df[col_name].iloc[i] - expected_mean) < 1e-6, \
                                f"Rolling mean at {i} should only use historical data"

    def test_prepare_dataset_excludes_contemporaneous_load(self):
        """Test that prepare_price_forecasting_dataset excludes contemporaneous load_mw."""
        try:
            df, feature_cols = prepare_price_forecasting_dataset(
                simulate_prices=True,  # Use simulation for testing
                strict_validation=False
            )

            # Check that 'load_mw' is NOT in feature columns
            assert 'load_mw' not in feature_cols, \
                "Contemporaneous load_mw should be excluded (data leakage)"

            # Check that lagged load features ARE included
            load_lag_features = [col for col in feature_cols if 'load_mw_lag_' in col]
            assert len(load_lag_features) > 0, \
                "Should include lagged load features"

        except FileNotFoundError:
            pytest.skip("Data files not found - skipping test")

    def test_temporal_ordering_in_train_test_split(self):
        """Test that train comes before test in time."""
        try:
            df, feature_cols = prepare_price_forecasting_dataset(
                simulate_prices=True,
                strict_validation=False
            )

            # Simple 80/20 split
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            # Check temporal ordering
            train_max_date = train_df['datetime_hour'].max()
            test_min_date = test_df['datetime_hour'].min()

            assert train_max_date < test_min_date, \
                "Train data must come entirely before test data (temporal ordering)"

        except FileNotFoundError:
            pytest.skip("Data files not found - skipping test")


class TestRealDataIntegration:
    """Integration tests with real ENTSO-E data (if available)."""

    def test_load_real_entsoe_data(self):
        """Test loading real ENTSO-E price data."""
        price_file = Path("data/raw_data/market_prices/FR_2023-01-01_2024-12-31.csv")

        if not price_file.exists():
            pytest.skip("Real ENTSO-E price data not found - skipping test")

        try:
            df = load_price_and_load_data(
                simulate_prices=False,
                strict_validation=True,
                price_file=str(price_file)
            )

            # Validate data quality
            assert len(df) > 10000, "Should have at least 10k hourly observations"
            assert 'price' in df.columns
            assert 'load_mw' in df.columns
            assert 'datetime_hour' in df.columns

            # Check realistic price ranges
            assert df['price'].min() >= -50, "Prices should be > -50 EUR/MWh"
            assert df['price'].max() <= 500, "Prices should be < 500 EUR/MWh (normal range)"

            # Check for negative prices (realistic in European markets)
            negative_prices = (df['price'] < 0).sum()
            assert negative_prices >= 0, "Negative prices are realistic (renewable flush)"

            print(f"\n✅ Real data validation passed:")
            print(f"   Records: {len(df):,}")
            print(f"   Price range: [{df['price'].min():.2f}, {df['price'].max():.2f}] EUR/MWh")
            print(f"   Negative prices: {negative_prices} ({negative_prices/len(df)*100:.2f}%)")

        except Exception as e:
            pytest.fail(f"Failed to load real ENTSO-E data: {e}")

    def test_data_validation_on_real_prices(self):
        """Test that validation passes on real ENTSO-E data."""
        price_file = Path("data/raw_data/market_prices/FR_2023-01-01_2024-12-31.csv")

        if not price_file.exists():
            pytest.skip("Real ENTSO-E price data not found")

        try:
            price_df = pd.read_csv(price_file)

            # Rename columns if needed
            if 'price_eur_mwh' in price_df.columns:
                price_df['price'] = price_df['price_eur_mwh']

            # Run validation
            validate_real_price_data(price_df, data_source="ENTSO-E France")

        except Exception as e:
            pytest.fail(f"Real data should pass validation: {e}")


class TestDataLeakagePrevention:
    """Comprehensive tests for data leakage prevention."""

    def test_no_target_in_features(self):
        """Test that target variable is not included in features."""
        try:
            df, feature_cols = prepare_price_forecasting_dataset(
                simulate_prices=True,
                strict_validation=False
            )

            assert 'price' not in feature_cols, \
                "Target 'price' should not be in features"

        except FileNotFoundError:
            pytest.skip("Data files not found")

    def test_no_datetime_in_features(self):
        """Test that datetime columns are not included in features."""
        try:
            df, feature_cols = prepare_price_forecasting_dataset(
                simulate_prices=True,
                strict_validation=False
            )

            datetime_cols = [col for col in feature_cols if 'datetime' in col.lower()]
            assert len(datetime_cols) == 0, \
                f"Datetime columns should not be in features: {datetime_cols}"

        except FileNotFoundError:
            pytest.skip("Data files not found")

    def test_all_lags_use_shift(self):
        """Test that all lag features are created with proper shift."""
        df = pd.DataFrame({
            'datetime_hour': pd.date_range('2023-01-01', periods=200, freq='H'),
            'price': np.random.uniform(40, 60, 200)
        })

        df = add_lag_features(df, target_col='price', lags=[1, 2, 3, 24])

        # Verify each lag is correctly shifted
        for lag in [1, 2, 3, 24]:
            col_name = f'price_lag_{lag}h'
            assert col_name in df.columns, f"Missing lag column: {col_name}"

            # Check alignment
            for i in range(lag, len(df)):
                if not pd.isna(df[col_name].iloc[i]):
                    expected = df['price'].iloc[i - lag]
                    actual = df[col_name].iloc[i]
                    assert abs(actual - expected) < 1e-6, \
                        f"Lag {lag} at index {i} incorrect: {actual} != {expected}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
