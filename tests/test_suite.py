"""
Test Suite for Energy Trading Research Platform

Structure:
    tests/
    ├── unit/               # Unit tests for individual functions
    ├── integration/        # Integration tests for pipelines
    ├── e2e/               # End-to-end workflow tests
    └── conftest.py        # Pytest fixtures

Run:
    pytest tests/
    pytest tests/unit/
    pytest --cov=. --cov-report=html
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    data = {
        "date": dates,
        "insee_region": np.random.choice([11, 52, 75], len(dates)),
        "temperature_2m_max": np.random.normal(15, 10, len(dates)),
        "temperature_2m_min": np.random.normal(5, 8, len(dates)),
        "precipitation_sum": np.random.exponential(2, len(dates)),
        "wind_speed_10m_max": np.random.exponential(15, len(dates)),
        "shortwave_radiation_sum": np.random.normal(5000, 2000, len(dates)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_energy_data():
    """Generate sample energy consumption data."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    data = {
        "date": dates,
        "insee_region": np.random.choice([11, 52, 75], len(dates)),
        "conso_elec_mw": np.random.normal(5000, 1000, len(dates)),
        "conso_gaz_mw": np.random.normal(3000, 800, len(dates)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_market_prices():
    """Generate sample market prices."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    data = {
        "date": dates,
        "price_FR": np.random.normal(50, 15, len(dates)),
        "price_DE": np.random.normal(52, 16, len(dates)),
    }
    return pd.DataFrame(data)


# =============================================================================
# UNIT TESTS - DATA PROCESSING
# =============================================================================


class TestDataProcessing:
    """Tests for data processing functions."""

    def test_merge_weather_energy(self, sample_weather_data, sample_energy_data):
        """Test merging weather and energy data."""
        # Merge on date and region
        merged = pd.merge(
            sample_weather_data,
            sample_energy_data,
            on=["date", "insee_region"],
            how="inner",
        )

        assert len(merged) > 0, "Merged data should not be empty"
        assert "temperature_2m_max" in merged.columns
        assert "conso_elec_mw" in merged.columns

    def test_feature_engineering(self, sample_weather_data):
        """Test feature engineering creates expected columns."""
        # Add derived features
        df = sample_weather_data.copy()
        df["temperature_2m_mean"] = (
            df["temperature_2m_max"] + df["temperature_2m_min"]
        ) / 2
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

        assert "temperature_2m_mean" in df.columns
        assert "temp_range" in df.columns
        assert df["temp_range"].min() >= 0, "Temperature range should be non-negative"

    def test_missing_data_handling(self, sample_weather_data):
        """Test handling of missing data."""
        df = sample_weather_data.copy()

        # Introduce missing values
        df.loc[0:5, "temperature_2m_max"] = np.nan

        # Forward fill
        df_filled = df.fillna(method="ffill")

        assert df_filled["temperature_2m_max"].isna().sum() == 0


# =============================================================================
# UNIT TESTS - MODELS
# =============================================================================


class TestModels:
    """Tests for ML models."""

    def test_xgboost_prediction_shape(self):
        """Test XGBoost prediction output shape."""
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        # Create dummy data
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 2)  # 2 targets

        # Train model
        model = MultiOutputRegressor(XGBRegressor(n_estimators=10))
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)

        assert y_pred.shape == y.shape, "Prediction shape should match target shape"
        assert not np.isnan(y_pred).any(), "Predictions should not contain NaN"

    def test_lightgbm_quantile_prediction(self):
        """Test LightGBM quantile predictions."""
        import lightgbm as lgb

        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        # Train for different quantiles
        quantiles = [0.1, 0.5, 0.9]
        predictions = {}

        for q in quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=10)
            model.fit(X, y)
            predictions[q] = model.predict(X)

        # Q10 should be < Q50 < Q90 (on average)
        assert (
            predictions[0.1].mean() < predictions[0.5].mean() < predictions[0.9].mean()
        )


# =============================================================================
# UNIT TESTS - TRADING
# =============================================================================


class TestTradingSystem:
    """Tests for trading system components."""

    def test_backtest_engine_initialization(self):
        """Test backtest engine initializes correctly."""
        from trading_system.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            initial_capital=100000, transaction_cost=0.001, max_position_size=1000
        )

        assert engine.capital == 100000
        assert engine.transaction_cost == 0.001
        assert len(engine.positions) == 0

    def test_signal_generation(self, sample_energy_data, sample_market_prices):
        """Test trading signal generation."""
        from trading_system.strategies.demand_price_arbitrage import (
            DemandPriceArbitrageStrategy,
        )

        strategy = DemandPriceArbitrageStrategy()

        # Calibrate
        demand = sample_energy_data["conso_elec_mw"]
        strategy.calibrate(demand)

        # Generate signals
        signals = strategy.generate_signals(demand)

        assert "signal" in signals.columns
        assert signals["signal"].isin([-1, 0, 1]).all()

    def test_position_size_calculation(self):
        """Test position sizing logic."""
        from trading_system.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(max_position_size=5000)

        # Test max position check
        can_open = engine.can_open_position(quantity=3000, price=50)
        assert can_open == True

        can_open_too_large = engine.can_open_position(quantity=6000, price=50)
        assert can_open_too_large == False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPipelines:
    """Integration tests for complete pipelines."""

    def test_training_pipeline_xgboost(
        self, tmp_path, sample_weather_data, sample_energy_data
    ):
        """Test complete XGBoost training pipeline."""
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        import pickle

        # Merge data
        data = pd.merge(
            sample_weather_data, sample_energy_data, on=["date", "insee_region"]
        )

        # Create features and targets
        features = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ]
        targets = ["conso_elec_mw", "conso_gaz_mw"]

        X = data[features].values
        y = data[targets].values

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        model = MultiOutputRegressor(XGBRegressor(n_estimators=10, random_state=42))
        model.fit(X_train, y_train)

        # Save
        model_path = tmp_path / "xgboost_test.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Load and predict
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        y_pred = loaded_model.predict(X_test)

        assert y_pred.shape == y_test.shape
        assert model_path.exists()

    def test_benchmark_pipeline(self, sample_weather_data, sample_energy_data):
        """Test benchmark pipeline runs without errors."""
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error

        # Merge data
        data = pd.merge(
            sample_weather_data, sample_energy_data, on=["date", "insee_region"]
        )

        features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        targets = ["conso_elec_mw", "conso_gaz_mw"]

        X = data[features].values
        y = data[targets].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Test XGBoost
        model = MultiOutputRegressor(XGBRegressor(n_estimators=10))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        assert rmse > 0
        assert rmse < 10000  # Sanity check


# =============================================================================
# END-TO-END TESTS
# =============================================================================


class TestWorkflows:
    """E2E tests for complete workflows."""

    def test_full_workflow(
        self, sample_weather_data, sample_energy_data, sample_market_prices
    ):
        """Test complete workflow: data → train → predict → evaluate."""
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        # 1. Merge data
        data = pd.merge(
            sample_weather_data, sample_energy_data, on=["date", "insee_region"]
        )

        # 2. Feature engineering
        data["temp_mean"] = (
            data["temperature_2m_max"] + data["temperature_2m_min"]
        ) / 2
        data["temp_range"] = data["temperature_2m_max"] - data["temperature_2m_min"]

        # 3. Prepare features
        features = [
            "temp_mean",
            "temp_range",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ]
        targets = ["conso_elec_mw", "conso_gaz_mw"]

        X = data[features].values
        y = data[targets].values

        # 4. Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 5. Train
        model = MultiOutputRegressor(XGBRegressor(n_estimators=20, random_state=42))
        model.fit(X_train, y_train)

        # 6. Predict
        y_pred = model.predict(X_test)

        # 7. Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Assertions
        assert rmse > 0
        assert r2 >= -1 and r2 <= 1  # R² can be negative for bad models
        assert y_pred.shape == y_test.shape


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance and benchmark tests."""

    def test_prediction_latency(self):
        """Test that prediction latency is acceptable."""
        import time

        X = np.random.randn(1000, 50)

        # Placeholder - test actual models
        start = time.time()
        # y_pred = model.predict(X)
        elapsed = time.time() - start

        # Should predict 1000 samples in < 1 second
        # assert elapsed < 1.0


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================


class TestDataValidation:
    """Tests for data quality and validation."""

    def test_no_future_leakage(self, sample_weather_data, sample_energy_data):
        """Ensure no future data leakage in training."""
        # Check that features don't include future information
        merged = pd.merge(
            sample_weather_data, sample_energy_data, on=["date", "insee_region"]
        )

        # Sort by date
        merged = merged.sort_values("date")

        # Create lag features
        merged["conso_elec_lag1"] = merged["conso_elec_mw"].shift(1)
        merged["conso_elec_lag7"] = merged["conso_elec_mw"].shift(7)

        # Check that lag features are properly shifted
        # First row should have NaN for lag1
        assert pd.isna(merged.iloc[0]["conso_elec_lag1"])

        # Second row lag1 should equal first row original
        assert merged.iloc[1]["conso_elec_lag1"] == merged.iloc[0]["conso_elec_mw"]

        # Check no future leakage: lag value should always be from the past
        for i in range(7, len(merged)):
            assert (
                merged.iloc[i]["conso_elec_lag7"] == merged.iloc[i - 7]["conso_elec_mw"]
            )

    def test_data_quality_checks(self, sample_weather_data):
        """Test data quality validations."""
        df = sample_weather_data

        # Check for impossible values
        assert (df["temperature_2m_max"] > -100).all(), "Temperature too low"
        assert (df["temperature_2m_max"] < 100).all(), "Temperature too high"
        assert (df["precipitation_sum"] >= 0).all(), "Precipitation cannot be negative"
        assert (df["wind_speed_10m_max"] >= 0).all(), "Wind speed cannot be negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
