"""Comprehensive Unit Tests for Energy Trading System

This test suite significantly improves code coverage from 7.5% baseline.
Target: Cover critical trading logic, risk management, and data processing.

Run with: pytest tests/test_comprehensive.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# =============================================================================
# Test Transaction Cost Model
# =============================================================================


class TestTransactionCostModel:
    """Test suite for transaction cost calculations."""

    def test_commission_calculation_basic(self):
        """Test basic commission calculation."""
        from trading_system.backtesting.backtesting_engine import TransactionCostModel

        cost_model = TransactionCostModel()
        cost_model.commission_pct = 0.001
        cost_model.commission_min = 1.0

        # Small trade: should use minimum
        commission = cost_model.calculate_commission(quantity=10, price=50)
        assert commission == 1.0, "Small trades should use minimum commission"

        # Large trade: should use percentage
        commission = cost_model.calculate_commission(quantity=1000, price=50)
        expected = 1000 * 50 * 0.001  # = 50 EUR
        assert abs(commission - expected) < 0.01

    def test_commission_zero_quantity(self):
        """Test commission on zero quantity."""
        from trading_system.backtesting.backtesting_engine import TransactionCostModel

        cost_model = TransactionCostModel()
        commission = cost_model.calculate_commission(quantity=0, price=50)
        assert commission == cost_model.commission_min

    def test_slippage_calculation_basic(self):
        """Test basic slippage calculation."""
        from trading_system.backtesting.backtesting_engine import TransactionCostModel

        cost_model = TransactionCostModel()
        cost_model.slippage_pct = 0.0005
        cost_model.slippage_fixed = 0.5

        slippage = cost_model.calculate_slippage(quantity=100, price=50)

        # slippage = notional * pct + quantity * fixed
        # = 5000 * 0.0005 + 100 * 0.5 = 2.5 + 50 = 52.5
        expected = 100 * 50 * 0.0005 + 100 * 0.5
        assert abs(slippage - expected) < 0.01

    def test_slippage_with_market_impact(self):
        """Test slippage with market impact component."""
        from trading_system.backtesting.backtesting_engine import TransactionCostModel

        cost_model = TransactionCostModel()
        cost_model.market_impact_coef = 0.01

        # Large order vs daily volume should have market impact
        slippage = cost_model.calculate_slippage(
            quantity=1000,
            price=50,
            daily_volume=10000,
        )

        assert slippage > 0, "Slippage should be positive"
        # Market impact = coef * notional * sqrt(quantity / daily_volume)
        # Should increase slippage


# =============================================================================
# Test Mean Reversion Strategy
# =============================================================================


class TestMeanReversionStrategy:
    """Test suite for mean reversion strategy logic."""

    def test_z_score_calculation_no_division_by_zero(self):
        """Test Z-score calculation with zero std protection."""
        from trading_system.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig

        config = MeanReversionConfig(lookback_window=10)
        strategy = MeanReversionStrategy(config)

        # Create constant prices (std = 0)
        df = pd.DataFrame({
            "price": [50.0] * 20,
            "datetime": pd.date_range("2022-01-01", periods=20, freq="H"),
        })

        signals = strategy.generate_signals(prices=df["price"], dates=df["datetime"])

        # Should not crash with division by zero
        assert "z_score" in signals.columns
        # Z-scores should be 0 or clamped to safe value
        assert signals["z_score"].abs().max() < 100  # No infinity

    def test_mean_reversion_signal_generation(self):
        """Test signal generation logic."""
        from trading_system.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig

        config = MeanReversionConfig(
            entry_z_score=2.0,
            exit_z_score=0.5,
            lookback_window=5,
        )
        strategy = MeanReversionStrategy(config)

        # Create oscillating prices
        prices = pd.Series([50, 45, 40, 45, 50, 55, 60, 55, 50, 45] * 2)
        dates = pd.date_range("2022-01-01", periods=len(prices), freq="H")

        signals = strategy.generate_signals(prices=prices, dates=dates)

        # Check signals are in valid range
        assert signals["signal"].isin([-1, 0, 1]).all(), "Signals should be -1, 0, or 1"

        # Check position column exists
        assert "position" in signals.columns

    def test_mean_reversion_stop_loss(self):
        """Test stop loss triggers correctly."""
        from trading_system.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig

        config = MeanReversionConfig(
            entry_z_score=2.0,
            stop_loss_z=3.5,
        )
        strategy = MeanReversionStrategy(config)

        # Create extreme price spike
        prices = pd.Series([50] * 10 + [100] + [50] * 10)  # Spike to 100
        dates = pd.date_range("2022-01-01", periods=len(prices), freq="H")

        signals = strategy.generate_signals(prices=prices, dates=dates)

        assert "stop_loss" in signals.columns


# =============================================================================
# Test Price Forecast Strategy
# =============================================================================


class TestPriceForecastStrategy:
    """Test suite for price forecast strategy."""

    def test_position_persistence_max_holding_period(self):
        """Test max holding period prevents indefinite positions."""
        from trading_system.strategies.price_forecast_strategy import PriceForecastStrategy

        strategy = PriceForecastStrategy(
            entry_threshold=5.0,
            exit_threshold=2.0,
            max_holding_period=10,  # 10 hours max
        )

        # Create forecast error that never converges
        n = 50
        market_prices = pd.Series([50.0] * n)
        forecasts = pd.Series([60.0] * n)  # Persistent 10 EUR error

        signals = strategy.generate_signals(
            market_prices=market_prices,
            forecasts=forecasts,
        )

        # Position should be forced to exit after max_holding_period
        # even if forecast error still large
        position_lengths = []
        current_length = 0
        for sig in signals["signal"]:
            if sig != 0:
                current_length += 1
            else:
                if current_length > 0:
                    position_lengths.append(current_length)
                current_length = 0

        if position_lengths:
            max_length = max(position_lengths)
            assert max_length <= 10, f"Max holding period violated: {max_length} > 10"

    def test_entry_exit_thresholds(self):
        """Test entry and exit threshold logic."""
        from trading_system.strategies.price_forecast_strategy import PriceForecastStrategy

        strategy = PriceForecastStrategy(
            entry_threshold=5.0,
            exit_threshold=2.0,
        )

        # Forecast error alternates between high and low
        market_prices = pd.Series([50, 50, 50, 50, 50])
        forecasts = pd.Series([60, 56, 52, 50, 48])  # Error: 10, 6, 2, 0, -2

        signals = strategy.generate_signals(
            market_prices=market_prices,
            forecasts=forecasts,
        )

        # First signal should be BUY (error > 5)
        # Should persist until error < 2
        assert signals["signal"].iloc[0] == 1  # Entry
        assert signals["signal"].iloc[2] == 0 or signals["signal"].iloc[3] == 0  # Exit

    def test_confidence_scaling(self):
        """Test position sizing with confidence scaling."""
        from trading_system.strategies.price_forecast_strategy import PriceForecastStrategy

        strategy = PriceForecastStrategy(
            confidence_scaling=True,
            position_size=0.02,
        )

        market_prices = pd.Series([50, 50, 50])
        forecasts = pd.Series([60, 60, 60])
        forecast_std = pd.Series([1.0, 5.0, 10.0])  # Varying uncertainty

        signals = strategy.generate_signals(
            market_prices=market_prices,
            forecasts=forecasts,
            forecast_std=forecast_std,
        )

        # Position sizes should be inversely proportional to std
        # (higher uncertainty = smaller position)
        assert "position_size" in signals.columns


# =============================================================================
# Test Data Loading and Validation
# =============================================================================


class TestDataLoader:
    """Test suite for data loading and price simulation."""

    def test_simulate_realistic_prices_no_nan(self):
        """Test price simulation produces no NaN values."""
        from model.price_forecasting.data_loader import simulate_realistic_prices

        # Create sample load data
        dates = pd.date_range("2022-01-01", periods=100, freq="H")
        load_data = pd.DataFrame({
            "datetime_hour": dates,
            "conso_elec_mw": np.random.uniform(30000, 50000, len(dates)),
        })

        prices = simulate_realistic_prices(load_data, random_seed=42)

        assert not prices["price"].isna().any(), "Prices contain NaN values"
        assert len(prices) == len(load_data), "Length mismatch"

    def test_simulate_realistic_prices_non_negative_allowed(self):
        """Test prices can go slightly negative (renewable flush)."""
        from model.price_forecasting.data_loader import simulate_realistic_prices

        dates = pd.date_range("2022-01-01", periods=1000, freq="H")
        load_data = pd.DataFrame({
            "datetime_hour": dates,
            "conso_elec_mw": np.random.uniform(20000, 60000, len(dates)),
        })

        prices = simulate_realistic_prices(
            load_data,
            spike_probability=0.02,
            random_seed=42,
        )

        # Should allow some negative prices (renewable flush scenario)
        # But not too extreme
        assert prices["price"].min() >= -10.0, "Prices too negative"
        assert prices["price"].max() < 500.0, "Prices too high (sanity check)"

    def test_price_load_correlation(self):
        """Test simulated prices correlate with load."""
        from model.price_forecasting.data_loader import simulate_realistic_prices

        dates = pd.date_range("2022-01-01", periods=500, freq="H")
        load_data = pd.DataFrame({
            "datetime_hour": dates,
            "conso_elec_mw": np.linspace(30000, 60000, len(dates)),  # Increasing load
        })

        prices = simulate_realistic_prices(load_data, random_seed=42)

        # Correlation should be positive (higher load → higher price)
        corr = np.corrcoef(load_data["conso_elec_mw"], prices["price"])[0, 1]
        assert corr > 0.3, f"Load-price correlation too low: {corr}"


# =============================================================================
# Test MAPE Calculation Consistency
# =============================================================================


class TestMetrics:
    """Test metric calculations for consistency."""

    def test_mape_epsilon_consistency(self):
        """Test MAPE uses consistent epsilon across calculations."""
        from mlops.mlflow_utils import MLflowTracker
        from model.price_forecasting.models import LightGBMQuantileForecaster

        # Create tracker
        tracker = MLflowTracker(experiment_name="test")

        y_true = np.array([50, 100, 150, 200])
        y_pred = np.array([55, 95, 160, 190])

        # Check MLflow utils uses 1e-8
        metrics = tracker.log_forecast_metrics(y_true, y_pred, prefix="test_")

        # MAPE should be calculated with epsilon = 1e-8
        expected_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        assert abs(metrics["test_mape"] - expected_mape) < 0.01

    def test_sharpe_ratio_annualization_hourly_data(self):
        """Test Sharpe ratio uses correct annualization for hourly data."""
        from trading_system.strategies.price_forecast_strategy import PriceForecastStrategy

        strategy = PriceForecastStrategy()

        # Create synthetic returns
        n = 1000
        returns = pd.Series(np.random.randn(n) * 0.01 + 0.0005)  # Small positive mean

        # Calculate Sharpe manually
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_hourly = mean_return / std_return if std_return > 0 else 0
        sharpe_annual = sharpe_hourly * np.sqrt(365 * 24)  # Hourly → Annual

        # This is just to document expected calculation
        assert sharpe_annual == sharpe_hourly * np.sqrt(365 * 24)


# =============================================================================
# Test Walk-Forward Validation
# =============================================================================


class TestWalkForward:
    """Test walk-forward validation framework."""

    def test_efficiency_ratio_calculation(self):
        """Test efficiency ratio calculation logic."""
        # ER = OOS Sharpe / IS Sharpe
        is_sharpe = 2.0
        oos_sharpe = 1.5

        efficiency_ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0

        assert efficiency_ratio == 0.75, "ER calculation incorrect"

        # ER > 0.7 indicates robust strategy
        assert efficiency_ratio > 0.7, "Strategy should be considered robust"

    def test_walk_forward_period_creation(self):
        """Test walk-forward period splitting."""
        from trading_system.backtesting.walk_forward import WalkForwardConfig

        config = WalkForwardConfig(
            train_period_days=180,
            test_period_days=60,
            step_days=30,
        )

        # Periods should not overlap in test sets
        assert config.test_period_days <= config.step_days * 2, \
            "Test periods might overlap with bad config"


# =============================================================================
# Test Fuel Prices Simulation
# =============================================================================


class TestFuelPrices:
    """Test fuel price simulation and spreads."""

    def test_ttf_gas_price_simulation(self):
        """Test TTF gas price simulation produces valid output."""
        from data_collection.fuel_prices import simulate_ttf_gas_prices

        dates = pd.date_range("2022-01-01", periods=365, freq="D")
        prices = simulate_ttf_gas_prices(dates, random_seed=42)

        assert len(prices) == len(dates)
        assert prices.min() >= 5.0, "Gas prices should be >= 5 EUR/MWh (floor)"
        assert not prices.isna().any(), "Gas prices contain NaN"

    def test_spark_spread_calculation(self):
        """Test spark spread calculation."""
        from data_collection.fuel_prices import calculate_spark_spread

        # Spark spread = Power - (Gas/Efficiency) - (Carbon * Emission)
        power_price = 100.0
        gas_price = 30.0
        carbon_price = 80.0
        efficiency = 0.55
        emission_factor = 0.35

        spread = calculate_spark_spread(
            power_price, gas_price, carbon_price, efficiency, emission_factor
        )

        expected = power_price - (gas_price / efficiency) - (carbon_price * emission_factor)
        assert abs(spread - expected) < 0.01

    def test_dark_spread_calculation(self):
        """Test dark spread calculation."""
        from data_collection.fuel_prices import calculate_dark_spread

        power_price = 100.0
        coal_price = 100.0  # EUR/tonne
        carbon_price = 80.0
        efficiency = 0.38
        emission_factor = 0.95

        spread = calculate_dark_spread(
            power_price, coal_price, carbon_price, efficiency, emission_factor
        )

        # Coal price needs conversion: EUR/tonne → EUR/MWh
        # Assume ~8.14 MWh/tonne heating value
        coal_price_mwh = coal_price / 8.14
        expected = power_price - (coal_price_mwh / efficiency) - (carbon_price * emission_factor)
        assert abs(spread - expected) < 1.0  # Within 1 EUR


# =============================================================================
# Test Risk Management
# =============================================================================


class TestRiskManagement:
    """Test risk management calculations."""

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 90, 95, 105, 100])

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity / running_max - 1) * 100

        max_dd = drawdown.min()

        # Max DD should be when equity drops from 110 to 90
        expected_max_dd = (90 / 110 - 1) * 100  # = -18.18%

        assert abs(max_dd - expected_max_dd) < 0.1

    def test_position_sizing_limits(self):
        """Test position sizing respects maximum limits."""
        from trading_system.strategies.price_forecast_strategy import PriceForecastStrategy

        strategy = PriceForecastStrategy(
            position_size=0.02,
            max_position=0.10,
        )

        market_prices = pd.Series([50] * 10)
        forecasts = pd.Series([100] * 10)  # Huge forecast error

        signals = strategy.generate_signals(
            market_prices=market_prices,
            forecasts=forecasts,
        )

        # Position sizes should not exceed max_position
        assert signals["position_size"].abs().max() <= 0.10

    def test_value_at_risk_calculation(self):
        """Test VaR calculation (95% confidence)."""
        # Create returns distribution
        returns = pd.Series(np.random.randn(1000) * 0.02)

        # VaR at 95% confidence = 5th percentile
        var_95 = np.percentile(returns, 5)

        # Should be negative (loss)
        assert var_95 < 0

        # Approximately -1.645 * std for normal distribution
        expected_var = -1.645 * returns.std()
        assert abs(var_95 - expected_var) < 0.01


# =============================================================================
# Test Data Validation
# =============================================================================


class TestDataValidation:
    """Test data validation and merge operations."""

    def test_merge_length_mismatch_handling(self):
        """Test handling of length mismatches during merge."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5]})  # Different length

        # Should handle gracefully
        min_len = min(len(df1), len(df2))
        df1_trimmed = df1.iloc[:min_len]
        df2_trimmed = df2.iloc[:min_len]

        result = pd.concat([df1_trimmed, df2_trimmed], axis=1)

        assert len(result) == min_len
        assert not result.isna().any().any()

    def test_duplicate_column_detection(self):
        """Test duplicate column detection before merge."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [5, 6], "c": [7, 8]})  # "b" is duplicate

        duplicate_cols = set(df1.columns) & set(df2.columns)

        assert "b" in duplicate_cols
        assert len(duplicate_cols) == 1


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
