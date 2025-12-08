"""
Comprehensive tests for production pipeline.

Tests cover:
1. Data validation
2. Feature engineering
3. Model training
4. Ensemble predictions
5. Trading strategy execution
6. End-to-end integration
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.config import UNIFIED_DATASET_PATH, FEATURE_FILE
from production.ensemble_model import EnsembleForecaster, train_ensemble_walk_forward
from production.trading_strategy import PeakShavingStrategy, backtest_strategy


class TestDataValidation:
    """Test data loading and validation."""

    def test_unified_dataset_exists(self):
        """Test that unified dataset file exists."""
        assert UNIFIED_DATASET_PATH.exists(), f"Dataset not found: {UNIFIED_DATASET_PATH}"

    def test_unified_dataset_structure(self):
        """Test that dataset has correct structure."""
        df = pd.read_csv(UNIFIED_DATASET_PATH)

        # Required columns
        required_cols = ['datetime_hour', 'price']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data types
        df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])
        assert df['datetime_hour'].dtype == 'datetime64[ns]', "datetime_hour must be datetime"
        assert df['price'].dtype in [np.float64, np.int64], "price must be numeric"

        # Check no missing values in critical columns
        assert df['datetime_hour'].notna().all(), "datetime_hour has missing values"
        assert df['price'].notna().all(), "price has missing values"

        # Check reasonable price range (EUR/MWh)
        assert df['price'].min() >= -200, "Price too low (< -200 EUR/MWh)"
        assert df['price'].max() <= 500, "Price too high (> 500 EUR/MWh)"

    def test_feature_list_exists(self):
        """Test that feature list file exists."""
        assert FEATURE_FILE.exists(), f"Feature list not found: {FEATURE_FILE}"

    def test_features_in_dataset(self):
        """Test that all features exist in dataset."""
        df = pd.read_csv(UNIFIED_DATASET_PATH)

        with open(FEATURE_FILE, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]

        missing_features = [f for f in feature_cols if f not in df.columns]
        assert len(missing_features) == 0, f"Missing features in dataset: {missing_features}"


class TestEnsembleModel:
    """Test ensemble model functionality."""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        np.random.seed(42)
        X_train = np.random.randn(1000, 10)
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(1000) * 0.1

        X_test = np.random.randn(100, 10)
        y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + np.random.randn(100) * 0.1

        return X_train, y_train, X_test, y_test

    def test_ensemble_creation(self):
        """Test that ensemble can be created."""
        ensemble = EnsembleForecaster(models=["ridge", "xgboost"])
        assert ensemble.models == ["ridge", "xgboost"]
        assert ensemble.weights == {"ridge": 0.3, "xgboost": 0.7}

    def test_ensemble_weights_validation(self):
        """Test that ensemble validates weights."""
        # Weights don't sum to 1
        with pytest.raises(ValueError):
            EnsembleForecaster(models=["ridge"], weights={"ridge": 0.5})

        # Mismatched keys
        with pytest.raises(ValueError):
            EnsembleForecaster(models=["ridge"], weights={"xgboost": 1.0})

    def test_ensemble_fit_predict(self, dummy_data):
        """Test that ensemble can fit and predict."""
        X_train, y_train, X_test, y_test = dummy_data

        ensemble = EnsembleForecaster(models=["ridge", "xgboost"])
        ensemble.fit(X_train, y_train)

        # Check that models are trained
        assert "ridge" in ensemble.trained_models
        assert "xgboost" in ensemble.trained_models

        # Check predictions
        y_pred = ensemble.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert y_pred.dtype in [np.float64, np.float32]

        # Check MAE is reasonable
        mae = np.mean(np.abs(y_test - y_pred))
        assert mae < 1.0, f"MAE too high: {mae}"

    def test_ensemble_individual_predictions(self, dummy_data):
        """Test that ensemble can return individual model predictions."""
        X_train, y_train, X_test, y_test = dummy_data

        ensemble = EnsembleForecaster(models=["ridge", "xgboost"])
        ensemble.fit(X_train, y_train)

        individual_preds = ensemble.predict_individual(X_test)

        assert "ridge" in individual_preds
        assert "xgboost" in individual_preds
        assert len(individual_preds["ridge"]) == len(y_test)
        assert len(individual_preds["xgboost"]) == len(y_test)


class TestTradingStrategy:
    """Test trading strategy functionality."""

    @pytest.fixture
    def dummy_predictions(self):
        """Create dummy predictions."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=72, freq='h')
        predictions = pd.DataFrame({
            'datetime_hour': dates,
            'actual': np.random.uniform(40, 100, 72),
            'predicted': np.random.uniform(40, 100, 72),
            'iteration': [1]*24 + [2]*24 + [3]*24
        })
        return predictions

    def test_peak_shaving_strategy_creation(self):
        """Test that peak shaving strategy can be created."""
        strategy = PeakShavingStrategy(initial_capital=10000, trade_volume=1.0)
        assert strategy.initial_capital == 10000
        assert strategy.trade_volume == 1.0

    def test_peak_shaving_execute_trades(self, dummy_predictions):
        """Test that peak shaving can execute trades."""
        strategy = PeakShavingStrategy()
        trades_df = strategy.execute_trades(dummy_predictions)

        # Should have 3 days of trades
        assert len(trades_df) == 3

        # Check required columns
        required_cols = ['iteration', 'date', 'buy_hour', 'sell_hour', 'pnl']
        for col in required_cols:
            assert col in trades_df.columns, f"Missing column: {col}"

        # Check PnL is numeric
        assert trades_df['pnl'].dtype in [np.float64, np.int64]

    def test_peak_shaving_metrics(self, dummy_predictions):
        """Test that trading metrics are calculated correctly."""
        strategy = PeakShavingStrategy(initial_capital=10000)
        trades_df = strategy.execute_trades(dummy_predictions)
        metrics = strategy.calculate_metrics(trades_df, initial_capital=10000)

        # Check all required metrics
        required_metrics = ['total_pnl', 'n_trades', 'win_rate', 'avg_profit',
                            'max_drawdown', 'sharpe_ratio', 'return_pct']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Check metric ranges
        assert 0 <= metrics['win_rate'] <= 100, "Win rate must be 0-100%"
        assert metrics['n_trades'] == 3, "Should have 3 trades"

    def test_backtest_strategy(self, dummy_predictions):
        """Test backtest function."""
        trades_df, metrics = backtest_strategy(
            dummy_predictions,
            strategy="peak_shaving",
            initial_capital=10000,
            trade_volume=1.0
        )

        assert len(trades_df) > 0, "No trades executed"
        assert 'total_pnl' in metrics, "Missing total_pnl metric"


class TestIntegration:
    """Integration tests for end-to-end pipeline."""

    def test_walk_forward_with_real_data(self):
        """Test walk-forward validation with actual data (small subset)."""
        # Load data
        df = pd.read_csv(UNIFIED_DATASET_PATH)
        df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])

        # Load features
        with open(FEATURE_FILE, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]

        # Use only first 1000 rows for fast test
        df_small = df.head(1000).copy()

        # Run walk-forward (very short)
        predictions_df, metrics = train_ensemble_walk_forward(
            df=df_small,
            feature_cols=feature_cols,
            target_col='price',
            initial_train_days=10,  # Small for test
            forecast_horizon_hours=24,
            models=["ridge", "xgboost"],
            weights={"ridge": 0.3, "xgboost": 0.7},
            verbose=False
        )

        # Check predictions were made
        assert len(predictions_df) > 0, "No predictions generated"
        assert 'predicted' in predictions_df.columns
        assert 'actual' in predictions_df.columns

        # Check metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mae'] > 0

    def test_end_to_end_pipeline(self):
        """Test complete pipeline: train → predict → trade."""
        # Load data (small subset)
        df = pd.read_csv(UNIFIED_DATASET_PATH)
        df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])

        with open(FEATURE_FILE, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]

        df_small = df.head(1000).copy()

        # 1. Train ensemble
        predictions_df, pred_metrics = train_ensemble_walk_forward(
            df=df_small,
            feature_cols=feature_cols,
            target_col='price',
            initial_train_days=10,
            forecast_horizon_hours=24,
            models=["ridge", "xgboost"],
            weights={"ridge": 0.3, "xgboost": 0.7},
            verbose=False
        )

        # 2. Execute trading
        trades_df, trading_metrics = backtest_strategy(
            predictions_df,
            strategy="peak_shaving",
            initial_capital=10000,
            trade_volume=1.0
        )

        # 3. Verify results
        assert len(predictions_df) > 0, "No predictions"
        assert len(trades_df) > 0, "No trades"
        assert 'total_pnl' in trading_metrics
        assert 'mae' in pred_metrics


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
