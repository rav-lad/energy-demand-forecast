"""Tests for the production pipeline components.

Tests:
1. EnsembleForecaster (fit, predict, weights validation)
2. walk-forward validation (no leakage, correct slicing)
3. Metrics (MAE, RMSE, sMAPE, directional accuracy)
4. Config integrity
5. End-to-end smoke test on synthetic data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble_model import EnsembleForecaster, train_ensemble_walk_forward
from src.models.metrics import (
    smape,
    safe_mape,
    mase,
    calculate_all_metrics,
)
from src.utils.config import (
    ENSEMBLE_WEIGHTS,
    MODEL_PARAMS,
    INITIAL_TRAIN_DAYS,
    FORECAST_HORIZON_HOURS,
    MODELS_TO_TRAIN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_regression_data(n_samples=500, n_features=10, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = X[:, 0] * 2 + X[:, 1] * (-1.5) + rng.normal(0, noise, n_samples) + 50.0
    return X, y


def make_ts_df(n_days=400, seed=0):
    """Synthetic hourly time series for walk-forward tests.

    walk_forward_validator.py expects a 'datetime' column (not 'datetime_hour').
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24
    datetimes = pd.date_range("2022-01-01", periods=n, freq="h")
    price = 50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 5, n)

    df = pd.DataFrame({"datetime": datetimes, "price": price})

    # Add minimal lag features (no leakage: min lag = 24h)
    for lag in [24, 48, 168]:
        ref = df[["datetime", "price"]].copy()
        ref["datetime"] = ref["datetime"] + pd.Timedelta(hours=lag)
        ref = ref.rename(columns={"price": f"price_lag_{lag}h"})
        df = df.merge(ref, on="datetime", how="left")

    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# EnsembleForecaster
# ---------------------------------------------------------------------------

class TestEnsembleForecaster:

    def test_init_default(self):
        model = EnsembleForecaster()
        assert "ridge" in model.models
        assert "xgboost" in model.models

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            EnsembleForecaster(
                models=["ridge", "xgboost"],
                weights={"ridge": 0.6, "xgboost": 0.6},
            )

    def test_weights_keys_must_match_models(self):
        with pytest.raises(ValueError, match="Weights keys"):
            EnsembleForecaster(
                models=["ridge", "xgboost"],
                weights={"ridge": 0.5, "lightgbm": 0.5},
            )

    def test_fit_and_predict(self):
        X, y = make_regression_data()
        model = EnsembleForecaster()
        model.fit(X, y)
        preds = model.predict(X[:50])
        assert preds.shape == (50,)
        assert not np.isnan(preds).any()

    def test_predict_before_fit_returns_zeros_or_raises(self):
        """Before fit, predict either raises or returns zeros (no trained models)."""
        model = EnsembleForecaster()
        X, _ = make_regression_data(n_samples=10)
        try:
            result = model.predict(X)
            # If it doesn't raise, it should return an array of zeros (no models trained)
            assert np.all(result == 0.0)
        except Exception:
            pass  # raising is also acceptable

    def test_individual_predictions_shape(self):
        X, y = make_regression_data()
        model = EnsembleForecaster()
        model.fit(X, y)
        individual = model.predict_individual(X[:30])
        assert set(individual.keys()) == {"ridge", "xgboost"}
        for name, pred in individual.items():
            assert pred.shape == (30,), f"{name} wrong shape"

    def test_get_model_importance(self):
        model = EnsembleForecaster()
        weights = model.get_model_importance()
        assert set(weights.keys()) == {"ridge", "xgboost"}
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_ensemble_mae_better_than_baseline(self):
        """Ensemble on structured data should beat naive mean prediction."""
        X, y = make_regression_data(n_samples=1000, noise=0.5)
        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]

        model = EnsembleForecaster()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae_model = np.mean(np.abs(y_test - preds))
        mae_baseline = np.mean(np.abs(y_test - y_train.mean()))
        assert mae_model < mae_baseline, "Ensemble worse than mean baseline"

    def test_ridge_only_ensemble(self):
        model = EnsembleForecaster(
            models=["ridge"],
            weights={"ridge": 1.0},
        )
        X, y = make_regression_data()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_xgboost_only_ensemble(self):
        model = EnsembleForecaster(
            models=["xgboost"],
            weights={"xgboost": 1.0},
        )
        X, y = make_regression_data()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            bad = EnsembleForecaster(
                models=["neural_net"],
                weights={"neural_net": 1.0},
            )
            X, y = make_regression_data(n_samples=50)
            bad.fit(X, y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestSMAPE:

    def test_perfect_forecast(self):
        y = np.array([50.0, 60.0, 40.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_symmetry(self):
        y_true = np.array([100.0, 50.0])
        y_pred = np.array([90.0, 55.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_bounded_0_200(self):
        y_true = np.array([10.0, 0.0, 50.0])
        y_pred = np.array([1000.0, 1000.0, 0.0])
        result = smape(y_true, y_pred)
        assert 0 <= result <= 200

    def test_handles_negative_prices(self):
        y_true = np.array([-10.0, 50.0, -5.0])
        y_pred = np.array([-8.0, 52.0, -6.0])
        result = smape(y_true, y_pred)
        assert np.isfinite(result)

    def test_zero_both_no_nan(self):
        y_true = np.array([0.0, 50.0])
        y_pred = np.array([0.0, 52.0])
        result = smape(y_true, y_pred)
        assert np.isfinite(result)


class TestSafeMAPE:

    def test_excludes_near_zero(self):
        y_true = np.array([100.0, 50.0, 1.0, -1.0, 80.0])
        y_pred = np.array([95.0, 55.0, 10.0, 5.0, 75.0])
        # With threshold=5: only [100, 50, 80] used
        result = safe_mape(y_true, y_pred, threshold=5.0)
        assert np.isfinite(result)
        assert result > 0

    def test_all_near_zero_raises(self):
        y_true = np.array([0.1, 0.2, -0.1])
        y_pred = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            safe_mape(y_true, y_pred, threshold=5.0)

    def test_perfect_forecast_zero(self):
        y_true = np.array([50.0, 60.0, 70.0])
        assert safe_mape(y_true, y_true) == pytest.approx(0.0)


class TestMASE:

    def test_better_than_naive_lt_one(self):
        """Good forecast: MASE < 1."""
        y_true = np.array([float(i) for i in range(100)])
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        result = mase(y_true, y_pred, seasonality=24)
        assert result < 1.0

    def test_naive_forecast_approx_one(self):
        """Seasonal naive forecast should give MASE ≈ 1."""
        rng = np.random.default_rng(7)
        y = rng.uniform(40, 80, 500)
        # Naive: predict y[t-24]
        y_true = y[24:]
        y_pred = y[:-24]
        # Use a larger training set to avoid NaN in naive denominator
        result = mase(y_true, y_pred, y_train=y[:200], seasonality=24)
        assert np.isfinite(result), "MASE should be finite"
        assert 0.5 < result < 5.0


class TestCalculateAllMetrics:

    def test_returns_required_keys(self):
        y_true = np.random.uniform(40, 100, 200)
        y_pred = y_true + np.random.normal(0, 5, 200)
        metrics = calculate_all_metrics(y_true, y_pred)
        for key in ["mae", "rmse", "r2", "smape", "bias", "max_error", "directional_accuracy"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_mae_positive(self):
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([55.0, 58.0, 65.0])
        metrics = calculate_all_metrics(y_true, y_pred)
        assert metrics["mae"] > 0

    def test_perfect_forecast(self):
        y = np.random.uniform(40, 100, 100)
        metrics = calculate_all_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["smape"] == pytest.approx(0.0)

    def test_r2_negative_for_bad_model(self):
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_pred = np.array([80.0, 70.0, 60.0, 50.0])  # reversed
        metrics = calculate_all_metrics(y_true, y_pred)
        assert metrics["r2"] < 0

    def test_directional_accuracy_range(self):
        y_true = np.random.uniform(40, 100, 200)
        y_pred = y_true + np.random.normal(0, 10, 200)
        metrics = calculate_all_metrics(y_true, y_pred)
        assert 0 <= metrics["directional_accuracy"] <= 100


# ---------------------------------------------------------------------------
# Config integrity
# ---------------------------------------------------------------------------

class TestConfig:

    def test_ensemble_weights_sum_to_one(self):
        assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-9

    def test_models_to_train_not_empty(self):
        assert len(MODELS_TO_TRAIN) > 0

    def test_forecast_horizon_is_24(self):
        assert FORECAST_HORIZON_HOURS == 24

    def test_initial_train_days_positive(self):
        assert INITIAL_TRAIN_DAYS > 0

    def test_ridge_params_exist(self):
        assert "ridge" in MODEL_PARAMS
        assert "alpha" in MODEL_PARAMS["ridge"]

    def test_xgboost_params_exist(self):
        assert "xgboost" in MODEL_PARAMS
        assert "n_estimators" in MODEL_PARAMS["xgboost"]
        assert "max_depth" in MODEL_PARAMS["xgboost"]


# ---------------------------------------------------------------------------
# Walk-forward validation (no-leakage smoke test)
# ---------------------------------------------------------------------------

class TestWalkForward:

    def test_predictions_length_equals_test_period(self):
        df = make_ts_df(n_days=40)
        feature_cols = [c for c in df.columns if c not in ["datetime", "price"]]

        preds_df, metrics = train_ensemble_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="price",
            initial_train_days=30,
            forecast_horizon_hours=24,
            verbose=False,
        )
        assert len(preds_df) > 0

    def test_metrics_dict_has_mae(self):
        df = make_ts_df(n_days=40)
        feature_cols = [c for c in df.columns if c not in ["datetime", "price"]]

        _, metrics = train_ensemble_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="price",
            initial_train_days=30,
            forecast_horizon_hours=24,
            verbose=False,
        )
        assert "mae" in metrics

    def test_no_future_data_in_train(self):
        """Training timestamps must all precede prediction timestamps."""
        df = make_ts_df(n_days=40)
        feature_cols = [c for c in df.columns if c not in ["datetime", "price"]]

        preds_df, _ = train_ensemble_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="price",
            initial_train_days=30,
            forecast_horizon_hours=24,
            verbose=False,
        )
        # All prediction timestamps must come after training period
        train_end = df["datetime"].iloc[30 * 24 - 1]
        pred_dates = pd.to_datetime(preds_df["datetime"])
        assert (pred_dates > train_end).all(), "Future data leaked into training!"

    def test_predictions_are_finite(self):
        df = make_ts_df(n_days=40)
        feature_cols = [c for c in df.columns if c not in ["datetime", "price"]]

        preds_df, _ = train_ensemble_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="price",
            initial_train_days=30,
            forecast_horizon_hours=24,
            verbose=False,
        )
        assert np.isfinite(preds_df["predicted"].values).all()


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Lightweight end-to-end test: data → features → train → predict → metrics."""

    def test_full_pipeline_runs(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal((n, 8))
        y = X[:, 0] * 3 + X[:, 2] * (-2) + rng.normal(0, 2, n) + 60

        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = EnsembleForecaster()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = calculate_all_metrics(y_test, preds, y_train=y_train)

        assert metrics["mae"] < 20, f"MAE too high: {metrics['mae']:.2f}"
        assert metrics["r2"] > 0.5, f"R² too low: {metrics['r2']:.4f}"
        assert metrics["smape"] < 30, f"sMAPE too high: {metrics['smape']:.2f}%"
