# MLOps Infrastructure

Professional MLflow-based experiment tracking and model management for energy trading research.

## Overview

This module provides a comprehensive MLflow integration for tracking:
- Model training experiments (price, load, renewable forecasting)
- Hyperparameter optimization
- Model performance metrics
- Trading strategy backtests
- Model versioning and registry

## Quick Start

### Basic Usage

```python
from mlops import MLflowTracker

# Track a forecasting experiment
with MLflowTracker("price_forecasting", run_name="lgbm-quantile-v1") as tracker:
    # Log hyperparameters
    tracker.log_params({
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
    })

    # Train model
    model = train_model(X_train, y_train)

    # Log metrics
    tracker.log_forecast_metrics(y_test, y_pred, prefix="test_")

    # Log model
    tracker.log_model(model, "model", registered_model_name="price_forecast_lgbm")
```

### Model Comparison

```python
from mlops import log_model_comparison

results = {
    "XGBoost": {"rmse": 12.5, "r2": 0.94, "sharpe": 1.65},
    "LightGBM": {"rmse": 11.8, "r2": 0.95, "sharpe": 1.72},
    "RandomForest": {"rmse": 14.2, "r2": 0.91, "sharpe": 1.48},
}

comparison_df = log_model_comparison(results, "model_comparison")
print(comparison_df)
```

### Load Best Model

```python
from mlops import load_best_model

# Load best model based on RMSE (lower is better)
best_model = load_best_model("energy-price-forecasting", "test_rmse", ascending=True)

# Load best strategy based on Sharpe ratio (higher is better)
best_strategy = load_best_model("trading-strategies-backtest", "sharpe_ratio", ascending=False)
```

## Experiment Categories

The system provides predefined experiment categories:

- `price_forecasting` - Energy price prediction models
- `load_forecasting` - Energy demand/load forecasting
- `renewable_forecasting` - Wind/solar generation forecasting
- `trading_strategies` - Strategy backtest results
- `model_comparison` - Cross-model performance analysis

## MLflow UI

View experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri file:///home/user/energy-demand-forecast/mlruns
```

Then open: http://localhost:5000

## Features

### Automatic Metric Logging

```python
# Forecasting metrics: RMSE, MAE, R², MAPE, Directional Accuracy
tracker.log_forecast_metrics(y_true, y_pred, prefix="test_")

# Trading metrics: Sharpe, Returns, Drawdown, Win Rate
tracker.log_trading_metrics(backtest_results, prefix="strategy_")
```

### Artifact Logging

```python
# Log matplotlib figures
tracker.log_figure(plt.gcf(), "forecast_plot.png")

# Log DataFrames
tracker.log_dataframe(results_df, "backtest_results.csv")

# Log JSON configs
tracker.log_dict(config, "config.json")
```

### Experiment Analysis

```python
from mlops import create_experiment_summary

# Get summary of all runs
summary = create_experiment_summary("energy-price-forecasting")
print(summary.sort_values("test_rmse").head())
```

## Integration Examples

### Price Forecasting Model

```python
from mlops import MLflowTracker
import lightgbm as lgb

with MLflowTracker("price_forecasting", run_name="lgbm-fuel-features") as tracker:
    # Log parameters
    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "n_estimators": 500,
        "learning_rate": 0.05,
    }
    tracker.log_params(params)

    # Train
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = tracker.log_forecast_metrics(y_test, y_pred, prefix="test_")

    # Log model
    tracker.log_model(model, "model")

    print(f"RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.3f}")
```

### Trading Strategy Backtest

```python
with MLflowTracker("trading_strategies", run_name="mean-reversion-v2") as tracker:
    # Log strategy parameters
    tracker.log_params({
        "half_life": 12,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "position_size": 0.02,
    })

    # Run backtest
    results = backtest_strategy(strategy, data)

    # Log performance metrics
    tracker.log_trading_metrics(results)

    # Log equity curve
    tracker.log_figure(plot_equity_curve(results), "equity_curve.png")
```

## Directory Structure

```
mlops/
├── __init__.py           # Package exports
├── mlflow_config.py      # Configuration settings
├── mlflow_utils.py       # Tracking utilities
└── README.md            # This file

mlruns/                   # MLflow tracking data (gitignored)
mlartifacts/             # Artifact storage (gitignored)
```

## Benefits for Interviews

This professional MLflow setup demonstrates:

1. **Reproducibility** - All experiments tracked with parameters and seeds
2. **Model Versioning** - Production-ready model registry
3. **Systematic Evaluation** - Consistent metrics across all models
4. **Professional Workflow** - Industry-standard MLOps practices
5. **Collaboration Ready** - Team can share and compare experiments

## Configuration

Edit `mlflow_config.py` to customize:
- Tracking URI (local file, remote server, database)
- Experiment names
- Default tags
- Artifact storage location

## Best Practices

1. **Always use context manager**: Ensures runs are properly closed
2. **Descriptive run names**: Include model type and version
3. **Log hyperparameters first**: Before training starts
4. **Prefix metrics**: Use `train_`, `test_`, `val_` prefixes
5. **Version models**: Use registered_model_name for production models
6. **Tag experiments**: Add meaningful tags for filtering

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
