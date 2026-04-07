"""Production configuration for energy price forecasting and trading."""

from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model/saved_models"
REPORTS_DIR = PROJECT_ROOT / "research/reports"

# Data paths
UNIFIED_DATASET_PATH = (
    DATA_DIR / "processed/price_forecasting_dataset_with_forecasts.csv"
)

# Feature configuration
FEATURE_FILE = (
    DATA_DIR / "processed/price_forecasting_dataset_with_forecasts_features.txt"
)

# Model configuration
MODELS_TO_TRAIN = ["ridge", "xgboost"]  # Add "lightgbm", "lstm" when ready
ENSEMBLE_WEIGHTS = {
    "ridge": 0.3,
    "xgboost": 0.7,
}

# Walk-forward validation parameters
INITIAL_TRAIN_DAYS = 365
FORECAST_HORIZON_HOURS = 24
RETRAIN_FREQUENCY_DAYS = 1

# Trading configuration
TRADING_STRATEGY = "peak_shaving"  # Buy at predicted low, sell at predicted high
INITIAL_CAPITAL = 10000  # EUR
TRADE_VOLUME = 1  # MWh per trade

# Model hyperparameters
MODEL_PARAMS = {
    "ridge": {"alpha": 10.0, "solver": "lsqr", "random_state": 42},
    "xgboost": {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    },
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "production/logs/production.log"

# Production mode
PRODUCTION_MODE = False  # Set to True to disable verbose output

# Adaptive Bayesian Tuning
TUNE_FREQUENCY_DAYS = 30  # Re-tune every 30 days
TUNING_ITERATIONS = 20  # Iterations for Bayesian Optimization
ADAPTIVE_MODEL_TYPE = "xgboost"
