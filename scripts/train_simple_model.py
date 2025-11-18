"""
Simple model training script for energy demand forecasting

This script trains a simple XGBoost model on the prepared data.

Usage:
    python train_simple_model.py --model xgboost
    python train_simple_model.py --model lightgbm
    python train_simple_model.py --model ridge
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent  # Go up to project root
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(frequency='daily'):
    """Load train and test data."""
    logger.info("Loading data...")

    train_file = DATA_DIR / f"train_{frequency}.csv"
    test_file = DATA_DIR / f"test_{frequency}.csv"

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    logger.info(f"  Train: {len(df_train)} records")
    logger.info(f"  Test:  {len(df_test)} records")

    return df_train, df_test


def prepare_features(df_train, df_test, target='load'):
    """Prepare X and y for training.

    Args:
        target: 'load' to predict load_mw, 'price' to predict price_eur_mwh
    """
    logger.info("Preparing features...")

    if target == 'load':
        # Predict load (remove current price to avoid leakage)
        drop_cols = ['datetime', 'country', 'price_eur_mwh']
        target_col = 'load_mw'
    elif target == 'price':
        # Predict price (remove current load to avoid leakage)
        drop_cols = ['datetime', 'country', 'load_mw']
        target_col = 'price_eur_mwh'
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'load' or 'price'")

    # Train
    X_train = df_train.drop(columns=drop_cols + [target_col], errors='ignore')
    y_train = df_train[target_col]

    # Test
    X_test = df_test.drop(columns=drop_cols + [target_col], errors='ignore')
    y_test = df_test[target_col]

    logger.info(f"  Features: {len(X_train.columns)}")
    logger.info(f"  Target: {target_col}")

    return X_train, y_train, X_test, y_test


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    logger.info("Training XGBoost...")

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM model."""
    logger.info("Training LightGBM...")

    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train)

        return model

    except ImportError:
        logger.error("LightGBM not installed. Install with: pip install lightgbm")
        return None


def train_ridge(X_train, y_train):
    """Train Ridge regression model."""
    logger.info("Training Ridge Regression...")

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    logger.info("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    logger.info(f"  MAE:  {mae:.2f} MW")
    logger.info(f"  RMSE: {rmse:.2f} MW")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }


def save_model(model, model_name, metrics, feature_names):
    """Save trained model and metadata."""
    logger.info("Saving model...")

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = model_dir / "model.pkl"
    joblib.dump(model, model_file)
    logger.info(f"  Model: {model_file}")

    # Save metrics
    metrics_file = model_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics: {metrics_file}")

    # Save features
    features_file = model_dir / "features.json"
    with open(features_file, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"  Features: {features_file}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train energy demand forecasting model")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "ridge", "random_forest"],
        default="xgboost",
        help="Model type to train"
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly"],
        default="daily",
        help="Data frequency"
    )
    parser.add_argument(
        "--target",
        choices=["load", "price"],
        default="load",
        help="Target variable: 'load' for load_mw, 'price' for price_eur_mwh"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"TRAINING {args.model.upper()} MODEL")
    logger.info("=" * 70)

    try:
        # Load data
        df_train, df_test = load_data(frequency=args.frequency)

        # Prepare features
        X_train, y_train, X_test, y_test = prepare_features(df_train, df_test, target=args.target)

        # Train model
        if args.model == "xgboost":
            model = train_xgboost(X_train, y_train)
        elif args.model == "lightgbm":
            model = train_lightgbm(X_train, y_train)
            if model is None:
                return False
        elif args.model == "ridge":
            model = train_ridge(X_train, y_train)
        elif args.model == "random_forest":
            model = train_random_forest(X_train, y_train)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save (add target suffix to model name)
        model_name = f"{args.model}_{args.target}" if args.target != "load" else args.model
        save_model(model, model_name, metrics, list(X_train.columns))

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
