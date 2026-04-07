"""Training script for price forecasting models with walk-forward validation.

Usage:
    python scripts/train.py                          # train all models with walk-forward
    python scripts/train.py --model xgboost          # train XGBoost only
    python scripts/train.py --model lightgbm         # train LightGBM only
    python scripts/train.py --model ridge            # train Ridge only
    python scripts/train.py --no-walk-forward        # simple 80/20 split (faster)
"""

import sys
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.walk_forward_validator import walk_forward_validate, save_walk_forward_results

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH     = project_root / "data/dataset.csv"
FEATURES_PATH = project_root / "data/features.txt"   # generated on first run
MODELS_DIR    = project_root / "outputs_pipeline/models"
REPORTS_DIR   = project_root / "outputs_pipeline/reports"

# ── Walk-forward parameters ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80         # 80% train / 20% test (computed dynamically from dataset)
RETRAIN_FREQ_DAYS = 1      # daily retrain (realistic for day-ahead market)
FORECAST_HORIZON = 24      # predict 24h ahead


def load_dataset():
    """Load dataset and derive feature list, writing features.txt if absent."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    # Normalise column name so the rest of the code uses datetime_hour
    df = df.rename(columns={"datetime": "datetime_hour"})
    df = df.sort_values("datetime_hour").reset_index(drop=True)
    # drop helper column added by build_dataset
    df = df.drop(columns=["gap"], errors="ignore")

    # Derive features: everything except datetime, price, load_mw (contemporaneous)
    EXCLUDE = {"datetime_hour", "price", "load_mw"}
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    # Persist so infer.py can read the same list
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text("\n".join(feature_cols))

    logger.info(f"Dataset: {len(df):,} rows, {len(feature_cols)} features")
    logger.info(f"Date range: {df['datetime_hour'].min().date()} to {df['datetime_hour'].max().date()}")
    logger.info(f"Price range: [{df['price'].min():.1f}, {df['price'].max():.1f}] EUR/MWh")
    return df, feature_cols


def build_models():
    """Return dict of model_name -> sklearn-compatible model."""
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    ridge_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0, random_state=42)),
    ])

    return {"xgboost": xgb_model, "lightgbm": lgb_model, "ridge": ridge_model}


def print_metrics(name, metrics):
    print(f"\n{'='*60}")
    print(f"  {name} - Walk-Forward Results")
    print(f"{'='*60}")
    print(f"  Predictions : {metrics['n_predictions']:,} hours")
    print(f"  MAE         : {metrics['mae']:.2f} EUR/MWh")
    print(f"  RMSE        : {metrics['rmse']:.2f} EUR/MWh")
    print(f"  R²          : {metrics['r2']:.3f}")
    print(f"  Direction   : {metrics['direction_accuracy']:.1f}%")
    print(f"  Bias        : {metrics['bias']:.2f} EUR/MWh")


def save_model(model, name):
    """Save model to outputs_pipeline/models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        path = MODELS_DIR / "xgboost_final.json"
        model.save_model(str(path))
    else:
        path = MODELS_DIR / f"{name}_final.pkl"
        joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


def compute_initial_train_days(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> int:
    """Compute initial training days so that the walk-forward test period covers
    (1 - train_ratio) of the total dataset duration."""
    total_days = (df["datetime_hour"].max() - df["datetime_hour"].min()).days
    train_days = int(total_days * train_ratio)
    test_days  = total_days - train_days
    split_date = df["datetime_hour"].min() + pd.Timedelta(days=train_days)
    logger.info(
        f"80/20 split: train {train_days}d "
        f"({df['datetime_hour'].min().date()} -> {split_date.date()}), "
        f"test {test_days}d "
        f"({split_date.date()} -> {df['datetime_hour'].max().date()})"
    )
    return train_days


def train_walk_forward(df, feature_cols, models_to_train):
    """Run walk-forward validation for each model and save results."""
    initial_train_days = compute_initial_train_days(df)
    all_predictions = {}

    for name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"  Starting walk-forward: {name.upper()}")
        print(f"  Train: {initial_train_days}d (80%), Test: {(df['datetime_hour'].max()-df['datetime_hour'].min()).days - initial_train_days}d (20%), retrain: daily")
        print(f"{'='*60}")

        predictions_df, metrics, daily_metrics = walk_forward_validate(
            df=df,
            model=model,
            feature_cols=feature_cols,
            target_col="price",
            initial_train_days=initial_train_days,
            forecast_horizon_hours=FORECAST_HORIZON,
            retrain_frequency_days=RETRAIN_FREQ_DAYS,
            expanding_window=True,
            verbose=True,
        )

        print_metrics(name, metrics)

        save_walk_forward_results(
            predictions_df=predictions_df,
            metrics=metrics,
            daily_metrics=daily_metrics,
            model_name=name,
            output_dir=str(REPORTS_DIR),
        )

        # Retrain on full data to save final model
        logger.info(f"Retraining {name} on full dataset for inference...")
        X_all = df[feature_cols].values
        y_all = df["price"].values
        model.fit(X_all, y_all)
        save_model(model, name)

        all_predictions[name] = predictions_df

    return all_predictions


def train_simple_split(df, feature_cols, models_to_train):
    """Simple 80/20 temporal split — quick sanity check, not for production eval."""
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["price"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["price"].values

    print(f"\nSimple split — train: {len(train_df):,}h, test: {len(test_df):,}h")

    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"  {name:12s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
        save_model(model, name)


def main():
    parser = argparse.ArgumentParser(description="Train price forecasting models")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "ridge", "all"],
        default="all",
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Use simple 80/20 split instead of walk-forward (faster)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  PRICE FORECASTING — MODEL TRAINING")
    print("="*60)

    df, feature_cols = load_dataset()

    all_models = build_models()
    if args.model != "all":
        models_to_train = {args.model: all_models[args.model]}
    else:
        models_to_train = all_models

    if args.no_walk_forward:
        train_simple_split(df, feature_cols, models_to_train)
    else:
        train_walk_forward(df, feature_cols, models_to_train)

    print("\n" + "="*60)
    print("  Training complete.")
    print(f"  Models saved in: {MODELS_DIR}")
    print(f"  Reports saved in: {REPORTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
