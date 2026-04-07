"""Train LightGBM quantile model (p10, p50, p90) with walk-forward validation.

Saves:
  outputs_pipeline/models/lightgbm_quantile_final.pkl   — full-dataset model for inference
  outputs_pipeline/reports/lightgbm_quantile_predictions.csv
  outputs_pipeline/reports/lightgbm_quantile_daily_metrics.csv
  outputs_pipeline/reports/lightgbm_quantile_metrics.txt

Usage:
    python scripts/train_quantile.py
    python scripts/train_quantile.py --no-walk-forward   # faster, just train + save
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.walk_forward_validator import walk_forward_validate, save_walk_forward_results
from src.models.quantile_model import QuantileWrapper

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH     = project_root / "data/dataset.csv"
FEATURES_PATH = project_root / "data/features.txt"
MODELS_DIR    = project_root / "outputs_pipeline/models"
REPORTS_DIR   = project_root / "outputs_pipeline/reports"

QUANTILES    = [0.05, 0.5, 0.95]   # wider interval → better coverage on fat-tail prices
TRAIN_RATIO  = 0.80

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=25,
    min_child_samples=50,      # larger leaf → smoother quantile boundary, avoids overfit
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=3.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)


# QuantileWrapper is defined in src/models/quantile_model.py (importable for pickle)


def load_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "datetime_hour"})
    df = df.sort_values("datetime_hour").reset_index(drop=True)
    df = df.drop(columns=["gap"], errors="ignore")

    EXCLUDE = {"datetime_hour", "price", "load_mw"}
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    logger.info(f"Dataset: {len(df):,} rows — {df['datetime_hour'].min().date()} to {df['datetime_hour'].max().date()}")
    logger.info(f"Price range: [{df['price'].min():.1f}, {df['price'].max():.1f}] EUR/MWh")
    return df, feature_cols


def compute_initial_train_days(df):
    total_days = (df["datetime_hour"].max() - df["datetime_hour"].min()).days
    train_days = int(total_days * TRAIN_RATIO)
    split_date = df["datetime_hour"].min() + pd.Timedelta(days=train_days)
    logger.info(
        f"80/20 split: train {train_days}d → {split_date.date()}, "
        f"test {total_days - train_days}d → {df['datetime_hour'].max().date()}"
    )
    return train_days


def save_quantile_walk_forward_results(predictions_df, daily_metrics_raw, metrics):
    """Save quantile-specific outputs (p10/p50/p90 columns)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_file = REPORTS_DIR / "lightgbm_quantile_predictions.csv"
    predictions_df.to_csv(pred_file, index=False)
    logger.info(f"Quantile predictions saved: {pred_file}")

    daily_df = pd.DataFrame(daily_metrics_raw)
    daily_file = REPORTS_DIR / "lightgbm_quantile_daily_metrics.csv"
    daily_df.to_csv(daily_file, index=False)
    logger.info(f"Quantile daily metrics saved: {daily_file}")

    metrics_file = REPORTS_DIR / "lightgbm_quantile_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("Walk-Forward Validation Results: lightgbm_quantile\n")
        f.write("=" * 80 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    logger.info(f"Quantile metrics saved: {metrics_file}")


def walk_forward_quantile(df, feature_cols, model):
    """Walk-forward validation that captures p10/p50/p90 per prediction day."""
    initial_train_days = compute_initial_train_days(df)
    train_end_idx = initial_train_days * 24

    df = df.sort_values("datetime_hour").reset_index(drop=True)

    X_init = df[feature_cols].iloc[:train_end_idx].values
    y_init = df["price"].iloc[:train_end_idx].values
    model.fit(X_init, y_init)
    logger.info(f"Initial model fitted on {initial_train_days} days.")

    all_rows = []
    daily_metrics = []

    first_test_dt    = df.iloc[train_end_idx]["datetime_hour"]
    first_pred_date  = (first_test_dt.normalize() + pd.Timedelta(days=1)).date()
    current_date     = pd.Timestamp(first_pred_date)
    iteration        = 0

    while current_date <= df["datetime_hour"].max():
        iteration += 1
        next_date = current_date + pd.Timedelta(days=1)
        mask = (df["datetime_hour"] >= current_date) & (df["datetime_hour"] < next_date)
        day_df = df[mask]

        if len(day_df) != 24:
            current_date = next_date
            continue

        X_day = day_df[feature_cols].values
        y_day = day_df["price"].values

        p10, p50, p90 = model.predict_interval(X_day)

        # Coverage: actual inside [p10, p90]
        in_interval = (y_day >= p10) & (y_day <= p90)

        for i, (_, row) in enumerate(day_df.iterrows()):
            all_rows.append({
                "datetime_hour":    row["datetime_hour"],
                "actual":           y_day[i],
                "pred_p10":         p10[i],
                "pred_p50":         p50[i],
                "pred_p90":         p90[i],
                "error_p50":        p50[i] - y_day[i],
                "interval_width":   p90[i] - p10[i],
                "in_interval":      bool(in_interval[i]),
                "iteration":        iteration,
                "date":             current_date.date(),
            })

        # Daily metrics
        mae  = mean_absolute_error(y_day, p50)
        rmse = np.sqrt(mean_squared_error(y_day, p50))
        r2   = r2_score(y_day, p50)
        cov  = in_interval.mean()
        daily_metrics.append({
            "datetime":          current_date,
            "iteration":         iteration,
            "mae_p50":           mae,
            "rmse_p50":          rmse,
            "r2_p50":            r2,
            "coverage_p10_p90":  cov,
            "mean_interval_width": (p90 - p10).mean(),
        })

        if iteration % 30 == 0:
            logger.info(
                f"Iter {iteration:4d}  {current_date.date()}  "
                f"MAE={mae:.1f}  Coverage={cov*100:.0f}%  Width={p90.mean()-p10.mean():.1f}"
            )

        # Retrain daily
        current_date = next_date
        train_mask = df["datetime_hour"] < current_date
        X_tr = df[feature_cols][train_mask].values
        y_tr = df["price"][train_mask].values
        model.fit(X_tr, y_tr)

    predictions_df = pd.DataFrame(all_rows)

    # Overall metrics
    actual = predictions_df["actual"].values
    p50_all = predictions_df["pred_p50"].values
    p10_all = predictions_df["pred_p10"].values
    p90_all = predictions_df["pred_p90"].values

    overall_metrics = {
        "mae_p50":           mean_absolute_error(actual, p50_all),
        "rmse_p50":          np.sqrt(mean_squared_error(actual, p50_all)),
        "r2_p50":            r2_score(actual, p50_all),
        "coverage_p10_p90":  float(np.mean((actual >= p10_all) & (actual <= p90_all))),
        "mean_interval_width": float(np.mean(p90_all - p10_all)),
        "n_predictions":     len(predictions_df),
        "n_iterations":      iteration,
    }

    logger.info(
        f"\n{'='*60}\n"
        f"QUANTILE WALK-FORWARD RESULTS\n"
        f"{'='*60}\n"
        f"  MAE (p50)          : {overall_metrics['mae_p50']:.2f} EUR/MWh\n"
        f"  RMSE (p50)         : {overall_metrics['rmse_p50']:.2f} EUR/MWh\n"
        f"  R² (p50)           : {overall_metrics['r2_p50']:.3f}\n"
        f"  Coverage p10-p90   : {overall_metrics['coverage_p10_p90']*100:.1f}%  (target: 80%)\n"
        f"  Mean interval width: {overall_metrics['mean_interval_width']:.1f} EUR/MWh\n"
        f"  Predictions        : {overall_metrics['n_predictions']:,}\n"
        f"{'='*60}"
    )

    return predictions_df, daily_metrics, overall_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM quantile model")
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation — only train on full dataset and save model",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LIGHTGBM QUANTILE — TRAINING")
    print("=" * 60)

    df, feature_cols = load_dataset()
    model = QuantileWrapper(quantiles=QUANTILES, **LGB_PARAMS)

    if not args.no_walk_forward:
        print("\nRunning walk-forward validation (this takes a few minutes)...")
        predictions_df, daily_metrics, overall_metrics = walk_forward_quantile(
            df, feature_cols, model
        )
        save_quantile_walk_forward_results(predictions_df, daily_metrics, overall_metrics)
    else:
        print("\nSkipping walk-forward (--no-walk-forward). Training on full dataset only.")

    # ── Retrain on full dataset for inference ──────────────────────────────────
    print("\nRetraining on full dataset for inference...")
    X_all = df[feature_cols].values
    y_all = df["price"].values
    model.fit(X_all, y_all)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lightgbm_quantile_final.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Quantile model saved: {model_path}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model : {model_path}")
    print(f"  Reports: {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
