"""Daily inference script — predict electricity price for tomorrow (J+1).

This script runs every morning. It:
1. Loads the latest saved models (trained weekly by scripts/train.py)
2. Builds the feature row for J+1 using data available today (J):
     - Lags from the processed dataset (price_lag_24h, etc.)
     - Weather forecasts for J+1 (Open-Meteo)
     - Load forecast for J+1
     - Calendar features for J+1
3. Runs each model and produces an ensemble prediction
4. Appends the prediction to outputs_pipeline/predictions/predictions.csv

Usage:
    python scripts/infer.py                  # predict tomorrow
    python scripts/infer.py --date 2024-06-01  # predict a specific date (backtest mode)
"""

import sys
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Must be imported so joblib can unpickle QuantileWrapper
from src.models.quantile_model import QuantileWrapper  # noqa: F401

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH    = project_root / "data/dataset.csv"
FEATURES_PATH   = project_root / "data/features.txt"
MODELS_DIR      = project_root / "outputs_pipeline/models"
PREDICTIONS_DIR = project_root / "outputs_pipeline/predictions"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all saved models from outputs_pipeline/models/."""
    models = {}

    xgb_path = MODELS_DIR / "xgboost_final.json"
    if xgb_path.exists():
        m = xgb.XGBRegressor()
        m.load_model(str(xgb_path))
        models["xgboost"] = m
        logger.info(f"Loaded XGBoost from {xgb_path}")

    # Standard LightGBM (point forecast)
    lgb_path = MODELS_DIR / "lightgbm_final.pkl"
    if lgb_path.exists():
        models["lightgbm"] = joblib.load(lgb_path)
        logger.info(f"Loaded LightGBM from {lgb_path}")

    # Quantile LightGBM (p10/p50/p90) — loaded separately, not in models dict
    lgb_q_path = MODELS_DIR / "lightgbm_quantile_final.pkl"
    if lgb_q_path.exists():
        models["_lightgbm_quantile"] = joblib.load(lgb_q_path)
        logger.info(f"Loaded LightGBM quantile from {lgb_q_path}")

    ridge_path = MODELS_DIR / "ridge_final.pkl"
    if ridge_path.exists():
        models["ridge"] = joblib.load(ridge_path)
        logger.info(f"Loaded Ridge from {ridge_path}")

    if not models:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_DIR}.\n"
            "Run first: python scripts/train.py"
        )

    return models


def load_feature_cols() -> list:
    """Load the feature column list."""
    with open(FEATURES_PATH) as f:
        return [line.strip() for line in f if line.strip()]


# ── Weather forecast ───────────────────────────────────────────────────────────

WIND_CAPACITY_MW  = 22_000
SOLAR_CAPACITY_MW = 17_000

def fetch_weather_forecast(target_date: datetime) -> pd.DataFrame | None:
    """Fetch hourly weather forecast for target_date from Open-Meteo forecast API.

    Uses the free Open-Meteo forecast API (no key required).
    Returns a DataFrame with 24 rows indexed by hour, or None on failure.

    This is legitimate data for J+1 prediction: weather forecasts for tomorrow
    are published by 12:00 today and are used by all market participants.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 48.8566,
                "longitude": 2.3522,
                "hourly": "temperature_2m,wind_speed_10m,precipitation,shortwave_radiation",
                "timezone": "Europe/Paris",
                "start_date": date_str,
                "end_date": date_str,
                "wind_speed_unit": "ms",
            },
            timeout=15,
        )
        resp.raise_for_status()
        h = resp.json()["hourly"]
        df = pd.DataFrame({
            "datetime_hour": pd.to_datetime(h["time"]),
            "temperature_2m":      h["temperature_2m"],
            "wind_speed_10m":      h["wind_speed_10m"],
            "precipitation":       h["precipitation"],
            "shortwave_radiation": h["shortwave_radiation"],
        })
        # Keep only the 24 hours of target_date
        df = df[df["datetime_hour"].dt.date == target_date.date()].reset_index(drop=True)

        # Derive renewable production features (same formulas as build_dataset.py)
        wind = df["wind_speed_10m"].values
        p = np.zeros_like(wind)
        mask_mid   = (wind >= 3) & (wind < 12)
        mask_rated = (wind >= 12) & (wind <= 25)
        p[mask_mid]   = ((wind[mask_mid] - 3) / (12 - 3)) ** 3
        p[mask_rated] = 1.0
        df["wind_production_mw"]      = p * WIND_CAPACITY_MW
        df["solar_production_mw"]     = (df["shortwave_radiation"].clip(lower=0) / 1000) * SOLAR_CAPACITY_MW * 0.17
        df["renewable_production_mw"] = df["wind_production_mw"] + df["solar_production_mw"]
        # net_load_mw and renewable_penetration_pct need load_mw — left NaN here, imputer handles it

        logger.info(
            f"Weather forecast fetched for {date_str}: "
            f"temp={df['temperature_2m'].mean():.1f}°C  "
            f"wind={df['wind_speed_10m'].mean():.1f}m/s  "
            f"solar={df['shortwave_radiation'].mean():.0f}W/m²"
        )
        return df

    except Exception as e:
        logger.warning(f"Weather forecast unavailable ({e}) — will use NaN for weather features")
        return None


# ── Feature building ───────────────────────────────────────────────────────────

def build_inference_features(target_date: datetime, df_hist: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Build the feature vector for target_date using only data available at J (today).

    For each hour h in target_date (J+1, hours 0-23):
    - price_lag_24h  = price at h-24h  → last known price (same hour yesterday = J)
    - price_lag_48h  = price at h-48h  → J-1
    - price_lag_168h = price at h-168h → same hour last week
    - weather features                 → Open-Meteo forecast API for J+1 (if target is future)
                                         or df_hist actuals (if target is past/present)
    - calendar features                → computed from target_date timestamps

    Args:
        target_date: The date to predict (J+1), as datetime (midnight)
        df_hist:     Full processed dataset (sorted by datetime_hour)
        feature_cols: List of feature columns

    Returns:
        DataFrame with 24 rows (one per hour) and feature_cols columns
    """
    # Hours to predict: J+1 00:00 → 23:00
    target_hours = pd.date_range(target_date, periods=24, freq="h")

    # Check we have the needed historical rows (lags go back up to 168h = 7 days)
    oldest_needed = target_date - timedelta(hours=168)
    available = df_hist[df_hist["datetime_hour"] < target_date]

    if available.empty:
        raise ValueError(f"No historical data available before {target_date}")

    oldest_available = available["datetime_hour"].min()
    if oldest_available > oldest_needed:
        logger.warning(
            f"Limited history: oldest available={oldest_available}, "
            f"needed for 168h lags={oldest_needed}. "
            "168h lag features may be NaN."
        )

    # Build feature rows for each target hour
    target_rows = pd.DataFrame({"datetime_hour": target_hours})

    # ── Calendar features (always computable) ────────────────────────────────
    dt = target_rows["datetime_hour"]
    target_rows["hour"]         = dt.dt.hour
    target_rows["day_of_week"]  = dt.dt.dayofweek
    target_rows["day_of_month"] = dt.dt.day
    target_rows["day_of_year"]  = dt.dt.dayofyear
    target_rows["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    target_rows["month"]        = dt.dt.month
    target_rows["quarter"]      = dt.dt.quarter
    target_rows["year"]         = dt.dt.year
    target_rows["hour_sin"]     = np.sin(2 * np.pi * target_rows["hour"] / 24)
    target_rows["hour_cos"]     = np.cos(2 * np.pi * target_rows["hour"] / 24)
    target_rows["dow_sin"]      = np.sin(2 * np.pi * target_rows["day_of_week"] / 7)
    target_rows["dow_cos"]      = np.cos(2 * np.pi * target_rows["day_of_week"] / 7)
    target_rows["month_sin"]    = np.sin(2 * np.pi * target_rows["month"] / 12)
    target_rows["month_cos"]    = np.cos(2 * np.pi * target_rows["month"] / 12)
    target_rows["is_weekend"]   = (target_rows["day_of_week"] >= 5).astype(int)
    target_rows["is_peak_hour"] = target_rows["hour"].between(8, 20).astype(int)
    target_rows["is_night"]     = (target_rows["hour"] < 6).astype(int)
    target_rows["is_winter"]    = target_rows["month"].isin([12, 1, 2]).astype(int)
    target_rows["is_summer"]    = target_rows["month"].isin([6, 7, 8]).astype(int)

    # ── Lag features (from historical data, time-based lookup) ───────────────
    for lag in [24, 48, 72, 168]:
        lag_times = target_hours - timedelta(hours=lag)
        lag_prices = df_hist.set_index("datetime_hour")["price"].reindex(lag_times).values
        lag_load   = df_hist.set_index("datetime_hour")["load_mw"].reindex(lag_times).values
        target_rows[f"price_lag_{lag}h"] = lag_prices
        target_rows[f"load_lag_{lag}h"]  = lag_load

    # ── Rolling stats (from available history) ────────────────────────────────
    # Use the most recent 168 hours before target_date
    history_window = df_hist[df_hist["datetime_hour"] < target_date].tail(168)
    if len(history_window) >= 12:
        target_rows["price_roll_mean_24h"]  = history_window["price"].tail(24).mean()
        target_rows["price_roll_std_24h"]   = history_window["price"].tail(24).std()
        target_rows["price_roll_mean_168h"] = history_window["price"].tail(168).mean()
        target_rows["price_roll_std_168h"]  = history_window["price"].tail(168).std()
        target_rows["load_roll_mean_24h"]   = history_window["load_mw"].tail(24).mean()
        target_rows["load_roll_std_24h"]    = history_window["load_mw"].tail(24).std()
        target_rows["load_roll_mean_168h"]  = history_window["load_mw"].tail(168).mean()
        target_rows["load_roll_std_168h"]   = history_window["load_mw"].tail(168).std()

    # ── Weather features ─────────────────────────────────────────────────────
    # Priority 1: df_hist actuals (for past/present dates in the dataset)
    # Priority 2: Open-Meteo forecast API (for future dates not yet in dataset)
    hist_indexed = df_hist.set_index("datetime_hour")

    WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "precipitation", "shortwave_radiation",
                    "wind_production_mw", "solar_production_mw", "renewable_production_mw"]

    # Check if target_date weather is already in df_hist
    hist_weather = {c: hist_indexed[c].reindex(target_hours).values
                    for c in WEATHER_COLS if c in hist_indexed.columns}
    weather_in_hist = any(~pd.isna(v).all() for v in hist_weather.values())

    if weather_in_hist:
        # Use actuals from dataset (backtest / recent past)
        for c, vals in hist_weather.items():
            target_rows[c] = vals
        logger.info("Weather: using actuals from dataset")
    else:
        # Fetch forecast from Open-Meteo (live J+1 inference)
        wx = fetch_weather_forecast(target_date)
        if wx is not None:
            wx_indexed = wx.set_index("datetime_hour")
            for c in WEATHER_COLS:
                if c in wx_indexed.columns:
                    target_rows[c] = wx_indexed[c].reindex(target_hours).values

    # ── Remaining features not yet filled ────────────────────────────────────
    for c in feature_cols:
        if c not in target_rows.columns:
            if c in hist_indexed.columns:
                target_rows[c] = hist_indexed[c].reindex(target_hours).values
            else:
                target_rows[c] = np.nan

    X = target_rows[feature_cols].copy()

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        feat_with_nan = X.columns[X.isna().any()].tolist()
        logger.warning(f"{nan_count} NaN values in inference features: {feat_with_nan}")

    return X, target_rows["datetime_hour"]


# ── Ensemble prediction ────────────────────────────────────────────────────────

def predict_ensemble(models: dict, X: pd.DataFrame) -> dict:
    """Run each model and return individual + equal-weight ensemble + quantile predictions."""
    preds = {}
    quantile_model = models.pop("_lightgbm_quantile", None)

    has_nan = X.isna().any().any()
    for name, model in models.items():
        # Ridge without imputer cannot handle NaN — check if pipeline has imputer
        ridge_has_imputer = (
            name == "ridge"
            and hasattr(model, "named_steps")
            and "imputer" in model.named_steps
        )
        if has_nan and name == "ridge" and not ridge_has_imputer:
            logger.warning(f"  ridge: skipped (NaN in features — no imputer in pipeline)")
            continue
        try:
            p = model.predict(X)
            preds[name] = p
            logger.info(f"  {name:12s}  mean={p.mean():.1f}  min={p.min():.1f}  max={p.max():.1f} EUR/MWh")
        except Exception as e:
            logger.error(f"  {name} prediction failed: {e}")

    if not preds:
        raise RuntimeError("All models failed to predict.")

    # Equal-weight ensemble (point forecast models only)
    preds["ensemble"] = np.mean(list(preds.values()), axis=0)

    # Quantile predictions (p10, p50, p90)
    if quantile_model is not None:
        try:
            p10, p50, p90 = quantile_model.predict_interval(X)
            preds["q10"] = p10
            preds["q50"] = p50
            preds["q90"] = p90
            logger.info(
                f"  quantile      "
                f"  p10={p10.mean():.1f}  p50={p50.mean():.1f}  p90={p90.mean():.1f} EUR/MWh"
            )
        except Exception as e:
            logger.error(f"  lightgbm_quantile prediction failed: {e}")

    return preds


# ── Output ─────────────────────────────────────────────────────────────────────

def save_predictions(target_date: datetime, datetimes: pd.Series, preds: dict):
    """Append predictions to outputs_pipeline/predictions/predictions.csv.

    Handles schema migration: if the existing file has fewer columns than the
    new predictions (e.g. added q10/q50/q90), it is rewritten with NaN fill
    for missing columns so the file stays readable.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    QUANTILE_KEYS = {"q10", "q50", "q90"}
    point_preds   = {k: v for k, v in preds.items() if k not in QUANTILE_KEYS}
    quant_preds   = {k: v for k, v in preds.items() if k in QUANTILE_KEYS}

    rows = []
    for i, dt in enumerate(datetimes):
        row = {
            "prediction_made_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_datetime": dt,
            "target_date": target_date.date(),
        }
        for model_name, p in point_preds.items():
            row[f"pred_{model_name}"] = round(float(p[i]), 2)
        for q_name, p in quant_preds.items():
            row[f"pred_{q_name}"] = round(float(p[i]), 2)
        rows.append(row)

    df_new = pd.DataFrame(rows)
    out_path = PREDICTIONS_DIR / "predictions.csv"

    if out_path.exists():
        try:
            df_old = pd.read_csv(out_path)
            # If schemas differ, add missing columns (filled with NaN) and rewrite
            new_cols = [c for c in df_new.columns if c not in df_old.columns]
            if new_cols:
                logger.info(f"Schema migration: adding columns {new_cols} to existing predictions")
                for c in new_cols:
                    df_old[c] = float("nan")
                # Reorder columns to match new schema
                df_old = df_old.reindex(columns=df_new.columns)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_csv(out_path, index=False)
            else:
                df_new.to_csv(out_path, mode="a", header=False, index=False)
        except Exception as e:
            logger.warning(f"Could not read existing predictions ({e}), overwriting.")
            df_new.to_csv(out_path, index=False)
    else:
        df_new.to_csv(out_path, index=False)

    logger.info(f"Predictions saved to {out_path}")
    return df_new


def print_summary(target_date: datetime, datetimes: pd.Series, preds: dict):
    ensemble = preds["ensemble"]
    print(f"\n{'='*60}")
    print(f"  Price forecast for {target_date.date()} (J+1)")
    print(f"{'='*60}")
    print(f"  Models used      : {', '.join(k for k in preds if k != 'ensemble')}")
    print(f"  Ensemble mean    : {ensemble.mean():.1f} EUR/MWh")
    print(f"  Ensemble min     : {ensemble.min():.1f} EUR/MWh  (hour {ensemble.argmin():02d}:00)")
    print(f"  Ensemble max     : {ensemble.max():.1f} EUR/MWh  (hour {ensemble.argmax():02d}:00)")
    peak_spread = ensemble.max() - ensemble.min()
    print(f"  Peak-off spread  : {peak_spread:.1f} EUR/MWh")
    print(f"{'='*60}")

    # Hourly table (compact)
    print(f"  {'Hour':>4}  {'Ensemble':>10}", end="")
    for name in preds:
        if name != "ensemble":
            print(f"  {name:>10}", end="")
    print()
    for i in range(24):
        line = f"  {i:02d}:00  {preds['ensemble'][i]:>10.1f}"
        for name, p in preds.items():
            if name != "ensemble":
                line += f"  {p[i]:>10.1f}"
        print(line)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily price inference — predict J+1")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date to predict (YYYY-MM-DD). Default: tomorrow.",
    )
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"  INFERENCE — target: {target_date.date()}")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Load models
    models = load_models()
    feature_cols = load_feature_cols()

    # 2. Load historical dataset
    logger.info(f"Loading dataset...")
    df_hist = pd.read_csv(DATASET_PATH, parse_dates=["datetime"])
    df_hist = df_hist.rename(columns={"datetime": "datetime_hour"})
    df_hist = df_hist.drop(columns=["gap"], errors="ignore")
    df_hist = df_hist.sort_values("datetime_hour").reset_index(drop=True)
    last_date = df_hist["datetime_hour"].max()
    logger.info(f"Dataset: {len(df_hist):,} rows, last date: {last_date.date()}")

    # Warn if dataset is stale relative to target date
    lag_days = (target_date - last_date).days
    if lag_days > 2:
        logger.warning(
            f"Dataset ends {last_date.date()}, target is {target_date.date()} "
            f"({lag_days}d gap). Run scripts/update_data.py to refresh."
        )
    if lag_days > 7:
        logger.warning(
            f"Gap of {lag_days}d — lag features will be NaN. "
            "Only NaN-tolerant models (XGBoost, LightGBM) will run. "
            "Predictions rely on calendar + weather features only."
        )

    # 3. Build features for J+1
    logger.info(f"Building features for {target_date.date()}...")
    X, datetimes = build_inference_features(target_date, df_hist, feature_cols)

    # 4. Predict
    logger.info("Running predictions...")
    preds = predict_ensemble(models, X)

    # 5. Print and save
    print_summary(target_date, datetimes, preds)
    df_out = save_predictions(target_date, datetimes, preds)

    return df_out


if __name__ == "__main__":
    main()
