"""Unified data pipeline for price forecasting with ALL required features.

This module combines:
1. Historical data (prices, load, weather)
2. Weather forecasts (temperature, wind, solar radiation)
3. Load forecasts (electricity demand)
4. Renewable production forecasts (wind + solar)
5. Fundamental data (gas, carbon prices - if available)
6. Calendar features

The output is a CLEAN dataset ready for ML models with NO DATA LEAKAGE.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_collection.weather_forecast import get_weather_for_price_forecasting
from data_collection.load_forecast import get_load_forecast_for_price_forecasting
from data_collection.renewable_forecast import add_renewable_features
from model.price_forecasting.data_loader import add_calendar_features

logger = logging.getLogger(__name__)


def load_historical_prices(
    start_date: str,
    end_date: str,
    data_path: str = "data/raw_data/market_prices/FR_2023-01-01_2024-12-31.csv"
) -> pd.DataFrame:
    """Load historical price data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_path: Path to price data CSV

    Returns:
        DataFrame with datetime_hour, price, load_actual
    """
    df = pd.read_csv(data_path)
    df["datetime_hour"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"price_eur_mwh": "price", "load_mw": "load_actual"})

    # Filter date range
    df = df[
        (df["datetime_hour"] >= start_date) &
        (df["datetime_hour"] <= end_date)
    ]

    df = df[["datetime_hour", "price", "load_actual"]].copy()

    logger.info(
        f"✅ Loaded historical prices\n"
        f"   Records: {len(df):,}\n"
        f"   Date range: {df['datetime_hour'].min()} to {df['datetime_hour'].max()}\n"
        f"   Price range: [{df['price'].min():.2f}, {df['price'].max():.2f}] EUR/MWh"
    )

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_cols: list = ["price", "load_actual"],
    lags: list = [24, 48, 72, 168],
    forecast_horizon: int = 24
) -> pd.DataFrame:
    """Create lagged features ensuring NO DATA LEAKAGE.

    For forecasting at t+24h, we can only use information up to time t.
    All lags must be >= forecast_horizon.

    Args:
        df: DataFrame sorted by datetime
        target_cols: Columns to create lags for
        lags: List of lag periods (hours)
        forecast_horizon: Forecast horizon (hours)

    Returns:
        DataFrame with lag features
    """
    df = df.copy()

    # Validate lags
    invalid_lags = [lag for lag in lags if lag < forecast_horizon]
    if invalid_lags:
        logger.warning(
            f"⚠️  Found lags < forecast horizon ({forecast_horizon}h): {invalid_lags}\n"
            f"    These will cause data leakage! Removing them."
        )
        lags = [lag for lag in lags if lag >= forecast_horizon]

    for col in target_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping lag features")
            continue

        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    # Rolling statistics (also shifted to avoid leakage)
    for col in target_cols:
        if col not in df.columns:
            continue

        for window in [24, 168]:  # 1 day, 1 week
            df[f"{col}_roll_mean_{window}h"] = (
                df[col].shift(forecast_horizon).rolling(window).mean()
            )
            df[f"{col}_roll_std_{window}h"] = (
                df[col].shift(forecast_horizon).rolling(window).std()
            )

    logger.info(
        f"✅ Created lag features\n"
        f"   Columns: {target_cols}\n"
        f"   Lags: {lags}\n"
        f"   Forecast horizon: {forecast_horizon}h (no leakage)"
    )

    return df


def align_forecast_features(
    df_historical: pd.DataFrame,
    df_forecasts: pd.DataFrame,
    forecast_horizon_hours: int = 24
) -> pd.DataFrame:
    """Align forecast features with historical data for proper time indexing.

    CRITICAL: Forecasts must be aligned so that at time t, we use:
    - Historical data up to t (lags)
    - Forecast data FOR time t+horizon

    Args:
        df_historical: DataFrame with historical data at time t
        df_forecasts: DataFrame with forecast data (FOR time t+horizon)
        forecast_horizon_hours: Forecast horizon

    Returns:
        Merged DataFrame with proper alignment
    """
    df_hist = df_historical.copy()
    df_fcst = df_forecasts.copy()

    # Normalize timezones - remove timezone info for consistent merging
    df_hist["datetime_hour"] = pd.to_datetime(df_hist["datetime_hour"]).dt.tz_localize(None)
    df_fcst["datetime_hour"] = pd.to_datetime(df_fcst["datetime_hour"]).dt.tz_localize(None)

    # Shift forecasts BACK by horizon to align them with prediction time
    # When we're at time t, forecasts are FOR t+horizon
    # But we store them at t so the model knows what's predicted
    df_fcst["datetime_hour"] = df_fcst["datetime_hour"] - timedelta(hours=forecast_horizon_hours)

    # Merge
    df = df_hist.merge(df_fcst, on="datetime_hour", how="left", suffixes=("", "_fcst"))

    logger.info(
        f"✅ Aligned forecast features with {forecast_horizon_hours}h horizon\n"
        f"   Historical records: {len(df_hist):,}\n"
        f"   Forecast records: {len(df_fcst):,}\n"
        f"   Merged records: {len(df):,}\n"
        f"   Missing forecasts: {df[df_fcst.columns[1]].isna().sum():,}"
    )

    return df


def build_unified_price_forecasting_dataset(
    start_date: str,
    end_date: str,
    forecast_horizon_hours: int = 24,
    mode: str = "backtest",
    include_fundamentals: bool = False
) -> Tuple[pd.DataFrame, list]:
    """Build complete dataset for price forecasting with all features.

    This is the MAIN FUNCTION that creates the clean, leak-free dataset.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        forecast_horizon_hours: Forecast horizon (default 24h)
        mode: "backtest" (simulated forecasts) or "realtime" (actual forecasts)
        include_fundamentals: Include gas/carbon prices (if available)

    Returns:
        Tuple of (DataFrame with all features, list of feature column names)
    """
    logger.info(
        f"\n" + "="*80 + "\n"
        f"🚀 BUILDING UNIFIED PRICE FORECASTING DATASET\n"
        f"="*80 + "\n"
        f"Date range: {start_date} to {end_date}\n"
        f"Forecast horizon: {forecast_horizon_hours}h\n"
        f"Mode: {mode}\n"
    )

    # =========================================================================
    # STEP 1: Load historical data (prices, actual load)
    # =========================================================================
    logger.info("\n📊 STEP 1/6: Loading historical prices and load...")
    df = load_historical_prices(start_date, end_date)

    # =========================================================================
    # STEP 2: Get weather forecasts
    # =========================================================================
    logger.info("\n🌤️  STEP 2/6: Getting weather forecasts...")
    df_weather = get_weather_for_price_forecasting(
        start_date, end_date,
        mode=mode,
        forecast_horizon_hours=forecast_horizon_hours
    )

    # =========================================================================
    # STEP 3: Generate load forecasts
    # =========================================================================
    logger.info("\n⚡ STEP 3/6: Generating load forecasts...")
    df_load_forecast = get_load_forecast_for_price_forecasting(
        df_weather,
        df_historical_load=df,
        mode="simple",  # Can change to "ml" if we have trained model
        forecast_horizon_hours=forecast_horizon_hours
    )

    # Merge load forecast with weather
    df_forecasts = df_weather.merge(df_load_forecast, on="datetime_hour", how="left")

    # =========================================================================
    # STEP 4: Calculate renewable production forecasts
    # =========================================================================
    logger.info("\n🌬️  STEP 4/6: Calculating renewable production forecasts...")
    df_forecasts = add_renewable_features(df_forecasts, include_net_load=True)

    # =========================================================================
    # STEP 5: Align forecasts with historical data
    # =========================================================================
    logger.info("\n🔗 STEP 5/6: Aligning forecast features...")
    df = align_forecast_features(df, df_forecasts, forecast_horizon_hours)

    # =========================================================================
    # STEP 6: Add calendar features and lags
    # =========================================================================
    logger.info("\n📅 STEP 6/6: Adding calendar and lag features...")

    # Calendar features
    df = add_calendar_features(df)

    # Lag features (NO LEAKAGE - all lags >= forecast_horizon)
    valid_lags = [lag for lag in [24, 48, 72, 168] if lag >= forecast_horizon_hours]
    df = create_lag_features(
        df,
        target_cols=["price", "load_actual"],
        lags=valid_lags,
        forecast_horizon=forecast_horizon_hours
    )

    # =========================================================================
    # FINALIZE: Drop NaN, identify features
    # =========================================================================
    logger.info("\n🧹 Finalizing dataset...")

    # Drop rows with NaN (from lags and forecasts)
    df_clean = df.dropna().reset_index(drop=True)

    # Identify feature columns (exclude target and metadata)
    exclude_cols = [
        "datetime_hour",
        "price",  # This is the TARGET
        "load_actual",  # Contemporaneous - use lags instead
        "temp_actual",  # Use forecast instead
        "wind10m_actual",
        "wind100m_actual",
        "solar_radiation_actual",
        "humidity_actual",
        "precipitation_actual",
        "cloudcover_actual"
    ]

    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    logger.info(
        f"\n" + "="*80 + "\n"
        f"✅ DATASET READY!\n"
        f"="*80 + "\n"
        f"📊 Statistics:\n"
        f"   Total records: {len(df_clean):,}\n"
        f"   Features: {len(feature_cols)}\n"
        f"   Date range: {df_clean['datetime_hour'].min()} to {df_clean['datetime_hour'].max()}\n"
        f"   Target (price) range: [{df_clean['price'].min():.2f}, {df_clean['price'].max():.2f}] EUR/MWh\n"
        f"\n"
        f"🎯 Feature categories:\n"
    )

    # Categorize features
    calendar_features = [c for c in feature_cols if any(x in c for x in ["hour", "day", "month", "week", "year", "sin", "cos", "weekend", "peak", "winter", "summer"])]
    lag_features = [c for c in feature_cols if "lag" in c or "roll" in c]
    forecast_features = [c for c in feature_cols if "forecast" in c]
    renewable_features = [c for c in feature_cols if any(x in c for x in ["wind", "solar", "renewable", "net_load", "capacity_factor", "penetration", "coverage", "curtailment"])]

    logger.info(f"   Calendar: {len(calendar_features)}")
    logger.info(f"   Historical lags: {len(lag_features)}")
    logger.info(f"   Weather forecasts: {len([c for c in forecast_features if 'temp' in c or 'wind' in c or 'solar' in c])}")
    logger.info(f"   Load forecast: {len([c for c in forecast_features if 'load' in c])}")
    logger.info(f"   Renewable forecasts: {len(renewable_features)}")

    return df_clean, feature_cols


def save_dataset(
    df: pd.DataFrame,
    feature_cols: list,
    output_path: str = "data/processed/price_forecasting_dataset_with_forecasts.csv"
):
    """Save the unified dataset to CSV.

    Args:
        df: Complete DataFrame
        feature_cols: List of feature column names
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save main dataset
    df.to_csv(output_file, index=False)

    # Save feature list
    feature_file = output_file.parent / (output_file.stem + "_features.txt")
    with open(feature_file, "w") as f:
        f.write("\n".join(feature_cols))

    logger.info(
        f"\n✅ Saved dataset:\n"
        f"   Data: {output_file}\n"
        f"   Features: {feature_file}\n"
        f"   Size: {len(df):,} rows × {len(df.columns)} columns"
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Build dataset
    df, features = build_unified_price_forecasting_dataset(
        start_date="2023-01-01",
        end_date="2024-12-31",
        forecast_horizon_hours=24,
        mode="backtest"
    )

    # Save to file
    save_dataset(df, features)

    # Show sample
    print("\n" + "="*80)
    print("📊 SAMPLE DATA")
    print("="*80)
    print(df[["datetime_hour", "price", "load_forecast", "net_load_forecast_mw",
              "wind_production_forecast_mw", "solar_production_forecast_mw"]].head(10))

    print("\n" + "="*80)
    print("📈 FEATURE SUMMARY")
    print("="*80)
    print(f"Total features: {len(features)}")
    print("\nFirst 20 features:")
    for i, feat in enumerate(features[:20], 1):
        print(f"  {i:2d}. {feat}")
