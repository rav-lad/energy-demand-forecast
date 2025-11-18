"""
Prepare Training Data for Energy Demand Forecasting

This script:
1. Loads raw data (ENTSO-E prices, ODRE consumption, Weather)
2. Merges all data sources
3. Engineers features
4. Splits into train/test sets
5. Saves ready-to-use datasets

Usage:
    python prepare_training_data.py --frequency daily
    python prepare_training_data.py --frequency hourly
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw_data"
MODIFIED_DIR = BASE_DIR / "data" / "modified_data"

# Create output directory
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)


def load_market_data():
    """Load ENTSO-E market data (prices and load)."""
    logger.info("Loading ENTSO-E market data...")

    market_file = RAW_DIR / "market_prices" / "FR_2023-01-01_2024-12-31.csv"
    df = pd.read_csv(market_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    logger.info(f"  Loaded {len(df):,} records")
    logger.info(f"  Period: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def load_odre_consumption():
    """Load ODRE regional consumption data."""
    logger.info("Loading ODRE consumption data...")

    odre_file = RAW_DIR / "energy_consumption_2023-2024.csv"
    df = pd.read_csv(odre_file)

    # Parse datetime
    if 'datetime_heure' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime_heure'])
    else:
        df['datetime'] = pd.to_datetime(df['date'])

    logger.info(f"  Loaded {len(df):,} records")
    logger.info(f"  Regions: {df['code_insee_region'].nunique()}")

    return df


def load_weather_data(frequency='daily'):
    """Load weather data."""
    logger.info(f"Loading {frequency} weather data...")

    weather_file = MODIFIED_DIR / f"weather_{frequency}_2023-01-01_2024-12-31.csv"
    df = pd.read_csv(weather_file)

    # Parse datetime
    if frequency == 'daily':
        df['datetime'] = pd.to_datetime(df['date'])
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

    logger.info(f"  Loaded {len(df):,} records")

    return df


def aggregate_daily(df_hourly):
    """Aggregate hourly data to daily."""
    logger.info("Aggregating hourly to daily...")

    df = df_hourly.copy()
    df['date'] = df['datetime'].dt.date

    # Aggregate
    agg_dict = {
        'price_eur_mwh': 'mean',
        'load_mw': 'mean',
    }

    df_daily = df.groupby('date').agg(agg_dict).reset_index()
    df_daily['datetime'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.drop('date', axis=1)

    logger.info(f"  Aggregated to {len(df_daily):,} daily records")

    return df_daily


def merge_all_data(df_market, df_weather, frequency='daily'):
    """Merge all data sources."""
    logger.info("Merging all data sources...")

    # Ensure datetime is timezone-naive for merging
    df_market['datetime'] = pd.to_datetime(df_market['datetime']).dt.tz_localize(None)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime']).dt.tz_localize(None)

    if frequency == 'daily':
        # Aggregate market to daily
        df_market = aggregate_daily(df_market)

    # Aggregate weather by averaging across regions
    weather_cols = [col for col in df_weather.columns if col not in ['datetime', 'insee_region', 'date']]
    df_weather_agg = df_weather.groupby('datetime')[weather_cols].mean().reset_index()

    # Merge market + weather
    df_merged = pd.merge(df_market, df_weather_agg, on='datetime', how='inner')

    logger.info(f"  Merged dataset: {len(df_merged):,} records")
    logger.info(f"  Features: {len(df_merged.columns)} columns")

    return df_merged


def engineer_features(df):
    """Engineer time-based and lag features."""
    logger.info("Engineering features...")

    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Time features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Is weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Lag features for load (target variable)
    for lag in [1, 2, 3, 7, 14]:
        df[f'load_lag_{lag}'] = df['load_mw'].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'load_rolling_mean_{window}'] = df['load_mw'].shift(1).rolling(window).mean()
        df[f'load_rolling_std_{window}'] = df['load_mw'].shift(1).rolling(window).std()

    # Price features
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['price_rolling_mean_7'] = df['price_eur_mwh'].shift(1).rolling(7).mean()

    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()

    logger.info(f"  Features after engineering: {len(df.columns)} columns")
    logger.info(f"  Records after dropna: {len(df):,}")

    return df


def split_train_test(df, test_size=0.2):
    """Split data into train and test sets (temporal split)."""
    logger.info("Splitting train/test...")

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Temporal split (last 20% for test)
    split_idx = int(len(df) * (1 - test_size))

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    logger.info(f"  Train: {len(df_train):,} records ({df_train['datetime'].min()} to {df_train['datetime'].max()})")
    logger.info(f"  Test:  {len(df_test):,} records ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    return df_train, df_test


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly"],
        default="daily",
        help="Data frequency"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PREPARING TRAINING DATA")
    logger.info("=" * 70)
    logger.info(f"Frequency: {args.frequency}")
    logger.info(f"Test size: {args.test_size * 100:.0f}%")

    try:
        # Load data
        df_market = load_market_data()
        df_weather = load_weather_data(frequency=args.frequency)

        # Merge
        df_merged = merge_all_data(df_market, df_weather, frequency=args.frequency)

        # Engineer features
        df_features = engineer_features(df_merged)

        # Split
        df_train, df_test = split_train_test(df_features, test_size=args.test_size)

        # Save
        train_file = MODIFIED_DIR / f"train_{args.frequency}.csv"
        test_file = MODIFIED_DIR / f"test_{args.frequency}.csv"

        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)

        logger.info("=" * 70)
        logger.info("DATA PREPARATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Train file: {train_file}")
        logger.info(f"Test file:  {test_file}")

        # Display sample
        logger.info("\nSample features:")
        logger.info(f"  Target: load_mw")
        logger.info(f"  Features: {[col for col in df_train.columns if col not in ['datetime', 'load_mw', 'country']][:10]}")
        logger.info("  ...")

        return True

    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
