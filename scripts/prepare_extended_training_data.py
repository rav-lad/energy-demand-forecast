"""Create extended training dataset from collected ENTSO-E + weather data."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"
MODIFIED_DATA_DIR = BASE_DIR / "data" / "modified_data"

def create_extended_dataset():
    """Create extended dataset with proper feature engineering."""

    print("="*80)
    print("CREATING EXTENDED TRAINING DATASET")
    print("="*80)

    # 1. Load ENTSO-E data
    print("\n1. Loading ENTSO-E data...")
    prices = pd.read_csv(RAW_DATA_DIR / "entsoe_prices_extended.csv")
    load = pd.read_csv(RAW_DATA_DIR / "entsoe_load_extended.csv")

    prices['datetime'] = pd.to_datetime(prices['datetime'])
    load['datetime'] = pd.to_datetime(load['datetime'])

    # Filter to only use data up to today (no future data!)
    today = pd.Timestamp.now(tz='UTC')
    prices = prices[prices['datetime'] <= today]
    load = load[load['datetime'] <= today]

    print(f"   Prices: {len(prices)} hourly records ({prices['datetime'].min()} to {prices['datetime'].max()})")
    print(f"   Load: {len(load)} hourly records ({load['datetime'].min()} to {load['datetime'].max()})")

    # 2. Aggregate to daily
    print("\n2. Aggregating to daily...")
    # Convert to local time (remove timezone for easier merging)
    prices['datetime'] = prices['datetime'].dt.tz_localize(None)
    load['datetime'] = load['datetime'].dt.tz_localize(None)

    prices_daily = prices[['datetime', 'price_eur_mwh']].set_index('datetime').resample('D').mean()
    load_daily = load[['datetime', 'load_mw']].set_index('datetime').resample('D').mean()

    # Merge price and load
    energy_data = prices_daily.join(load_daily, how='inner')
    energy_data = energy_data.reset_index()

    print(f"   Daily energy data: {len(energy_data)} days")

    # 3. Load weather data
    print("\n3. Loading weather data...")
    weather = pd.read_csv(RAW_DATA_DIR / "weather_extended.csv")
    weather['date'] = pd.to_datetime(weather['date'])
    weather = weather.rename(columns={'date': 'datetime'})

    print(f"   Weather data: {len(weather)} days")

    # 4. Merge all data
    print("\n4. Merging datasets...")
    merged = energy_data.merge(weather, on='datetime', how='inner')
    merged = merged.sort_values('datetime').reset_index(drop=True)

    print(f"   Merged: {len(merged)} days ({merged['datetime'].min()} to {merged['datetime'].max()})")

    # 5. Add time features
    print("\n5. Adding time features...")
    merged['year'] = merged['datetime'].dt.year
    merged['month'] = merged['datetime'].dt.month
    merged['day'] = merged['datetime'].dt.day
    merged['dayofweek'] = merged['datetime'].dt.dayofweek
    merged['quarter'] = merged['datetime'].dt.quarter
    merged['dayofyear'] = merged['datetime'].dt.dayofyear
    merged['weekofyear'] = merged['datetime'].dt.isocalendar().week
    merged['is_weekend'] = (merged['dayofweek'] >= 5).astype(int)

    # Cyclical encoding
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
    merged['day_sin'] = np.sin(2 * np.pi * merged['day'] / 31)
    merged['day_cos'] = np.cos(2 * np.pi * merged['day'] / 31)
    merged['dayofweek_sin'] = np.sin(2 * np.pi * merged['dayofweek'] / 7)
    merged['dayofweek_cos'] = np.cos(2 * np.pi * merged['dayofweek'] / 7)

    # 6. Add lag features (CRITICAL: No future leakage!)
    print("\n6. Adding lag features...")

    # Price lags (only keep most important: 1, 2, 3, 7 days)
    for lag in [1, 2, 3, 7]:
        merged[f'price_lag_{lag}'] = merged['price_eur_mwh'].shift(lag)

    # Load lags (only keep most important: 1, 2, 3, 7 days)
    for lag in [1, 2, 3, 7]:
        merged[f'load_lag_{lag}'] = merged['load_mw'].shift(lag)

    # Rolling statistics (only 7-day window to minimize data loss)
    merged['load_rolling_mean_7'] = merged['load_mw'].shift(1).rolling(7).mean()
    merged['load_rolling_std_7'] = merged['load_mw'].shift(1).rolling(7).std()
    merged['price_rolling_mean_7'] = merged['price_eur_mwh'].shift(1).rolling(7).mean()

    # Drop rows with NaN (from lag/rolling features) - only lose ~14 rows now instead of 30
    initial_len = len(merged)
    merged = merged.dropna()
    print(f"   Dropped {initial_len - len(merged)} rows with NaN (from lag features)")

    # 7. Save full dataset
    print("\n7. Saving dataset...")
    output_file = PROCESSED_DATA_DIR / "daily_data_extended_full.csv"
    merged.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"EXTENDED DATASET CREATED:")
    print(f"  File: {output_file}")
    print(f"  Total records: {len(merged)} days")
    print(f"  Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
    print(f"  Features: {len(merged.columns)}")
    print(f"{'='*80}")

    # 8. Create train/test split (80/20 for larger dataset)
    print("\n8. Creating train/test split (80/20)...")
    train_size = int(len(merged) * 0.8)

    train_df = merged[:train_size].copy()
    test_df = merged[train_size:].copy()

    train_df.to_csv(MODIFIED_DATA_DIR / "train_daily_extended.csv", index=False)
    test_df.to_csv(MODIFIED_DATA_DIR / "test_daily_extended.csv", index=False)

    print(f"   Train: {len(train_df)} days ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"   Test: {len(test_df)} days ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    print(f"\n   Saved to: {MODIFIED_DATA_DIR}")


if __name__ == "__main__":
    create_extended_dataset()
