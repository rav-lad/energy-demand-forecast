#!/usr/bin/env python
"""Collect extended historical data (2020-2025) for better model training.

This script collects ~5 years of data instead of 2 years to improve
deep learning model performance.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Add data_collection to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data_collection.entsoe_connector import EntsoeConnector
from data_collection.odre_collector import OdreCollector
from data_collection.weather_collector import WeatherCollector

# Load environment
load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

# Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_extended_data(start_date="2020-01-01", end_date=None):
    """Collect extended historical data.

    Args:
        start_date: Start date (default 2020-01-01 for ~5 years)
        end_date: End date (default today)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print("="*80)
    print(f"COLLECTING EXTENDED DATA: {start_date} to {end_date}")
    print("="*80)

    # 1. ENTSO-E Data (prices + load) - Collect year by year to avoid API limits
    print("\n1. Collecting ENTSO-E data (prices + load)...")
    entsoe = EntsoeConnector(api_key=ENTSOE_API_KEY)

    # Split into yearly chunks
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_prices = []
    all_load = []

    current_year = start_dt.year
    while current_year <= end_dt.year:
        year_start = f"{current_year}-01-01" if current_year > start_dt.year else start_date
        year_end = f"{current_year}-12-31" if current_year < end_dt.year else end_date

        print(f"   - Fetching {current_year}...")

        try:
            year_prices = entsoe.get_day_ahead_prices(
                country="FR",
                start=year_start,
                end=year_end
            )
            all_prices.append(year_prices)
            print(f"     OK Prices: {len(year_prices)} records")
        except Exception as e:
            print(f"     WARNING Prices failed: {e}")

        try:
            year_load = entsoe.get_actual_load(
                country="FR",
                start=year_start,
                end=year_end
            )
            all_load.append(year_load)
            print(f"     OK Load: {len(year_load)} records")
        except Exception as e:
            print(f"     WARNING Load failed: {e}")

        current_year += 1

    # Combine all years
    prices = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
    load = pd.concat(all_load, ignore_index=True) if all_load else pd.DataFrame()

    # Save raw ENTSO-E data
    prices.to_csv(RAW_DATA_DIR / "entsoe_prices_extended.csv", index=False)
    load.to_csv(RAW_DATA_DIR / "entsoe_load_extended.csv", index=False)
    print(f"\n   OK Total: {len(prices)} price records, {len(load)} load records")

    # 2. ODRE Data (backup for load)
    print("\n2. Collecting ODRE data (backup load)...")
    odre = OdreCollector()

    try:
        odre_data = odre.fetch_data(
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"   WARNING ODRE fetch failed: {e}")
        odre_data = None

    if odre_data is not None and len(odre_data) > 0:
        odre_data.to_csv(RAW_DATA_DIR / "odre_consumption_extended.csv", index=False)
        print(f"   OK Saved {len(odre_data)} ODRE records")
    else:
        print("   WARNING ODRE data collection failed (using ENTSO-E only)")

    # 3. Weather Data
    print("\n3. Collecting weather data...")
    weather = WeatherCollector()

    # Paris coordinates (France)
    weather_data = weather.get_historical_weather(
        latitude=48.8566,
        longitude=2.3522,
        start_date=start_date,
        end_date=end_date
    )

    weather_data.to_csv(RAW_DATA_DIR / "weather_extended.csv", index=False)
    print(f"   OK Saved {len(weather_data)} weather records")

    # 4. Merge all data
    print("\n4. Merging datasets...")

    # Convert to daily
    prices['date'] = pd.to_datetime(prices['datetime']).dt.date
    load['date'] = pd.to_datetime(load['datetime']).dt.date
    weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

    # Aggregate to daily
    daily_prices = prices.groupby('date').agg({
        'price_eur_mwh': 'mean'
    }).reset_index()

    daily_load = load.groupby('date').agg({
        'load_mw': 'mean'
    }).reset_index()

    # Merge
    merged = daily_prices.merge(daily_load, on='date', how='inner')
    merged = merged.merge(weather_data, on='date', how='inner')

    # Add time features
    merged['datetime'] = pd.to_datetime(merged['date'])
    merged['year'] = merged['datetime'].dt.year
    merged['month'] = merged['datetime'].dt.month
    merged['day'] = merged['datetime'].dt.day
    merged['dayofweek'] = merged['datetime'].dt.dayofweek
    merged['quarter'] = merged['datetime'].dt.quarter
    merged['dayofyear'] = merged['datetime'].dt.dayofyear
    merged['weekofyear'] = merged['datetime'].dt.isocalendar().week

    # Cyclical encoding
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
    merged['day_sin'] = np.sin(2 * np.pi * merged['day'] / 31)
    merged['day_cos'] = np.cos(2 * np.pi * merged['day'] / 31)
    merged['dayofweek_sin'] = np.sin(2 * np.pi * merged['dayofweek'] / 7)
    merged['dayofweek_cos'] = np.cos(2 * np.pi * merged['dayofweek'] / 7)
    merged['is_weekend'] = (merged['dayofweek'] >= 5).astype(int)

    # Add lag features
    for lag in [1, 2, 3, 7, 14]:
        merged[f'load_lag_{lag}'] = merged['load_mw'].shift(lag)
        if lag <= 3:
            merged[f'price_lag_{lag}'] = merged['price_eur_mwh'].shift(lag)

    # Rolling features
    for window in [7, 14, 30]:
        merged[f'load_rolling_mean_{window}'] = merged['load_mw'].shift(1).rolling(window).mean()
        merged[f'load_rolling_std_{window}'] = merged['load_mw'].shift(1).rolling(window).std()

    merged['price_rolling_mean_7'] = merged['price_eur_mwh'].shift(1).rolling(7).mean()

    # Drop NaN from lag features
    merged = merged.dropna()

    # Drop date column, keep datetime
    merged = merged.drop(columns=['date'])

    # Save
    output_file = PROCESSED_DATA_DIR / "daily_data_extended.csv"
    merged.to_csv(output_file, index=False)

    print(f"\nOK Merged dataset saved: {output_file}")
    print(f"  Total records: {len(merged)}")
    print(f"  Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
    print(f"  Features: {len(merged.columns)}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nPrice (EUR/MWh):")
    print(f"  Mean: {merged['price_eur_mwh'].mean():.2f}")
    print(f"  Std:  {merged['price_eur_mwh'].std():.2f}")
    print(f"  Min:  {merged['price_eur_mwh'].min():.2f}")
    print(f"  Max:  {merged['price_eur_mwh'].max():.2f}")

    print(f"\nLoad (MW):")
    print(f"  Mean: {merged['load_mw'].mean():.2f}")
    print(f"  Std:  {merged['load_mw'].std():.2f}")
    print(f"  Min:  {merged['load_mw'].min():.2f}")
    print(f"  Max:  {merged['load_mw'].max():.2f}")

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    collect_extended_data(start_date=args.start, end_date=args.end)
