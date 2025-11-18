"""
Create synthetic data matching the notebook structure.
This creates train_daily.csv and test_daily.csv with realistic energy market data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create data directories
DATA_DIR = Path("data")
MODIFIED_DIR = DATA_DIR / "modified_data"
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)

# Generate dates: 700 days total (560 train + 140 test)
start_date = pd.Timestamp('2023-01-31')
n_days = 700
dates = pd.date_range(start=start_date, periods=n_days, freq='D')

# Generate base features with realistic patterns
def generate_data(dates):
    n = len(dates)

    # Time features
    data = {
        'datetime': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'dayofweek': dates.dayofweek,
        'quarter': dates.quarter,
        'dayofyear': dates.dayofyear,
        'week': dates.isocalendar().week.astype(int)
    }

    # Seasonal patterns
    day_of_year = np.array([d.dayofyear for d in dates])
    seasonal_pattern = np.sin(2 * np.pi * day_of_year / 365.25)
    weekly_pattern = np.sin(2 * np.pi * day_of_year / 7)

    # Weather features (realistic French climate)
    base_temp = 12 + 10 * seasonal_pattern  # Seasonal temperature variation
    data['temperature_2m_max'] = base_temp + 5 + np.random.normal(0, 3, n)
    data['temperature_2m_min'] = base_temp - 5 + np.random.normal(0, 2.5, n)
    data['precipitation_sum'] = np.abs(2.5 + 2 * (1 - seasonal_pattern) + np.random.exponential(2, n))
    data['wind_speed_10m_max'] = 15 + 8 * (1 - seasonal_pattern) + np.random.gamma(2, 3, n)
    data['shortwave_radiation_sum'] = 10 + 8 * seasonal_pattern + np.random.normal(0, 2, n)
    data['et0_fao_evapotranspiration'] = 1 + 1.5 * seasonal_pattern + np.random.normal(0, 0.3, n)

    # Energy load (MW) - correlate with temperature and time
    # Higher load in winter (heating) and summer (cooling), weekly patterns
    base_load = 48000
    temp_effect = -300 * (data['temperature_2m_max'] - 15)  # Higher load when temp far from 15°C
    weekend_effect = -4000 * (data['dayofweek'] >= 5).astype(int)  # Lower on weekends
    seasonal_load = 8000 * (1 - seasonal_pattern)  # Higher in winter

    data['load_mw'] = (base_load + temp_effect + weekend_effect + seasonal_load +
                       np.random.normal(0, 2000, n))

    # Electricity price (EUR/MWh) - complex relationship with load, renewables, time
    # Merit order effect: price increases non-linearly with demand
    load_normalized = (data['load_mw'] - 40000) / 10000
    price_from_load = 50 + 20 * load_normalized + 5 * load_normalized**2

    # Renewable effect (wind reduces prices)
    renewable_effect = -data['wind_speed_10m_max'] * 0.8

    # Time effects (peak hours in real markets)
    hour_effect = 10 * weekly_pattern

    # Random volatility
    volatility = np.random.normal(0, 15, n)

    data['price_eur_mwh'] = np.maximum(0, price_from_load + renewable_effect + hour_effect + volatility)

    # Create lag features for load
    df = pd.DataFrame(data)
    df['load_lag_1'] = df['load_mw'].shift(1)
    df['load_lag_7'] = df['load_mw'].shift(7)
    df['load_lag_14'] = df['load_mw'].shift(14)

    # Rolling features
    df['load_rolling_mean_7'] = df['load_mw'].rolling(window=7, min_periods=1).mean()
    df['load_rolling_std_7'] = df['load_mw'].rolling(window=7, min_periods=1).std()
    df['load_rolling_mean_14'] = df['load_mw'].rolling(window=14, min_periods=1).mean()
    df['load_rolling_std_14'] = df['load_mw'].rolling(window=14, min_periods=1).std()
    df['load_rolling_mean_30'] = df['load_mw'].rolling(window=30, min_periods=1).mean()
    df['load_rolling_std_30'] = df['load_mw'].rolling(window=30, min_periods=1).std()

    # Price lags
    df['price_lag_1'] = df['price_eur_mwh'].shift(1)
    df['price_rolling_mean_7'] = df['price_eur_mwh'].rolling(window=7, min_periods=1).mean()

    # Fill NaN from lags with forward fill
    df = df.fillna(method='bfill')

    return df

# Generate full dataset
print("Generating synthetic data...")
df_full = generate_data(dates)

# Split into train and test
train_size = 560
df_train = df_full.iloc[:train_size].copy()
df_test = df_full.iloc[train_size:].copy()

print(f"Train set: {len(df_train)} days ({df_train['datetime'].min()} to {df_train['datetime'].max()})")
print(f"Test set: {len(df_test)} days ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

# Save to CSV
train_path = MODIFIED_DIR / 'train_daily.csv'
test_path = MODIFIED_DIR / 'test_daily.csv'

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"\n✅ Data saved:")
print(f"   {train_path}")
print(f"   {test_path}")

# Display sample
print(f"\nSample data:")
print(df_train.head())
print(f"\nColumns: {df_train.columns.tolist()}")
print(f"\nData statistics:")
print(df_train[['price_eur_mwh', 'load_mw', 'temperature_2m_max']].describe())
