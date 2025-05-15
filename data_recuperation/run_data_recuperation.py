import pandas as pd
from pathlib import Path

# Input folders
HOURLY_DIR = Path("data/raw_data/meteo_hourly_by_region")
DAILY_DIR = Path("data/raw_data/meteo_daily_by_region")
MODIFIED_DIR = Path("data/modified_data")
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1. Merge and sort hourly data
# ---------------------------
hourly_files = list(HOURLY_DIR.glob("*.csv"))
all_hourly = []
for file in hourly_files:
    df = pd.read_csv(file)
    # Standardize timestamp
    df["datetime_hour"] = pd.to_datetime(df.get("datetime", df.columns[0]), errors='coerce')
    df["date"] = df["datetime_hour"].dt.date
    all_hourly.append(df)

# Concatenate all hourly files
meteo_hourly = pd.concat(all_hourly, ignore_index=True)
# Sort only by datetime_hour to preserve chronological order
meteo_hourly = meteo_hourly.sort_values(by=["datetime_hour"]).reset_index(drop=True)
meteo_hourly.to_csv(MODIFIED_DIR / "meteo_hourly_regional.csv", index=False)
print(f"Hourly data merged and sorted: {len(meteo_hourly)} rows")

# ---------------------------
# 2. Merge and sort daily data
# ---------------------------
daily_files = list(DAILY_DIR.glob("*.csv"))
all_daily = []
for file in daily_files:
    df = pd.read_csv(file)
    # Ensure date column is in datetime format
    df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.date
    all_daily.append(df)

# Concatenate all daily files
meteo_daily = pd.concat(all_daily, ignore_index=True)
# Sort only by date to preserve chronological order
meteo_daily = meteo_daily.sort_values(by=["date"]).reset_index(drop=True)
meteo_daily.to_csv(MODIFIED_DIR / "meteo_daily_regional.csv", index=False)
print(f"Daily data merged and sorted: {len(meteo_daily)} rows")
