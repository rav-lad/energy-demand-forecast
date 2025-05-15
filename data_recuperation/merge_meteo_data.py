import pandas as pd
from pathlib import Path

# Input folders
HOURLY_DIR = Path("data/raw_data/meteo_hourly_by_region")
DAILY_DIR = Path("data/raw_data/meteo_daily_by_region")
MODIFIED_DIR = Path("data/modified_data")
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1. Merge and shuffle hourly data
# ---------------------------
hourly_files = list(HOURLY_DIR.glob("*.csv"))
all_hourly = []
for file in hourly_files:
    df = pd.read_csv(file)
    df["datetime_hour"] = pd.to_datetime(df.get("datetime", df.columns[0]), errors='coerce')
    df["date"] = df["datetime_hour"].dt.date
    all_hourly.append(df)

meteo_hourly = pd.concat(all_hourly, ignore_index=True)

# Randomly shuffle regions within each datetime_hour
meteo_hourly = (
    meteo_hourly
    .groupby("datetime_hour", group_keys=False)
    .apply(lambda x: x.sample(frac=1, random_state=42), include_groups=False)
    .sort_values("datetime_hour")
    .reset_index(drop=True)
)

meteo_hourly.to_csv(MODIFIED_DIR / "meteo_hourly_regional.csv", index=False)
print(f"Hourly data merged and shuffled: {len(meteo_hourly)} rows")

# ---------------------------
# 2. Merge and shuffle daily data
# ---------------------------
daily_files = list(DAILY_DIR.glob("*.csv"))
all_daily = []
for file in daily_files:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.date
    all_daily.append(df)

meteo_daily = pd.concat(all_daily, ignore_index=True)

meteo_daily = (
    meteo_daily
    .groupby("date", group_keys=False)
    .apply(lambda x: x.sample(frac=1, random_state=42))
    .sort_values("date")
    .reset_index(drop=True)
)

meteo_daily.to_csv(MODIFIED_DIR / "meteo_daily_regional.csv", index=False)
print(f"Daily data merged and shuffled: {len(meteo_daily)} rows")
