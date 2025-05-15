import pandas as pd
from pathlib import Path

# Input file paths
ENERGY_DAILY_PATH = "data/modified_data/energy_daily_regional.csv"
ENERGY_HOURLY_PATH = "data/modified_data/energy_hourly_regional.csv"
METEO_DAILY_PATH = "data/modified_data/meteo_daily_regional.csv"
METEO_HOURLY_PATH = "data/modified_data/meteo_hourly_regional.csv"

# Output file paths
MERGED_DAILY_PATH = "data/modified_data/merged_daily_regional.csv"
MERGED_HOURLY_PATH = "data/modified_data/merged_hourly_regional.csv"

# Create modified_data directory if needed
Path("data/modified_data").mkdir(parents=True, exist_ok=True)

# ------------------------
# 1. Daily merge
# ------------------------
df_energy_day = pd.read_csv(ENERGY_DAILY_PATH)
df_meteo_day = pd.read_csv(METEO_DAILY_PATH)

# Convert to datetime64[ns]
df_energy_day["date"] = pd.to_datetime(df_energy_day["date"])
df_meteo_day["date"] = pd.to_datetime(df_meteo_day["date"])

# Merge on date + insee_region
df_merged_day = pd.merge(df_energy_day, df_meteo_day, on=["date", "insee_region"], how="inner")

# Chronological sorting
df_merged_day = df_merged_day.sort_values(by=["date", "insee_region"]).reset_index(drop=True)

# Save
df_merged_day.to_csv(MERGED_DAILY_PATH, index=False)
print(f"Daily merge completed successfully: {len(df_merged_day):,} rows")

# ------------------------
# 2. Hourly merge
# ------------------------
df_energy_hour = pd.read_csv(ENERGY_HOURLY_PATH)
df_meteo_hour = pd.read_csv(METEO_HOURLY_PATH)

# Convert to datetime64[ns]
df_energy_hour["datetime_hour"] = pd.to_datetime(df_energy_hour["datetime_hour"])
df_meteo_hour["datetime_hour"] = pd.to_datetime(df_meteo_hour["datetime_hour"])

# Filter to April 2024 cutoff
cutoff = pd.Timestamp("2024-04-30 23:00:00")
df_energy_hour = df_energy_hour[df_energy_hour["datetime_hour"] <= cutoff]
df_meteo_hour = df_meteo_hour[df_meteo_hour["datetime_hour"] <= cutoff]

# Merge on datetime_hour + insee_region
df_merged_hour = pd.merge(df_energy_hour, df_meteo_hour, on=["datetime_hour", "insee_region"], how="inner")

# Chronological sorting
df_merged_hour = df_merged_hour.sort_values(by=["datetime_hour", "insee_region"]).reset_index(drop=True)

# Save
df_merged_hour.to_csv(MERGED_HOURLY_PATH, index=False)
print(f"Hourly merge completed successfully: {len(df_merged_hour):,} rows")
